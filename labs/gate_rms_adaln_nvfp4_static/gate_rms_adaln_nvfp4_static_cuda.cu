/*
 * MiniMax H3 attention gate + exact RMSNorm + segmented AdaLN -> supplied-
 * scale NVFP4 for FC1.
 *
 * This isolated SM120 candidate composes two already-qualified contracts.  A
 * 128-thread CTA owns one row, applies the exact BF16 fmaf residual mutation,
 * retains the rounded residual in registers, reproduces PyTorch cf30153's RMS
 * reduction tree, stages AdaLN through the observable BF16 boundaries, and
 * emits TensorCoreNVFP4Layout data with the v3 bridge's FP32 scale.  No global
 * activation-amax pass or intermediate activation reread is required.
 */

#include <torch/extension.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/GradMode.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kWidth = 5376;
constexpr double kCertifiedEps = 1.0e-5;
constexpr int kVec = 4;
constexpr int kWarpCount = 4;
constexpr int kThreads = 32 * kWarpCount;
constexpr int kVectorCount = kWidth / kVec;
constexpr int kVectorsPerThread =
    (kVectorCount + kThreads - 1) / kThreads;  // 11
constexpr int kFp4BlockSize = 16;
constexpr int kThreadsPerScale = kFp4BlockSize / kVec;  // 4
constexpr int kScaleCols = kWidth / kFp4BlockSize;
constexpr int kPackedRowBytes = kWidth / 2;

static_assert(kWidth % 256 == 0,
              "certified RMS vector/reduction tiling requires width % 256");
static_assert(kWidth % kFp4BlockSize == 0,
              "NVFP4 scale blocks must divide the H3 width");
static_assert(kScaleCols % 4 == 0,
              "H3 scale rows require no right-side scale padding");
static_assert(kVectorCount == 1344 && kVectorsPerThread == 11,
              "register schedule is specialized to H3 width 5376");

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
  T value[N];
};

using Bf16x4 = AlignedVector<__nv_bfloat16, kVec>;

__device__ __forceinline__ size_t scale_factor_swizzled_offset(
    size_t row_idx, size_t col_idx, uint32_t col_length) {
  // TensorCoreNVFP4Layout / cuBLAS block-scale swizzle:
  // (row_block, col_block_group, 32, 4, 4).
  constexpr uint32_t kRowsPerTile = 128;
  constexpr uint32_t kRowsPerTileColumn = 32;
  constexpr uint32_t kColsPerTileColumn = 4;
  const size_t row_block = row_idx / kRowsPerTile;
  const size_t rem = row_idx % kRowsPerTile;
  const size_t d4 = rem / kRowsPerTileColumn;
  const size_t d3 = rem % kRowsPerTileColumn;
  const size_t col_block_group = col_idx / kColsPerTileColumn;
  const size_t d5 = col_idx % kColsPerTileColumn;
  const size_t groups =
      (col_length + kColsPerTileColumn - 1) / kColsPerTileColumn;
  return ((row_block * groups + col_block_group) * kRowsPerTileColumn + d3)
      * 16 + d4 * kColsPerTileColumn + d5;
}

__device__ __forceinline__ __nv_bfloat16 weighted_rms_bf16(
    __nv_bfloat16 x, __nv_bfloat16 weight, float rstd) {
  // Match vectorized_layer_norm_kernel: gamma * (rstd * x), then BF16 store.
  const float normalized = rstd * __bfloat162float(x);
  const float weighted = __bfloat162float(weight) * normalized;
  return __float2bfloat16_rn(weighted);
}

__device__ __forceinline__ __nv_bfloat16 staged_adaln_bf16(
    __nv_bfloat16 normed, float scale, float shift) {
  // Match h.mul_(1 + scale.to(bf16)).add_(shift.to(bf16)), preserving all
  // three observable BF16 storage boundaries.
  const __nv_bfloat16 scale_bf16 = __float2bfloat16_rn(scale);
  const __nv_bfloat16 multiplier_bf16 = __float2bfloat16_rn(
      1.0f + __bfloat162float(scale_bf16));
  const __nv_bfloat16 product_bf16 = __float2bfloat16_rn(
      __bfloat162float(normed) * __bfloat162float(multiplier_bf16));
  const __nv_bfloat16 shift_bf16 = __float2bfloat16_rn(shift);
  return __float2bfloat16_rn(
      __bfloat162float(product_bf16) + __bfloat162float(shift_bf16));
}

template <bool EmitRstd>
__global__ __launch_bounds__(kThreads)
void gate_rms_adaln_nvfp4_static_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ other,
    const float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    const float* __restrict__ global_scale,
    float* __restrict__ rstd_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t original_rows,
    int64_t gate_stride,
    int64_t scale_stride,
    int64_t shift_stride,
    int64_t modulation_rows,
    float eps) {
  __shared__ float warp_scratch[kWarpCount];
  __shared__ volatile float rstd_shared;

  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = lane + warp * 32;

  // Up to fifteen canonical padding rows follow the real activation.  Block
  // scales for these rows are already zero-initialized by the host allocation.
  if (row >= original_rows) {
    uint8_t* row_output = output + row * kPackedRowBytes;
    for (int byte = tid; byte < kPackedRowBytes; byte += kThreads) {
      row_output[byte] = 0;
    }
    return;
  }

  const int32_t selected_mod_row = row_ids[row];
  // The Python identity token proves the map. Clamp as the final native OOB
  // boundary without adding a device-to-host synchronization.
  const int32_t mod_row =
      selected_mod_row >= 0 && selected_mod_row < modulation_rows
      ? selected_mod_row : 0;
  const int64_t gate_base = static_cast<int64_t>(mod_row) * gate_stride;
  const int64_t scale_base = static_cast<int64_t>(mod_row) * scale_stride;
  const int64_t shift_base = static_cast<int64_t>(mod_row) * shift_stride;

  Bf16x4* x_vectors = reinterpret_cast<Bf16x4*>(
      x + row * static_cast<int64_t>(kWidth));
  const Bf16x4* other_vectors = reinterpret_cast<const Bf16x4*>(
      other + row * static_cast<int64_t>(kWidth));

  // Apply the exact stock residual expression once.  fmaf happens in FP32,
  // and the result is rounded to BF16 before it contributes to RMS statistics.
  // Retaining those 44 BF16 values also removes the post-gate activation reread.
  Bf16x4 cached[kVectorsPerThread];
  float sigma2 = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < kVectorsPerThread; ++iteration) {
    const int vector_index = tid + iteration * kThreads;
    Bf16x4 updated{};
    if (vector_index < kVectorCount) {
      const Bf16x4 x_data = x_vectors[vector_index];
      const Bf16x4 other_data = other_vectors[vector_index];
#pragma unroll
      for (int element = 0; element < kVec; ++element) {
        const int feature = vector_index * kVec + element;
        const __nv_bfloat16 gate_bf16 = __float2bfloat16_rn(
            gate[gate_base + feature]);
        const float gated = fmaf(
            __bfloat162float(other_data.value[element]),
            __bfloat162float(gate_bf16),
            __bfloat162float(x_data.value[element]));
        updated.value[element] = __float2bfloat16_rn(gated);
        const float rounded = __bfloat162float(updated.value[element]);
        sigma2 = sigma2 + rounded * rounded;
      }
      x_vectors[vector_index] = updated;
    }
    cached[iteration] = updated;
  }

  // Exact PyTorch cf30153 RMS tree: per-thread serial accumulation, warp
  // shuffle tree, then (warp0 + warp2) + (warp1 + warp3).
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sigma2 = sigma2 + __shfl_down_sync(0xffffffffu, sigma2, offset);
  }
  if (lane == 0) {
    warp_scratch[warp] = sigma2;
  }
  __syncthreads();
  if (lane == 0 && warp == 0) {
    const float left = warp_scratch[0] + warp_scratch[2];
    const float right = warp_scratch[1] + warp_scratch[3];
    rstd_shared = rsqrtf(
        (left + right) / static_cast<float>(kWidth) + eps);
  }
  __syncthreads();

  const float rstd = rstd_shared;
  if constexpr (EmitRstd) {
    if (tid == 0) {
      rstd_output[row] = rstd;
    }
  }
  const float global_decode_scale = global_scale[0];
  uint8_t* row_output = output + row * kPackedRowBytes;

#pragma unroll
  for (int iteration = 0; iteration < kVectorsPerThread; ++iteration) {
    const int vector_index = tid + iteration * kThreads;
    if (vector_index < kVectorCount) {
      const int col = vector_index * kVec;
      float values[kVec];
#pragma unroll
      for (int element = 0; element < kVec; ++element) {
        const int feature = col + element;
        const __nv_bfloat16 normed = weighted_rms_bf16(
            cached[iteration].value[element], weight[feature], rstd);
        values[element] = __bfloat162float(staged_adaln_bf16(
            normed,
            scale[scale_base + feature],
            shift[shift_base + feature]));
      }

      float absmax = fabsf(values[0]);
#pragma unroll
      for (int element = 1; element < kVec; ++element) {
        absmax = fmaxf(absmax, fabsf(values[element]));
      }
#pragma unroll
      for (int offset = kThreadsPerScale / 2; offset >= 1; offset >>= 1) {
        absmax = fmaxf(
            absmax,
            __shfl_down_sync(
                0xffffffffu, absmax, offset, kThreadsPerScale));
      }
      const int leader = (lane / kThreadsPerScale) * kThreadsPerScale;
      absmax = __shfl_sync(0xffffffffu, absmax, leader, 32);

      float decode_scale = __fdividef(
          __fdividef(absmax, 6.0f), global_decode_scale);
      decode_scale = fminf(decode_scale, 448.0f);
      const __nv_fp8_e4m3 scale_fp8 =
          static_cast<__nv_fp8_e4m3>(decode_scale);
      const float decode_scale_fp8 = static_cast<float>(scale_fp8);

      if ((tid % kThreadsPerScale) == 0) {
        const size_t scale_offset = scale_factor_swizzled_offset(
            static_cast<size_t>(row),
            static_cast<size_t>(col / kFp4BlockSize),
            kScaleCols);
        reinterpret_cast<__nv_fp8_e4m3*>(block_scales)[scale_offset] =
            scale_fp8;
      }

      const float encode_scale = fminf(
          __fdividef(
              1.0f, decode_scale_fp8 * global_decode_scale),
          FLT_MAX);
#pragma unroll
      for (int element = 0; element < kVec; ++element) {
        values[element] *= encode_scale;
      }

      union PackedFp4 {
        uint16_t u16;
        __nv_fp4x2_storage_t fp4x2[2];
      } packed;
      // Reverse each pair for TensorCoreNVFP4Layout's high-nibble-first order.
      packed.fp4x2[0] = __nv_cvt_float2_to_fp4x2(
          float2{values[1], values[0]}, __NV_E2M1, cudaRoundNearest);
      packed.fp4x2[1] = __nv_cvt_float2_to_fp4x2(
          float2{values[3], values[2]}, __NV_E2M1, cudaRoundNearest);
      *reinterpret_cast<uint16_t*>(row_output + col / 2) = packed.u16;
    }
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& other,
    const torch::Tensor& gate,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    const torch::Tensor& global_scale,
    double eps) {
  TORCH_CHECK(!c10::GradMode::is_enabled(),
              "native H3 gate/RMS/AdaLN producer is inference-only");
  TORCH_CHECK(eps == kCertifiedEps,
              "eps must equal the certified H3 value 1e-5");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "x must be CUDA BF16");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) == kWidth,
              "x must have nonempty shape [S, 5376]");
  TORCH_CHECK(x.is_contiguous() && !x.requires_grad(),
              "x must be contiguous and inference-only");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(x.data_ptr())
          % (sizeof(__nv_bfloat16) * kVec) == 0,
      "x storage offset must preserve 8-byte vector alignment");

  TORCH_CHECK(other.is_cuda() && other.scalar_type() == torch::kBFloat16,
              "other must be CUDA BF16");
  TORCH_CHECK(other.sizes() == x.sizes() && other.is_contiguous()
              && !other.requires_grad(),
              "other must match contiguous inference-only x");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(other.data_ptr())
          % (sizeof(__nv_bfloat16) * kVec) == 0,
      "other storage offset must preserve 8-byte vector alignment");

  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kBFloat16,
              "weight must be CUDA BF16");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kWidth
              && weight.is_contiguous() && !weight.requires_grad(),
              "weight must be contiguous inference-only [5376]");

  TORCH_CHECK(gate.is_cuda() && scale.is_cuda() && shift.is_cuda(),
              "gate/scale/shift must be CUDA");
  TORCH_CHECK(gate.scalar_type() == torch::kFloat32
              && scale.scalar_type() == torch::kFloat32
              && shift.scalar_type() == torch::kFloat32,
              "gate/scale/shift must use qualified FP32 modulation");
  TORCH_CHECK(scale.dim() == 2 && shift.sizes() == scale.sizes()
              && gate.sizes() == scale.sizes()
              && scale.size(0) > 0 && scale.size(1) == kWidth,
              "gate/scale/shift must share shape [M, 5376]");
  TORCH_CHECK(gate.stride(1) == 1 && scale.stride(1) == 1
              && shift.stride(1) == 1,
              "gate/scale/shift feature dimensions must be contiguous");
  TORCH_CHECK(!gate.requires_grad() && !scale.requires_grad()
              && !shift.requires_grad(),
              "gate/scale/shift must be inference-only");

  TORCH_CHECK(row_ids.is_cuda() && row_ids.scalar_type() == torch::kInt32
              && row_ids.dim() == 1 && row_ids.size(0) == x.size(0)
              && row_ids.is_contiguous() && !row_ids.requires_grad(),
              "row_ids must be contiguous inference-only int32 [S]");
  TORCH_CHECK(global_scale.is_cuda()
              && global_scale.scalar_type() == torch::kFloat32
              && global_scale.dim() == 0 && global_scale.is_contiguous()
              && !global_scale.requires_grad(),
              "global_scale must be one contiguous CUDA FP32 scalar");
  TORCH_CHECK(x.get_device() == other.get_device()
              && x.get_device() == gate.get_device()
              && x.get_device() == weight.get_device()
              && x.get_device() == scale.get_device()
              && x.get_device() == shift.get_device()
              && x.get_device() == row_ids.get_device()
              && x.get_device() == global_scale.get_device(),
              "all tensors must share one CUDA device");

  at::assert_no_internal_overlap(x);
  at::assert_no_overlap(x, other);
  at::assert_no_overlap(x, gate);
  at::assert_no_overlap(x, weight);
  at::assert_no_overlap(x, scale);
  at::assert_no_overlap(x, shift);
  at::assert_no_overlap(x, row_ids);
  at::assert_no_overlap(x, global_scale);
}

template <bool EmitRstd>
std::vector<torch::Tensor> gate_rms_adaln_nvfp4_static_impl(
    torch::Tensor x,
    torch::Tensor other,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    torch::Tensor global_scale,
    double eps) {
  check_inputs(
      x, other, gate, weight, scale, shift, row_ids, global_scale, eps);
  const c10::cuda::CUDAGuard guard_device(x.device());
  const int64_t padded_rows = ((x.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto bytes = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({padded_rows, kPackedRowBytes}, bytes);
  torch::Tensor block_scales = torch::zeros({scale_rows, kScaleCols}, bytes);
  torch::Tensor rstd = EmitRstd
      ? torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32))
      : torch::Tensor();
  const dim3 threads(32, kWarpCount, 1);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();

  gate_rms_adaln_nvfp4_static_kernel<EmitRstd><<<
      static_cast<unsigned int>(padded_rows), threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(other.data_ptr()),
      gate.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      scale.data_ptr<float>(), shift.data_ptr<float>(),
      row_ids.data_ptr<int32_t>(), global_scale.data_ptr<float>(),
      EmitRstd ? rstd.data_ptr<float>() : nullptr,
      packed.data_ptr<uint8_t>(), block_scales.data_ptr<uint8_t>(),
      x.size(0), gate.stride(0), scale.stride(0), shift.stride(0),
      scale.size(0), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Mirror the one logical in-place residual operation for aliases and saved
  // tensor version checks.  There is no legal stock fallback after this point.
  torch::autograd::increment_version(x);
  if constexpr (EmitRstd) {
    return {packed, block_scales, rstd};
  }
  return {packed, block_scales};
}

std::vector<torch::Tensor> gate_rms_adaln_nvfp4_static(
    torch::Tensor x,
    torch::Tensor other,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    torch::Tensor global_scale,
    double eps) {
  return gate_rms_adaln_nvfp4_static_impl<false>(
      x, other, gate, weight, scale, shift, row_ids, global_scale, eps);
}

std::vector<torch::Tensor> gate_rms_adaln_nvfp4_static_debug(
    torch::Tensor x,
    torch::Tensor other,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    torch::Tensor global_scale,
    double eps) {
  return gate_rms_adaln_nvfp4_static_impl<true>(
      x, other, gate, weight, scale, shift, row_ids, global_scale, eps);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("abi_version", []() { return 1; }, "Native static gate ABI version");
  m.def("build_id", []() {
    return "h3-gate-rms-adaln-nvfp4-static-torch2.13-cf30153-sm120-gb10-safe-v1";
  }, "Immutable native build capability ID");
  m.def("gate_rms_adaln_nvfp4_static",
        &gate_rms_adaln_nvfp4_static,
        "H3 residual gate + RMSNorm + AdaLN -> supplied-scale NVFP4 (CUDA)");
  m.def("gate_rms_adaln_nvfp4_static_debug",
        &gate_rms_adaln_nvfp4_static_debug,
        "Static gate fusion plus row rstd for byte-exact qualification (CUDA)");
}
