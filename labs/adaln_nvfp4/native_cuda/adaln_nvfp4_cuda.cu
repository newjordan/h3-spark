/*
 * Experimental exact-staged H3 segmented AdaLN -> NVFP4 fusion.
 *
 * Input is the already-materialized BF16 output of stock weighted RMSNorm.
 * Keeping that boundary makes the RMS reduction and its BF16 output identical
 * to the production path.  This kernel removes only the following full-size
 * BF16 AdaLN temporary before QKV/FC1 activation quantization.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kWidth = 5376;
constexpr int kFp4BlockSize = 16;
constexpr int kScaleCols = kWidth / kFp4BlockSize;  // 336
constexpr int kValuesPerThread = 8;
constexpr int kThreadsPerScale = kFp4BlockSize / kValuesPerThread;
constexpr int kThreadsPerBlock = 128;

static_assert(kWidth % kFp4BlockSize == 0);
static_assert((kWidth / kValuesPerThread) % 32 == 0,
              "Each H3 row must end on a warp boundary");
static_assert(kScaleCols % 4 == 0,
              "H3 scale rows require no right-side scale padding");

__device__ __forceinline__ size_t scale_factor_swizzled_offset(
    size_t row_idx, size_t col_idx, uint32_t col_length) {
  // cuBLAS block-scale layout: (row_block, col_block_group, 32, 4, 4).
  constexpr uint32_t kRowsPerTile = 128;
  constexpr uint32_t kRowsPerTileColumn = 32;
  constexpr uint32_t kColsPerTileColumn = 4;

  const size_t row_block = row_idx / kRowsPerTile;
  const size_t rem = row_idx % kRowsPerTile;
  const size_t d4 = rem / kRowsPerTileColumn;
  const size_t d3 = rem % kRowsPerTileColumn;
  const size_t col_block_group = col_idx / kColsPerTileColumn;
  const size_t d5 = col_idx % kColsPerTileColumn;
  const size_t col_block_group_count =
      (col_length + kColsPerTileColumn - 1) / kColsPerTileColumn;

  return ((row_block * col_block_group_count + col_block_group)
              * kRowsPerTileColumn
          + d3)
          * 16
      + d4 * kColsPerTileColumn + d5;
}

template <typename T>
__device__ __forceinline__ float modulation_to_float(T value);

template <>
__device__ __forceinline__ float modulation_to_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ float modulation_to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename ModT>
__device__ __forceinline__ __nv_bfloat16 staged_adaln_bf16(
    __nv_bfloat16 normed,
    ModT scale,
    ModT shift) {
  // Match exactly:
  //   h.mul_(1.0 + scale.to(bfloat16)).add_(shift.to(bfloat16))
  // The explicit BF16 values prevent contraction across any storage boundary.
  const __nv_bfloat16 scale_bf16 =
      __float2bfloat16_rn(modulation_to_float(scale));
  const __nv_bfloat16 multiplier_bf16 = __float2bfloat16_rn(
      1.0f + __bfloat162float(scale_bf16));
  const __nv_bfloat16 product_bf16 = __float2bfloat16_rn(
      __bfloat162float(normed) * __bfloat162float(multiplier_bf16));
  const __nv_bfloat16 shift_bf16 =
      __float2bfloat16_rn(modulation_to_float(shift));
  return __float2bfloat16_rn(
      __bfloat162float(product_bf16) + __bfloat162float(shift_bf16));
}

__device__ __forceinline__ float dynamic_scale_from_bf16_bits(
    const uint32_t* __restrict__ global_max_bits) {
  const __nv_bfloat16 absmax = __ushort_as_bfloat16(
      static_cast<unsigned short>(global_max_bits[0]));
  // TensorCoreNVFP4Layout performs this division in BF16, then widens.
  const __nv_bfloat16 scale_bf16 = __float2bfloat16_rn(
      __bfloat162float(absmax) / 2688.0f);
  return __bfloat162float(scale_bf16);
}

template <typename ModT>
__global__ void adaln_amax_bf16_bits_kernel(
    const __nv_bfloat16* __restrict__ normed,
    const ModT* __restrict__ scale,
    const ModT* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    uint32_t* __restrict__ global_max_bits,
    int64_t scale_row_stride,
    int64_t shift_row_stride,
    int64_t num_values) {
  uint32_t local_max = 0;
  // The H3 width is divisible by kValuesPerThread, so each vector starts and
  // ends within one row.  Resolve the expensive row division and modulation
  // row lookup once for eight adjacent BF16 values instead of once per scalar.
  // The result is still an integer max over the exact staged BF16 bit
  // patterns, making reduction order irrelevant for finite activations.
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x * kValuesPerThread;
  for (int64_t base =
           (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x)
               * kValuesPerThread;
       base < num_values;
       base += stride) {
    const int64_t row = base / kWidth;
    const int col = static_cast<int>(base - row * kWidth);
    const int32_t mod_row = row_ids[row];
    const int64_t scale_base =
        static_cast<int64_t>(mod_row) * scale_row_stride + col;
    const int64_t shift_base =
        static_cast<int64_t>(mod_row) * shift_row_stride + col;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      const __nv_bfloat16 value = staged_adaln_bf16(
          normed[base + i], scale[scale_base + i], shift[shift_base + i]);
      const uint32_t abs_bits = static_cast<uint32_t>(
          __bfloat16_as_ushort(value) & 0x7fffu);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
  }

  constexpr unsigned int kFullMask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1) {
    const uint32_t other =
        __shfl_down_sync(kFullMask, local_max, offset);
    local_max = local_max < other ? other : local_max;
  }

  __shared__ uint32_t warp_maxima[kThreadsPerBlock / 32];
  if (lane == 0) {
    warp_maxima[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    constexpr int kWarpCount = kThreadsPerBlock / 32;
    local_max = lane < kWarpCount ? warp_maxima[lane] : 0;
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
      const uint32_t other =
          __shfl_down_sync(kFullMask, local_max, offset);
      local_max = local_max < other ? other : local_max;
    }
    if (lane == 0) {
      atomicMax(
          reinterpret_cast<unsigned int*>(global_max_bits),
          static_cast<unsigned int>(local_max));
    }
  }
}

template <typename ModT, bool DynamicScale>
__global__ void adaln_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ normed,
    const ModT* __restrict__ scale,
    const ModT* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    const float* __restrict__ supplied_scale,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t scale_row_stride,
    int64_t shift_row_stride,
    int64_t orig_rows,
    int64_t padded_rows) {
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int kThreadsPerRow = kWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * kThreadsPerRow;
  if (thread_linear >= total_threads) {
    return;
  }

  const int64_t row = thread_linear / kThreadsPerRow;
  const int col = static_cast<int>(
      (thread_linear - row * kThreadsPerRow) * kValuesPerThread);

  float global_decode_scale;
  if constexpr (DynamicScale) {
    global_decode_scale = dynamic_scale_from_bf16_bits(global_max_bits);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      dynamic_scale_output[0] = global_decode_scale;
    }
  } else {
    global_decode_scale = supplied_scale[0];
  }

  float values[kValuesPerThread];
  if (row < orig_rows) {
    const int32_t mod_row = row_ids[row];
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      values[i] = __bfloat162float(staged_adaln_bf16(
          normed[row * kWidth + col + i],
          scale[static_cast<int64_t>(mod_row) * scale_row_stride + col + i],
          shift[static_cast<int64_t>(mod_row) * shift_row_stride + col + i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      values[i] = 0.0f;
    }
  }

  float absmax = fabsf(values[0]);
#pragma unroll
  for (int i = 1; i < kValuesPerThread; ++i) {
    absmax = fmaxf(absmax, fabsf(values[i]));
  }

  constexpr unsigned int kFullMask = 0xffffffffu;
  absmax = fmaxf(
      absmax,
      __shfl_down_sync(kFullMask, absmax, 1, kThreadsPerScale));
  const int lane = threadIdx.x & 31;
  const int leader = (lane / kThreadsPerScale) * kThreadsPerScale;
  absmax = __shfl_sync(kFullMask, absmax, leader, 32);

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
  const __nv_fp8_e4m3 scale_fp8 =
      static_cast<__nv_fp8_e4m3>(decode_scale);
  const float decode_scale_fp8 = static_cast<float>(scale_fp8);

  if ((threadIdx.x % kThreadsPerScale) == 0) {
    const int64_t scale_col = col / kFp4BlockSize;
    const size_t scale_offset = scale_factor_swizzled_offset(
        static_cast<size_t>(row),
        static_cast<size_t>(scale_col),
        kScaleCols);
    reinterpret_cast<__nv_fp8_e4m3*>(block_scales)[scale_offset] = scale_fp8;
  }

  const float encode_scale = fminf(
      __fdividef(1.0f, decode_scale_fp8 * global_decode_scale), FLT_MAX);
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    values[i] *= encode_scale;
  }

  union PackedFp4 {
    uint32_t u32;
    __nv_fp4x2_storage_t fp4x2[4];
  } packed;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // CUDA places args.x in the low nibble. Reverse each pair so logical even
    // elements use comfy-kitchen's high-nibble-first convention.
    packed.fp4x2[i] = __nv_cvt_float2_to_fp4x2(
        float2{values[2 * i + 1], values[2 * i]},
        __NV_E2M1,
        cudaRoundNearest);
  }
  *reinterpret_cast<uint32_t*>(
      output + thread_linear * (kValuesPerThread / 2)) = packed.u32;
}

void check_inputs(
    const torch::Tensor& normed,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids) {
  TORCH_CHECK(normed.is_cuda(), "normed must be a CUDA tensor");
  TORCH_CHECK(normed.scalar_type() == torch::kBFloat16,
              "normed must have dtype torch.bfloat16");
  TORCH_CHECK(normed.dim() == 2 && normed.size(1) == kWidth,
              "normed must have shape [S, 5376]");
  TORCH_CHECK(normed.size(0) > 0, "normed must contain at least one row");
  TORCH_CHECK(normed.is_contiguous(), "normed must be contiguous");

  TORCH_CHECK(scale.is_cuda() && shift.is_cuda(),
              "scale and shift must be CUDA tensors");
  TORCH_CHECK(
      scale.scalar_type() == torch::kBFloat16
          || scale.scalar_type() == torch::kFloat32,
      "scale must have dtype torch.bfloat16 or torch.float32");
  TORCH_CHECK(shift.scalar_type() == scale.scalar_type(),
              "scale and shift must have the same dtype");
  TORCH_CHECK(scale.dim() == 2 && shift.dim() == 2,
              "scale and shift must be 2D tensors");
  TORCH_CHECK(scale.sizes() == shift.sizes(),
              "scale and shift must have matching shapes");
  TORCH_CHECK(scale.size(0) > 0 && scale.size(1) == kWidth,
              "scale and shift must have shape [M, 5376]");
  TORCH_CHECK(scale.stride(1) == 1 && shift.stride(1) == 1,
              "scale and shift must be contiguous in their feature dimension");

  TORCH_CHECK(row_ids.is_cuda(), "row_ids must be a CUDA tensor");
  TORCH_CHECK(row_ids.scalar_type() == torch::kInt32,
              "row_ids must have dtype torch.int32");
  TORCH_CHECK(row_ids.dim() == 1 && row_ids.size(0) == normed.size(0),
              "row_ids must have shape [S]");
  TORCH_CHECK(row_ids.is_contiguous(), "row_ids must be contiguous");
  TORCH_CHECK(
      normed.get_device() == scale.get_device()
          && normed.get_device() == shift.get_device()
          && normed.get_device() == row_ids.get_device(),
      "all inputs must be on the same CUDA device");
}

void check_global_scale(
    const torch::Tensor& normed,
    const torch::Tensor& global_scale) {
  TORCH_CHECK(global_scale.is_cuda(), "global_scale must be a CUDA tensor");
  TORCH_CHECK(global_scale.scalar_type() == torch::kFloat32,
              "global_scale must have dtype torch.float32");
  TORCH_CHECK(global_scale.numel() == 1,
              "global_scale must contain exactly one value");
  TORCH_CHECK(global_scale.is_contiguous(), "global_scale must be contiguous");
  TORCH_CHECK(normed.get_device() == global_scale.get_device(),
              "normed and global_scale must be on the same CUDA device");
}

std::vector<torch::Tensor> allocate_outputs(const torch::Tensor& normed) {
  const int64_t padded_rows = ((normed.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto byte_options = normed.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty(
      {padded_rows, kWidth / 2}, byte_options);
  torch::Tensor block_scales = torch::zeros(
      {scale_rows, kScaleCols}, byte_options);
  return {packed, block_scales};
}

template <typename ModT>
void launch_quantizer(
    const torch::Tensor& normed,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    const torch::Tensor& supplied_scale,
    const torch::Tensor& max_bits,
    const torch::Tensor& dynamic_scale,
    std::vector<torch::Tensor>& outputs,
    bool dynamic) {
  const int64_t padded_rows = outputs[0].size(0);
  constexpr int kThreadsPerRow = kWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * kThreadsPerRow;
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      normed.get_device()).stream();

  if (dynamic) {
    adaln_nvfp4_kernel<ModT, true><<<
        blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(normed.data_ptr()),
        reinterpret_cast<const ModT*>(scale.data_ptr()),
        reinterpret_cast<const ModT*>(shift.data_ptr()),
        row_ids.data_ptr<int32_t>(),
        nullptr,
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        dynamic_scale.data_ptr<float>(),
        outputs[0].data_ptr<uint8_t>(),
        outputs[1].data_ptr<uint8_t>(),
        scale.stride(0),
        shift.stride(0),
        normed.size(0),
        padded_rows);
  } else {
    adaln_nvfp4_kernel<ModT, false><<<
        blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(normed.data_ptr()),
        reinterpret_cast<const ModT*>(scale.data_ptr()),
        reinterpret_cast<const ModT*>(shift.data_ptr()),
        row_ids.data_ptr<int32_t>(),
        supplied_scale.data_ptr<float>(),
        nullptr,
        nullptr,
        outputs[0].data_ptr<uint8_t>(),
        outputs[1].data_ptr<uint8_t>(),
        scale.stride(0),
        shift.stride(0),
        normed.size(0),
        padded_rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ModT>
void launch_amax(
    const torch::Tensor& normed,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    torch::Tensor& max_bits) {
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t value_groups =
      (normed.numel() + kValuesPerThread - 1) / kValuesPerThread;
  const int64_t needed_blocks =
      (value_groups + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      normed.get_device()).stream();
  adaln_amax_bf16_bits_kernel<ModT><<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(normed.data_ptr()),
      reinterpret_cast<const ModT*>(scale.data_ptr()),
      reinterpret_cast<const ModT*>(shift.data_ptr()),
      row_ids.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      scale.stride(0),
      shift.stride(0),
      normed.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> adaln_nvfp4(
    torch::Tensor normed,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    torch::Tensor global_scale) {
  check_inputs(normed, scale, shift, row_ids);
  check_global_scale(normed, global_scale);
  const c10::cuda::CUDAGuard device_guard(normed.device());
  std::vector<torch::Tensor> outputs = allocate_outputs(normed);
  const torch::Tensor empty;
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_quantizer<__nv_bfloat16>(
        normed, scale, shift, row_ids, global_scale, empty, empty, outputs, false);
  } else {
    launch_quantizer<float>(
        normed, scale, shift, row_ids, global_scale, empty, empty, outputs, false);
  }
  return outputs;
}

std::vector<torch::Tensor> adaln_nvfp4_dynamic(
    torch::Tensor normed,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids) {
  check_inputs(normed, scale, shift, row_ids);
  const c10::cuda::CUDAGuard device_guard(normed.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, normed.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, normed.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_outputs(normed);

  if (scale.scalar_type() == torch::kBFloat16) {
    launch_amax<__nv_bfloat16>(
        normed, scale, shift, row_ids, max_bits);
    launch_quantizer<__nv_bfloat16>(
        normed, scale, shift, row_ids, torch::Tensor(), max_bits,
        global_scale, outputs, true);
  } else {
    launch_amax<float>(normed, scale, shift, row_ids, max_bits);
    launch_quantizer<float>(
        normed, scale, shift, row_ids, torch::Tensor(), max_bits,
        global_scale, outputs, true);
  }
  outputs.push_back(global_scale);
  return outputs;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "adaln_nvfp4",
      &adaln_nvfp4,
      "Exact-staged H3 segmented AdaLN to NVFP4 (CUDA)",
      pybind11::arg("normed"),
      pybind11::arg("scale"),
      pybind11::arg("shift"),
      pybind11::arg("row_ids"),
      pybind11::arg("global_scale"));
  m.def(
      "adaln_nvfp4_dynamic",
      &adaln_nvfp4_dynamic,
      "Exact-staged H3 segmented AdaLN to dynamic NVFP4 (CUDA)",
      pybind11::arg("normed"),
      pybind11::arg("scale"),
      pybind11::arg("shift"),
      pybind11::arg("row_ids"));
}
