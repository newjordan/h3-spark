/*
 * Saturation certificate for supplied-scale NVFP4 pack.
 *
 * The static pack already computes per-16-element absmax and clamps
 * decode_scale = absmax / 6 / G at 448 (E4M3 max). That clamp is the
 * overflow: T > G * 2688. Layer quarantine of the four dynamic
 * exclusions is the wrong object; this flag is the per-call predicate.
 *
 * Lab-only. Does not patch serving.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kActivationWidth = 14336;
constexpr int kRawWidth = 2 * kActivationWidth;
constexpr int kFp4BlockSize = 16;
constexpr int kThreadsPerBlock = 128;
constexpr int kDirectValuesPerThread = 8;
constexpr int kDirectThreadsPerScale = kFp4BlockSize / kDirectValuesPerThread;
constexpr int kScaleCols = kActivationWidth / kFp4BlockSize;

__device__ __forceinline__ size_t scale_factor_swizzled_offset(
    size_t row_idx, size_t col_idx, uint32_t col_length) {
  constexpr uint32_t kTotalRowsPerBaseBlock = 128;
  constexpr uint32_t kRowsPerBaseBlockCol = 32;
  constexpr uint32_t kColsPerBaseBlockCol = 4;
  const size_t rb = row_idx / kTotalRowsPerBaseBlock;
  const size_t rem = row_idx % kTotalRowsPerBaseBlock;
  const size_t d4 = rem / kRowsPerBaseBlockCol;
  const size_t d3 = rem % kRowsPerBaseBlockCol;
  const size_t cbg = col_idx / kColsPerBaseBlockCol;
  const size_t d5 = col_idx % kColsPerBaseBlockCol;
  const size_t cbg_cnt =
      (col_length + kColsPerBaseBlockCol - 1) / kColsPerBaseBlockCol;
  return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16
      + d4 * kColsPerBaseBlockCol + d5;
}

__device__ __forceinline__ float swiglu_like_eager_bf16(
    __nv_bfloat16 gate_bf16, __nv_bfloat16 up_bf16) {
  const float gate = __bfloat162float(gate_bf16);
  const float up = __bfloat162float(up_bf16);
  const float silu = gate / (1.0f + expf(-gate));
  const float silu_bf16 = __bfloat162float(__float2bfloat16_rn(silu));
  return __bfloat162float(__float2bfloat16_rn(silu_bf16 * up));
}

__global__ void swiglu_pack_from_scale_sat_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const float* __restrict__ supplied_scale,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int* __restrict__ saturate,
    int64_t orig_rows,
    int64_t padded_rows) {
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t total_threads = padded_rows * threads_per_row;
  if (thread_linear >= total_threads) {
    return;
  }

  const float global_decode_scale = supplied_scale[0];
  const int64_t row = thread_linear / threads_per_row;
  const int col = static_cast<int>(
      (thread_linear - row * threads_per_row) * kDirectValuesPerThread);

  float values[kDirectValuesPerThread];
  if (row < orig_rows) {
    const __nv_bfloat16* gate = raw + row * kRawWidth + col;
    const __nv_bfloat16* up = gate + kActivationWidth;
    const uint4 gate_vec = *reinterpret_cast<const uint4*>(gate);
    const uint4 up_vec = *reinterpret_cast<const uint4*>(up);
    const __nv_bfloat16* gate_v =
        reinterpret_cast<const __nv_bfloat16*>(&gate_vec);
    const __nv_bfloat16* up_v =
        reinterpret_cast<const __nv_bfloat16*>(&up_vec);
#pragma unroll
    for (int i = 0; i < kDirectValuesPerThread; ++i) {
      values[i] = swiglu_like_eager_bf16(gate_v[i], up_v[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kDirectValuesPerThread; ++i) {
      values[i] = 0.0f;
    }
  }

  float absmax = fabsf(values[0]);
#pragma unroll
  for (int i = 1; i < kDirectValuesPerThread; ++i) {
    absmax = fmaxf(absmax, fabsf(values[i]));
  }

  constexpr unsigned int kFullMask = 0xffffffffu;
#pragma unroll
  for (int offset = kDirectThreadsPerScale / 2; offset >= 1; offset /= 2) {
    absmax = fmaxf(
        absmax,
        __shfl_down_sync(
            kFullMask, absmax, offset, kDirectThreadsPerScale));
  }
  const int lane = threadIdx.x & 31;
  const int leader =
      (lane / kDirectThreadsPerScale) * kDirectThreadsPerScale;
  absmax = __shfl_sync(kFullMask, absmax, leader, 32);

  // Exact validator overflow: T > G * 6 * 448. The stock pack clamps
  // this to 448 and keeps going. The certificate is the pre-clamp test.
  const float decode_unclamped =
      __fdividef(__fdividef(absmax, 6.0f), global_decode_scale);
  if ((threadIdx.x % kDirectThreadsPerScale) == 0 &&
      decode_unclamped > 448.0f) {
    atomicOr(saturate, 1);
  }

  float decode_scale = fminf(decode_unclamped, 448.0f);
  const __nv_fp8_e4m3 scale_fp8 =
      static_cast<__nv_fp8_e4m3>(decode_scale);
  const float decode_scale_fp8 = static_cast<float>(scale_fp8);

  if ((threadIdx.x % kDirectThreadsPerScale) == 0) {
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
  for (int i = 0; i < kDirectValuesPerThread; ++i) {
    values[i] *= encode_scale;
  }

  union PackedFp4 {
    uint32_t u32;
    __nv_fp4x2_storage_t fp4x2[4];
  } packed;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed.fp4x2[i] = __nv_cvt_float2_to_fp4x2(
        float2{values[2 * i + 1], values[2 * i]},
        __NV_E2M1,
        cudaRoundNearest);
  }
  *reinterpret_cast<uint32_t*>(
      output + thread_linear * (kDirectValuesPerThread / 2)) = packed.u32;
}

}  // namespace

std::vector<torch::Tensor> swiglu_static_rebind_sat(
    torch::Tensor raw, torch::Tensor global_scale) {
  TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth && raw.is_contiguous());
  TORCH_CHECK(raw.size(0) > 0);
  TORCH_CHECK(global_scale.is_cuda() && global_scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(global_scale.numel() == 1 && global_scale.is_contiguous());
  TORCH_CHECK(raw.get_device() == global_scale.get_device());

  const c10::cuda::CUDAGuard device_guard(raw.device());
  const int64_t orig_rows = raw.size(0);
  // Same padding as allocate_direct_outputs: pack rows % 16, scale rows % 128.
  const int64_t padded_rows = ((orig_rows + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto byte_options = raw.options().dtype(torch::kUInt8);
  auto packed = torch::empty({padded_rows, kActivationWidth / 2}, byte_options);
  auto block_scales = torch::zeros({scale_rows, kScaleCols}, byte_options);
  auto saturate = torch::zeros({1}, raw.options().dtype(torch::kInt32));

  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(raw.get_device()).stream();
  swiglu_pack_from_scale_sat_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      global_scale.data_ptr<float>(),
      packed.data_ptr<uint8_t>(),
      block_scales.data_ptr<uint8_t>(),
      saturate.data_ptr<int>(),
      orig_rows,
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, block_scales, saturate};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "swiglu_static_rebind_sat",
      &swiglu_static_rebind_sat,
      "Static SwiGLU NVFP4 pack plus per-call saturation certificate");
}
