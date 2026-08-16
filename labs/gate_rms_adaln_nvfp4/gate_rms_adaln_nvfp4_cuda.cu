/*
 * Isolated H3 attention gate + exact RMSNorm + segmented AdaLN -> dynamic
 * NVFP4 producer.
 *
 * The row reduction deliberately mirrors PyTorch cf30153's vectorized CUDA
 * RMSNorm: 128 threads as four warps, vector width four, the same per-thread
 * accumulation order, the same warp tree, and the same inter-warp tree.  The
 * mutating first pass applies the BF16 residual gate, computes row rstd, and
 * finds the exact post-AdaLN BF16 global maximum.  The second pass rereads the
 * gated residual and directly writes cuBLAS NVFP4 data and block scales.
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
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kWidth = 5376;
constexpr double kCertifiedEps = 1.0e-5;
constexpr int kVec = 4;
constexpr int kRmsWarpCount = 4;
constexpr int kRmsThreads = 32 * kRmsWarpCount;
constexpr int kFp4BlockSize = 16;
constexpr int kScaleCols = kWidth / kFp4BlockSize;
constexpr int kValuesPerThread = 8;
constexpr int kQuantThreads = 128;
constexpr int kVectorsPerRmsThread =
    ((kWidth / kVec) + kRmsThreads - 1) / kRmsThreads;

static_assert(kWidth % 256 == 0, "H3 RMS vector/reduction tiles must divide width");
static_assert(kWidth % kFp4BlockSize == 0, "H3 width must divide FP4 blocks");
static_assert(kScaleCols % 4 == 0, "H3 scale rows require no right padding");

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
  T value[N];
};

__device__ __forceinline__ size_t scale_factor_swizzled_offset(
    size_t row_idx, size_t col_idx, uint32_t col_length) {
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

template <typename T>
__device__ __forceinline__ float mod_float(T value);

template <>
__device__ __forceinline__ float mod_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ float mod_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 weighted_rms_bf16(
    __nv_bfloat16 x, __nv_bfloat16 weight, float rstd) {
  // Match vectorized_layer_norm_kernel's gamma * (rstd * x), then BF16 store.
  const float normalized = rstd * __bfloat162float(x);
  const float weighted = __bfloat162float(weight) * normalized;
  return __float2bfloat16_rn(weighted);
}

template <typename ModT>
__device__ __forceinline__ __nv_bfloat16 staged_adaln_bf16(
    __nv_bfloat16 normed, ModT scale, ModT shift) {
  const __nv_bfloat16 scale_bf16 = __float2bfloat16_rn(mod_float(scale));
  const __nv_bfloat16 multiplier_bf16 = __float2bfloat16_rn(
      1.0f + __bfloat162float(scale_bf16));
  const __nv_bfloat16 product_bf16 = __float2bfloat16_rn(
      __bfloat162float(normed) * __bfloat162float(multiplier_bf16));
  const __nv_bfloat16 shift_bf16 = __float2bfloat16_rn(mod_float(shift));
  return __float2bfloat16_rn(
      __bfloat162float(product_bf16) + __bfloat162float(shift_bf16));
}

__device__ __forceinline__ float rms_row_rstd(
    const __nv_bfloat16* __restrict__ row,
    int width,
    float eps,
    float* __restrict__ warp_scratch,
    volatile float* __restrict__ rstd_shared) {
  using Vec = AlignedVector<__nv_bfloat16, kVec>;
  const Vec* row_vec = reinterpret_cast<const Vec*>(row);
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = lane + warp * 32;
  const int vector_count = width / kVec;
  float sigma2 = 0.0f;
  for (int i = tid; i < vector_count; i += kRmsThreads) {
    const Vec data = row_vec[i];
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float value = __bfloat162float(data.value[j]);
      sigma2 = sigma2 + value * value;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sigma2 = sigma2 + __shfl_down_sync(0xffffffffu, sigma2, offset);
  }

  // Same four-warp binary tree used by PyTorch compute_stats():
  // (warp0 + warp2) + (warp1 + warp3). Store all four first so the exact
  // tree is explicit and no shared-memory slot is reused between barriers.
  if (lane == 0) {
    warp_scratch[warp] = sigma2;
  }
  __syncthreads();
  if (lane == 0 && warp == 0) {
    const float left = warp_scratch[0] + warp_scratch[2];
    const float right = warp_scratch[1] + warp_scratch[3];
    sigma2 = left + right;
    rstd_shared[0] = rsqrtf(sigma2 / static_cast<float>(width) + eps);
  }
  __syncthreads();
  return rstd_shared[0];
}

template <typename ModT>
__global__ void gate_rms_adaln_first_pass(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ other,
    const ModT* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    const ModT* __restrict__ scale,
    const ModT* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    float* __restrict__ rstd_output,
    uint32_t* __restrict__ global_max_bits,
    __nv_bfloat16* __restrict__ adaln_output,
    int64_t gate_stride,
    int64_t scale_stride,
    int64_t shift_stride,
    int64_t modulation_rows,
    float eps) {
  __shared__ float rms_scratch[kRmsWarpCount];
  __shared__ volatile float rstd_shared;
  __shared__ uint32_t max_scratch[kRmsWarpCount];
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = lane + warp * 32;
  const int32_t selected_mod_row = row_ids[row];
  const int32_t mod_row =
      selected_mod_row >= 0 && selected_mod_row < modulation_rows
      ? selected_mod_row : 0;

  using Vec = AlignedVector<__nv_bfloat16, kVec>;
  Vec cached[kVectorsPerRmsThread];
  const int vector_count = kWidth / kVec;
  Vec* x_vec = reinterpret_cast<Vec*>(x + row * static_cast<int64_t>(kWidth));
  const Vec* other_vec = reinterpret_cast<const Vec*>(
      other + row * static_cast<int64_t>(kWidth));
  float sigma2 = 0.0f;
  #pragma unroll
  for (int n = 0; n < kVectorsPerRmsThread; ++n) {
    const int i = tid + n * kRmsThreads;
    if (i < vector_count) {
      const Vec x_data = x_vec[i];
      const Vec other_data = other_vec[i];
      Vec updated;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const int col = i * kVec + j;
        const __nv_bfloat16 gate_bf16 = __float2bfloat16_rn(mod_float(
            gate[static_cast<int64_t>(mod_row) * gate_stride + col]));
        const float value = fmaf(
            __bfloat162float(other_data.value[j]),
            __bfloat162float(gate_bf16),
            __bfloat162float(x_data.value[j]));
        updated.value[j] = __float2bfloat16_rn(value);
        const float rounded = __bfloat162float(updated.value[j]);
        sigma2 = sigma2 + rounded * rounded;
      }
      cached[n] = updated;
      x_vec[i] = updated;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sigma2 = sigma2 + __shfl_down_sync(0xffffffffu, sigma2, offset);
  }
  if (lane == 0) {
    rms_scratch[warp] = sigma2;
  }
  __syncthreads();
  if (lane == 0 && warp == 0) {
    const float left = rms_scratch[0] + rms_scratch[2];
    const float right = rms_scratch[1] + rms_scratch[3];
    rstd_shared = rsqrtf(
        (left + right) / static_cast<float>(kWidth) + eps);
  }
  __syncthreads();
  const float rstd = rstd_shared;
  if (tid == 0) {
    rstd_output[row] = rstd;
  }

  uint32_t local_max = 0;
  #pragma unroll
  for (int n = 0; n < kVectorsPerRmsThread; ++n) {
    const int i = tid + n * kRmsThreads;
    if (i < vector_count) {
      const Vec data = cached[n];
      Vec adaln;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const int col = i * kVec + j;
        const __nv_bfloat16 normed = weighted_rms_bf16(
            data.value[j], weight[col], rstd);
        const __nv_bfloat16 value = staged_adaln_bf16(
            normed,
            scale[static_cast<int64_t>(mod_row) * scale_stride + col],
            shift[static_cast<int64_t>(mod_row) * shift_stride + col]);
        adaln.value[j] = value;
        const uint32_t bits = static_cast<uint32_t>(
            __bfloat16_as_ushort(value) & 0x7fffu);
        local_max = local_max < bits ? bits : local_max;
      }
      if (adaln_output != nullptr) {
        reinterpret_cast<Vec*>(
            adaln_output + row * static_cast<int64_t>(kWidth))[i] = adaln;
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const uint32_t other_max =
        __shfl_down_sync(0xffffffffu, local_max, offset);
    local_max = local_max < other_max ? other_max : local_max;
  }
  if (lane == 0) {
    max_scratch[warp] = local_max;
  }
  __syncthreads();
  if (warp == 0) {
    local_max = lane < kRmsWarpCount ? max_scratch[lane] : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const uint32_t other_max =
          __shfl_down_sync(0xffffffffu, local_max, offset);
      local_max = local_max < other_max ? other_max : local_max;
    }
    if (lane == 0) {
      atomicMax(global_max_bits, local_max);
    }
  }
}

template <typename ModT, bool EmitOutput, bool FindGlobalMax>
__global__ void rms_adaln_first_pass(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const ModT* __restrict__ scale,
    const ModT* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    float* __restrict__ rstd_output,
    __nv_bfloat16* __restrict__ materialized_output,
    uint32_t* __restrict__ global_max_bits,
    int64_t scale_stride,
    int64_t shift_stride,
    int width,
    int64_t modulation_rows,
    float eps) {
  __shared__ float rms_scratch[kRmsWarpCount];
  __shared__ volatile float rstd_shared;
  __shared__ uint32_t max_scratch[kRmsWarpCount];
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = lane + warp * 32;
  const float rstd = rms_row_rstd(
      x + row * static_cast<int64_t>(width), width, eps, rms_scratch,
      &rstd_shared);
  if (tid == 0) {
    rstd_output[row] = rstd;
  }

  const int32_t selected_mod_row = row_ids[row];
  const int32_t mod_row =
      selected_mod_row >= 0 && selected_mod_row < modulation_rows
      ? selected_mod_row : 0;
  uint32_t local_max = 0;
  using Vec = AlignedVector<__nv_bfloat16, kVec>;
  const Vec* x_vec = reinterpret_cast<const Vec*>(
      x + row * static_cast<int64_t>(width));
  Vec* out_vec = EmitOutput
      ? reinterpret_cast<Vec*>(materialized_output + row * static_cast<int64_t>(width))
      : nullptr;
  const int vector_count = width / kVec;
  for (int i = tid; i < vector_count; i += kRmsThreads) {
    const Vec data = x_vec[i];
    Vec out;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const int col = i * kVec + j;
      const __nv_bfloat16 normed = weighted_rms_bf16(
          data.value[j], weight[col], rstd);
      const __nv_bfloat16 value = staged_adaln_bf16(
          normed,
          scale[static_cast<int64_t>(mod_row) * scale_stride + col],
          shift[static_cast<int64_t>(mod_row) * shift_stride + col]);
      out.value[j] = value;
      if constexpr (FindGlobalMax) {
        const uint32_t bits = static_cast<uint32_t>(
            __bfloat16_as_ushort(value) & 0x7fffu);
        local_max = local_max < bits ? bits : local_max;
      }
    }
    if constexpr (EmitOutput) {
      out_vec[i] = out;
    }
  }

  if constexpr (FindGlobalMax) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const uint32_t other =
          __shfl_down_sync(0xffffffffu, local_max, offset);
      local_max = local_max < other ? other : local_max;
    }
    if (lane == 0) {
      max_scratch[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
      local_max = lane < kRmsWarpCount ? max_scratch[lane] : 0;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        const uint32_t other =
            __shfl_down_sync(0xffffffffu, local_max, offset);
        local_max = local_max < other ? other : local_max;
      }
      if (lane == 0) {
        atomicMax(global_max_bits, local_max);
      }
    }
  }
}

template <typename ModT, bool EmitOutput, bool FindGlobalMax>
__global__ void rms_adaln_first_pass_cached(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const ModT* __restrict__ scale,
    const ModT* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    float* __restrict__ rstd_output,
    __nv_bfloat16* __restrict__ materialized_output,
    uint32_t* __restrict__ global_max_bits,
    int64_t scale_stride,
    int64_t shift_stride,
    int width,
    int64_t modulation_rows,
    float eps) {
  // Same RMS tree and AdaLN as rms_adaln_first_pass, but the row is
  // loaded once into registers. Removes the second HBM read of x.
  __shared__ float rms_scratch[kRmsWarpCount];
  __shared__ volatile float rstd_shared;
  __shared__ uint32_t max_scratch[kRmsWarpCount];
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = lane + warp * 32;
  using Vec = AlignedVector<__nv_bfloat16, kVec>;
  const Vec* x_vec = reinterpret_cast<const Vec*>(
      x + row * static_cast<int64_t>(width));
  const int vector_count = width / kVec;
  Vec cached[kVectorsPerRmsThread];
  float sigma2 = 0.0f;
#pragma unroll
  for (int n = 0; n < kVectorsPerRmsThread; ++n) {
    const int i = tid + n * kRmsThreads;
    if (i < vector_count) {
      const Vec data = x_vec[i];
      cached[n] = data;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float value = __bfloat162float(data.value[j]);
        sigma2 = sigma2 + value * value;
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sigma2 = sigma2 + __shfl_down_sync(0xffffffffu, sigma2, offset);
  }
  if (lane == 0) {
    rms_scratch[warp] = sigma2;
  }
  __syncthreads();
  if (lane == 0 && warp == 0) {
    const float left = rms_scratch[0] + rms_scratch[2];
    const float right = rms_scratch[1] + rms_scratch[3];
    sigma2 = left + right;
    rstd_shared = rsqrtf(sigma2 / static_cast<float>(width) + eps);
  }
  __syncthreads();
  const float rstd = rstd_shared;
  if (tid == 0) {
    rstd_output[row] = rstd;
  }

  const int32_t selected_mod_row = row_ids[row];
  const int32_t mod_row =
      selected_mod_row >= 0 && selected_mod_row < modulation_rows
      ? selected_mod_row : 0;
  uint32_t local_max = 0;
  Vec* out_vec = EmitOutput
      ? reinterpret_cast<Vec*>(materialized_output + row * static_cast<int64_t>(width))
      : nullptr;
#pragma unroll
  for (int n = 0; n < kVectorsPerRmsThread; ++n) {
    const int i = tid + n * kRmsThreads;
    if (i < vector_count) {
      const Vec data = cached[n];
      Vec out;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const int col = i * kVec + j;
        const __nv_bfloat16 normed = weighted_rms_bf16(
            data.value[j], weight[col], rstd);
        const __nv_bfloat16 value = staged_adaln_bf16(
            normed,
            scale[static_cast<int64_t>(mod_row) * scale_stride + col],
            shift[static_cast<int64_t>(mod_row) * shift_stride + col]);
        out.value[j] = value;
        if constexpr (FindGlobalMax) {
          const uint32_t bits = static_cast<uint32_t>(
              __bfloat16_as_ushort(value) & 0x7fffu);
          local_max = local_max < bits ? bits : local_max;
        }
      }
      if constexpr (EmitOutput) {
        out_vec[i] = out;
      }
    }
  }

  if constexpr (FindGlobalMax) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const uint32_t other =
          __shfl_down_sync(0xffffffffu, local_max, offset);
      local_max = local_max < other ? other : local_max;
    }
    if (lane == 0) {
      max_scratch[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
      local_max = lane < kRmsWarpCount ? max_scratch[lane] : 0;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        const uint32_t other =
            __shfl_down_sync(0xffffffffu, local_max, offset);
        local_max = local_max < other ? other : local_max;
      }
      if (lane == 0) {
        atomicMax(global_max_bits, local_max);
      }
    }
  }
}

__device__ __forceinline__ float dynamic_scale_from_bf16_bits(
    const uint32_t* __restrict__ global_max_bits) {
  const __nv_bfloat16 absmax = __ushort_as_bfloat16(
      static_cast<unsigned short>(global_max_bits[0]));
  const __nv_bfloat16 scale_bf16 = __float2bfloat16_rn(
      __bfloat162float(absmax) / 2688.0f);
  return __bfloat162float(scale_bf16);
}

template <typename ModT>
__global__ void rms_adaln_nvfp4_second_pass(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const ModT* __restrict__ scale,
    const ModT* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    const float* __restrict__ rstd,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t scale_stride,
    int64_t shift_stride,
    int64_t original_rows,
    int64_t padded_rows,
    int64_t modulation_rows) {
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
  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (thread_linear == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

  float values[kValuesPerThread];
  if (row < original_rows) {
    const int32_t selected_mod_row = row_ids[row];
    const int32_t mod_row =
        selected_mod_row >= 0 && selected_mod_row < modulation_rows
        ? selected_mod_row : 0;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      const __nv_bfloat16 normed = weighted_rms_bf16(
          x[row * kWidth + col + i], weight[col + i], rstd[row]);
      values[i] = __bfloat162float(staged_adaln_bf16(
          normed,
          scale[static_cast<int64_t>(mod_row) * scale_stride + col + i],
          shift[static_cast<int64_t>(mod_row) * shift_stride + col + i]));
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
  absmax = fmaxf(
      absmax,
      __shfl_down_sync(0xffffffffu, absmax, 1, kFp4BlockSize / kValuesPerThread));
  const int lane = threadIdx.x & 31;
  const int leader = (lane / 2) * 2;
  absmax = __shfl_sync(0xffffffffu, absmax, leader, 32);

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
  const __nv_fp8_e4m3 scale_fp8 = static_cast<__nv_fp8_e4m3>(decode_scale);
  const float decode_scale_fp8 = static_cast<float>(scale_fp8);
  if ((threadIdx.x & 1) == 0) {
    const size_t scale_offset = scale_factor_swizzled_offset(
        static_cast<size_t>(row), static_cast<size_t>(col / kFp4BlockSize),
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
    packed.fp4x2[i] = __nv_cvt_float2_to_fp4x2(
        float2{values[2 * i + 1], values[2 * i]},
        __NV_E2M1,
        cudaRoundNearest);
  }
  *reinterpret_cast<uint32_t*>(
      output + thread_linear * (kValuesPerThread / 2)) = packed.u32;
}

template <typename ModT>
__device__ __forceinline__ void load_mod8(
    const ModT* __restrict__ ptr, ModT out[kValuesPerThread]) {
  if constexpr (sizeof(ModT) == 2) {
    const uint4 vec = *reinterpret_cast<const uint4*>(ptr);
    const ModT* src = reinterpret_cast<const ModT*>(&vec);
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      out[i] = src[i];
    }
  } else {
    const uint4 v0 = *reinterpret_cast<const uint4*>(ptr);
    const uint4 v1 = *reinterpret_cast<const uint4*>(ptr + 4);
    const ModT* s0 = reinterpret_cast<const ModT*>(&v0);
    const ModT* s1 = reinterpret_cast<const ModT*>(&v1);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[i] = s0[i];
      out[i + 4] = s1[i];
    }
  }
}

template <typename ModT>
__global__ void rms_adaln_nvfp4_second_pass_vec(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const ModT* __restrict__ scale,
    const ModT* __restrict__ shift,
    const int32_t* __restrict__ row_ids,
    const float* __restrict__ rstd,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t scale_stride,
    int64_t shift_stride,
    int64_t original_rows,
    int64_t padded_rows,
    int64_t modulation_rows) {
  // Same AdaLN and NVFP4 algebra as the scalar second pass. 8-wide
  // packed loads of residual, weight, and modulation. rstd is reused;
  // the RMS tree is not recomputed.
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
  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (thread_linear == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

  float values[kValuesPerThread];
  if (row < original_rows) {
    const int32_t selected_mod_row = row_ids[row];
    const int32_t mod_row =
        selected_mod_row >= 0 && selected_mod_row < modulation_rows
        ? selected_mod_row : 0;
    const uint4 x_vec = *reinterpret_cast<const uint4*>(
        x + row * kWidth + col);
    const uint4 w_vec = *reinterpret_cast<const uint4*>(weight + col);
    const __nv_bfloat16* x_v =
        reinterpret_cast<const __nv_bfloat16*>(&x_vec);
    const __nv_bfloat16* w_v =
        reinterpret_cast<const __nv_bfloat16*>(&w_vec);
    ModT scale_v[kValuesPerThread];
    ModT shift_v[kValuesPerThread];
    load_mod8(
        scale + static_cast<int64_t>(mod_row) * scale_stride + col,
        scale_v);
    load_mod8(
        shift + static_cast<int64_t>(mod_row) * shift_stride + col,
        shift_v);
    const float row_rstd = rstd[row];
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      const __nv_bfloat16 normed = weighted_rms_bf16(
          x_v[i], w_v[i], row_rstd);
      values[i] = __bfloat162float(
          staged_adaln_bf16(normed, scale_v[i], shift_v[i]));
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
  absmax = fmaxf(
      absmax,
      __shfl_down_sync(0xffffffffu, absmax, 1, kFp4BlockSize / kValuesPerThread));
  const int lane = threadIdx.x & 31;
  const int leader = (lane / 2) * 2;
  absmax = __shfl_sync(0xffffffffu, absmax, leader, 32);

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
  const __nv_fp8_e4m3 scale_fp8 = static_cast<__nv_fp8_e4m3>(decode_scale);
  const float decode_scale_fp8 = static_cast<float>(scale_fp8);
  if ((threadIdx.x & 1) == 0) {
    const size_t scale_offset = scale_factor_swizzled_offset(
        static_cast<size_t>(row), static_cast<size_t>(col / kFp4BlockSize),
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
    packed.fp4x2[i] = __nv_cvt_float2_to_fp4x2(
        float2{values[2 * i + 1], values[2 * i]},
        __NV_E2M1,
        cudaRoundNearest);
  }
  *reinterpret_cast<uint32_t*>(
      output + thread_linear * (kValuesPerThread / 2)) = packed.u32;
}

__global__ void bf16_nvfp4_from_amax_bits_kernel(
    const __nv_bfloat16* __restrict__ input,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t original_rows,
    int64_t padded_rows) {
  // Pack already-materialized AdaLN BF16. Same quantizer as the second pass.
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
  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (thread_linear == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

  float values[kValuesPerThread];
  if (row < original_rows) {
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      values[i] = __bfloat162float(input[row * kWidth + col + i]);
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
  absmax = fmaxf(
      absmax,
      __shfl_down_sync(0xffffffffu, absmax, 1, kFp4BlockSize / kValuesPerThread));
  const int lane = threadIdx.x & 31;
  const int leader = (lane / 2) * 2;
  absmax = __shfl_sync(0xffffffffu, absmax, leader, 32);

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
  const __nv_fp8_e4m3 scale_fp8 = static_cast<__nv_fp8_e4m3>(decode_scale);
  const float decode_scale_fp8 = static_cast<float>(scale_fp8);
  if ((threadIdx.x & 1) == 0) {
    const size_t scale_offset = scale_factor_swizzled_offset(
        static_cast<size_t>(row), static_cast<size_t>(col / kFp4BlockSize),
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
    packed.fp4x2[i] = __nv_cvt_float2_to_fp4x2(
        float2{values[2 * i + 1], values[2 * i]},
        __NV_E2M1,
        cudaRoundNearest);
  }
  *reinterpret_cast<uint32_t*>(
      output + thread_linear * (kValuesPerThread / 2)) = packed.u32;
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    double eps) {
  TORCH_CHECK(!c10::GradMode::is_enabled(),
              "native H3 RMS/AdaLN producers are inference-only");
  TORCH_CHECK(eps == kCertifiedEps,
              "eps must equal the certified H3 value 1e-5");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
              "x must be CUDA BF16");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) == kWidth,
              "x must have nonempty shape [S, 5376]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(!x.requires_grad(), "x must not require gradients");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(x.data_ptr())
          % (sizeof(__nv_bfloat16) * kVec) == 0,
      "x storage offset must preserve 8-byte vector alignment");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kBFloat16,
              "weight must be CUDA BF16");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kWidth
              && weight.is_contiguous(), "weight must be contiguous [5376]");
  TORCH_CHECK(!weight.requires_grad(), "weight must not require gradients");
  TORCH_CHECK(scale.is_cuda() && shift.is_cuda(), "scale/shift must be CUDA");
  TORCH_CHECK(scale.scalar_type() == torch::kBFloat16
              || scale.scalar_type() == torch::kFloat32,
              "scale must be BF16 or FP32");
  TORCH_CHECK(shift.scalar_type() == scale.scalar_type(),
              "scale/shift dtype mismatch");
  TORCH_CHECK(scale.dim() == 2 && shift.sizes() == scale.sizes()
              && scale.size(0) > 0 && scale.size(1) == kWidth,
              "scale/shift must be [M, 5376]");
  TORCH_CHECK(scale.stride(1) == 1 && shift.stride(1) == 1,
              "scale/shift feature dimension must be contiguous");
  TORCH_CHECK(!scale.requires_grad() && !shift.requires_grad(),
              "scale/shift must not require gradients");
  TORCH_CHECK(row_ids.is_cuda() && row_ids.scalar_type() == torch::kInt32
              && row_ids.dim() == 1 && row_ids.size(0) == x.size(0)
              && row_ids.is_contiguous(), "row_ids must be contiguous int32 [S]");
  TORCH_CHECK(x.get_device() == weight.get_device()
              && x.get_device() == scale.get_device()
              && x.get_device() == shift.get_device()
              && x.get_device() == row_ids.get_device(),
              "all tensors must share one CUDA device");
  at::assert_no_internal_overlap(x);
  // The integration identity token proves the production map. Kernels still
  // device-check every selected row before any table dereference, keeping the
  // native ABI memory-safe without GPU reductions or host synchronizations.
}

void check_gate_inputs(
    const torch::Tensor& x,
    const torch::Tensor& other,
    const torch::Tensor& gate,
    const torch::Tensor& scale) {
  TORCH_CHECK(other.is_cuda() && other.scalar_type() == torch::kBFloat16,
              "other must be CUDA BF16");
  TORCH_CHECK(other.sizes() == x.sizes() && other.is_contiguous(),
              "other must match contiguous x");
  TORCH_CHECK(!other.requires_grad(), "other must not require gradients");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(other.data_ptr())
          % (sizeof(__nv_bfloat16) * kVec) == 0,
      "other storage offset must preserve 8-byte vector alignment");
  TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == scale.scalar_type(),
              "gate must be CUDA and match modulation dtype");
  TORCH_CHECK(gate.dim() == 2 && gate.size(0) == scale.size(0)
              && gate.size(1) == kWidth && gate.stride(1) == 1,
              "gate must match modulation rows with contiguous width 5376");
  TORCH_CHECK(!gate.requires_grad(), "gate must not require gradients");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
              "gate producer is qualified only for FP32 modulation");
  TORCH_CHECK(x.get_device() == other.get_device()
              && x.get_device() == gate.get_device(),
              "gate inputs must share x's CUDA device");
  at::assert_no_overlap(x, other);
  at::assert_no_overlap(x, gate);
  at::assert_no_overlap(x, scale);
}

void check_mutation_aliases(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids) {
  at::assert_no_overlap(x, weight);
  at::assert_no_overlap(x, shift);
  at::assert_no_overlap(x, row_ids);
}

template <typename ModT, bool EmitOutput, bool FindGlobalMax>
void launch_first_pass(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    torch::Tensor& rstd,
    const torch::Tensor& materialized,
    const torch::Tensor& max_bits,
    double eps) {
  const dim3 threads(32, kRmsWarpCount, 1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  rms_adaln_first_pass<ModT, EmitOutput, FindGlobalMax><<<
      static_cast<unsigned int>(x.size(0)), threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const ModT*>(scale.data_ptr()),
      reinterpret_cast<const ModT*>(shift.data_ptr()),
      row_ids.data_ptr<int32_t>(),
      rstd.data_ptr<float>(),
      EmitOutput
          ? reinterpret_cast<__nv_bfloat16*>(materialized.data_ptr()) : nullptr,
      FindGlobalMax
          ? reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()) : nullptr,
      scale.stride(0), shift.stride(0), kWidth, scale.size(0),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ModT>
void launch_gate_first_pass(
    torch::Tensor& x,
    const torch::Tensor& other,
    const torch::Tensor& gate,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    torch::Tensor& rstd,
    torch::Tensor& max_bits,
    double eps,
    __nv_bfloat16* adaln_output = nullptr) {
  const dim3 threads(32, kRmsWarpCount, 1);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  gate_rms_adaln_first_pass<ModT><<<
      static_cast<unsigned int>(x.size(0)), threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(other.data_ptr()),
      reinterpret_cast<const ModT*>(gate.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const ModT*>(scale.data_ptr()),
      reinterpret_cast<const ModT*>(shift.data_ptr()),
      row_ids.data_ptr<int32_t>(), rstd.data_ptr<float>(),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      adaln_output,
      gate.stride(0), scale.stride(0), shift.stride(0), scale.size(0),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> rms_adaln_bf16(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    double eps) {
  check_inputs(x, weight, scale, shift, row_ids, eps);
  const c10::cuda::CUDAGuard guard(x.device());
  torch::Tensor output = torch::empty_like(x);
  torch::Tensor rstd = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  const torch::Tensor empty;
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_first_pass<__nv_bfloat16, true, false>(
        x, weight, scale, shift, row_ids, rstd, output, empty, eps);
  } else {
    launch_first_pass<float, true, false>(
        x, weight, scale, shift, row_ids, rstd, output, empty, eps);
  }
  return {output, rstd};
}

std::vector<torch::Tensor> rms_adaln_nvfp4_dynamic(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    double eps) {
  check_inputs(x, weight, scale, shift, row_ids, eps);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t padded_rows = ((x.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto bytes = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({padded_rows, kWidth / 2}, bytes);
  torch::Tensor block_scales = torch::zeros({scale_rows, kScaleCols}, bytes);
  torch::Tensor rstd = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  torch::Tensor max_bits = torch::zeros({1}, x.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty({1}, x.options().dtype(torch::kFloat32));
  const torch::Tensor empty;
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_first_pass<__nv_bfloat16, false, true>(
        x, weight, scale, shift, row_ids, rstd, empty, max_bits, eps);
  } else {
    launch_first_pass<float, false, true>(
        x, weight, scale, shift, row_ids, rstd, empty, max_bits, eps);
  }

  constexpr int kThreadsPerRow = kWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * kThreadsPerRow;
  const int blocks = static_cast<int>((total_threads + kQuantThreads - 1) / kQuantThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (scale.scalar_type() == torch::kBFloat16) {
    rms_adaln_nvfp4_second_pass<__nv_bfloat16><<<blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr()),
        row_ids.data_ptr<int32_t>(), rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  } else {
    rms_adaln_nvfp4_second_pass<float><<<blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        scale.data_ptr<float>(), shift.data_ptr<float>(), row_ids.data_ptr<int32_t>(),
        rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, block_scales, global_scale};
}

template <typename ModT, bool EmitOutput, bool FindGlobalMax>
void launch_first_pass_cached(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    torch::Tensor& rstd,
    const torch::Tensor& materialized,
    const torch::Tensor& max_bits,
    double eps) {
  const dim3 threads(32, kRmsWarpCount, 1);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  rms_adaln_first_pass_cached<ModT, EmitOutput, FindGlobalMax><<<
      static_cast<unsigned int>(x.size(0)), threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const ModT*>(scale.data_ptr()),
      reinterpret_cast<const ModT*>(shift.data_ptr()),
      row_ids.data_ptr<int32_t>(),
      rstd.data_ptr<float>(),
      EmitOutput
          ? reinterpret_cast<__nv_bfloat16*>(materialized.data_ptr()) : nullptr,
      FindGlobalMax
          ? reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()) : nullptr,
      scale.stride(0), shift.stride(0), kWidth, scale.size(0),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> rms_adaln_nvfp4_dynamic_cached(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    double eps) {
  check_inputs(x, weight, scale, shift, row_ids, eps);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t padded_rows = ((x.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto bytes = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({padded_rows, kWidth / 2}, bytes);
  torch::Tensor block_scales = torch::zeros({scale_rows, kScaleCols}, bytes);
  torch::Tensor rstd = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  torch::Tensor max_bits = torch::zeros({1}, x.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty({1}, x.options().dtype(torch::kFloat32));
  const torch::Tensor empty;
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_first_pass_cached<__nv_bfloat16, false, true>(
        x, weight, scale, shift, row_ids, rstd, empty, max_bits, eps);
  } else {
    launch_first_pass_cached<float, false, true>(
        x, weight, scale, shift, row_ids, rstd, empty, max_bits, eps);
  }

  constexpr int kThreadsPerRow = kWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * kThreadsPerRow;
  const int blocks = static_cast<int>((total_threads + kQuantThreads - 1) / kQuantThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (scale.scalar_type() == torch::kBFloat16) {
    rms_adaln_nvfp4_second_pass<__nv_bfloat16><<<blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr()),
        row_ids.data_ptr<int32_t>(), rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  } else {
    rms_adaln_nvfp4_second_pass<float><<<blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        scale.data_ptr<float>(), shift.data_ptr<float>(), row_ids.data_ptr<int32_t>(),
        rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, block_scales, global_scale, rstd};
}

std::vector<torch::Tensor> gate_rms_adaln_nvfp4_dynamic(
    torch::Tensor x,
    torch::Tensor other,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    double eps) {
  check_inputs(x, weight, scale, shift, row_ids, eps);
  check_gate_inputs(x, other, gate, scale);
  check_mutation_aliases(x, weight, shift, row_ids);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t padded_rows = ((x.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto bytes = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({padded_rows, kWidth / 2}, bytes);
  torch::Tensor block_scales = torch::zeros({scale_rows, kScaleCols}, bytes);
  torch::Tensor rstd = torch::empty(
      {x.size(0)}, x.options().dtype(torch::kFloat32));
  torch::Tensor max_bits = torch::zeros(
      {1}, x.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, x.options().dtype(torch::kFloat32));
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_gate_first_pass<__nv_bfloat16>(
        x, other, gate, weight, scale, shift, row_ids, rstd, max_bits, eps);
  } else {
    launch_gate_first_pass<float>(
        x, other, gate, weight, scale, shift, row_ids, rstd, max_bits, eps);
  }
  // The first launch is the mutation boundary.  Mirror a logical PyTorch
  // in-place residual update so aliases and saved-version checks observe it.
  torch::autograd::increment_version(x);

  constexpr int kThreadsPerRow = kWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * kThreadsPerRow;
  const int blocks = static_cast<int>(
      (total_threads + kQuantThreads - 1) / kQuantThreads);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (scale.scalar_type() == torch::kBFloat16) {
    rms_adaln_nvfp4_second_pass<__nv_bfloat16><<<
        blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr()),
        row_ids.data_ptr<int32_t>(), rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  } else {
    rms_adaln_nvfp4_second_pass<float><<<
        blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        row_ids.data_ptr<int32_t>(), rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, block_scales, global_scale, rstd};
}

torch::Tensor adaln_workspace(const torch::Tensor& x) {
  static std::mutex mutex;
  static std::unordered_map<int, torch::Tensor> cache;
  const int device = x.get_device();
  std::lock_guard<std::mutex> lock(mutex);
  torch::Tensor& slot = cache[device];
  if (!slot.defined() || slot.size(0) < x.size(0) ||
      slot.device() != x.device()) {
    slot = torch::empty({x.size(0), kWidth}, x.options());
  }
  return slot.narrow(0, 0, x.size(0));
}

std::vector<torch::Tensor> gate_rms_adaln_nvfp4_dynamic_keep(
    torch::Tensor x,
    torch::Tensor other,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    double eps) {
  // First pass already evaluates AdaLN for the amax. Keep those BF16
  // values and pack from them so the second pass does not recompute
  // RMS/AdaLN. Residual mutation is unchanged.
  check_inputs(x, weight, scale, shift, row_ids, eps);
  check_gate_inputs(x, other, gate, scale);
  check_mutation_aliases(x, weight, shift, row_ids);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t padded_rows = ((x.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto bytes = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({padded_rows, kWidth / 2}, bytes);
  torch::Tensor block_scales = torch::zeros({scale_rows, kScaleCols}, bytes);
  torch::Tensor rstd = torch::empty(
      {x.size(0)}, x.options().dtype(torch::kFloat32));
  torch::Tensor max_bits = torch::zeros(
      {1}, x.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, x.options().dtype(torch::kFloat32));
  torch::Tensor adaln = adaln_workspace(x);
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_gate_first_pass<__nv_bfloat16>(
        x, other, gate, weight, scale, shift, row_ids, rstd, max_bits, eps,
        reinterpret_cast<__nv_bfloat16*>(adaln.data_ptr()));
  } else {
    launch_gate_first_pass<float>(
        x, other, gate, weight, scale, shift, row_ids, rstd, max_bits, eps,
        reinterpret_cast<__nv_bfloat16*>(adaln.data_ptr()));
  }
  torch::autograd::increment_version(x);

  constexpr int kThreadsPerRow = kWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * kThreadsPerRow;
  const int blocks = static_cast<int>(
      (total_threads + kQuantThreads - 1) / kQuantThreads);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  bf16_nvfp4_from_amax_bits_kernel<<<blocks, kQuantThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(adaln.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
      block_scales.data_ptr<uint8_t>(), x.size(0), padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, block_scales, global_scale, rstd};
}

void launch_second_pass_vec(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& row_ids,
    const torch::Tensor& rstd,
    const torch::Tensor& max_bits,
    torch::Tensor& global_scale,
    torch::Tensor& packed,
    torch::Tensor& block_scales,
    int64_t padded_rows) {
  constexpr int kThreadsPerRow = kWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * kThreadsPerRow;
  const int blocks = static_cast<int>(
      (total_threads + kQuantThreads - 1) / kQuantThreads);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (scale.scalar_type() == torch::kBFloat16) {
    rms_adaln_nvfp4_second_pass_vec<__nv_bfloat16><<<
        blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr()),
        row_ids.data_ptr<int32_t>(), rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  } else {
    rms_adaln_nvfp4_second_pass_vec<float><<<
        blocks, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        row_ids.data_ptr<int32_t>(), rstd.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
        global_scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
        block_scales.data_ptr<uint8_t>(), scale.stride(0), shift.stride(0),
        x.size(0), padded_rows, scale.size(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> rms_adaln_nvfp4_dynamic_vec(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    double eps) {
  check_inputs(x, weight, scale, shift, row_ids, eps);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t padded_rows = ((x.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto bytes = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({padded_rows, kWidth / 2}, bytes);
  torch::Tensor block_scales = torch::zeros({scale_rows, kScaleCols}, bytes);
  torch::Tensor rstd = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  torch::Tensor max_bits = torch::zeros({1}, x.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty({1}, x.options().dtype(torch::kFloat32));
  const torch::Tensor empty;
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_first_pass<__nv_bfloat16, false, true>(
        x, weight, scale, shift, row_ids, rstd, empty, max_bits, eps);
  } else {
    launch_first_pass<float, false, true>(
        x, weight, scale, shift, row_ids, rstd, empty, max_bits, eps);
  }
  launch_second_pass_vec(
      x, weight, scale, shift, row_ids, rstd, max_bits, global_scale,
      packed, block_scales, padded_rows);
  return {packed, block_scales, global_scale, rstd};
}

std::vector<torch::Tensor> gate_rms_adaln_nvfp4_dynamic_vec(
    torch::Tensor x,
    torch::Tensor other,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    torch::Tensor row_ids,
    double eps) {
  check_inputs(x, weight, scale, shift, row_ids, eps);
  check_gate_inputs(x, other, gate, scale);
  check_mutation_aliases(x, weight, shift, row_ids);
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t padded_rows = ((x.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const auto bytes = x.options().dtype(torch::kUInt8);
  torch::Tensor packed = torch::empty({padded_rows, kWidth / 2}, bytes);
  torch::Tensor block_scales = torch::zeros({scale_rows, kScaleCols}, bytes);
  torch::Tensor rstd = torch::empty(
      {x.size(0)}, x.options().dtype(torch::kFloat32));
  torch::Tensor max_bits = torch::zeros(
      {1}, x.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, x.options().dtype(torch::kFloat32));
  if (scale.scalar_type() == torch::kBFloat16) {
    launch_gate_first_pass<__nv_bfloat16>(
        x, other, gate, weight, scale, shift, row_ids, rstd, max_bits, eps);
  } else {
    launch_gate_first_pass<float>(
        x, other, gate, weight, scale, shift, row_ids, rstd, max_bits, eps);
  }
  torch::autograd::increment_version(x);
  launch_second_pass_vec(
      x, weight, scale, shift, row_ids, rstd, max_bits, global_scale,
      packed, block_scales, padded_rows);
  return {packed, block_scales, global_scale, rstd};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("abi_version", []() { return 2; }, "Native adapter ABI version");
  m.def("build_id", []() {
    return "h3-gate-rms-adaln-nvfp4-torch2.13-cf30153-sm120-gb10-safe-v2";
  }, "Immutable native build capability ID");
  m.def("rms_adaln_bf16", &rms_adaln_bf16,
        "Exact H3 RMSNorm + segmented AdaLN materialization (CUDA)");
  m.def("rms_adaln_nvfp4_dynamic", &rms_adaln_nvfp4_dynamic,
        "Exact H3 RMSNorm + segmented AdaLN -> dynamic NVFP4 (CUDA)");
  m.def("rms_adaln_nvfp4_dynamic_cached", &rms_adaln_nvfp4_dynamic_cached,
        "RMS/AdaLN dynamic NVFP4 with one-load first pass");
  m.def("gate_rms_adaln_nvfp4_dynamic", &gate_rms_adaln_nvfp4_dynamic,
        "H3 residual gate + RMSNorm + AdaLN -> dynamic NVFP4 (CUDA)");
  m.def("gate_rms_adaln_nvfp4_dynamic_keep",
        &gate_rms_adaln_nvfp4_dynamic_keep,
        "Gate+RMS+AdaLN dynamic NVFP4 keeping first-pass AdaLN for pack");
  m.def("rms_adaln_nvfp4_dynamic_vec", &rms_adaln_nvfp4_dynamic_vec,
        "RMS/AdaLN dynamic NVFP4 with 8-wide second-pass loads");
  m.def("gate_rms_adaln_nvfp4_dynamic_vec",
        &gate_rms_adaln_nvfp4_dynamic_vec,
        "Gate+RMS+AdaLN dynamic NVFP4 with 8-wide second-pass loads");
}
