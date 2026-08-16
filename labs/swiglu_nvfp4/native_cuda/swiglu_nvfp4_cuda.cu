/*
 * Experimental H3 BF16 SwiGLU -> NVFP4 fusion.
 *
 * The NVFP4 packing and scale layout intentionally follow comfy-kitchen
 * v0.2.27's Apache-2.0 CUDA implementation:
 *   comfy_kitchen/backends/cuda/ops/quantize_nvfp4.cu
 *   comfy_kitchen/backends/cuda/float_utils.cuh
 *
 * This file is a stand-alone research prototype and is not wired into
 * ComfyUI.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kActivationWidth = 14336;
constexpr int kRawWidth = 2 * kActivationWidth;
constexpr int kValuesPerThread = 4;
constexpr int kFp4BlockSize = 16;
constexpr int kThreadsPerScale = kFp4BlockSize / kValuesPerThread;
constexpr int kThreadsPerBlock = 128;
constexpr int kScaleCols = kActivationWidth / kFp4BlockSize;  // 896
constexpr int kDirectValuesPerThread = 8;
constexpr int kDirectThreadsPerScale =
    kFp4BlockSize / kDirectValuesPerThread;
constexpr int kFc1K = 5376;
constexpr int kFc1TileM = 32;
constexpr int kFc1TileN = 128;
constexpr int kFc1TileK = 64;
constexpr int kSwizzleTileR = 128;
constexpr int kSwizzleTileC = 64;
constexpr int kSwizzleScaleGroup = 4;
constexpr int kWmma = 16;
constexpr int kWmmaTile = 32;
constexpr int kWmmaThreads = 128;

static_assert(kFc1K % kFc1TileK == 0, "FC1 K must tile");
static_assert(kActivationWidth % kFc1TileN == 0, "FC1 N must tile");
static_assert(kActivationWidth % kSwizzleTileC == 0,
              "Swizzle tile must divide the product width");
static_assert(kScaleCols % kSwizzleScaleGroup == 0,
              "Scale columns must be groups of 4");

// These are the activation widths of H3's four main NVFP4 linears. FC2 is
// handled by the SwiGLU-specialized path below; the direct path targets QKV,
// attention output, and FC1 without becoming a process-wide quantizer.
__host__ __device__ constexpr bool is_h3_activation_width(int64_t width) {
  return width == 5376 || width == 7168 || width == 14336;
}

static_assert(kActivationWidth % 512 == 0,
              "Each CUDA block must remain within one H3 row");
static_assert(kScaleCols % 4 == 0,
              "H3's scale columns require no right-side scale padding");
static_assert(kFp4BlockSize % kDirectValuesPerThread == 0,
              "A direct-kernel lane group must own one FP4 scale block");

__device__ __forceinline__ size_t scale_factor_swizzled_offset(
    size_t row_idx, size_t col_idx, uint32_t col_length) {
  // cuBLAS block-scale layout: (row_block, col_block_group, 32, 4, 4).
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
  // ComfyUI's eager path is F.silu(gate).mul_(up). Because both tensors are
  // BF16, that path rounds once after SiLU and again after multiplication.
  // Preserve those two storage-boundary roundings before quantization.
  const float gate = __bfloat162float(gate_bf16);
  const float up = __bfloat162float(up_bf16);
  const float silu = gate / (1.0f + expf(-gate));
  const float silu_bf16 = __bfloat162float(__float2bfloat16_rn(silu));
  return __bfloat162float(__float2bfloat16_rn(silu_bf16 * up));
}

__device__ __forceinline__ float swiglu_fp32(
    __nv_bfloat16 gate_bf16, __nv_bfloat16 up_bf16) {
  const float gate = __bfloat162float(gate_bf16);
  const float up = __bfloat162float(up_bf16);
  return (gate / (1.0f + expf(-gate))) * up;
}

template <bool EagerBf16Rounding>
__global__ void swiglu_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const float* __restrict__ global_scale,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row = kActivationWidth / kValuesPerThread;
  const int64_t total_threads = padded_rows * threads_per_row;
  if (thread_linear >= total_threads) {
    return;
  }

  const int64_t row = thread_linear / threads_per_row;
  const int col = static_cast<int>(
      (thread_linear - row * threads_per_row) * kValuesPerThread);

  float values[kValuesPerThread];
  if (row < orig_rows) {
    const __nv_bfloat16* gate = raw + row * kRawWidth + col;
    const __nv_bfloat16* up = gate + kActivationWidth;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      if constexpr (EagerBf16Rounding) {
        values[i] = swiglu_like_eager_bf16(gate[i], up[i]);
      } else {
        values[i] = swiglu_fp32(gate[i], up[i]);
      }
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

  // Four adjacent lanes jointly own one 16-value NVFP4 scale block.
  constexpr unsigned int kFullMask = 0xffffffffu;
#pragma unroll
  for (int offset = kThreadsPerScale / 2; offset >= 1; offset /= 2) {
    absmax = fmaxf(
        absmax,
        __shfl_down_sync(kFullMask, absmax, offset, kThreadsPerScale));
  }
  const int lane = threadIdx.x & 31;
  const int leader = (lane / kThreadsPerScale) * kThreadsPerScale;
  absmax = __shfl_sync(kFullMask, absmax, leader, 32);

  const float global_decode_scale = global_scale[0];
  // comfy-kitchen builds this quantizer with --use_fast_math. Keep the
  // quantizer fast divides explicit while leaving SiLU expf precise.
  float decode_scale = __fdividef(__fdividef(absmax, 6.0f), global_decode_scale);
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

  // This reproduces comfy-kitchen's quantizer, including use of the rounded
  // E4M3 scale for the reciprocal and its high-nibble-first convention.
  const float encode_scale = fminf(
      __fdividef(1.0f, decode_scale_fp8 * global_decode_scale), FLT_MAX);
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    values[i] *= encode_scale;
  }

  union PackedFp4 {
    uint16_t u16;
    __nv_fp4x2_storage_t fp4x2[2];
  } packed;
  // CUDA's converter places args.x in the low nibble and args.y in the high
  // nibble. Reverse each pair so even logical elements occupy high nibbles.
  packed.fp4x2[0] = __nv_cvt_float2_to_fp4x2(
      float2{values[1], values[0]}, __NV_E2M1, cudaRoundNearest);
  packed.fp4x2[1] = __nv_cvt_float2_to_fp4x2(
      float2{values[3], values[2]}, __NV_E2M1, cudaRoundNearest);
  *reinterpret_cast<uint16_t*>(output + thread_linear * 2) = packed.u16;
}

__global__ void swiglu_amax_bf16_bits_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t num_values) {
  uint32_t local_max = 0;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < num_values;
       idx += stride) {
    const int64_t row = idx / kActivationWidth;
    const int64_t col = idx - row * kActivationWidth;
    const __nv_bfloat16 gate = raw[row * kRawWidth + col];
    const __nv_bfloat16 up =
        raw[row * kRawWidth + kActivationWidth + col];
    // The helper already performs both eager BF16 storage roundings.
    const __nv_bfloat16 value =
        __float2bfloat16_rn(swiglu_like_eager_bf16(gate, up));
    const uint32_t abs_bits =
        static_cast<uint32_t>(__bfloat16_as_ushort(value) & 0x7fffu);
    local_max = local_max < abs_bits ? abs_bits : local_max;
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

  __shared__ uint32_t warp_maxima[8];
  if (lane == 0) {
    warp_maxima[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = blockDim.x >> 5;
    local_max = lane < warp_count ? warp_maxima[lane] : 0;
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

__global__ void swiglu_store_amax_bf16_bits_kernel(
    const __nv_bfloat16* __restrict__ raw,
    __nv_bfloat16* __restrict__ activated,
    uint32_t* __restrict__ global_max_bits,
    int64_t num_values) {
  // One SwiGLU evaluation: keep the BF16 activation and the exact bit-max
  // used by TensorCoreNVFP4Layout. The pack pass then reads `activated`
  // instead of recomputing SiLU on the 28,672-wide raw pair.
  uint32_t local_max = 0;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < num_values;
       idx += stride) {
    const int64_t row = idx / kActivationWidth;
    const int64_t col = idx - row * kActivationWidth;
    const __nv_bfloat16 gate = raw[row * kRawWidth + col];
    const __nv_bfloat16 up =
        raw[row * kRawWidth + kActivationWidth + col];
    const __nv_bfloat16 value =
        __float2bfloat16_rn(swiglu_like_eager_bf16(gate, up));
    activated[idx] = value;
    const uint32_t abs_bits =
        static_cast<uint32_t>(__bfloat16_as_ushort(value) & 0x7fffu);
    local_max = local_max < abs_bits ? abs_bits : local_max;
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

  __shared__ uint32_t warp_maxima[8];
  if (lane == 0) {
    warp_maxima[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = blockDim.x >> 5;
    local_max = lane < warp_count ? warp_maxima[lane] : 0;
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

__global__ void finalize_dynamic_scale_kernel(
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ global_scale) {
  const __nv_bfloat16 absmax = __ushort_as_bfloat16(
      static_cast<unsigned short>(global_max_bits[0]));
  // Match TensorCoreNVFP4Layout exactly:
  // (BF16 amax / Python scalar 2688.0) yields BF16, then widens to FP32.
  const __nv_bfloat16 scale_bf16 = __float2bfloat16_rn(
      __bfloat162float(absmax) / 2688.0f);
  global_scale[0] = __bfloat162float(scale_bf16);
}

__device__ __forceinline__ float dynamic_scale_from_bf16_bits(
    const uint32_t* __restrict__ global_max_bits) {
  const __nv_bfloat16 absmax = __ushort_as_bfloat16(
      static_cast<unsigned short>(global_max_bits[0]));
  // TensorCoreNVFP4Layout evaluates this division in BF16 before widening the
  // scalar to FP32. Keeping that storage boundary makes the generic path byte
  // compatible with the stock ComfyUI quantizer.
  const __nv_bfloat16 scale_bf16 = __float2bfloat16_rn(
      __bfloat162float(absmax) / 2688.0f);
  return __bfloat162float(scale_bf16);
}

// min_x SiLU(x) = x*sigmoid(x) at x≈-1.27846454 is -0.278464542761...
// Any BF16-rounded |SiLU(g)| for g<0 is therefore strictly below this.
constexpr float kSiluNegAbsMax = 0.27846456f;

__device__ __forceinline__ uint32_t swiglu_abs_bits(
    __nv_bfloat16 gate_bf16, __nv_bfloat16 up_bf16) {
  const __nv_bfloat16 value =
      __float2bfloat16_rn(swiglu_like_eager_bf16(gate_bf16, up_bf16));
  return static_cast<uint32_t>(__bfloat16_as_ushort(value) & 0x7fffu);
}

__device__ __forceinline__ uint32_t swiglu_abs_bound_bits(
    __nv_bfloat16 gate_bf16, __nv_bfloat16 up_bf16) {
  // Exact identity: |round_bf16(round_bf16(silu(g))*u)| <= ru_bf16(bound)
  // because |silu(g)| <= g for g>=0 (g already BF16) and |silu(g)| <
  // kSiluNegAbsMax for g<0, and round-nearest cannot exceed round-up.
  const float gate = __bfloat162float(gate_bf16);
  const float up = __bfloat162float(up_bf16);
  const float mag = gate >= 0.0f ? gate : kSiluNegAbsMax;
  const float ub = mag * fabsf(up);
  return static_cast<uint32_t>(
      __bfloat16_as_ushort(__float2bfloat16_ru(ub)) & 0x7fffu);
}

__device__ __forceinline__ void reduce_atomic_max_bits(
    uint32_t local_max, uint32_t* __restrict__ global_max_bits) {
  constexpr unsigned int kFullMask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1) {
    const uint32_t other = __shfl_down_sync(kFullMask, local_max, offset);
    local_max = local_max < other ? other : local_max;
  }

  __shared__ uint32_t warp_maxima[8];
  if (lane == 0) {
    warp_maxima[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = blockDim.x >> 5;
    local_max = lane < warp_count ? warp_maxima[lane] : 0;
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
      const uint32_t other = __shfl_down_sync(kFullMask, local_max, offset);
      local_max = local_max < other ? other : local_max;
    }
    if (lane == 0) {
      atomicMax(
          reinterpret_cast<unsigned int*>(global_max_bits),
          static_cast<unsigned int>(local_max));
    }
  }
}

__global__ void swiglu_amax_bound_vec_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t orig_rows) {
  // Same BF16 abs-bit max as the scalar two-pass, but 8-wide packed loads
  // matching the pack tile and skip exact SiLU when the round-up bound
  // cannot beat the thread-local max. Max is associative; skipped lanes
  // cannot change the bits.
  uint32_t local_max = 0;
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
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
      if (swiglu_abs_bound_bits(gate_v[i], up_v[i]) <= local_max) {
        continue;
      }
      const uint32_t abs_bits = swiglu_abs_bits(gate_v[i], up_v[i]);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
}

__global__ void swiglu_amax_vec_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t orig_rows) {
  // Same exact SiLU amax as the scalar two-pass. 8-wide loads match the
  // pack tile. Every lane evaluates SiLU; max of abs-bits is associative.
  uint32_t local_max = 0;
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
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
      const uint32_t abs_bits = swiglu_abs_bits(gate_v[i], up_v[i]);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
}

__device__ __forceinline__ __nv_bfloat16 silu_like_eager_bf16(
    __nv_bfloat16 gate_bf16) {
  const float gate = __bfloat162float(gate_bf16);
  return __float2bfloat16_rn(gate / (1.0f + expf(-gate)));
}

__global__ void swiglu_amax_inplace_silu_kernel(
    __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t orig_rows) {
  // One SiLU: write silu(gate) over the dead gate half. Amax bits are
  // still abs(round(silu*up)). Pack then only multiplies.
  uint32_t local_max = 0;
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
    __nv_bfloat16* gate = raw + row * kRawWidth + col;
    const __nv_bfloat16* up = gate + kActivationWidth;
    uint4 gate_vec = *reinterpret_cast<uint4*>(gate);
    const uint4 up_vec = *reinterpret_cast<const uint4*>(up);
    __nv_bfloat16* gate_v = reinterpret_cast<__nv_bfloat16*>(&gate_vec);
    const __nv_bfloat16* up_v =
        reinterpret_cast<const __nv_bfloat16*>(&up_vec);
#pragma unroll
    for (int i = 0; i < kDirectValuesPerThread; ++i) {
      const __nv_bfloat16 silu = silu_like_eager_bf16(gate_v[i]);
      gate_v[i] = silu;
      const __nv_bfloat16 prod = __float2bfloat16_rn(
          __bfloat162float(silu) * __bfloat162float(up_v[i]));
      const uint32_t abs_bits = static_cast<uint32_t>(
          __bfloat16_as_ushort(prod) & 0x7fffu);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
    *reinterpret_cast<uint4*>(gate) = gate_vec;
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
}

__global__ void swiglu_pack_from_silu_gate_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  // Gate half already holds eager silu(gate). Product rounding matches
  // F.silu(gate).mul_(up).
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t total_threads = padded_rows * threads_per_row;
  if (thread_linear >= total_threads) {
    return;
  }

  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

  const int64_t row = thread_linear / threads_per_row;
  const int col = static_cast<int>(
      (thread_linear - row * threads_per_row) * kDirectValuesPerThread);

  float values[kDirectValuesPerThread];
  if (row < orig_rows) {
    const __nv_bfloat16* silu_gate = raw + row * kRawWidth + col;
    const __nv_bfloat16* up = silu_gate + kActivationWidth;
    const uint4 gate_vec = *reinterpret_cast<const uint4*>(silu_gate);
    const uint4 up_vec = *reinterpret_cast<const uint4*>(up);
    const __nv_bfloat16* gate_v =
        reinterpret_cast<const __nv_bfloat16*>(&gate_vec);
    const __nv_bfloat16* up_v =
        reinterpret_cast<const __nv_bfloat16*>(&up_vec);
#pragma unroll
    for (int i = 0; i < kDirectValuesPerThread; ++i) {
      values[i] = __bfloat162float(__float2bfloat16_rn(
          __bfloat162float(gate_v[i]) * __bfloat162float(up_v[i])));
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

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
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

__global__ void swiglu_amax_inplace_prod_kernel(
    __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t orig_rows) {
  // One SiLU: write eager product over the dead up half. Amax bits are
  // abs(round(silu*up)). Pack then reads only that 14,336-wide half.
  uint32_t local_max = 0;
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
    const __nv_bfloat16* gate = raw + row * kRawWidth + col;
    __nv_bfloat16* up = raw + row * kRawWidth + kActivationWidth + col;
    const uint4 gate_vec = *reinterpret_cast<const uint4*>(gate);
    uint4 up_vec = *reinterpret_cast<uint4*>(up);
    const __nv_bfloat16* gate_v =
        reinterpret_cast<const __nv_bfloat16*>(&gate_vec);
    __nv_bfloat16* up_v = reinterpret_cast<__nv_bfloat16*>(&up_vec);
#pragma unroll
    for (int i = 0; i < kDirectValuesPerThread; ++i) {
      const __nv_bfloat16 prod = __float2bfloat16_rn(
          swiglu_like_eager_bf16(gate_v[i], up_v[i]));
      up_v[i] = prod;
      const uint32_t abs_bits = static_cast<uint32_t>(
          __bfloat16_as_ushort(prod) & 0x7fffu);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
    *reinterpret_cast<uint4*>(up) = up_vec;
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
}

__global__ void swiglu_pack_from_up_prod_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  // Up half already holds eager round(silu*up). Pack is half-width BF16
  // NVFP4: no SiLU, no multiply, one 14,336-col load per row.
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t total_threads = padded_rows * threads_per_row;
  if (thread_linear >= total_threads) {
    return;
  }

  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

  const int64_t row = thread_linear / threads_per_row;
  const int col = static_cast<int>(
      (thread_linear - row * threads_per_row) * kDirectValuesPerThread);

  float values[kDirectValuesPerThread];
  if (row < orig_rows) {
    const __nv_bfloat16* prod =
        raw + row * kRawWidth + kActivationWidth + col;
    const uint4 prod_vec = *reinterpret_cast<const uint4*>(prod);
    const __nv_bfloat16* prod_v =
        reinterpret_cast<const __nv_bfloat16*>(&prod_vec);
#pragma unroll
    for (int i = 0; i < kDirectValuesPerThread; ++i) {
      values[i] = __bfloat162float(prod_v[i]);
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

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
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

__global__ void fc1_epilogue_assoc_kernel(
    const __nv_bfloat16* __restrict__ raw,
    unsigned long long* __restrict__ mismatches,
    int64_t orig_rows) {
  // Virtual FP32 accumulator = BF16 store + 1 FP32 ULP. That still
  // round-trips to the same BF16 (FC1 store bits) but SiLU(acc) is a
  // different function than SiLU(store). The in-GEMM epilogue must use
  // the eager association, not fused FP32 SiLU on the accumulator.
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
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
      const float g = __bfloat162float(gate_v[i]);
      const float u = __bfloat162float(up_v[i]);
      const float acc_g = nextafterf(g, copysignf(FLT_MAX, g));
      const float acc_u = nextafterf(u, copysignf(FLT_MAX, u));
      const __nv_bfloat16 store_g = __float2bfloat16_rn(acc_g);
      const __nv_bfloat16 store_u = __float2bfloat16_rn(acc_u);
      const __nv_bfloat16 eager = __float2bfloat16_rn(
          swiglu_like_eager_bf16(store_g, store_u));
      const float silu_acc = acc_g / (1.0f + expf(-acc_g));
      const __nv_bfloat16 fused = __float2bfloat16_rn(silu_acc * acc_u);
      local += static_cast<unsigned long long>(
          __bfloat16_as_ushort(eager) != __bfloat16_as_ushort(fused));
    }
  }
  if (local != 0) {
    atomicAdd(mismatches, local);
  }
}

template <bool StoreProduct>
__global__ void fc1_paired_mma_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int64_t rows) {
  // Same MMA for both arms: each CTA owns a (row tile, product-col tile)
  // and accumulates gate and up together. StoreProduct writes the eager
  // SwiGLU product (the in-GEMM epilogue). Otherwise it writes [gate|up].
  const int n = blockIdx.x * kFc1TileN + threadIdx.x;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * kFc1TileM;
  if (n >= kActivationWidth || row0 >= rows) {
    return;
  }

  float acc_g[kFc1TileM];
  float acc_u[kFc1TileM];
#pragma unroll
  for (int r = 0; r < kFc1TileM; ++r) {
    acc_g[r] = 0.0f;
    acc_u[r] = 0.0f;
  }

  __shared__ __nv_bfloat16 smem_x[kFc1TileM][kFc1TileK];
  __shared__ __nv_bfloat16 smem_wg[kFc1TileN][kFc1TileK];
  __shared__ __nv_bfloat16 smem_wu[kFc1TileN][kFc1TileK];

  const __nv_bfloat16* wgrow =
      weight + static_cast<int64_t>(n) * kFc1K;
  const __nv_bfloat16* wurow =
      weight + (static_cast<int64_t>(n) + kActivationWidth) * kFc1K;

  for (int k0 = 0; k0 < kFc1K; k0 += kFc1TileK) {
#pragma unroll
    for (int i = 0; i < kFc1TileK; i += 8) {
      *reinterpret_cast<uint4*>(&smem_wg[threadIdx.x][i]) =
          *reinterpret_cast<const uint4*>(wgrow + k0 + i);
      *reinterpret_cast<uint4*>(&smem_wu[threadIdx.x][i]) =
          *reinterpret_cast<const uint4*>(wurow + k0 + i);
    }

    const int x_base = threadIdx.x * 16;
    const int xr = x_base / kFc1TileK;
    const int xc = x_base - xr * kFc1TileK;
    const int64_t mr = row0 + xr;
    if (mr < rows) {
      const __nv_bfloat16* xsrc = input + mr * kFc1K + k0 + xc;
      *reinterpret_cast<uint4*>(&smem_x[xr][xc]) =
          *reinterpret_cast<const uint4*>(xsrc);
      *reinterpret_cast<uint4*>(&smem_x[xr][xc + 8]) =
          *reinterpret_cast<const uint4*>(xsrc + 8);
    } else {
      *reinterpret_cast<uint4*>(&smem_x[xr][xc]) = uint4{0, 0, 0, 0};
      *reinterpret_cast<uint4*>(&smem_x[xr][xc + 8]) = uint4{0, 0, 0, 0};
    }
    __syncthreads();

#pragma unroll 4
    for (int r = 0; r < kFc1TileM; ++r) {
      if (row0 + r >= rows) {
        break;
      }
      float ag = acc_g[r];
      float au = acc_u[r];
#pragma unroll
      for (int kk = 0; kk < kFc1TileK; ++kk) {
        const float x = __bfloat162float(smem_x[r][kk]);
        ag += x * __bfloat162float(smem_wg[threadIdx.x][kk]);
        au += x * __bfloat162float(smem_wu[threadIdx.x][kk]);
      }
      acc_g[r] = ag;
      acc_u[r] = au;
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < kFc1TileM; ++r) {
    const int64_t m = row0 + r;
    if (m >= rows) {
      break;
    }
    const __nv_bfloat16 g = __float2bfloat16_rn(acc_g[r]);
    const __nv_bfloat16 u = __float2bfloat16_rn(acc_u[r]);
    if constexpr (StoreProduct) {
      output[m * kActivationWidth + n] = __float2bfloat16_rn(
          swiglu_like_eager_bf16(g, u));
    } else {
      output[m * kRawWidth + n] = g;
      output[m * kRawWidth + kActivationWidth + n] = u;
    }
  }
}

template <bool StoreProduct>
__global__ void fc1_paired_wmma_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int64_t rows) {
  // Tensor-core paired-N: each CTA owns a 32x32 product tile, four warps
  // of 16x16 WMMA. Same A fragment feeds gate and up. Epilogue is eager
  // BF16 SwiGLU. Not the kitchen NVFP4 MMA.
  using namespace nvcuda;
  const int warp = threadIdx.x >> 5;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;
  const int tile_n = blockIdx.x * kWmmaTile;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kWmmaTile;

  wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __nv_bfloat16,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __nv_bfloat16,
                 wmma::col_major>
      bg_frag, bu_frag;
  wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> cg_frag,
      cu_frag;
  wmma::fill_fragment(cg_frag, 0.0f);
  wmma::fill_fragment(cu_frag, 0.0f);

  __shared__ __nv_bfloat16 smem_a[kWmmaTile][kWmma];
  __shared__ __nv_bfloat16 smem_bg[kWmmaTile][kWmma];
  __shared__ __nv_bfloat16 smem_bu[kWmmaTile][kWmma];

  for (int k0 = 0; k0 < kFc1K; k0 += kWmma) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int a_idx = threadIdx.x + s * kWmmaThreads;
      const int ar = a_idx / kWmma;
      const int ac = a_idx - ar * kWmma;
      const int64_t am = tile_m + ar;
      smem_a[ar][ac] = (am < rows)
          ? input[am * kFc1K + k0 + ac]
          : __nv_bfloat16{};
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int b_idx = threadIdx.x + s * kWmmaThreads;
      const int n_local = b_idx & (kWmmaTile - 1);
      const int k_local = b_idx / kWmmaTile;
      const int n = tile_n + n_local;
      const __nv_bfloat16 zg = (n < kActivationWidth)
          ? weight[static_cast<int64_t>(n) * kFc1K + k0 + k_local]
          : __nv_bfloat16{};
      const __nv_bfloat16 zu = (n < kActivationWidth)
          ? weight[(static_cast<int64_t>(n) + kActivationWidth) * kFc1K
                   + k0 + k_local]
          : __nv_bfloat16{};
      smem_bg[n_local][k_local] = zg;
      smem_bu[n_local][k_local] = zu;
    }
    __syncthreads();

    wmma::load_matrix_sync(
        a_frag, &smem_a[warp_m * kWmma][0], kWmma);
    wmma::load_matrix_sync(
        bg_frag, &smem_bg[warp_n * kWmma][0], kWmma);
    wmma::load_matrix_sync(
        bu_frag, &smem_bu[warp_n * kWmma][0], kWmma);
    wmma::mma_sync(cg_frag, a_frag, bg_frag, cg_frag);
    wmma::mma_sync(cu_frag, a_frag, bu_frag, cu_frag);
    __syncthreads();
  }

  __shared__ float smem_cg[kWmmaTile][kWmmaTile];
  __shared__ float smem_cu[kWmmaTile][kWmmaTile];
  wmma::store_matrix_sync(
      &smem_cg[warp_m * kWmma][warp_n * kWmma], cg_frag, kWmmaTile,
      wmma::mem_row_major);
  wmma::store_matrix_sync(
      &smem_cu[warp_m * kWmma][warp_n * kWmma], cu_frag, kWmmaTile,
      wmma::mem_row_major);
  __syncthreads();

  for (int i = threadIdx.x; i < kWmmaTile * kWmmaTile; i += kWmmaThreads) {
    const int r = i / kWmmaTile;
    const int c = i - r * kWmmaTile;
    const int64_t m = tile_m + r;
    const int n = tile_n + c;
    if (m >= rows || n >= kActivationWidth) {
      continue;
    }
    const __nv_bfloat16 g = __float2bfloat16_rn(smem_cg[r][c]);
    const __nv_bfloat16 u = __float2bfloat16_rn(smem_cu[r][c]);
    if constexpr (StoreProduct) {
      output[m * kActivationWidth + n] = __float2bfloat16_rn(
          swiglu_like_eager_bf16(g, u));
    } else {
      output[m * kRawWidth + n] = g;
      output[m * kRawWidth + kActivationWidth + n] = u;
    }
  }
}

constexpr int kNvfp4TileM = 16;
constexpr int kNvfp4TileN = 8;
constexpr int kNvfp4TileK = 64;
constexpr int kNvfp4PackedK = kFc1K / 2;
constexpr uint32_t kE4M3One = 0x38383838u;

__device__ __forceinline__ void nvfp4_mma_m16n8k64(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa), "h"(static_cast<unsigned short>(0)),
        "h"(static_cast<unsigned short>(0)),
        "r"(sfb), "h"(static_cast<unsigned short>(0)),
        "h"(static_cast<unsigned short>(0)));
}

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ w_packed,
    __nv_bfloat16* __restrict__ output,
    int64_t rows) {
  // SM120 NVFP4 tensor-core paired-N: one warp owns 16x8 product columns.
  // Same A fragment feeds gate and up. Epilogue is eager BF16 SwiGLU.
  // Not cuBLASLt; unit-E4M3 scales keep the atom legal.
  const int lane = threadIdx.x;
  const int tile_n = blockIdx.x * kNvfp4TileN;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kNvfp4TileM;
  if (tile_n >= kActivationWidth) {
    return;
  }

  float gg0 = 0.0f, gg1 = 0.0f, gg2 = 0.0f, gg3 = 0.0f;
  float uu0 = 0.0f, uu1 = 0.0f, uu2 = 0.0f, uu3 = 0.0f;
  __shared__ uint8_t smem_a[kNvfp4TileM][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bg[kNvfp4TileN][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bu[kNvfp4TileN][kNvfp4TileK / 2];

  for (int k0 = 0; k0 < kFc1K; k0 += kNvfp4TileK) {
    const int packed_k = k0 / 2;
    for (int i = lane; i < kNvfp4TileM * (kNvfp4TileK / 2); i += 32) {
      const int r = i / (kNvfp4TileK / 2);
      const int c = i - r * (kNvfp4TileK / 2);
      const int64_t m = tile_m + r;
      smem_a[r][c] = (m < rows)
          ? a_packed[m * kNvfp4PackedK + packed_k + c]
          : static_cast<uint8_t>(0);
    }
    for (int i = lane; i < kNvfp4TileN * (kNvfp4TileK / 2); i += 32) {
      const int r = i / (kNvfp4TileK / 2);
      const int c = i - r * (kNvfp4TileK / 2);
      const int n = tile_n + r;
      smem_bg[r][c] = w_packed[
          static_cast<int64_t>(n) * kNvfp4PackedK + packed_k + c];
      smem_bu[r][c] = w_packed[
          (static_cast<int64_t>(n) + kActivationWidth) * kNvfp4PackedK
          + packed_k + c];
    }
    __syncwarp();

    const int ar = lane & 15;
    const int ac = (lane >> 4) * 16;
    const uint4 av = *reinterpret_cast<const uint4*>(&smem_a[ar][ac]);
    const int br = lane & 7;
    const int bc = (lane >> 3) * 8;
    const uint2 bg = *reinterpret_cast<const uint2*>(&smem_bg[br][bc]);
    const uint2 bu = *reinterpret_cast<const uint2*>(&smem_bu[br][bc]);
    nvfp4_mma_m16n8k64(
        gg0, gg1, gg2, gg3, av.x, av.y, av.z, av.w, bg.x, bg.y,
        gg0, gg1, gg2, gg3, kE4M3One, kE4M3One);
    nvfp4_mma_m16n8k64(
        uu0, uu1, uu2, uu3, av.x, av.y, av.z, av.w, bu.x, bu.y,
        uu0, uu1, uu2, uu3, kE4M3One, kE4M3One);
    __syncwarp();
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const float g_vals[4] = {gg0, gg1, gg2, gg3};
  const float u_vals[4] = {uu0, uu1, uu2, uu3};
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t m = tile_m + r0 + r_off[i];
    const int n = tile_n + c0 + c_off[i];
    if (m >= rows || n >= kActivationWidth) {
      continue;
    }
    const __nv_bfloat16 g = __float2bfloat16_rn(g_vals[i]);
    const __nv_bfloat16 u = __float2bfloat16_rn(u_vals[i]);
    if constexpr (StoreProduct) {
      output[m * kActivationWidth + n] = __float2bfloat16_rn(
          swiglu_like_eager_bf16(g, u));
    } else {
      output[m * kRawWidth + n] = g;
      output[m * kRawWidth + kActivationWidth + n] = u;
    }
  }
}

constexpr int kNvfp4BigM = 64;
constexpr int kNvfp4BigN = 64;
constexpr int kNvfp4NSub = kNvfp4BigN / kNvfp4TileN;
constexpr int kNvfp4TiledThreads = 128;

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_tiled_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ w_packed,
    __nv_bfloat16* __restrict__ output,
    int64_t rows) {
  // Persistent N-owner: one CTA keeps a 64-wide product panel, streams
  // every M tile, and reuses the A K-slab across 8 n-subtiles and both
  // arms. Same m16n8k64 atom and eager epilogue as the 1-warp kernel.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kNvfp4BigN;
  __shared__ uint8_t smem_a[kNvfp4BigM][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bg[kNvfp4BigN][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bu[kNvfp4BigN][kNvfp4TileK / 2];

  for (int64_t tile_m = 0; tile_m < rows; tile_m += kNvfp4BigM) {
    float acc_g[kNvfp4NSub][4];
    float acc_u[kNvfp4NSub][4];
#pragma unroll
    for (int ni = 0; ni < kNvfp4NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ni][i] = 0.0f;
        acc_u[ni][i] = 0.0f;
      }
    }

    for (int k0 = 0; k0 < kFc1K; k0 += kNvfp4TileK) {
      const int packed_k = k0 / 2;
      const int a_r = threadIdx.x >> 1;
      const int a_c = (threadIdx.x & 1) * 16;
      const int64_t am = tile_m + a_r;
      const uint4 a_src = (am < rows)
          ? *reinterpret_cast<const uint4*>(
                a_packed + am * kNvfp4PackedK + packed_k + a_c)
          : uint4{0, 0, 0, 0};
      *reinterpret_cast<uint4*>(&smem_a[a_r][a_c]) = a_src;

      const int b_n = threadIdx.x >> 1;
      const int b_c = (threadIdx.x & 1) * 16;
      const int n = tile_n + b_n;
      *reinterpret_cast<uint4*>(&smem_bg[b_n][b_c]) =
          *reinterpret_cast<const uint4*>(
              w_packed + static_cast<int64_t>(n) * kNvfp4PackedK
              + packed_k + b_c);
      *reinterpret_cast<uint4*>(&smem_bu[b_n][b_c]) =
          *reinterpret_cast<const uint4*>(
              w_packed
              + (static_cast<int64_t>(n) + kActivationWidth) * kNvfp4PackedK
              + packed_k + b_c);
      __syncthreads();

      const int ar = (warp * kNvfp4TileM) + (lane & 15);
      const int ac = (lane >> 4) * 16;
      const uint4 av = *reinterpret_cast<const uint4*>(&smem_a[ar][ac]);
#pragma unroll
      for (int ni = 0; ni < kNvfp4NSub; ++ni) {
        const int br = ni * kNvfp4TileN + (lane & 7);
        const int bc = (lane >> 3) * 8;
        const uint2 bg = *reinterpret_cast<const uint2*>(&smem_bg[br][bc]);
        const uint2 bu = *reinterpret_cast<const uint2*>(&smem_bu[br][bc]);
        nvfp4_mma_m16n8k64(
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            av.x, av.y, av.z, av.w, bg.x, bg.y,
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            kE4M3One, kE4M3One);
        nvfp4_mma_m16n8k64(
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            av.x, av.y, av.z, av.w, bu.x, bu.y,
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            kE4M3One, kE4M3One);
      }
      __syncthreads();
    }

    const int r0 = (lane / 4);
    const int c0 = (lane % 4) * 2;
    const int r_off[4] = {0, 0, 8, 8};
    const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
    for (int ni = 0; ni < kNvfp4NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        const __nv_bfloat16 g = __float2bfloat16_rn(acc_g[ni][i]);
        const __nv_bfloat16 u = __float2bfloat16_rn(acc_u[ni][i]);
        if constexpr (StoreProduct) {
          output[m * kActivationWidth + n] = __float2bfloat16_rn(
              swiglu_like_eager_bf16(g, u));
        } else {
          output[m * kRawWidth + n] = g;
          output[m * kRawWidth + kActivationWidth + n] = u;
        }
      }
    }
  }
}

// PTX m16n8k64 e2m1 fragment (ISA 9.3 §9.7.15.5.11) plus scale_vec::4X
// selectors (ISA 9.3 §9.7.15.3). Kitchen cublasLt uses the same atom with
// real UE4M3 block scales and an FP32 alpha applied before the BF16 store.
// The unit-scale lab kernels load 32 consecutive K of one row per lane and
// therefore compute a different bilinear form.
__device__ __forceinline__ uint32_t pack_four_scales(
    const uint8_t* __restrict__ scales,
    int64_t row,
    int kcol0,
    int scale_cols) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed |= static_cast<uint32_t>(
        scales[scale_factor_swizzled_offset(
            static_cast<size_t>(row),
            static_cast<size_t>(kcol0 + i),
            static_cast<uint32_t>(scale_cols))])
        << (8 * i);
  }
  return packed;
}

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ w_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // One warp owns 16x8 product columns. Same A fragment feeds gate and up.
  // Epilogue is eager BF16 SwiGLU after alpha * acc.
  const int lane = threadIdx.x;
  const int tile_n = blockIdx.x * kNvfp4TileN;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kNvfp4TileM;
  if (tile_n >= kActivationWidth) {
    return;
  }

  float gg0 = 0.0f, gg1 = 0.0f, gg2 = 0.0f, gg3 = 0.0f;
  float uu0 = 0.0f, uu1 = 0.0f, uu2 = 0.0f, uu3 = 0.0f;
  __shared__ uint8_t smem_a[kNvfp4TileM][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bg[kNvfp4TileN][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bu[kNvfp4TileN][kNvfp4TileK / 2];
  __shared__ uint32_t smem_sa[kNvfp4TileM];
  __shared__ uint32_t smem_sbg[kNvfp4TileN];
  __shared__ uint32_t smem_sbu[kNvfp4TileN];

  for (int k0 = 0; k0 < kFc1K; k0 += kNvfp4TileK) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    for (int i = lane; i < kNvfp4TileM * (kNvfp4TileK / 2); i += 32) {
      const int r = i / (kNvfp4TileK / 2);
      const int c = i - r * (kNvfp4TileK / 2);
      const int64_t m = tile_m + r;
      smem_a[r][c] = (m < rows)
          ? a_packed[m * kNvfp4PackedK + packed_k + c]
          : static_cast<uint8_t>(0);
    }
    for (int i = lane; i < kNvfp4TileN * (kNvfp4TileK / 2); i += 32) {
      const int r = i / (kNvfp4TileK / 2);
      const int c = i - r * (kNvfp4TileK / 2);
      const int n = tile_n + r;
      smem_bg[r][c] = w_packed[
          static_cast<int64_t>(n) * kNvfp4PackedK + packed_k + c];
      smem_bu[r][c] = w_packed[
          (static_cast<int64_t>(n) + kActivationWidth) * kNvfp4PackedK
          + packed_k + c];
    }
    if (lane < kNvfp4TileM) {
      const int64_t m = tile_m + lane;
      smem_sa[lane] = (m < rows)
          ? pack_four_scales(a_scales, m, kcol0, a_scale_cols)
          : 0u;
    }
    if (lane < kNvfp4TileN) {
      const int n = tile_n + lane;
      smem_sbg[lane] = pack_four_scales(
          w_scales, n, kcol0, w_scale_cols);
      smem_sbu[lane] = pack_four_scales(
          w_scales, n + kActivationWidth, kcol0, w_scale_cols);
    }
    __syncwarp();

    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int packed_k0 = tidg * 4;
    const uint32_t a0 =
        *reinterpret_cast<const uint32_t*>(&smem_a[group][packed_k0]);
    const uint32_t a1 =
        *reinterpret_cast<const uint32_t*>(&smem_a[group + 8][packed_k0]);
    const uint32_t a2 =
        *reinterpret_cast<const uint32_t*>(&smem_a[group][packed_k0 + 16]);
    const uint32_t a3 =
        *reinterpret_cast<const uint32_t*>(&smem_a[group + 8][packed_k0 + 16]);
    const uint32_t bg0 =
        *reinterpret_cast<const uint32_t*>(&smem_bg[group][packed_k0]);
    const uint32_t bg1 =
        *reinterpret_cast<const uint32_t*>(&smem_bg[group][packed_k0 + 16]);
    const uint32_t bu0 =
        *reinterpret_cast<const uint32_t*>(&smem_bu[group][packed_k0]);
    const uint32_t bu1 =
        *reinterpret_cast<const uint32_t*>(&smem_bu[group][packed_k0 + 16]);
    const int sfa_row = (lane & 1) ? (group + 8) : group;
    const uint32_t sfa = smem_sa[sfa_row];
    const uint32_t sfb_g = smem_sbg[group];
    const uint32_t sfb_u = smem_sbu[group];
    nvfp4_mma_m16n8k64(
        gg0, gg1, gg2, gg3, a0, a1, a2, a3, bg0, bg1,
        gg0, gg1, gg2, gg3, sfa, sfb_g);
    nvfp4_mma_m16n8k64(
        uu0, uu1, uu2, uu3, a0, a1, a2, a3, bu0, bu1,
        uu0, uu1, uu2, uu3, sfa, sfb_u);
    __syncwarp();
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const float g_vals[4] = {gg0, gg1, gg2, gg3};
  const float u_vals[4] = {uu0, uu1, uu2, uu3};
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t m = tile_m + r0 + r_off[i];
    const int n = tile_n + c0 + c_off[i];
    if (m >= rows || n >= kActivationWidth) {
      continue;
    }
    const __nv_bfloat16 g = __float2bfloat16_rn(g_vals[i] * alpha);
    const __nv_bfloat16 u = __float2bfloat16_rn(u_vals[i] * alpha);
    if constexpr (StoreProduct) {
      output[m * kActivationWidth + n] = __float2bfloat16_rn(
          swiglu_like_eager_bf16(g, u));
    } else {
      output[m * kRawWidth + n] = g;
      output[m * kRawWidth + kActivationWidth + n] = u;
    }
  }
}

__device__ __forceinline__ void cp_async16(
    void* dst, const void* src, unsigned nbytes) {
  const unsigned addr = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
               "r"(addr), "l"(src), "r"(nbytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait0() {
  asm volatile("cp.async.wait_group 0;\n");
}

constexpr int kPipeStages = 2;

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_piped_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ w_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 2D 64x64 tile, 4 warps. Double-buffered cp.async K pipeline.
  // Same PTX fragment + kitchen UE4M3 map as the 1-warp scaled kernel.
  // A K-slab is reused across 8 n-subtiles and both arms.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kNvfp4BigN;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kNvfp4BigM;
  __shared__ uint8_t smem_a[kPipeStages][kNvfp4BigM][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bg[kPipeStages][kNvfp4BigN][kNvfp4TileK / 2];
  __shared__ uint8_t smem_bu[kPipeStages][kNvfp4BigN][kNvfp4TileK / 2];
  __shared__ uint32_t smem_sa[kPipeStages][kNvfp4BigM];
  __shared__ uint32_t smem_sbg[kPipeStages][kNvfp4BigN];
  __shared__ uint32_t smem_sbu[kPipeStages][kNvfp4BigN];

  const int a_r = threadIdx.x >> 1;
  const int a_c = (threadIdx.x & 1) * 16;
  const int b_n = threadIdx.x >> 1;
  const int b_c = (threadIdx.x & 1) * 16;

  auto issue_stage = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    const int64_t m = tile_m + a_r;
    const uint8_t* a_src =
        a_packed + (m < rows ? m : 0) * kNvfp4PackedK + packed_k + a_c;
    cp_async16(&smem_a[buf][a_r][a_c], a_src, m < rows ? 16u : 0u);
    const int n = tile_n + b_n;
    cp_async16(
        &smem_bg[buf][b_n][b_c],
        w_packed + static_cast<int64_t>(n) * kNvfp4PackedK + packed_k + b_c,
        16u);
    cp_async16(
        &smem_bu[buf][b_n][b_c],
        w_packed
            + (static_cast<int64_t>(n) + kActivationWidth) * kNvfp4PackedK
            + packed_k + b_c,
        16u);
    if (threadIdx.x < kNvfp4BigM) {
      const int64_t sm = tile_m + threadIdx.x;
      smem_sa[buf][threadIdx.x] = (sm < rows)
          ? pack_four_scales(a_scales, sm, kcol0, a_scale_cols)
          : 0u;
      const int sn = tile_n + threadIdx.x;
      smem_sbg[buf][threadIdx.x] = pack_four_scales(
          w_scales, sn, kcol0, w_scale_cols);
      smem_sbu[buf][threadIdx.x] = pack_four_scales(
          w_scales, sn + kActivationWidth, kcol0, w_scale_cols);
    }
    cp_async_commit();
  };

  float acc_g[kNvfp4NSub][4];
  float acc_u[kNvfp4NSub][4];
#pragma unroll
  for (int ni = 0; ni < kNvfp4NSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc_g[ni][i] = 0.0f;
      acc_u[ni][i] = 0.0f;
    }
  }

  issue_stage(0, 0);
  cp_async_wait0();
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < kFc1K; k0 += kNvfp4TileK) {
    const int next = k0 + kNvfp4TileK;
    if (next < kFc1K) {
      issue_stage(next, buf ^ 1);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int packed_k0 = tidg * 4;
    const int a_row0 = warp * kNvfp4TileM + group;
    const uint32_t a0 =
        *reinterpret_cast<const uint32_t*>(&smem_a[buf][a_row0][packed_k0]);
    const uint32_t a1 =
        *reinterpret_cast<const uint32_t*>(&smem_a[buf][a_row0 + 8][packed_k0]);
    const uint32_t a2 =
        *reinterpret_cast<const uint32_t*>(
            &smem_a[buf][a_row0][packed_k0 + 16]);
    const uint32_t a3 =
        *reinterpret_cast<const uint32_t*>(
            &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
    const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
    const uint32_t sfa = smem_sa[buf][sfa_row];
#pragma unroll
    for (int ni = 0; ni < kNvfp4NSub; ++ni) {
      const int b_row = ni * kNvfp4TileN + group;
      const uint32_t bg0 =
          *reinterpret_cast<const uint32_t*>(&smem_bg[buf][b_row][packed_k0]);
      const uint32_t bg1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_bg[buf][b_row][packed_k0 + 16]);
      const uint32_t bu0 =
          *reinterpret_cast<const uint32_t*>(&smem_bu[buf][b_row][packed_k0]);
      const uint32_t bu1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_bu[buf][b_row][packed_k0 + 16]);
      const uint32_t sfb_g = smem_sbg[buf][b_row];
      const uint32_t sfb_u = smem_sbu[buf][b_row];
      nvfp4_mma_m16n8k64(
          acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
          a0, a1, a2, a3, bg0, bg1,
          acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
          sfa, sfb_g);
      nvfp4_mma_m16n8k64(
          acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
          a0, a1, a2, a3, bu0, bu1,
          acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
          sfa, sfb_u);
    }

    if (next < kFc1K) {
      cp_async_wait0();
    }
    __syncthreads();
    buf ^= 1;
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kNvfp4NSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kActivationWidth) {
        continue;
      }
      const __nv_bfloat16 g = __float2bfloat16_rn(acc_g[ni][i] * alpha);
      const __nv_bfloat16 u = __float2bfloat16_rn(acc_u[ni][i] * alpha);
      if constexpr (StoreProduct) {
        output[m * kActivationWidth + n] = __float2bfloat16_rn(
            swiglu_like_eager_bf16(g, u));
      } else {
        output[m * kRawWidth + n] = g;
        output[m * kRawWidth + kActivationWidth + n] = u;
      }
    }
  }
}

constexpr int kTmaM = 128;
constexpr int kTmaN = 128;
constexpr int kTmaNSub = kTmaN / kNvfp4TileN;
constexpr int kTmaThreads = 256;
constexpr int kTmaPackedK = kNvfp4TileK / 2;
constexpr unsigned kTmaBytes =
    static_cast<unsigned>(3 * kTmaM * kTmaPackedK);

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  const unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::
               "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(
    uint64_t* bar, unsigned tx) {
  const unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::
      "r"(addr), "r"(tx));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  const unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}

__device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, int parity) {
  const unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "WAIT_TMA: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_TMA;\n\t"
      "}\n" ::
      "r"(addr), "r"(parity));
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d(
    void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
  const unsigned d = __cvta_generic_to_shared(dst);
  const unsigned b = __cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::
      "r"(d), "l"(map), "r"(x), "r"(y), "r"(b)
      : "memory");
}

CUtensorMap make_nvfp4_tmap(
    void* ptr, uint64_t dim0, uint64_t dim1, uint32_t box0, uint32_t box1,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  alignas(64) CUtensorMap map{};
  const cuuint64_t global_dim[2] = {dim0, dim1};
  const cuuint64_t global_strides[1] = {dim0};
  const cuuint32_t box_dim[2] = {box0, box1};
  const cuuint32_t element_strides[2] = {1, 1};
  const CUresult err = cuTensorMapEncodeTiled(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      2,
      ptr,
      global_dim,
      global_strides,
      box_dim,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(err == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ",
              static_cast<int>(err));
  return map;
}

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_tma_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 128x128 2D tile, 8 warps. TMA 2-stage K pipeline + mbarrier.
  // Same PTX fragment + kitchen UE4M3 map as the scale-aware atom.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTmaN;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTmaM;
  __shared__ __align__(128) uint8_t smem_a[kPipeStages][kTmaM][kTmaPackedK];
  __shared__ __align__(128) uint8_t smem_bg[kPipeStages][kTmaN][kTmaPackedK];
  __shared__ __align__(128) uint8_t smem_bu[kPipeStages][kTmaN][kTmaPackedK];
  __shared__ uint32_t smem_sa[kPipeStages][kTmaM];
  __shared__ uint32_t smem_sbg[kPipeStages][kTmaN];
  __shared__ uint32_t smem_sbu[kPipeStages][kTmaN];
  __shared__ __align__(8) uint64_t mbar[kPipeStages];

  if (threadIdx.x == 0) {
    mbar_init(&mbar[0], 1);
    mbar_init(&mbar[1], 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar[buf], kTmaBytes);
      tma_load_2d(
          &smem_a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar[buf]);
      tma_load_2d(
          &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
      tma_load_2d(
          &smem_bu[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar[buf]);
    }
  };

  auto load_scales = [&](int k0, int buf) {
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x < kTmaM) {
      const int64_t sm = tile_m + threadIdx.x;
      smem_sa[buf][threadIdx.x] = (sm < rows)
          ? pack_four_scales(a_scales, sm, kcol0, a_scale_cols)
          : 0u;
      const int sn = tile_n + threadIdx.x;
      smem_sbg[buf][threadIdx.x] = pack_four_scales(
          w_scales, sn, kcol0, w_scale_cols);
      smem_sbu[buf][threadIdx.x] = pack_four_scales(
          w_scales, sn + kActivationWidth, kcol0, w_scale_cols);
    }
  };

  float acc_g[kTmaNSub][4];
  float acc_u[kTmaNSub][4];
#pragma unroll
  for (int ni = 0; ni < kTmaNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc_g[ni][i] = 0.0f;
      acc_u[ni][i] = 0.0f;
    }
  }

  issue_tma(0, 0);
  load_scales(0, 0);
  mbar_wait_parity(&mbar[0], 0);
  __syncthreads();

  int buf = 0;
  int phase[kPipeStages] = {1, 0};
  for (int k0 = 0; k0 < kFc1K; k0 += kNvfp4TileK) {
    const int next = k0 + kNvfp4TileK;
    if (next < kFc1K) {
      issue_tma(next, buf ^ 1);
      load_scales(next, buf ^ 1);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int packed_k0 = tidg * 4;
    const int a_row0 = warp * kNvfp4TileM + group;
    const uint32_t a0 =
        *reinterpret_cast<const uint32_t*>(&smem_a[buf][a_row0][packed_k0]);
    const uint32_t a1 =
        *reinterpret_cast<const uint32_t*>(&smem_a[buf][a_row0 + 8][packed_k0]);
    const uint32_t a2 =
        *reinterpret_cast<const uint32_t*>(
            &smem_a[buf][a_row0][packed_k0 + 16]);
    const uint32_t a3 =
        *reinterpret_cast<const uint32_t*>(
            &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
    const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
    const uint32_t sfa = smem_sa[buf][sfa_row];
#pragma unroll
    for (int ni = 0; ni < kTmaNSub; ++ni) {
      const int b_row = ni * kNvfp4TileN + group;
      const uint32_t bg0 =
          *reinterpret_cast<const uint32_t*>(&smem_bg[buf][b_row][packed_k0]);
      const uint32_t bg1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_bg[buf][b_row][packed_k0 + 16]);
      const uint32_t bu0 =
          *reinterpret_cast<const uint32_t*>(&smem_bu[buf][b_row][packed_k0]);
      const uint32_t bu1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_bu[buf][b_row][packed_k0 + 16]);
      const uint32_t sfb_g = smem_sbg[buf][b_row];
      const uint32_t sfb_u = smem_sbu[buf][b_row];
      nvfp4_mma_m16n8k64(
          acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
          a0, a1, a2, a3, bg0, bg1,
          acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
          sfa, sfb_g);
      nvfp4_mma_m16n8k64(
          acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
          a0, a1, a2, a3, bu0, bu1,
          acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
          sfa, sfb_u);
    }

    if (next < kFc1K) {
      const int nbuf = buf ^ 1;
      mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
    buf ^= 1;
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kTmaNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kActivationWidth) {
        continue;
      }
      const __nv_bfloat16 g = __float2bfloat16_rn(acc_g[ni][i] * alpha);
      const __nv_bfloat16 u = __float2bfloat16_rn(acc_u[ni][i] * alpha);
      if constexpr (StoreProduct) {
        output[m * kActivationWidth + n] = __float2bfloat16_rn(
            swiglu_like_eager_bf16(g, u));
      } else {
        output[m * kRawWidth + n] = g;
        output[m * kRawWidth + kActivationWidth + n] = u;
      }
    }
  }
}

constexpr int kScaleSlabBytes = 512;
constexpr unsigned kTmaSfBytes =
    kTmaBytes + static_cast<unsigned>(3 * kScaleSlabBytes);

__device__ __forceinline__ void cp_async_bulk(
    void* dst, const void* src, unsigned bytes, uint64_t* bar) {
  const unsigned d = __cvta_generic_to_shared(dst);
  const unsigned b = __cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n" ::
      "r"(d), "l"(src), "r"(bytes), "r"(b)
      : "memory");
}

__device__ __forceinline__ const uint8_t* scale_slab_ptr(
    const uint8_t* scales, int row0, int kcol0, int scale_cols) {
  // One cuBLAS 128x4 E4M3 slab is 512 consecutive bytes.
  const int cbg_cnt = (scale_cols + kSwizzleScaleGroup - 1)
      / kSwizzleScaleGroup;
  const int rb = row0 / kTmaM;
  const int cbg = kcol0 / kSwizzleScaleGroup;
  return scales + (static_cast<int64_t>(rb) * cbg_cnt + cbg) * kScaleSlabBytes;
}

__device__ __forceinline__ uint32_t scale_from_slab(
    const uint8_t* slab, int row) {
  // Slab order is (d3, d4, d5) with row = d4*32+d3. Four K-scales are
  // consecutive at d5=0..3, matching pack_four_scales for that row.
  return *reinterpret_cast<const uint32_t*>(
      slab + (row & 31) * 16 + (row >> 5) * 4);
}

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_tma_sf_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // Same 128x128 TMA MMA as the SWIZZLE_NONE kernel, but the UE4M3
  // vectors come from one 512-byte cuBLAS slab per K-tile per operand
  // instead of 128 scalar swizzle walks.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTmaN;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTmaM;
  __shared__ __align__(128) uint8_t smem_a[kPipeStages][kTmaM][kTmaPackedK];
  __shared__ __align__(128) uint8_t smem_bg[kPipeStages][kTmaN][kTmaPackedK];
  __shared__ __align__(128) uint8_t smem_bu[kPipeStages][kTmaN][kTmaPackedK];
  __shared__ __align__(16) uint8_t slab_sa[kPipeStages][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kPipeStages][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kPipeStages][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar[kPipeStages];

  if (threadIdx.x == 0) {
    mbar_init(&mbar[0], 1);
    mbar_init(&mbar[1], 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar[buf], kTmaSfBytes);
      tma_load_2d(
          &smem_a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar[buf]);
      tma_load_2d(
          &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
      tma_load_2d(
          &smem_bu[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar[buf]);
      cp_async_bulk(
          &slab_sa[buf][0],
          scale_slab_ptr(
              a_scales, static_cast<int>(tile_m), kcol0, a_scale_cols),
          kScaleSlabBytes, &mbar[buf]);
      cp_async_bulk(
          &slab_sbg[buf][0],
          scale_slab_ptr(w_scales, tile_n, kcol0, w_scale_cols),
          kScaleSlabBytes, &mbar[buf]);
      cp_async_bulk(
          &slab_sbu[buf][0],
          scale_slab_ptr(
              w_scales, tile_n + kActivationWidth, kcol0, w_scale_cols),
          kScaleSlabBytes, &mbar[buf]);
    }
  };

  float acc_g[kTmaNSub][4];
  float acc_u[kTmaNSub][4];
#pragma unroll
  for (int ni = 0; ni < kTmaNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc_g[ni][i] = 0.0f;
      acc_u[ni][i] = 0.0f;
    }
  }

  issue_tma(0, 0);
  mbar_wait_parity(&mbar[0], 0);
  __syncthreads();

  int buf = 0;
  int phase[kPipeStages] = {1, 0};
  for (int k0 = 0; k0 < kFc1K; k0 += kNvfp4TileK) {
    const int next = k0 + kNvfp4TileK;
    if (next < kFc1K) {
      issue_tma(next, buf ^ 1);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int packed_k0 = tidg * 4;
    const int a_row0 = warp * kNvfp4TileM + group;
    const uint32_t a0 =
        *reinterpret_cast<const uint32_t*>(&smem_a[buf][a_row0][packed_k0]);
    const uint32_t a1 =
        *reinterpret_cast<const uint32_t*>(&smem_a[buf][a_row0 + 8][packed_k0]);
    const uint32_t a2 =
        *reinterpret_cast<const uint32_t*>(
            &smem_a[buf][a_row0][packed_k0 + 16]);
    const uint32_t a3 =
        *reinterpret_cast<const uint32_t*>(
            &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
    const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
    const uint32_t sfa = scale_from_slab(slab_sa[buf], sfa_row);
#pragma unroll
    for (int ni = 0; ni < kTmaNSub; ++ni) {
      const int b_row = ni * kNvfp4TileN + group;
      const uint32_t bg0 =
          *reinterpret_cast<const uint32_t*>(&smem_bg[buf][b_row][packed_k0]);
      const uint32_t bg1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_bg[buf][b_row][packed_k0 + 16]);
      const uint32_t bu0 =
          *reinterpret_cast<const uint32_t*>(&smem_bu[buf][b_row][packed_k0]);
      const uint32_t bu1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_bu[buf][b_row][packed_k0 + 16]);
      const uint32_t sfb_g = scale_from_slab(slab_sbg[buf], b_row);
      const uint32_t sfb_u = scale_from_slab(slab_sbu[buf], b_row);
      nvfp4_mma_m16n8k64(
          acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
          a0, a1, a2, a3, bg0, bg1,
          acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
          sfa, sfb_g);
      nvfp4_mma_m16n8k64(
          acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
          a0, a1, a2, a3, bu0, bu1,
          acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
          sfa, sfb_u);
    }

    if (next < kFc1K) {
      const int nbuf = buf ^ 1;
      mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
    buf ^= 1;
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kTmaNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kActivationWidth) {
        continue;
      }
      const __nv_bfloat16 g = __float2bfloat16_rn(acc_g[ni][i] * alpha);
      const __nv_bfloat16 u = __float2bfloat16_rn(acc_u[ni][i] * alpha);
      if constexpr (StoreProduct) {
        output[m * kActivationWidth + n] = __float2bfloat16_rn(
            swiglu_like_eager_bf16(g, u));
      } else {
        output[m * kRawWidth + n] = g;
        output[m * kRawWidth + kActivationWidth + n] = u;
      }
    }
  }
}

constexpr int kTma256M = 256;
constexpr int kTma256N = 64;
constexpr int kTma256NSub = kTma256N / kNvfp4TileN;
constexpr int kTma256Threads = 256;
constexpr int kTma3Stages = 3;
constexpr unsigned kTma256Bytes =
    static_cast<unsigned>(
        kTma256M * kTmaPackedK + 2 * kTma256N * kTmaPackedK
        + 4 * kScaleSlabBytes);

template <bool Swizzle32>
__device__ __forceinline__ uint32_t ld_tma_u32(
    const uint8_t* row, int row_idx, int col) {
  // SWIZZLE_32B permutes 16B chunks inside each 32B row. Measured on
  // sm_121a: phys_col = col XOR ((row & 4) << 2) — swap halves on
  // rows 4..7 of every 8-row / 256B group. Identity when Swizzle32 is false.
  if constexpr (Swizzle32) {
    col ^= (row_idx & 4) << 2;
  }
  return *reinterpret_cast<const uint32_t*>(row + col);
}

template <bool StoreProduct, bool Swizzle32>
__global__ void fc1_paired_nvfp4_scaled_tma256_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 256x64 2D tile, 8 warps, 3-stage TMA K pipeline. Same PTX
  // fragment + cuBLAS 128x4 scale-slab map. Two A slabs cover 256 M.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ __align__(128) uint8_t smem_a[kTma3Stages][kTma256M][kTmaPackedK];
  __shared__ __align__(128) uint8_t smem_bg[kTma3Stages][kTma256N][kTmaPackedK];
  __shared__ __align__(128) uint8_t smem_bu[kTma3Stages][kTma256N][kTmaPackedK];
  __shared__ __align__(16) uint8_t slab_sa0[kTma3Stages][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sa1[kTma3Stages][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTma3Stages][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTma3Stages][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar[kTma3Stages];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTma3Stages; ++s) {
      mbar_init(&mbar[s], 1);
    }
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar[buf], kTma256Bytes);
      tma_load_2d(
          &smem_a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar[buf]);
      tma_load_2d(
          &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
      tma_load_2d(
          &smem_bu[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar[buf]);
      cp_async_bulk(
          &slab_sa0[buf][0],
          scale_slab_ptr(
              a_scales, static_cast<int>(tile_m), kcol0, a_scale_cols),
          kScaleSlabBytes, &mbar[buf]);
      cp_async_bulk(
          &slab_sa1[buf][0],
          scale_slab_ptr(
              a_scales, static_cast<int>(tile_m) + kTmaM, kcol0,
              a_scale_cols),
          kScaleSlabBytes, &mbar[buf]);
      cp_async_bulk(
          &slab_sbg[buf][0],
          scale_slab_ptr(w_scales, tile_n, kcol0, w_scale_cols),
          kScaleSlabBytes, &mbar[buf]);
      cp_async_bulk(
          &slab_sbu[buf][0],
          scale_slab_ptr(
              w_scales, tile_n + kActivationWidth, kcol0, w_scale_cols),
          kScaleSlabBytes, &mbar[buf]);
    }
  };

  float acc_g[2][kTma256NSub][4];
  float acc_u[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  constexpr int kTilesK = kFc1K / kNvfp4TileK;
  issue_tma(0, 0);
  if (kTilesK > 1) {
    issue_tma(kNvfp4TileK, 1);
  }
  mbar_wait_parity(&mbar[0], 0);
  __syncthreads();

  int phase[kTma3Stages] = {1, 0, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int buf = ki % kTma3Stages;
    const int ahead = ki + 2;
    if (ahead < kTilesK) {
      issue_tma(ahead * kNvfp4TileK, ahead % kTma3Stages);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
    const int packed_k0 = tidg * 4;
#pragma unroll
    for (int ms = 0; ms < 2; ++ms) {
      const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
      const uint32_t a0 = ld_tma_u32<Swizzle32>(
          &smem_a[buf][a_row0][0], a_row0, packed_k0);
      const uint32_t a1 = ld_tma_u32<Swizzle32>(
          &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0);
      const uint32_t a2 = ld_tma_u32<Swizzle32>(
          &smem_a[buf][a_row0][0], a_row0, packed_k0 + 16);
      const uint32_t a3 = ld_tma_u32<Swizzle32>(
          &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0 + 16);
      const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
      const uint32_t sfa = scale_from_slab(
          sfa_row < kTmaM ? slab_sa0[buf] : slab_sa1[buf],
          sfa_row & (kTmaM - 1));
#pragma unroll
      for (int ni = 0; ni < kTma256NSub; ++ni) {
        const int b_row = ni * kNvfp4TileN + group;
        const uint32_t bg0 = ld_tma_u32<Swizzle32>(
            &smem_bg[buf][b_row][0], b_row, packed_k0);
        const uint32_t bg1 = ld_tma_u32<Swizzle32>(
            &smem_bg[buf][b_row][0], b_row, packed_k0 + 16);
        const uint32_t bu0 = ld_tma_u32<Swizzle32>(
            &smem_bu[buf][b_row][0], b_row, packed_k0);
        const uint32_t bu1 = ld_tma_u32<Swizzle32>(
            &smem_bu[buf][b_row][0], b_row, packed_k0 + 16);
        const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
        const uint32_t sfb_g = scale_from_slab(slab_sbg[buf], sfb_row);
        const uint32_t sfb_u = scale_from_slab(slab_sbu[buf], sfb_row);
        nvfp4_mma_m16n8k64(
            acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
            acc_g[ms][ni][3],
            a0, a1, a2, a3, bg0, bg1,
            acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
            acc_g[ms][ni][3],
            sfa, sfb_g);
        nvfp4_mma_m16n8k64(
            acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
            acc_u[ms][ni][3],
            a0, a1, a2, a3, bu0, bu1,
            acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
            acc_u[ms][ni][3],
            sfa, sfb_u);
      }
    }

    const int nxt = ki + 1;
    if (nxt < kTilesK) {
      const int nbuf = nxt % kTma3Stages;
      mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        const __nv_bfloat16 g =
            __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
        const __nv_bfloat16 u =
            __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
        if constexpr (StoreProduct) {
          output[m * kActivationWidth + n] = __float2bfloat16_rn(
              swiglu_like_eager_bf16(g, u));
        } else {
          output[m * kRawWidth + n] = g;
          output[m * kRawWidth + kActivationWidth + n] = u;
        }
      }
    }
  }
}

constexpr int kTmaK2Packed = 64;
constexpr int kTmaK2 = 128;
constexpr unsigned kTmaK2Bytes =
    static_cast<unsigned>(
        kTma256M * kTmaK2Packed + 2 * kTma256N * kTmaK2Packed
        + 8 * kScaleSlabBytes);

template <bool Swizzle64>
__device__ __forceinline__ uint32_t ld_tma64_u32(
    const uint8_t* row, int row_idx, int col) {
  // SWIZZLE_64B permutes 16B chunks inside each 64B row. Measured on
  // sm_121a: phys_col = col XOR (((row >> 1) & 3) << 4). Same row.
  if constexpr (Swizzle64) {
    col ^= ((row_idx >> 1) & 3) << 4;
  }
  return *reinterpret_cast<const uint32_t*>(row + col);
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
    const void* addr) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(addr));
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(s));
}

template <bool StoreProduct, bool Swizzle64, bool ReduceAmax = false,
          bool LdmA = false, bool LdmB = false, bool PipeB = false,
          bool LeadS = false, bool PipeA = false>
__global__ void fc1_paired_nvfp4_scaled_tma256k2_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha,
    uint32_t* __restrict__ global_max_bits) {
  // 256x64 tile, 8 warps, 3-stage TMA of K=128 (64 packed bytes).
  // One load feeds two m16n8k64 atoms. Two 128x4 scale slabs per
  // operand cover the 8 K-scale columns. Same PTX fragment as K=64.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ __align__(128) uint8_t smem_a[kTma3Stages][kTma256M][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(16) uint8_t slab_sa0[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sa1[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar[kTma3Stages];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTma3Stages; ++s) {
      mbar_init(&mbar[s], 1);
    }
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar[buf], kTmaK2Bytes);
      tma_load_2d(
          &smem_a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar[buf]);
      tma_load_2d(
          &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
      tma_load_2d(
          &smem_bu[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar[buf]);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &slab_sa0[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sa1[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                a_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sbg[buf][kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sbu[buf][kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
      }
    }
  };

  float acc_g[2][kTma256NSub][4];
  float acc_u[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK2;
  issue_tma(0, 0);
  if (kTilesK > 1) {
    issue_tma(kTmaK2, 1);
  }
  mbar_wait_parity(&mbar[0], 0);
  __syncthreads();

  int phase[kTma3Stages] = {1, 0, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int buf = ki % kTma3Stages;
    const int ahead = ki + 2;
    if (ahead < kTilesK) {
      issue_tma(ahead * kTmaK2, ahead % kTma3Stages);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
    if constexpr (PipeA) {
      // Next K=64 A/SFA overlaps m16n8k64. Same fragment as scalar k2.
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
        auto load_a = [&](int ks, uint32_t& a0, uint32_t& a1,
                          uint32_t& a2, uint32_t& a3, uint32_t& sfa) {
          const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
          a0 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0][0], a_row0, packed_k0);
          a1 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0);
          a2 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0][0], a_row0, packed_k0 + 16);
          a3 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0 + 16);
          const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
          sfa = scale_from_slab(
              sfa_row < kTmaM ? slab_sa0[buf][ks] : slab_sa1[buf][ks],
              sfa_row & (kTmaM - 1));
        };
        uint32_t a0, a1, a2, a3, sfa;
        load_a(0, a0, a1, a2, a3, sfa);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t na0 = 0, na1 = 0, na2 = 0, na3 = 0, nsfa = 0;
          if (ks + 1 < 2) {
            load_a(ks + 1, na0, na1, na2, na3, nsfa);
          }
          const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
          for (int ni = 0; ni < kTma256NSub; ++ni) {
            const int b_row = ni * kNvfp4TileN + group;
            const uint32_t bg0 = ld_tma64_u32<Swizzle64>(
                &smem_bg[buf][b_row][0], b_row, packed_k0);
            const uint32_t bg1 = ld_tma64_u32<Swizzle64>(
                &smem_bg[buf][b_row][0], b_row, packed_k0 + 16);
            const uint32_t bu0 = ld_tma64_u32<Swizzle64>(
                &smem_bu[buf][b_row][0], b_row, packed_k0);
            const uint32_t bu1 = ld_tma64_u32<Swizzle64>(
                &smem_bu[buf][b_row][0], b_row, packed_k0 + 16);
            const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
            const uint32_t sfb_g =
                scale_from_slab(slab_sbg[buf][ks], sfb_row);
            const uint32_t sfb_u =
                scale_from_slab(slab_sbu[buf][ks], sfb_row);
            nvfp4_mma_m16n8k64(
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                a0, a1, a2, a3, bg0, bg1,
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                sfa, sfb_g);
            nvfp4_mma_m16n8k64(
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                a0, a1, a2, a3, bu0, bu1,
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                sfa, sfb_u);
          }
          if (ks + 1 < 2) {
            a0 = na0;
            a1 = na1;
            a2 = na2;
            a3 = na3;
            sfa = nsfa;
          }
        }
      }
    } else {
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
        uint32_t a0, a1, a2, a3;
        if constexpr (LdmA) {
          const int ar = ms * kTmaM + warp * kNvfp4TileM + (lane & 15);
          const int ac = ks * (kNvfp4TileK / 2) + (lane >> 4) * 16;
          ldmatrix_x4(a0, a1, a2, a3, &smem_a[buf][ar][ac]);
        } else {
          a0 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0][0], a_row0, packed_k0);
          a1 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0);
          a2 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0][0], a_row0, packed_k0 + 16);
          a3 = ld_tma64_u32<Swizzle64>(
              &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0 + 16);
        }
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        // scale_vec::4X: thread-id-a=0 reads SFA from lanes where
        // ((lane>>1)&1)==0 (16 threads, one row each). thread-id-b=0
        // reads SFB from (lane&3)==0 (8 threads, one N-col each).
        const uint32_t sfa =
            (!LeadS || ((lane >> 1) & 1) == 0)
                ? scale_from_slab(
                      sfa_row < kTmaM ? slab_sa0[buf][ks]
                                      : slab_sa1[buf][ks],
                      sfa_row & (kTmaM - 1))
                : 0u;
        if constexpr (LdmB) {
          // One ldmatrix.x4 covers two 8-N subtiles. Same A-fragment
          // map: d0/d2 = ni, d1/d3 = ni+8. x2 B is a different
          // thread permutation and is not this atom.
#pragma unroll
          for (int ni = 0; ni < kTma256NSub; ni += 2) {
            const int br = ni * kNvfp4TileN + (lane & 15);
            const int bc =
                ks * (kNvfp4TileK / 2) + (lane >> 4) * 16;
            uint32_t bg0, bg0n, bg1, bg1n;
            uint32_t bu0, bu0n, bu1, bu1n;
            ldmatrix_x4(
                bg0, bg0n, bg1, bg1n, &smem_bg[buf][br][bc]);
            ldmatrix_x4(
                bu0, bu0n, bu1, bu1n, &smem_bu[buf][br][bc]);
            const int b_row = ni * kNvfp4TileN + group;
            const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
            const uint32_t sfb_g =
                scale_from_slab(slab_sbg[buf][ks], sfb_row);
            const uint32_t sfb_u =
                scale_from_slab(slab_sbu[buf][ks], sfb_row);
            const uint32_t sfb_g1 = scale_from_slab(
                slab_sbg[buf][ks], sfb_row + kNvfp4TileN);
            const uint32_t sfb_u1 = scale_from_slab(
                slab_sbu[buf][ks], sfb_row + kNvfp4TileN);
            nvfp4_mma_m16n8k64(
                acc_g[ms][ni][0], acc_g[ms][ni][1],
                acc_g[ms][ni][2], acc_g[ms][ni][3],
                a0, a1, a2, a3, bg0, bg1,
                acc_g[ms][ni][0], acc_g[ms][ni][1],
                acc_g[ms][ni][2], acc_g[ms][ni][3],
                sfa, sfb_g);
            nvfp4_mma_m16n8k64(
                acc_u[ms][ni][0], acc_u[ms][ni][1],
                acc_u[ms][ni][2], acc_u[ms][ni][3],
                a0, a1, a2, a3, bu0, bu1,
                acc_u[ms][ni][0], acc_u[ms][ni][1],
                acc_u[ms][ni][2], acc_u[ms][ni][3],
                sfa, sfb_u);
            nvfp4_mma_m16n8k64(
                acc_g[ms][ni + 1][0], acc_g[ms][ni + 1][1],
                acc_g[ms][ni + 1][2], acc_g[ms][ni + 1][3],
                a0, a1, a2, a3, bg0n, bg1n,
                acc_g[ms][ni + 1][0], acc_g[ms][ni + 1][1],
                acc_g[ms][ni + 1][2], acc_g[ms][ni + 1][3],
                sfa, sfb_g1);
            nvfp4_mma_m16n8k64(
                acc_u[ms][ni + 1][0], acc_u[ms][ni + 1][1],
                acc_u[ms][ni + 1][2], acc_u[ms][ni + 1][3],
                a0, a1, a2, a3, bu0n, bu1n,
                acc_u[ms][ni + 1][0], acc_u[ms][ni + 1][1],
                acc_u[ms][ni + 1][2], acc_u[ms][ni + 1][3],
                sfa, sfb_u1);
          }
        } else if constexpr (PipeB) {
          int b_row = group;
          uint32_t bg0 = ld_tma64_u32<Swizzle64>(
              &smem_bg[buf][b_row][0], b_row, packed_k0);
          uint32_t bg1 = ld_tma64_u32<Swizzle64>(
              &smem_bg[buf][b_row][0], b_row, packed_k0 + 16);
          uint32_t bu0 = ld_tma64_u32<Swizzle64>(
              &smem_bu[buf][b_row][0], b_row, packed_k0);
          uint32_t bu1 = ld_tma64_u32<Swizzle64>(
              &smem_bu[buf][b_row][0], b_row, packed_k0 + 16);
          int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
          uint32_t sfb_g = scale_from_slab(slab_sbg[buf][ks], sfb_row);
          uint32_t sfb_u = scale_from_slab(slab_sbu[buf][ks], sfb_row);
#pragma unroll
          for (int ni = 0; ni < kTma256NSub; ++ni) {
            uint32_t nbg0 = 0, nbg1 = 0, nbu0 = 0, nbu1 = 0;
            uint32_t nsfb_g = 0, nsfb_u = 0;
            if (ni + 1 < kTma256NSub) {
              const int nb_row = (ni + 1) * kNvfp4TileN + group;
              nbg0 = ld_tma64_u32<Swizzle64>(
                  &smem_bg[buf][nb_row][0], nb_row, packed_k0);
              nbg1 = ld_tma64_u32<Swizzle64>(
                  &smem_bg[buf][nb_row][0], nb_row, packed_k0 + 16);
              nbu0 = ld_tma64_u32<Swizzle64>(
                  &smem_bu[buf][nb_row][0], nb_row, packed_k0);
              nbu1 = ld_tma64_u32<Swizzle64>(
                  &smem_bu[buf][nb_row][0], nb_row, packed_k0 + 16);
              const int nsfb_row = (tile_n & (kTmaM - 1)) + nb_row;
              nsfb_g = scale_from_slab(slab_sbg[buf][ks], nsfb_row);
              nsfb_u = scale_from_slab(slab_sbu[buf][ks], nsfb_row);
            }
            nvfp4_mma_m16n8k64(
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                a0, a1, a2, a3, bg0, bg1,
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                sfa, sfb_g);
            nvfp4_mma_m16n8k64(
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                a0, a1, a2, a3, bu0, bu1,
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                sfa, sfb_u);
            if (ni + 1 < kTma256NSub) {
              bg0 = nbg0;
              bg1 = nbg1;
              bu0 = nbu0;
              bu1 = nbu1;
              sfb_g = nsfb_g;
              sfb_u = nsfb_u;
            }
          }
        } else {
#pragma unroll
          for (int ni = 0; ni < kTma256NSub; ++ni) {
            const int b_row = ni * kNvfp4TileN + group;
            const uint32_t bg0 = ld_tma64_u32<Swizzle64>(
                &smem_bg[buf][b_row][0], b_row, packed_k0);
            const uint32_t bg1 = ld_tma64_u32<Swizzle64>(
                &smem_bg[buf][b_row][0], b_row, packed_k0 + 16);
            const uint32_t bu0 = ld_tma64_u32<Swizzle64>(
                &smem_bu[buf][b_row][0], b_row, packed_k0);
            const uint32_t bu1 = ld_tma64_u32<Swizzle64>(
                &smem_bu[buf][b_row][0], b_row, packed_k0 + 16);
            const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
            const uint32_t sfb_g =
                (!LeadS || (lane & 3) == 0)
                    ? scale_from_slab(slab_sbg[buf][ks], sfb_row)
                    : 0u;
            const uint32_t sfb_u =
                (!LeadS || (lane & 3) == 0)
                    ? scale_from_slab(slab_sbu[buf][ks], sfb_row)
                    : 0u;
            nvfp4_mma_m16n8k64(
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                a0, a1, a2, a3, bg0, bg1,
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                sfa, sfb_g);
            nvfp4_mma_m16n8k64(
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                a0, a1, a2, a3, bu0, bu1,
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                sfa, sfb_u);
          }
        }
      }
    }
    }

    const int nxt = ki + 1;
    if (nxt < kTilesK) {
      const int nbuf = nxt % kTma3Stages;
      mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
  uint32_t local_max = 0;
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        const __nv_bfloat16 g =
            __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
        const __nv_bfloat16 u =
            __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
        if constexpr (StoreProduct) {
          output[m * kActivationWidth + n] = __float2bfloat16_rn(
              swiglu_like_eager_bf16(g, u));
        } else {
          output[m * kRawWidth + n] = g;
          output[m * kRawWidth + kActivationWidth + n] = u;
        }
        if constexpr (ReduceAmax) {
          const uint32_t bits = swiglu_abs_bits(g, u);
          local_max = local_max < bits ? bits : local_max;
        }
      }
    }
  }
  if constexpr (ReduceAmax) {
    reduce_atomic_max_bits(local_max, global_max_bits);
  }
}

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_tma256k2p_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // Persistent 8-warp k2: one CTA owns 256 M and walks every
  // 64-wide N tile. Same m16n8k64 atom, 2x8x4 acc, and 3-stage
  // K=128 TMA as k2. A for this panel is streamed across N so
  // L2 can reuse it. Smem matches k2 (under 99 KiB).
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ __align__(128) uint8_t smem_a[kTma3Stages][kTma256M][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(16) uint8_t slab_sa0[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sa1[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar[kTma3Stages];

  constexpr int kNTiles = kActivationWidth / kTma256N;
  constexpr int kTilesK = kFc1K / kTmaK2;
  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};

  for (int nt = static_cast<int>(blockIdx.x); nt < kNTiles;
       nt += static_cast<int>(gridDim.x)) {
    const int tile_n = nt * kTma256N;
    if (threadIdx.x == 0) {
#pragma unroll
      for (int s = 0; s < kTma3Stages; ++s) {
        mbar_init(&mbar[s], 1);
      }
    }
    __syncthreads();

    auto issue_tma = [&](int k0, int buf) {
      const int packed_k = k0 / 2;
      const int kcol0 = k0 / kFp4BlockSize;
      if (threadIdx.x == 0) {
        mbar_arrive_expect_tx(&mbar[buf], kTmaK2Bytes);
        tma_load_2d(
            &smem_a[buf][0][0], &a_map, packed_k,
            static_cast<int>(tile_m), &mbar[buf]);
        tma_load_2d(
            &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
        tma_load_2d(
            &smem_bu[buf][0][0], &w_map, packed_k,
            tile_n + kActivationWidth, &mbar[buf]);
#pragma unroll
        for (int kg = 0; kg < 2; ++kg) {
          const int kc = kcol0 + kg * kSwizzleScaleGroup;
          cp_async_bulk(
              &slab_sa0[buf][kg][0],
              scale_slab_ptr(
                  a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
              kScaleSlabBytes, &mbar[buf]);
          cp_async_bulk(
              &slab_sa1[buf][kg][0],
              scale_slab_ptr(
                  a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                  a_scale_cols),
              kScaleSlabBytes, &mbar[buf]);
          cp_async_bulk(
              &slab_sbg[buf][kg][0],
              scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
              kScaleSlabBytes, &mbar[buf]);
          cp_async_bulk(
              &slab_sbu[buf][kg][0],
              scale_slab_ptr(
                  w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
              kScaleSlabBytes, &mbar[buf]);
        }
      }
    };

    float acc_g[2][kTma256NSub][4];
    float acc_u[2][kTma256NSub][4];
#pragma unroll
    for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
      for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc_g[ms][ni][i] = 0.0f;
          acc_u[ms][ni][i] = 0.0f;
        }
      }
    }

    issue_tma(0, 0);
    if (kTilesK > 1) {
      issue_tma(kTmaK2, 1);
    }
    mbar_wait_parity(&mbar[0], 0);
    __syncthreads();

    int phase[kTma3Stages] = {1, 0, 0};
    for (int ki = 0; ki < kTilesK; ++ki) {
      const int buf = ki % kTma3Stages;
      const int ahead = ki + 2;
      if (ahead < kTilesK) {
        issue_tma(ahead * kTmaK2, ahead % kTma3Stages);
      }

      const int group = lane >> 2;
      const int tidg = lane & 3;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
        for (int ms = 0; ms < 2; ++ms) {
          const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
          const uint32_t a0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_a[buf][a_row0][packed_k0]);
          const uint32_t a1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_a[buf][a_row0 + 8][packed_k0]);
          const uint32_t a2 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_a[buf][a_row0][packed_k0 + 16]);
          const uint32_t a3 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
          const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
          const uint32_t sfa = scale_from_slab(
              sfa_row < kTmaM ? slab_sa0[buf][ks] : slab_sa1[buf][ks],
              sfa_row & (kTmaM - 1));
#pragma unroll
          for (int ni = 0; ni < kTma256NSub; ++ni) {
            const int b_row = ni * kNvfp4TileN + group;
            const uint32_t bg0 =
                *reinterpret_cast<const uint32_t*>(
                    &smem_bg[buf][b_row][packed_k0]);
            const uint32_t bg1 =
                *reinterpret_cast<const uint32_t*>(
                    &smem_bg[buf][b_row][packed_k0 + 16]);
            const uint32_t bu0 =
                *reinterpret_cast<const uint32_t*>(
                    &smem_bu[buf][b_row][packed_k0]);
            const uint32_t bu1 =
                *reinterpret_cast<const uint32_t*>(
                    &smem_bu[buf][b_row][packed_k0 + 16]);
            const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
            const uint32_t sfb_g =
                scale_from_slab(slab_sbg[buf][ks], sfb_row);
            const uint32_t sfb_u =
                scale_from_slab(slab_sbu[buf][ks], sfb_row);
            nvfp4_mma_m16n8k64(
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                a0, a1, a2, a3, bg0, bg1,
                acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
                acc_g[ms][ni][3],
                sfa, sfb_g);
            nvfp4_mma_m16n8k64(
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                a0, a1, a2, a3, bu0, bu1,
                acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
                acc_u[ms][ni][3],
                sfa, sfb_u);
          }
        }
      }

      const int nxt = ki + 1;
      if (nxt < kTilesK) {
        const int nbuf = nxt % kTma3Stages;
        mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
        phase[nbuf] ^= 1;
      }
      __syncthreads();
    }

#pragma unroll
    for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
      for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int64_t m =
              tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
          const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
          if (m >= rows || n >= kActivationWidth) {
            continue;
          }
          const __nv_bfloat16 g =
              __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
          const __nv_bfloat16 u =
              __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
          if constexpr (StoreProduct) {
            output[m * kActivationWidth + n] = __float2bfloat16_rn(
                swiglu_like_eager_bf16(g, u));
          } else {
            output[m * kRawWidth + n] = g;
            output[m * kRawWidth + kActivationWidth + n] = u;
          }
        }
      }
    }
  }
}

constexpr int kTmaK2WsThreads = 288;
constexpr int kTmaK2WsMmaWarps = 8;

template <bool StoreProduct>
__global__ void __launch_bounds__(kTmaK2WsThreads, 1)
fc1_paired_nvfp4_scaled_tma256k2ws_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // Warp-specialized k2: warp 0 produces TMA, warps 1-8 consume
  // the same 256x64 m16n8k64 atom. Smem matches k2 (under 99 KiB).
  // Full/empty mbarriers let the producer stay two K-tiles ahead
  // without a CTA syncthreads in the mainloop.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ __align__(128) uint8_t smem_a[kTma3Stages][kTma256M][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(16) uint8_t slab_sa0[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sa1[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t full[kTma3Stages];
  __shared__ __align__(8) uint64_t empty[kTma3Stages];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTma3Stages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], kTmaK2WsMmaWarps);
    }
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    mbar_arrive_expect_tx(&full[buf], kTmaK2Bytes);
    tma_load_2d(
        &smem_a[buf][0][0], &a_map, packed_k,
        static_cast<int>(tile_m), &full[buf]);
    tma_load_2d(
        &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &full[buf]);
    tma_load_2d(
        &smem_bu[buf][0][0], &w_map, packed_k,
        tile_n + kActivationWidth, &full[buf]);
#pragma unroll
    for (int kg = 0; kg < 2; ++kg) {
      const int kc = kcol0 + kg * kSwizzleScaleGroup;
      cp_async_bulk(
          &slab_sa0[buf][kg][0],
          scale_slab_ptr(
              a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
          kScaleSlabBytes, &full[buf]);
      cp_async_bulk(
          &slab_sa1[buf][kg][0],
          scale_slab_ptr(
              a_scales, static_cast<int>(tile_m) + kTmaM, kc,
              a_scale_cols),
          kScaleSlabBytes, &full[buf]);
      cp_async_bulk(
          &slab_sbg[buf][kg][0],
          scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
          kScaleSlabBytes, &full[buf]);
      cp_async_bulk(
          &slab_sbu[buf][kg][0],
          scale_slab_ptr(
              w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
          kScaleSlabBytes, &full[buf]);
    }
  };

  constexpr int kTilesK = kFc1K / kTmaK2;
  if (warp == 0) {
    if (lane == 0) {
      int empty_phase[kTma3Stages] = {0, 0, 0};
      for (int ki = 0; ki < kTilesK; ++ki) {
        const int buf = ki % kTma3Stages;
        if (ki >= kTma3Stages) {
          mbar_wait_parity(&empty[buf], empty_phase[buf]);
          empty_phase[buf] ^= 1;
        }
        issue_tma(ki * kTmaK2, buf);
      }
    }
    return;
  }

  const int mma_warp = warp - 1;
  const int group = lane >> 2;
  const int tidg = lane & 3;
  float acc_g[2][kTma256NSub][4];
  float acc_u[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  mbar_wait_parity(&full[0], 0);
  int full_phase[kTma3Stages] = {1, 0, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int buf = ki % kTma3Stages;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + mma_warp * kNvfp4TileM + group;
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0][packed_k0]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0 + 8][packed_k0]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0][packed_k0 + 16]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        const uint32_t sfa = scale_from_slab(
            sfa_row < kTmaM ? slab_sa0[buf][ks] : slab_sa1[buf][ks],
            sfa_row & (kTmaM - 1));
#pragma unroll
        for (int ni = 0; ni < kTma256NSub; ++ni) {
          const int b_row = ni * kNvfp4TileN + group;
          const uint32_t bg0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[buf][b_row][packed_k0]);
          const uint32_t bg1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[buf][b_row][packed_k0 + 16]);
          const uint32_t bu0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[buf][b_row][packed_k0]);
          const uint32_t bu1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[buf][b_row][packed_k0 + 16]);
          const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb_g =
              scale_from_slab(slab_sbg[buf][ks], sfb_row);
          const uint32_t sfb_u =
              scale_from_slab(slab_sbu[buf][ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              a0, a1, a2, a3, bg0, bg1,
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              sfa, sfb_g);
          nvfp4_mma_m16n8k64(
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              a0, a1, a2, a3, bu0, bu1,
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              sfa, sfb_u);
        }
      }
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(&empty[buf]);
    }
    const int nxt = ki + 1;
    if (nxt < kTilesK) {
      const int nbuf = nxt % kTma3Stages;
      mbar_wait_parity(&full[nbuf], full_phase[nbuf]);
      full_phase[nbuf] ^= 1;
    }
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + mma_warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        const __nv_bfloat16 g =
            __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
        const __nv_bfloat16 u =
            __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
        if constexpr (StoreProduct) {
          output[m * kActivationWidth + n] = __float2bfloat16_rn(
              swiglu_like_eager_bf16(g, u));
        } else {
          output[m * kRawWidth + n] = g;
          output[m * kRawWidth + kActivationWidth + n] = u;
        }
      }
    }
  }
}

// Kitchen 12-warp split on the k2 tile: 4 producer warps issue
// A / Bg / Bu / scales in parallel; 8 MMA warps keep the 256x64
// K=128 atom. 9-warp k2ws serializes every TMA on one lane.
constexpr int kTmaK2Ws4ProdWarps = 4;
constexpr int kTmaK2Ws4MmaWarps = 8;
constexpr int kTmaK2Ws4Threads = 384;
constexpr unsigned kTmaK2Ws4ABytes =
    static_cast<unsigned>(kTma256M * kTmaK2Packed);
constexpr unsigned kTmaK2Ws4BBytes =
    static_cast<unsigned>(kTma256N * kTmaK2Packed);
constexpr unsigned kTmaK2Ws4SBytes =
    static_cast<unsigned>(8 * kScaleSlabBytes);

__global__ void __launch_bounds__(kTmaK2Ws4Threads, 1)
fc1_paired_nvfp4_scaled_tma256k2ws4_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ __align__(128) uint8_t smem_a[kTma3Stages][kTma256M][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTma3Stages][kTma256N][kTmaK2Packed];
  __shared__ __align__(16) uint8_t slab_sa0[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sa1[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t full[kTma3Stages];
  __shared__ __align__(8) uint64_t empty[kTma3Stages];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTma3Stages; ++s) {
      mbar_init(&full[s], kTmaK2Ws4ProdWarps);
      mbar_init(&empty[s], kTmaK2Ws4MmaWarps);
    }
  }
  __syncthreads();

  constexpr int kTilesK = kFc1K / kTmaK2;
  if (warp < kTmaK2Ws4ProdWarps) {
    if (lane == 0) {
      int empty_phase[kTma3Stages] = {0, 0, 0};
      for (int ki = 0; ki < kTilesK; ++ki) {
        const int buf = ki % kTma3Stages;
        if (ki >= kTma3Stages) {
          mbar_wait_parity(&empty[buf], empty_phase[buf]);
          empty_phase[buf] ^= 1;
        }
        const int k0 = ki * kTmaK2;
        const int packed_k = k0 / 2;
        const int kcol0 = k0 / kFp4BlockSize;
        if (warp == 0) {
          mbar_arrive_expect_tx(&full[buf], kTmaK2Ws4ABytes);
          tma_load_2d(
              &smem_a[buf][0][0], &a_map, packed_k,
              static_cast<int>(tile_m), &full[buf]);
        } else if (warp == 1) {
          mbar_arrive_expect_tx(&full[buf], kTmaK2Ws4BBytes);
          tma_load_2d(
              &smem_bg[buf][0][0], &w_map, packed_k, tile_n,
              &full[buf]);
        } else if (warp == 2) {
          mbar_arrive_expect_tx(&full[buf], kTmaK2Ws4BBytes);
          tma_load_2d(
              &smem_bu[buf][0][0], &w_map, packed_k,
              tile_n + kActivationWidth, &full[buf]);
        } else {
          mbar_arrive_expect_tx(&full[buf], kTmaK2Ws4SBytes);
#pragma unroll
          for (int kg = 0; kg < 2; ++kg) {
            const int kc = kcol0 + kg * kSwizzleScaleGroup;
            cp_async_bulk(
                &slab_sa0[buf][kg][0],
                scale_slab_ptr(
                    a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
                kScaleSlabBytes, &full[buf]);
            cp_async_bulk(
                &slab_sa1[buf][kg][0],
                scale_slab_ptr(
                    a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                    a_scale_cols),
                kScaleSlabBytes, &full[buf]);
            cp_async_bulk(
                &slab_sbg[buf][kg][0],
                scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
                kScaleSlabBytes, &full[buf]);
            cp_async_bulk(
                &slab_sbu[buf][kg][0],
                scale_slab_ptr(
                    w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
                kScaleSlabBytes, &full[buf]);
          }
        }
      }
    }
    return;
  }

  const int mma_warp = warp - kTmaK2Ws4ProdWarps;
  const int group = lane >> 2;
  const int tidg = lane & 3;
  float acc_g[2][kTma256NSub][4];
  float acc_u[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  mbar_wait_parity(&full[0], 0);
  int full_phase[kTma3Stages] = {1, 0, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int buf = ki % kTma3Stages;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + mma_warp * kNvfp4TileM + group;
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0][packed_k0]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0 + 8][packed_k0]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0][packed_k0 + 16]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        const uint32_t sfa = scale_from_slab(
            sfa_row < kTmaM ? slab_sa0[buf][ks] : slab_sa1[buf][ks],
            sfa_row & (kTmaM - 1));
#pragma unroll
        for (int ni = 0; ni < kTma256NSub; ++ni) {
          const int b_row = ni * kNvfp4TileN + group;
          const uint32_t bg0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[buf][b_row][packed_k0]);
          const uint32_t bg1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[buf][b_row][packed_k0 + 16]);
          const uint32_t bu0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[buf][b_row][packed_k0]);
          const uint32_t bu1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[buf][b_row][packed_k0 + 16]);
          const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb_g =
              scale_from_slab(slab_sbg[buf][ks], sfb_row);
          const uint32_t sfb_u =
              scale_from_slab(slab_sbu[buf][ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              a0, a1, a2, a3, bg0, bg1,
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              sfa, sfb_g);
          nvfp4_mma_m16n8k64(
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              a0, a1, a2, a3, bu0, bu1,
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              sfa, sfb_u);
        }
      }
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(&empty[buf]);
    }
    const int nxt = ki + 1;
    if (nxt < kTilesK) {
      const int nbuf = nxt % kTma3Stages;
      mbar_wait_parity(&full[nbuf], full_phase[nbuf]);
      full_phase[nbuf] ^= 1;
    }
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + mma_warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        output[m * kRawWidth + n] =
            __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
        output[m * kRawWidth + kActivationWidth + n] =
            __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
      }
    }
  }
}

constexpr int kTmaK4M = 128;
constexpr int kTmaK4Packed = 128;
constexpr int kTmaK4 = 256;
constexpr int kTmaK4Stages = 2;
constexpr int kTmaK4Kg = 4;
constexpr unsigned kTmaK4Bytes =
    static_cast<unsigned>(
        kTmaK4M * kTmaK4Packed + 2 * kTma256N * kTmaK4Packed
        + (1 + 2) * kTmaK4Kg * kScaleSlabBytes);

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_tma128k4_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 128x64 tile, 8 warps, 2-stage TMA of K=256 (128 packed bytes).
  // One load feeds four m16n8k64 atoms. Four 128x4 scale slabs per
  // operand cover the 16 K-scale columns. Same PTX fragment.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTmaK4M;
  __shared__ __align__(128) uint8_t smem_a[kTmaK4Stages][kTmaK4M][kTmaK4Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTmaK4Stages][kTma256N][kTmaK4Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTmaK4Stages][kTma256N][kTmaK4Packed];
  __shared__ __align__(16) uint8_t slab_sa[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar[kTmaK4Stages];

  if (threadIdx.x == 0) {
    mbar_init(&mbar[0], 1);
    mbar_init(&mbar[1], 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar[buf], kTmaK4Bytes);
      tma_load_2d(
          &smem_a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar[buf]);
      tma_load_2d(
          &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
      tma_load_2d(
          &smem_bu[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar[buf]);
#pragma unroll
      for (int kg = 0; kg < kTmaK4Kg; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &slab_sa[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sbg[buf][kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sbu[buf][kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
      }
    }
  };

  float acc_g[kTma256NSub][4];
  float acc_u[kTma256NSub][4];
#pragma unroll
  for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc_g[ni][i] = 0.0f;
      acc_u[ni][i] = 0.0f;
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK4;
  issue_tma(0, 0);
  mbar_wait_parity(&mbar[0], 0);
  __syncthreads();

  int buf = 0;
  int phase[kTmaK4Stages] = {1, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK4, buf ^ 1);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < kTmaK4Kg; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
      const int a_row0 = warp * kNvfp4TileM + group;
      const uint32_t a0 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0][packed_k0]);
      const uint32_t a1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0 + 8][packed_k0]);
      const uint32_t a2 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0][packed_k0 + 16]);
      const uint32_t a3 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
      const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
      const uint32_t sfa = scale_from_slab(slab_sa[buf][ks], sfa_row);
#pragma unroll
      for (int ni = 0; ni < kTma256NSub; ++ni) {
        const int b_row = ni * kNvfp4TileN + group;
        const uint32_t bg0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bg[buf][b_row][packed_k0]);
        const uint32_t bg1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bg[buf][b_row][packed_k0 + 16]);
        const uint32_t bu0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bu[buf][b_row][packed_k0]);
        const uint32_t bu1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bu[buf][b_row][packed_k0 + 16]);
        const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
        const uint32_t sfb_g = scale_from_slab(slab_sbg[buf][ks], sfb_row);
        const uint32_t sfb_u = scale_from_slab(slab_sbu[buf][ks], sfb_row);
        nvfp4_mma_m16n8k64(
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            a0, a1, a2, a3, bg0, bg1,
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            sfa, sfb_g);
        nvfp4_mma_m16n8k64(
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            a0, a1, a2, a3, bu0, bu1,
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            sfa, sfb_u);
      }
    }

    if (next < kTilesK) {
      const int nbuf = buf ^ 1;
      mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
    buf ^= 1;
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kActivationWidth) {
        continue;
      }
      const __nv_bfloat16 g = __float2bfloat16_rn(acc_g[ni][i] * alpha);
      const __nv_bfloat16 u = __float2bfloat16_rn(acc_u[ni][i] * alpha);
      if constexpr (StoreProduct) {
        output[m * kActivationWidth + n] = __float2bfloat16_rn(
            swiglu_like_eager_bf16(g, u));
      } else {
        output[m * kRawWidth + n] = g;
        output[m * kRawWidth + kActivationWidth + n] = u;
      }
    }
  }
}

constexpr int kTmaKitN = 128;
constexpr int kTmaKitNSub = kTmaKitN / kNvfp4TileN;
constexpr unsigned kTmaKitBytes =
    static_cast<unsigned>(
        kTmaK4M * kTmaK4Packed + kTmaKitN * kTmaK4Packed
        + 2 * kTmaK4Kg * kScaleSlabBytes);

template <bool Swizzle128>
__device__ __forceinline__ uint32_t ld_tma128_u32(
    const uint8_t* row, int row_idx, int col) {
  // SWIZZLE_128B permutes 16B chunks inside each 128B row. Measured
  // on sm_121a: phys_col = col XOR ((row & 7) << 4). Identity when
  // Swizzle128 is false.
  if constexpr (Swizzle128) {
    col ^= (row_idx & 7) << 4;
  }
  return *reinterpret_cast<const uint32_t*>(row + col);
}

template <bool Swizzle128, bool Pipe = false, bool LdmA = false>
__global__ void fc1_nvfp4_scaled_tma128n128k4_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // Kitchen tile: 128x128x256, 8 warps, 2-stage, ONE N operand.
  // cublasLt name: 128x128x256_1x1x1 s16864. Pairing both arms
  // doubles B and does not fit; this is the single-N geometry.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTmaKitN;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTmaK4M;
  __shared__ __align__(128) uint8_t smem_a[kTmaK4Stages][kTmaK4M][kTmaK4Packed];
  __shared__ __align__(128) uint8_t smem_b[kTmaK4Stages][kTmaKitN][kTmaK4Packed];
  __shared__ __align__(16) uint8_t slab_sa[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sb[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar[kTmaK4Stages];

  if (threadIdx.x == 0) {
    mbar_init(&mbar[0], 1);
    mbar_init(&mbar[1], 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar[buf], kTmaKitBytes);
      tma_load_2d(
          &smem_a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar[buf]);
      tma_load_2d(
          &smem_b[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
#pragma unroll
      for (int kg = 0; kg < kTmaK4Kg; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &slab_sa[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sb[buf][kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
      }
    }
  };

  float acc[kTmaKitNSub][4];
#pragma unroll
  for (int ni = 0; ni < kTmaKitNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[ni][i] = 0.0f;
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK4;
  issue_tma(0, 0);
  mbar_wait_parity(&mbar[0], 0);
  __syncthreads();

  int buf = 0;
  int phase[kTmaK4Stages] = {1, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK4, buf ^ 1);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < kTmaK4Kg; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
      const int a_row0 = warp * kNvfp4TileM + group;
      uint32_t a0, a1, a2, a3;
      if constexpr (LdmA) {
        const int ar = warp * kNvfp4TileM + (lane & 15);
        const int ac = ks * (kNvfp4TileK / 2) + (lane >> 4) * 16;
        ldmatrix_x4(a0, a1, a2, a3, &smem_a[buf][ar][ac]);
      } else {
        a0 = ld_tma128_u32<Swizzle128>(
            &smem_a[buf][a_row0][0], a_row0, packed_k0);
        a1 = ld_tma128_u32<Swizzle128>(
            &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0);
        a2 = ld_tma128_u32<Swizzle128>(
            &smem_a[buf][a_row0][0], a_row0, packed_k0 + 16);
        a3 = ld_tma128_u32<Swizzle128>(
            &smem_a[buf][a_row0 + 8][0], a_row0 + 8, packed_k0 + 16);
      }
      const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
      const uint32_t sfa = scale_from_slab(slab_sa[buf][ks], sfa_row);
      if constexpr (Pipe) {
        int b_row = group;
        uint32_t b0 = ld_tma128_u32<Swizzle128>(
            &smem_b[buf][b_row][0], b_row, packed_k0);
        uint32_t b1 = ld_tma128_u32<Swizzle128>(
            &smem_b[buf][b_row][0], b_row, packed_k0 + 16);
        uint32_t sfb = scale_from_slab(
            slab_sb[buf][ks], (tile_n & (kTmaM - 1)) + b_row);
#pragma unroll
        for (int ni = 0; ni < kTmaKitNSub; ++ni) {
          uint32_t nb0 = 0, nb1 = 0, nsfb = 0;
          if (ni + 1 < kTmaKitNSub) {
            const int nb_row = (ni + 1) * kNvfp4TileN + group;
            nb0 = ld_tma128_u32<Swizzle128>(
                &smem_b[buf][nb_row][0], nb_row, packed_k0);
            nb1 = ld_tma128_u32<Swizzle128>(
                &smem_b[buf][nb_row][0], nb_row, packed_k0 + 16);
            nsfb = scale_from_slab(
                slab_sb[buf][ks], (tile_n & (kTmaM - 1)) + nb_row);
          }
          nvfp4_mma_m16n8k64(
              acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
              a0, a1, a2, a3, b0, b1,
              acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
              sfa, sfb);
          if (ni + 1 < kTmaKitNSub) {
            b0 = nb0;
            b1 = nb1;
            sfb = nsfb;
          }
        }
      } else {
#pragma unroll
        for (int ni = 0; ni < kTmaKitNSub; ++ni) {
          const int b_row = ni * kNvfp4TileN + group;
          const uint32_t b0 = ld_tma128_u32<Swizzle128>(
              &smem_b[buf][b_row][0], b_row, packed_k0);
          const uint32_t b1 = ld_tma128_u32<Swizzle128>(
              &smem_b[buf][b_row][0], b_row, packed_k0 + 16);
          const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb = scale_from_slab(slab_sb[buf][ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
              a0, a1, a2, a3, b0, b1,
              acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
              sfa, sfb);
        }
      }
    }

    if (next < kTilesK) {
      const int nbuf = buf ^ 1;
      mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
    buf ^= 1;
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kTmaKitNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kRawWidth) {
        continue;
      }
      output[m * kRawWidth + n] =
          __float2bfloat16_rn(acc[ni][i] * alpha);
    }
  }
}

// Kitchen launch: 384 threads (4 producer + 8 MMA), 88064 B dynamic.
// Tile and atom match tma128n128k4 (already byte-exact). Producers
// split A / B / SFA / SFB so four TMA issues run concurrently.
constexpr int kKitWsThreads = 384;
constexpr int kKitWsProdWarps = 4;
constexpr int kKitWsMmaWarps = 8;
constexpr int kKitWsDynSmem = 88064;
constexpr unsigned kKitWsABytes =
    static_cast<unsigned>(kTmaK4M * kTmaK4Packed);
constexpr unsigned kKitWsBBytes =
    static_cast<unsigned>(kTmaKitN * kTmaK4Packed);
constexpr unsigned kKitWsSBytes =
    static_cast<unsigned>(kTmaK4Kg * kScaleSlabBytes);

struct alignas(128) KitWsSmem {
  uint8_t a[kTmaK4Stages][kTmaK4M][kTmaK4Packed];
  uint8_t b[kTmaK4Stages][kTmaKitN][kTmaK4Packed];
  uint8_t sa[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  uint8_t sb[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  uint64_t full[kTmaK4Stages];
  uint64_t empty[kTmaK4Stages];
};

static_assert(sizeof(KitWsSmem) <= kKitWsDynSmem,
              "12-warp kitchen tile must fit kitchen's 86 KiB dynamic");

__global__ void __launch_bounds__(kKitWsThreads, 1)
fc1_nvfp4_scaled_tma128n128k4ws_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  extern __shared__ __align__(128) char kit_ws_raw[];
  auto* sm = reinterpret_cast<KitWsSmem*>(kit_ws_raw);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTmaKitN;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTmaK4M;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTmaK4Stages; ++s) {
      mbar_init(&sm->full[s], kKitWsProdWarps);
      mbar_init(&sm->empty[s], kKitWsMmaWarps);
    }
  }
  __syncthreads();

  constexpr int kTilesK = kFc1K / kTmaK4;
  if (warp < kKitWsProdWarps) {
    if (lane == 0) {
      int empty_phase[kTmaK4Stages] = {0, 0};
      for (int ki = 0; ki < kTilesK; ++ki) {
        const int buf = ki & 1;
        if (ki >= kTmaK4Stages) {
          mbar_wait_parity(&sm->empty[buf], empty_phase[buf]);
          empty_phase[buf] ^= 1;
        }
        const int k0 = ki * kTmaK4;
        const int packed_k = k0 / 2;
        const int kcol0 = k0 / kFp4BlockSize;
        if (warp == 0) {
          mbar_arrive_expect_tx(&sm->full[buf], kKitWsABytes);
          tma_load_2d(
              &sm->a[buf][0][0], &a_map, packed_k,
              static_cast<int>(tile_m), &sm->full[buf]);
        } else if (warp == 1) {
          mbar_arrive_expect_tx(&sm->full[buf], kKitWsBBytes);
          tma_load_2d(
              &sm->b[buf][0][0], &w_map, packed_k, tile_n,
              &sm->full[buf]);
        } else if (warp == 2) {
          mbar_arrive_expect_tx(&sm->full[buf], kKitWsSBytes);
          cp_async_bulk(
              &sm->sa[buf][0][0],
              scale_slab_ptr(
                  a_scales, static_cast<int>(tile_m), kcol0,
                  a_scale_cols),
              kKitWsSBytes, &sm->full[buf]);
        } else {
          mbar_arrive_expect_tx(&sm->full[buf], kKitWsSBytes);
          cp_async_bulk(
              &sm->sb[buf][0][0],
              scale_slab_ptr(w_scales, tile_n, kcol0, w_scale_cols),
              kKitWsSBytes, &sm->full[buf]);
        }
      }
    }
    return;
  }

  const int mma_warp = warp - kKitWsProdWarps;
  const int group = lane >> 2;
  const int tidg = lane & 3;
  float acc[kTmaKitNSub][4];
#pragma unroll
  for (int ni = 0; ni < kTmaKitNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[ni][i] = 0.0f;
    }
  }

  mbar_wait_parity(&sm->full[0], 0);
  int full_phase[kTmaK4Stages] = {1, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int buf = ki & 1;
#pragma unroll
    for (int ks = 0; ks < kTmaK4Kg; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
      const int a_row0 = mma_warp * kNvfp4TileM + group;
      const uint32_t a0 =
          *reinterpret_cast<const uint32_t*>(
              &sm->a[buf][a_row0][packed_k0]);
      const uint32_t a1 =
          *reinterpret_cast<const uint32_t*>(
              &sm->a[buf][a_row0 + 8][packed_k0]);
      const uint32_t a2 =
          *reinterpret_cast<const uint32_t*>(
              &sm->a[buf][a_row0][packed_k0 + 16]);
      const uint32_t a3 =
          *reinterpret_cast<const uint32_t*>(
              &sm->a[buf][a_row0 + 8][packed_k0 + 16]);
      const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
      const uint32_t sfa = scale_from_slab(sm->sa[buf][ks], sfa_row);
#pragma unroll
      for (int ni = 0; ni < kTmaKitNSub; ++ni) {
        const int b_row = ni * kNvfp4TileN + group;
        const uint32_t b0 =
            *reinterpret_cast<const uint32_t*>(
                &sm->b[buf][b_row][packed_k0]);
        const uint32_t b1 =
            *reinterpret_cast<const uint32_t*>(
                &sm->b[buf][b_row][packed_k0 + 16]);
        const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
        const uint32_t sfb = scale_from_slab(sm->sb[buf][ks], sfb_row);
        nvfp4_mma_m16n8k64(
            acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
            a0, a1, a2, a3, b0, b1,
            acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
            sfa, sfb);
      }
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(&sm->empty[buf]);
    }
    const int nxt = ki + 1;
    if (nxt < kTilesK) {
      const int nbuf = nxt & 1;
      mbar_wait_parity(&sm->full[nbuf], full_phase[nbuf]);
      full_phase[nbuf] ^= 1;
    }
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kTmaKitNSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m =
          tile_m + mma_warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kRawWidth) {
        continue;
      }
      output[m * kRawWidth + n] =
          __float2bfloat16_rn(acc[ni][i] * alpha);
    }
  }
}

constexpr int kTmaN128 = 128;
constexpr int kTmaN128Sub = kTmaN128 / kNvfp4TileN;
constexpr unsigned kTmaN128Bytes =
    static_cast<unsigned>(
        kTmaK4M * kTmaK2Packed + 2 * kTmaN128 * kTmaK2Packed
        + 6 * kScaleSlabBytes);

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_tma128k2n_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 128x128 tile, 8 warps, 3-stage TMA of K=128. Each A load feeds
  // 16 n-subtiles and both arms. Same PTX fragment as k2. Acc is
  // 16*2*4 floats — same register budget as 256x64 dual-strip.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTmaN128;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTmaK4M;
  __shared__ __align__(128) uint8_t smem_a[kTma3Stages][kTmaK4M][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTma3Stages][kTmaN128][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTma3Stages][kTmaN128][kTmaK2Packed];
  __shared__ __align__(16) uint8_t slab_sa[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTma3Stages][2][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar[kTma3Stages];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTma3Stages; ++s) {
      mbar_init(&mbar[s], 1);
    }
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar[buf], kTmaN128Bytes);
      tma_load_2d(
          &smem_a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar[buf]);
      tma_load_2d(
          &smem_bg[buf][0][0], &w_map, packed_k, tile_n, &mbar[buf]);
      tma_load_2d(
          &smem_bu[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar[buf]);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &slab_sa[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sbg[buf][kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
        cp_async_bulk(
            &slab_sbu[buf][kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &mbar[buf]);
      }
    }
  };

  float acc_g[kTmaN128Sub][4];
  float acc_u[kTmaN128Sub][4];
#pragma unroll
  for (int ni = 0; ni < kTmaN128Sub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc_g[ni][i] = 0.0f;
      acc_u[ni][i] = 0.0f;
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK2;
  issue_tma(0, 0);
  if (kTilesK > 1) {
    issue_tma(kTmaK2, 1);
  }
  mbar_wait_parity(&mbar[0], 0);
  __syncthreads();

  int phase[kTma3Stages] = {1, 0, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int buf = ki % kTma3Stages;
    const int ahead = ki + 2;
    if (ahead < kTilesK) {
      issue_tma(ahead * kTmaK2, ahead % kTma3Stages);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
      const int a_row0 = warp * kNvfp4TileM + group;
      const uint32_t a0 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0][packed_k0]);
      const uint32_t a1 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0 + 8][packed_k0]);
      const uint32_t a2 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0][packed_k0 + 16]);
      const uint32_t a3 =
          *reinterpret_cast<const uint32_t*>(
              &smem_a[buf][a_row0 + 8][packed_k0 + 16]);
      const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
      const uint32_t sfa = scale_from_slab(slab_sa[buf][ks], sfa_row);
#pragma unroll
      for (int ni = 0; ni < kTmaN128Sub; ++ni) {
        const int b_row = ni * kNvfp4TileN + group;
        const uint32_t bg0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bg[buf][b_row][packed_k0]);
        const uint32_t bg1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bg[buf][b_row][packed_k0 + 16]);
        const uint32_t bu0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bu[buf][b_row][packed_k0]);
        const uint32_t bu1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_bu[buf][b_row][packed_k0 + 16]);
        const uint32_t sfb_g = scale_from_slab(slab_sbg[buf][ks], b_row);
        const uint32_t sfb_u = scale_from_slab(slab_sbu[buf][ks], b_row);
        nvfp4_mma_m16n8k64(
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            a0, a1, a2, a3, bg0, bg1,
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            sfa, sfb_g);
        nvfp4_mma_m16n8k64(
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            a0, a1, a2, a3, bu0, bu1,
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            sfa, sfb_u);
      }
    }

    const int nxt = ki + 1;
    if (nxt < kTilesK) {
      const int nbuf = nxt % kTma3Stages;
      mbar_wait_parity(&mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kTmaN128Sub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kActivationWidth) {
        continue;
      }
      const __nv_bfloat16 g = __float2bfloat16_rn(acc_g[ni][i] * alpha);
      const __nv_bfloat16 u = __float2bfloat16_rn(acc_u[ni][i] * alpha);
      if constexpr (StoreProduct) {
        output[m * kActivationWidth + n] = __float2bfloat16_rn(
            swiglu_like_eager_bf16(g, u));
      } else {
        output[m * kRawWidth + n] = g;
        output[m * kRawWidth + kActivationWidth + n] = u;
      }
    }
  }
}

constexpr unsigned kTma256K4Bytes =
    static_cast<unsigned>(
        kTma256M * kTmaK4Packed + 2 * kTma256N * kTmaK4Packed
        + (2 + 2) * kTmaK4Kg * kScaleSlabBytes);

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_tma256k4_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 256x64 tile, 8 warps, 1-stage TMA of K=256 (128 packed bytes).
  // One load feeds four m16n8k64 atoms. Four 128x4 scale slabs per
  // operand cover the 16 K-scale columns. Same PTX fragment and
  // 2x8x4 register acc as k2. 2-stage at this box is 112 KiB and
  // exceeds the 99 KiB static smem cap (0x18c00).
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ __align__(128) uint8_t smem_a[kTma256M][kTmaK4Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTma256N][kTmaK4Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTma256N][kTmaK4Packed];
  __shared__ __align__(16) uint8_t slab_sa0[kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sa1[kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[kTmaK4Kg][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar;

  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar, kTma256K4Bytes);
      tma_load_2d(
          &smem_a[0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar);
      tma_load_2d(
          &smem_bg[0][0], &w_map, packed_k, tile_n, &mbar);
      tma_load_2d(
          &smem_bu[0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar);
#pragma unroll
      for (int kg = 0; kg < kTmaK4Kg; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &slab_sa0[kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &mbar);
        cp_async_bulk(
            &slab_sa1[kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                a_scale_cols),
            kScaleSlabBytes, &mbar);
        cp_async_bulk(
            &slab_sbg[kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &mbar);
        cp_async_bulk(
            &slab_sbu[kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &mbar);
      }
    }
  };

  float acc_g[2][kTma256NSub][4];
  float acc_u[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK4;
  issue_tma(0);
  mbar_wait_parity(&mbar, 0);
  __syncthreads();

  int phase = 1;
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < kTmaK4Kg; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0][packed_k0]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0 + 8][packed_k0]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0][packed_k0 + 16]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0 + 8][packed_k0 + 16]);
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        const uint32_t sfa = scale_from_slab(
            sfa_row < kTmaM ? slab_sa0[ks] : slab_sa1[ks],
            sfa_row & (kTmaM - 1));
#pragma unroll
        for (int ni = 0; ni < kTma256NSub; ++ni) {
          const int b_row = ni * kNvfp4TileN + group;
          const uint32_t bg0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[b_row][packed_k0]);
          const uint32_t bg1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[b_row][packed_k0 + 16]);
          const uint32_t bu0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[b_row][packed_k0]);
          const uint32_t bu1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[b_row][packed_k0 + 16]);
          const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb_g = scale_from_slab(slab_sbg[ks], sfb_row);
          const uint32_t sfb_u = scale_from_slab(slab_sbu[ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              a0, a1, a2, a3, bg0, bg1,
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              sfa, sfb_g);
          nvfp4_mma_m16n8k64(
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              a0, a1, a2, a3, bu0, bu1,
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              sfa, sfb_u);
        }
      }
    }

    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK4);
      mbar_wait_parity(&mbar, phase);
      phase ^= 1;
      __syncthreads();
    }
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        const __nv_bfloat16 g =
            __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
        const __nv_bfloat16 u =
            __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
        if constexpr (StoreProduct) {
          output[m * kActivationWidth + n] = __float2bfloat16_rn(
              swiglu_like_eager_bf16(g, u));
        } else {
          output[m * kRawWidth + n] = g;
          output[m * kRawWidth + kActivationWidth + n] = u;
        }
      }
    }
  }
}

// 1-stage k2: same 256x64 K=128 atom, no K pipeline. Smem ~29 KiB
// so 2 CTAs/SM is legal if the register file allows.
constexpr unsigned kTma256K2s1Bytes =
    static_cast<unsigned>(
        kTma256M * kTmaK2Packed + 2 * kTma256N * kTmaK2Packed
        + 8 * kScaleSlabBytes);

__global__ void fc1_paired_nvfp4_scaled_tma256k2s1_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ __align__(128) uint8_t smem_a[kTma256M][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bg[kTma256N][kTmaK2Packed];
  __shared__ __align__(128) uint8_t smem_bu[kTma256N][kTmaK2Packed];
  __shared__ __align__(16) uint8_t slab_sa0[2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sa1[2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbg[2][kScaleSlabBytes];
  __shared__ __align__(16) uint8_t slab_sbu[2][kScaleSlabBytes];
  __shared__ __align__(8) uint64_t mbar;

  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&mbar, kTma256K2s1Bytes);
      tma_load_2d(
          &smem_a[0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &mbar);
      tma_load_2d(
          &smem_bg[0][0], &w_map, packed_k, tile_n, &mbar);
      tma_load_2d(
          &smem_bu[0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &mbar);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &slab_sa0[kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &mbar);
        cp_async_bulk(
            &slab_sa1[kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                a_scale_cols),
            kScaleSlabBytes, &mbar);
        cp_async_bulk(
            &slab_sbg[kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &mbar);
        cp_async_bulk(
            &slab_sbu[kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &mbar);
      }
    }
  };

  float acc_g[2][kTma256NSub][4];
  float acc_u[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK2;
  issue_tma(0);
  mbar_wait_parity(&mbar, 0);
  __syncthreads();

  int phase = 1;
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0][packed_k0]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0 + 8][packed_k0]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0][packed_k0 + 16]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(
                &smem_a[a_row0 + 8][packed_k0 + 16]);
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        const uint32_t sfa = scale_from_slab(
            sfa_row < kTmaM ? slab_sa0[ks] : slab_sa1[ks],
            sfa_row & (kTmaM - 1));
#pragma unroll
        for (int ni = 0; ni < kTma256NSub; ++ni) {
          const int b_row = ni * kNvfp4TileN + group;
          const uint32_t bg0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[b_row][packed_k0]);
          const uint32_t bg1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bg[b_row][packed_k0 + 16]);
          const uint32_t bu0 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[b_row][packed_k0]);
          const uint32_t bu1 =
              *reinterpret_cast<const uint32_t*>(
                  &smem_bu[b_row][packed_k0 + 16]);
          const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb_g = scale_from_slab(slab_sbg[ks], sfb_row);
          const uint32_t sfb_u = scale_from_slab(slab_sbu[ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              a0, a1, a2, a3, bg0, bg1,
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              sfa, sfb_g);
          nvfp4_mma_m16n8k64(
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              a0, a1, a2, a3, bu0, bu1,
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              sfa, sfb_u);
        }
      }
    }

    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK2);
      mbar_wait_parity(&mbar, phase);
      phase ^= 1;
      __syncthreads();
    }
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        output[m * kRawWidth + n] =
            __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
        output[m * kRawWidth + kActivationWidth + n] =
            __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
      }
    }
  }
}

// Paired 256x64 K=256 2-stage is 112 KiB. Dropping the second B
// operand (kitchen single-N) makes 2-stage legal: 94 KiB.
constexpr unsigned kTma256K4n1Bytes =
    static_cast<unsigned>(
        kTma256M * kTmaK4Packed + kTma256N * kTmaK4Packed
        + 3 * kTmaK4Kg * kScaleSlabBytes);

struct alignas(128) Tma256K4n1Smem {
  uint8_t a[kTmaK4Stages][kTma256M][kTmaK4Packed];
  uint8_t b[kTmaK4Stages][kTma256N][kTmaK4Packed];
  uint8_t sa0[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  uint8_t sa1[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  uint8_t sb[kTmaK4Stages][kTmaK4Kg][kScaleSlabBytes];
  uint64_t mbar[kTmaK4Stages];
};

// 128x64 K=128 2-stage paired. Legal TMA boxes (128 M, 64 N).
// Acc is 8x4x2 = 64 floats. Smem ~39 KiB so 2 CTAs/SM if
// regs stay under 128.
constexpr unsigned kTma128K2Bytes =
    static_cast<unsigned>(
        kTmaK4M * kTmaK2Packed + 2 * kTma256N * kTmaK2Packed
        + 6 * kScaleSlabBytes);

struct alignas(128) Tma128K2Smem {
  uint8_t a[kTmaK4Stages][kTmaK4M][kTmaK2Packed];
  uint8_t bg[kTmaK4Stages][kTma256N][kTmaK2Packed];
  uint8_t bu[kTmaK4Stages][kTma256N][kTmaK2Packed];
  uint8_t sa[kTmaK4Stages][2][kScaleSlabBytes];
  uint8_t sbg[kTmaK4Stages][2][kScaleSlabBytes];
  uint8_t sbu[kTmaK4Stages][2][kScaleSlabBytes];
  uint64_t mbar[kTmaK4Stages];
};

static_assert(sizeof(Tma128K2Smem) * 2 <= 101376,
              "two 128x64 K=128 2-stage CTAs must fit GB10 99 KiB");

__global__ void fc1_paired_nvfp4_scaled_tma128k2_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTmaK4M;
  __shared__ Tma128K2Smem sm;

  if (threadIdx.x == 0) {
    mbar_init(&sm.mbar[0], 1);
    mbar_init(&sm.mbar[1], 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&sm.mbar[buf], kTma128K2Bytes);
      tma_load_2d(
          &sm.a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &sm.mbar[buf]);
      tma_load_2d(
          &sm.bg[buf][0][0], &w_map, packed_k, tile_n, &sm.mbar[buf]);
      tma_load_2d(
          &sm.bu[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &sm.mbar[buf]);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &sm.sa[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &sm.mbar[buf]);
        cp_async_bulk(
            &sm.sbg[buf][kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &sm.mbar[buf]);
        cp_async_bulk(
            &sm.sbu[buf][kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &sm.mbar[buf]);
      }
    }
  };

  float acc_g[kTma256NSub][4];
  float acc_u[kTma256NSub][4];
#pragma unroll
  for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc_g[ni][i] = 0.0f;
      acc_u[ni][i] = 0.0f;
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK2;
  issue_tma(0, 0);
  mbar_wait_parity(&sm.mbar[0], 0);
  __syncthreads();

  int buf = 0;
  int phase[kTmaK4Stages] = {1, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK2, buf ^ 1);
    }
    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
      const int a_row0 = warp * kNvfp4TileM + group;
      const uint32_t a0 =
          *reinterpret_cast<const uint32_t*>(
              &sm.a[buf][a_row0][packed_k0]);
      const uint32_t a1 =
          *reinterpret_cast<const uint32_t*>(
              &sm.a[buf][a_row0 + 8][packed_k0]);
      const uint32_t a2 =
          *reinterpret_cast<const uint32_t*>(
              &sm.a[buf][a_row0][packed_k0 + 16]);
      const uint32_t a3 =
          *reinterpret_cast<const uint32_t*>(
              &sm.a[buf][a_row0 + 8][packed_k0 + 16]);
      const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
      const uint32_t sfa = scale_from_slab(sm.sa[buf][ks], sfa_row);
#pragma unroll
      for (int ni = 0; ni < kTma256NSub; ++ni) {
        const int b_row = ni * kNvfp4TileN + group;
        const uint32_t bg0 =
            *reinterpret_cast<const uint32_t*>(
                &sm.bg[buf][b_row][packed_k0]);
        const uint32_t bg1 =
            *reinterpret_cast<const uint32_t*>(
                &sm.bg[buf][b_row][packed_k0 + 16]);
        const uint32_t bu0 =
            *reinterpret_cast<const uint32_t*>(
                &sm.bu[buf][b_row][packed_k0]);
        const uint32_t bu1 =
            *reinterpret_cast<const uint32_t*>(
                &sm.bu[buf][b_row][packed_k0 + 16]);
        const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
        const uint32_t sfb_g = scale_from_slab(sm.sbg[buf][ks], sfb_row);
        const uint32_t sfb_u = scale_from_slab(sm.sbu[buf][ks], sfb_row);
        nvfp4_mma_m16n8k64(
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            a0, a1, a2, a3, bg0, bg1,
            acc_g[ni][0], acc_g[ni][1], acc_g[ni][2], acc_g[ni][3],
            sfa, sfb_g);
        nvfp4_mma_m16n8k64(
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            a0, a1, a2, a3, bu0, bu1,
            acc_u[ni][0], acc_u[ni][1], acc_u[ni][2], acc_u[ni][3],
            sfa, sfb_u);
      }
    }
    if (next < kTilesK) {
      const int nbuf = buf ^ 1;
      mbar_wait_parity(&sm.mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
    buf ^= 1;
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t m = tile_m + warp * kNvfp4TileM + r0 + r_off[i];
      const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
      if (m >= rows || n >= kActivationWidth) {
        continue;
      }
      output[m * kRawWidth + n] =
          __float2bfloat16_rn(acc_g[ni][i] * alpha);
      output[m * kRawWidth + kActivationWidth + n] =
          __float2bfloat16_rn(acc_u[ni][i] * alpha);
    }
  }
}

// Half-N k2: compute 32 N (64-float acc). TMA B box is 64
// (SM120 2D tile faults at N=32). 1-stage so two CTAs fit.
constexpr int kTma32N = 32;
constexpr int kTma32NSub = kTma32N / kNvfp4TileN;
constexpr int kTma32TmaN = 64;
constexpr unsigned kTma256N32Bytes =
    static_cast<unsigned>(
        kTma256M * kTmaK2Packed + 2 * kTma32TmaN * kTmaK2Packed
        + 8 * kScaleSlabBytes);

struct alignas(128) Tma256N32Smem {
  uint8_t a[kTma256M][kTmaK2Packed];
  uint8_t bg[kTma32TmaN][kTmaK2Packed];
  uint8_t bu[kTma32TmaN][kTmaK2Packed];
  uint8_t sa0[2][kScaleSlabBytes];
  uint8_t sa1[2][kScaleSlabBytes];
  uint8_t sbg[2][kScaleSlabBytes];
  uint8_t sbu[2][kScaleSlabBytes];
  uint64_t mbar;
};

static_assert(sizeof(Tma256N32Smem) * 2 <= 101376,
              "two 256x32 K=128 1-stage CTAs must fit GB10 99 KiB");

__global__ void fc1_paired_nvfp4_scaled_tma256n32_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma32N;
  const int tma_n = tile_n & ~(kTma32TmaN - 1);
  const int smem_n0 = tile_n - tma_n;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ Tma256N32Smem sm;

  if (threadIdx.x == 0) {
    mbar_init(&sm.mbar, 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&sm.mbar, kTma256N32Bytes);
      tma_load_2d(
          &sm.a[0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &sm.mbar);
      tma_load_2d(
          &sm.bg[0][0], &w_map, packed_k, tma_n, &sm.mbar);
      tma_load_2d(
          &sm.bu[0][0], &w_map, packed_k,
          tma_n + kActivationWidth, &sm.mbar);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &sm.sa0[kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &sm.mbar);
        cp_async_bulk(
            &sm.sa1[kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                a_scale_cols),
            kScaleSlabBytes, &sm.mbar);
        cp_async_bulk(
            &sm.sbg[kg][0],
            scale_slab_ptr(w_scales, tma_n, kc, w_scale_cols),
            kScaleSlabBytes, &sm.mbar);
        cp_async_bulk(
            &sm.sbu[kg][0],
            scale_slab_ptr(
                w_scales, tma_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &sm.mbar);
      }
    }
  };

  float acc_g[2][kTma32NSub][4];
  float acc_u[2][kTma32NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma32NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK2;
  issue_tma(0);
  mbar_wait_parity(&sm.mbar, 0);
  __syncthreads();

  int phase = 1;
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[a_row0][packed_k0]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[a_row0 + 8][packed_k0]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[a_row0][packed_k0 + 16]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[a_row0 + 8][packed_k0 + 16]);
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        const uint32_t sfa = scale_from_slab(
            sfa_row < kTmaM ? sm.sa0[ks] : sm.sa1[ks],
            sfa_row & (kTmaM - 1));
#pragma unroll
        for (int ni = 0; ni < kTma32NSub; ++ni) {
          const int b_row = smem_n0 + ni * kNvfp4TileN + group;
          const uint32_t bg0 =
              *reinterpret_cast<const uint32_t*>(
                  &sm.bg[b_row][packed_k0]);
          const uint32_t bg1 =
              *reinterpret_cast<const uint32_t*>(
                  &sm.bg[b_row][packed_k0 + 16]);
          const uint32_t bu0 =
              *reinterpret_cast<const uint32_t*>(
                  &sm.bu[b_row][packed_k0]);
          const uint32_t bu1 =
              *reinterpret_cast<const uint32_t*>(
                  &sm.bu[b_row][packed_k0 + 16]);
          const int sfb_row = (tma_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb_g =
              scale_from_slab(sm.sbg[ks], sfb_row);
          const uint32_t sfb_u =
              scale_from_slab(sm.sbu[ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              a0, a1, a2, a3, bg0, bg1,
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              sfa, sfb_g);
          nvfp4_mma_m16n8k64(
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              a0, a1, a2, a3, bu0, bu1,
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              sfa, sfb_u);
        }
      }
    }

    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK2);
      mbar_wait_parity(&sm.mbar, phase);
      phase ^= 1;
      __syncthreads();
    }
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma32NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kActivationWidth) {
          continue;
        }
        output[m * kRawWidth + n] =
            __float2bfloat16_rn(acc_g[ms][ni][i] * alpha);
        output[m * kRawWidth + kActivationWidth + n] =
            __float2bfloat16_rn(acc_u[ms][ni][i] * alpha);
      }
    }
  }
}

static_assert(sizeof(Tma256K4n1Smem) <= 101376,
              "256x64 K=256 2-stage single-N must fit GB10 99 KiB");

__global__ void fc1_nvfp4_scaled_tma256k4n1_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 256x64, 8 warps, 2-stage TMA of K=256, ONE N operand.
  // Four m16n8k64 atoms per box. Same 2x8x4 acc half as k2.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTma256N;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;
  __shared__ Tma256K4n1Smem sm;

  if (threadIdx.x == 0) {
    mbar_init(&sm.mbar[0], 1);
    mbar_init(&sm.mbar[1], 1);
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&sm.mbar[buf], kTma256K4n1Bytes);
      tma_load_2d(
          &sm.a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &sm.mbar[buf]);
      tma_load_2d(
          &sm.b[buf][0][0], &w_map, packed_k, tile_n, &sm.mbar[buf]);
#pragma unroll
      for (int kg = 0; kg < kTmaK4Kg; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &sm.sa0[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &sm.mbar[buf]);
        cp_async_bulk(
            &sm.sa1[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                a_scale_cols),
            kScaleSlabBytes, &sm.mbar[buf]);
        cp_async_bulk(
            &sm.sb[buf][kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &sm.mbar[buf]);
      }
    }
  };

  float acc[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[ms][ni][i] = 0.0f;
      }
    }
  }

  constexpr int kTilesK = kFc1K / kTmaK4;
  issue_tma(0, 0);
  mbar_wait_parity(&sm.mbar[0], 0);
  __syncthreads();

  int buf = 0;
  int phase[kTmaK4Stages] = {1, 0};
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK4, buf ^ 1);
    }

    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < kTmaK4Kg; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[buf][a_row0][packed_k0]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[buf][a_row0 + 8][packed_k0]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[buf][a_row0][packed_k0 + 16]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(
                &sm.a[buf][a_row0 + 8][packed_k0 + 16]);
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        const uint32_t sfa = scale_from_slab(
            sfa_row < kTmaM ? sm.sa0[buf][ks] : sm.sa1[buf][ks],
            sfa_row & (kTmaM - 1));
#pragma unroll
        for (int ni = 0; ni < kTma256NSub; ++ni) {
          const int b_row = ni * kNvfp4TileN + group;
          const uint32_t b0 =
              *reinterpret_cast<const uint32_t*>(
                  &sm.b[buf][b_row][packed_k0]);
          const uint32_t b1 =
              *reinterpret_cast<const uint32_t*>(
                  &sm.b[buf][b_row][packed_k0 + 16]);
          const int sfb_row = (tile_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb = scale_from_slab(sm.sb[buf][ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc[ms][ni][0], acc[ms][ni][1], acc[ms][ni][2],
              acc[ms][ni][3],
              a0, a1, a2, a3, b0, b1,
              acc[ms][ni][0], acc[ms][ni][1], acc[ms][ni][2],
              acc[ms][ni][3],
              sfa, sfb);
        }
      }
    }

    if (next < kTilesK) {
      const int nbuf = buf ^ 1;
      mbar_wait_parity(&sm.mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
    buf ^= 1;
  }

  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t m =
            tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
        const int n = tile_n + ni * kNvfp4TileN + c0 + c_off[i];
        if (m >= rows || n >= kRawWidth) {
          continue;
        }
        output[m * kRawWidth + n] =
            __float2bfloat16_rn(acc[ms][ni][i] * alpha);
      }
    }
  }
}

constexpr int kTmaN2 = 128;
constexpr int kTmaN2Stages = 2;
constexpr int kTmaN2AccFloats = 2 * kTma256NSub * 4 * 2;

struct alignas(128) TmaN2Smem {
  uint8_t a[kTmaN2Stages][kTma256M][kTmaK2Packed];
  uint8_t bg0[kTmaN2Stages][kTma256N][kTmaK2Packed];
  uint8_t bu0[kTmaN2Stages][kTma256N][kTmaK2Packed];
  uint8_t bg1[kTmaN2Stages][kTma256N][kTmaK2Packed];
  uint8_t bu1[kTmaN2Stages][kTma256N][kTmaK2Packed];
  uint8_t sa0[kTmaN2Stages][2][kScaleSlabBytes];
  uint8_t sa1[kTmaN2Stages][2][kScaleSlabBytes];
  uint8_t sbg0[kTmaN2Stages][2][kScaleSlabBytes];
  uint8_t sbu0[kTmaN2Stages][2][kScaleSlabBytes];
  uint8_t sbg1[kTmaN2Stages][2][kScaleSlabBytes];
  uint8_t sbu1[kTmaN2Stages][2][kScaleSlabBytes];
  uint64_t mbar[kTmaN2Stages];
  float acc[kTma256Threads][kTmaN2AccFloats];
};

static_assert(sizeof(TmaN2Smem) <= 227328,
              "N-halves opt-in smem must fit GB10 block cap");

constexpr unsigned kTmaN2Bytes =
    static_cast<unsigned>(
        kTma256M * kTmaK2Packed + 4 * kTma256N * kTmaK2Packed
        + 12 * kScaleSlabBytes);

__device__ __forceinline__ void n2_swap_acc(
    float acc_g[2][kTma256NSub][4],
    float acc_u[2][kTma256NSub][4],
    float* spill) {
  float* p = spill;
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float t = *p;
        *p++ = acc_g[ms][ni][i];
        acc_g[ms][ni][i] = t;
      }
    }
  }
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float t = *p;
        *p++ = acc_u[ms][ni][i];
        acc_u[ms][ni][i] = t;
      }
    }
  }
}

template <bool StoreProduct>
__global__ void fc1_paired_nvfp4_scaled_tma256k2n2_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap w_map,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ w_scales,
    __nv_bfloat16* __restrict__ output,
    int64_t rows,
    int a_scale_cols,
    int w_scale_cols,
    float alpha) {
  // 256x128 logical tile, 8 warps, k2 atom. One A stream feeds two
  // sequential 64-wide N-halves. Acc0 stays in the 2x8x4 register
  // budget; acc1 lives in opt-in smem and swaps in for its MMA.
  extern __shared__ __align__(128) uint8_t dyn_n2[];
  auto* sm = reinterpret_cast<TmaN2Smem*>(dyn_n2);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x * kTmaN2;
  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kTma256M;

  if (threadIdx.x == 0) {
    mbar_init(&sm->mbar[0], 1);
    mbar_init(&sm->mbar[1], 1);
  }
  float* spill = sm->acc[threadIdx.x];
#pragma unroll
  for (int i = 0; i < kTmaN2AccFloats; ++i) {
    spill[i] = 0.0f;
  }
  __syncthreads();

  auto issue_tma = [&](int k0, int buf) {
    const int packed_k = k0 / 2;
    const int kcol0 = k0 / kFp4BlockSize;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(&sm->mbar[buf], kTmaN2Bytes);
      tma_load_2d(
          &sm->a[buf][0][0], &a_map, packed_k,
          static_cast<int>(tile_m), &sm->mbar[buf]);
      tma_load_2d(
          &sm->bg0[buf][0][0], &w_map, packed_k, tile_n, &sm->mbar[buf]);
      tma_load_2d(
          &sm->bu0[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth, &sm->mbar[buf]);
      tma_load_2d(
          &sm->bg1[buf][0][0], &w_map, packed_k,
          tile_n + kTma256N, &sm->mbar[buf]);
      tma_load_2d(
          &sm->bu1[buf][0][0], &w_map, packed_k,
          tile_n + kActivationWidth + kTma256N, &sm->mbar[buf]);
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        const int kc = kcol0 + kg * kSwizzleScaleGroup;
        cp_async_bulk(
            &sm->sa0[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m), kc, a_scale_cols),
            kScaleSlabBytes, &sm->mbar[buf]);
        cp_async_bulk(
            &sm->sa1[buf][kg][0],
            scale_slab_ptr(
                a_scales, static_cast<int>(tile_m) + kTmaM, kc,
                a_scale_cols),
            kScaleSlabBytes, &sm->mbar[buf]);
        cp_async_bulk(
            &sm->sbg0[buf][kg][0],
            scale_slab_ptr(w_scales, tile_n, kc, w_scale_cols),
            kScaleSlabBytes, &sm->mbar[buf]);
        cp_async_bulk(
            &sm->sbu0[buf][kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth, kc, w_scale_cols),
            kScaleSlabBytes, &sm->mbar[buf]);
        cp_async_bulk(
            &sm->sbg1[buf][kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kTma256N, kc, w_scale_cols),
            kScaleSlabBytes, &sm->mbar[buf]);
        cp_async_bulk(
            &sm->sbu1[buf][kg][0],
            scale_slab_ptr(
                w_scales, tile_n + kActivationWidth + kTma256N, kc,
                w_scale_cols),
            kScaleSlabBytes, &sm->mbar[buf]);
      }
    }
  };

  float acc_g[2][kTma256NSub][4];
  float acc_u[2][kTma256NSub][4];
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[ms][ni][i] = 0.0f;
        acc_u[ms][ni][i] = 0.0f;
      }
    }
  }

  auto mma_half = [&](int buf, const uint8_t bg[][kTmaK2Packed],
                      const uint8_t bu[][kTmaK2Packed],
                      const uint8_t slab_sbg[][kScaleSlabBytes],
                      const uint8_t slab_sbu[][kScaleSlabBytes],
                      int half_n) {
    const int group = lane >> 2;
    const int tidg = lane & 3;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int packed_k0 = ks * (kNvfp4TileK / 2) + tidg * 4;
#pragma unroll
      for (int ms = 0; ms < 2; ++ms) {
        const int a_row0 = ms * kTmaM + warp * kNvfp4TileM + group;
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(
                &sm->a[buf][a_row0][packed_k0]);
        const uint32_t a1 =
            *reinterpret_cast<const uint32_t*>(
                &sm->a[buf][a_row0 + 8][packed_k0]);
        const uint32_t a2 =
            *reinterpret_cast<const uint32_t*>(
                &sm->a[buf][a_row0][packed_k0 + 16]);
        const uint32_t a3 =
            *reinterpret_cast<const uint32_t*>(
                &sm->a[buf][a_row0 + 8][packed_k0 + 16]);
        const int sfa_row = (lane & 1) ? (a_row0 + 8) : a_row0;
        const uint32_t sfa = scale_from_slab(
            sfa_row < kTmaM ? sm->sa0[buf][ks] : sm->sa1[buf][ks],
            sfa_row & (kTmaM - 1));
#pragma unroll
        for (int ni = 0; ni < kTma256NSub; ++ni) {
          const int b_row = ni * kNvfp4TileN + group;
          const uint32_t bg0 =
              *reinterpret_cast<const uint32_t*>(&bg[b_row][packed_k0]);
          const uint32_t bg1 =
              *reinterpret_cast<const uint32_t*>(&bg[b_row][packed_k0 + 16]);
          const uint32_t bu0 =
              *reinterpret_cast<const uint32_t*>(&bu[b_row][packed_k0]);
          const uint32_t bu1 =
              *reinterpret_cast<const uint32_t*>(&bu[b_row][packed_k0 + 16]);
          const int sfb_row = (half_n & (kTmaM - 1)) + b_row;
          const uint32_t sfb_g = scale_from_slab(slab_sbg[ks], sfb_row);
          const uint32_t sfb_u = scale_from_slab(slab_sbu[ks], sfb_row);
          nvfp4_mma_m16n8k64(
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              a0, a1, a2, a3, bg0, bg1,
              acc_g[ms][ni][0], acc_g[ms][ni][1], acc_g[ms][ni][2],
              acc_g[ms][ni][3],
              sfa, sfb_g);
          nvfp4_mma_m16n8k64(
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              a0, a1, a2, a3, bu0, bu1,
              acc_u[ms][ni][0], acc_u[ms][ni][1], acc_u[ms][ni][2],
              acc_u[ms][ni][3],
              sfa, sfb_u);
        }
      }
    }
  };

  constexpr int kTilesK = kFc1K / kTmaK2;
  issue_tma(0, 0);
  mbar_wait_parity(&sm->mbar[0], 0);
  __syncthreads();

  int buf = 0;
  int phase[kTmaN2Stages] = {1, 0};
  bool n1_hot = false;
  for (int ki = 0; ki < kTilesK; ++ki) {
    const int next = ki + 1;
    if (next < kTilesK) {
      issue_tma(next * kTmaK2, buf ^ 1);
    }

    if (!n1_hot) {
      mma_half(buf, sm->bg0[buf], sm->bu0[buf], sm->sbg0[buf],
               sm->sbu0[buf], tile_n);
      n2_swap_acc(acc_g, acc_u, spill);
      mma_half(buf, sm->bg1[buf], sm->bu1[buf], sm->sbg1[buf],
               sm->sbu1[buf], tile_n + kTma256N);
      n1_hot = true;
    } else {
      mma_half(buf, sm->bg1[buf], sm->bu1[buf], sm->sbg1[buf],
               sm->sbu1[buf], tile_n + kTma256N);
      n2_swap_acc(acc_g, acc_u, spill);
      mma_half(buf, sm->bg0[buf], sm->bu0[buf], sm->sbg0[buf],
               sm->sbu0[buf], tile_n);
      n1_hot = false;
    }

    if (next < kTilesK) {
      const int nbuf = buf ^ 1;
      mbar_wait_parity(&sm->mbar[nbuf], phase[nbuf]);
      phase[nbuf] ^= 1;
    }
    __syncthreads();
    buf ^= 1;
  }

  // 42 K-tiles is even, so acc0 is in registers and acc1 is in spill.
  const int r0 = (lane / 4);
  const int c0 = (lane % 4) * 2;
  const int r_off[4] = {0, 0, 8, 8};
  const int c_off[4] = {0, 1, 0, 1};
  auto store_half = [&](const float g[][kTma256NSub][4],
                        const float u[][kTma256NSub][4], int n_base) {
#pragma unroll
    for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
      for (int ni = 0; ni < kTma256NSub; ++ni) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int64_t m =
              tile_m + ms * kTmaM + warp * kNvfp4TileM + r0 + r_off[i];
          const int n = n_base + ni * kNvfp4TileN + c0 + c_off[i];
          if (m >= rows || n >= kActivationWidth) {
            continue;
          }
          const __nv_bfloat16 gv =
              __float2bfloat16_rn(g[ms][ni][i] * alpha);
          const __nv_bfloat16 uv =
              __float2bfloat16_rn(u[ms][ni][i] * alpha);
          if constexpr (StoreProduct) {
            output[m * kActivationWidth + n] = __float2bfloat16_rn(
                swiglu_like_eager_bf16(gv, uv));
          } else {
            output[m * kRawWidth + n] = gv;
            output[m * kRawWidth + kActivationWidth + n] = uv;
          }
        }
      }
    }
  };
  store_half(acc_g, acc_u, tile_n);
  n2_swap_acc(acc_g, acc_u, spill);
  store_half(acc_g, acc_u, tile_n + kTma256N);
}

__global__ void swiglu_amax_swizzle_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t orig_rows) {
  // One CTA owns a 128x64 product tile: one cuBLAS E4M3 slab.
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kSwizzleTileR
      + threadIdx.x;
  const int col0 = blockIdx.x * kSwizzleTileC;
  uint32_t local_max = 0;
  if (row < orig_rows) {
    const __nv_bfloat16* gate = raw + row * kRawWidth + col0;
    const __nv_bfloat16* up = gate + kActivationWidth;
#pragma unroll
    for (int c = 0; c < kSwizzleTileC; c += kDirectValuesPerThread) {
      const uint4 gate_vec = *reinterpret_cast<const uint4*>(gate + c);
      const uint4 up_vec = *reinterpret_cast<const uint4*>(up + c);
      const __nv_bfloat16* gate_v =
          reinterpret_cast<const __nv_bfloat16*>(&gate_vec);
      const __nv_bfloat16* up_v =
          reinterpret_cast<const __nv_bfloat16*>(&up_vec);
#pragma unroll
      for (int i = 0; i < kDirectValuesPerThread; ++i) {
        const uint32_t abs_bits = swiglu_abs_bits(gate_v[i], up_v[i]);
        local_max = local_max < abs_bits ? abs_bits : local_max;
      }
    }
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
}

__global__ void swiglu_pack_swizzle_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  // Same 128x64 tile as amax. The 128x4 E4M3 slab is 512 consecutive
  // bytes in the cuBLAS swizzle, so the CTA writes one contiguous scale
  // region. Algebra matches pack-from-amax-bits.
  const int64_t row = static_cast<int64_t>(blockIdx.y) * kSwizzleTileR
      + threadIdx.x;
  const int col0 = blockIdx.x * kSwizzleTileC;
  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }
  if (row >= padded_rows) {
    return;
  }

  float values[kSwizzleTileC];
  if (row < orig_rows) {
    const __nv_bfloat16* gate = raw + row * kRawWidth + col0;
    const __nv_bfloat16* up = gate + kActivationWidth;
#pragma unroll
    for (int c = 0; c < kSwizzleTileC; c += kDirectValuesPerThread) {
      const uint4 gate_vec = *reinterpret_cast<const uint4*>(gate + c);
      const uint4 up_vec = *reinterpret_cast<const uint4*>(up + c);
      const __nv_bfloat16* gate_v =
          reinterpret_cast<const __nv_bfloat16*>(&gate_vec);
      const __nv_bfloat16* up_v =
          reinterpret_cast<const __nv_bfloat16*>(&up_vec);
#pragma unroll
      for (int i = 0; i < kDirectValuesPerThread; ++i) {
        values[c + i] = swiglu_like_eager_bf16(gate_v[i], up_v[i]);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < kSwizzleTileC; ++i) {
      values[i] = 0.0f;
    }
  }

#pragma unroll
  for (int blk = 0; blk < kSwizzleScaleGroup; ++blk) {
    const int base = blk * kFp4BlockSize;
    float absmax = fabsf(values[base]);
#pragma unroll
    for (int i = 1; i < kFp4BlockSize; ++i) {
      absmax = fmaxf(absmax, fabsf(values[base + i]));
    }
    float decode_scale = __fdividef(
        __fdividef(absmax, 6.0f), global_decode_scale);
    decode_scale = fminf(decode_scale, 448.0f);
    const __nv_fp8_e4m3 scale_fp8 =
        static_cast<__nv_fp8_e4m3>(decode_scale);
    const float decode_scale_fp8 = static_cast<float>(scale_fp8);
    const int64_t scale_col = (col0 / kFp4BlockSize) + blk;
    const size_t scale_offset = scale_factor_swizzled_offset(
        static_cast<size_t>(row),
        static_cast<size_t>(scale_col),
        kScaleCols);
    reinterpret_cast<__nv_fp8_e4m3*>(block_scales)[scale_offset] = scale_fp8;

    const float encode_scale = fminf(
        __fdividef(1.0f, decode_scale_fp8 * global_decode_scale), FLT_MAX);
    union PackedFp4 {
      uint64_t u64;
      __nv_fp4x2_storage_t fp4x2[8];
    } packed;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float a = values[base + 2 * i] * encode_scale;
      const float b = values[base + 2 * i + 1] * encode_scale;
      packed.fp4x2[i] = __nv_cvt_float2_to_fp4x2(
          float2{b, a}, __NV_E2M1, cudaRoundNearest);
    }
    *reinterpret_cast<uint64_t*>(
        output + row * (kActivationWidth / 2) + col0 / 2 + blk * 8) =
        packed.u64;
  }
}

__global__ void swiglu_nvfp4_dynamic_coop_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  // One launch: bound-filtered amax, grid.sync(), 8-wide pack from bits.
  // Same algebra as the two-kernel fused producer. No 558 MiB store.
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();

  uint32_t local_max = 0;
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
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
      if (swiglu_abs_bound_bits(gate_v[i], up_v[i]) <= local_max) {
        continue;
      }
      const uint32_t abs_bits = swiglu_abs_bits(gate_v[i], up_v[i]);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
  grid.sync();

  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

  const int64_t total_pack =
      padded_rows * groups_per_row;
  for (int64_t thread_linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       thread_linear < total_pack;
       thread_linear += stride) {
    const int64_t row = thread_linear / groups_per_row;
    const int col = static_cast<int>(
        (thread_linear - row * groups_per_row) * kDirectValuesPerThread);
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

    float decode_scale = __fdividef(
        __fdividef(absmax, 6.0f), global_decode_scale);
    decode_scale = fminf(decode_scale, 448.0f);
    const __nv_fp8_e4m3 scale_fp8 =
        static_cast<__nv_fp8_e4m3>(decode_scale);
    const float decode_scale_fp8 = static_cast<float>(scale_fp8);
    if ((threadIdx.x % kDirectThreadsPerScale) == 0) {
      const int64_t scale_col = col / kFp4BlockSize;
      const size_t scale_offset = scale_factor_swizzled_offset(
          static_cast<size_t>(row),
          static_cast<size_t>(scale_col),
          kScaleCols);
      reinterpret_cast<__nv_fp8_e4m3*>(block_scales)[scale_offset] =
          scale_fp8;
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
}

__device__ __forceinline__ void reduce_bound_winner(
    uint32_t& bound, unsigned& gate_bits, unsigned& up_bits) {
  constexpr unsigned int kFullMask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1) {
    const uint32_t other_bound = __shfl_down_sync(kFullMask, bound, offset);
    const unsigned other_gate = __shfl_down_sync(kFullMask, gate_bits, offset);
    const unsigned other_up = __shfl_down_sync(kFullMask, up_bits, offset);
    if (other_bound > bound) {
      bound = other_bound;
      gate_bits = other_gate;
      up_bits = other_up;
    }
  }
}

__global__ void swiglu_amax_bound_winner_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    uint32_t* __restrict__ global_bound_bits,
    int64_t orig_rows) {
  // Cheap pass: no SiLU. Each CTA keeps the (gate, up) pair with the
  // largest ru_bf16 bound, then writes the exact abs-bits of that pair.
  // The resulting L is a lower bound on the true amax. Exact identity:
  // bound_bits(x) >= exact_bits(x), so any x with bound_bits <= L cannot
  // raise the bit-max.
  uint32_t local_bound = 0;
  unsigned win_gate_bits = 0;
  unsigned win_up_bits = 0;
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
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
      const uint32_t bound = swiglu_abs_bound_bits(gate_v[i], up_v[i]);
      if (bound > local_bound) {
        local_bound = bound;
        win_gate_bits = static_cast<unsigned>(
            __bfloat16_as_ushort(gate_v[i]));
        win_up_bits = static_cast<unsigned>(
            __bfloat16_as_ushort(up_v[i]));
      }
    }
  }

  reduce_bound_winner(local_bound, win_gate_bits, win_up_bits);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __shared__ uint32_t warp_bound[8];
  __shared__ unsigned warp_gate[8];
  __shared__ unsigned warp_up[8];
  if (lane == 0) {
    warp_bound[warp] = local_bound;
    warp_gate[warp] = win_gate_bits;
    warp_up[warp] = win_up_bits;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = blockDim.x >> 5;
    local_bound = lane < warp_count ? warp_bound[lane] : 0;
    win_gate_bits = lane < warp_count ? warp_gate[lane] : 0;
    win_up_bits = lane < warp_count ? warp_up[lane] : 0;
    reduce_bound_winner(local_bound, win_gate_bits, win_up_bits);
    if (lane == 0 && local_bound != 0) {
      const __nv_bfloat16 gate = __ushort_as_bfloat16(
          static_cast<unsigned short>(win_gate_bits));
      const __nv_bfloat16 up = __ushort_as_bfloat16(
          static_cast<unsigned short>(win_up_bits));
      atomicMax(
          reinterpret_cast<unsigned int*>(global_max_bits),
          static_cast<unsigned int>(swiglu_abs_bits(gate, up)));
      if (global_bound_bits != nullptr) {
        atomicMax(
            reinterpret_cast<unsigned int*>(global_bound_bits),
            static_cast<unsigned int>(local_bound));
      }
    }
  }
}

__global__ void swiglu_amax_sparse_exact_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    unsigned long long* __restrict__ exact_evals,
    const uint32_t* __restrict__ global_bound_bits,
    int32_t* __restrict__ scanned,
    int64_t orig_rows) {
  // Second amax pass: exact SiLU only when bound_bits > L. Every skipped
  // lane has exact_bits <= bound_bits <= L and cannot change the max.
  // If bound bits U are supplied and G(L)==G(U), the stock scale is
  // already determined and this launch must not touch HBM.
  if (global_bound_bits != nullptr
      && dynamic_scale_from_bf16_bits(global_max_bits)
          == dynamic_scale_from_bf16_bits(global_bound_bits)) {
    return;
  }
  if (global_bound_bits != nullptr && scanned != nullptr
      && blockIdx.x == 0 && threadIdx.x == 0) {
    scanned[0] = 1;
  }
  const uint32_t lower = global_max_bits[0];
  uint32_t local_max = lower;
  unsigned long long local_exact = 0;
  const int64_t groups_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kDirectValuesPerThread);
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
      if (swiglu_abs_bound_bits(gate_v[i], up_v[i]) <= lower) {
        continue;
      }
      ++local_exact;
      const uint32_t abs_bits = swiglu_abs_bits(gate_v[i], up_v[i]);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
  if (exact_evals != nullptr && local_exact != 0) {
    atomicAdd(exact_evals, local_exact);
  }
}

__global__ void swiglu_nvfp4_pack_from_amax_bits_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  // Pack-only second pass: recompute SwiGLU, derive the stock BF16-rounded
  // global scale from supplied amax bits. No 558 MiB activation store.
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row =
      kActivationWidth / kDirectValuesPerThread;
  const int64_t total_threads = padded_rows * threads_per_row;
  if (thread_linear >= total_threads) {
    return;
  }

  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

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

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
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

constexpr int kVec16 = 16;

__global__ void swiglu_amax_vec16_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t orig_rows) {
  // One thread owns one NVFP4 scale block (16 values). Same exact SiLU
  // abs-bit max. Block amax is thread-local; no 2-thread shuffle.
  uint32_t local_max = 0;
  const int64_t groups_per_row = kActivationWidth / kVec16;
  const int64_t num_groups = orig_rows * groups_per_row;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < num_groups;
       group += stride) {
    const int64_t row = group / groups_per_row;
    const int col = static_cast<int>(
        (group - row * groups_per_row) * kVec16);
    const __nv_bfloat16* gate = raw + row * kRawWidth + col;
    const __nv_bfloat16* up = gate + kActivationWidth;
    const uint4 gate0 = *reinterpret_cast<const uint4*>(gate);
    const uint4 gate1 = *reinterpret_cast<const uint4*>(gate + 8);
    const uint4 up0 = *reinterpret_cast<const uint4*>(up);
    const uint4 up1 = *reinterpret_cast<const uint4*>(up + 8);
    const __nv_bfloat16* gv0 =
        reinterpret_cast<const __nv_bfloat16*>(&gate0);
    const __nv_bfloat16* gv1 =
        reinterpret_cast<const __nv_bfloat16*>(&gate1);
    const __nv_bfloat16* uv0 =
        reinterpret_cast<const __nv_bfloat16*>(&up0);
    const __nv_bfloat16* uv1 =
        reinterpret_cast<const __nv_bfloat16*>(&up1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t b0 = swiglu_abs_bits(gv0[i], uv0[i]);
      const uint32_t b1 = swiglu_abs_bits(gv1[i], uv1[i]);
      local_max = local_max < b0 ? b0 : local_max;
      local_max = local_max < b1 ? b1 : local_max;
    }
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
}

__global__ void swiglu_nvfp4_pack_from_amax_bits_vec16_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row = kActivationWidth / kVec16;
  const int64_t total_threads = padded_rows * threads_per_row;
  if (thread_linear >= total_threads) {
    return;
  }

  const float global_decode_scale =
      dynamic_scale_from_bf16_bits(global_max_bits);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    dynamic_scale_output[0] = global_decode_scale;
  }

  const int64_t row = thread_linear / threads_per_row;
  const int col = static_cast<int>(
      (thread_linear - row * threads_per_row) * kVec16);

  float values[kVec16];
  if (row < orig_rows) {
    const __nv_bfloat16* gate = raw + row * kRawWidth + col;
    const __nv_bfloat16* up = gate + kActivationWidth;
    const uint4 gate0 = *reinterpret_cast<const uint4*>(gate);
    const uint4 gate1 = *reinterpret_cast<const uint4*>(gate + 8);
    const uint4 up0 = *reinterpret_cast<const uint4*>(up);
    const uint4 up1 = *reinterpret_cast<const uint4*>(up + 8);
    const __nv_bfloat16* gv0 =
        reinterpret_cast<const __nv_bfloat16*>(&gate0);
    const __nv_bfloat16* gv1 =
        reinterpret_cast<const __nv_bfloat16*>(&gate1);
    const __nv_bfloat16* uv0 =
        reinterpret_cast<const __nv_bfloat16*>(&up0);
    const __nv_bfloat16* uv1 =
        reinterpret_cast<const __nv_bfloat16*>(&up1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = swiglu_like_eager_bf16(gv0[i], uv0[i]);
      values[i + 8] = swiglu_like_eager_bf16(gv1[i], uv1[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kVec16; ++i) {
      values[i] = 0.0f;
    }
  }

  float absmax = fabsf(values[0]);
#pragma unroll
  for (int i = 1; i < kVec16; ++i) {
    absmax = fmaxf(absmax, fabsf(values[i]));
  }

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
  const __nv_fp8_e4m3 scale_fp8 =
      static_cast<__nv_fp8_e4m3>(decode_scale);
  const float decode_scale_fp8 = static_cast<float>(scale_fp8);

  const int64_t scale_col = col / kFp4BlockSize;
  const size_t scale_offset = scale_factor_swizzled_offset(
      static_cast<size_t>(row),
      static_cast<size_t>(scale_col),
      kScaleCols);
  reinterpret_cast<__nv_fp8_e4m3*>(block_scales)[scale_offset] = scale_fp8;

  const float encode_scale = fminf(
      __fdividef(1.0f, decode_scale_fp8 * global_decode_scale), FLT_MAX);
#pragma unroll
  for (int i = 0; i < kVec16; ++i) {
    values[i] *= encode_scale;
  }

  union PackedFp4 {
    uint64_t u64;
    __nv_fp4x2_storage_t fp4x2[8];
  } packed;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    packed.fp4x2[i] = __nv_cvt_float2_to_fp4x2(
        float2{values[2 * i + 1], values[2 * i]},
        __NV_E2M1,
        cudaRoundNearest);
  }
  *reinterpret_cast<uint64_t*>(
      output + thread_linear * (kVec16 / 2)) = packed.u64;
}

__global__ void swiglu_nvfp4_pack_from_scale_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const float* __restrict__ supplied_scale,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  // Pack-only static rebind: same 8-wide SwiGLU tile as the dynamic pack,
  // but the global decode scale is a supplied fingerprint, not a live amax.
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

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
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

__global__ void bf16_amax_bits_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint32_t* __restrict__ global_max_bits,
    int64_t num_values) {
  uint32_t local_max = 0;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < num_values;
       idx += stride) {
    const uint32_t abs_bits = static_cast<uint32_t>(
        __bfloat16_as_ushort(input[idx]) & 0x7fffu);
    local_max = local_max < abs_bits ? abs_bits : local_max;
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

  __shared__ uint32_t warp_maxima[8];
  if (lane == 0) {
    warp_maxima[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = blockDim.x >> 5;
    local_max = lane < warp_count ? warp_maxima[lane] : 0;
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

template <bool DynamicScale>
__global__ void bf16_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ supplied_scale,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows,
    int width,
    int scale_cols) {
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row = width / kDirectValuesPerThread;
  const int64_t total_threads = padded_rows * threads_per_row;
  if (thread_linear >= total_threads) {
    return;
  }

  const int64_t row = thread_linear / threads_per_row;
  const int col = static_cast<int>(
      (thread_linear - row * threads_per_row) * kDirectValuesPerThread);

  float global_decode_scale;
  if constexpr (DynamicScale) {
    global_decode_scale = dynamic_scale_from_bf16_bits(global_max_bits);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      dynamic_scale_output[0] = global_decode_scale;
    }
  } else {
    global_decode_scale = supplied_scale[0];
  }

  float values[kDirectValuesPerThread];
  if (row < orig_rows) {
#pragma unroll
    for (int i = 0; i < kDirectValuesPerThread; ++i) {
      values[i] = __bfloat162float(input[row * width + col + i]);
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

  float decode_scale = __fdividef(
      __fdividef(absmax, 6.0f), global_decode_scale);
  decode_scale = fminf(decode_scale, 448.0f);
  const __nv_fp8_e4m3 scale_fp8 =
      static_cast<__nv_fp8_e4m3>(decode_scale);
  const float decode_scale_fp8 = static_cast<float>(scale_fp8);

  if ((threadIdx.x % kDirectThreadsPerScale) == 0) {
    const int64_t scale_col = col / kFp4BlockSize;
    const size_t scale_offset = scale_factor_swizzled_offset(
        static_cast<size_t>(row),
        static_cast<size_t>(scale_col),
        static_cast<uint32_t>(scale_cols));
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

void check_direct_input(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
              "input must have dtype torch.bfloat16");
  TORCH_CHECK(input.dim() == 2,
              "input must be a 2D tensor");
  TORCH_CHECK(is_h3_activation_width(input.size(1)),
              "input width must be an H3 NVFP4 width (5376, 7168, or 14336)");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.size(0) > 0,
              "NVFP4 quantization is undefined for an empty input");
}

std::vector<torch::Tensor> allocate_direct_outputs(
    const torch::Tensor& input) {
  const int64_t padded_rows = ((input.size(0) + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;
  const int64_t width = input.size(1);
  const auto byte_options = input.options().dtype(torch::kUInt8);
  torch::Tensor qx = torch::empty(
      {padded_rows, width / 2}, byte_options);
  torch::Tensor sx = torch::zeros(
      {scale_rows, width / kFp4BlockSize}, byte_options);
  return {qx, sx};
}

std::vector<torch::Tensor> bf16_nvfp4(
    torch::Tensor input,
    torch::Tensor global_scale) {
  check_direct_input(input);
  TORCH_CHECK(global_scale.is_cuda(), "global_scale must be a CUDA tensor");
  TORCH_CHECK(global_scale.scalar_type() == torch::kFloat32,
              "global_scale must have dtype torch.float32");
  TORCH_CHECK(global_scale.numel() == 1,
              "global_scale must contain exactly one value");
  TORCH_CHECK(global_scale.is_contiguous(), "global_scale must be contiguous");
  TORCH_CHECK(input.get_device() == global_scale.get_device(),
              "input and global_scale must be on the same CUDA device");

  const c10::cuda::CUDAGuard device_guard(input.device());
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(input);
  const int64_t padded_rows = outputs[0].size(0);
  const int width = static_cast<int>(input.size(1));
  const int scale_cols = width / kFp4BlockSize;
  const int64_t total_threads =
      padded_rows * (width / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      input.get_device()).stream();

  bf16_nvfp4_kernel<false><<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      global_scale.data_ptr<float>(),
      nullptr,
      nullptr,
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      input.size(0),
      padded_rows,
      width,
      scale_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

std::vector<torch::Tensor> bf16_nvfp4_dynamic(torch::Tensor input) {
  check_direct_input(input);
  const c10::cuda::CUDAGuard device_guard(input.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, input.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, input.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(input);

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_values = input.numel();
  const int64_t needed_blocks =
      (num_values + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      input.get_device()).stream();

  bf16_amax_bits_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      num_values);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int width = static_cast<int>(input.size(1));
  const int scale_cols = width / kFp4BlockSize;
  const int64_t total_threads =
      padded_rows * (width / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  bf16_nvfp4_kernel<true><<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      nullptr,
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      input.size(0),
      padded_rows,
      width,
      scale_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4(
    torch::Tensor raw,
    torch::Tensor global_scale,
    bool eager_bf16_rounding) {
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(global_scale.is_cuda(), "global_scale must be a CUDA tensor");
  TORCH_CHECK(global_scale.scalar_type() == torch::kFloat32,
              "global_scale must have dtype torch.float32");
  TORCH_CHECK(global_scale.numel() == 1,
              "global_scale must contain exactly one value");
  TORCH_CHECK(global_scale.is_contiguous(), "global_scale must be contiguous");
  TORCH_CHECK(raw.get_device() == global_scale.get_device(),
              "raw and global_scale must be on the same CUDA device");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  const int64_t orig_rows = raw.size(0);
  const int64_t padded_rows = ((orig_rows + 15) / 16) * 16;
  const int64_t scale_rows = ((padded_rows + 127) / 128) * 128;

  const auto byte_options = raw.options().dtype(torch::kUInt8);
  torch::Tensor qx = torch::empty(
      {padded_rows, kActivationWidth / 2}, byte_options);
  // The cuBLAS scale tile includes rows beyond padded_rows up to a multiple of
  // 128. They must be deterministic zero, just as in comfy-kitchen.
  torch::Tensor sx = torch::zeros({scale_rows, kScaleCols}, byte_options);

  if (padded_rows == 0) {
    return {qx, sx};
  }

  const int64_t total_threads =
      padded_rows * (kActivationWidth / kValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  if (eager_bf16_rounding) {
    swiglu_nvfp4_kernel<true><<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
        global_scale.data_ptr<float>(),
        qx.data_ptr<uint8_t>(),
        sx.data_ptr<uint8_t>(),
        orig_rows,
        padded_rows);
  } else {
    swiglu_nvfp4_kernel<false><<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
        global_scale.data_ptr<float>(),
        qx.data_ptr<uint8_t>(),
        sx.data_ptr<uint8_t>(),
        orig_rows,
        padded_rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qx, sx};
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic(torch::Tensor raw) {
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));

  const int64_t num_values = raw.size(0) * kActivationWidth;
  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t needed_blocks =
      (num_values + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_bf16_bits_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      num_values);
  finalize_dynamic_scale_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<torch::Tensor> outputs =
      swiglu_nvfp4(raw, global_scale, true);
  outputs.push_back(global_scale);
  return outputs;
}

torch::Tensor activation_workspace(const torch::Tensor& raw) {
  // Grow-only per-device cache: 50 FC2 calls per eval must not allocate
  // 558 MiB each time. Rows may only increase (production S is stable).
  static std::mutex mutex;
  static std::unordered_map<int, torch::Tensor> cache;
  const int device = raw.get_device();
  std::lock_guard<std::mutex> lock(mutex);
  torch::Tensor& slot = cache[device];
  if (!slot.defined() || slot.size(0) < raw.size(0) ||
      slot.device() != raw.device()) {
    slot = torch::empty({raw.size(0), kActivationWidth}, raw.options());
  }
  return slot.narrow(0, 0, raw.size(0));
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_oneshot(torch::Tensor raw) {
  // Exact identity: amax bits come from the stored BF16 SwiGLU, then the
  // existing dynamic pack reads that activation instead of recomputing SiLU.
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor activated = activation_workspace(raw);
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(activated);

  const int64_t num_values = raw.size(0) * kActivationWidth;
  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t needed_blocks =
      (num_values + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_store_amax_bf16_bits_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(activated.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      num_values);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int width = kActivationWidth;
  const int scale_cols = kScaleCols;
  const int64_t total_threads =
      padded_rows * (width / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  bf16_nvfp4_kernel<true><<<blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(activated.data_ptr()),
      nullptr,
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows,
      width,
      scale_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_fused(torch::Tensor raw) {
  // Exact two-pass producer without the 558 MiB activation store:
  //   1. 8-wide bound-filtered amax (skip SiLU when ru_bf16 bound cannot
  //      change the bit-max)
  //   2. 8-wide pack that inlines (amax/2688).to(float32)
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_bound_vec_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  swiglu_nvfp4_pack_from_amax_bits_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_twolevel(torch::Tensor raw) {
  // Exact producer that removes almost all amax SiLU:
  //   1. bound-only scan; exact SiLU only at each CTA's bound-argmax (L)
  //   2. exact SiLU only where bound > L
  //   3. 8-wide pack from amax bits (still recomputes SwiGLU; no 558 MiB)
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor exact_evals = torch::zeros(
      {1}, raw.options().dtype(torch::kInt64));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_bound_winner_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      nullptr,
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  swiglu_amax_sparse_exact_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      reinterpret_cast<unsigned long long*>(exact_evals.data_ptr<int64_t>()),
      nullptr,
      nullptr,
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  swiglu_nvfp4_pack_from_amax_bits_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  outputs.push_back(exact_evals);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_interval(torch::Tensor raw) {
  // Same two-level amax, but skip the sparse HBM scan when the stock
  // global scale is invariant on [L, U]: G(x)=(bf16(x)/2688).to(bf16)
  // is nondecreasing, so G(L)==G(U) implies G(amax)==G(L).
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor bound_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor exact_evals = torch::zeros(
      {1}, raw.options().dtype(torch::kInt64));
  torch::Tensor scanned = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_bound_winner_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(bound_bits.data_ptr<int32_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  swiglu_amax_sparse_exact_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      reinterpret_cast<unsigned long long*>(exact_evals.data_ptr<int64_t>()),
      reinterpret_cast<const uint32_t*>(bound_bits.data_ptr<int32_t>()),
      scanned.data_ptr<int32_t>(),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  swiglu_nvfp4_pack_from_amax_bits_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  outputs.push_back(exact_evals);
  outputs.push_back(scanned);
  outputs.push_back(max_bits);
  outputs.push_back(bound_bits);
  return outputs;
}

std::vector<torch::Tensor> swiglu_winner_lu(torch::Tensor raw) {
  // Bound-argmax exact L and tensor-wide bound U. No sparse scan, no pack.
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0, "empty raw");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor bound_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();
  swiglu_amax_bound_winner_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(bound_bits.data_ptr<int32_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {max_bits, bound_bits};
}

std::vector<torch::Tensor> swiglu_nvfp4_static_rebind(
    torch::Tensor raw, torch::Tensor global_scale) {
  // Fail-closed static pack: skip the amax pass. Byte-identical to the
  // dynamic two-pass iff global_scale is the stock
  // (amax/2688).to(bf16).to(fp32) for this raw tensor. Fingerprint is
  // shape/dtype/device/contiguity, not a raised margin.
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "static rebind is undefined for an empty input");
  TORCH_CHECK(global_scale.is_cuda(), "global_scale must be a CUDA tensor");
  TORCH_CHECK(global_scale.scalar_type() == torch::kFloat32,
              "global_scale must have dtype torch.float32");
  TORCH_CHECK(global_scale.numel() == 1,
              "global_scale must contain exactly one value");
  TORCH_CHECK(global_scale.is_contiguous(), "global_scale must be contiguous");
  TORCH_CHECK(raw.get_device() == global_scale.get_device(),
              "raw and global_scale must be on the same CUDA device");
  TORCH_CHECK(!global_scale.requires_grad(),
              "global_scale must not require grad");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));
  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();
  swiglu_nvfp4_pack_from_scale_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_coop(torch::Tensor raw) {
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  const int device = raw.get_device();
  int coop = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &coop, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(coop != 0, "device does not support cooperative launch");

  constexpr int kThreads = 256;
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      swiglu_nvfp4_dynamic_coop_kernel,
      kThreads,
      0));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int blocks = blocks_per_sm * prop.multiProcessorCount;
  TORCH_CHECK(blocks > 0, "cooperative occupancy is zero");

  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));
  int64_t padded_rows = outputs[0].size(0);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();

  const __nv_bfloat16* raw_ptr =
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr());
  uint32_t* max_ptr =
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>());
  float* scale_ptr = global_scale.data_ptr<float>();
  uint8_t* qx_ptr = outputs[0].data_ptr<uint8_t>();
  uint8_t* sx_ptr = outputs[1].data_ptr<uint8_t>();
  int64_t orig_rows = raw.size(0);
  void* args[] = {
      &raw_ptr,
      &max_ptr,
      &scale_ptr,
      &qx_ptr,
      &sx_ptr,
      &orig_rows,
      &padded_rows,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(swiglu_nvfp4_dynamic_coop_kernel),
      dim3(blocks),
      dim3(kThreads),
      args,
      0,
      stream));
  outputs.push_back(global_scale);
  torch::Tensor coop_blocks = torch::empty(
      {1}, raw.options().dtype(torch::kInt32));
  coop_blocks.fill_(blocks);
  outputs.push_back(coop_blocks);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_vec(torch::Tensor raw) {
  // Two-pass with pack-tile amax: every SiLU, 8-wide loads, then 8-wide
  // pack from amax bits. No bound skip, no extra store, no finalize kernel.
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_vec_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  swiglu_nvfp4_pack_from_amax_bits_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_vec16(torch::Tensor raw) {
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kVec16);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_vec16_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kVec16);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  swiglu_nvfp4_pack_from_amax_bits_vec16_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_inplace(torch::Tensor raw) {
  // Mutates raw: gate half becomes eager silu(gate). Not a serving path.
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");
  TORCH_CHECK(!raw.requires_grad(), "raw must not require grad");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_inplace_silu_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  swiglu_pack_from_silu_gate_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_inplace_prod(
    torch::Tensor raw) {
  // Mutates raw: up half becomes eager round(silu*up). Not a serving path.
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");
  TORCH_CHECK(!raw.requires_grad(), "raw must not require grad");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_inplace_prod_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t padded_rows = outputs[0].size(0);
  const int64_t total_threads =
      padded_rows * (kActivationWidth / kDirectValuesPerThread);
  const int blocks = static_cast<int>(
      (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
  swiglu_pack_from_up_prod_kernel<<<
      blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      raw.size(0),
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

torch::Tensor fc1_epilogue_assoc_mismatches(torch::Tensor raw) {
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0, "raw must contain at least one row");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor mismatches = torch::zeros(
      {1}, raw.options().dtype(torch::kInt64));
  constexpr int kReductionThreads = 256;
  constexpr int kMaximumReductionBlocks = 4096;
  const int64_t num_groups =
      raw.size(0) * (kActivationWidth / kDirectValuesPerThread);
  const int64_t needed_blocks =
      (num_groups + kReductionThreads - 1) / kReductionThreads;
  const int blocks = static_cast<int>(
      needed_blocks < kMaximumReductionBlocks
          ? needed_blocks
          : kMaximumReductionBlocks);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();
  fc1_epilogue_assoc_kernel<<<blocks, kReductionThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<unsigned long long*>(mismatches.data_ptr<int64_t>()),
      raw.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return mismatches;
}

torch::Tensor fc1_paired_store(
    torch::Tensor input,
    torch::Tensor weight,
    bool product) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(),
              "input and weight must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16
                  && weight.scalar_type() == torch::kBFloat16,
              "input and weight must have dtype torch.bfloat16");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kFc1K,
              "input must have shape [S, 5376]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kRawWidth
                  && weight.size(1) == kFc1K,
              "weight must have shape [28672, 5376]");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
              "input and weight must be contiguous");
  TORCH_CHECK(input.size(0) > 0, "input must contain at least one row");
  TORCH_CHECK(input.get_device() == weight.get_device(),
              "input and weight must be on the same CUDA device");
  TORCH_CHECK(!input.requires_grad() && !weight.requires_grad(),
              "input and weight must not require grad");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const int64_t rows = input.size(0);
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      input.options());
  const dim3 block(kFc1TileN);
  const dim3 grid(
      kActivationWidth / kFc1TileN,
      static_cast<unsigned int>(
          (rows + kFc1TileM - 1) / kFc1TileM));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      input.get_device()).stream();
  const __nv_bfloat16* in_ptr =
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  const __nv_bfloat16* w_ptr =
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  __nv_bfloat16* out_ptr =
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  if (product) {
    fc1_paired_mma_kernel<true><<<grid, block, 0, stream>>>(
        in_ptr, w_ptr, out_ptr, rows);
  } else {
    fc1_paired_mma_kernel<false><<<grid, block, 0, stream>>>(
        in_ptr, w_ptr, out_ptr, rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_wmma(
    torch::Tensor input,
    torch::Tensor weight,
    bool product) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(),
              "input and weight must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16
                  && weight.scalar_type() == torch::kBFloat16,
              "input and weight must have dtype torch.bfloat16");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kFc1K,
              "input must have shape [S, 5376]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kRawWidth
                  && weight.size(1) == kFc1K,
              "weight must have shape [28672, 5376]");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
              "input and weight must be contiguous");
  TORCH_CHECK(input.size(0) > 0, "input must contain at least one row");
  TORCH_CHECK(input.get_device() == weight.get_device(),
              "input and weight must be on the same CUDA device");
  TORCH_CHECK(!input.requires_grad() && !weight.requires_grad(),
              "input and weight must not require grad");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const int64_t rows = input.size(0);
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      input.options());
  const dim3 block(kWmmaThreads);
  const dim3 grid(
      kActivationWidth / kWmmaTile,
      static_cast<unsigned int>((rows + kWmmaTile - 1) / kWmmaTile));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      input.get_device()).stream();
  const __nv_bfloat16* in_ptr =
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  const __nv_bfloat16* w_ptr =
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  __nv_bfloat16* out_ptr =
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  if (product) {
    fc1_paired_wmma_kernel<true><<<grid, block, 0, stream>>>(
        in_ptr, w_ptr, out_ptr, rows);
  } else {
    fc1_paired_wmma_kernel<false><<<grid, block, 0, stream>>>(
        in_ptr, w_ptr, out_ptr, rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4(
    torch::Tensor a_packed,
    torch::Tensor w_packed,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda(),
              "packed tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8,
              "packed tensors must be uint8");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous(),
              "packed tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device(),
              "packed tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const int64_t rows = a_packed.size(0);
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(32);
  const dim3 grid(
      kActivationWidth / kNvfp4TileN,
      static_cast<unsigned int>((rows + kNvfp4TileM - 1) / kNvfp4TileM));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  if (product) {
    fc1_paired_nvfp4_kernel<true><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows);
  } else {
    fc1_paired_nvfp4_kernel<false><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_tiled(
    torch::Tensor a_packed,
    torch::Tensor w_packed,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda(),
              "packed tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8,
              "packed tensors must be uint8");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous(),
              "packed tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device(),
              "packed tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const int64_t rows = a_packed.size(0);
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kNvfp4TiledThreads);
  const dim3 grid(kActivationWidth / kNvfp4BigN);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  if (product) {
    fc1_paired_nvfp4_tiled_kernel<true><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows);
  } else {
    fc1_paired_nvfp4_tiled_kernel<false><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(32);
  const dim3 grid(
      kActivationWidth / kNvfp4TileN,
      static_cast<unsigned int>((rows + kNvfp4TileM - 1) / kNvfp4TileM));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_kernel<true><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_kernel<false><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_piped(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kNvfp4TiledThreads);
  const dim3 grid(
      kActivationWidth / kNvfp4BigN,
      static_cast<unsigned int>((rows + kNvfp4BigM - 1) / kNvfp4BigM));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_piped_kernel<true><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_piped_kernel<false><<<grid, block, 0, stream>>>(
        a_packed.data_ptr<uint8_t>(), w_packed.data_ptr<uint8_t>(),
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTmaM));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTmaN));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTmaThreads);
  const dim3 grid(
      kActivationWidth / kTmaN,
      static_cast<unsigned int>((rows + kTmaM - 1) / kTmaM));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma_kernel<true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma_kernel<false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma_sf(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTmaM));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTmaN));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTmaThreads);
  const dim3 grid(
      kActivationWidth / kTmaN,
      static_cast<unsigned int>((rows + kTmaM - 1) / kTmaM));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma_sf_kernel<true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma_sf_kernel<false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma256_kernel<true, false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma256_kernel<false, false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;

}

torch::Tensor fc1_paired_nvfp4_scaled_tma256_sw(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTma256M),
      CU_TENSOR_MAP_SWIZZLE_32B);
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaPackedK),
      static_cast<uint32_t>(kTma256N),
      CU_TENSOR_MAP_SWIZZLE_32B);
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma256_kernel<true, true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma256_kernel<false, true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma256k2_kernel<true, false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host, nullptr);
  } else {
    fc1_paired_nvfp4_scaled_tma256k2_kernel<false, false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2_ldm(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2_kernel<false, false, false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2_ldmb(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2_kernel<false, false, false, false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2_pipe(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2_kernel<
      false, false, false, false, false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2_leads(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2_kernel<
      false, false, false, false, false, false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2_pipea(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2_kernel<
      false, false, false, false, false, false, false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2s1(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2s1_kernel<<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256n32(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma32TmaN));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma32N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256n32_kernel<<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

pybind11::dict fc1_paired_nvfp4_scaled_tma256n32_attrs() {
  cudaFuncAttributes a{};
  TORCH_CHECK(
      cudaFuncGetAttributes(
          &a, fc1_paired_nvfp4_scaled_tma256n32_kernel) == cudaSuccess,
      "n32 attrs");
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, fc1_paired_nvfp4_scaled_tma256n32_kernel, kTma256Threads, 0);
  pybind11::dict out;
  out["regs"] = a.numRegs;
  out["smem"] = static_cast<int>(a.sharedSizeBytes);
  out["occupancy"] = occ;
  out["smem_struct"] = static_cast<int>(sizeof(Tma256N32Smem));
  return out;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma128k2(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTmaK4M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma128k2_kernel<<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

pybind11::dict fc1_paired_nvfp4_scaled_tma128k2_attrs() {
  cudaFuncAttributes a{};
  TORCH_CHECK(
      cudaFuncGetAttributes(
          &a, fc1_paired_nvfp4_scaled_tma128k2_kernel) == cudaSuccess,
      "128k2 attrs");
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, fc1_paired_nvfp4_scaled_tma128k2_kernel, kTma256Threads, 0);
  pybind11::dict out;
  out["regs"] = a.numRegs;
  out["smem"] = static_cast<int>(a.sharedSizeBytes);
  out["occupancy"] = occ;
  out["smem_struct"] = static_cast<int>(sizeof(Tma128K2Smem));
  return out;
}

pybind11::dict fc1_paired_nvfp4_scaled_tma256k2s1_attrs() {
  cudaFuncAttributes a{};
  cudaFuncAttributes b{};
  TORCH_CHECK(
      cudaFuncGetAttributes(
          &a, fc1_paired_nvfp4_scaled_tma256k2s1_kernel) == cudaSuccess,
      "s1 attrs");
  TORCH_CHECK(
      cudaFuncGetAttributes(
          &b,
          fc1_paired_nvfp4_scaled_tma256k2_kernel<
              false, false, false, false, false, false, false, false>)
          == cudaSuccess,
      "k2 attrs");
  int occ_s1 = 0;
  int occ_k2 = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_s1, fc1_paired_nvfp4_scaled_tma256k2s1_kernel, kTma256Threads, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_k2,
      fc1_paired_nvfp4_scaled_tma256k2_kernel<
          false, false, false, false, false, false, false, false>,
      kTma256Threads, 0);
  pybind11::dict out;
  out["s1_regs"] = a.numRegs;
  out["s1_smem"] = static_cast<int>(a.sharedSizeBytes);
  out["s1_occupancy"] = occ_s1;
  out["k2_regs"] = b.numRegs;
  out["k2_smem"] = static_cast<int>(b.sharedSizeBytes);
  out["k2_occupancy"] = occ_k2;
  return out;
}

std::vector<torch::Tensor> fc1_paired_nvfp4_scaled_tma256k2_amax(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  torch::Tensor max_bits = torch::zeros(
      {1}, a_packed.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, a_packed.options().dtype(torch::kFloat32));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2_kernel<false, false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host,
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  finalize_dynamic_scale_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, global_scale};
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2p(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  // One CTA per 256-row panel walks every N tile (A-stationary).
  const dim3 grid(
      1,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma256k2p_kernel<true>
        <<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma256k2p_kernel<false>
        <<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2ws(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTmaK2WsThreads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma256k2ws_kernel<true>
        <<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma256k2ws_kernel<false>
        <<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2ws4(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTmaK2Ws4Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_paired_nvfp4_scaled_tma256k2ws4_kernel<<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

pybind11::dict fc1_paired_nvfp4_scaled_tma256k2ws4_attrs() {
  cudaFuncAttributes a{};
  TORCH_CHECK(
      cudaFuncGetAttributes(
          &a, fc1_paired_nvfp4_scaled_tma256k2ws4_kernel) == cudaSuccess,
      "ws4 attrs");
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, fc1_paired_nvfp4_scaled_tma256k2ws4_kernel, kTmaK2Ws4Threads, 0);
  pybind11::dict out;
  out["regs"] = a.numRegs;
  out["smem"] = static_cast<int>(a.sharedSizeBytes);
  out["occupancy"] = occ;
  out["threads"] = kTmaK2Ws4Threads;
  out["warps"] = kTmaK2Ws4Threads / 32;
  out["prod_warps"] = kTmaK2Ws4ProdWarps;
  return out;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2_sw(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M),
      CU_TENSOR_MAP_SWIZZLE_64B);
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N),
      CU_TENSOR_MAP_SWIZZLE_64B);
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma256k2_kernel<true, true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host, nullptr);
  } else {
    fc1_paired_nvfp4_scaled_tma256k2_kernel<false, true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma128k4(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaK4M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma128k4_kernel<true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma128k4_kernel<false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_nvfp4_scaled_tma128n128k4(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaK4M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaKitN));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kRawWidth / kTmaKitN,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_nvfp4_scaled_tma128n128k4_kernel<false><<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_nvfp4_scaled_tma128n128k4_pipe(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaK4M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaKitN));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kRawWidth / kTmaKitN,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_nvfp4_scaled_tma128n128k4_kernel<false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_nvfp4_scaled_tma128n128k4_ldm(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaK4M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaKitN));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kRawWidth / kTmaKitN,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_nvfp4_scaled_tma128n128k4_kernel<false, false, true>
      <<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_nvfp4_scaled_tma128n128k4_sw(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaK4M),
      CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaKitN),
      CU_TENSOR_MAP_SWIZZLE_128B);
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kRawWidth / kTmaKitN,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_nvfp4_scaled_tma128n128k4_kernel<true><<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void ensure_kit_ws_smem() {
  static std::once_flag once;
  std::call_once(once, [] {
    const cudaError_t err = cudaFuncSetAttribute(
        fc1_nvfp4_scaled_tma128n128k4ws_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kKitWsDynSmem);
    TORCH_CHECK(err == cudaSuccess,
                "opt-in smem ", kKitWsDynSmem, " failed: ",
                cudaGetErrorString(err));
  });
}

pybind11::dict fc1_nvfp4_scaled_tma128n128k4ws_attrs() {
  ensure_kit_ws_smem();
  cudaFuncAttributes a{};
  const cudaError_t err = cudaFuncGetAttributes(
      &a, fc1_nvfp4_scaled_tma128n128k4ws_kernel);
  TORCH_CHECK(err == cudaSuccess, "cudaFuncGetAttributes: ",
              cudaGetErrorString(err));
  pybind11::dict out;
  out["threads"] = kKitWsThreads;
  out["warps"] = kKitWsThreads / 32;
  out["producer_warps"] = kKitWsProdWarps;
  out["mma_warps"] = kKitWsMmaWarps;
  out["dynamic_smem"] = kKitWsDynSmem;
  out["smem_struct_bytes"] = static_cast<int>(sizeof(KitWsSmem));
  out["num_regs"] = a.numRegs;
  out["local_size_bytes"] = static_cast<int>(a.localSizeBytes);
  out["max_dynamic_shared_size_bytes"] = a.maxDynamicSharedSizeBytes;
  out["shared_size_bytes"] = a.sharedSizeBytes;
  return out;
}

torch::Tensor fc1_nvfp4_scaled_tma128n128k4ws(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  ensure_kit_ws_smem();
  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaK4M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTmaKitN));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kKitWsThreads);
  const dim3 grid(
      kRawWidth / kTmaKitN,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_nvfp4_scaled_tma128n128k4ws_kernel<<<
      grid, block, kKitWsDynSmem, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma128k2n(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTmaK4M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTmaN128));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTmaN128,
      static_cast<unsigned int>((rows + kTmaK4M - 1) / kTmaK4M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma128k2n_kernel<true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma128k2n_kernel<false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k4(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  if (product) {
    fc1_paired_nvfp4_scaled_tma256k4_kernel<true><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma256k4_kernel<false><<<grid, block, 0, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fc1_nvfp4_scaled_tma256k4n1(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK4Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, kRawWidth}, a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kRawWidth / kTma256N,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  fc1_nvfp4_scaled_tma256k4n1_kernel<<<grid, block, 0, stream>>>(
      a_map, w_map,
      a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
      static_cast<int>(a_scales.size(1)),
      static_cast<int>(w_scales.size(1)),
      alpha_host);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void ensure_tma256k2n2_smem() {
  static std::once_flag once;
  std::call_once(once, [] {
    const int bytes = static_cast<int>(sizeof(TmaN2Smem));
    int optin = 0;
    const cudaError_t qerr = cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    TORCH_CHECK(qerr == cudaSuccess, "cudaDevAttrMaxSharedMemoryPerBlockOptin");
    TORCH_CHECK(
        optin >= bytes,
        "sequential N-halves need ", bytes,
        " B smem (", kTma256Threads * kTmaN2AccFloats * 4,
        " B FP32 acc1 + TMA); device opt-in is ", optin, " B");
    auto set = [bytes](auto* fn) {
      const cudaError_t err = cudaFuncSetAttribute(
          fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      TORCH_CHECK(err == cudaSuccess,
                  "opt-in smem ", bytes, " failed: ",
                  cudaGetErrorString(err));
    };
    set(fc1_paired_nvfp4_scaled_tma256k2n2_kernel<true>);
    set(fc1_paired_nvfp4_scaled_tma256k2n2_kernel<false>);
  });
}

torch::Tensor fc1_paired_nvfp4_scaled_tma256k2n2(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor w_packed,
    torch::Tensor w_scales,
    torch::Tensor alpha,
    bool product) {
  TORCH_CHECK(a_packed.is_cuda() && w_packed.is_cuda()
                  && a_scales.is_cuda() && w_scales.is_cuda()
                  && alpha.is_cuda(),
              "packed, scale, and alpha tensors must be CUDA");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8
                  && w_packed.scalar_type() == torch::kUInt8
                  && a_scales.scalar_type() == torch::kUInt8
                  && w_scales.scalar_type() == torch::kUInt8,
              "packed and scale tensors must be uint8");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat
                  && alpha.numel() == 1 && alpha.is_contiguous(),
              "alpha must be a contiguous one-element float32 tensor");
  TORCH_CHECK(a_packed.dim() == 2 && a_packed.size(1) == kNvfp4PackedK,
              "a_packed must have shape [S, 2688]");
  TORCH_CHECK(w_packed.dim() == 2 && w_packed.size(0) == kRawWidth
                  && w_packed.size(1) == kNvfp4PackedK,
              "w_packed must have shape [28672, 2688]");
  constexpr int kScaleColsK = kFc1K / kFp4BlockSize;
  TORCH_CHECK(a_scales.dim() == 2 && a_scales.size(1) == kScaleColsK,
              "a_scales must have shape [roundup(S,128), 336]");
  TORCH_CHECK(w_scales.dim() == 2 && w_scales.size(0) >= kRawWidth
                  && w_scales.size(1) == kScaleColsK,
              "w_scales must have shape [>=28672, 336]");
  TORCH_CHECK(a_packed.is_contiguous() && w_packed.is_contiguous()
                  && a_scales.is_contiguous() && w_scales.is_contiguous(),
              "packed and scale tensors must be contiguous");
  TORCH_CHECK(a_packed.size(0) > 0, "a_packed must contain at least one row");
  const int64_t rows = a_packed.size(0);
  const int64_t scale_rows = ((rows + 127) / 128) * 128;
  TORCH_CHECK(a_scales.size(0) >= scale_rows,
              "a_scales rows must cover roundup(S, 128)");
  TORCH_CHECK(a_packed.get_device() == w_packed.get_device()
                  && a_packed.get_device() == a_scales.get_device()
                  && a_packed.get_device() == w_scales.get_device()
                  && a_packed.get_device() == alpha.get_device(),
              "all tensors must share a device");

  ensure_tma256k2n2_smem();
  const c10::cuda::CUDAGuard device_guard(a_packed.device());
  const CUtensorMap a_map = make_nvfp4_tmap(
      a_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(rows),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256M));
  const CUtensorMap w_map = make_nvfp4_tmap(
      w_packed.data_ptr<uint8_t>(),
      static_cast<uint64_t>(kNvfp4PackedK),
      static_cast<uint64_t>(kRawWidth),
      static_cast<uint32_t>(kTmaK2Packed),
      static_cast<uint32_t>(kTma256N));
  torch::Tensor output = torch::empty(
      {rows, product ? kActivationWidth : kRawWidth},
      a_packed.options().dtype(torch::kBFloat16));
  const dim3 block(kTma256Threads);
  const dim3 grid(
      kActivationWidth / kTmaN2,
      static_cast<unsigned int>((rows + kTma256M - 1) / kTma256M));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      a_packed.get_device()).stream();
  const float alpha_host = alpha.item<float>();
  const size_t dyn = sizeof(TmaN2Smem);
  if (product) {
    fc1_paired_nvfp4_scaled_tma256k2n2_kernel<true>
        <<<grid, block, dyn, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  } else {
    fc1_paired_nvfp4_scaled_tma256k2n2_kernel<false>
        <<<grid, block, dyn, stream>>>(
        a_map, w_map,
        a_scales.data_ptr<uint8_t>(), w_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows,
        static_cast<int>(a_scales.size(1)),
        static_cast<int>(w_scales.size(1)),
        alpha_host);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> swiglu_nvfp4_dynamic_swizzle(torch::Tensor raw) {
  // Two-pass with 128x64 tiles matching one cuBLAS E4M3 slab.
  TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
  TORCH_CHECK(raw.scalar_type() == torch::kBFloat16,
              "raw must have dtype torch.bfloat16");
  TORCH_CHECK(raw.dim() == 2 && raw.size(1) == kRawWidth,
              "raw must have shape [S, 28672]");
  TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
  TORCH_CHECK(raw.size(0) > 0,
              "dynamic scaling is undefined for an empty input");

  const c10::cuda::CUDAGuard device_guard(raw.device());
  torch::Tensor max_bits = torch::zeros(
      {1}, raw.options().dtype(torch::kInt32));
  torch::Tensor global_scale = torch::empty(
      {1}, raw.options().dtype(torch::kFloat32));
  std::vector<torch::Tensor> outputs = allocate_direct_outputs(
      raw.narrow(1, 0, kActivationWidth));

  const int64_t orig_rows = raw.size(0);
  const int64_t padded_rows = outputs[0].size(0);
  const dim3 block(kSwizzleTileR);
  const dim3 grid(
      kActivationWidth / kSwizzleTileC,
      static_cast<unsigned int>(
          (padded_rows + kSwizzleTileR - 1) / kSwizzleTileR));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(
      raw.get_device()).stream();

  swiglu_amax_swizzle_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<uint32_t*>(max_bits.data_ptr<int32_t>()),
      orig_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  swiglu_pack_swizzle_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(raw.data_ptr()),
      reinterpret_cast<const uint32_t*>(max_bits.data_ptr<int32_t>()),
      global_scale.data_ptr<float>(),
      outputs[0].data_ptr<uint8_t>(),
      outputs[1].data_ptr<uint8_t>(),
      orig_rows,
      padded_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  outputs.push_back(global_scale);
  return outputs;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "bf16_nvfp4",
      &bf16_nvfp4,
      "H3 BF16 activation to cuBLAS-layout NVFP4 (CUDA)",
      pybind11::arg("input"),
      pybind11::arg("global_scale"));
  m.def(
      "bf16_nvfp4_dynamic",
      &bf16_nvfp4_dynamic,
      "Two-pass dynamic-scale H3 BF16 activation to NVFP4 (CUDA)",
      pybind11::arg("input"));
  m.def(
      "swiglu_nvfp4",
      &swiglu_nvfp4,
      "Fused H3 BF16 SwiGLU to cuBLAS-layout NVFP4 (CUDA)",
      pybind11::arg("raw"),
      pybind11::arg("global_scale"),
      pybind11::arg("eager_bf16_rounding") = true);
  m.def(
      "swiglu_nvfp4_dynamic",
      &swiglu_nvfp4_dynamic,
      "Two-pass dynamic-scale H3 BF16 SwiGLU to NVFP4 (CUDA)",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_dynamic_oneshot",
      &swiglu_nvfp4_dynamic_oneshot,
      "One-SwiGLU dynamic-scale H3 BF16 SwiGLU to NVFP4 (store+pack)",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_dynamic_fused",
      &swiglu_nvfp4_dynamic_fused,
      "Bound-filtered amax + 8-wide pack from amax bits (no activation store)",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_dynamic_twolevel",
      &swiglu_nvfp4_dynamic_twolevel,
      "Two-level bound-argmax amax + 8-wide pack from amax bits",
      pybind11::arg("raw"));
  m.def(
      "swiglu_winner_lu",
      &swiglu_winner_lu,
      "Bound-argmax exact L and tensor-wide bound U (no pack)");
  m.def(
      "swiglu_nvfp4_dynamic_interval",
      &swiglu_nvfp4_dynamic_interval,
      "Two-level amax with G(L)==G(U) sparse-scan skip",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_static_rebind",
      &swiglu_nvfp4_static_rebind,
      "Fail-closed 8-wide SwiGLU NVFP4 pack from a supplied global scale",
      pybind11::arg("raw"),
      pybind11::arg("global_scale"));
  m.def(
      "swiglu_nvfp4_dynamic_coop",
      &swiglu_nvfp4_dynamic_coop,
      "One-launch cooperative bound-filtered amax + 8-wide pack",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_dynamic_vec",
      &swiglu_nvfp4_dynamic_vec,
      "8-wide exact amax + 8-wide pack from amax bits",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_dynamic_vec16",
      &swiglu_nvfp4_dynamic_vec16,
      "16-wide exact two-pass: one thread owns one NVFP4 scale block",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_dynamic_inplace",
      &swiglu_nvfp4_dynamic_inplace,
      "One-SiLU amax that overwrites gate; pack multiplies only",
      pybind11::arg("raw"));
  m.def(
      "swiglu_nvfp4_dynamic_inplace_prod",
      &swiglu_nvfp4_dynamic_inplace_prod,
      "One-SiLU amax that overwrites up with the product; pack is half-width",
      pybind11::arg("raw"));
  m.def(
      "fc1_epilogue_assoc_mismatches",
      &fc1_epilogue_assoc_mismatches,
      "Count eager-BF16 vs fused-FP32-acc SwiGLU product mismatches",
      pybind11::arg("raw"));
  m.def(
      "fc1_paired_store",
      &fc1_paired_store,
      "Paired-N FC1 MMA: store [gate|up] or the eager SwiGLU product",
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("product"));
  m.def(
      "swiglu_nvfp4_dynamic_swizzle",
      &swiglu_nvfp4_dynamic_swizzle,
      "Two-pass SwiGLU NVFP4 tiled to the 128x4 cuBLAS E4M3 slab",
      pybind11::arg("raw"));
  m.def(
      "fc1_paired_wmma",
      &fc1_paired_wmma,
      "Tensor-core paired-N FC1: store [gate|up] or eager SwiGLU product",
      pybind11::arg("input"),
      pybind11::arg("weight"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4",
      &fc1_paired_nvfp4,
      "SM120 NVFP4 paired-N: store [gate|up] or eager SwiGLU product",
      pybind11::arg("a_packed"),
      pybind11::arg("w_packed"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_tiled",
      &fc1_paired_nvfp4_tiled,
      "Persistent tiled SM120 NVFP4 paired-N (A reused across N)",
      pybind11::arg("a_packed"),
      pybind11::arg("w_packed"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled",
      &fc1_paired_nvfp4_scaled,
      "Scale-aware SM120 NVFP4 paired-N: kitchen atom + eager product",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_piped",
      &fc1_paired_nvfp4_scaled_piped,
      "Pipelined scale-aware SM120 NVFP4 paired-N (cp.async K + A reuse)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma",
      &fc1_paired_nvfp4_scaled_tma,
      "TMA 128x128 scale-aware SM120 NVFP4 paired-N (tensor map + mbarrier)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma_sf",
      &fc1_paired_nvfp4_scaled_tma_sf,
      "TMA 128x128 paired-N with cuBLAS 128x4 scale-slab bulk loads",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256",
      &fc1_paired_nvfp4_scaled_tma256,
      "TMA 256x64 3-stage paired-N (larger M tile + deeper K pipeline)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256_sw",
      &fc1_paired_nvfp4_scaled_tma256_sw,
      "TMA 256x64 3-stage paired-N with SWIZZLE_32B remapped to PTX fragment",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2",
      &fc1_paired_nvfp4_scaled_tma256k2,
      "TMA 256x64 K=128 box: two m16n8k64 atoms per load",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2_ldm",
      &fc1_paired_nvfp4_scaled_tma256k2_ldm,
      "TMA 256x64 K=128 A fragment via ldmatrix.x4",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2_ldmb",
      &fc1_paired_nvfp4_scaled_tma256k2_ldmb,
      "TMA 256x64 K=128 B fragment via ldmatrix.x4 (two N-subtiles)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2_pipe",
      &fc1_paired_nvfp4_scaled_tma256k2_pipe,
      "TMA 256x64 K=128 next-N B/SFB overlapped on m16n8k64",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2_leads",
      &fc1_paired_nvfp4_scaled_tma256k2_leads,
      "TMA 256x64 K=128 SFA/SFB only on contributing lanes",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2_pipea",
      &fc1_paired_nvfp4_scaled_tma256k2_pipea,
      "TMA 256x64 K=128 next-K A/SFA overlapped on m16n8k64",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2s1",
      &fc1_paired_nvfp4_scaled_tma256k2s1,
      "TMA 256x64 K=128 1-stage paired-N (occupancy vs 3-stage k2)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2s1_attrs",
      &fc1_paired_nvfp4_scaled_tma256k2s1_attrs,
      "Occupancy/regs/smem for 1-stage k2 vs 3-stage k2");
  m.def(
      "fc1_paired_nvfp4_scaled_tma256n32",
      &fc1_paired_nvfp4_scaled_tma256n32,
      "TMA 256x32 K=128 2-stage paired-N (half acc, occupancy)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256n32_attrs",
      &fc1_paired_nvfp4_scaled_tma256n32_attrs,
      "Occupancy/regs/smem for 256x32 K=128 2-stage");
  m.def(
      "fc1_paired_nvfp4_scaled_tma128k2",
      &fc1_paired_nvfp4_scaled_tma128k2,
      "TMA 128x64 K=128 2-stage paired-N (64-float acc, occupancy)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma128k2_attrs",
      &fc1_paired_nvfp4_scaled_tma128k2_attrs,
      "Occupancy/regs/smem for 128x64 K=128 2-stage");
  m.def(
      "fc1_nvfp4_scaled_tma256k4n1",
      &fc1_nvfp4_scaled_tma256k4n1,
      "TMA 256x64 K=256 2-stage single-N (paired 2-stage is 112 KiB)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2_amax",
      &fc1_paired_nvfp4_scaled_tma256k2_amax,
      "TMA 256x64 K=128 [gate|up] store + epilogue live amax",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2_sw",
      &fc1_paired_nvfp4_scaled_tma256k2_sw,
      "TMA 256x64 K=128 box with SWIZZLE_64B remapped to PTX fragment",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma128k4",
      &fc1_paired_nvfp4_scaled_tma128k4,
      "TMA 128x64 K=256 box: four m16n8k64 atoms per load",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma128k2n",
      &fc1_paired_nvfp4_scaled_tma128k2n,
      "TMA 128x128 K=128 box: 16 n-subtiles, register-legal wider N",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k4",
      &fc1_paired_nvfp4_scaled_tma256k4,
      "TMA 256x64 K=256 box: four m16n8k64 atoms per load",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2n2",
      &fc1_paired_nvfp4_scaled_tma256k2n2,
      "TMA 256x128 sequential N-halves: one A stream, k2 acc + opt-in spill",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2ws",
      &fc1_paired_nvfp4_scaled_tma256k2ws,
      "TMA 256x64 K=128 warp-specialized: producer + 8 MMA warps",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2ws4",
      &fc1_paired_nvfp4_scaled_tma256k2ws4,
      "TMA 256x64 K=128 12-warp: 4 producers + 8 MMA (kitchen split on k2)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2ws4_attrs",
      &fc1_paired_nvfp4_scaled_tma256k2ws4_attrs,
      "Occupancy/regs/smem for 12-warp 4-producer k2");
  m.def(
      "fc1_paired_nvfp4_scaled_tma256k2p",
      &fc1_paired_nvfp4_scaled_tma256k2p,
      "TMA 256x64 K=128 persistent: one CTA walks all N for a 256-row A panel",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"),
      pybind11::arg("product"));
  m.def(
      "fc1_nvfp4_scaled_tma128n128k4",
      &fc1_nvfp4_scaled_tma128n128k4,
      "TMA 128x128x256 single-N kitchen tile (one B operand, not paired)",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_nvfp4_scaled_tma128n128k4_pipe",
      &fc1_nvfp4_scaled_tma128n128k4_pipe,
      "TMA 128x128x256 with software-pipelined B fragments",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_nvfp4_scaled_tma128n128k4_ldm",
      &fc1_nvfp4_scaled_tma128n128k4_ldm,
      "TMA 128x128x256 A fragment via ldmatrix.x4",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_nvfp4_scaled_tma128n128k4_sw",
      &fc1_nvfp4_scaled_tma128n128k4_sw,
      "TMA 128x128x256 SWIZZLE_128B remapped to the PTX fragment",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_nvfp4_scaled_tma128n128k4ws",
      &fc1_nvfp4_scaled_tma128n128k4ws,
      "TMA 128x128x256 12-warp: 4 producer + 8 MMA, 86 KiB dynamic",
      pybind11::arg("a_packed"),
      pybind11::arg("a_scales"),
      pybind11::arg("w_packed"),
      pybind11::arg("w_scales"),
      pybind11::arg("alpha"));
  m.def(
      "fc1_nvfp4_scaled_tma128n128k4ws_attrs",
      &fc1_nvfp4_scaled_tma128n128k4ws_attrs,
      "Launch attrs for the 12-warp kitchen-tile mainloop");
}
