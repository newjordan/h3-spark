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

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
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

std::vector<torch::Tensor> bf16_nvfp4_from_amax_bits(
    torch::Tensor input,
    torch::Tensor max_bits) {
  check_direct_input(input);
  TORCH_CHECK(max_bits.is_cuda(), "max_bits must be a CUDA tensor");
  TORCH_CHECK(max_bits.scalar_type() == torch::kInt32,
              "max_bits must have dtype torch.int32");
  TORCH_CHECK(max_bits.numel() == 1,
              "max_bits must contain exactly one value");
  TORCH_CHECK(max_bits.is_contiguous(), "max_bits must be contiguous");
  TORCH_CHECK(input.get_device() == max_bits.get_device(),
              "input and max_bits must be on the same CUDA device");

  const c10::cuda::CUDAGuard device_guard(input.device());
  torch::Tensor global_scale = torch::empty(
      {1}, input.options().dtype(torch::kFloat32));
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

  // The producer has already reduced these exact BF16 output bytes. Reuse its
  // device-side magnitude bits and run only the packing pass.
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

__global__ void swiglu_nvfp4_from_amax_bits_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int64_t orig_rows,
    int64_t padded_rows) {
  // Same 4-wide eager tile as swiglu_nvfp4_kernel<true>. G is
  // dynamic_scale_from_bf16_bits — the old <<<1,1>>> finalize, inlined.
  const int64_t thread_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t threads_per_row = kActivationWidth / kValuesPerThread;
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
      (thread_linear - row * threads_per_row) * kValuesPerThread);

  float values[kValuesPerThread];
  if (row < orig_rows) {
    const __nv_bfloat16* gate = raw + row * kRawWidth + col;
    const __nv_bfloat16* up = gate + kActivationWidth;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      values[i] = swiglu_like_eager_bf16(gate[i], up[i]);
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
#pragma unroll
  for (int offset = kThreadsPerScale / 2; offset >= 1; offset /= 2) {
    absmax = fmaxf(
        absmax,
        __shfl_down_sync(kFullMask, absmax, offset, kThreadsPerScale));
  }
  const int lane = threadIdx.x & 31;
  const int leader = (lane / kThreadsPerScale) * kThreadsPerScale;
  absmax = __shfl_sync(kFullMask, absmax, leader, 32);

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
  packed.fp4x2[0] = __nv_cvt_float2_to_fp4x2(
      float2{values[1], values[0]}, __NV_E2M1, cudaRoundNearest);
  packed.fp4x2[1] = __nv_cvt_float2_to_fp4x2(
      float2{values[3], values[2]}, __NV_E2M1, cudaRoundNearest);
  *reinterpret_cast<uint16_t*>(output + thread_linear * 2) = packed.u16;
}

__device__ __forceinline__ void reduce_atomic_max_bits(
    uint32_t local_max, uint32_t* global_max_bits) {
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

__global__ void swiglu_amax_vec_kernel(
    const __nv_bfloat16* __restrict__ raw,
    uint32_t* __restrict__ global_max_bits,
    int64_t orig_rows) {
  uint32_t local_max = 0;
  const int64_t groups_per_row = kActivationWidth / kDirectValuesPerThread;
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
      const __nv_bfloat16 value =
          __float2bfloat16_rn(swiglu_like_eager_bf16(gate_v[i], up_v[i]));
      const uint32_t abs_bits = static_cast<uint32_t>(
          __bfloat16_as_ushort(value) & 0x7fffu);
      local_max = local_max < abs_bits ? abs_bits : local_max;
    }
  }
  reduce_atomic_max_bits(local_max, global_max_bits);
}

__global__ void swiglu_nvfp4_pack_from_amax_bits_kernel(
    const __nv_bfloat16* __restrict__ raw,
    const uint32_t* __restrict__ global_max_bits,
    float* __restrict__ dynamic_scale_output,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
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
      "bf16_nvfp4_from_amax_bits",
      &bf16_nvfp4_from_amax_bits,
      "Pack H3 BF16 activation using producer-supplied exact amax bits (CUDA)",
      pybind11::arg("input"),
      pybind11::arg("max_bits"));
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
}
