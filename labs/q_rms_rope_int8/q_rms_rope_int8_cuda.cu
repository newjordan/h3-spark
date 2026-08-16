/*
 * H3-specific fused Q/K RMSNorm + split-half RoPE and Sage Q INT8 quantizer.
 *
 * This is an isolated research kernel. It deliberately has a narrow contract:
 * BF16, batch 1, 56 heads, head dim 128 / rot dim 96, NHD, 32-token Sage
 * groups. Dimensions 96:128 take the production RMS-only tail path.
 * K is transformed in place; Q is rounded exactly like comfy-kitchen but only
 * materialized as Sage's INT8 tensor and FP32 per-warp scale.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kHeads = 56;
constexpr int kHeadDim = 128;
constexpr int kRotDim = 96;
constexpr int kPairs = kRotDim / 2;
constexpr int kTokensPerGroup = 32;
constexpr int kQkvRowWidth = kHeads * kHeadDim * 3;
constexpr const char* kBuildId =
    "h3-q-rms-rope96-qint8-v3-torch2.13-sm120";

__device__ __forceinline__ float warp_reduce_sum_ck(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

// This is SageAttention's XOR-tree max reduction, including the second XOR
// tree over warp leaders. Only thread zero consumes the final result.
template <int WarpsPerBlock>
__device__ __forceinline__ float block_reduce_max_sage(float value) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
  }

  __shared__ float warp_maxima[32];
  if (lane == 0) {
    warp_maxima[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < 32 && lane < WarpsPerBlock
      ? warp_maxima[lane]
      : -1.0e20f;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
  }
  return value;
}

__device__ __forceinline__ int8_t sage_float_to_int8_rn(float value) {
  uint32_t result;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
  return reinterpret_cast<const int8_t&>(result);
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32) void qk_rms_rope_q_int8_kernel(
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ freqs,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    int8_t* __restrict__ q_int8,
    float* __restrict__ q_scale,
    int sequence,
    int64_t q_sequence_stride,
    int64_t k_sequence_stride,
    int64_t freqs_sequence_stride,
    float epsilon) {
  static_assert(kTokensPerGroup % WarpsPerBlock == 0);
  constexpr int TokensPerWarp = kTokensPerGroup / WarpsPerBlock;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int head = static_cast<int>(blockIdx.y);

  // comfy-kitchen's contiguous split-half specialization assigns two adjacent
  // pairs to each lane. Keep each BF16 storage boundary: one after RMSNorm and
  // scaling and a second after the FP32 2x2 rotation. A warp may process more
  // than one token serially; each token retains the identical one-warp RMS
  // reduction while smaller CTAs permit substantially better SM occupancy.
  // Each lane owns up to two rotated values in each split half and one
  // normalized-only tail value. Lanes 24-31 have no rotated pairs.
  __nv_bfloat16 q_values[TokensPerWarp][5];
  float local_amax = 0.0000001f;
#pragma unroll
  for (int token_slot = 0; token_slot < TokensPerWarp; ++token_slot) {
    const int token_in_group = warp + token_slot * WarpsPerBlock;
    const int token =
        static_cast<int>(blockIdx.x) * kTokensPerGroup + token_in_group;
    const bool valid = token < sequence;
    const int64_t q_base =
        static_cast<int64_t>(token) * q_sequence_stride + head * kHeadDim;
    const int64_t k_base =
        static_cast<int64_t>(token) * k_sequence_stride + head * kHeadDim;

    // Match comfy-kitchen's one-warp-per-token RMS accumulation exactly: each
    // lane visits d, d+32, d+64, d+96 with fmaf and the same shuffle-down tree.
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    if (valid) {
#pragma unroll
      for (int d = lane; d < kHeadDim; d += 32) {
        const float q_value = __bfloat162float(q[q_base + d]);
        const float k_value = __bfloat162float(k[k_base + d]);
        q_sum = fmaf(q_value, q_value, q_sum);
        k_sum = fmaf(k_value, k_value, k_sum);
      }
    }
    q_sum = warp_reduce_sum_ck(q_sum);
    k_sum = warp_reduce_sum_ck(k_sum);
    const float q_rrms =
        rsqrtf(q_sum / static_cast<float>(kHeadDim) + epsilon);
    const float k_rrms =
        rsqrtf(k_sum / static_cast<float>(kHeadDim) + epsilon);

#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int pair = lane * 2 + p;
      const int high = pair + kPairs;

      __nv_bfloat16 q_out0 = __float2bfloat16_rn(0.0f);
      __nv_bfloat16 q_out1 = __float2bfloat16_rn(0.0f);
      if (valid && pair < kPairs) {
        float q0 = __bfloat162float(q[q_base + pair]);
        float q1 = __bfloat162float(q[q_base + high]);
        float k0 = __bfloat162float(k[k_base + pair]);
        float k1 = __bfloat162float(k[k_base + high]);

        q0 = __bfloat162float(__float2bfloat16_rn(
            q0 * q_rrms * __bfloat162float(q_weight[pair])));
        q1 = __bfloat162float(__float2bfloat16_rn(
            q1 * q_rrms * __bfloat162float(q_weight[high])));
        k0 = __bfloat162float(__float2bfloat16_rn(
            k0 * k_rrms * __bfloat162float(k_weight[pair])));
        k1 = __bfloat162float(__float2bfloat16_rn(
            k1 * k_rrms * __bfloat162float(k_weight[high])));

        const int64_t freq_base =
            static_cast<int64_t>(token) * freqs_sequence_stride + pair * 4;
        const float f00 = __bfloat162float(freqs[freq_base]);
        const float f01 = __bfloat162float(freqs[freq_base + 1]);
        const float f10 = __bfloat162float(freqs[freq_base + 2]);
        const float f11 = __bfloat162float(freqs[freq_base + 3]);

        q_out0 = __float2bfloat16_rn(f00 * q0 + f01 * q1);
        q_out1 = __float2bfloat16_rn(f10 * q0 + f11 * q1);
        k[k_base + pair] = __float2bfloat16_rn(f00 * k0 + f01 * k1);
        k[k_base + high] = __float2bfloat16_rn(f10 * k0 + f11 * k1);
      }

      q_values[token_slot][p] = q_out0;
      q_values[token_slot][p + 2] = q_out1;
      local_amax = fmaxf(local_amax, fabsf(__bfloat162float(q_out0)));
      local_amax = fmaxf(local_amax, fabsf(__bfloat162float(q_out1)));
    }

    // comfy-kitchen's norm-only tail starts at rot_dim and performs exactly
    // one BF16 cast after `input * rrms * weight`, with no RoPE operation.
    const int tail = kRotDim + lane;
    __nv_bfloat16 q_tail = __float2bfloat16_rn(0.0f);
    if (valid) {
      q_tail = __float2bfloat16_rn(
          __bfloat162float(q[q_base + tail]) * q_rrms *
          __bfloat162float(q_weight[tail]));
      k[k_base + tail] = __float2bfloat16_rn(
          __bfloat162float(k[k_base + tail]) * k_rrms *
          __bfloat162float(k_weight[tail]));
    }
    q_values[token_slot][4] = q_tail;
    local_amax = fmaxf(local_amax, fabsf(__bfloat162float(q_tail)));
  }

  __shared__ float group_amax;
  const float reduced_amax = block_reduce_max_sage<WarpsPerBlock>(local_amax);
  if (threadIdx.x == 0) {
    group_amax = reduced_amax;
    q_scale[static_cast<int64_t>(head) * gridDim.x + blockIdx.x] =
        reduced_amax / 127.0f;
  }
  __syncthreads();

  const float encode_scale = 127.0f / group_amax;
#pragma unroll
  for (int token_slot = 0; token_slot < TokensPerWarp; ++token_slot) {
    const int token_in_group = warp + token_slot * WarpsPerBlock;
    const int token =
        static_cast<int>(blockIdx.x) * kTokensPerGroup + token_in_group;
    if (token < sequence) {
      const int64_t output_base =
          (static_cast<int64_t>(token) * kHeads + head) * kHeadDim;
      const int pair = lane * 2;
      if (pair < kPairs) {
        q_int8[output_base + pair] = sage_float_to_int8_rn(
            __bfloat162float(q_values[token_slot][0]) * encode_scale);
        q_int8[output_base + pair + 1] = sage_float_to_int8_rn(
            __bfloat162float(q_values[token_slot][1]) * encode_scale);
        q_int8[output_base + pair + kPairs] = sage_float_to_int8_rn(
            __bfloat162float(q_values[token_slot][2]) * encode_scale);
        q_int8[output_base + pair + kPairs + 1] = sage_float_to_int8_rn(
            __bfloat162float(q_values[token_slot][3]) * encode_scale);
      }
      q_int8[output_base + kRotDim + lane] = sage_float_to_int8_rn(
          __bfloat162float(q_values[token_slot][4]) * encode_scale);
    }
  }
}

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    torch::ScalarType dtype,
    const torch::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.device() == device, name, " must be on the Q device");
}

std::vector<torch::Tensor> qk_rms_rope_q_int8(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor freqs,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double epsilon,
    int warps_per_block) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  const torch::Device device = q.device();
  check_tensor(q, "q", torch::kBFloat16, device);
  check_tensor(k, "k", torch::kBFloat16, device);
  check_tensor(freqs, "freqs", torch::kBFloat16, device);
  check_tensor(q_weight, "q_weight", torch::kBFloat16, device);
  check_tensor(k_weight, "k_weight", torch::kBFloat16, device);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q and k must be rank 4 NHD tensors");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k shapes must match");
  TORCH_CHECK(q.size(0) == 1 && q.size(2) == kHeads && q.size(3) == kHeadDim,
              "only H3 [1,S,56,128] q/k are supported");
  TORCH_CHECK(q.size(1) > 1024,
              "H3 Q fusion requires a sequence longer than 1024 tokens");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1,
              "q/k head dimensions must be contiguous");
  TORCH_CHECK(q.stride(2) == kHeadDim && k.stride(2) == kHeadDim,
              "q/k heads must be densely packed");
  TORCH_CHECK(q.stride(1) == kQkvRowWidth &&
              k.stride(1) == kQkvRowWidth,
              "q/k must be views of production packed QKV rows");
  TORCH_CHECK(q.stride(0) == q.size(1) * kQkvRowWidth &&
              k.stride(0) == k.size(1) * kQkvRowWidth,
              "q/k batch strides must match production packed QKV");
  TORCH_CHECK(k.storage_offset() == q.storage_offset() + kHeads * kHeadDim,
              "k must immediately follow q in the same packed QKV storage");
  TORCH_CHECK(q.storage().unsafeGetStorageImpl() ==
              k.storage().unsafeGetStorageImpl(),
              "q and k must share the same packed QKV storage");
  TORCH_CHECK(q.size(1) <= std::numeric_limits<int>::max(), "sequence is too long");
  TORCH_CHECK(q_weight.dim() == 1 && q_weight.numel() == kHeadDim &&
              q_weight.is_contiguous(), "q_weight must be contiguous [128]");
  TORCH_CHECK(k_weight.dim() == 1 && k_weight.numel() == kHeadDim &&
              k_weight.is_contiguous(), "k_weight must be contiguous [128]");
  TORCH_CHECK(freqs.dim() == 6 && freqs.size(0) == 1 &&
              freqs.size(1) == q.size(1) && freqs.size(2) == 1 &&
              freqs.size(3) == kPairs && freqs.size(4) == 2 &&
              freqs.size(5) == 2 && freqs.is_contiguous(),
              "freqs must be contiguous [1,S,1,48,2,2]");
  TORCH_CHECK(epsilon == 1.0e-5,
              "only the production H3 q/k RMS epsilon 1e-5 is supported");
  TORCH_CHECK(warps_per_block == 8 || warps_per_block == 16 ||
              warps_per_block == 32,
              "warps_per_block must be 8, 16, or 32");
  TORCH_CHECK(!q.requires_grad() && !k.requires_grad() &&
              !freqs.requires_grad() && !q_weight.requires_grad() &&
              !k_weight.requires_grad(), "inference tensors must not require grad");

  const c10::cuda::CUDAGuard device_guard(device);
  torch::Tensor q_int8 = torch::empty(q.sizes(), q.options().dtype(torch::kInt8));
  // Sage lays scales out as ceil(S / BLKQ) * (BLKQ / WARPQ), not ceil(S /
  // WARPQ). For a partial final 128-token block this intentionally retains
  // all four 32-token scale slots; all-invalid slots receive the 1e-7 floor.
  const int groups = static_cast<int>(((q.size(1) + 127) / 128) * 4);
  torch::Tensor q_scale = torch::empty(
      {1, kHeads, groups}, q.options().dtype(torch::kFloat32));
  const dim3 grid(groups, kHeads, 1);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q.get_device()).stream();
#define H3_LAUNCH(WARPS)                                                        \
  qk_rms_rope_q_int8_kernel<WARPS><<<grid, WARPS * 32, 0, stream>>>(           \
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),                    \
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),                          \
      reinterpret_cast<const __nv_bfloat16*>(freqs.data_ptr()),                \
      reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),             \
      reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr()),             \
      q_int8.data_ptr<int8_t>(), q_scale.data_ptr<float>(),                     \
      static_cast<int>(q.size(1)), q.stride(1), k.stride(1), freqs.stride(1),  \
      static_cast<float>(epsilon))
  if (warps_per_block == 8) {
    H3_LAUNCH(8);
  } else if (warps_per_block == 16) {
    H3_LAUNCH(16);
  } else {
    H3_LAUNCH(32);
  }
#undef H3_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // K is mutated through its raw data pointer.  Preserve PyTorch's in-place
  // mutation semantics for aliases and saved tensors even though this path is
  // inference-only and deliberately bypasses the dispatcher.
  torch::autograd::increment_version(k);
  return {q_int8, q_scale, k};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("build_id", []() { return std::string(kBuildId); });
  module.def(
      "qk_rms_rope_q_int8",
      &qk_rms_rope_q_int8,
      "H3 Q/K RMS+RoPE with Sage per-warp Q INT8 (K mutated in place)");
}
