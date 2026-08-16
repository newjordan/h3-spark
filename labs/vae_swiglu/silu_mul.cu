// Byte-exact VAE SwiGLU: out = round(silu(round? gate)) * up
// Matches Comfy eager F.silu(gate).mul_(up) two-rounding association on
// BF16/FP16 (SiLU stored, then product stored). FP32 is one mul.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

__device__ __forceinline__ __nv_bfloat16 silu_mul_bf16(
    __nv_bfloat16 gate_bf16, __nv_bfloat16 up_bf16) {
  const float gate = __bfloat162float(gate_bf16);
  const float up = __bfloat162float(up_bf16);
  const float silu = gate / (1.0f + expf(-gate));
  const float silu_b = __bfloat162float(__float2bfloat16_rn(silu));
  return __float2bfloat16_rn(silu_b * up);
}

__device__ __forceinline__ __half silu_mul_fp16(__half gate_h, __half up_h) {
  const float gate = __half2float(gate_h);
  const float up = __half2float(up_h);
  const float silu = gate / (1.0f + expf(-gate));
  const float silu_h = __half2float(__float2half_rn(silu));
  return __float2half_rn(silu_h * up);
}

template <typename T>
__global__ void silu_mul_kernel(
    const T* __restrict__ raw, T* __restrict__ out,
    int64_t rows, int64_t inner) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = rows * inner;
  if (idx >= n) {
    return;
  }
  const int64_t row = idx / inner;
  const int64_t col = idx - row * inner;
  const T* rowp = raw + row * (inner * 2);
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    out[idx] = silu_mul_bf16(rowp[col], rowp[inner + col]);
  } else if constexpr (std::is_same<T, __half>::value) {
    out[idx] = silu_mul_fp16(rowp[col], rowp[inner + col]);
  } else {
    const float g = rowp[col];
    const float u = rowp[inner + col];
    out[idx] = (g / (1.0f + expf(-g))) * u;
  }
}

template <typename T>
int launch(const void* raw, void* out, long long rows, long long inner,
           void* stream) {
  if (raw == nullptr || out == nullptr || rows <= 0 || inner <= 0) {
    return 1;
  }
  const int64_t n = static_cast<int64_t>(rows) * static_cast<int64_t>(inner);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  auto s = static_cast<cudaStream_t>(stream);
  silu_mul_kernel<T><<<blocks, threads, 0, s>>>(
      static_cast<const T*>(raw), static_cast<T*>(out), rows, inner);
  return 0;
}

}  // namespace

extern "C" {

int h3_silu_mul_bf16(const void* raw, void* out, long long rows,
                     long long inner, void* stream) {
  return launch<__nv_bfloat16>(raw, out, rows, inner, stream);
}

int h3_silu_mul_fp16(const void* raw, void* out, long long rows,
                     long long inner, void* stream) {
  return launch<__half>(raw, out, rows, inner, stream);
}

int h3_silu_mul_fp32(const void* raw, void* out, long long rows,
                     long long inner, void* stream) {
  return launch<float>(raw, out, rows, inner, stream);
}

}  // extern "C"
