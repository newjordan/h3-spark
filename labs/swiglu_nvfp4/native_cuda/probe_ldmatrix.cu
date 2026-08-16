// Probe: can ldmatrix.x4 emit the kitchen m16n8k64 A fragment
// (group/tidg 4x u32) from a 16x32 packed-K smem tile?

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ void ldmatrix_x4(uint32_t& d0, uint32_t& d1,
                                            uint32_t& d2, uint32_t& d3,
                                            const void* addr) {
  const unsigned s = __cvta_generic_to_shared(addr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& d0, uint32_t& d1,
                                            const void* addr) {
  const unsigned s = __cvta_generic_to_shared(addr);
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(d0), "=r"(d1)
      : "r"(s));
}

__global__ void probe(int* mismatches, int formula) {
  __shared__ __align__(16) uint8_t smem[16][32];
  const int lane = threadIdx.x;
  for (int i = lane; i < 16 * 32; i += 32) {
    smem[i / 32][i % 32] = static_cast<uint8_t>(i);
  }
  __syncthreads();

  const int group = lane >> 2;
  const int tidg = lane & 3;
  const int packed_k0 = tidg * 4;
  const uint32_t a0 =
      *reinterpret_cast<const uint32_t*>(&smem[group][packed_k0]);
  const uint32_t a1 =
      *reinterpret_cast<const uint32_t*>(&smem[group + 8][packed_k0]);
  const uint32_t a2 =
      *reinterpret_cast<const uint32_t*>(&smem[group][packed_k0 + 16]);
  const uint32_t a3 =
      *reinterpret_cast<const uint32_t*>(&smem[group + 8][packed_k0 + 16]);

  uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  if (formula == 0) {
    // Standard SM80 16B-per-(lane%8) address.
    ldmatrix_x4(d0, d1, d2, d3, &smem[lane & 7][(lane >> 3) * 16]);
  } else if (formula == 1) {
    // Row = lane%16, 16B at col 0 or 16 (1-warp uint4 map).
    ldmatrix_x4(d0, d1, d2, d3, &smem[lane & 15][(lane >> 4) * 16]);
  } else if (formula == 2) {
    // Kitchen fragment addresses themselves.
    ldmatrix_x4(d0, d1, d2, d3, &smem[group][packed_k0]);
  } else {
    // 8-row pair: even lanes row, odd row+8.
    ldmatrix_x4(d0, d1, d2, d3, &smem[(lane & 7) + 8 * (lane & 1)]
                                    [((lane >> 3) & 1) * 16]);
  }

  const int bad = (d0 != a0) || (d1 != a1) || (d2 != a2) || (d3 != a3);
  atomicAdd(mismatches, bad);
}

__global__ void probe_b(int* mismatches) {
  __shared__ __align__(16) uint8_t smem[8][32];
  const int lane = threadIdx.x;
  for (int i = lane; i < 8 * 32; i += 32) {
    smem[i / 32][i % 32] = static_cast<uint8_t>(i + 1);
  }
  __syncthreads();
  const int group = lane >> 2;
  const int tidg = lane & 3;
  const int packed_k0 = tidg * 4;
  const uint32_t b0 =
      *reinterpret_cast<const uint32_t*>(&smem[group][packed_k0]);
  const uint32_t b1 =
      *reinterpret_cast<const uint32_t*>(&smem[group][packed_k0 + 16]);
  uint32_t d0 = 0, d1 = 0;
  ldmatrix_x2(d0, d1, &smem[lane & 7][(lane >> 3) * 8]);
  atomicAdd(mismatches, (d0 != b0) || (d1 != b1));
}

__global__ void probe_b2(int* mismatches) {
  __shared__ __align__(16) uint8_t smem[8][32];
  const int lane = threadIdx.x;
  for (int i = lane; i < 8 * 32; i += 32) {
    smem[i / 32][i % 32] = static_cast<uint8_t>(i + 1);
  }
  __syncthreads();
  const int group = lane >> 2;
  const int tidg = lane & 3;
  const int packed_k0 = tidg * 4;
  const uint32_t b0 =
      *reinterpret_cast<const uint32_t*>(&smem[group][packed_k0]);
  const uint32_t b1 =
      *reinterpret_cast<const uint32_t*>(&smem[group][packed_k0 + 16]);
  uint32_t d0 = 0, d1 = 0;
  ldmatrix_x2(d0, d1, &smem[lane & 7][(lane >> 4) * 16]);
  atomicAdd(mismatches, (d0 != b0) || (d1 != b1));
}

int main() {
  int* d = nullptr;
  cudaMalloc(&d, 4);
  for (int f = 0; f < 4; ++f) {
    cudaMemset(d, 0, 4);
    probe<<<1, 32>>>(d, f);
    int h = -1;
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaDeviceSynchronize();
    std::printf("formula %d mismatches %d cuda %s\n", f, h,
                cudaGetErrorString(err));
  }
  cudaMemset(d, 0, 4);
  probe_b<<<1, 32>>>(d);
  int h = -1;
  cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaDeviceSynchronize();
  std::printf("B x2 addr (lane&7,(lane>>3)*8) mismatches %d cuda %s\n", h,
              cudaGetErrorString(err));
  cudaMemset(d, 0, 4);
  probe_b2<<<1, 32>>>(d);
  cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
  err = cudaDeviceSynchronize();
  std::printf("B x2 addr (lane&7,(lane>>4)*16) mismatches %d cuda %s\n", h,
              cudaGetErrorString(err));
  return 0;
}
