// Probe SM 12.1 TMA SWIZZLE_128B XOR map on a 128-byte K-row.
// Fills g[row, col] = col, TMA-loads into smem, prints phys vs linear.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

__device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, int parity) {
  const unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT;\n\t"
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

__global__ void dump_swizzle(
    const __grid_constant__ CUtensorMap map, uint8_t* out) {
  __shared__ __align__(128) uint8_t smem[128][128];
  __shared__ __align__(8) uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_arrive_expect_tx(&mbar, 128u * 128u);
    tma_load_2d(&smem[0][0], &map, 0, 0, &mbar);
  }
  __syncthreads();
  mbar_wait_parity(&mbar, 0);
  __syncthreads();
  for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
    out[i] = smem[i / 128][i % 128];
  }
}

static void check(CUresult e, const char* what) {
  if (e != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(e, &s);
    std::fprintf(stderr, "%s: %s\n", what, s ? s : "?");
    std::exit(2);
  }
}

static void check_rt(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

int main() {
  check(cuInit(0), "cuInit");
  uint8_t* g = nullptr;
  uint8_t* o = nullptr;
  check_rt(cudaMalloc(&g, 128 * 128), "malloc g");
  check_rt(cudaMalloc(&o, 128 * 128), "malloc o");
  uint8_t host[128 * 128];
  for (int r = 0; r < 128; ++r) {
    for (int c = 0; c < 128; ++c) {
      host[r * 128 + c] = static_cast<uint8_t>(c);
    }
  }
  check_rt(cudaMemcpy(g, host, 128 * 128, cudaMemcpyHostToDevice), "h2d");

  alignas(64) CUtensorMap map{};
  const cuuint64_t dim[2] = {128, 128};
  const cuuint64_t stride[1] = {128};
  const cuuint32_t box[2] = {128, 128};
  const cuuint32_t elem[2] = {1, 1};
  check(cuTensorMapEncodeTiled(
            &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, g, dim, stride, box,
            elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "encode");

  dump_swizzle<<<1, 128>>>(map, o);
  check_rt(cudaDeviceSynchronize(), "sync");
  check_rt(cudaMemcpy(host, o, 128 * 128, cudaMemcpyDeviceToHost), "d2h");

  std::printf("row xor16  pred=(row&7)<<4  match\n");
  int mismatches = 0;
  for (int r = 0; r < 16; ++r) {
    int xor_key = -1;
    bool xor_ok = true;
    for (int c = 0; c < 128; ++c) {
      const int phys = static_cast<int>(host[r * 128 + c]);
      const int key = c ^ phys;
      if (xor_key < 0) {
        xor_key = key;
      } else if (key != xor_key) {
        xor_ok = false;
      }
    }
    const int pred = (r & 7) << 4;
    std::printf(
        "%3d  0x%02x  0x%02x  %s%s\n", r, xor_ok ? xor_key : 0xff, pred,
        xor_ok ? "xor" : "NOT_XOR", xor_ok && xor_key == pred ? " MATCH" : "");
    if (!xor_ok || xor_key != pred) {
      ++mismatches;
    }
  }
  // Full 128-row check of hypothesized map.
  int bad = 0;
  for (int r = 0; r < 128; ++r) {
    const int pred = (r & 7) << 4;
    for (int c = 0; c < 128; ++c) {
      if (static_cast<int>(host[r * 128 + c]) != (c ^ pred)) {
        ++bad;
      }
    }
  }
  std::printf("full_128x128_pred_mismatches %d\n", bad);
  // Also print 16B-chunk permutation for row 0..7.
  for (int r = 0; r < 8; ++r) {
    std::printf("row %d chunks:", r);
    for (int k = 0; k < 8; ++k) {
      std::printf(" %d", static_cast<int>(host[r * 128 + k * 16]));
    }
    std::printf("\n");
  }
  return bad == 0 ? 0 : 1;
}
