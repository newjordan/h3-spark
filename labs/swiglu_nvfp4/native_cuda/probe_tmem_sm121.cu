// Probe: does SM 12.1 accept tcgen05 TMEM alloc?
// Kitchen-legal NVFP4 on this chip is mma.m16n8k64 (registers), not TMEM.

#include <cstdint>

__global__ void probe_tcgen05_alloc() {
  __shared__ uint32_t dst;
  const unsigned addr = __cvta_generic_to_shared(&dst);
  const unsigned n_cols = 32;
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::
      "r"(addr), "r"(n_cols)
      : "memory");
}
