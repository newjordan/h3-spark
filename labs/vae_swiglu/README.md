# VAE SwiGLU `silu(gate)*up`

Byte-exact native replacement for `F.silu(gate).mul_(up)` on the VAE
MLP last dimension. Association is eager two-rounding on BF16/FP16.

- Source: `silu_mul.cu`
- Serving control: `COMFY_MINIMAX_H3_VAE_SILU_MUL=auto` (rollback `off`)
- Library path: `COMFY_MINIMAX_H3_VAE_SILU_MUL_LIBRARY` or
  `silu_mul_native_v1.so` next to `vae.py`
- Generated `.so` is not committed
