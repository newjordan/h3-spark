#!/usr/bin/env python3
"""Fail-closed resource identity: a second 256x64 FP32 acc cannot live
on GB10 in smem, registers, a 2-CTA cluster, or TMEM.

One A stream feeding two N-halves needs acc0+acc1. Prior tick: smem
opt-in is 99 KiB < 128 KiB acc1. This tick closes the other homes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch


THREADS = 256
ACC0_FLOATS = 2 * 8 * 4 * 2  # 2 strips x 8 n-subs x 4 frag x 2 arms
ACC1_FLOATS = ACC0_FLOATS
K2_SMEM = 86040
ACC1_BYTES = THREADS * ACC1_FLOATS * 4


def device_attr(code: int) -> int:
    import ctypes

    cudart = ctypes.CDLL("libcudart.so")
    val = ctypes.c_int()
    err = cudart.cudaDeviceGetAttribute(ctypes.byref(val), code, 0)
    if err != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute({code}) err={err}")
    return int(val.value)


def probe_tmem() -> dict:
    src = Path(__file__).with_name("probe_tmem_sm121.cu")
    out = Path("/tmp/probe_tmem_sm121.o")
    cmd = [
        "/usr/local/cuda/bin/nvcc",
        "-c",
        "-arch=sm_121a",
        "-o",
        str(out),
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stderr or "") + (proc.stdout or "")
    return {
        "nvcc_cmd": " ".join(cmd),
        "nvcc_returncode": proc.returncode,
        "tmem_accepted": proc.returncode == 0,
        "nvcc_log_tail": "\n".join(log.strip().splitlines()[-12:]),
    }


def main() -> int:
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    optin = int(getattr(props, "shared_memory_per_block_optin", 0))
    regs_sm = int(props.regs_per_multiprocessor)
    cluster_launch = device_attr(119)
    max_blocks_cluster = device_attr(120)
    tmem = probe_tmem()
    acc0_regs = THREADS * ACC0_FLOATS
    dual_regs = THREADS * (ACC0_FLOATS + ACC1_FLOATS)
    payload = {
        "identity": (
            "A second 256x64 FP32 acc has no home on GB10: smem opt-in "
            "99 KiB < 128 KiB acc1; 256 threads x 256 floats fills the "
            "64K register file; MaxBlocksPerCluster is 1 so A cannot be "
            "multicast to a partner CTA; tcgen05 TMEM is SM_100a/101a, "
            "not sm_121a."
        ),
        "device": torch.cuda.get_device_name(device),
        "sm": f"{props.major}.{props.minor}",
        "sm_count": int(props.multi_processor_count),
        "smem_optin": optin,
        "acc1_bytes": ACC1_BYTES,
        "k2_smem_bytes": K2_SMEM,
        "acc1_exceeds_optin": ACC1_BYTES > optin,
        "regs_per_sm": regs_sm,
        "acc0_regs": acc0_regs,
        "acc0_plus_acc1_regs": dual_regs,
        "dual_acc_fills_regfile": dual_regs >= regs_sm,
        "cluster_launch": cluster_launch,
        "max_blocks_per_cluster": max_blocks_cluster,
        "cluster_cannot_pair_n_halves": max_blocks_cluster < 2,
        **tmem,
    }
    payload["pass"] = (
        payload["acc1_exceeds_optin"]
        and payload["dual_acc_fills_regfile"]
        and payload["cluster_cannot_pair_n_halves"]
        and not payload["tmem_accepted"]
        and optin == 101376
        and regs_sm == 65536
        and max_blocks_cluster == 1
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_fc1_nvfp4_acc_resources_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
