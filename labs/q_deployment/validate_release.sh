#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 PATCHED_SAGE_ROOT PATCHED_Q_ROOT PATCHED_CONSUMER_ROOT Q_BINARY CONSUMER_BINARY QUALIFIED_Q QUALIFIED_CONSUMER" >&2
  exit 2
fi

bundle_root=$(cd "$(dirname "$0")" && pwd)
sage_root=$(realpath "$1")
q_root=$(realpath "$2")
consumer_root=$(realpath "$3")
q_binary=$(realpath "$4")
consumer_binary=$(realpath "$5")
qualified_q=$(realpath "$6")
qualified_consumer=$(realpath "$7")
h3_python=${H3_PYTHON:?set H3_PYTHON to the target ComfyUI Python}
torch_lib=$($h3_python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')

(cd "$bundle_root" && sha256sum -c PATCH_MANIFEST.sha256)
(cd "$sage_root" && sha256sum -c "$bundle_root/SAGE_SOURCE_MANIFEST.sha256")
(cd "$q_root" && sha256sum -c "$bundle_root/Q_SOURCE_MANIFEST.sha256")
(cd "$consumer_root" && sha256sum -c "$bundle_root/CONSUMER_SOURCE_MANIFEST.sha256")

check_sha() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [[ $actual == "$expected" ]] || {
    echo "SHA mismatch for $path: expected $expected, got $actual" >&2
    exit 1
  }
}

check_sha b97b30c30222276d490d2a3d4cf2fff1081d7ff3a66d8e01395589feccd2da1f "$q_binary"
check_sha 45b21fe1957e535a860fcd28014b6b194666a29b5b98f4672c7e3b99230edb8f "$consumer_binary"
check_sha 26561e557be0c501c2a4a1e7fb733db8da378bb55d7a92c5dc0136d37eee55df "$qualified_q"
check_sha 8b24f011cba209994c9e8e426b2432f60c472023da83a26e425a48607f543e7b "$qualified_consumer"

"$h3_python" - <<'PY'
import torch
assert torch.__version__ == "2.13.0+cu130"
assert torch.version.git_version == "cf30153c4c131c8164ee7798e5022d810682e2cb"
assert torch.version.cuda == "13.0"
print("runtime", torch.__version__, torch.version.git_version, torch.version.cuda)
PY

PYTHONPATH="$sage_root" "$h3_python" - <<'PY'
from sageattention import _fused
from sageattention import sm89_compile
assert _fused.h3_build_id() == "h3-sage-fused-current-stream-v1-torch2.13-sm120"
assert sm89_compile._qattn_sm89.h3_build_id() == "h3-sage-qattn-output-amax-v2-torch2.13-sm120"
assert callable(sm89_compile._qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_with_output_amax)
print("sage IDs OK")
PY

"$h3_python" - "$q_binary" "$consumer_binary" <<'PY'
import importlib.util
import pathlib
import sys

def load(name, value):
    path = pathlib.Path(value)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

q = load("h3_q_rms_rope_int8_native_v1", sys.argv[1])
assert q.build_id() == "h3-q-rms-rope96-qint8-v3-torch2.13-sm120"
assert callable(q.qk_rms_rope_q_int8)

c = load("h3_swiglu_nvfp4_native_v1", sys.argv[2])
assert c.h3_supplied_amax_build_id() == "h3-nvfp4-supplied-amax-v1-torch2.13-sm120"
for name in (
    "bf16_nvfp4", "bf16_nvfp4_dynamic", "bf16_nvfp4_from_amax_bits",
    "swiglu_nvfp4", "swiglu_nvfp4_dynamic",
):
    assert callable(getattr(c, name, None)), name
print("Q and consumer IDs/entrypoints OK")
PY

sage_fused=$(find "$sage_root/sageattention" -maxdepth 1 -type f -name '_fused*.so' -print -quit)
sage_qattn=$(find "$sage_root/sageattention" -maxdepth 1 -type f -name '_qattn_sm89*.so' -print -quit)
[[ -n $sage_fused && -n $sage_qattn ]]

for binary in "$q_binary" "$consumer_binary" "$sage_fused" "$sage_qattn"; do
  cuobjdump --list-elf "$binary" | grep -q 'sm_120.cubin'
  readelf -d "$binary" | grep -F "$torch_lib" >/dev/null
done

grep -q 'CMAKE_BUILD_TYPE:STRING=Release' "$(dirname "$q_binary")/CMakeCache.txt"
grep -q 'CMAKE_BUILD_TYPE:STRING=Release' "$(dirname "$consumer_binary")/CMakeCache.txt"
grep -q -- '-DNDEBUG' "$(dirname "$q_binary")/compile_commands.json"
grep -q -- '-DNDEBUG' "$(dirname "$consumer_binary")/compile_commands.json"

[[ $(grep -c 'torch::autograd::increment_version(k);' "$q_root/q_rms_rope_int8_cuda.cu") -eq 1 ]]
launch_line=$(grep -n 'C10_CUDA_KERNEL_LAUNCH_CHECK();' "$q_root/q_rms_rope_int8_cuda.cu" | tail -1 | cut -d: -f1)
version_line=$(grep -n 'torch::autograd::increment_version(k);' "$q_root/q_rms_rope_int8_cuda.cu" | cut -d: -f1)
[[ $launch_line -lt $version_line ]]
grep -q 'q.size(1) > 1024' "$q_root/q_rms_rope_int8_cuda.cu"

scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
normalize_sass() {
  cuobjdump --dump-sass "$1" | sed -E '/Function :/d; /code for sm_/d; /Fatbin elf code/d' > "$2"
}
compare_sass() {
  local candidate=$1
  local baseline=$2
  local label=$3
  normalize_sass "$candidate" "$scratch/${label}.candidate.sass"
  normalize_sass "$baseline" "$scratch/${label}.baseline.sass"
  cmp "$scratch/${label}.candidate.sass" "$scratch/${label}.baseline.sass"
  sha256sum "$scratch/${label}.candidate.sass" "$scratch/${label}.baseline.sass"
}

compare_sass "$q_binary" "$qualified_q" q
compare_sass "$consumer_binary" "$qualified_consumer" consumer
{
  echo "# UNQUALIFIED rebuilt Sage artifacts; source/ABI/cubin/RUNPATH checked, CUDA parity still required"
  sha256sum "$sage_fused" "$sage_qattn"
} > "$bundle_root/REBUILT_SAGE_RECEIPT.sha256"

echo "H3 Q/consumer release validation: PASS"
echo "Sage source/import validation: PASS; CUDA promotion: HOLD"
