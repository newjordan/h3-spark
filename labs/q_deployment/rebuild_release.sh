#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PATCHED_SAGE_ROOT PATCHED_Q_ROOT PATCHED_CONSUMER_ROOT" >&2
  exit 2
fi

sage_root=$(realpath "$1")
q_root=$(realpath "$2")
consumer_root=$(realpath "$3")
h3_python=${H3_PYTHON:?set H3_PYTHON to the target ComfyUI Python}
h3_jobs=${H3_BUILD_JOBS:-4}

readarray -t runtime < <("$h3_python" - <<'PY'
import pathlib
import torch
print(torch.__version__)
print(torch.version.git_version)
print(torch.version.cuda)
print(torch.utils.cmake_prefix_path)
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)

[[ ${runtime[0]} == 2.13.0+cu130 ]]
[[ ${runtime[1]} == cf30153c4c131c8164ee7798e5022d810682e2cb ]]
[[ ${runtime[2]} == 13.0 ]]
torch_cmake=${runtime[3]}
torch_lib=${runtime[4]}

cmake -S "$q_root" -B "$q_root/build-h3-release" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$h3_python" \
  -DCMAKE_PREFIX_PATH="$torch_cmake" \
  -DH3_TORCH_CUDA_ARCH_LIST=12.0
cmake --build "$q_root/build-h3-release" --parallel "$h3_jobs"

cmake -S "$consumer_root" -B "$consumer_root/build-h3-release" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$h3_python" \
  -DCMAKE_PREFIX_PATH="$torch_cmake" \
  -DH3_TORCH_CUDA_ARCH_LIST=12.0
cmake --build "$consumer_root/build-h3-release" --parallel "$h3_jobs"

(
  cd "$sage_root"
  env \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS="$h3_jobs" \
    LDFLAGS="-Wl,-rpath,$torch_lib" \
    "$h3_python" setup.py build_ext --inplace --force
)
