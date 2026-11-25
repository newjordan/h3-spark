#!/usr/bin/env bash
# Configure and build every labs/**/CMakeLists.txt project into its build/.
# Requires H3_PYTHON (the interpreter that imports torch). Does not download
# weights and does not launch a render.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" == "--list" ]]; then
  find "$REPO_DIR/labs" -name CMakeLists.txt \
    -not -path '*/build/*' \
    -not -path '*/cmake-build*/*' \
    -printf '%h\n' | sed "s|^$REPO_DIR/||" | sort
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--list]" >&2
  exit 2
fi

if [[ -z "${H3_PYTHON:-}" ]]; then
  echo "error: set H3_PYTHON to the interpreter that has torch" >&2
  exit 2
fi
if [[ ! -x "$H3_PYTHON" ]]; then
  echo "error: H3_PYTHON is not executable: $H3_PYTHON" >&2
  exit 2
fi

torch_cmake="$("$H3_PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')"

mapfile -t projects < <(
  find "$REPO_DIR/labs" -name CMakeLists.txt \
    -not -path '*/build/*' \
    -not -path '*/cmake-build*/*' \
    -printf '%h\n' | sort
)

if [[ ${#projects[@]} -eq 0 ]]; then
  echo "error: no labs/**/CMakeLists.txt projects found" >&2
  exit 2
fi

for src in "${projects[@]}"; do
  cmake_args=(
    -S "$src"
    -B "$src/build"
    -DCMAKE_BUILD_TYPE=Release
    -DPython_EXECUTABLE="$H3_PYTHON"
    -DCMAKE_PREFIX_PATH="$torch_cmake"
  )
  if [[ -n "${H3_TORCH_CUDA_ARCH_LIST:-}" ]]; then
    cmake_args+=(-DH3_TORCH_CUDA_ARCH_LIST="$H3_TORCH_CUDA_ARCH_LIST")
  fi
  echo "configuring ${src#"$REPO_DIR"/}"
  cmake "${cmake_args[@]}"
  cmake --build "$src/build" --parallel
done

echo "built ${#projects[@]} native projects"
