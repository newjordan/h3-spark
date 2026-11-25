#!/usr/bin/env bash
# Install integration/comfyui into a clean ComfyUI checkout of the qualified
# revision. Tests and this overlay's README stay in the h3-spark tree.
set -euo pipefail

readonly EXPECTED_REVISION="6db4fa2fcd38a92d4e0364c917cd8503620e2a74"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly OVERLAY_DIR="$REPO_DIR/integration/comfyui"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/clean/ComfyUI" >&2
  echo "Installs the H3 overlay into a clean checkout of ComfyUI ${EXPECTED_REVISION}." >&2
  echo "Refuses a dirty worktree. Repository tests are not copied." >&2
  exit 2
fi

if [[ ! -d "$1" ]]; then
  echo "target directory does not exist: $1" >&2
  exit 2
fi

target_dir="$(cd -- "$1" && pwd)"
if [[ ! -d "$target_dir/.git" ]]; then
  echo "target is not a Git checkout: $target_dir" >&2
  exit 2
fi

expected_revision="${H3_COMFYUI_REVISION:-$EXPECTED_REVISION}"
if [[ -n "${H3_COMFYUI_REVISION:-}" && "$H3_COMFYUI_REVISION" != "$EXPECTED_REVISION" ]]; then
  echo "warning: H3_COMFYUI_REVISION overrides the qualified ComfyUI pin ${EXPECTED_REVISION}" >&2
fi

actual_revision="$(git -C "$target_dir" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "expected ComfyUI $expected_revision, found $actual_revision" >&2
  exit 2
fi

if ! git -C "$target_dir" diff --quiet || ! git -C "$target_dir" diff --cached --quiet; then
  echo "target ComfyUI worktree is dirty; preserve or commit it first" >&2
  exit 2
fi

installed=0
while IFS= read -r -d '' source_file; do
  relative_path="${source_file#"$OVERLAY_DIR/"}"
  case "$relative_path" in
    README.md|tests|tests/*) continue ;;
  esac
  destination="$target_dir/$relative_path"
  install -D -m 0644 -- "$source_file" "$destination"
  installed=$((installed + 1))
done < <(find "$OVERLAY_DIR" -type f -print0)

echo "Installed ${installed} H3 overlay files into $target_dir"
echo "Tests remain in ${OVERLAY_DIR}/tests and were not copied."
echo "Source config/h3_spark_serving.env, then build native libraries with scripts/build_native.sh before enabling auto fusion."
