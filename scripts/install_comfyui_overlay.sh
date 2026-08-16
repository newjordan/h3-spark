#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_REVISION="6db4fa2fcd38a92d4e0364c917cd8503620e2a74"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly OVERLAY_DIR="$REPO_DIR/integration/comfyui"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/clean/ComfyUI" >&2
  exit 2
fi

target_dir="$(cd -- "$1" && pwd)"
if [[ ! -d "$target_dir/.git" ]]; then
  echo "target is not a Git checkout: $target_dir" >&2
  exit 2
fi

actual_revision="$(git -C "$target_dir" rev-parse HEAD)"
if [[ "$actual_revision" != "$EXPECTED_REVISION" ]]; then
  echo "expected ComfyUI $EXPECTED_REVISION, found $actual_revision" >&2
  exit 2
fi

if ! git -C "$target_dir" diff --quiet || ! git -C "$target_dir" diff --cached --quiet; then
  echo "target ComfyUI worktree is dirty; preserve or commit it first" >&2
  exit 2
fi

while IFS= read -r -d '' source_file; do
  relative_path="${source_file#"$OVERLAY_DIR/"}"
  [[ "$relative_path" == "README.md" ]] && continue
  destination="$target_dir/$relative_path"
  install -D -m 0644 -- "$source_file" "$destination"
done < <(find "$OVERLAY_DIR" -type f -print0)

echo "Installed H3 Spark source overlay into $target_dir"
echo "Build/install the promoted native libraries before enabling auto fusion."
