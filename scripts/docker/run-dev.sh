#!/usr/bin/env bash
# [host] Run the dev container: host uid/gid, NPU passthrough, repo mounted at /workspace.
# Extra args run as the command (default: bash).  e.g. ./run-dev.sh ./scripts/build/build.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
root="$(git -C "$here" rev-parse --show-toplevel)"

args=(--rm -it --user "$(id -u):$(id -g)")
npu="$(npu_device)"
[ -n "$npu" ] && args+=(--device="/dev/accel/$npu")
args+=(-v "$root:/workspace" -w /workspace -e HOME=/workspace -e PEANO_INSTALL_DIR=/workspace/llvm-aie)

exec docker run "${args[@]}" "$IMAGE_DEV" "${@:-bash}"
