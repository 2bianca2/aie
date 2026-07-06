#!/usr/bin/env bash
# [host, 받는 측] Run the deploy container with NPU passthrough. Extra args = command (default bash).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"

args=(--rm -it)
npu="$(npu_device)"
[ -n "$npu" ] && args+=(--device="/dev/accel/$npu")

exec docker run "${args[@]}" "$IMAGE_DEPLOY" "${@:-bash}"
