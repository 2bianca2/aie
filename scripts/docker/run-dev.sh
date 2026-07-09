#!/usr/bin/env bash
# [host] Run the dev container: host uid/gid, NPU passthrough, repo mounted at /workspace.
# Extra args run as the command (default: bash).  e.g. ./run-dev.sh ./scripts/build/build.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
root="$(git -C "$here" rev-parse --show-toplevel)"

args=(--rm -it --user "$(id -u):$(id -g)")
# resolve host uid/gid to names inside the container (avoids "I have no name!" /
# "groups: cannot find name" when the host uid/gid isn't the image's ubuntu 1000)
args+=(-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro)
npu="$(npu_device)"
if [ -n "$npu" ]; then
  args+=(--device="/dev/accel/$npu")
  # add the device's owning group so the non-root container user can open it
  # even when the node is 0660 root:render (not just the amdxdna-udev 0666 case)
  gid="$(device_gid "$npu")"
  [ -n "$gid" ] && args+=(--group-add "$gid")
fi
args+=(-v "$root:/workspace" -w /workspace -e HOME=/workspace -e PEANO_INSTALL_DIR=/workspace/llvm-aie)

exec docker run "${args[@]}" "$IMAGE_DEV" "${@:-bash}"
