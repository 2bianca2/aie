#!/usr/bin/env bash
# [host] Run the pipeline-dump debug tool inside the dev container.
# Activates the venv + PYTHONPATH, then runs scripts/debug/pipeline_dump.py "$@".
# NPU passthrough + repo mounted at /workspace (same as run-dev.sh), non-interactive.
#   e.g. ./scripts/docker/run-debug.sh \
#          --model models/mlp_2layer/mlp_2layer.onnx --function mlp_2layer \
#          --input x.npy --input w1.npy --input w2.npy --label t1
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
root="$(git -C "$here" rev-parse --show-toplevel)"

args=(--rm --user "$(id -u):$(id -g)")
npu="$(npu_device)"
[ -n "$npu" ] && args+=(--device="/dev/accel/$npu")
args+=(-v "$root:/workspace" -w /workspace -e HOME=/workspace -e PEANO_INSTALL_DIR=/workspace/llvm-aie)

exec docker run "${args[@]}" "$IMAGE_DEV" bash -lc '
  source /opt/venv/bin/activate
  export PYTHONPATH=/workspace/build/compiler/bindings/python
  exec python3 scripts/debug/pipeline_dump.py "$@"
' _ "$@"
