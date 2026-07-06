#!/usr/bin/env bash
# [host] Build the dev image. Runnable from any directory.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
root="$(git -C "$here" rev-parse --show-toplevel)"   # docker build context (repo root, has Dockerfile)
exec docker build --target dev -t "$IMAGE_DEV" "$root"
