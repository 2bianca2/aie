#!/usr/bin/env bash
# [container] Incremental build with adaptive -j (bounded by cores AND RAM). Run from /workspace.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
exec cmake --build build -j "$JOBS" "$@"
