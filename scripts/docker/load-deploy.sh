#!/usr/bin/env bash
# [host, 받는 측] Load a deploy image from tar.gz.  Usage: load-deploy.sh [tar.gz]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
tar="${1:-$DEPLOY_TAR}"
gunzip -c "$tar" | docker load
