#!/usr/bin/env bash
# [host] Build the deploy (runtime) image from a pinned commit, then save it as tar.gz.
# Usage: build-deploy.sh <배포 커밋 SHA>
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
commit="${1:?사용법: build-deploy.sh <배포 커밋 SHA>}"

docker build --target runtime \
  --build-arg IREE_AMD_AIE_COMMIT="$commit" \
  --build-arg BUILD_JOBS="$JOBS" \
  -t "$IMAGE_DEPLOY" "$REPO_URL#$BRANCH"

docker save "$IMAGE_DEPLOY" | gzip > "$DEPLOY_TAR"
echo "배포 파일 생성: $DEPLOY_TAR"
