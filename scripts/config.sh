#!/usr/bin/env bash
# Shared config, sourced by the other scripts. Every value is overridable via env var.
# (No `set -e` here — this file is sourced.)

IMAGE_DEV="${IMAGE_DEV:-iree-amd-aie:dev}"
IMAGE_DEPLOY="${IMAGE_DEPLOY:-iree-amd-aie:deploy}"
REPO_URL="${REPO_URL:-https://github.com/ace-knu/iree-amd-aie.git}"
BRANCH="${BRANCH:-dev}"
DEPLOY_TAR="${DEPLOY_TAR:-iree-amd-aie-deploy.tar.gz}"

# Adaptive build parallelism: leave 2 cores for the OS (avoids full-core freeze).
JOBS="${JOBS:-$(nproc --ignore=2)}"

# First NPU accel device name (e.g. accel0); empty if none present.
npu_device() { ls /dev/accel/ 2>/dev/null | head -1; }
