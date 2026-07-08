#!/usr/bin/env bash
# [container] cmake configure for IREE + amd-aie (dev flags: frontends ON, assertions ON, bindings ON).
# Run from /workspace. Extra args pass through to cmake.
set -euo pipefail
cmake -B build -S third_party/iree -G Ninja \
  -DIREE_CMAKE_PLUGIN_PATHS="$PWD" -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_INPUT_TORCH=ON -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TOSA=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
  -DIREE_BUILD_TESTS=ON -DIREE_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  "$@"
