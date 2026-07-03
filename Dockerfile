# iree-amd-aie dev/deploy container (Option B, multi-stage)
# Phase 1 (this iteration): base-deps + dev only.
# Phase 2 (later): builder + runtime for a self-contained deploy image.
ARG UBUNTU_BASE=ubuntu:24.04

# ============================================================
# base-deps : common build environment (no source baked in)
#   Rationale for the dependency set:
#     - cmake: IREE requires 3.26...3.29; Ubuntu 24.04 apt cmake (3.28.3) satisfies it.
#     - python3 + python3-numpy: IREE_BUILD_PYTHON_BINDINGS=ON needs NumPy headers
#       (FindPython3 NumPy component). apt python3-numpy avoids PEP 668 pip issues.
#     - libudev-dev / uuid-dev: build deps for third_party/XRT (amdxdna SHIM),
#       which is vendored as a submodule -> no separate amd/xdna-driver clone needed.
#     - ccache: optional (speeds up rebuilds). Remove it together with the
#       CMAKE_*_COMPILER_LAUNCHER=ccache flags if a truly minimal image is wanted.
# ============================================================
FROM ${UBUNTU_BASE} AS base-deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git \
      build-essential cmake ninja-build clang lld \
      libudev-dev uuid-dev \
      python3 python3-dev python3-venv python3-pip python3-numpy \
      ccache \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# dev : development environment (source is bind-mounted, not baked)
#   Source + submodules are prepared on the HOST (recursive clone) and mounted
#   at /workspace. The container only builds/runs.
# ============================================================
FROM base-deps AS dev
# ubuntu:24.04 ships a default 'ubuntu' user at uid/gid 1000. Reuse it so files
# created through the bind-mount are owned by the host user (typically uid 1000),
# not root. VS Code Dev Containers remaps this uid to the host's automatically
# (updateRemoteUserUID defaults to true); for plain `docker run` on a host whose
# uid is not 1000, pass `--user $(id -u):$(id -g)`.
USER ubuntu
WORKDIR /workspace

# ============================================================
# builder / runtime : deploy image  <- Phase 2 (not authored yet)
#   builder (FROM base-deps): clone fork at a pinned commit ->
#     bash build_tools/download_peano.sh ->
#     cmake build with -DCMAKE_INSTALL_PREFIX=/opt/iree-amd-aie -> install target.
#   runtime (FROM ${UBUNTU_BASE}): install only runtime shared libs, then
#     COPY --from=builder the install output (/opt/iree-amd-aie, llvm-aie, XRT libs).
#     No source, no toolchain, no mount -> recipient just runs.
# ============================================================
