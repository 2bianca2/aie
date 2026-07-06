# iree-amd-aie 컨테이너 사용자 가이드

따라만 하면 개발/배포 환경이 구축되는 최소 명령 모음이다. 배경·이유·문제해결 등 자세한 내용은
[`HOST_PREREQUISITES.md`](HOST_PREREQUISITES.md) · [`DEV_CONTAINER.md`](DEV_CONTAINER.md) ·
[`RUN.md`](RUN.md) 참고.

- 대상: Ubuntu 24.04 + AMD NPU(amdxdna) 호스트
- 구조: 소스/빌드 산출물은 호스트, 컨테이너는 빌드/실행 환경만 제공(bind-mount)

---

## Part 1. Prerequisites (호스트 준비)

### 1-1. 확인 (다 있으면 Part 2로)
```bash
. /etc/os-release; echo "$PRETTY_NAME"   # Ubuntu 24.04 계열
uname -r                                  # 커널 (6.11+/6.14+ 권장)
lsmod | grep amdxdna                      # NPU 커널 드라이버 로드됨
modinfo amdxdna | grep ^version           # 로드된 KMD 버전 (SHIM과 호환 계열인지 판단용)
ls -l /dev/accel/                         # NPU 디바이스 노드(accelN) 존재
ls /usr/lib/firmware/amdnpu               # NPU 펌웨어 존재
docker --version                          # Docker 설치·동작
```

### 1-2. 없을 때 설치
```bash
# Docker (한 줄 설치 후 재로그인)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

# NPU 드라이버/펌웨어 (amdxdna가 없거나 SHIM과 다른 버전 계열일 때 설치)
#   컨테이너 SHIM은 git(third_party/XRT)에 고정 → 호환되는 드라이버 커밋은 iree-amd-aie repo의
#   README(Dependencies > Driver)가 지정한다. 그 커밋으로 KMD+펌웨어+XRT를 빌드·설치:
#     git clone https://github.com/amd/xdna-driver.git
#     cd xdna-driver && git checkout <README가 지정한 커밋> && git submodule update --init --recursive
#     # 이후 빌드·설치는 xdna-driver repo 절차를 따름 (DKMS로 설치되면 커널 업데이트 시 자동 재빌드)
#   (커널 6.14+/24.04 HWE 6.11+엔 in-tree amdxdna도 있으나, SHIM 정합을 위해 위 방식을 권장)
```
참고: 버전 숫자를 여기 박지 않는다 — 호환 드라이버는 repo README가 SHIM과 함께 git에서 관리한다.
설치 후 `xrt-smi examine`으로 NPU 인식을 확인하고, Part 2/3의 샘플 모델이 실행되면(rc=0) 호환 확인이다.

---

## Part 2. 개발용 컨테이너 (빌드 + 검증)

### 2-1. 소스 준비 (호스트)
```bash
mkdir -p ~/Projects
# recursive clone(끊기면 재개 말고 지우고 다시). 컨테이너 설정이 담긴 dev-container 브랜치.
git clone --recursive --branch dev-container \
  https://github.com/ace-knu/iree-amd-aie.git ~/Projects/iree-amd-aie
cd ~/Projects/iree-amd-aie
bash build_tools/download_peano.sh        # Peano(llvm-aie) v21 → ./llvm-aie
```

### 2-2. 이미지 빌드 & 컨테이너 기동
```bash
docker build --target dev -t iree-amd-aie:dev .

NPU=$(ls /dev/accel/ | head -1)           # 보통 accel0
docker run --rm -it \
  --user "$(id -u):$(id -g)" \            # 생성 파일을 본인 소유로
  --device=/dev/accel/$NPU \              # NPU passthrough
  -v ~/Projects/iree-amd-aie:/workspace \
  -e HOME=/workspace \
  -e PEANO_INSTALL_DIR=/workspace/llvm-aie \
  iree-amd-aie:dev bash
```

### 2-3. 빌드 & 테스트 (컨테이너 내부)
```bash
cmake -B build -S third_party/iree -G Ninja \
  -DIREE_CMAKE_PLUGIN_PATHS=$PWD -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_INPUT_TORCH=ON -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TOSA=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
  -DIREE_BUILD_TESTS=ON -DIREE_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"

cmake --build build          # ninja 기본은 전 코어. 노트북 등은 -j 6 처럼 낮춰라(전 코어는 먹통 위험)
ctest --test-dir build -R amd-aie --output-on-failure   # 가속하려면 -j N (코어 수보다 낮게)
# 기대: 100% tests passed, 0 failed out of 214
```

### 2-4. (선택) 모델 → NPU 실행 (컨테이너 내부)
```bash
# ONNX → MLIR
PYTHONPATH=/workspace/build/compiler/bindings/python \
  /opt/venv/bin/python -m iree.compiler.tools.import_onnx model.onnx -o model.mlir
#   PyTorch는 iree.compiler.extras.fx_importer (torch.export) 사용

# MLIR → NPU (peano 경로는 '플래그로' 전달 필수. 일부 워크로드는 stack-size 증대)
build/tools/iree-compile --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=/workspace/llvm-aie --iree-amdaie-stack-size=2048 \
  model.mlir -o model.vmfb
build/tools/iree-run-module --device=amdxdna --module=model.vmfb --function=<name> --input=<...>
```
참고: iree-amdaie-target-device 값은 NPU 세대에 맞춘다 (Phoenix는 npu1_4col, Strix는 npu4).

---

## Part 3. 배포용 컨테이너 (실행 전용 이미지)

받는 쪽이 **모델을 컴파일·실행만** 하는 self-contained 이미지. 소스/툴체인 없음.

### 3-1. 이미지 빌드 (고정 커밋을 hermetic하게)
```bash
docker build --target runtime \
  --build-arg IREE_AMD_AIE_COMMIT=<배포할 커밋 SHA> \
  --build-arg BUILD_JOBS=6 \                       # 서버는 코어 수만큼 올려 가속
  -t iree-amd-aie:deploy \
  https://github.com/ace-knu/iree-amd-aie.git#dev-container
```
참고: 배포할 커밋 SHA는 fork에 push된 커밋이어야 한다 (git ls-remote 로 dev-container 브랜치의 SHA 확인).

### 3-2. 실행 (NPU) — 컨테이너 내부
```bash
NPU=$(ls /dev/accel/ | head -1)
docker run --rm -it --device=/dev/accel/$NPU iree-amd-aie:deploy bash

# 컨테이너 안: 모델 → MLIR → NPU
python3 -m iree.compiler.tools.import_onnx model.onnx -o model.mlir   # PyTorch는 fx_importer
iree-compile --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir="$PEANO_INSTALL_DIR" --iree-amdaie-stack-size=2048 \
  model.mlir -o model.vmfb
iree-run-module --device=amdxdna --module=model.vmfb --function=<name> --input=<...>
```
