# iree-amd-aie 배포 이미지 실행 가이드 (Phase 2 / runtime)

`iree-amd-aie:deploy`(runtime) 이미지를 NPU 호스트에서 실행하는 문서다. 이미지는 **컴파일+실행
도구만** 담은 self-contained 이미지로, 소스·툴체인·mount가 없다. 받는 기관은 **모델을 컴파일하고
NPU에서 실행**하기만 하면 된다.

## 이미지에 들어있는 것
- `iree-compile` (+ `libIREECompiler.so`) — MLIR 모델을 amd-aie(NPU)용으로 컴파일
- `iree-run-module` — 컴파일된 `.vmfb`를 NPU에서 실행 (amdxdna SHIM 정적 링크)
- **모델 importer(버전 정합, 우리 빌드의 python 바인딩)** — 원본 모델을 MLIR로 변환:
  - ONNX: `python -m iree.compiler.tools.import_onnx`
  - PyTorch: `iree.compiler.extras.fx_importer` (torch.export 기반)
- python + `onnx`/`torch`(CPU)/`sympy` (importer 실행용, `/opt/venv`)
- `/opt/llvm-aie` — Peano(AIE core 컴파일러), `PEANO_INSTALL_DIR`로 이미 설정됨

---

## 0. 사전조건
호스트에 **NPU 스택(amdxdna KMD + 펌웨어) + Docker**가 준비돼 있어야 한다 →
[`HOST_PREREQUISITES.md`](HOST_PREREQUISITES.md) 참조. **호스트 KMD와 이미지 SHIM은 ABI 호환**돼야
실제 실행이 안전하다(호환 드라이버 버전은 iree-amd-aie README가 지정; `xrt-smi examine` + 샘플 실행으로 검증).

## 1. 이미지 확보
```bash
# (A) 제공자: 고정 커밋에서 빌드 → tar.gz로 저장해 전달
#   BUILD_JOBS는 OS 몫 2코어를 남겨 전 코어 점유(먹통)를 방지 (2-3/USER_GUIDE와 동일)
docker build --target runtime \
  --build-arg IREE_AMD_AIE_COMMIT=<배포 커밋 SHA> \
  --build-arg BUILD_JOBS="$(nproc --ignore=2)" \
  -t iree-amd-aie:deploy https://github.com/ace-knu/iree-amd-aie.git#dev
docker save iree-amd-aie:deploy | gzip > iree-amd-aie-deploy.tar.gz

# (B) 받는 측: 전달받은 이미지 로드
gunzip -c iree-amd-aie-deploy.tar.gz | docker load
```

## 2. 실행 (NPU passthrough)
```bash
NPU=$(ls /dev/accel/ | head -1)          # 보통 accel0 (인덱스라 고정 아님)
docker run --rm -it --device=/dev/accel/$NPU iree-amd-aie:deploy bash
```

## 3. 모델 → MLIR → NPU 실행 (컨테이너 내부)

**입력이 원본 모델(ONNX/PyTorch)이면 먼저 MLIR로 변환한다. 이미 `.mlir`이면 3-2로.**

### 3-1. 원본 모델 → MLIR (importer, 버전 정합)
```bash
# ONNX
python -m iree.compiler.tools.import_onnx model.onnx -o model.mlir

# PyTorch: torch.export로 저장한 ExportedProgram(model.pt2)을 fx_importer로 변환
python - <<'PY'
import torch
from iree.compiler.extras.fx_importer import FxImporter
ep = torch.export.load("model.pt2")             # 입력 shape 포함 → example_input 불필요
imp = FxImporter(); imp.import_frozen_program(ep)
open("model.mlir","w").write(str(imp.module))
PY
```
> model.pt2는 모델 저장 시 `torch.export.save(torch.export.export(model, (예시입력,)), "model.pt2")`로 만든다
> (기존 .pt 모델도 torch 2.x에서 이렇게 변환; 입력 shape은 export 시점에 지정).

### 3-2. MLIR → NPU 실행
```bash
# 컴파일: amd-aie 타깃. target-device는 NPU 세대에 맞춘다
#   Phoenix=npu1_4col, Strix(예: Ryzen AI 9 HX 370)=npu4
#   --iree-amd-aie-peano-install-dir 는 필수(환경변수 PEANO_INSTALL_DIR만으론 iree-compile이 못 찾음)
iree-compile --iree-hal-target-backends=amd-aie \
  --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir="$PEANO_INSTALL_DIR" \
  model.mlir -o model.vmfb
#   일부 워크로드는 AIE 코어 스택이 부족할 수 있다("insufficient stack" 에러) →
#   --iree-amdaie-stack-size=2048 처럼 늘린다. 성능 경로는 bf16 + --iree-amdaie-enable-ukernels=all.

# 실행: amdxdna HAL로 NPU에서 구동
#   --function/--input은 model.mlir의 `func.func @<이름>(%arg0: tensor<...>)`를 보고 맞춘다
#   (함수명=@<이름>; 입력=각 인자의 shape·dtype을 인자 수만큼). 아래는 예시.
iree-run-module --device=amdxdna --module=model.vmfb \
  --function=main --input="1x3x224x224xf32=1"
```
> 검증됨: 배포 이미지에서 ONNX matmul을 import→compile→`iree-run-module --device=amdxdna`로 NPU에서
> 실행(rc=0) 확인. (naive f32 matmul은 수치 정확도가 낮을 수 있음 — 실제 워크로드는 bf16+ukernel 권장.)
> 정확한 플래그/타깃 디바이스와 e2e 예시는 repo의 `build_tools/ci/run_matmul_test.sh`와
> `compiler/plugins/target/AMD-AIE/iree-amd-aie/Test/samples/`를 참조한다. PyTorch fx_importer의
> 정확한 API는 `iree.compiler.extras.fx_importer` 소스를 확인한다(torch 버전에 민감).

## 4. 빠른 검증
```bash
iree-compile --version                                  # 도구 동작
python3 -c "import torch,onnx; import iree.compiler"     # importer 스택 로드
# ONNX matmul을 만들어 import→compile→run 하는 전체 확인은 아래처럼:
python3 -c "import onnx; from onnx import helper as h,TensorProto as T; \
m=h.make_model(h.make_graph([h.make_node('MatMul',['A','B'],['Y'])],'mm', \
[h.make_tensor_value_info('A',T.FLOAT,[64,64]),h.make_tensor_value_info('B',T.FLOAT,[64,64])], \
[h.make_tensor_value_info('Y',T.FLOAT,[64,64])]),opset_imports=[h.make_opsetid('',20)]); \
onnx.save(m,'/tmp/mm.onnx')"
python3 -m iree.compiler.tools.import_onnx /tmp/mm.onnx -o /tmp/mm.mlir
iree-compile --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir="$PEANO_INSTALL_DIR" --iree-amdaie-stack-size=2048 \
  /tmp/mm.mlir -o /tmp/mm.vmfb
iree-run-module --device=amdxdna --module=/tmp/mm.vmfb --function=mm \
  --input="64x64xf32=2" --input="64x64xf32=3"     # NPU에서 실행되면 rc=0
```

---

## 참고 / 주의
- 이미지는 **import(ONNX/PyTorch→MLIR) + 컴파일 + 실행** 구성이다. IREE를 소스에서 빌드하지 않는다.
- **버전 정합**: 호스트 KMD ↔ 이미지 SHIM(xdna-driver 계열)이 어긋나면 컴파일은 되어도 NPU 실행에서
  실패할 수 있다(`HOST_PREREQUISITES.md`).
- 이미지 크기: Peano(~418MB) + `libIREECompiler.so`(native + python 바인딩) + torch(CPU) 때문에
  **수 GB**다. import 계층(python/torch/onnx)이 필요 없다면(미리 `.mlir`로 받는 경우) Dockerfile에서
  `IREE_BUILD_PYTHON_BINDINGS=OFF` + runtime의 python/torch/onnx를 빼 훨씬 작게 만들 수 있다.
- **torch 버전**: PyTorch fx_importer는 `torch.export` API에 의존해 torch 버전에 민감하다. Dockerfile은
  검증된 `torch==2.12.1+cpu`로 pin되어 있다(dev/배포 동일). torch를 올릴 때는 실제 PyTorch 모델 import를
  다시 검증한다.
- **환경변수**: 배포 이미지는 `PATH`(venv 포함)·`PYTHONPATH`·`PEANO_INSTALL_DIR`가 이미 설정돼 있어,
  dev(2-4)와 달리 `source`/`export` 없이 `python3 -m ...`/`iree-compile`이 바로 동작한다.
