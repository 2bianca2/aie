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
[`HOST_PREREQUISITES.md`](HOST_PREREQUISITES.md) 참조. 특히 **호스트 KMD와 이미지의 SHIM은
버전 정합**(xdna-driver `20e1f74` 계열)이어야 실제 실행이 안전하다.

## 1. 이미지 확보
```bash
# (배포 방식 A) fork의 고정 커밋에서 직접 빌드 — 서버 권장(빌드 몇십 분~)
docker build --target runtime \
  --build-arg IREE_AMD_AIE_COMMIT=<배포 커밋 SHA> \
  -t iree-amd-aie:deploy https://github.com/ace-knu/iree-amd-aie.git#<브랜치>
#   제약 호스트(노트북)면 --build-arg BUILD_JOBS=6 등으로 병렬도 낮춤

# (배포 방식 B) 이미지 tar를 받은 경우
docker load -i iree-amd-aie-deploy.tar
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

# PyTorch: torch.export한 프로그램을 fx_importer로 변환 (짧은 파이썬 스크립트)
python - <<'PY'
import torch
from iree.compiler.extras.fx_importer import FxImporter
ep = torch.export.export(MyModule().eval(), (example_input,))
imp = FxImporter(); imp.import_frozen_program(ep)
open("model.mlir","w").write(str(imp.module))
PY
```

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
iree-run-module --device=amdxdna --module=model.vmfb \
  --function=<name> --input=<...>
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
- **torch 버전 정합(빌드 시 검증 필요)**: PyTorch fx_importer는 `torch.export` API에 의존해 torch
  버전에 민감하다. 배포 빌드 후 실제 PyTorch 모델로 import를 검증하고, 확인된 torch 버전을
  Dockerfile에 pin하는 것이 좋다(현재는 CPU 최신 wheel).
