# 컴파일 파이프라인 디버깅 (단계별 MLIR 덤프)

입력 모델(ONNX)의 AMD-AIE 풀스택 컴파일을 **단계별로 추적**하기 위한 도구다.
`iree-compile`을 여러 번 호출해 컴파일 phase마다의 MLIR 스냅샷과 백엔드 아티팩트를 모으고,
최종 vmfb를 NPU에서 실행한 뒤 numpy 레퍼런스와 비교한다. 실행한 모든 명령을 `MANIFEST.txt`에
기록해 각 단계를 손으로 재현할 수 있다.

- 도구: `scripts/debug/pipeline_dump.py` (dev 컨테이너 안 실행)
- 호스트 래퍼: `scripts/docker/run-debug.sh` (venv + PYTHONPATH 세팅 후 컨테이너에서 실행)
- 산출물: gitignore된 `debug_out/<model>_<label>/` (실행마다 덮어씀)
- e2e 실행 흐름 자체는 `USER_GUIDE.md` §2-4 참고

## 실행

```bash
# 호스트에서 (repo 루트)
./scripts/docker/run-debug.sh \
  --model models/mlp_2layer/mlp_2layer.onnx \
  --function mlp_2layer \
  --input x.npy --input w1.npy --input w2.npy \   # 순서 = func arg 순서, .npy만
  --expected expected.npy \                        # (선택) numpy 비교 대상
  --pass iree-amdaie-lower-to-aie \                # (선택) 특정 pass before/after 추출
  --label run1
```

경로 규칙: `--model`/`--input`/`--expected`는 **repo 루트 기준 상대경로**로 준다(컨테이너에서 repo가
`/workspace`로 마운트되고 그 기준으로 해석됨). 호스트 절대경로는 안 되며, 파일은 repo 안에 있어야 한다.

주요 옵션:
- `--target-device` : `npu4`(Strix, 기본) / `npu1_4col`(Phoenix)
- `--stack-size` : 기본 2048 ('insufficient stack' 시 증대)
- `--num-outputs` : 결과 텐서 개수(기본 1)
- `--rtol/--atol` : 비교 허용오차(기본 1e-3)
- `--elide` : `--mlir-elide-elementsattrs-if-larger` (기본 64, 덤프 가독성)
- `--outdir` : 기본 `debug_out`

컨테이너 안에서 직접 실행하려면 (`run-dev.sh` 진입 후):
```bash
source /opt/venv/bin/activate
export PYTHONPATH=/workspace/build/compiler/bindings/python
python3 scripts/debug/pipeline_dump.py --model … --function … --input …
```

### 입력 npy 준비 (예시)
도구는 입력을 생성하지 않으므로 `.npy`를 미리 만들어 둔다. numpy로 만들어 repo 안(예:
`debug_out/_inputs/`)에 저장하고, 실행 시 repo 루트 기준 상대경로로 넘긴다:
```python
# (컨테이너 안, venv 활성화 후) 입력 + numpy 레퍼런스 생성
import numpy as np
d = "/workspace/debug_out/_inputs"; import os; os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(0)
x  = rng.standard_normal((128,128)).astype(np.float32); np.save(f"{d}/x.npy",  x)
w1 = rng.standard_normal((128,128)).astype(np.float32); np.save(f"{d}/w1.npy", w1)
w2 = rng.standard_normal((128,128)).astype(np.float32); np.save(f"{d}/w2.npy", w2)
np.save(f"{d}/expected.npy", (np.maximum(x@w1,0)@w2).astype(np.float32))  # (선택) 비교용
```
```bash
# 위에서 만든 파일들을 상대경로로 전달
./scripts/docker/run-debug.sh --model models/mlp_2layer/mlp_2layer.onnx --function mlp_2layer \
  --input debug_out/_inputs/x.npy --input debug_out/_inputs/w1.npy --input debug_out/_inputs/w2.npy \
  --expected debug_out/_inputs/expected.npy --label run1
```

## 산출물 레이아웃 (`debug_out/<model>_<label>/`)

```
00_source.mlir                 # import_onnx 결과 (.mlir 입력이면 복사본)
phases/
  01_input.mlir                # --compile-to=input      (frontend: torch->linalg+ABI)
  02_flow.mlir                 #             flow         (dispatch 형성)
  03_stream.mlir               #             stream       (async 스케줄/버퍼)
  04_executable-configurations.mlir  #       (codegen 직전, lowering strategy 선택 후)
  05_executable-targets.mlir   #             (amdaie->aie codegen 완료; 물리화 IR, 큼)
  06_vm.mlir                   #             vm           (vmfb 직전)
  NN_<phase>.log               # 각 phase 컴파일 stderr
passes/                        # --pass 준 경우만 (MLIR IR-print 파일 트리)
backend/
  dump_files/…                 # --iree-hal-dump-executable-files-to
  dump_intermediates/…         # --iree-hal-dump-executable-intermediates-to
  aie2xclbin.log               # 최종 컴파일 stderr (--aie2xclbin-print-ir-after-all 포함)
model.vmfb                     # 최종 컴파일 산출물
run/
  out0.npy …                   # NPU 실행 출력
  compare.txt                  # --expected 준 경우 비교 리포트
  run.log                      # iree-run-module stderr
MANIFEST.txt                   # 실행한 모든 명령 + rc + 소요시간 (재현용)
```

## npy 입력/출력 규약 (중요)

입력은 `.npy`만 받으며 iree-run-module에 **shape 프리픽스 없이** `--input=@x.npy`로 넘긴다.
`128x128xf32=@x.npy`처럼 프리픽스를 붙이면 iree가 파일을 raw 바이트로 읽어 npy 헤더(128B)까지
데이터로 먹어 **garbage/inf**가 나온다. 도구가 이 규약을 강제한다(그래서 `.bin`이 아닌 `.npy`).
`--output=@out.npy`도 동일하게 헤더 포함 npy로 저장된다.

## 특정 pass 추출 (`--pass`)

`--pass`에는 **pass의 CLI arg 이름**(예: `iree-amdaie-lower-to-aie`)을 준다 — C++ 클래스명
(`AMDAIELowerToAIE`)이 아니다. arg 이름은 `Transforms/Passes.td`의 `Pass<"...">` 첫 인자다.
여러 번 지정 가능. 결과는 `passes/` 아래 MLIR IR-print 트리로 저장되며, dispatch/파일별로
`…_0_<pass>.mlir`(before) / `…_1_<pass>.mlir`(after) 쌍이 생긴다.

## 재현

`MANIFEST.txt`의 각 명령은 그대로 복붙 실행하면 동일 산출물을 재생성한다. 특정 단계만 다시
보고 싶을 때 해당 `iree-compile … --compile-to=<phase>` 줄만 실행하면 된다.
