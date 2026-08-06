# 컴파일 파이프라인 디버깅 (단계별 MLIR 덤프)

입력 모델(ONNX)의 AMD-AIE 풀스택 컴파일을 **단계별로 추적**하기 위한 도구다.
기본(full) 모드는 `iree-compile` **한 번 실행에 `--dump-compilation-phases-to`로 모든 phase의 MLIR
스냅샷**과 백엔드 아티팩트를 모으고, 최종 vmfb를 NPU에서 실행한 뒤 numpy 레퍼런스와 비교한다.
패스를 고쳐가며 연구할 때는 **재개(resume) 모드**(`--from-phase`)로 이전 덤프의 특정 phase IR부터
`--compile-from`으로 **구간만 재실행**하고 그 구간/특정 패스의 입출력 MLIR을 빠르게 확인할 수 있다
(프론트엔드를 건너뛰어 빠름). 실행한 모든 명령을 `MANIFEST.txt`에 기록해 각 단계를 손으로 재현할 수 있다.

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
  --input x.npy \                                  # 순서 = func arg 순서, .npy만
  --expected expected.npy \                        # (선택) 비교 대상
  --pass iree-amdaie-lower-to-aie \                # (선택) 특정 pass before/after 추출
  --label run1
```

경로 규칙: `--model`/`--input`/`--expected`는 **repo 루트 기준 상대경로**로 준다(컨테이너에서 repo가
`/workspace`로 마운트되고 그 기준으로 해석됨). 호스트 절대경로는 안 되며, 파일은 repo 안에 있어야 한다.

주요 옵션:
- `--target-device` : `npu4`(Strix, 기본) / `npu1_4col`(Phoenix)
- `--device-flag` : iree-compile의 디바이스 선언 플래그(반복 가능). **주면 기본값
  `--iree-hal-target-backends=amd-aie`를 대체한다.** CPU+NPU hetero 컴파일을 디버깅할 때 쓴다.
- `--compile-flag` : iree-compile에 그대로 덧붙일 플래그(반복 가능). conv 모델에 필요한
  demote/im2col 계열을 여기로 넘긴다 (조합은 `models/vgg16/README.md` 참고).
- `--run-device` : iree-run-module의 `--device`(반복 가능, 기본 `amdxdna`). hetero vmfb는
  `amdxdna`와 `local-task` 둘 다 필요하다.
- `--stack-size` : 기본 2048 ('insufficient stack' 시 증대)
- `--num-outputs` : 결과 텐서 개수(기본 1)
- `--rtol/--atol` : 비교 허용오차(기본 1e-3)
- `--elide` : `--mlir-elide-elementsattrs-if-larger` (기본 64, 덤프 가독성)
- `--outdir` : 기본 `debug_out`
- `--from-phase` : 재개 모드 진입 — 같은 `--label`의 이전 덤프 `phases/N.<phase>.mlir`부터 재실행
  (아래 "패스 연구 워크플로우" 참고). 지정 시 import/프론트엔드를 건너뛰고 NPU 실행은 하지 않는다.
- `--to-phase` : 재개 정지점(기본 `executable-targets` = AIE 코드젠). `--from-phase`와 함께 쓴다.

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
x = rng.standard_normal((128,128)).astype(np.float32); np.save(f"{d}/x.npy", x)
```
`mlp_2layer`는 가중치가 모델에 상수로 구워져 있어 **런타임 입력이 `x` 하나뿐이다**
(`models/mlp_2layer/export_mlp_2layer.py` 참고). 따라서 `--expected`를 numpy로 직접 계산할 수 없고,
같은 모델을 CPU로 컴파일해 얻는다:
```bash
# (컨테이너 안) golden = 동일 모델의 llvm-cpu 실행 결과
/workspace/build/tools/iree-compile models/mlp_2layer/mlp_2layer.mlir \
  --iree-hal-target-backends=llvm-cpu -o /tmp/mlp_cpu.vmfb
/workspace/build/tools/iree-run-module --device=local-task --module=/tmp/mlp_cpu.vmfb \
  --function=mlp_2layer --input=@debug_out/_inputs/x.npy \
  --output=@debug_out/_inputs/expected.npy
```
```bash
# 위에서 만든 파일들을 상대경로로 전달
./scripts/docker/run-debug.sh --model models/mlp_2layer/mlp_2layer.onnx --function mlp_2layer \
  --input debug_out/_inputs/x.npy \
  --expected debug_out/_inputs/expected.npy --label run1
```

### hetero(CPU+NPU) 컴파일 디버깅
기본값은 NPU 단일 디바이스라 pooling·layout op가 섞인 모델은 이 형태로 넘긴다:
```bash
./scripts/docker/run-debug.sh --model models/mlp_2layer/mlp_2layer.onnx --function mlp_2layer \
  --input debug_out/_inputs/x.npy --expected debug_out/_inputs/expected.npy \
  --device-flag=--iree-hal-target-device=npu=amdxdna \
  --device-flag=--iree-hal-target-device=cpu=local \
  --device-flag=--iree-hal-local-target-device-backends=llvm-cpu \
  --device-flag=--iree-hal-default-device=npu \
  --compile-flag=--iree-amdaie-demote-contraction-inputs-to-bf16 \
  --compile-flag=--iree-amdaie-enable-vectorization-passes=false \
  --run-device amdxdna --run-device local-task \
  --rtol 5e-2 --atol 5e-2 --label hetero
```
허용오차를 푼 이유: golden은 CPU f32인데 위 컴파일은 입력을 bf16으로 demote한다. 기본값(1e-3)이면
정상 실행도 `FAIL`로 보고된다 — 수치 차이지 버그가 아니다.

## 산출물 레이아웃 (`debug_out/<model>_<label>/`)

파일명 `N.<phase>.mlir`의 `N`은 IREE의 `IREEVMPipelinePhase` enum 순번이다. full 컴파일 1회에
`--dump-compilation-phases-to`가 아래 phase들을 모두 자동으로 덤프한다(중간에 실패해도 그 이전
phase 덤프는 남는다). 별도의 phase별 `.log`는 없고, 통합 stderr는 `backend/aie2xclbin.log`에 있다.

```
00_source.mlir                 # import_onnx 결과 (.mlir 입력이면 복사본)
phases/                        # --dump-compilation-phases-to (full 모드 1회 호출로 전체 덤프)
  1.input.mlir                 # frontend: torch->linalg+ABI
  6.flow.mlir                  # dispatch 형성
  7.stream.mlir                # async 스케줄/버퍼
  8.executable-sources.mlir    # hal.executable 구성 직전 (codegen 제외)
  9.executable-configurations.mlir  # codegen 직전, lowering strategy 선택 후
  10.executable-targets.mlir   # amdaie->aie codegen(translation) 완료; 물리화 IR, 큼
  11.hal.mlir                  # hal 확정
  12.vm.mlir                   # vmfb 직전
  # 이 외 2.abi / 3.preprocessing / 4.global-optimization / 5.dispatch-creation 도 덤프됨
passes/                        # full 모드에서 --pass 준 경우만 (MLIR IR-print 파일 트리)
backend/
  dump_files/…                 # --iree-hal-dump-executable-files-to
  dump_intermediates/…         # --iree-hal-dump-executable-intermediates-to
  aie2xclbin.log               # 최종 컴파일 stderr (--aie2xclbin-print-ir-after-all 포함)
model.vmfb                     # 최종 컴파일 산출물
run/
  out0.npy …                   # NPU 실행 출력
  compare.txt                  # --expected 준 경우 비교 리포트
  run.log                      # iree-run-module stderr
resume/                        # 재개 모드(--from-phase) 산출물. baseline은 건드리지 않음
  <from>_to_<to>.mlir          # 재개 구간 출력 IR
  <from>_to_<to>.log           # 재개 컴파일 stderr
  passes/                      # 재개 중 --pass 준 경우 (before/after IR 트리)
  MANIFEST.txt                 # 재개 명령 기록
MANIFEST.txt                   # full 실행 명령 + rc + 소요시간 (재현용)
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

## 패스 연구 워크플로우 (구간 실행 + I/O MLIR)

파이프라인의 특정 단계에서 패스를 수정·추가하며 반복 개발할 때 쓴다. 매번 ONNX 프론트엔드부터
전체를 재컴파일하지 않고, **이전 덤프의 phase IR부터 구간만 재실행**해서 빠르게 확인한다.

3-step 루프:
```bash
# 1) baseline: 모든 phase IR 1회 덤프 + vmfb + NPU 실행 (full 모드)
./scripts/docker/run-debug.sh --model models/mlp_2layer/mlp_2layer.onnx --function mlp_2layer \
  --input debug_out/_inputs/x.npy --input debug_out/_inputs/w1.npy --input debug_out/_inputs/w2.npy \
  --label base
#   -> debug_out/mlp_2layer_base/phases/{1.input … 10.executable-targets … 12.vm}.mlir + model.vmfb

# 2) 대상 패스 수정 (예: compiler/.../Transforms/Passes.cpp) 후 컨테이너에서 증분 빌드
./scripts/build/build.sh

# 3) 구간만 재개(프론트엔드 skip) + 특정 패스 before/after IR 확인
./scripts/docker/run-debug.sh --model models/mlp_2layer/mlp_2layer.onnx --function mlp_2layer \
  --label base --from-phase executable-configurations --to-phase executable-targets \
  --pass iree-amdaie-lower-to-aie
#   -> debug_out/mlp_2layer_base/resume/executable-configurations_to_executable-targets.mlir
#      debug_out/mlp_2layer_base/resume/passes/…  (해당 패스 before/after)
```

- `--from-phase X`는 baseline(같은 `--label`)의 `phases/N.X.mlir`을 입력으로 `--compile-from=X`를 실행한다.
  `--to-phase Y`(기본 `executable-targets`)가 `--compile-to=Y`로 정지점을 정한다.
- 재개 모드는 IR 전용이라 NPU 실행/비교를 하지 않고, baseline 산출물(`phases/`, `model.vmfb`)을 덮어쓰지 않는다.
  결과는 `resume/` 아래에 쌓인다. baseline 덤프가 없으면 먼저 full 모드로 만들라고 에러를 낸다.
- 수정한 패스가 도는 구간을 골라 `--from/--to-phase`를 준다. AIE 코드젠 패스는 대개
  `executable-configurations → executable-targets` 구간이다.

### 수치까지 검증하려면 (수동)
재개 모드는 IR만 본다. 패스 수정 후 **NPU 출력 수치**까지 확인하려면 (a) full 모드로 재실행하거나,
(b) 덤프한 phase IR에서 vmfb까지 이어 컴파일한 뒤 실행한다(프론트엔드 skip):
```bash
# 컨테이너 안. <dir> = debug_out/<model>_<label>
iree-compile <dir>/phases/9.executable-configurations.mlir \
  --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=$PEANO_INSTALL_DIR --iree-amdaie-stack-size=2048 \
  --compile-from=executable-configurations -o /tmp/resumed.vmfb
iree-run-module --device=amdxdna --module=/tmp/resumed.vmfb --function=mlp_2layer \
  --input=@x.npy --input=@w1.npy --input=@w2.npy --output=@/tmp/out0.npy
```

## phase 덤프/재개 플래그 (배경)

- `--dump-compilation-phases-to=<dir>`: full 컴파일 1회가 각 phase 종료 시 `<dir>/N.<phase>.mlir`을
  자동 저장한다(`N` = `IREEVMPipelinePhase` 순번). HAL 서브페이즈(executable-sources/configurations/targets)도
  포함된다. → phase별로 `--compile-to`를 여러 번 돌릴 필요가 없다.
- `--compile-from=X`: X **이전** phase는 모두 skip하고 X **다음**부터 실행한다. 즉 "X phase의 출력물"인
  `N.X.mlir`을 입력으로 주고 `--compile-from=X`를 붙이면 그 다음 구간이 이어서 돈다.
  `--compile-to=Y`와 조합해 X~Y 구간만 실행할 수 있다(`compile-from < compile-to` 필수).

## 재현

`MANIFEST.txt`(및 `resume/MANIFEST.txt`)의 각 명령은 그대로 복붙 실행하면 동일 산출물을 재생성한다.
특정 phase 단독은 `iree-compile … --compile-to=<phase>`, 특정 구간은
`iree-compile <phases/N.X.mlir> --compile-from=X --compile-to=Y` 줄만 실행하면 된다.
