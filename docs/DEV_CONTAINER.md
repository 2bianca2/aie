# iree-amd-aie 개발 컨테이너 재현 가이드 (Phase 1 / dev)

이 문서는 다른 사람/기관이 **똑같이 따라 해서** iree-amd-aie 개발 컨테이너를 구축할 수 있도록,
실제로 실행할 명령과 그 과정의 판단(문제·수정 포함)을 기록한다.

- 구조: **소스는 호스트에 두고 컨테이너에 mount하는 방식(bind-mount)**. 소스+submodule+Peano는
  **호스트가 준비**하고, 컨테이너는 `/workspace`로 mount해서 **빌드/실행만** 한다(이미지에 소스를 굽지 않음).
- 사용자: 컨테이너는 **호스트 사용자와 같은 uid/gid로 실행**한다(`docker run --user "$(id -u):$(id -g)"`
  또는 VS Code가 자동 매핑). 이렇게 해야 mount로 생성되는 파일이 호스트에서 **본인 소유로 남는다**
  (root 소유가 되어 sudo가 필요해지는 문제 방지).
- 범위: **개발(dev) 컨테이너만**. 배포(runtime) 이미지는 Phase 2에서 추가한다.

> **왜 venv가 아니라 Docker인가?** 소스·산출물이 호스트에 남는다는 점은 venv와 비슷하지만, venv는
> **python 패키지만** 격리한다. 이 빌드는 clang/lld·cmake·ninja 같은 **네이티브 C/C++ 툴체인**과
> 시스템 dev 라이브러리(`libudev-dev`, `uuid-dev`), glibc까지 버전에 민감한데 — Docker는 이 **OS
> 유저스페이스 전체를 고정**해 기관마다 다른 툴체인 차이를 없앤다(말하자면 "유저스페이스 전체용 venv").

---

## 0. 전제: 호스트 준비 완료

이 컨테이너는 **유저스페이스(빌드/실행 환경)만** 담는다. 커널 드라이버(amdxdna KMD)·NPU 펌웨어·Docker
등 **호스트 준비는 먼저 [`HOST_PREREQUISITES.md`](HOST_PREREQUISITES.md)를 완료**한다(확인 항목 +
없을 때 설치 방법 포함).

**호스트 준비 이후는 모든 호스트에서 동일**하다. 호스트-특정 항목(커널/KMD/디바이스/펌웨어/Docker)만
각자 맞추면, 그 다음(**동일 fork `git clone --recursive` → `download_peano.sh` v21 → `docker build` →
컨테이너에서 고정 플래그로 빌드**)은 **정해진 소스/버전**이라 동일하다. 특히:

- **유저스페이스 XRT SHIM(amdxdna HAL 드라이버)은 git에 포함되어 빌드된다** — repo의
  `runtime/src/iree-amd-aie/driver/amdxdna/`(SHIM 소스) + `third_party/XRT` submodule에서
  `-DIREE_EXTERNAL_HAL_DRIVERS=amdxdna`로 컴파일된다. **호스트의 `/opt/xilinx/xrt`를 쓰지 않으며**
  별도 설치도 필요 없다(이미지에 그 경로 없음). → 재현 가능. 단 이 SHIM은 **호스트 KMD와 ABI 호환**돼야
  실제 NPU 실행이 안전하다(`HOST_PREREQUISITES.md`의 버전 정합).
- 나머지 변동 요인(nightly/submodule 만료, apt·base image 드리프트)은 §7에 정리.

---

## 1. 호스트: 소스 준비 (recursive clone + Peano)

**소스와 모든 submodule은 호스트가 준비**한다. 컨테이너는 빌드/실행 환경만 담고, 마운트된
소스를 사용한다. fork(`ace-knu/iree-amd-aie`)에는 `main` 브랜치만 존재하여 `main` 기준으로 진행한다.

```bash
mkdir -p ~/Projects
# 모든 submodule까지 한 번에 받는다 (recursive). 중간에 끊기면 재개가 아니라
# 처음부터 다시 받는 것이 안전하다(부분 clone은 'Unable to find current revision' 유발).
git clone --recursive https://github.com/ace-knu/iree-amd-aie.git ~/Projects/iree-amd-aie
cd ~/Projects/iree-amd-aie
git remote add upstream https://github.com/nod-ai/iree-amd-aie.git

# Peano(llvm-aie) - AIE core 컴파일용 (aie2xclbin, e2e/실행 테스트에 필요).
# repo 루트에서 실행하면 프로젝트 안 './llvm-aie'에 설치된다(원샷). pin은 v21(5절 참조).
bash build_tools/download_peano.sh
```

> Peano는 repo 루트의 `llvm-aie/`(다운로드 아티팩트, `.gitignore`에 등록됨)에 설치된다. 컨테이너는
> 이를 mount로 보므로, Peano가 필요한 테스트/e2e는 `PEANO_INSTALL_DIR=/workspace/llvm-aie`로 지정한다.
> 예: `docker run ... -e PEANO_INSTALL_DIR=/workspace/llvm-aie ... bash -lc 'ctest --test-dir build -R amd-aie'`.
>
> 이 방식이라 컨테이너는 submodule/Peano를 받지 않는다(그래서 `devcontainer.json`에 postCreate가 없다).
> 호스트가 recursive clone + download_peano로 전부 준비하고, 컨테이너는 순수 빌드/실행 환경이다.

---

## 2. 컨테이너 파일 (repo 루트에 커밋)

- `Dockerfile` — `base-deps`(공통 빌드 환경) + `dev`(non-root) 2스테이지. builder/runtime은 Phase 2 주석.
- `.dockerignore` — dev 스테이지는 소스를 COPY하지 않으므로 빌드 컨텍스트 최소화.
- `.devcontainer/devcontainer.json` — dev 타깃 빌드 + `/workspace` mount + `--device=/dev/accel/accel0`
  + `remoteUser: ubuntu`. (submodule/Peano는 호스트가 준비하므로 postCreate 없음.)
- `.gitignore` — `.claude/`(Claude Code, git/컨테이너 미포함) + `/llvm-aie/`·`/llvm_aie-*.dist-info/`
  (download_peano 아티팩트, 실수 커밋 방지) 추가.
- `build_tools/peano_commit_linux.txt` — Peano pin을 만료된 v19 → **v21**로 변경(§5).
- `docs/` — 이 가이드(`DEV_CONTAINER.md`) + 호스트 준비(`HOST_PREREQUISITES.md`).

의존성 근거(소스 대조):
- cmake: IREE 요구 `3.26...3.29` ⊇ Ubuntu 24.04 apt cmake `3.28.3` → apt 사용(별도 install_cmake.sh 불필요).
- python: repo에 버전 강제 없음 → 24.04 기본 python3(3.12) 사용.
- **python3-numpy: `IREE_BUILD_PYTHON_BINDINGS=ON`에 필수**(FindPython3 NumPy 컴포넌트). apt 패키지로
  설치해 PEP 668(pip) 문제를 피한다.
- `libudev-dev`, `uuid-dev`: `third_party/XRT`(amdxdna SHIM) 빌드 의존성 → 별도 xdna-driver clone 불필요.

---

## 3. dev 컨테이너 빌드 & 기동

### 방법 A: VS Code Dev Containers
1절에서 **호스트가 recursive clone**으로 소스+submodule을 이미 준비했으므로, 폴더
(`~/Projects/iree-amd-aie`) 열기 → "Reopen in Container"만 하면 바로 빌드 가능한 환경이 된다.

> 검증 완료: `docker build --target dev`가 약 24초에 성공. 이미지 내 툴체인 —
> cmake `3.28.3`, ninja `1.11.1`, clang/lld `18.1.3`, ccache `4.9.1`, python `3.12.3`, git `2.43.0`.
> `--user "$(id -u):$(id -g)"`로 실행하면 컨테이너가 호스트 사용자 uid로 돌아 mount 쓰기가 정상이다.

### 방법 B: CLI
```bash
cd ~/Projects/iree-amd-aie
docker build --target dev -t iree-amd-aie:dev .

# NPU 디바이스 노드는 인덱스 기반(accelN)이라 고정 보장이 아니다. 먼저 확인:
#   ls /dev/accel/        (단일 NPU면 보통 accel0; 여러 개면 다를 수 있음)
NPU=$(ls /dev/accel/ | head -1)   # 예: accel0
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  --device=/dev/accel/$NPU \
  -v ~/Projects/iree-amd-aie:/workspace \
  -e HOME=/workspace \
  -e PEANO_INSTALL_DIR=/workspace/llvm-aie \
  iree-amd-aie:dev bash
# --user "$(id -u):$(id -g)": 호스트 사용자 uid/gid로 실행 → 생성 파일이 본인 소유(uid 1000 아니어도 OK).
#   (본인이 clone한 소스라 소유자와 uid가 일치해 쓰기 정상. -e HOME=/workspace 는 임의 uid에서 ccache 등 HOME 의존 도구 대비.)
# 소스+submodule은 호스트가 준비했으므로 컨테이너에서 바로 4절 빌드로 진행한다.
# (빌드만 할 거면 --device / PEANO_INSTALL_DIR 없이도 되지만, ctest amd-aie 전체 통과엔 PEANO_INSTALL_DIR 필요.)
```

---

## 4. 첫 빌드 검증 (컨테이너 내부, frontend 전부 ON)

```bash
cmake -B build -S third_party/iree -G Ninja \
  -DIREE_CMAKE_PLUGIN_PATHS=$PWD -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_INPUT_TORCH=ON -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TOSA=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
  -DIREE_BUILD_TESTS=ON -DIREE_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
cmake --build build
ctest --test-dir build -R amd-aie --output-on-failure -j $(nproc)
# 기대: 100% tests passed, 0 failed out of 214
# (IREE_ENABLE_ASSERTIONS=ON + PEANO_INSTALL_DIR 필수 — 하나라도 빠지면 일부 테스트 실패)
```

빌드가 통과하면 dev 기반이 확정된다(배포 이미지도 동일 base-deps를 공유하므로 재현된다).

> 검증 결과(구축 당시, `--cpus=6 -j 6`): **빌드 성공(rc=0, 약 25분)**. `ctest -R amd-aie`는
> Release(assertions OFF) 빌드에서 **214개 중 201개 통과(94%)**.
>
> 실패 13개는 **Peano와 무관**하다. 전부 `aie_cdo_gen_test`/`aie_elf_files_gen_test`/`iree-opt`를
> 실행하는 테스트이고, 기대하는 `XAIE API: ... with args:` 트레이스는 `iree_aie_runtime.h`의
> **`LLVM_DEBUG(...)`** 로 출력된다. `LLVM_DEBUG`는 **assertions 빌드에서만** 활성이므로, Release
> (`-DNDEBUG`) 빌드에서는 트레이스가 사라져 FileCheck가 실패한다(이 테스트들엔 `REQUIRES: asserts`
> 가드가 없어 skip이 아니라 fail). upstream CI도 `assertions:[ON,OFF]` 매트릭스로 빌드하며 이 트레이스
> 테스트들은 ON에서만 통과한다.
>
> → 전부 통과시키려면 **`-DIREE_ENABLE_ASSERTIONS=ON`으로 재빌드**한다(NDEBUG 제거 = LLVM 포함 전체
> 재컴파일). dev 컨테이너에는 assertions ON이 사실 더 적합한 기본값이다.
>
> **최종 검증**: assertions ON 재빌드(rc=0) 후 `ctest -R amd-aie`는 **211/214**로 개선(트레이스
> 10개 통과). 남은 3개(`ctrlpkt_gen`, `elf_pm_size`, `convert_device_to_control_packets`)는
> `aie_elf_files_gen_test`가 `PEANO_INSTALL_DIR`을 요구하는 진짜 Peano 의존 테스트였고, Peano v21
> (`21.0.0.2026070201`)을 `PEANO_INSTALL_DIR`로 지정하니 통과 → **214/214 전부 통과**. (Peano v21은
> 5절대로 pin에 반영됨.)

---

## 5. 알려진 이슈 / 주의

- **assertions 빌드 (ctest amd-aie 100% 통과 조건)**: 다수 Target/Transforms 테스트가 `LLVM_DEBUG`
  트레이스(`XAIE API: ...`)를 FileCheck한다. `LLVM_DEBUG`는 assertions 빌드에서만 활성이라, Release
  (`-DNDEBUG`) 빌드에서는 이 13개가 실패한다. `-DIREE_ENABLE_ASSERTIONS=ON`으로 빌드하면 통과한다.
  (dev 컨테이너 기본값으로 권장. upstream CI도 `assertions:[ON,OFF]` 매트릭스 사용.)
- **Peano pin을 v21로 bump함 (해결)**: 기존 pin `llvm_aie==19.0.0.2025052701+31d2aa6e`는 Xilinx/llvm-aie
  **nightly 자산 보관 기간 만료로 삭제**되어 `download_peano.sh`로 받을 수 없었다. 현재 공개된
  `21.0.0.2026070201+4617c73e`로 호환성을 검증(위 3개 Peano 테스트 + 전체 amd-aie **214/214 통과**)한 뒤
  `build_tools/peano_commit_linux.txt`를 **v21로 pin 변경**했다(`19 -> 21` 메이저 점프지만 호환됨).
  `download_peano.sh`가 새 pin으로 정상 동작함도 확인. → **다른 기관 재현 가능.**
  주의: v21도 **nightly**라 언젠가 다시 만료된다. 장기적으로는 특정 릴리스 미러링/vendoring이 더 안전하다.
  (`peano_commit_windows.txt`는 이번에 미변경 — linux만 대상.)
- **submodule은 무중단 recursive clone**: `git clone --recursive`를 중간에 끊으면 부분 clone이 남아
  재실행 시 `fatal: Unable to find current revision`가 난다. 끊겼으면 해당 submodule(또는 clone 전체)을
  삭제하고 처음부터 다시 받는다.
- **NumPy**: `IREE_BUILD_PYTHON_BINDINGS=ON`은 NumPy가 없으면 configure 단계에서 실패한다
  (`Could NOT find Python3 ... NumPy`). base-deps의 `python3-numpy`로 해결.
- **non-root 소유권**: 컨테이너를 root로 실행하면 mount로 생성된 build 산출물이 호스트에서
  `root:root`가 되어 호스트 사용자가 sudo 없이 다루지 못한다. 그래서 **호스트 사용자 uid/gid로 실행**한다
  — CLI는 `docker run --user "$(id -u):$(id -g)"`, VS Code devcontainer는 `updateRemoteUserUID`(기본 true)가
  이미지의 `ubuntu` 사용자 uid를 호스트 uid로 자동 remap. (이미지 기본 사용자는 `ubuntu`이지만 uid 1000에
  고정 의존하지 않는다.)
- **frontend 전부 ON**: `IREE_INPUT_STABLEHLO/TORCH/TOSA=ON`은 관련 submodule(stablehlo, torch-mlir)이
  필요하다. recursive clone이 이를 포함한다.
- **제약 호스트(노트북 등)에서의 자원 제한**: 이 빌드는 LLVM 포함이라 전 코어를 오래 100%로 점유한다.
  코어가 적거나 발열/응답성이 걱정되는 호스트(예: 노트북)에서는 `-j`를 낮추고 컨테이너 CPU를 제한한다.
  예: `docker run --cpus=6 ... iree-amd-aie:dev bash -lc 'nice -n 19 cmake --build build -j 6'`.
  이는 **실행 단위(per-run) 선택**일 뿐 이미지/`devcontainer.json`에는 넣지 않는다(빌드 서버는 풀 병렬 사용).
  (구축 당시 실측: `-j 24` 풀 병렬은 12코어 노트북에서 OS까지 CPU 기아 → 먹통/강제종료. `--cpus=6 -j 6`으로
  CPU를 600%로 캡하니 응답성 유지하며 정상 진행.)

---

## 6. Phase 2 (배포 이미지) 예고

받는 기관이 **실행만** 하도록, builder에서 특정 커밋을 clone/build/install하고 runtime에 산출물만
`COPY`하는 self-contained 이미지를 만든다. 호스트 준비는 이미 작성된 [`HOST_PREREQUISITES.md`](HOST_PREREQUISITES.md)를
그대로 쓰고, 배포 이미지 실행/검증용 `docs/RUN.md`를 Phase 2에서 추가한다.

---

## 7. 재현성 견고성 / 만료 위험 (후속 개선 후보 — 현재는 문서화만, 미적용)

이 셋업은 clone 시 외부 소스 여러 곳 + nightly 바이너리 + 버전 미고정 apt에 의존한다. "몇 년 뒤 /
다른 기관에서 그대로 재현"을 보장하려면 아래 위험을 인지하고, 필요 시 미러링/스냅샷을 도입한다.

**만료·소멸 위험 (구축 당시 실측 기준):**

| 의존물 | 출처 | 위험 | 이유 |
|---|---|---|---|
| Peano (llvm-aie) | Xilinx/llvm-aie **nightly** | 높음 | nightly 자산 주기적 prune. v19 이미 삭제됨, v21도 언젠가 만료 |
| XRT, mlir-air | **nod-ai fork**, feature 브랜치 | 중간 | 브랜치 force-push/삭제 시 pin 커밋 orphan → `clone --recursive` 영구 실패 |
| openssl-cmake | viaduck (소규모 3rd-party) | 중간 | 소규모 저장소 소멸 가능성 |
| iree, iree-org/llvm-project | iree-org (fork) | 낮음~중간 | rc 커밋, fork |
| apt 패키지 (cmake 등), `ubuntu:24.04` | Ubuntu 24.04 archive (버전 미고정) | 낮음 | 24.04 내 드리프트 + EOL 후 old-releases 이동(수년) |

**개선 옵션 (도입 시):**
- **Peano 미러링(가장 높은 위험/가장 싼 해결)**: v21 wheel을 ace-knu fork의 **GitHub Release 자산**으로
  올리고(release 자산은 nightly와 달리 자동 prune 안 됨) `download_peano.sh`가 그 URL을 받게 수정.
- **전체 소스트리 스냅샷(가장 견고)**: 완전히 채워진 트리(submodule + peano)를 tarball로 durable 보관 →
  live clone 실패 시 fallback. submodule orphan까지 대비. 수 GB.
- **base 이미지/apt 고정(선택)**: `ubuntu:24.04@sha256:...` 다이제스트 pin, 필요 시 apt 버전 pin.

> 현재는 v21 pin으로 재현 가능한 상태이며, 위 미러링/스냅샷은 **환경을 개선·정리할 때 재검토**한다.
