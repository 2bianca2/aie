# 이종 device 배치 (Heterogeneous device placement)

amd-aie 백엔드에서 여러 device(가속기 + host CPU)에 dispatch를 나눠 배치하는 방식과, 그 구조를 어떻게 확장하는지 정리한다.

## 개요

- **패스**: `AMDAIEAssignDeviceAffinities`
  (`compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEAssignDeviceAffinities.cpp`)
- **역할**: amd-aie가 codegen하는 op(contraction/convolution)을 **가속기**에, 그 외(layout transpose, demote가 만든
  f32→bf16 cast, 기타 elementwise)를 **host(CPU)**에 배치한다. Flow 단계 끝(dispatch outlining 후, Stream affinity
  solver 전)에서 각 `flow.dispatch`에 `stream.affinity`를 부여하고, cross-device 버퍼 해결을 위한 `stream.topology`를 주입한다.
- **훅**: `PluginRegistration.cpp`
  - `extendPreprocessingPassPipeline` — topology 조기 주입(host device global이 Flow 전에 DCE되는 것 방지) + demote.
  - `extendFlowTransformPassPipeline` — dispatch 형성 후 affinity 부여.

## device-capability 테이블 (단일 출처)

지원 device의 역할과 device 쌍의 topology capability를 **한 곳**에 두고, 분류와 topology 두 seam이 이 테이블을 참조한다.

| 조회 | 현재 값 |
|---|---|
| `deviceRole(backend)` → Host / Accelerator / Unknown | `llvm-cpu`=Host, `amd-aie`=Accelerator, 그 외=Unknown(무시) |
| `linkFlags(srcBackend, dstBackend)` → `{unified_memory, transparent_access}` | `amd-aie↔llvm-cpu`=`transparent_access:true`(통합 APU, 검증됨), 그 외 쌍=둘 다 `false`(보수적 staging) |

> `transparent_access`는 장치쌍·방향에 따라 다른 하드웨어 속성이라(IREE `HALAttrs.td`의 예시가 쌍마다 다름) **무조건 true는
> 일반적으로 틀리다.** IREE는 topology를 자동 유도하지 않고 소비만 하므로(`ResolveTopologyQueries`/`ElideAsyncCopies`),
> 이 테이블이 알려진 값을 인코딩하고, 사용자가 `stream.topology`를 직접 주면 그걸 우선한다.

## 현재 동작

1. **분류** — 선언된 모든 device global(`--iree-hal-target-device` 옵션)의 backend를 `deviceRole`로 조회 →
   `accelerators` / `hosts` 리스트. 미등록 backend는 skip(배정·topology 어디에도 안 들어가 DCE).
2. **조건** — `host ≥ 1` 필수(가속기는 선택). 단일 host 구성도 동작.
3. **정책(현재 front)** — `hostTarget = hosts.front()`; `accelTarget = accelerators.empty() ? hostTarget :
   accelerators.front()`. contraction/conv → `accelTarget`, 그 외 → `hostTarget`. (가속기 없으면 contraction도 host로.)
4. **topology** — 실제 배정된 device만 mesh(`used = {accelTarget, hostTarget}`), 링크 플래그는 테이블에서 조회. 사용자
   제공 `stream.topology`가 있으면 생성/덮어쓰지 않음.

## 확장 가이드

| 상황 | 필요한 작업 |
|---|---|
| **같은 backend의 새 device** (예: npu4 외 다른 amd-aie 타겟) | 테이블 수정 불필요 — 이미 amd-aie로 분류됨. 옵션으로 타겟하면 그대로 동작. |
| **새 accelerator backend** (예: 다른 종류 NPU) | `deviceRole`에 role 추가 + `linkFlags`에 쌍 추가. **단, 그 backend의 실제 codegen 지원이 있어야 함** — 이 플러그인은 amd-aie만 codegen하므로, codegen 없는 backend에 contraction을 배정하면 downstream 실패. |
| **다중 동종 device 실제 분배** (예: NPU 2개를 나눠 씀) | 테이블만으론 부족 — **정책 seam(front) 교체 필요**(front는 첫 device만 사용). |

## 미래: 지원연산 + 성능 기반 배치 정책의 기반

각 가속기가 실행 가능한 연산이 다르고, 정책이 "각 device의 지원 연산 + 성능"을 종합해 배치를 결정하는 형태로 확장할 때,
이 테이블 + 정책 seam이 **그대로 기반**이 된다.

```
device-capability 테이블 (device 당)
  role         : Host / Accelerator            ← (현재)
  linkFlags    : 쌍별 topology 링크 capability   ← (현재)
  supportedOps : 이 device가 실행 가능한 op 집합   ← (추가)
  cost         : (device, opKind) → 성능/비용     ← (추가)

정책(현재 front → 미래):
  각 dispatch(op kind, shape 등)에 대해
    1) supportedOps로 후보 device 필터
    2) cost로 최적 device 선택 (연산 성능 + cross-device 전송비용 종합)
```

설계상 이점:
- **role을 supportedOps에서 파생 가능** — "contraction/conv를 지원하면 accelerator, 범용 fallback이면 host"로 두면
  `supportedOps`가 `role` 컬럼을 흡수해 더 단일화된다.
- **전송비용까지 한 곳에서** — topology 링크 정보(`linkFlags`, transparent/staging)가 같은 테이블에 있으니, cost 정책이
  연산 성능뿐 아니라 cross-device 전송비용까지 종합해 비교할 수 있다.
- **정책 seam이 이미 분리됨** — 현재 `front` 선택 로직 자리가 그대로 정책 교체 지점.
- **timing 확장점 표시됨** — 성능/op 기반은 dispatch를 봐야 결정되는 per-dispatch 정책이라, 그때는 preprocessing에서
  후보 전체를 보수적으로 keep-alive mesh → Flow에서 실제 used-set으로 narrowing하는 **2단계 topology**가 필요하다(현재
  front는 dispatch 이전 결정 가능해 불필요; `AMDAIEAssignDeviceAffinities.cpp` topology 주석에 확장점 명시).

정직한 단서(규모 확장 시):
- **supportedOps 출처**: 소수 backend는 테이블 하드코딩으로 충분하나, 규모가 커지면 각 backend의 실제 codegen 능력을
  **질의(registry)** 하는 편이 이상적(backend가 자기가 lower 가능한 op을 앎).
- **cost 모델**: 실제 비용 모델(shape·메모리·전송 고려)은 그 자체로 큰 작업. 단순 힌트(상대 처리량)에서 점진 확장 가능.
- **scope**: 진짜 다중-가속기 성능 정책은 여러 backend codegen을 아우르므로 IREE-core/멀티-플러그인 성격도 있다. 다만
  "테이블 + 정책 seam" 패턴이 그 로컬 기반으로 맞고, core 메커니즘과 정렬시키기 좋다.

## 참고

- 패스·테이블: `Transforms/AMDAIEAssignDeviceAffinities.cpp` (`deviceRole`/`linkFlags`, `runOnOperation`).
- 훅: `PluginRegistration.cpp` (`extendPreprocessingPassPipeline`/`extendFlowTransformPassPipeline`).
- HAL topology attr: IREE `compiler/src/iree/compiler/Dialect/HAL/IR/HALAttrs.td` (`DeviceTopologyAttr`/`DeviceLinkAttr`),
  소비처 `Dialect/HAL/Transforms/ResolveTopologyQueries.cpp`, `Dialect/Stream/Transforms/ElideAsyncCopies.cpp`.
- 실행 레시피(2-장치 npu+cpu): `docs/2026-07-06_env_setup/USER_GUIDE.md`.
