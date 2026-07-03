# iree-amd-aie 호스트 사전조건 (NPU 스택 + Docker)

dev/배포 컨테이너를 쓰기 **전에 호스트를 준비**하는 문서다. 컨테이너는 유저스페이스(빌드/실행)만
담고, **커널 드라이버(amdxdna KMD) · NPU 펌웨어 · Docker는 호스트(컨테이너 밖) 책임**이다.

흐름: **1. 확인 → 2. 없으면 설치 → 3. 준비되면 `DEV_CONTAINER.md`로.**

---

## 1. 확인 (호스트마다 다를 수 있음)

아래는 **동작에 필수**이며 머신마다 다를 수 있으니 각자 확인한다. 오른쪽 "레퍼런스 값"은
**이 가이드를 만들고 검증한 환경의 값**(예시일 뿐, 본인 값과 같아야 하는 건 아님)이다.

| 확인 항목 | 왜 호스트마다 다른가 | 확인 명령 | 레퍼런스 값 (검증 환경) |
|---|---|---|---|
| OS/배포판 | 패키지·기본 버전이 다름 (본 가이드는 Ubuntu 24.04 기준) | `. /etc/os-release; echo "$PRETTY_NAME"` | Ubuntu 24.04.4 LTS |
| 커널 버전 | `amdxdna` in-tree 지원(6.11+/6.14+)인지, DKMS 필요인지 | `uname -r` | 6.14.0-37-generic |
| NPU 커널 드라이버(KMD) | 호스트가 로드해야 함. **컨테이너 SHIM과 ABI 호환** 필요 | `lsmod \| grep amdxdna` | amdxdna 로드됨 |
| NPU 디바이스 노드·권한 | `accelN` 번호는 **인덱스 기반**이라 고정 아님. 권한이 다르면 `--group-add` 필요 | `ls -l /dev/accel/` | `/dev/accel/accel0` (grp `render`, mode `666` → group-add 불필요) |
| NPU 펌웨어 | 호스트에 있어야 함 | `ls /usr/lib/firmware/amdnpu` | 존재 (`1502_00`, `17f0_*`) |
| Docker | Engine + daemon 필요 | `docker --version` / `docker info` | 29.5.3 |

> 참고(동작 필수는 아님): **CPU/코어 수**(`nproc`) — 빌드 병렬도·발열 판단용, 코어 적거나 노트북이면
> `DEV_CONTAINER.md §5`의 자원 제한 참고(검증 환경은 Ryzen AI 9 HX 370, 12c/24t). **호스트 XRT**(선택) —
> 빌드엔 불필요, `source /opt/xilinx/xrt/setup.sh && xrt-smi examine`로 NPU 인식만 확인(검증 환경 2.20.0).
> 호스트 사용자 uid/gid는 컨테이너 실행 시 `docker run --user "$(id -u):$(id -g)"`가 자동 처리한다.

---

## 2. 없으면 설치

### Docker (없을 때)
가장 간단한 방법 하나만 쓰면 된다.

```bash
# (권장) 공식 설치 스크립트 — 한 줄, 최신 Docker Engine
curl -fsSL https://get.docker.com | sudo sh

# 설치 후 공통: sudo 없이 쓰려면 docker 그룹에 추가 (재로그인 필요)
sudo usermod -aG docker "$USER"
```
> 더 간단히 Ubuntu 기본 패키지도 가능: `sudo apt-get install -y docker.io` (버전은 낮지만 이 프로젝트의
> `docker build`/`docker run`엔 충분). 공식 apt 저장소를 직접 등록하는 다단계 방식도 있으나(키링+repo
> 여러 줄) 위 한 줄로 충분하다.

### NPU 드라이버(amdxdna KMD) + 펌웨어 + 호스트 XRT (없을 때)
`lsmod | grep amdxdna`가 비었거나 `/dev/accel/`·펌웨어가 없으면 아래 둘 중 하나. **이 부분은 버전에
민감하므로 `amd/xdna-driver`의 자체 지시(README/`build.sh`)를 따른다**(여기서 임의 명령을 만들지 않음).

- **(A) amdxdna in-tree 커널 사용** — mainline **6.14+** 또는 Ubuntu 24.04 **HWE 6.11+**. KMD가 커널에
  포함되어 별도 드라이버 빌드가 불필요(레퍼런스 환경이 이 경우: 6.14 + amdxdna 로드).
- **(B) out-of-tree DKMS 빌드** — `amd/xdna-driver`를 **커밋 `20e1f74`** 로 빌드·설치. 이 빌드가
  **KMD + NPU 펌웨어 + XRT(호스트 유저스페이스 도구)** 를 함께 설치한다(iree-amd-aie README §Dependencies/
  Driver도 이 커밋을 지정).
  ```bash
  git clone https://github.com/amd/xdna-driver.git
  cd xdna-driver && git checkout 20e1f74 && git submodule update --init --recursive
  # 빌드/설치 절차는 xdna-driver README(build.sh)를 그대로 따른다.
  # (기존 xrt 패키지가 있으면 충돌 방지로 제거: dpkg -l | awk '/^ii/&&$2~/^xrt/{print $2}' | xargs -r sudo apt-get remove -y)
  ```

### 버전 정합 (중요)
호스트 KMD와 **컨테이너 SHIM**(`third_party/XRT`)이 xdna-driver 계열에서 **일치**해야 실제 NPU 실행이
안전하다. 그래서 호스트 KMD도 컨테이너 SHIM과 같은 계열(`20e1f74` 기준)을 권장한다. (SHIM은 git에서
빌드되어 컨테이너에 포함되므로 별도 설치가 필요 없다 — `DEV_CONTAINER.md §0` 참조.)

---

## 3. 준비되면

위가 모두 확인되면 **`DEV_CONTAINER.md`의 1절**(fork recursive clone → Peano → docker build → 빌드/검증)
부터 진행한다. 이후 단계는 정해진 소스/버전이라 모든 호스트에서 동일하다.
