# 다른 Claude 세션에 그대로 붙여넣을 프롬프트

아래 `--- 여기부터 ---` ~ `--- 여기까지 ---` 사이를 통째로 복사해서 새 Claude Code 세션에 붙여넣으면 된다.
(작업 디렉터리를 `/Users/user/Desktop/FEniCSx/ksme_test_layers` 로 열고 시작할 것)

--- 여기부터 ---

너는 FEniCSx(DOLFINx) 기반 CFD/음향 해석 파이프라인을 **재현 실행**하는 역할이다.
새로 설계하지 말고, 이 디렉터리에 이미 있는 코드를 **그대로 실행**해서 동일한 결과를 재생성해라.

## 0. 대상 연구

40 kHz 초음파(TinyLev 방식, 트랜스듀서 720소자)로 부양시킨 직경 2 mm 물 액적 **60개**
(10 기둥 × 기둥당 6 액적, 4-layer 엇갈림 배열 3-2-3-2)를 필터로 써서,
210 × 32 mm 덕트를 흐르는 공기 중 10 µm 미세입자를 포집하는 시스템의 전산해석이다.

해석 전략은 **규모 분리(multi-scale)** 다. 절대 한 메시에 다 넣으려 하지 마라(자유도 10⁸ 급이라 불가능).
- 장치 규모: 음향장 = Rayleigh–Sommerfeld 해석적 중첩 + Gor'kov 포텐셜 (FEM 아님)
- 기둥 규모: 3D 정상 Navier–Stokes, Taylor–Hood P2/P1, Newton(SNES)+MUMPS LU
- 액적 규모: 표면 응력적분 + Lagrangian RK4 입자추적
- 배열 규모: 포집 단면적 `A_cap` 기반 layer 캐스케이드 모델

**시작 전에 반드시 `METHODS.md` 를 먼저 읽어라.** 지배방정식·수치기법·DOE·검증·한계가 전부 거기 있다.
그 다음 `README.md`, `RESUME.md` 순으로 읽으면 전체 그림이 잡힌다.

## 1. 환경

```bash
conda activate fenicsx-env      # 또는 /Users/user/miniforge3/envs/fenicsx-env/bin/python3 를 직접 사용
export OMP_NUM_THREADS=6
```
검증된 버전: Python 3.14.6 (conda-forge) · DOLFINx 0.11.0 · gmsh 4.15.2 · NumPy 2.5.3 · SciPy 1.18.0 · Matplotlib 3.11.1

`python3 -c "import dolfinx; print(dolfinx.__version__)"` 로 0.11.0 이 나오는지 먼저 확인할 것.
버전이 다르면 API가 깨질 수 있으니, 임의로 코드를 고치지 말고 먼저 나에게 보고해라.

**메모리 제약**: 16 GB RAM 에서 MUMPS 직접해법의 실질 상한은 자유도 **약 35만**이다.
그 이상이면 스왑으로 사실상 멈춘다. 메시를 임의로 키우지 마라.

## 2. 실행 순서 (이대로)

```bash
cd /Users/user/Desktop/FEniCSx/ksme_test_layers

python3 params_layers.py                          # 설계 제원 확인 (즉시)
python3 acoustics_layers.py                       # 720소자 음향 기준장 → acoustic_layers_ref.npz (약 4분)
python3 mesh_column.py                            # 1/4 단위셀 메시 5종 mesh_col_r0~r4.msh (약 10초)
OMP_NUM_THREADS=6 python3 -u run_sweeps_layers.py 2>&1 | tee -a sweeps.log   # DOE 50조건 (약 142분)
OMP_NUM_THREADS=6 python3 -u validate_layers.py   2>&1 | tee -a validation.log  # 검증 V1~V5 (약 20분)
python3 neck_model.py                             # 넥 유입 · 트랜스듀서 오염 · 덕트 바닥 침강
python3 make_report_layers.py                     # RESULTS.txt + figures/ 그림 15종
python3 make_3d_renders.py                        # 유속 5단계 3D 유동궤적 fig13 5장
```

전체를 한 번에 돌리려면 `./finalize.sh` (단, finalize.sh 는 검증·넥·3D 렌더는 빼고 1~5단계만 돈다).

**중요**: `run_sweeps_layers.py` 는 `results/` 에 해당 조건 JSON 이 이미 있으면 건너뛴다(이어하기 지원).
**처음부터 다시 계산하고 싶으면** `results/*.json` 을 지우고 시작해라. 그냥 두면 즉시 끝나버린다.
`acoustic_layers_ref.npz` 도 마찬가지로 있으면 재사용되므로, 재생성하려면 삭제 후 실행.

DOE 는 2~4시간 걸리니 백그라운드로 돌리고 로그를 주기적으로 확인하는 방식으로 진행해라.

## 3. DOE 정의 (params_layers.py 에 있음, 바꾸지 말 것)

- 실험1 — 유속 스윕: `SWEEP_U = [0.5, 1.0, 1.5, 2.0, 2.5] m/s`, 구동 V/V0 = 1.0 고정
- 실험2 — 구동 스윕: `SWEEP_DRIVE = [0.70, 0.85, 1.00, 1.15, 1.30]`, U = 1.5 m/s 고정
- 각 조건 **5회 반복** (`NREP=5`) — 반복은 난수 시드(`SEED_REP=[1000..5000]`)와
  액적 높이 섭동, 그리고 메시 실현 5종(`mesh_col_r0~r4`)을 함께 바꾸는 방식이다. 총 50조건.
- 입자 시딩: 중요도 표본추출 `N_SEED_IMP=7200`(액적당 1200), 직접법 `N_SEED_DIR=8000`
- 주요 물성: `D_DROP=2.0 mm`, `D_PART=10 µm`, `C_MASS_IN=10 mg/m³`,
  `P_ANTINODE_REF=2400 Pa`, `N_PER_COL=6`, 덕트 `210 × 32 mm`

조건을 바꾸라는 지시가 따로 없으면 **`params_layers.py` 를 절대 수정하지 마라.**
바꿔야 할 일이 생기면 `METHODS.md` §7 의 변경표를 따르고, 메시/음향장 재생성 필요 여부를 확인해라.

## 4. 기대 결과 (재현 여부 판정 기준)

- `results/` 에 조건별 JSON **50개** + 궤적 npz + 유동해 vtu
- `RESULTS.txt` 6개 절 (설계제원 / 음향 60트랩 / 실험1 / 실험2 / 넥·침강 / 입자수지 / 운전지도)
- `figures/fig01`~`fig15` PNG
- 검증 **V1~V5 전부 PASS**
- 참고 수치 (이전 실행값, 이 정도면 재현 성공):
  - U=0.5 m/s: 기둥합 항력 ≈ 4.39 µN, Cd 오차 ≈ +10.7 %, A_col ≈ 0.7 mm², 제거율 ≈ 0.11 %
  - 실험2 U=1.5 m/s: V/V0=0.70 → 부양 0/60 (제거율 0 %), 0.85 → 50/60 (0.87 %),
    1.00 → 60/60 (1.01 %), 1.15 → 60/60 (1.00 %), 1.30 → 60/60 (1.00 %)
  - DOE 총 소요 ≈ 141.7 분
  - V5 입자수 수렴: 1200 → 5.81 mm², 3000 → 6.58 mm², 6000 → 7.13 mm²

수치가 몇 % 차이나는 건 정상(난수·메시·BLAS 차이). **부호나 자릿수가 다르면 그건 문제니 보고해라.**

## 5. 선택 — 정전 포집 실험

`electro_column.py` 는 위 파이프라인에 정전기력(입자 −대전 / 액적 +대전, 쿨롱력 + 도체구 영상력)을
추가한 확장 실험이다. 기본 파이프라인과 별개이며 `finalize.sh` 에 포함돼 있지 않다.
환경변수로 조건을 준다:

```bash
U=1.5 MESH=mesh_col_r4.msh NPER=400 TAG=electro_sweep python3 -u electro_column.py
```
기본 스윕 `CASES` 는 구동 1.0 · 코로나 전계 3e5 V/m 에서 액적 전하비 Q/Q_lim = 0, 0.25, 0.5, 1.0, 2.0.
`results/electro_sweep.json` 에 저장된다.
주의: 현재 저장돼 있는 `results/electro_sweep.json` 은 **이전 버전 코드의 출력**이라 스키마가 다르다
(`U` 키 없음). 다시 돌리면 새 스키마로 덮어써진다.

## 6. 작업 방식

- 긴 계산은 백그라운드 실행 + 로그 tail 로 진행 상황을 확인해라. 중간에 죽었으면 이어하기가 되니 재실행.
- 코드를 임의로 리팩터링하거나 "개선"하지 마라. 재현이 목적이다.
- 실행 중 에러가 나면 원인을 짚어서 보고하고, 고쳐야 할 부분이 있으면 최소 수정으로 제안해라.
- 끝나면 `RESULTS.txt` 의 핵심 수치와 위 §4 기대값을 비교해서 재현 성공 여부를 요약해 보고해라.

--- 여기까지 ---
