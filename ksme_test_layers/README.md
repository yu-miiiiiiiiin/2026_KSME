# 다층 초음파 부양 액적 필터 — 해석 패키지

초음파로 부양시킨 물 액적 **60개**(10 기둥 × 6 액적, 4 layer 엇갈림)를 필터로 사용해
10 µm 미세입자를 포집하는 시스템의 전산해석.

| 파일 | 내용 |
|---|---|
| **`METHODS.md`** | **해석 방법론 — 지배방정식·수치기법·DOE·검증·재현 방법 (먼저 읽을 것)** |
| **`RESULTS.txt`** | 전체 결과 표 (설계 제원 · 60개 액적 트랩 · 실험1 · 실험2 · 안정성 지도) |
| `operating_map` (RESULTS.txt [6]) | **운전 지도 — 유속 x 구동세기별 흡착율** |
| `figures/` | 그림 15종 (배치도, 음향장, 힘 수지, 유동장, 궤적, 스윕 결과, 검증, 입자수지) |
| `figures/fig13_flow3d_U*.png` | **유속 5단계 장치 전체 3D 유동궤적** (SolidWorks 스타일) |
| `results/` | 조건별 원자료 JSON, 입자 궤적 npz, 유동해 vtu |
| `sweeps.log` | DOE 실행 로그 |
| `validation.log` | 검증 실행 로그 |

## 바로 실행

```bash
./finalize.sh          # 전체 (약 3~5시간)
```

## 개별 실행

```bash
python3 params_layers.py                        # 설계 제원 확인
python3 acoustics_layers.py                     # 음향 기준장 (약 4분)
python3 mesh_column.py                          # 메시 (약 10초)
OMP_NUM_THREADS=6 python3 run_sweeps_layers.py  # DOE (약 3시간, 이어하기 지원)
OMP_NUM_THREADS=6 python3 validate_layers.py    # 검증
python3 neck_model.py                           # 넥 유입·트랜스듀서 오염
python3 make_report_layers.py                   # 표·그림 (RESULTS.txt + figures/)
python3 make_3d_renders.py                      # 유속 5단계 3D 유동궤적
```

조건을 바꾸려면 `params_layers.py` 만 고친다 (`METHODS.md` §7 표 참조).
