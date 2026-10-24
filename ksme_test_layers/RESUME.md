# 작업 완료 요약 (2026-09-08)

`METHODS.md`(방법론·재현법) → `RESULTS.txt`(전체 결과표) → `figures/`(그림) 순으로 보면 된다.

## 무엇을 했나

초음파로 부양시킨 물 액적 **60개**(10 기둥 × 6 액적, 4 layer 엇갈림 배열)를 필터로
쓰는 시스템을, 요청하신 네 가지 축으로 전부 해석했다.

1. **액적에 걸리는 힘** — 중력 · 초음파(Helmholtz + Gor'kov) · 유동(Navier–Stokes)
   을 지배방정식에서 출발해 60개 액적 각각에 대해 산출
2. **흡착율** — 액적 통과 전후 미세입자 농도를 두 가지 독립 방법으로 비교
3. **유속 5단계 × 5회 / 초음파 구동 5단계 × 5회** (총 50조건, 141.7분)
4. **덕트–부양기 직결 설계** (멤브레인 없이 액적 교체 가능) 반영

## 최종 산출물

| 파일 | 내용 |
|---|---|
| `METHODS.md` | 방법론 — 지배방정식, 규모분리, 수치기법, DOE, 검증, 재현 절차 |
| `RESULTS.txt` | 결과표 6절 (설계제원 / 음향 60트랩 / 실험1 / 실험2 / 넥·침강 / 입자수지 / 운전지도) |
| `figures/fig01`~`fig15` | 그림 15종 |
| `figures/fig13_flow3d_U*.png` | **유속 5단계 장치 전체 3D 유동궤적** (SolidWorks 스타일) |
| `results/` | 조건별 원자료 JSON 50개, 궤적 npz, 유동해 vtu |
| `sweeps.log`, `validation.log` | 실행·검증 로그 |

## 재실행

```bash
./finalize.sh                                    # 전체 (약 3시간)
# 또는 개별
python3 acoustics_layers.py                      # 음향 기준장 (4분)
python3 mesh_column.py                           # 메시 5종 (10초)
OMP_NUM_THREADS=6 python3 run_sweeps_layers.py   # DOE 50조건 (142분, 이어하기 지원)
OMP_NUM_THREADS=6 python3 validate_layers.py     # 검증 V1~V5
python3 make_report_layers.py                    # RESULTS.txt + figures/
python3 neck_model.py                            # 넥 유입·트랜스듀서 오염
python3 make_3d_renders.py                       # 3D 유동궤적 5장
```

조건을 바꾸려면 `params_layers.py` 만 고친다 (`METHODS.md` §7 표 참조).
