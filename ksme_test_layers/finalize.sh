#!/bin/bash
# 전체 파이프라인 실행 (처음부터 끝까지)
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}

echo "[1/5] 설계 제원"
python3 params_layers.py

echo "[2/5] 720소자 음향 기준장"
[ -f acoustic_layers_ref.npz ] || python3 acoustics_layers.py

echo "[3/5] 단위셀 메시 5종"
python3 mesh_column.py

echo "[4/5] DOE 실행 (실험1 25조건 + 실험2 25조건)"
python3 -u run_sweeps_layers.py 2>&1 | tee -a sweeps.log

echo "[5/5] 집계 · 표 · 그림"
python3 make_report_layers.py

echo "완료.  RESULTS.txt / figures/ 확인"
