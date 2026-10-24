"""
유속 5단계별 장치 전체 3D 유동궤적 렌더링
==========================================
run_sweeps_layers.py 가 실현 0 에서 저장한 단위셀 해(results/sol_E1_U*_r0.vtu)를
10개 기둥 위치에 타일링해 장치 전체 속도장을 만들고, 입구에서 뿌린 유선을
속도로 채색해 SolidWorks Flow Simulation 스타일로 렌더링한다.

  python3 make_3d_renders.py            # 5단계 전부
  python3 make_3d_renders.py 1.5        # 특정 유속만
"""
import os, sys, glob, numpy as np
import params_layers as P
import mesh_column as MC
import layers_model as LM
import viz3d_layers as V3

FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def load_cell(U):
    fn = f"results/sol_E1_U{U:g}_r0.vtu"
    if not os.path.exists(fn):
        return None
    import pyvista as pv
    return pv.read(fn)


def device_grid_from_vtu(cell, U_bulk, h=1.0e-3, U_layers=None):
    """단위셀 VTU 를 10개 기둥에 타일링해 장치 전체 균일격자 속도장을 만든다."""
    import pyvista as pv
    nx = int((V3.DEV_X1-V3.DEV_X0)/h)+1
    ny = int(P.DUCT_W/h)+1
    nz = int(P.DUCT_H/h)+1
    gx = np.linspace(V3.DEV_X0, V3.DEV_X1, nx)
    gy = np.linspace(-P.DUCT_W/2, P.DUCT_W/2, ny)
    gz = np.linspace(-P.DUCT_H/2, P.DUCT_H/2, nz)
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing='ij')
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], 1)
    UL = U_layers if U_layers is not None else [U_bulk]*len(P.LAYER_NCOL)

    vel = np.zeros_like(pts)
    vel[:, 0] = P.inlet_profile(pts[:, 2], U_bulk)     # 기본 = 덕트 프로파일
    for L, cols in enumerate(P.layer_columns()):
        for (xc, yc) in cols:
            loc = np.stack([pts[:, 0]-xc, pts[:, 1]-yc, pts[:, 2]], 1)
            m = ((loc[:, 0] >= P.CELL_X0) & (loc[:, 0] <= P.CELL_X1) &
                 (np.abs(loc[:, 1]) <= P.CELL_YMAX*0.98))
            if not m.any():
                continue
            probe = pv.PolyData(np.ascontiguousarray(loc[m])).sample(
                cell, tolerance=1e-6)
            v = np.asarray(probe.point_data["u"])
            valid = np.asarray(probe.point_data["vtkValidPointMask"]).astype(bool)
            idx = np.flatnonzero(m)
            vel[idx[valid]] = v[valid]*(UL[L]/U_bulk)
    g = pv.StructuredGrid(XX, YY, ZZ)
    g.point_data["u"] = vel
    g.point_data["speed"] = np.linalg.norm(vel, axis=1)
    return g


def render_one(U, vmax=None):
    cell = load_cell(U)
    if cell is None:
        print(f"  U={U}: 단위셀 해 없음 (results/sol_E1_U{U:g}_r0.vtu) — 건너뜀")
        return None
    # layer 별 접근속도 (후류 결손 보정)
    import json
    js = sorted(glob.glob(f"results/E1_U{U:g}_r*.json"))
    wC = np.mean([json.load(open(f))["wake_C"] for f in js]) if js else 0.0
    UL, dz = LM.layer_velocities(U, wC)
    g = device_grid_from_vtu(cell, U, U_layers=UL)
    fn = f"{FIG}/fig13_flow3d_U{U:g}.png"
    title = (f"유속 U = {U} m/s  ·  Re_d = {P.reynolds(U):.0f}  ·  "
             f"Q = {U*P.A_DUCT*6e4:.0f} L/min\n"
             f"액적 60개 (10 기둥 x 6) · 4 layer · 구동 V/V0 = 1.00")
    out, vm = V3.render(g, fn, U, vmax=vmax, title=title)
    print(f"  U={U}: {out}  (|u|max = {vm:.3f} m/s, 후류결손 {100*dz:.1f} %)")
    return vm


if __name__ == "__main__":
    Us = [float(a) for a in sys.argv[1:]] or P.SWEEP_U
    # 색 범위를 모든 유속에서 각각 최대값에 맞춘다(유속별 구조를 보기 위함)
    print("장치 전체 3D 유동궤적 렌더링")
    for U in Us:
        render_one(U)
