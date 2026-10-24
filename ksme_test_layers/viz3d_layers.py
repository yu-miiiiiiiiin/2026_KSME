"""
장치 전체 3D 유동궤적 렌더링 (SolidWorks Flow Simulation 스타일)
=================================================================
단위셀(기둥 1개) FE 해를 10개 기둥 위치에 타일링해 장치 전체 속도장을 구성하고,
입구에서 뿌린 유선(streamline)을 속도로 채색해 렌더링한다.

  * 타일링 근거 : 기둥 간 거리(최소 69.5 mm = 35 D)가 단위셀 영향권(~10 D)보다
    훨씬 크므로, 각 기둥의 교란은 서로 겹치지 않는다. layer 별 접근속도는
    후류 결손 보정(layers_model)을 적용한다.
  * 액적 60개, 부양기 넥 20개(상/하), 덕트 외곽을 함께 그린다.
"""
import os, numpy as np
import params_layers as P
import mesh_column as MC

KFONT = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
NBAND = 10
DEV_X0, DEV_X1 = -0.08, 0.26          # 장치 x 범위 [m]


def device_grid(probe_of_layer, h=1.0e-3, U_bulk=1.5, U_layers=None):
    """
    장치 전체 균일격자 위 속도장을 구성한다.
    probe_of_layer : layer 인덱스 -> 단위셀 프로브 (호출가능)
    """
    import pyvista as pv
    nx = int((DEV_X1-DEV_X0)/h)+1
    ny = int(P.DUCT_W/h)+1
    nz = int(P.DUCT_H/h)+1
    gx = np.linspace(DEV_X0, DEV_X1, nx)
    gy = np.linspace(-P.DUCT_W/2, P.DUCT_W/2, ny)
    gz = np.linspace(-P.DUCT_H/2, P.DUCT_H/2, nz)
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing='ij')
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], 1)
    UL = U_layers if U_layers is not None else [U_bulk]*len(P.LAYER_NCOL)

    vel = np.zeros_like(pts)
    vel[:, 0] = P.inlet_profile(pts[:, 2], U_bulk)
    cols = [(L, xc, yc) for L, cs in enumerate(P.layer_columns()) for (xc, yc) in cs]
    for L, xc, yc in cols:
        loc = np.stack([pts[:, 0]-xc, pts[:, 1]-yc, pts[:, 2]], 1)
        m = ((loc[:, 0] >= P.CELL_X0) & (loc[:, 0] <= P.CELL_X1) &
             (np.abs(loc[:, 1]) <= P.CELL_YMAX))
        if not m.any():
            continue
        v = probe_of_layer(L)(loc[m])
        vel[m] = v*(UL[L]/U_bulk)
    g = pv.StructuredGrid(XX, YY, ZZ)
    g.point_data["u"] = vel
    g.point_data["speed"] = np.linalg.norm(vel, axis=1)
    return g


def device_geometry():
    """덕트 상자, 액적 60개, 넥 20개 메쉬."""
    import pyvista as pv
    drops = []
    for L, cs in enumerate(P.layer_columns()):
        for (xc, yc) in cs:
            for zk in MC.full_column_z():
                drops.append(pv.Sphere(radius=P.A_DROP, center=(xc, yc, zk),
                                       theta_resolution=26, phi_resolution=26))
    necks = []
    for L, cs in enumerate(P.layer_columns()):
        for (xc, yc) in cs:
            for s in (+1, -1):
                hn = 0.45*P.NECK_H          # 시야를 가리지 않도록 일부만 표시
                necks.append(pv.Cylinder(
                    center=(xc, yc, s*(P.DUCT_H/2 + hn/2)),
                    direction=(0, 0, 1), radius=P.R_OPEN, height=hn,
                    capping=False, resolution=40))
    duct = pv.Box(bounds=(DEV_X0, DEV_X1, -P.DUCT_W/2, P.DUCT_W/2,
                          -P.DUCT_H/2, P.DUCT_H/2))
    return drops, necks, duct


def draw_legend(png, vmin, vmax, unit="속도 [m/s]", cmap="jet", nband=NBAND,
                title=None):
    """좌상단 이산 색 사각형 + 수치 범례 (SolidWorks 스타일)."""
    import matplotlib as mpl
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(png).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    cm = mpl.colormaps[cmap]
    bh = max(14, int(H*0.026)); bw = int(bh*1.15)
    x0, y0 = int(W*0.014), int(H*0.014)
    try:
        f = ImageFont.truetype(KFONT, int(bh*0.78))
        ft = ImageFont.truetype(KFONT, int(bh*0.95))
    except OSError:
        f = ft = ImageFont.load_default()
    vals = np.linspace(vmax, vmin, nband+1)
    for i in range(nband+1):
        c = tuple(int(255*v) for v in cm(1.0 - i/nband)[:3])
        y = y0 + i*bh
        d.rectangle([x0, y, x0+bw, y+bh-2], fill=c, outline=(0, 0, 0), width=1)
        d.text((x0+int(bw*1.9), y+int(bh*0.06)),
               f"{vals[i]:.3f}" if vals[i] else "0", font=f, fill=(0, 0, 0))
    yy = y0 + (nband+1)*bh + int(H*0.018)
    d.text((x0, yy), unit, font=f, fill=(0, 0, 0))
    if title:
        nl = title.count("\n") + 1
        d.text((x0, H - int(bh*(nl+1.6))), title, font=ft, fill=(0, 0, 0))
    im.save(png)
    return png


def render(grid, fname, U_bulk, *, n_lines=150, size=(1900, 1150),
           tube=None, vmax=None, title=None, elev=21.0, azim=32.0):
    import pyvista as pv
    spd = grid.point_data["speed"]
    vmax = float(np.nanmax(spd)) if vmax is None else vmax
    tube = tube if tube is not None else 3.4e-4

    # 시드 : 입구 단면에 격자로 균일 배치 + 액적 기둥 높이에 집중
    ny, nz = 15, 7
    yy = np.linspace(-P.DUCT_W/2*0.92, P.DUCT_W/2*0.92, ny)
    zz = np.linspace(-P.DUCT_H/2*0.85, P.DUCT_H/2*0.85, nz)
    YY, ZZ = np.meshgrid(yy, zz, indexing="ij")
    s1 = np.stack([np.full(YY.size, DEV_X0+3e-3), YY.ravel(), ZZ.ravel()], 1)
    # 액적 기둥 정면을 지나는 유선을 반드시 포함 (포집 경로를 보이기 위함)
    ycs = sorted({yc for cs in P.layer_columns() for (_, yc) in cs})
    zs = MC.full_column_z()
    s2 = np.array([[DEV_X0+3e-3, yc + dy, zk]
                   for yc in ycs for zk in zs for dy in (-6e-4, 6e-4)])
    seed = pv.PolyData(np.vstack([s1, s2]))
    sl = grid.streamlines_from_source(
        seed, vectors="u", integration_direction="forward",
        max_length=4.0*(DEV_X1-DEV_X0), initial_step_length=0.5,
        max_step_length=1.0, max_steps=20000,
        terminal_speed=1e-8, compute_vorticity=False)

    drops, necks, duct = device_geometry()
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("#f4f6f8", top="#ccd5de")
    for m in necks:
        p.add_mesh(m, color="#98a0a8", opacity=0.13, show_scalar_bar=False)
    if sl.n_points:
        p.add_mesh(sl.tube(radius=tube), scalars="speed", cmap="jet",
                   clim=(0, vmax), n_colors=NBAND, show_scalar_bar=False,
                   lighting=False)
    for m in drops:
        p.add_mesh(m, color="#12305c", opacity=1.0, smooth_shading=True,
                   specular=0.7, show_scalar_bar=False)
    p.add_mesh(duct.extract_all_edges(), color="#3a3a3a", line_width=1.6)
    p.add_axes(line_width=4, viewport=(0.0, 0.0, 0.10, 0.18))

    e, a = np.radians(elev), np.radians(azim)
    dvec = np.array([np.cos(e)*np.cos(a), -np.cos(e)*np.sin(a), np.sin(e)])
    c = np.array([0.5*(DEV_X0+DEV_X1), 0.0, 0.0])
    p.enable_parallel_projection()
    p.camera.parallel_scale = 0.112
    p.camera.focal_point = tuple(c)
    p.camera.position = tuple(c + dvec*2.0)
    p.camera.up = (0, 0, 1)
    p.screenshot(fname)
    p.close()
    draw_legend(fname, 0.0, vmax, title=title)
    return fname, vmax
