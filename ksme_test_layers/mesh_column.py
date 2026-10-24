"""
액적 기둥 단위셀 메시 (gmsh) — 부양기 직결(챔버) 설계
=======================================================
액적 교체가 가능해야 하므로 덕트 개구를 멤브레인으로 덮지 않고, 덕트를 부양기에
직접 연결한다. 각 기둥 위치에서 덕트 천장/바닥이 열려 부양기 챔버로 이어진다.

  직관부 : |z| <= DUCT_H/2 = 16 mm
  개구/넥: rho <= R_OPEN = 20 mm 의 원통, z = 16 -> 53.4 mm (부양기 하우징)
           (덕트 벽면에서 집속빔 반경은 10.45 mm 뿐이므로 20 mm 면 충분하다)
  넥 천장: 트랜스듀서 배열면 = 음향 초점 중심 반경 57 mm 구면

  ** 생산 계산에서는 넥을 격자로 뚫지 않는다 (build(neck=False)). **
  개구 지름 기준 Re_D = 1.3x10^3 ~ 6.6x10^3 로 공동 전단층이 비정상이라 정상
  Navier-Stokes 해가 존재하지 않는다. 막다른 공동은 순 질량교환이 0 이므로
  시간평균 효과로서 개구면을 자유활동 평면으로 두고 주유동을 푼다.
  넥 유입/트랜스듀서 오염은 neck_model.py 에서 전단층 유입 상관식으로 추정한다.

대칭성 이용 (계산량 1/4)
  * y = 0                : 기둥 축을 포함하는 거울대칭면   -> u_y = 0
  * z = 0                : 기둥 중앙면(액적 상하 대칭)     -> u_z = 0
  * y = CELL_YMAX = 35 mm: 이웃 기둥과의 중앙면            -> u_y = 0
따라서 1/4 도메인에는 액적 3개의 반구와 챔버의 1/2 이 들어간다.

경계
  x = x0   입구  : 1/7 승 법칙 난류 코어 프로파일 u_x(z)
  x = x1   출구  : 무응력(자연)
  y = 0         : 대칭 (u_y = 0)
  y = 26 mm     : 원방 (u = (u_in(z),0,0))
  z = 0         : 대칭 (u_z = 0)
  z = 16 mm 평면: 덕트 벽, 자유활동 (난류 코어 가정)
  넥 원통면     : 부양기 하우징 벽, 무활동
  구면 돔       : 트랜스듀서 배열면, 무활동  <- 입자 부착 = 트랜스듀서 오염
  액적면 x3     : 무활동 (액적마다 별도 태그 -> 개별 항력 적분)
"""
import numpy as np, gmsh
import params_layers as P

INLET, OUTLET, SYM_Y0, FAR_Y, SYM_Z, WALL_DUCT, WALL_NECK, WALL_ARRAY = \
    2, 3, 4, 5, 6, 7, 8, 9
DROP_TAG0 = 20                       # 액적 k 의 태그 = DROP_TAG0 + k
FLUID = 1

# 1/4 도메인 안의 액적 z 위치 [m] (덕트 중심 기준, 상하 대칭화된 값)
Z_DROPS = np.array([2.3248e-3, 7.0168e-3, 11.7295e-3])
Z_DUCT_OFFSET = -0.553e-3            # 덕트 중심 - 음향 초점 [m]


def build(fname, h_drop=1.40e-4, h_wake=None, h_far=2.2e-3, h_cham=5.0e-3,
          neck=False, verbose=False):
    x0, x1 = P.CELL_X0, P.CELL_X1
    ymax, zduct = P.CELL_YMAX, P.CELL_ZMAX
    a, D = P.A_DROP, P.D_DROP
    h_wake = h_wake if h_wake is not None else 5.0*h_drop

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("droplet_column_chamber")
    occ = gmsh.model.occ

    # 직관부 + 챔버(원통 ∩ 구) 를 합치고 액적을 뺀다
    duct = occ.addBox(x0, 0.0, 0.0, x1-x0, ymax, zduct)
    if neck:
        # (참고용) 넥을 실제로 뚫은 형상. 개구 위 전단층이 비정상이라
        # 정상 N-S 해가 존재하지 않아 생산 계산에는 쓰지 않는다.
        cyl = occ.addCylinder(0, 0, zduct, 0, 0, P.Z_APEX-zduct+1e-3, P.R_OPEN)
        sph = occ.addSphere(0, 0, 0, P.R_CURV)
        clip = occ.addBox(x0, 0.0, zduct, x1-x0, ymax, P.Z_APEX-zduct+2e-3)
        ch, _ = occ.intersect([(3, cyl)], [(3, sph)], removeObject=True,
                              removeTool=True)
        ch, _ = occ.intersect(ch, [(3, clip)], removeObject=True, removeTool=True)
        fused, _ = occ.fuse([(3, duct)], ch, removeObject=True, removeTool=True)
    else:
        fused = [(3, duct)]
    tools = [(3, occ.addSphere(0.0, 0.0, float(zk), a)) for zk in Z_DROPS]
    occ.cut(fused, tools, removeObject=True, removeTool=True)
    occ.synchronize()

    vols = gmsh.model.getEntities(3)
    assert len(vols) == 1, f"volume 분할 이상: {len(vols)}"
    gmsh.model.addPhysicalGroup(3, [vols[0][1]], FLUID, "fluid")

    # 경계 분류. 곡면은 무게중심이 면 위에 없으므로 bounding box 로 판정한다.
    tol = 1e-5
    g = {k: [] for k in (INLET, OUTLET, SYM_Y0, FAR_Y, SYM_Z,
                         WALL_DUCT, WALL_NECK, WALL_ARRAY)}
    gd = {k: [] for k in range(len(Z_DROPS))}
    other = []
    for (_, s_) in gmsh.model.getEntities(2):
        cx, cy, cz = occ.getCenterOfMass(2, s_)
        bx0, by0, bz0, bx1, by1, bz1 = gmsh.model.getBoundingBox(2, s_)
        hit = None
        for k, zk in enumerate(Z_DROPS):
            if np.sqrt(cx**2 + cy**2 + (cz-zk)**2) < 1.6*a:
                hit = k; break
        if hit is not None:                         gd[hit].append(s_)
        elif abs(bx1-x0) < tol:                     g[INLET].append(s_)
        elif abs(bx0-x1) < tol:                     g[OUTLET].append(s_)
        elif abs(by1) < tol:                        g[SYM_Y0].append(s_)
        elif abs(by0-ymax) < tol:                   g[FAR_Y].append(s_)
        elif abs(bz1) < tol:                        g[SYM_Z].append(s_)
        elif abs(bz0-zduct) < tol and abs(bz1-zduct) < tol:
            g[WALL_DUCT].append(s_)
        elif abs(bz0-zduct) < tol and abs(bz1-P.Z_NECK) < 1e-4:
            g[WALL_NECK].append(s_)
        elif bz0 > zduct - tol:                     g[WALL_ARRAY].append(s_)
        else: other.append((s_, cx, cy, cz, bz0, bz1))
    assert not other, f"미분류 경계면 {other}"
    names = {INLET: "inlet", OUTLET: "outlet", SYM_Y0: "sym_y0",
             FAR_Y: "far_y", SYM_Z: "sym_z", WALL_DUCT: "wall_duct",
             WALL_NECK: "wall_neck", WALL_ARRAY: "wall_array"}
    need = (INLET, OUTLET, SYM_Y0, FAR_Y, SYM_Z, WALL_DUCT)
    for tag, ss in g.items():
        assert ss or (tag not in need), f"경계 {names[tag]} 분류 실패"
        if ss:
            gmsh.model.addPhysicalGroup(2, ss, tag, names[tag])
    for k, ss in gd.items():
        assert ss, f"액적 {k} 표면 분류 실패"
        gmsh.model.addPhysicalGroup(2, ss, DROP_TAG0+k, f"droplet{k}")

    # ---------------- 메시 크기장
    allsph = [s for ss in gd.values() for s in ss]
    f = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f, "SurfacesList", allsph)
    gmsh.model.mesh.field.setNumber(f, "Sampling", 300)
    t = gmsh.model.mesh.field.add("Threshold")
    for k, v in dict(InField=f, SizeMin=h_drop, SizeMax=h_far,
                     DistMin=0.6*a, DistMax=3*D).items():
        gmsh.model.mesh.field.setNumber(t, k, v)
    bx = gmsh.model.mesh.field.add("Box")        # 기둥 + 후류
    for k, v in dict(VIn=h_wake, VOut=h_far, XMin=-1.2*D, XMax=6*D,
                     YMin=-1.4*D, YMax=1.4*D,
                     ZMin=-0.8*D, ZMax=float(Z_DROPS[-1])+1.4*D,
                     Thickness=1.2*D).items():
        gmsh.model.mesh.field.setNumber(bx, k, v)
    flds = [t, bx]
    cb = gmsh.model.mesh.field.add("Box")        # 넥(느린 재순환) 은 성기게
    for k, v in dict(VIn=h_cham, VOut=h_far, XMin=x0, XMax=x1,
                     YMin=-ymax, YMax=ymax, ZMin=zduct+4e-3, ZMax=P.Z_APEX+1e-3,
                     Thickness=5e-3).items():
        gmsh.model.mesh.field.setNumber(cb, k, v)
    if neck:
        mx = gmsh.model.mesh.field.add("Max")
        gmsh.model.mesh.field.setNumbers(mx, "FieldsList", [cb])
        flds.append(mx)
    mn = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(mn, "FieldsList", flds)
    gmsh.model.mesh.field.setAsBackgroundMesh(mn)
    for o in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature",
              "Mesh.MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber(o, 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Optimize", 1)

    gmsh.model.mesh.generate(3)
    gmsh.write(fname)
    ne = len(gmsh.model.mesh.getElementsByType(4)[0])
    nn = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    return dict(file=fname, tets=ne, nodes=nn)


def cell_bounds():
    return (P.CELL_X0, P.CELL_X1, P.CELL_YMAX, P.CELL_ZTOP)


def full_column_z():
    """전체 기둥 6개 액적의 z 위치 (덕트 중심 기준) [m]."""
    return np.concatenate([-Z_DROPS[::-1], Z_DROPS])


if __name__ == "__main__":
    import time
    print(f"1/4 단위셀: x[{P.CELL_X0*1e3:.0f},{P.CELL_X1*1e3:.0f}] "
          f"y[0,{P.CELL_YMAX*1e3:.0f}] z[0,{P.CELL_ZTOP*1e3:.0f}] mm")
    print(f"  직관부 반높이 {P.CELL_ZMAX*1e3:.0f} mm, 개구 반경 {P.R_OPEN*1e3:.1f} mm, "
          f"칼라 {P.COLLAR_H*1e3:.1f} mm, 돔 정점 {P.Z_APEX*1e3:.1f} mm")
    print(f"  반구 {len(Z_DROPS)}개 z={np.round(Z_DROPS*1e3,3)} mm")
    for k, h in enumerate(P.H_DROP_REP):
        t0 = time.time(); i = build(f"mesh_col_r{k}.msh", h_drop=h)
        print(f"  mesh_col_r{k}.msh: {i['tets']:,} tets, {i['nodes']:,} nodes "
              f"(h={h*1e6:.0f} um, {time.time()-t0:.1f}s)")
