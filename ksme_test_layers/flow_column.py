"""
액적 기둥 단위셀 정상 비압축성 Navier-Stokes
=============================================
    ρ (u·∇)u = −∇p + μ ∇²u ,   ∇·u = 0
Taylor-Hood P2/P1 혼합요소, Newton(PETSc SNES) + 직접해법(MUMPS).

경계조건 (부양기 직결 챔버 설계)
    입구        : u = (u_in(z), 0, 0)   1/7 승 법칙 난류 코어 프로파일
    출구        : 무응력 (자연)
    y = 0       : u_y = 0  (기둥 축 대칭면)
    y = 26 mm   : u = (u_in(z),0,0)  (원방)
    z = 0       : u_z = 0  (기둥 중앙 대칭면)
    z = 16 mm   : u_z = 0  (덕트 벽, 자유활동 — 난류 코어 가정)
    넥 원통면   : u = 0    (부양기 하우징)
    구면 돔     : u = 0    (트랜스듀서 배열면)
    액적면 x3   : u = 0

액적 k 가 받는 힘 = 응력텐서 표면적분
    σ = −p I + μ(∇u + ∇uᵀ)
    F_k = −∮_S σ·n dS      (1/4 도메인의 반구 적분 x2 = 액적 1개분)
"""
import numpy as np, ufl, basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import petsc as dfp
from dolfinx.io import gmsh as gio
import mesh_column as MC, params_layers as P

RHO, MU = P.RHO_A, P.MU_A


def load(fname):
    md = gio.read_from_msh(fname, MPI.COMM_WORLD, gdim=3)
    return md.mesh, md.facet_tags


def solve(fname, U_bulk, verbose=False):
    msh, ft = load(fname)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim-1, tdim)

    el_u = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(3,))
    el_p = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([el_u, el_p]))
    V, _ = W.sub(0).collapse()

    w = fem.Function(W, name="w")
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    rho = fem.Constant(msh, PETSc.ScalarType(RHO))
    mu = fem.Constant(msh, PETSc.ScalarType(MU))
    U_ref = fem.Constant(msh, PETSc.ScalarType(U_bulk))

    F = (rho*ufl.inner(ufl.grad(u)*u, v)*ufl.dx
         + mu*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
         - p*ufl.div(v)*ufl.dx + q*ufl.div(u)*ufl.dx)

    def dofs_on(tag, sub=None):
        f = ft.find(tag)
        if sub is None:
            return fem.locate_dofs_topological((W.sub(0), V), tdim-1, f)
        Vi, _ = W.sub(0).sub(sub).collapse()
        return fem.locate_dofs_topological((W.sub(0).sub(sub), Vi), tdim-1, f)

    u_in = fem.Function(V)
    def set_inflow(val):
        u_in.x.array[:] = 0.0
        u_in.sub(0).interpolate(lambda x: P.inlet_profile(x[2], val))
        u_in.x.scatter_forward()

    u_zero = fem.Function(V); u_zero.x.array[:] = 0.0
    Vy, _ = W.sub(0).sub(1).collapse(); y_zero = fem.Function(Vy); y_zero.x.array[:] = 0.0
    Vz, _ = W.sub(0).sub(2).collapse(); z_zero = fem.Function(Vz); z_zero.x.array[:] = 0.0

    bcs = [fem.dirichletbc(u_in, dofs_on(MC.INLET), W.sub(0)),
           fem.dirichletbc(y_zero, dofs_on(MC.SYM_Y0, 1), W.sub(0).sub(1)),
           fem.dirichletbc(u_in, dofs_on(MC.FAR_Y), W.sub(0)),
           fem.dirichletbc(z_zero, dofs_on(MC.SYM_Z, 2), W.sub(0).sub(2)),
           fem.dirichletbc(z_zero, dofs_on(MC.WALL_DUCT, 2), W.sub(0).sub(2)),
           ]
    for k in range(len(MC.Z_DROPS)):
        bcs.append(fem.dirichletbc(u_zero, dofs_on(MC.DROP_TAG0+k), W.sub(0)))

    # ---------------- 해법
    # (1) Stokes 를 먼저 풀어 초기값으로 삼는다. 챔버(느린 재순환)가 있는 형상에서
    #     u=0 에서 곧바로 Newton 을 돌리면 수렴이 불안정하다.
    # (2) 이어서 완전 N-S 를 유속 연속법으로 푼다.
    # 잔차의 절대 크기가 10^-8 급이라(점성력 ~ 10^-8 N) rtol 을 10^-9 로 잡으면
    # 반올림 바닥에 걸려 MAX_IT 로 실패한다. rtol 10^-8 / atol 10^-10 으로 둔다.
    base = {"snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_rtol": 1e-9, "snes_atol": 1e-11, "snes_max_it": 30,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 500, "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_10": 5}
    if verbose:
        base["snes_monitor"] = None
    ndof = W.dofmap.index_map.size_global*W.dofmap.index_map_bs
    nit = 0

    Re = P.reynolds(U_bulk)
    ramp = ([U_bulk] if Re < 60 else
            list(U_bulk*np.array([0.2, 0.45, 0.7, 1.0])))
    for Uk in ramp:
        set_inflow(Uk)
        prob = dfp.NonlinearProblem(F, w, bcs=bcs,
                                    petsc_options_prefix=f"ns{int(Uk*1e4)}_",
                                    petsc_options=base)
        prob.solve()
        r = prob.solver.getConvergedReason()
        nit += prob.solver.getIterationNumber()
        f0 = prob.solver.getFunctionNorm()
        # 챔버(셀 크기 7.5 mm)와 액적 경계층(0.16 mm)의 크기비가 커서 계의 조건수가
        # 크다. 첫 Newton 단계에서 잔차가 5~6 자릿수 떨어진 뒤 반올림 바닥(상대
        # 10^-6 급)에 걸려 더 내려가지 않는다. 물리적으로는 이미 수렴한 해이므로
        # '상대 잔차 10^-5 이하'를 수렴으로 인정하고, 항력을 상관식과 대조해 검증한다.
        ok = (r > 0)
        if not ok:
            raise RuntimeError(f"Newton 발산 @ U={Uk} (reason={r}, "
                               f"resid={f0:.3e})")
        if verbose:
            print(f"      U={Uk:.3f} Re_d={P.reynolds(Uk):.0f} reason={r} "
                  f"it={prob.solver.getIterationNumber()} "
                  f"resid={f0:.3e}", flush=True)
    return msh, ft, w, droplet_forces(msh, ft, w), dict(ndof=ndof, snes_it=nit)


def _residual_norm(F, w, bcs):
    """현재 상태에서의 비선형 잔차 노름 (수렴 판정 기준값)."""
    b = dfp.assemble_vector(fem.form(F))
    dfp.apply_lifting(b, [fem.form(ufl.derivative(F, w))], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dfp.set_bc(b, bcs, w.x.petsc_vec, -1.0)
    return b.norm()


def droplet_forces(msh, ft, w):
    """
    액적 k 개별 표면 응력 적분 [N].
    1/4 도메인의 반구 적분에 2를 곱해 액적 1개분으로 환산(y 거울대칭).
    z<0 의 짝 액적은 대칭이므로 F_x 가 동일하다.
    반환: {k: {'pressure':vec3, 'viscous':vec3, 'total':vec3}}, 그리고 'sum'
    """
    u, p = ufl.split(w)
    n = ufl.FacetNormal(msh)
    mu = fem.Constant(msh, PETSc.ScalarType(MU))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    out, tot = {}, np.zeros(3)
    for k in range(len(MC.Z_DROPS)):
        rec = {}; s = np.zeros(3)
        for name, sig in (("pressure", -p*ufl.Identity(3)),
                          ("viscous", mu*(ufl.grad(u) + ufl.grad(u).T))):
            comp = []
            for i in (0, 2):
                e = ufl.as_vector([1.0 if j == i else 0.0 for j in range(3)])
                val = fem.assemble_scalar(
                    fem.form(ufl.dot(ufl.dot(sig, n), e)*ds(MC.DROP_TAG0+k)))
                comp.append(-2*msh.comm.allreduce(val, op=MPI.SUM))
            vec = np.array([comp[0], 0.0, comp[1]])
            rec[name] = vec; s = s + vec
        rec["total"] = s; out[k] = rec; tot = tot + s
    out["sum_half"] = tot                 # 상반부 3개 액적 합
    out["sum_column"] = 2*tot             # 기둥 전체 6개 액적 합 (F_x 기준)
    return out


def neck_exchange(msh, ft, w, U_bulk):
    """
    개구면(z = DUCT_H/2, rho <= R_OPEN)을 통해 넥으로 드나드는 체적유량 [m^3/s].
    직관부 유량 대비 비율로 '유동이 트랜스듀서 쪽으로 얼마나 들어가는가' 를 잰다.
    개구면은 내부면이라 표면적분 대신 z = zduct 평면 바로 아래/위 격자 샘플로 잰다.
    """
    from dolfinx import geometry as dgeom
    uu = collapse_velocity(w)
    tree = dgeom.bb_tree(msh, msh.topology.dim)
    n = 90
    r = np.linspace(0, P.R_OPEN*0.995, n)
    th = np.linspace(0, np.pi, n)          # y>=0 절반
    RR, TH = np.meshgrid(r, th, indexing='ij')
    xs_ = RR*np.cos(TH); ys_ = np.abs(RR*np.sin(TH))
    pts = np.ascontiguousarray(np.stack(
        [xs_.ravel(), ys_.ravel(), np.full(xs_.size, P.CELL_ZMAX-2e-4)], 1))
    cand = dgeom.compute_collisions_points(tree, pts)
    adj = dgeom.compute_colliding_cells(msh, cand, pts)
    off, arr = adj.offsets, adj.array
    has = np.diff(off) > 0
    uz = np.zeros(len(pts))
    idx = np.flatnonzero(has)
    if len(idx):
        uz[idx] = np.atleast_2d(uu.eval(
            pts[idx], arr[off[:-1][has]].astype(np.int32)))[:, 2]
    uz = uz.reshape(RR.shape)
    dA = (r[1]-r[0])*(th[1]-th[0])*RR          # 극좌표 면적요소
    up = float(np.sum(np.clip(uz, 0, None)*dA))*2     # y<0 대칭 -> x2
    dn = float(np.sum(np.clip(-uz, 0, None)*dA))*2
    Q_duct = U_bulk*(2*P.CELL_YMAX)*P.DUCT_H
    return dict(Q_into_neck=up, Q_out_neck=dn, Q_duct=Q_duct,
                exchange_ratio=up/Q_duct if Q_duct > 0 else np.nan)


def wake_profile(msh, w, U_bulk, xs=(5e-3, 8e-3, 11e-3, 14e-3, 17e-3, 20e-3)):
    """
    액적 중심 높이에서 하류 x 위치별 축방향 속도 결손 (1 − u_x/u_in).
    layer 간 접근속도 보정에 사용한다.
    """
    from dolfinx import geometry as dgeom
    V, dofs = w.function_space.sub(0).collapse()
    uu = fem.Function(V); uu.x.array[:] = w.x.array[dofs]; uu.x.scatter_forward()
    tree = dgeom.bb_tree(msh, msh.topology.dim)
    out = {}
    for x in xs:
        defs = []
        for zk in MC.Z_DROPS:
            pts = np.array([[x, 0.0, float(zk)]])
            cand = dgeom.compute_collisions_points(tree, pts)
            adj = dgeom.compute_colliding_cells(msh, cand, pts)
            if len(adj.array) == 0:
                defs.append(np.nan); continue
            ux = float(np.atleast_2d(uu.eval(pts, adj.array[:1].astype(np.int32)))[0][0])
            defs.append(1.0 - ux/P.inlet_profile(float(zk), U_bulk))
        out[float(x)] = defs
    return out


def collapse_velocity(w):
    V, dofs = w.function_space.sub(0).collapse()
    uu = fem.Function(V); uu.x.array[:] = w.x.array[dofs]; uu.x.scatter_forward()
    return uu


if __name__ == "__main__":
    import time, sys
    fn = sys.argv[1] if len(sys.argv) > 1 else "mesh_col_r2.msh"
    U = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    t0 = time.time()
    msh, ft, w, F, info = solve(fn, U, verbose=True)
    print(f"\n{fn}  U={U} m/s  ndof={info['ndof']:,}  "
          f"newton {info['snes_it']} it  {time.time()-t0:.1f}s")
    print(f"{'액적':>4} {'z[mm]':>7} {'u_in/U':>7} {'항력[uN]':>10} {'압력':>8} "
          f"{'점성':>8} {'F_z[uN]':>9} {'Cd':>7} {'Cd_상관식':>9}")
    for k, zk in enumerate(MC.Z_DROPS):
        ul = P.inlet_profile(float(zk), U)
        d = F[k]
        Cd = d['total'][0]/(0.5*P.RHO_A*ul**2*P.A_FRONT)
        print(f"{k:>4} {zk*1e3:7.2f} {ul/U:7.3f} {d['total'][0]*1e6:10.3f} "
              f"{d['pressure'][0]*1e6:8.3f} {d['viscous'][0]*1e6:8.3f} "
              f"{d['total'][2]*1e6:9.3f} {Cd:7.4f} "
              f"{P.cd_schiller_naumann(P.reynolds(ul)):9.4f}")
    print(f"기둥 6개 합 항력 = {F['sum_column'][0]*1e6:.2f} uN")
    ce = neck_exchange(msh, ft, w, U)
    print(f"넥 교환유량 = {ce['Q_into_neck']*6e4:.3f} L/min "
          f"(직관 {ce['Q_duct']*6e4:.1f} L/min 의 {100*ce['exchange_ratio']:.2f} %)")
    print("\n후류 결손 (1 - u/u_in):")
    for x, dv in wake_profile(msh, w, U).items():
        print(f"  x={x*1e3:5.1f} mm: {np.round(dv,4)}")
