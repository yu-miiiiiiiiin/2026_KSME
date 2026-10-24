"""
미세입자 Lagrangian 추적 (액적 기둥 단위셀)
============================================
지배방정식 (뉴턴 운동방정식 + 항력)

    m_p dv/dt = F_drag + F_grav + F_ac
    F_drag = 3π μ d_p (u_f − v) · f(Re_p),  f = 1 + 0.15 Re_p^0.687  (Schiller-Naumann)
    F_grav = m_p g (1 − ρ_a/ρ_p)
    F_ac   = −∇U_G |_(a = d_p/2)      (입자에 대한 Gor'kov 복사력, drive² 스케일)
    dx/dt  = v

유동장은 1/4 도메인 FE 해를 y, z 거울대칭으로 펼쳐 전체 기둥(액적 6개)에서 평가한다.
    u_x(x,-y,z)= u_x(x,y,z),  u_y(x,-y,z)=−u_y,  u_z(x,-y,z)= u_z
    u_x(x,y,-z)= u_x(x,y,z),  u_y(x,y,-z)= u_y,  u_z(x,y,-z)=−u_z
"""
import numpy as np
from dolfinx import geometry as dgeom
from scipy.interpolate import RegularGridInterpolator
import params_layers as P
import mesh_column as MC
import acoustics_layers as AC

Z_FULL = MC.full_column_z()          # 액적 6개의 z (덕트 중심 기준)


class MirrorProbe:
    """1/4 도메인 FE 속도장을 전체 단위셀로 펼쳐 평가. 도메인 밖은 입구 프로파일."""
    def __init__(self, msh, u_fn, U_bulk):
        self.msh, self.u, self.U = msh, u_fn, U_bulk
        self.tree = dgeom.bb_tree(msh, msh.topology.dim)

    def __call__(self, pts):
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        sy = np.where(pts[:, 1] < 0, -1.0, 1.0)
        sz = np.where(pts[:, 2] < 0, -1.0, 1.0)
        q = np.stack([pts[:, 0], np.abs(pts[:, 1]), np.abs(pts[:, 2])], 1)
        q = np.ascontiguousarray(q)
        out = np.zeros_like(q)
        out[:, 0] = P.inlet_profile(pts[:, 2], self.U)      # 기본값 = 자유류
        cand = dgeom.compute_collisions_points(self.tree, q)
        adj = dgeom.compute_colliding_cells(self.msh, cand, q)
        off, arr = adj.offsets, adj.array
        has = np.diff(off) > 0
        if has.any():
            idx = np.flatnonzero(has)
            val = np.atleast_2d(self.u.eval(q[idx], arr[off[:-1][has]].astype(np.int32)))
            out[idx] = val
        out[:, 1] *= sy
        out[:, 2] *= sz
        return out


class GridProbe:
    """
    FE 속도장을 계층 격자에 한 번만 샘플링해 두고 삼선형 보간으로 평가한다.
    (매 RK4 단계마다 dolfinx 셀 탐색을 하면 비용이 지배적이라 10~20배 느리다)

    * 격자는 1/4 도메인 좌표(y>=0, z>=0)에 만들고, 질의 시 거울대칭으로 펼친다.
    * 액적 내부 격자점은 무활동 조건에 맞춰 u = 0 으로 채운다. 자유류 값으로
      채우면 액적 표면 바로 바깥에서 보간이 오염되어 포집이 과대평가된다.
    * 도메인 밖(상류/하류/측면)은 입구 프로파일로 채운다.
    """
    #  (셀 크기, (x0,x1,y1,z1))  — 뒤로 갈수록 조밀. 액적 기둥 근방을 가장 조밀하게.
    LEVELS = ((8.0e-4, (-8.0e-3, 24.0e-3,  7.0e-3, 16.0e-3)),
              (2.5e-4, (-6.0e-3, 12.0e-3,  4.0e-3, 15.0e-3)),
              (8.0e-5, (-2.5e-3,  5.0e-3,  2.2e-3, 13.4e-3)))

    def __init__(self, msh, u_fn, U_bulk, verbose=False,
                 levels=None, centers=None, uniform=False):
        from dolfinx import geometry as dgeom
        self.U, self.uniform = U_bulk, uniform
        tree = dgeom.bb_tree(msh, msh.topology.dim)
        cen = (np.asarray(centers, float) if centers is not None else
               np.stack([np.zeros(len(MC.Z_DROPS)), np.zeros(len(MC.Z_DROPS)),
                         MC.Z_DROPS], 1))
        self.LEVELS = levels if levels is not None else GridProbe.LEVELS
        self.grids, self.itps, self.npts = [], [], 0
        for h, (x0, x1, y1, z1) in self.LEVELS:
            gx = np.arange(x0, x1+h/2, h)
            gy = np.arange(0.0, y1+h/2, h)
            gz = np.arange(0.0, z1+h/2, h)
            XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing='ij')
            q = np.ascontiguousarray(np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], 1))
            val = np.zeros((len(q), 3))
            val[:, 0] = (U_bulk if uniform
                         else P.inlet_profile(q[:, 2], U_bulk))   # 기본 = 자유류
            inside = (np.linalg.norm(q[:, None, :]-cen[None, :, :], axis=2)
                      .min(axis=1) <= P.A_DROP)
            val[inside] = 0.0                                  # 액적 내부 = 무활동
            cand = dgeom.compute_collisions_points(tree, q)
            adj = dgeom.compute_colliding_cells(msh, cand, q)
            off, arr = adj.offsets, adj.array
            has = np.diff(off) > 0
            idx = np.flatnonzero(has)
            if len(idx):
                val[idx] = np.atleast_2d(
                    u_fn.eval(q[idx], arr[off[:-1][has]].astype(np.int32)))
            self.grids.append((gx, gy, gz))
            self.itps.append([RegularGridInterpolator(
                (gx, gy, gz), val[:, c].reshape(XX.shape),
                bounds_error=False, fill_value=None) for c in range(3)])
            self.npts += len(q)
            if verbose:
                print(f"    격자 h={h*1e6:.0f} um : {len(q):,}점 "
                      f"(FE 적중 {100*len(idx)/len(q):.1f} %)", flush=True)

    def __call__(self, pts):
        pts = np.asarray(pts, float)
        sy = np.where(pts[:, 1] < 0, -1.0, 1.0)
        sz = np.where(pts[:, 2] < 0, -1.0, 1.0)
        q = np.stack([pts[:, 0], np.abs(pts[:, 1]), np.abs(pts[:, 2])], 1)
        out = np.zeros_like(q)
        out[:, 0] = (self.U if self.uniform
                     else P.inlet_profile(pts[:, 2], self.U))
        for (gx, gy, gz), itp in zip(self.grids, self.itps):
            m = ((q[:, 0] >= gx[0]) & (q[:, 0] <= gx[-1]) &
                 (q[:, 1] <= gy[-1]) & (q[:, 2] <= gz[-1]))
            if m.any():
                qq = q[m]
                out[m] = np.stack([f(qq) for f in itp], axis=1)
        out[:, 1] *= sy
        out[:, 2] *= sz
        return out


class HybridProbe(GridProbe):
    """
    액적 근방(중심에서 R_near 이내)은 FE 해를 직접 평가하고, 그 밖은 격자 보간.

    순수 격자 보간은 액적 표면의 경계층(δ ~ D/√Re ~ 0.14 mm)을 담지 못해
    표면 근방 속도를 과대평가하고 포집을 절반 수준으로 과소평가한다
    (validate_layers.py V3 참조). 반대로 전 구간을 FE 평가하면 셀 탐색 비용이
    지배적이라 10배 이상 느리다. 궤적 중 액적 근방에 머무는 구간은 10~15 %
    뿐이므로, 그 구간만 정확 평가하면 정확도와 속도를 모두 얻는다.
    """
    def __init__(self, msh, u_fn, U_bulk, R_near=3.0*P.A_DROP, **kw):
        from dolfinx import geometry as dgeom
        super().__init__(msh, u_fn, U_bulk, **kw)
        self.msh, self.u_fn = msh, u_fn
        self.tree = dgeom.bb_tree(msh, msh.topology.dim)
        self.R_near = R_near
        cen = kw.get('centers')
        self.cen_full = (np.asarray(cen, float) if cen is not None else
                         np.stack([np.zeros_like(Z_FULL), np.zeros_like(Z_FULL),
                                   Z_FULL], 1))
        self.n_exact = self.n_call = 0

    def __call__(self, pts):
        from dolfinx import geometry as dgeom
        pts = np.asarray(pts, float)
        out = super().__call__(pts)
        d = np.linalg.norm(pts[:, None, :] - self.cen_full[None, :, :],
                           axis=2).min(axis=1)
        m = d <= self.R_near
        self.n_call += len(pts); self.n_exact += int(m.sum())
        if m.any():
            sub = pts[m]
            sy = np.where(sub[:, 1] < 0, -1.0, 1.0)
            sz = np.where(sub[:, 2] < 0, -1.0, 1.0)
            q = np.ascontiguousarray(np.stack(
                [sub[:, 0], np.abs(sub[:, 1]), np.abs(sub[:, 2])], 1))
            val = out[m].copy()
            cand = dgeom.compute_collisions_points(self.tree, q)
            adj = dgeom.compute_colliding_cells(self.msh, cand, q)
            off, arr = adj.offsets, adj.array
            has = np.diff(off) > 0
            idx = np.flatnonzero(has)
            if len(idx):
                v = np.atleast_2d(self.u_fn.eval(
                    q[idx], arr[off[:-1][has]].astype(np.int32)))
                v[:, 1] *= sy[idx]; v[:, 2] *= sz[idx]
                val[idx] = v
            out[m] = val
        return out


class AcousticForceCell:
    """
    입자에 대한 Gor'kov 복사력을 단위셀 3D 격자에 사전계산해 두고 보간.
    셀 좌표 z 는 덕트 중심 기준이므로 음향 좌표로 z + Z_DUCT_OFFSET 을 쓴다.
    힘은 drive² 에 비례한다.
    """
    def __init__(self, h=5.0e-4, pad=1.5e-3, c_p=2500.0):
        P0 = AC.calibrate()
        pos, nrm = AC.unit_geometry()
        ph = np.zeros(len(pos))
        gx = np.arange(P.CELL_X0-pad, P.CELL_X1+pad+h/2, h)
        gy = np.arange(-P.CELL_YMAX-pad, P.CELL_YMAX+pad+h/2, h)
        gz = np.arange(-P.CELL_ZMAX-pad, P.CELL_ZMAX+pad+h/2, h)
        _, _, (Fx, Fy, Fz) = AC.gorkov_on_grid(
            gx, gy, gz + MC.Z_DUCT_OFFSET, pos, nrm, ph, P0,
            P.D_PART/2, P.RHO_PART, c_p)
        self.itp = [RegularGridInterpolator((gx, gy, gz), Fc, bounds_error=False,
                                            fill_value=0.0) for Fc in (Fx, Fy, Fz)]
        self.grid_pts = Fx.size
        self.Fmax = float(np.max(np.sqrt(Fx**2+Fy**2+Fz**2)))

    def __call__(self, x, drive=1.0):
        return drive*drive*np.stack([f(x) for f in self.itp], axis=1)


class ParticleTracker:
    """상태 코드: 0=진행, 1..6=액적 k 포집, 7=넥 유입, 8=덕트벽, 9=유출"""
    def __init__(self, probe, acoustic=None, drive=1.0,
                 d_p=P.D_PART, rho_p=P.RHO_PART, centers=None, bounds=None):
        self.probe = probe
        self.dp, self.rp = d_p, rho_p
        self.tau = rho_p*d_p**2/(18*P.MU_A)
        self.mp = rho_p*np.pi/6*d_p**3
        self.acoustic, self.drive = acoustic, drive
        self.centers = (np.asarray(centers, float) if centers is not None else
                        np.stack([np.zeros_like(Z_FULL), np.zeros_like(Z_FULL),
                                  Z_FULL], 1))          # (n_drop,3)
        self.bounds = bounds if bounds is not None else (
            P.CELL_X0, P.CELL_X1, P.CELL_YMAX, P.CELL_ZMAX)

    def accel(self, x, v):
        uf = self.probe(x)
        rel = uf - v
        Re_p = P.RHO_A*np.linalg.norm(rel, axis=1)*self.dp/P.MU_A
        f = 1.0 + 0.15*np.clip(Re_p, 0.0, None)**0.687
        a = (f[:, None]*rel)/self.tau
        a[:, 2] += -P.G*(1.0 - P.RHO_A/self.rp)
        if self.acoustic is not None:
            a += self.acoustic(x, self.drive)/self.mp
        return a

    def run(self, x0, v0, dt, nstep, keep_traj=False):
        x, v = x0.copy(), v0.copy()
        state = np.zeros(len(x), dtype=int)
        traj = [x.copy()] if keep_traj else None
        rcap = P.A_DROP + self.dp/2
        x0b, x1b, ymax, zmax = self.bounds
        for _ in range(nstep):
            act = state == 0
            if not act.any():
                break
            xa, va = x[act], v[act]
            k1x, k1v = va, self.accel(xa, va)
            k2x, k2v = va+0.5*dt*k1v, self.accel(xa+0.5*dt*k1x, va+0.5*dt*k1v)
            k3x, k3v = va+0.5*dt*k2v, self.accel(xa+0.5*dt*k2x, va+0.5*dt*k2v)
            k4x, k4v = va+dt*k3v,     self.accel(xa+dt*k3x,     va+dt*k3v)
            x[act] = xa + dt/6*(k1x+2*k2x+2*k3x+k4x)
            v[act] = va + dt/6*(k1v+2*k2v+2*k3v+k4v)
            # 액적 포집
            d = np.linalg.norm(x[:, None, :] - self.centers[None, :, :], axis=2)
            near = d.min(axis=1) <= rcap
            hit = (state == 0) & near
            if hit.any():
                state[hit] = 1 + np.argmin(d[hit], axis=1)
            # 덕트 천장/바닥 도달. 개구(rho <= R_OPEN) 안이면 부양기 넥으로 빨려
            # 들어간 것(-> 트랜스듀서 오염 경로), 밖이면 덕트 벽 부착.
            top = (state == 0) & (np.abs(x[:, 2]) >= zmax)
            if top.any():
                rho = np.hypot(x[top, 0], x[top, 1])
                idx = np.flatnonzero(top)
                state[idx[rho <= P.R_OPEN]] = 7      # 넥 유입
                state[idx[rho > P.R_OPEN]] = 8       # 덕트 벽
            # 유출
            out = ((x[:, 0] > x1b) | (x[:, 0] < x0b) | (np.abs(x[:, 1]) > ymax))
            state[(state == 0) & out] = 9
            if keep_traj:
                traj.append(x.copy())
        return (np.array(traj) if keep_traj else None), state


def capture_area(n_cap, n_seed, A_seed):
    """포집 단면적 [m²] 과 이항분포 표준오차."""
    q = n_cap/n_seed
    A = q*A_seed
    dA = np.sqrt(max(q*(1-q), 1e-12)/n_seed)*A_seed
    return A, dA


def study(probe, acf, drive, U_bulk, seed, n_imp=P.N_SEED_IMP, n_dir=P.N_SEED_DIR,
          keep_traj=False):
    """
    (A) 중요도 표본추출 : 액적 6개 각각의 포집가능 반경 안에 조밀 시딩 -> 액적별 A_cap
    (B) 직접법          : 단위셀 입구 단면 균일 시딩 -> 입출구 개수비 직접 비교
    """
    trk = ParticleTracker(probe, acoustic=acf, drive=drive)
    rng = np.random.default_rng(seed)
    dt = min(0.20*trk.tau, 0.05*P.D_DROP/U_bulk)
    nstep = int(2.6*(P.CELL_X1-P.CELL_X0)/(U_bulk*dt))
    xin = P.CELL_X0 + 5e-4

    # ---------------- (A) 적응형 중요도 표본추출 (액적별)
    #   1차(예비) : 넓은 반경 R0 로 포집 반경을 개산
    #   2차(본)   : R = clip(2.2 r_cap) 로 좁혀 같은 입자수로 통계오차를 줄인다
    #               (저유속에서는 포집반경이 0.2 mm 수준이라 넓게 뿌리면 낭비)
    def _seed_run(radii, npd, rng, off=None):
        off = off if off is not None else [(0.0, 0.0)]*len(Z_FULL)
        xs, tg = [], []
        for j, zk in enumerate(Z_FULL):
            R = radii[j]; oy, oz = off[j]
            rr = R*np.sqrt(rng.random(npd)); th = rng.random(npd)*2*np.pi
            xs.append(np.stack([np.full(npd, xin), oy + rr*np.cos(th),
                                zk + oz + rr*np.sin(th)], 1))
            tg.append(np.full(npd, j))
        xa = np.concatenate(xs); tg = np.concatenate(tg)
        va = np.stack([P.inlet_profile(xa[:, 2], U_bulk),
                       np.zeros(len(xa)), np.zeros(len(xa))], 1)
        return xa, tg, va

    R0 = P.R_SEED_FAC*P.A_DROP
    n_pilot = 240
    xp, tp, vp = _seed_run([R0]*len(Z_FULL), n_pilot, rng)
    _, sp = trk.run(xp, vp, dt, nstep)
    radii, offs = [], []
    for j in range(len(Z_FULL)):
        m = tp == j
        cap = sp[m] == j+1
        q = cap.mean()
        r_cap = np.sqrt(max(q, 1e-4))*R0
        # 포집된 입자의 시딩 위치 중심 (중력 침강으로 액적 중심에서 벗어난다)
        if cap.sum() >= 5:
            oy = float(xp[m][cap][:, 1].mean())
            oz = float(xp[m][cap][:, 2].mean() - Z_FULL[j])
        else:
            oy = oz = 0.0
        radii.append(float(np.clip(2.6*r_cap + 1.5e-4, 0.5*P.A_DROP, R0)))
        offs.append((oy, oz))

    per = int(np.ceil(n_imp/len(Z_FULL)))
    xa, tag, va = _seed_run(radii, per, rng, offs)
    tja, sa = trk.run(xa, va, dt, nstep, keep_traj=keep_traj)

    per_drop, A_tot, var_tot = [], 0.0, 0.0
    for j in range(len(Z_FULL)):
        m = tag == j
        nc = int((sa[m] == j+1).sum())          # 자기 액적에 잡힌 개수
        A, dA = capture_area(nc, int(m.sum()), np.pi*radii[j]**2)
        # 안전점검: 포집된 입자의 시딩 반경이 R 의 90 % 를 넘으면 R 이 부족
        rr0 = np.hypot(xa[m, 1]-offs[j][0],
                       xa[m, 2]-Z_FULL[j]-offs[j][1])[sa[m] == j+1]
        frac_edge = float((rr0 > 0.9*radii[j]).mean()) if nc else 0.0
        ul = float(P.inlet_profile(Z_FULL[j], U_bulk))
        per_drop.append(dict(j=int(j), z_m=float(Z_FULL[j]), u_local=ul,
                             n_cap=nc, n_seed=int(m.sum()),
                             R_seed_m=radii[j], seed_off=offs[j],
                             frac_edge=frac_edge,
                             A_cap_m2=A, A_cap_err_m2=dA,
                             eta=A/(np.pi*P.A_DROP**2)))
        A_tot += A; var_tot += dA*dA
    Rs = float(np.mean(radii))
    A_col = A_tot                                # 기둥 1개(액적 6개) 총 포집 단면적
    dA_col = np.sqrt(var_tot)

    # ---------------- (B) 직접법
    yd = rng.uniform(-P.CELL_YMAX, P.CELL_YMAX, n_dir)
    zd = rng.uniform(-P.CELL_ZMAX, P.CELL_ZMAX, n_dir)
    xd = np.stack([np.full(n_dir, xin), yd, zd], 1)
    vd = np.stack([P.inlet_profile(zd, U_bulk), np.zeros(n_dir), np.zeros(n_dir)], 1)
    _, sd = trk.run(xd, vd, dt, nstep)
    A_seed_dir = (2*P.CELL_YMAX)*(2*P.CELL_ZMAX)
    ncd = int(((sd >= 1) & (sd <= 6)).sum())
    A_dir, dA_dir = capture_area(ncd, n_dir, A_seed_dir)
    n_wall = int((sd == 8).sum())
    n_neck = int((sd == 7).sum())

    return dict(per_drop=per_drop, A_col_m2=A_col, A_col_err_m2=dA_col,
                eta_col=A_col/(len(Z_FULL)*np.pi*P.A_DROP**2),
                A_col_direct_m2=A_dir, A_col_direct_err_m2=dA_dir,
                n_cap_direct=ncd, n_seed_direct=n_dir, A_seed_direct_m2=A_seed_dir,
                wall_frac=n_wall/n_dir, neck_frac=n_neck/n_dir,
                dt=dt, nstep=nstep,
                n_seed_imp=int(len(xa)), R_seed_m=Rs), (tja, sa, xa, tag)
