"""
다중 초음파 부양기(10 유닛 x 72 트랜스듀서 = 720 소자) 음향장과 액적 복사력
=============================================================================
지배방정식
---------
(1) 정상 음향장 : Helmholtz 방정식  ∇²p + k²p = 0
    각 트랜스듀서를 원형 피스톤 소스로 보고 자유장 Green 함수를 소스면 위에서
    적분(Rayleigh-Sommerfeld)한 형태로 중첩한다.

        p(r) = Σ_j  P0 · D_j(θ)/d_j · exp( i(k d_j + φ_j) )
        D(θ) = 2 J1(k a_t sinθ)/(k a_t sinθ)         (원형 피스톤 지향성)

    10개 유닛 전체(720 소자)를 한 번에 중첩하므로 유닛 간 간섭(crosstalk)이
    자동으로 포함된다.

(2) 음향 복사력 : 반경 a 구에 대한 Gor'kov 포텐셜

        U_G = (4/3)πa³ [ f1·(1/2)κ0⟨p²⟩ − f2·(3/4)ρ0⟨|v|²⟩ ]
        f1 = 1 − κp/κ0 ,  f2 = 2(ρp−ρ0)/(2ρp+ρ0)
        ⟨p²⟩ = |p|²/2 ,  ⟨|v|²⟩ = |∇p|²/(2(ωρ0)²)
        F_ac = −∇U_G

(3) 구동 세기 : 음압 p0 ∝ 인가전압 V  ->  F_ac ∝ p0² ∝ V²
    drive = V/V0 로 두면  F_ac(drive) = drive² · F_ac(1)
    (전류도 전압에 비례하므로 전기 입력전력 ∝ V² ∝ F_ac 이다)
"""
import numpy as np
from scipy.special import j1
import params_layers as P


# ================================================================ 배열 기하
def _one_face(center, z_focus, flip, R=P.R_CURV, counts=P.N_RING,
              off=(0.0, np.pi/12, np.pi/18)):
    """구면 캡 한 면(36 소자)의 트랜스듀서 위치와 법선."""
    s = np.array([2*P.A_TRANS/(2*np.sin(np.pi/n)) for n in counts])  # 링 접선 조건
    pos, nrm = [], []
    for th, n, o in zip(s/R, counts, off):
        phi = np.arange(n)*2*np.pi/n + o
        z = z_focus + (R*np.cos(th) if flip else -R*np.cos(th))
        p = np.stack([center[0] + R*np.sin(th)*np.cos(phi),
                      center[1] + R*np.sin(th)*np.sin(phi),
                      np.full(n, z)], 1)
        d = np.array([center[0], center[1], z_focus]) - p
        nrm.append(d/np.linalg.norm(d, axis=1, keepdims=True))
        pos.append(p)
    return np.concatenate(pos), np.concatenate(nrm)


def unit_geometry(center=(0.0, 0.0)):
    """유닛 1개(72 소자)의 위치/법선."""
    pb, nb = _one_face(center, 0.0, False)
    pt, nt = _one_face(center, 0.0, True)
    return np.concatenate([pb, pt]), np.concatenate([nb, nt])


def array_geometry(coherent=True, seed=0):
    """
    10 유닛(720 소자) 전체 기하.
    coherent=True  : 모든 유닛이 같은 위상 기준 (단일 발진기 구동)
    coherent=False : 유닛별 독립 위상 (개별 드라이버) -> 난수 위상 오프셋
    반환: pos(720,3), nrm(720,3), phase(720,), unit_id(720,), centers(10,2)
    """
    rng = np.random.default_rng(seed)
    pos, nrm, ph, uid, cen = [], [], [], [], []
    k = 0
    for L, cols in enumerate(P.layer_columns()):
        for (xc, yc) in cols:
            p, n = unit_geometry((xc, yc))
            dphi = 0.0 if coherent else float(rng.uniform(0, 2*np.pi))
            pos.append(p); nrm.append(n)
            ph.append(np.full(len(p), dphi)); uid.append(np.full(len(p), k))
            cen.append((xc, yc, L)); k += 1
    return (np.concatenate(pos), np.concatenate(nrm), np.concatenate(ph),
            np.concatenate(uid), np.array(cen))


# ================================================================ 압력장
def pressure(points, pos, nrm, phase, p0, chunk=60_000):
    """복소 압력 진폭 [Pa]."""
    pts = np.atleast_2d(np.asarray(points, float))
    out = np.zeros(len(pts), complex)
    ka = P.K_WAVE*P.A_TRANS
    for i in range(0, len(pts), chunk):
        r = pts[i:i+chunk]
        dv = r[:, None, :] - pos[None, :, :]
        d = np.maximum(np.linalg.norm(dv, axis=2), 1e-9)
        ct = np.einsum('mnj,nj->mn', dv, nrm)/d
        arg = ka*np.sqrt(np.clip(1-ct**2, 0, 1))
        safe = np.where(arg < 1e-9, 1.0, arg)
        D = np.where(arg < 1e-9, 1.0, 2*j1(safe)/safe)
        # Rayleigh-Sommerfeld 1종 경사인자 (1+cosθ)/2 : 후방 방사를 매끄럽게 억제.
        # θ=90° 에서 D 를 0 으로 자르면 불연속이 생겨 ∇p, ∇U 계산이 오염된다
        # (단일 격자점 스파이크). 자기 유닛 축 위에서는 cosθ≈1 이라 영향이 없다.
        obl = 0.5*(1.0 + ct)
        out[i:i+chunk] = np.sum(p0*D*obl/d
                                * np.exp(1j*(P.K_WAVE*d + phase)), axis=1)
    return out


def calibrate(target=P.P_ANTINODE_REF):
    """
    고립된 유닛 1개의 축상 최대 |p| 가 target 이 되도록 소자 세기 P0 [Pa·m] 결정.
    (10 유닛 모두 동일 구동이므로 보정은 유닛 단위로 한다)
    """
    pos, nrm = unit_geometry()
    ph = np.zeros(len(pos))
    z = np.linspace(-0.03, 0.03, 6001)
    a = np.abs(pressure(np.stack([0*z, 0*z, z], 1), pos, nrm, ph, 1.0))
    return target/a.max()


# ================================================================ Gor'kov
def contrast(rho_p, c_p):
    kap0 = 1.0/(P.RHO_A*P.C0**2)
    f1 = 1.0 - (1.0/(rho_p*c_p**2))/kap0
    f2 = 2.0*(rho_p - P.RHO_A)/(2.0*rho_p + P.RHO_A)
    return kap0, f1, f2


def phi_contrast(rho_p, c_p):
    """평면정재파 표준형 음향 대비인자 Φ = (f1 + 1.5 f2)/3."""
    _, f1, f2 = contrast(rho_p, c_p)
    return (f1 + 1.5*f2)/3.0


def gorkov(Pc, spacing, a_sphere, rho_p, c_p):
    """복소 압력장 -> Gor'kov 포텐셜 U [J] 와 힘 F = −∇U [N]."""
    kap0, f1, f2 = contrast(rho_p, c_p)
    grads = np.gradient(Pc, *spacing)
    if Pc.ndim == 1:
        grads = [grads]
    v2 = sum(np.abs(g)**2 for g in grads)/(P.OMEGA*P.RHO_A)**2
    U = (4/3)*np.pi*a_sphere**3*(f1*0.5*kap0*(0.5*np.abs(Pc)**2)
                                 - f2*0.75*P.RHO_A*(0.5*v2))
    F = np.gradient(U, *spacing)
    if Pc.ndim == 1:
        F = [F]
    return U, tuple(-g for g in F)


def gorkov_along(pts, axis, ds, pos, nrm, ph, P0, a_sphere,
                 rho_p=P.RHO_W, c_p=P.C_W, h_t=2.0e-5):
    """
    직선 위에서 Gor'kov 포텐셜과 힘 3성분을 계산한다.

    주의 : U_G 는 |∇p|² 를 포함하므로 직선 방향 미분만으로는 계산할 수 없다.
    (예: 트랩 높이에서 x 방향으로 훑으면 지배적인 ∂p/∂z 성분이 누락된다)
    따라서 직선 주위에 두께 3점(±h_t)의 얇은 3D 슬랩을 만들어 ∇p 를 온전히
    구한 뒤, 슬랩 중앙선의 U 를 다시 미분해 F = −∇U 를 얻는다.

    pts  : (n,3) 직선 위 점들 (axis 방향으로 등간격 ds)
    axis : 0/1/2 (x/y/z)
    반환 : Pc(n), U(n), (Fx, Fy, Fz) 각 (n,)
    """
    n = len(pts)
    o1, o2 = [i for i in range(3) if i != axis]
    off = np.array([-h_t, 0.0, h_t])
    grid = np.repeat(pts[:, None, None, :], 3, axis=1).repeat(3, axis=2)
    grid[..., o1] += off[None, :, None]
    grid[..., o2] += off[None, None, :]
    Pc = pressure(grid.reshape(-1, 3), pos, nrm, ph, P0).reshape(n, 3, 3)
    sp = [0.0]*3
    sp[axis], sp[o1], sp[o2] = ds, h_t, h_t
    U3, F3 = gorkov(Pc, (sp[axis], h_t, h_t), a_sphere, rho_p, c_p)
    F = [None]*3
    F[axis], F[o1], F[o2] = F3[0][:, 1, 1], F3[1][:, 1, 1], F3[2][:, 1, 1]
    return Pc[:, 1, 1], U3[:, 1, 1], tuple(F)


def force_on_line(pts, ds, pos, nrm, ph, P0, a_sphere,
                  rho_p=P.RHO_W, c_p=P.C_W, axis=2):
    """gorkov_along 의 축방향 성분만 돌려주는 편의 함수."""
    Pc, U, F = gorkov_along(pts, axis, ds, pos, nrm, ph, P0, a_sphere, rho_p, c_p)
    return Pc, U, F[axis]


def gorkov_on_grid(x, y, z, pos, nrm, ph, P0, a_sphere, rho_p, c_p):
    """3D 격자 위 U_G 와 (Fx,Fy,Fz)."""
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
    Pc = pressure(np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], 1),
                  pos, nrm, ph, P0).reshape(XX.shape)
    sp = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
    U, F = gorkov(Pc, sp, a_sphere, rho_p, c_p)
    return Pc, U, F


# ================================================================ 트랩 탐색
def _local_max(v):
    return np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1


def _local_min(v):
    return np.where((v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:]))[0] + 1


def column_traps(xc, yc, pos, nrm, ph, P0, n_keep=P.N_PER_COL,
                 zhalf=0.026, nz=5201):
    """
    기둥 (xc,yc) 축을 따라 액적 트랩을 찾는다.

    반환 리스트 원소(액적 1개):
      z_node    압력 노드 위치 [m]            (무중력 평형)
      z_eq      중력 하 평형 위치 [m]         (F_ac,z = W)
      sag       z_node − z_eq  [m]            (중력 처짐)
      F_cap     트랩이 낼 수 있는 최대 상향력 [N] (= 평형 아래쪽 F_z 극대)
      k_ax      축방향 강성 −dF_z/dz  [N/m]
      p_node    노드에서의 잔류 |p| [Pa]
    """
    z = np.linspace(-zhalf, zhalf, nz)
    pts = np.stack([np.full(nz, xc), np.full(nz, yc), z], 1)
    Pc, U, Fz = force_on_line(pts, z[1]-z[0], pos, nrm, ph, P0, P.A_DROP, axis=2)
    A = np.abs(Pc)
    nodes = _local_min(A)                       # 압력 노드
    caps = _local_max(Fz)                       # 상향 최대력 지점
    out = []
    for i in nodes:
        zn = z[i]
        below = caps[(z[caps] < zn) & (z[caps] > zn - 0.9*P.LAM/2)]
        if len(below) == 0:
            continue
        j = below[np.argmax(Fz[below])]
        Fcap = float(Fz[j])
        # 평형점: F_z = W 이고 감소하는 교차 (극대점 j 위쪽에서 탐색)
        seg = slice(j, min(j + int(0.9*(P.LAM/2)/(z[1]-z[0])), nz-1))
        g = Fz[seg] - P.W_DROP
        zz = z[seg]
        idx = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
        if len(idx) == 0:
            z_eq, k_ax = np.nan, np.nan
        else:
            m = idx[0]
            z_eq = float(np.interp(0.0, [g[m+1], g[m]], [zz[m+1], zz[m]]))
            h = 2e-5
            k_ax = float(-(np.interp(z_eq+h, z, Fz) - np.interp(z_eq-h, z, Fz))/(2*h))
        out.append(dict(z_node=float(zn), z_eq=z_eq, sag=float(zn-z_eq),
                        F_cap=Fcap, k_ax=k_ax, p_node=float(A[i]),
                        Fz_over_W=Fcap/P.W_DROP))
    out = [o for o in out if np.isfinite(o['z_eq'])]
    out.sort(key=lambda o: -o['F_cap'])
    keep = sorted(out[:n_keep], key=lambda o: o['z_node'])
    return keep, dict(z=z, absP=A, U=U, Fz=Fz)


def lateral_scan(xc, yc, z0, pos, nrm, ph, P0, direction, half=0.004, n=801):
    """
    트랩 높이 z0 에서 direction('x' 또는 'y') 방향 복사력 프로파일.
    반환: s(변위), F(복원력 성분), k_lat(강성), F_esc(최대 복원력), s_esc(그 위치)
    """
    s = np.linspace(-half, half, n)
    if direction == 'x':
        pts = np.stack([xc+s, np.full(n, yc), np.full(n, z0)], 1)
    else:
        pts = np.stack([np.full(n, xc), yc+s, np.full(n, z0)], 1)
    _, _, F = force_on_line(pts, s[1]-s[0], pos, nrm, ph, P0, P.A_DROP,
                            axis=0 if direction == 'x' else 1)
    h = 2e-5
    k = float(-(np.interp(h, s, F) - np.interp(-h, s, F))/(2*h))
    pos_side = s >= 0
    F_esc = float(np.max(-F[pos_side]))
    s_esc = float(s[pos_side][np.argmax(-F[pos_side])])
    return s, F, k, F_esc, s_esc


# ================================================================ 기준장 생성
def build_reference(fname="acoustic_layers_ref.npz", coherent=True, seed=0,
                    verbose=True):
    """
    10 유닛 전체 배열의 기준(drive=1) 음향장을 계산해 60개 액적의 트랩 특성을 저장.
    유닛 단독(72소자) 결과도 함께 계산해 crosstalk 영향을 정량화한다.
    """
    P0 = calibrate()
    pos, nrm, ph, uid, cen = array_geometry(coherent=coherent, seed=seed)
    # 단독 유닛 (원점 중심)
    sp, sn = unit_geometry(); sph = np.zeros(len(sp))

    recs = []
    for c, (xc, yc, L) in enumerate(cen):
        traps, prof = column_traps(xc, yc, pos, nrm, ph, P0)
        strap, _ = column_traps(0.0, 0.0, sp, sn, sph, P0)   # 단독 기준(1회면 충분)
        for m, t in enumerate(traps):
            sx, Fx, kx, Fex, sex = lateral_scan(xc, yc, t['z_eq'], pos, nrm, ph, P0, 'x')
            sy, Fy, ky, Fey, sey = lateral_scan(xc, yc, t['z_eq'], pos, nrm, ph, P0, 'y')
            r = dict(col=c, layer=int(L), xc=float(xc), yc=float(yc), node=m, **t)
            r.update(k_lat_x=kx, F_esc_x=Fex, s_esc_x=sex,
                     k_lat_y=ky, F_esc_y=Fey, s_esc_y=sey)
            if m < len(strap):
                r.update(F_cap_solo=strap[m]['F_cap'], z_node_solo=strap[m]['z_node'])
            recs.append(r)
        if verbose:
            print(f"  기둥 {c+1:2d}/{len(cen)} (layer {L+1}, x={xc*1e3:5.0f}, "
                  f"y={yc*1e3:+5.0f} mm): 트랩 {len(traps)}개, "
                  f"F_cap {min(t['F_cap'] for t in traps)*1e6:.1f}~"
                  f"{max(t['F_cap'] for t in traps)*1e6:.1f} uN", flush=True)

    # 대표 기둥(중앙, layer1)의 축/평면 프로파일 저장
    traps0, prof0 = column_traps(0.0, 0.0, pos, nrm, ph, P0)
    keys = ['col', 'layer', 'xc', 'yc', 'node', 'z_node', 'z_eq', 'sag', 'F_cap',
            'k_ax', 'p_node', 'Fz_over_W', 'k_lat_x', 'F_esc_x', 's_esc_x',
            'k_lat_y', 'F_esc_y', 's_esc_y', 'F_cap_solo', 'z_node_solo']
    tab = np.array([[r.get(k, np.nan) for k in keys] for r in recs], float)
    np.savez_compressed(fname, P0=P0, pos=pos, nrm=nrm, phase=ph, unit_id=uid,
                        centers=cen, table=tab, keys=np.array(keys),
                        prof_z=prof0['z'], prof_absP=prof0['absP'],
                        prof_U=prof0['U'], prof_Fz=prof0['Fz'],
                        coherent=coherent, seed=seed)
    return fname, recs


def load_table(fname="acoustic_layers_ref.npz"):
    d = np.load(fname, allow_pickle=False)
    keys = [str(k) for k in d['keys']]
    tab = d['table']
    return {k: tab[:, i] for i, k in enumerate(keys)}, d


def scale_drive(rec, drive):
    """구동비 drive 에서의 힘/강성 (F ∝ drive², 위치는 평형 재계산 필요)."""
    s = drive*drive
    return dict(F_cap=rec['F_cap']*s, k_ax=rec['k_ax']*s,
                k_lat_x=rec['k_lat_x']*s, k_lat_y=rec['k_lat_y']*s,
                F_esc_x=rec['F_esc_x']*s, F_esc_y=rec['F_esc_y']*s,
                p_antinode=P.P_ANTINODE_REF*drive)


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("="*78)
    print("다중 부양기 음향장 (720 소자) 기준 계산")
    print("="*78)
    f, recs = build_reference()
    tabd, d = load_table(f)
    print(f"\n소자 세기 P0 = {float(d['P0']):.4f} Pa·m, "
          f"안티노드 {P.P_ANTINODE_REF:.0f} Pa @ drive=1")
    print(f"액적 수 {len(recs)}개, 계산 {time.time()-t0:.1f}s")
    print(f"\n트랩 z 위치 (기둥1): "
          f"{np.round(tabd['z_node'][:6]*1e3,3)} mm")
    print(f"중력 처짐 sag      : {np.round(tabd['sag'][:6]*1e3,3)} mm")
    print(f"트랩 용량 F_cap/W  : {np.round(tabd['Fz_over_W'][:6],3)}")
    ct = 100*(tabd['F_cap']/tabd['F_cap_solo'] - 1)
    print(f"\ncrosstalk (전체배열 vs 단독유닛) F_cap 변화: "
          f"{ct.min():+.2f} ~ {ct.max():+.2f} %  (평균 {ct.mean():+.2f} %)")
    print(f"F_cap 전체 범위 {tabd['F_cap'].min()*1e6:.2f} ~ "
          f"{tabd['F_cap'].max()*1e6:.2f} uN  (중력 {P.W_DROP*1e6:.2f} uN)")
    print(f"모든 액적 부양 가능: {bool(np.all(tabd['F_cap'] > P.W_DROP))}")
