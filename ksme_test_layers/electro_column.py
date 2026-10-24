"""
정전 포집 실험 — 미세입자(−) / 액적(+) 대전
=============================================
기존 해석(유동 + 음향 + 중력)에 정전기력을 추가한다.

    m_p dv/dt = F_drag + F_grav + F_ac + F_coul + F_image

  F_coul  = (1/4πε₀) q_p Q_d (x−c_k)/|x−c_k|³        (입자−액적 쿨롱력)
            q_p < 0, Q_d > 0 이므로 인력
  F_image = −(1/4πε₀) q_p² a³(2r²−a²) / (r³(r²−a²)²) r̂   (도체구 영상력, 항상 인력)
            액적은 전도성(물, τ_e = ε/σ ≪ 통과시간)이므로 도체구로 본다

입자 대전 (Pauthenier 전계대전 포화전하)
    q_s = 3ε_r/(ε_r+2) · π ε₀ E_c d_p²          E_c = 코로나 대전기 전계

액적 대전의 상한 두 가지
  (1) Rayleigh 한계   Q_R = 8π √(ε₀ σ a³)          (액적 분열)
  (2) 기둥 내 상호반발 : 같은 부호 액적 6개가 λ/2 간격으로 늘어서 있어
      최하단 액적은 반발력(위→아래 방향 성분)과 중력을 함께 받는다.
          F_rep·Σ + W < F_cap·drive²
      → 실제로는 (2)가 (1)보다 약 5배 엄격하다 (Q_lim = 0.20 Q_R).
"""
import os, sys, json, time
import numpy as np
import params_layers as P, mesh_column as MC, flow_column as FC
import particles_column as PC, acoustics_layers as AC, layers_model as LM

EPS0 = 8.8541878e-12
KE   = 1.0/(4*np.pi*EPS0)
OUT  = "results"

# ---------------------------------------------------------------- 대전 조건
E_CORONA = 3.0e5          # 코로나 대전기 전계 [V/m] (3 kV/mm, 공기 절연파괴의 40 %)
EPS_R_P  = 4.0            # 미세입자 비유전율 (광물 분진)


def q_particle(E_c=E_CORONA, eps_r=EPS_R_P, d=P.D_PART):
    """Pauthenier 전계대전 포화전하 [C] (음극성이므로 부호는 −)."""
    return -3*eps_r/(eps_r+2)*np.pi*EPS0*E_c*d*d


def q_rayleigh(a=P.A_DROP, sigma=P.SIGMA):
    return 8*np.pi*np.sqrt(EPS0*sigma*a**3)


def q_column_limit(F_cap, drive=1.0, n=P.N_PER_COL, d=None):
    """
    기둥 내 상호반발로 인한 액적 전하 상한 [C].
    최하단(또는 최상단) 액적이 나머지 5개로부터 받는 반발력의 축방향 합
        F_rep = k Q² Σ_{m=1..n-1} 1/(m d)²
    이 '음향 포획력 여유' F_cap·drive² − W 를 넘지 않아야 한다.
    """
    d = d if d is not None else P.LAM/2
    S = sum(1.0/m**2 for m in range(1, n))
    head = F_cap*drive*drive - P.W_DROP
    if head <= 0:
        return 0.0
    return float(np.sqrt(head/(KE*S/d**2)))


class ElectroTracker(PC.ParticleTracker):
    """PC.ParticleTracker + 쿨롱력 + 영상력."""
    def __init__(self, *a, q_p=0.0, Q_d=0.0, image=True, **kw):
        super().__init__(*a, **kw)
        self.q_p, self.Q_d, self.image = q_p, Q_d, image

    def accel(self, x, v):
        a = super().accel(x, v)
        if self.q_p == 0.0:
            return a
        rv = x[:, None, :] - self.centers[None, :, :]        # (N,K,3)
        r = np.linalg.norm(rv, axis=2)
        r = np.maximum(r, P.A_DROP + self.dp/2)
        F = np.zeros_like(x)
        if self.Q_d != 0.0:
            F += KE*self.q_p*self.Q_d*np.sum(rv/r[:, :, None]**3, axis=1)
        if self.image:
            aa = P.A_DROP
            mag = -KE*self.q_p**2*aa**3*(2*r**2-aa**2)/(r**3*(r**2-aa**2)**2)
            F += np.sum(mag[:, :, None]*rv/r[:, :, None], axis=1)
        return a + F/self.mp


def study_charged(probe, acf, U_bulk, seed, q_p, Q_d, n_per=400, drive=1.0):
    """적응형 중요도 표본추출로 액적 6개의 포집 단면적 A_cap 을 구한다."""
    trk = ElectroTracker(probe, acoustic=acf, drive=drive, q_p=q_p, Q_d=Q_d)
    rng = np.random.default_rng(seed)
    dt = min(0.20*trk.tau, 0.05*P.D_DROP/U_bulk)
    nstep = int(2.6*(P.CELL_X1-P.CELL_X0)/(U_bulk*dt))
    xin = P.CELL_X0 + 5e-4
    ZF = PC.Z_FULL

    def seed_run(radii, npd, offs):
        xs, tg = [], []
        for j, zk in enumerate(ZF):
            R = radii[j]; oy, oz = offs[j]
            rr = R*np.sqrt(rng.random(npd)); th = rng.random(npd)*2*np.pi
            xs.append(np.stack([np.full(npd, xin), oy+rr*np.cos(th),
                                zk+oz+rr*np.sin(th)], 1))
            tg.append(np.full(npd, j))
        xa = np.concatenate(xs); tg = np.concatenate(tg)
        va = np.stack([P.inlet_profile(xa[:, 2], U_bulk),
                       np.zeros(len(xa)), np.zeros(len(xa))], 1)
        return xa, tg, va

    # --- 1차: 넓게 뿌려 포집반경 개산 (정전 인력으로 반경이 커질 수 있다)
    R0 = 5.0e-3
    xp, tp, vp = seed_run([R0]*len(ZF), 200, [(0.0, 0.0)]*len(ZF))
    _, sp = trk.run(xp, vp, dt, nstep)
    radii, offs = [], []
    for j in range(len(ZF)):
        m = tp == j; cap = sp[m] == j+1
        r_cap = np.sqrt(max(cap.mean(), 1e-4))*R0
        if cap.sum() >= 5:
            oy = float(xp[m][cap][:, 1].mean())
            oz = float(xp[m][cap][:, 2].mean() - ZF[j])
        else:
            oy = oz = 0.0
        radii.append(float(np.clip(1.6*r_cap + 2.0e-4, 0.5*P.A_DROP, 6.0e-3)))
        offs.append((oy, oz))

    # --- 2차: 본 표본
    xa, tag, va = seed_run(radii, n_per, offs)
    _, sa = trk.run(xa, va, dt, nstep)
    per, A_tot, var = [], 0.0, 0.0
    for j in range(len(ZF)):
        m = tag == j
        nc = int((sa[m] == j+1).sum())
        A, dA = PC.capture_area(nc, int(m.sum()), np.pi*radii[j]**2)
        per.append(dict(j=j, z_m=float(ZF[j]), n_cap=nc, n_seed=int(m.sum()),
                        R_seed_m=radii[j], A_cap_m2=A, A_cap_err_m2=dA,
                        r_cap_eff_m=float(np.sqrt(A/np.pi)),
                        eta=A/(np.pi*P.A_DROP**2)))
        A_tot += A; var += dA*dA
    return dict(per_drop=per, A_col_m2=A_tot, A_col_err_m2=float(np.sqrt(var)),
                eta_col=A_tot/(len(ZF)*np.pi*P.A_DROP**2),
                dt=dt, nstep=nstep, n_seed=int(len(xa)))


def cascade_removal(A_col, U_bulk, wake_C, m_scale=0.255):
    UL, dz = LM.layer_velocities(U_bulk, wake_C)
    rows, pen = [], 1.0
    for L, n in enumerate(P.LAYER_NCOL):
        f = min(max(n*A_col*(UL[L]/U_bulk)**m_scale/P.A_DUCT, 0.0), 1.0)
        rows.append(dict(layer=L+1, n_col=n, U_L=float(UL[L]), f=float(f),
                         C_in_rel=pen, C_out_rel=pen*(1-f)))
        pen *= (1-f)
    return dict(rows=rows, penetration=float(pen), removal=float(1-pen),
                wake_deficit=float(dz), U_layers=UL.tolist())


# ---------------------------------------------------------------- 드라이버
def parse_cases(s):
    """'drive:E_c:Q_d/Q_lim' 목록 문자열 -> [(drive, E_c, fq), ...]"""
    out = []
    for tok in s.split(","):
        d, e, f = tok.strip().split(":")
        out.append((float(d), float(e), float(f)))
    return out


# 기본 스윕 = 기준 구동(V/V0=1.0)에서 액적 전하 4단계
CASES_DEFAULT = "1.0:3e5:0, 1.0:3e5:0.25, 1.0:3e5:0.5, 1.0:3e5:1.0, 1.0:3e5:2.0"


def main():
    U     = float(os.environ.get("U", P.U_FIXED_FOR_DRIVE))
    mesh  = os.environ.get("MESH", "mesh_col_r4.msh")
    n_per = int(os.environ.get("NPER", 400))
    tag   = os.environ.get("TAG", "electro_sweep")
    cases = parse_cases(os.environ.get("CASES", CASES_DEFAULT))
    t_all = time.time()

    tab, _ = AC.load_table()
    F_cap0 = float(np.min(tab['F_cap']))          # 구동 1.0 에서의 최소 포획력
    Q_R    = q_rayleigh()

    print("="*78)
    print("정전 포집 실험 — 미세입자(−) / 액적(+)")
    print("="*78)
    print(f"Rayleigh 한계 Q_R = {Q_R*1e9:.3f} nC "
          f"(F_cap,min(V/V0=1) = {F_cap0*1e6:.2f} uN, W = {P.W_DROP*1e6:.2f} uN)")
    print("액적 전하 상한 Q_lim(drive) = sqrt((F_cap·drive² − W)/(k·Σ/d²)) "
          "— Rayleigh 가 아니라 기둥 부양 안정성이 결정\n")

    print("[음향력 격자]", flush=True)
    acf = PC.AcousticForceCell()
    print(f"  최대 |F_ac| = {acf.Fmax*1e12:.2f} pN, {time.time()-t_all:.1f}s")

    print(f"[유동해] {mesh}, U={U} m/s", flush=True)
    t0 = time.time()
    msh, ft, w, Ff, info = FC.solve(mesh, U)
    uu = FC.collapse_velocity(w)
    wake = FC.wake_profile(msh, w, U)
    wC, _ = LM.fit_wake_decay(wake)
    print(f"  ndof={info['ndof']:,}, 기둥 항력 {Ff['sum_column'][0]*1e6:.2f} uN, "
          f"{time.time()-t0:.0f}s", flush=True)
    probe = PC.HybridProbe(msh, uu, U, R_near=2.0*P.A_DROP)

    S = sum(1.0/m**2 for m in range(1, P.N_PER_COL))    # 기둥 반발 합
    print(f"\n[스윕] 액적당 {n_per}개 x 6 액적, U={U} m/s, 조건 {len(cases)}개")
    print(f"{'V/V0':>5} {'Ec[kV/cm]':>10} {'q_p[fC]':>8} {'Q/Qlim':>7} {'Q_d[pC]':>8} "
          f"{'Q/QR':>6} {'r_cap[mm]':>10} {'A_col[mm2]':>11} {'A/A0':>6} "
          f"{'제거율[%]':>10} {'C_out':>8} {'기둥안정':>9}")
    recs, A0 = [], None
    for drive, E_c, fq in cases:
        t1 = time.time()
        q_p   = q_particle(E_c)
        F_cap = F_cap0*drive*drive
        Q_lim = q_column_limit(F_cap0, drive)
        Q_d   = fq*Q_lim
        pr = study_charged(probe, acf, U, 12345+int(fq*1000+E_c/1e3+drive*7),
                           q_p, Q_d, n_per=n_per, drive=drive)
        casc = cascade_removal(pr['A_col_m2'], U, wC)
        eng = LM.summarize(casc, U, float(Ff['sum_column'][0]))
        F_rep = KE*Q_d**2*S/(P.LAM/2)**2
        margin = (F_cap - P.W_DROP - F_rep)/F_cap        # 축방향 잔여 여유
        stable = margin > 0
        rec = dict(drive=drive, E_corona=E_c, Q_over_lim=fq, Q_d_C=Q_d, q_p_C=q_p,
                   U=U, F_rep_N=F_rep, F_cap_N=F_cap, axial_margin=margin,
                   Q_over_rayleigh=Q_d/Q_R, column_stable=bool(stable),
                   particles=pr, cascade=casc, wake_C=float(wC), **eng)
        recs.append(rec)
        if A0 is None:
            A0 = pr['A_col_m2']
        rc = np.mean([d['r_cap_eff_m'] for d in pr['per_drop']])
        lab = ('한계' if abs(margin) < 0.02 else ('OK' if stable else 'FAIL'))
        print(f"{drive:5.2f} {E_c/1e5:10.1f} {q_p*1e15:8.3f} {fq:7.2f} "
              f"{Q_d*1e12:8.1f} {Q_d/Q_R:6.2f} {rc*1e3:10.3f} "
              f"{pr['A_col_m2']*1e6:11.3f} {pr['A_col_m2']/A0:6.2f} "
              f"{100*casc['removal']:10.3f} {eng['C_out_mg_m3']:8.3f} {lab:>9}"
              f"   ({time.time()-t1:.0f}s)", flush=True)

    # 액적 전하 중화시간 (포집된 −전하가 +전하를 상쇄) — 마지막 조건 기준
    base = recs[-1]
    I = (P.C_NUM_IN*U*base['particles']['A_col_m2']/P.N_PER_COL
         * abs(base['q_p_C']))
    t_neut = base['Q_d_C']/I if I > 0 else np.inf
    print(f"\n액적 전하 중화시간 (마지막 조건) : I = {I*1e15:.2f} fA "
          f"-> t = {t_neut:.0f} s = {t_neut/60:.1f} 분")
    rho_s = P.C_NUM_IN*abs(base['q_p_C'])
    E_sc = rho_s*(P.DUCT_H/2)/EPS0
    print(f"입자 공간전하 rho = {rho_s*1e9:.2f} nC/m3 -> 덕트 벽면 전계 "
          f"{E_sc:.1f} V/m (자기침전 성분, 무시 가능)")

    json.dump(dict(cases=recs, Q_rayleigh_C=Q_R, F_cap0_N=F_cap0,
                   mesh=mesh, n_per=n_per, U=U),
              open(f"{OUT}/{tag}.json", "w"), indent=1,
              ensure_ascii=False, default=float)
    print(f"\n총 {time.time()-t_all:.0f}s -> {OUT}/{tag}.json")


if __name__ == "__main__":
    main()
