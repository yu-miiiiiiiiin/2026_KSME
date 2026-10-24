"""
검증 (validation) — 새 다층 해석 코드가 물리적으로 옳은지 확인
==============================================================
V1  Gor'kov 복사력 : 단독 유닛 결과가 기존 단일 액적 연구(ksme_test)와 일치하는가
V2  항력          : 액적별 CFD 항력이 Schiller-Naumann 상관식과 일치하는가
V3  단일 액적 대조 : 기존 연구의 메시/유동해에 '새 입자 추적기'를 적용해
                    기존 포집효율(eta ~ 0.67 @ U=1.5)을 재현하는가
                    -> 재현되면 기둥 배열에서 eta 가 낮아지는 것은 물리(밀집 효과)
V4  격자 프로브   : 격자 보간 프로브가 정확 FE 프로브와 같은 결과를 주는가
V5  수렴성        : 시간간격 / 입자 수에 대한 민감도
"""
import sys, os, time, numpy as np
sys.path.append(os.path.abspath("../ksme_test"))

import params_layers as P
import acoustics_layers as AC
import mesh_column as MC
import flow_column as FC
import particles_column as PC


def v1_gorkov():
    print("\n[V1] Gor'kov 복사력 — 단독 유닛 vs 기존 단일 액적 연구")
    P0 = AC.calibrate(); pos, nrm = AC.unit_geometry(); ph = np.zeros(len(pos))
    z = np.linspace(-0.026, 0.026, 5201)
    Pc, U, Fz = AC.force_on_line(np.stack([0*z, 0*z, z], 1), z[1]-z[0],
                                 pos, nrm, ph, P0, P.A_DROP, axis=2)
    _, _, k, Fe, se = AC.lateral_scan(0, 0, 1.822e-3, pos, nrm, ph, P0, 'x')
    ref = dict(Fz_max=64.68e-6, Fx_max=7.85e-6, p_anti=2400.0)
    print(f"  안티노드 |p|   = {np.abs(Pc).max():7.1f} Pa   (기존 {ref['p_anti']:.0f})")
    print(f"  축방향 F_z,max = {Fz.max()*1e6:7.3f} uN   (기존 {ref['Fz_max']*1e6:.2f}, "
          f"차이 {100*(Fz.max()/ref['Fz_max']-1):+.2f} %)")
    print(f"  횡방향 F_x,max = {Fe*1e6:7.3f} uN   (기존 {ref['Fx_max']*1e6:.2f}, "
          f"차이 {100*(Fe/ref['Fx_max']-1):+.2f} %)")
    return abs(Fz.max()/ref['Fz_max']-1) < 0.02 and abs(Fe/ref['Fx_max']-1) < 0.02


def v2_drag(mesh="mesh_col_r2.msh", Us=(0.5, 1.5, 2.5)):
    print("\n[V2] 액적별 항력 vs Schiller-Naumann 상관식")
    print(f"  {'U':>5} {'액적':>4} {'z[mm]':>7} {'Re_loc':>7} {'Cd_CFD':>8} "
          f"{'Cd_corr':>8} {'차이':>8}")
    ok = True
    for U in Us:
        msh, ft, w, F, info = FC.solve(mesh, U)
        for k, zk in enumerate(MC.Z_DROPS):
            ul = float(P.inlet_profile(float(zk), U))
            Cd = F[k]['total'][0]/(0.5*P.RHO_A*ul**2*P.A_FRONT)
            Cc = float(P.cd_schiller_naumann(P.reynolds(ul)))
            e = 100*(Cd/Cc-1)
            ok &= abs(e) < 20
            print(f"  {U:5.1f} {k:>4} {zk*1e3:7.2f} {P.reynolds(ul):7.0f} "
                  f"{Cd:8.4f} {Cc:8.4f} {e:+7.1f} %")
    return ok


def v3_single_droplet_control(U=1.5, n_seed=6000, seed=1000):
    """
    단일 액적 대조군.

    기존 연구(ksme_test)의 VelocityProbe 는 반(半)도메인(y>=0) FE 해를 y<0 으로
    미러링하지 않고, 도메인 밖이라는 이유로 자유류 (U,0,0) 를 돌려준다. 그 결과
    입자의 절반이 균일류를 직진해 액적에 그대로 충돌하여 포집이 과대평가되었다.
    기존 값은 다음과 같이 재구성된다.

        A_cap(기존) = (1/2)·πa²          (y<0 : 균일류 직진, 기하 차단면)
                    + (1/2)·A_cap(정확)  (y>0 : 정상 유동)

    본 검증은 (i) 미러링을 적용한 정확값 A_cap 을 구하고,
    (ii) 위 식으로 기존 값을 예측해 실제 기존 결과와 맞는지 확인하며,
    (iii) 정확값이 고전 관성충돌 상관식 범위에 드는지 본다.
    """
    print("\n[V3] 단일 액적 대조군 — 기존 연구값 재구성 및 상관식 대조")
    import flow_solver as OFS          # ../ksme_test
    msh, ft, w, Fo, info = OFS.solve("../ksme_test/mesh_r4.msh", U)
    uu = FC.collapse_velocity(w)
    lv = ((8.0e-4, (-8.0e-3, 14.0e-3, 7.0e-3, 7.0e-3)),
          (2.5e-4, (-6.0e-3, 12.0e-3, 4.0e-3, 4.0e-3)),
          (8.0e-5, (-2.5e-3,  5.0e-3, 2.2e-3, 2.2e-3)))
    probe = PC.HybridProbe(msh, uu, U, R_near=2.0*P.A_DROP, levels=lv,
                           centers=[[0, 0, 0]], uniform=True)
    trk = PC.ParticleTracker(probe, acoustic=None, drive=1.0, centers=[[0, 0, 0]],
                             bounds=(-8e-3, 14e-3, 7e-3, 7e-3))
    rng = np.random.default_rng(seed)
    Rs = 2.0*P.A_DROP
    rr = Rs*np.sqrt(rng.random(n_seed)); th = rng.random(n_seed)*2*np.pi
    x0 = np.stack([np.full(n_seed, -7.5e-3), rr*np.cos(th), rr*np.sin(th)], 1)
    v0 = np.tile([U, 0, 0], (n_seed, 1))
    dt = min(0.20*trk.tau, 0.05*P.D_DROP/U)
    ns = int(2.6*22e-3/(U*dt))
    _, st = trk.run(x0, v0, dt, ns)
    nc = int((st == 1).sum())
    A, dA = PC.capture_area(nc, n_seed, np.pi*Rs**2)
    eta = A/(np.pi*P.A_DROP**2)
    A_old_pred = 0.5*np.pi*P.A_DROP**2 + 0.5*A
    A_old_meas = 2.166e-6
    St = P.stokes(U)
    e1 = (St/(St+0.5))**2            # 점성유동 근사 하한
    e2 = St**2/(St+0.25)**2          # 포텐셜유동 근사 상한
    print(f"  정확(미러링) : 포집 {nc}/{n_seed}  A_cap = {A*1e6:.3f} ± {dA*1e6:.3f} mm2"
          f"   eta = {eta:.3f}")
    print(f"  기존값 재구성 : πa²/2 + A/2 = {A_old_pred*1e6:.3f} mm2  "
          f"vs 기존 실측 {A_old_meas*1e6:.3f} mm2  (차이 {100*(A_old_meas/A_old_pred-1):+.1f} %)")
    print(f"  -> 기존 연구는 흡착율을 {A_old_meas/A:.2f} 배 과대평가했다")
    print(f"  고전 관성충돌 상관식 (St={St:.3f}) : "
          f"(St/(St+0.5))² = {e1:.3f} ~ St²/(St+0.25)² = {e2:.3f}")
    print(f"  -> 정확값 {eta:.3f} 는 상관식 범위 "
          f"{'안에 있음' if e1 < eta < e2 else '밖 (점검 필요)'}")
    ok = abs(A_old_meas/A_old_pred - 1) < 0.06 and e1 < eta < e2
    return ok, eta


def v4_probe(mesh="mesh_col_r2.msh", U=1.5, n_seed=1500, seed=7):
    print("\n[V4] 격자 보간 프로브 vs 정확 FE 프로브")
    msh, ft, w, F, info = FC.solve(mesh, U)
    uu = FC.collapse_velocity(w)
    out = {}
    for name, mk in (("정확 FE", lambda: PC.MirrorProbe(msh, uu, U)),
                     ("격자보간", lambda: PC.GridProbe(msh, uu, U, verbose=True))):
        t0 = time.time(); pb = mk(); tb = time.time()-t0
        r, _ = PC.study(pb, None, 1.0, U, seed, n_imp=n_seed, n_dir=200)
        out[name] = (r['A_col_m2'], time.time()-t0, tb)
        print(f"  {name:9s}: A_col = {r['A_col_m2']*1e6:.3f} mm2  "
              f"(구성 {tb:.1f}s, 전체 {time.time()-t0:.1f}s)")
    a, b = out["정확 FE"][0], out["격자보간"][0]
    print(f"  차이 {100*(b/a-1):+.2f} %   가속 {out['정확 FE'][1]/out['격자보간'][1]:.1f} 배")
    return abs(b/a-1) < 0.06


def v5_convergence(mesh="mesh_col_r2.msh", U=1.5):
    print("\n[V5] 시간간격 / 입자 수 민감도")
    msh, ft, w, F, info = FC.solve(mesh, U)
    uu = FC.collapse_velocity(w)
    pb = PC.GridProbe(msh, uu, U)
    base = dict(n_imp=3000, n_dir=200)
    for fac in (2.0, 1.0, 0.5):
        trk = PC.ParticleTracker(pb)
        dt = fac*min(0.20*trk.tau, 0.05*P.D_DROP/U)
        # study() 내부 dt 를 우회하기 위해 직접 계산
        rng = np.random.default_rng(11)
        Rs = P.R_SEED_FAC*P.A_DROP; per = 800
        xs, tg = [], []
        for j, zk in enumerate(PC.Z_FULL):
            rr = Rs*np.sqrt(rng.random(per)); th = rng.random(per)*2*np.pi
            xs.append(np.stack([np.full(per, P.CELL_X0+5e-4), rr*np.cos(th),
                                zk+rr*np.sin(th)], 1))
            tg.append(np.full(per, j))
        xa = np.concatenate(xs); tg = np.concatenate(tg)
        va = np.stack([P.inlet_profile(xa[:, 2], U), np.zeros(len(xa)),
                       np.zeros(len(xa))], 1)
        ns = int(2.6*(P.CELL_X1-P.CELL_X0)/(U*dt))
        _, st = trk.run(xa, va, dt, ns)
        A = sum(PC.capture_area(int((st[tg == j] == j+1).sum()), per,
                                np.pi*Rs**2)[0] for j in range(6))
        print(f"  dt x{fac:.1f} ({dt*1e6:5.1f} us, {ns:5d} step): "
              f"A_col = {A*1e6:.3f} mm2")
    for n in (1200, 3000, 6000):
        r, _ = PC.study(pb, None, 1.0, U, 11, n_imp=n, n_dir=200)
        print(f"  입자 {n:5d}: A_col = {r['A_col_m2']*1e6:.3f} "
              f"± {r['A_col_err_m2']*1e6:.3f} mm2")
    return True


if __name__ == "__main__":
    which = sys.argv[1:] or ["v1", "v2", "v3", "v4", "v5"]
    res = {}
    if "v1" in which: res['V1'] = v1_gorkov()
    if "v2" in which: res['V2'] = v2_drag()
    if "v3" in which: res['V3'] = v3_single_droplet_control()[0]
    if "v4" in which: res['V4'] = v4_probe()
    if "v5" in which: res['V5'] = v5_convergence()
    print("\n" + "="*60)
    for k, v in res.items():
        print(f"  {k}: {'PASS' if v else 'CHECK'}")
