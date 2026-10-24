"""
실험계획 실행 — 60 액적 4-layer 필터
=====================================
실험 1 : 미세입자 유속 5단계 (초음파 구동 100 % 고정), 조건당 독립 실현 5회
실험 2 : 초음파 구동(전압/전류) 5단계 (유속 1.5 m/s 고정), 조건당 독립 실현 5회

실현 k = (메시 k, 입자 난수시드 k) 로 이산화 오차 + 몬테카를로 오차를 함께 반영.

실험 2 에서 유동해를 재사용하는 근거 : 음향 스트리밍 속도는
(3/16)u_ac²/c₀ ~ 20 mm/s 로 운전 유속의 1 % 미만이라 단방향 결합(음향→힘)만
고려하면 된다. 구동 세기는 Gor'kov 복사력을 drive² 로 스케일할 뿐이다.
"""
import os, json, time, numpy as np
from mpi4py import MPI
import params_layers as P, mesh_column as MC, flow_column as FC
import particles_column as PC, acoustics_layers as AC, layers_model as LM

OUT = "results"
os.makedirs(OUT, exist_ok=True)
K_OF_J = [2, 1, 0, 0, 1, 2]          # 액적 level j(0=최하단) -> 1/4 도메인 반구 index
M_SCALE_DEFAULT = 0.255              # A_col ∝ U^m (후류 보정용 국소 지수)


# ---------------------------------------------------------------- 힘 수지
def droplet_force_table(F, U_bulk):
    """액적 level j = 0..5 의 유동 힘. z<0 짝은 대칭이므로 F_x 동일, F_z 부호반전."""
    rows = []
    for j, k in enumerate(K_OF_J):
        zj = float(PC.Z_FULL[j])
        ul = float(P.inlet_profile(zj, U_bulk))
        d = F[k]
        fx = float(d['total'][0])
        fz = float(d['total'][2])*(1.0 if zj > 0 else -1.0)
        rows.append(dict(j=j, z_m=zj, u_local=ul,
                         F_drag_N=fx,
                         F_drag_pressure_N=float(d['pressure'][0]),
                         F_drag_viscous_N=float(d['viscous'][0]),
                         F_lift_N=fz,
                         Cd_cfd=fx/(0.5*P.RHO_A*ul**2*P.A_FRONT),
                         Cd_corr=float(P.cd_schiller_naumann(P.reynolds(ul))),
                         Re_local=float(P.reynolds(ul))))
    return rows


def acoustic_status(tab, drive, drag_by_j, U_layers, U_bulk):
    """
    60개 액적 각각의 부양 안정성.
      축   : F_cap·drive² > W
      횡   : F_esc_x·drive² > F_drag(해당 layer 유속 보정)
    반환: 액적별 레코드와 layer 별 생존 수
    """
    s = drive*drive
    recs = []
    n_lev = len(PC.Z_FULL)
    for i in range(len(tab['F_cap'])):
        L = int(tab['layer'][i]); j = int(tab['node'][i])
        scale = (U_layers[L]/U_bulk)**2 if U_bulk > 0 else 1.0
        Fd = drag_by_j[j]['F_drag_N']*scale
        st = LM.stability(tab['F_cap'][i], abs(tab['F_esc_x'][i]), Fd, drive)
        recs.append(dict(col=int(tab['col'][i]), layer=L, j=j,
                         F_cap_N=float(tab['F_cap'][i])*s,
                         F_esc_x_N=float(abs(tab['F_esc_x'][i]))*s,
                         k_ax=float(tab['k_ax'][i])*s,
                         k_lat_x=float(tab['k_lat_x'][i])*s,
                         sag_m=float(tab['sag'][i]),
                         F_drag_N=Fd, **st))
    surv = np.zeros(len(P.LAYER_NCOL), int)
    for r in recs:
        if r['stable']:
            surv[r['layer']] += 1
    return recs, surv


# ---------------------------------------------------------------- 캐스케이드
def build_cascade(A_by_j, surv_by_layer_j, U_bulk, wake_C, m_scale):
    """
    layer 별 제거율. 생존한 액적만 포집에 기여한다.
      f_L = Σ_(생존 액적) A_cap(level j) · (U_L/U)^m / A_duct
    """
    UL, dz = LM.layer_velocities(U_bulk, wake_C)
    rows, pen = [], 1.0
    for L, n in enumerate(P.LAYER_NCOL):
        sc = (UL[L]/U_bulk)**m_scale if U_bulk > 0 else 1.0
        A = sum(A_by_j[j]*surv_by_layer_j[L][j]*sc for j in range(len(A_by_j)))
        f = min(max(A/P.A_DUCT, 0.0), 1.0)
        rows.append(dict(layer=L+1, n_col=n,
                         n_drop_stable=int(sum(surv_by_layer_j[L])),
                         U_L=float(UL[L]), A_layer_m2=float(A), f=float(f),
                         C_in_rel=pen, C_out_rel=pen*(1-f)))
        pen *= (1-f)
    return dict(rows=rows, penetration=float(pen), removal=float(1-pen),
                wake_deficit=float(dz), U_layers=UL.tolist())


def survivors_matrix(ac_recs):
    """layer x level 생존 액적 수 행렬."""
    M = [[0]*len(PC.Z_FULL) for _ in P.LAYER_NCOL]
    for r in ac_recs:
        if r['stable']:
            M[r['layer']][r['j']] += 1
    return M


# ---------------------------------------------------------------- 1회 실행
def one_case(msh, uu, F, wake, U_bulk, drive, seed, tab, keep_traj=False,
             m_scale=M_SCALE_DEFAULT):
    drag = droplet_force_table(F, U_bulk)
    wC, _ = LM.fit_wake_decay(wake)
    UL, _ = LM.layer_velocities(U_bulk, wC)
    ac_recs, surv = acoustic_status(tab, drive, drag, UL, U_bulk)
    probe = PC.HybridProbe(msh, uu, U_bulk, R_near=2.0*P.A_DROP)
    pr, extra = PC.study(probe, ONE_CASE_ACF, drive, U_bulk, seed,
                         keep_traj=keep_traj)
    A_by_j = [d['A_cap_m2'] for d in pr['per_drop']]
    casc = build_cascade(A_by_j, survivors_matrix(ac_recs), U_bulk, wC, m_scale)
    eng = LM.summarize(casc, U_bulk, float(F['sum_column'][0]))
    n_stable = int(sum(surv))
    rec = dict(U=U_bulk, drive=drive, Re_d=float(P.reynolds(U_bulk)),
               Re_duct=float(P.duct_Re(U_bulk)), St=float(P.stokes(U_bulk)),
               We=float(P.weber(U_bulk)), p_antinode_Pa=P.P_ANTINODE_REF*drive,
               F_gravity_N=P.W_DROP,
               drag=drag, F_drag_column_N=float(F['sum_column'][0]),
               wake=({str(k): list(map(float, v)) for k, v in wake.items()}),
               wake_C=float(wC),
               n_stable=n_stable, n_stable_by_layer=surv.tolist(),
               survivors=survivors_matrix(ac_recs),
               ac_margin_axial_min=float(min(r['margin_axial'] for r in ac_recs)),
               ac_margin_axial_max=float(max(r['margin_axial'] for r in ac_recs)),
               ac_margin_lateral_min=float(min(r['margin_lateral'] for r in ac_recs)),
               ac_margin_lateral_max=float(max(r['margin_lateral'] for r in ac_recs)),
               F_cap_min_N=float(min(r['F_cap_N'] for r in ac_recs)),
               F_cap_max_N=float(max(r['F_cap_N'] for r in ac_recs)),
               F_esc_x_mean_N=float(np.mean([r['F_esc_x_N'] for r in ac_recs])),
               particles=pr, cascade=casc, **eng)
    return rec, extra


# ---------------------------------------------------------------- 드라이버
ONE_CASE_ACF = None


def run_all():
    global ONE_CASE_ACF
    t_all = time.time()
    if not os.path.exists("acoustic_layers_ref.npz"):
        print("[음향 기준장 생성]", flush=True)
        AC.build_reference()
    tab, _ = AC.load_table()
    print("[입자용 음향력 격자 사전계산]", flush=True)
    t0 = time.time()
    ONE_CASE_ACF = PC.AcousticForceCell()
    print(f"  격자점 {ONE_CASE_ACF.grid_pts:,}, 최대 |F_ac,입자| = "
          f"{ONE_CASE_ACF.Fmax*1e12:.3f} pN (입자 무게 {P.M_PART*P.G*1e12:.3f} pN), "
          f"{time.time()-t0:.1f}s\n", flush=True)

    for k, h in enumerate(P.H_DROP_REP):
        if not os.path.exists(f"mesh_col_r{k}.msh"):
            MC.build(f"mesh_col_r{k}.msh", h_drop=h)

    # ================================ 실험 1 : 유속 5단계
    for U in P.SWEEP_U:
        for rep in range(P.NREP):
            tag = f"E1_U{U:g}_r{rep}"
            if os.path.exists(f"{OUT}/{tag}.json"):
                continue
            t0 = time.time()
            print(f"=== [실험1] U = {U} m/s | 실현 {rep+1}/{P.NREP} ===", flush=True)
            msh, ft, w, F, info = FC.solve(f"mesh_col_r{rep}.msh", U)
            uu = FC.collapse_velocity(w)
            wake = FC.wake_profile(msh, w, U)
            rec, extra = one_case(msh, uu, F, wake, U, 1.0, P.SEED_REP[rep], tab,
                                  keep_traj=(rep == 0))
            fd = " ".join("%.2f" % (d["F_drag_N"]*1e6) for d in rec["drag"][3:])
            cderr = 100*(rec["drag"][3]["Cd_cfd"]/rec["drag"][3]["Cd_corr"] - 1)
            print(f"  항력(액적 z>0) {fd} uN | 기둥합 "
                  f"{rec['F_drag_column_N']*1e6:.2f} uN | Cd 오차 {cderr:+.1f} %",
                  flush=True)
            print(f"  A_col = {rec['particles']['A_col_m2']*1e6:.3f} mm2 "
                  f"(직접법 {rec['particles']['A_col_direct_m2']*1e6:.3f}) | "
                  f"부양 {rec['n_stable']}/60 | 제거율 {100*rec['removal']:.3f} % | "
                  f"C_out {rec['C_out_mg_m3']:.3f} mg/m3 | dP {rec['dP_droplets_Pa']:.4f} Pa",
                  flush=True)
            rec.update(exp="E1", rep=rep, h_drop=P.H_DROP_REP[rep],
                       n_tets=int(msh.topology.index_map(3).size_global),
                       ndof=info['ndof'], wall_s=time.time()-t0)
            json.dump(rec, open(f"{OUT}/{tag}.json", "w"), indent=1, ensure_ascii=False, default=float)
            if rep == 0:
                tja, sa, xa, tg = extra
                if tja is not None:
                    np.savez_compressed(f"{OUT}/traj_{tag}.npz",
                                        traj=tja.astype(np.float32), state=sa,
                                        seeds=xa, tag=tg)
                import viz_layers as VZ
                VZ.save_solution(msh, w, f"{OUT}/sol_{tag}.vtu")
            print(f"  {time.time()-t0:.0f}s\n", flush=True)

    # ================================ 실험 2 : 구동 5단계 (유동해 재사용)
    U = P.U_FIXED_FOR_DRIVE
    for rep in range(P.NREP):
        need = [d for d in P.SWEEP_DRIVE
                if not os.path.exists(f"{OUT}/E2_D{d:.2f}_r{rep}.json")]
        if not need:
            continue
        t0 = time.time()
        print(f"=== [실험2] 유동해 (U={U} m/s, 실현 {rep+1}/{P.NREP}) ===", flush=True)
        msh, ft, w, F, info = FC.solve(f"mesh_col_r{rep}.msh", U)
        uu = FC.collapse_velocity(w)
        wake = FC.wake_profile(msh, w, U)
        for drive in need:
            t1 = time.time()
            rec, extra = one_case(msh, uu, F, wake, U, drive,
                                  P.SEED_REP[rep]+int(drive*100), tab,
                                  keep_traj=(rep == 0))
            print(f"  V/V0={drive:.2f} p0={rec['p_antinode_Pa']:.0f} Pa | "
                  f"부양 {rec['n_stable']}/60 "
                  f"(축여유 {rec['ac_margin_axial_min']:.2f}~{rec['ac_margin_axial_max']:.2f}, "
                  f"횡여유 {rec['ac_margin_lateral_min']:.2f}~"
                  f"{rec['ac_margin_lateral_max']:.2f}) | "
                  f"제거율 {100*rec['removal']:.3f} % | C_out {rec['C_out_mg_m3']:.3f}",
                  flush=True)
            tag = f"E2_D{drive:.2f}_r{rep}"
            rec.update(exp="E2", rep=rep, h_drop=P.H_DROP_REP[rep],
                       n_tets=int(msh.topology.index_map(3).size_global),
                       ndof=info['ndof'], wall_s=time.time()-t1)
            json.dump(rec, open(f"{OUT}/{tag}.json", "w"), indent=1, ensure_ascii=False, default=float)
            if rep == 0:
                tja, sa, xa, tg = extra
                if tja is not None:
                    np.savez_compressed(f"{OUT}/traj_{tag}.npz",
                                        traj=tja.astype(np.float32), state=sa,
                                        seeds=xa, tag=tg)
        if rep == 0:
            import viz_layers as VZ
            VZ.save_solution(msh, w, f"{OUT}/sol_E2_U{U:g}.vtu")
        print(f"  {time.time()-t0:.0f}s\n", flush=True)

    print(f"ALL DONE  총 {(time.time()-t_all)/60:.1f} 분", flush=True)


if __name__ == "__main__":
    run_all()
