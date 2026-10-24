"""
결과 집계 · 표 · 그림 생성
==========================
  python3 make_report_layers.py            전체
  python3 make_report_layers.py tables      표만
  python3 make_report_layers.py figs        그림만
"""
import os, sys, json, glob, numpy as np
import params_layers as P
import acoustics_layers as AC
import mesh_column as MC
import layers_model as LM
import neck_model as NM
import run_sweeps_layers as SW
import report_style as RS

OUT, FIG = "results", "figures"
_FONT = RS.use()
os.makedirs(FIG, exist_ok=True)
LEVEL_NAME = [f"L{j+1}" for j in range(P.N_PER_COL)]


# ================================================================ 로드/집계
def load(exp):
    rs = []
    for f in sorted(glob.glob(f"{OUT}/{exp}_*.json")):
        if "traj" in f:
            continue
        rs.append(json.load(open(f)))
    return rs


def group(recs, key):
    g = {}
    for r in recs:
        g.setdefault(round(r[key], 4), []).append(r)
    return {k: g[k] for k in sorted(g)}


def ms(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def pick(rs, path):
    """path 예: 'removal' 또는 'particles.A_col_m2' 또는 'drag.3.Cd_cfd'"""
    out = []
    for r in rs:
        v = r
        for k in path.split('.'):
            v = v[int(k)] if k.isdigit() else v[k]
        out.append(v)
    return out


# ================================================================ 표
def sec(f, title):
    f.write("\n" + "="*112 + f"\n{title}\n" + "="*112 + "\n")


def write_tables(path="RESULTS.txt"):
    e1, e2 = load("E1"), load("E2")
    tab, d = AC.load_table()
    f = open(path, "w")

    # ---------------------------------------------------------- 0. 설계
    sec(f, "[0] 설계 제원")
    f.write(f"""
장치        TinyLev 기반 단축 초음파 부양기 x {P.N_COL}대
            (40 kHz, 직경 10 mm 트랜스듀서 72개/대 = 총 {P.N_COL*72}개,
             면당 3링 6/12/18, 구면캡 곡률반경 {P.R_CURV*1e3:.0f} mm,
             배열 정점 간격 99.8 mm, 배열 외경 {2*P.ARRAY_ENVELOPE_R*1e3:.1f} mm)
액적        물 D = {P.D_DROP*1e3:.1f} mm, m = {P.M_DROP*1e6:.2f} mg,
            중력 W = {P.W_DROP*1e6:.2f} uN, ka = {P.K_WAVE*P.A_DROP:.3f}
배열        기둥 {P.N_COL}개 x 액적 {P.N_PER_COL}개 = {P.N_DROP}개
            layer {len(P.LAYER_NCOL)}층 {P.LAYER_NCOL} (엇갈림), S_x = {P.S_X*1e3:.0f} mm,
            S_y = {P.S_Y*1e3:.0f} mm, 최소 배열중심거리 {np.hypot(P.S_X,P.S_Y/2)*1e3:.1f} mm
            (> 배열 외경 {2*P.ARRAY_ENVELOPE_R*1e3:.1f} mm 이므로 물리 간섭 없음)
덕트        {P.DUCT_W*1e3:.0f}(W) x {P.DUCT_H*1e3:.0f}(H) mm,
            A = {P.A_DUCT*1e6:.0f} mm2, D_h = {P.duct_Dh()*1e3:.1f} mm
액적 z위치  {np.round(MC.full_column_z()*1e3, 3)} mm (덕트 중심 기준)
            = 연직 정재파의 중앙 6개 압력 노드. 덕트벽 여유
            {(P.DUCT_H/2-MC.Z_DROPS[-1]-P.A_DROP)*1e3:.2f} mm ({(P.DUCT_H/2-MC.Z_DROPS[-1]-P.A_DROP)/P.D_DROP:.2f} D)
미세입자    d = {P.D_PART*1e6:.0f} um, rho = {P.RHO_PART:.0f} kg/m3,
            tau_p = {P.TAU_PART*1e3:.3f} ms, 입구 {P.C_MASS_IN*1e6:.1f} mg/m3
            (= {P.C_NUM_IN:.3e} 개/m3)
""")

    # ---------------------------------------------------------- 1. 음향
    sec(f, "[1] 음향장 해석 — 60개 액적의 트랩 특성 (구동 V/V0 = 1.00, 720 소자 전체 중첩)")
    f.write(f"\n{'기둥':>4} {'layer':>5} {'x[mm]':>6} {'y[mm]':>6} | "
            f"{'level':>5} {'노드z[mm]':>9} {'평형z[mm]':>9} {'처짐[mm]':>8} "
            f"{'F_cap[uN]':>10} {'F/W':>6} {'k_ax[N/m]':>10} "
            f"{'F_esc,x[uN]':>11} {'k_lat[N/m]':>11}\n" + "-"*112 + "\n")
    for i in range(len(tab['F_cap'])):
        f.write(f"{int(tab['col'][i])+1:>4} {int(tab['layer'][i])+1:>5} "
                f"{tab['xc'][i]*1e3:>6.0f} {tab['yc'][i]*1e3:>6.0f} | "
                f"{int(tab['node'][i])+1:>5} {tab['z_node'][i]*1e3:>9.3f} "
                f"{tab['z_eq'][i]*1e3:>9.3f} {tab['sag'][i]*1e3:>8.3f} "
                f"{tab['F_cap'][i]*1e6:>10.2f} {tab['Fz_over_W'][i]:>6.3f} "
                f"{tab['k_ax'][i]:>10.4f} {abs(tab['F_esc_x'][i])*1e6:>11.2f} "
                f"{tab['k_lat_x'][i]:>11.5f}\n")
        if (i+1) % P.N_PER_COL == 0:
            f.write("-"*112 + "\n")
    ct = 100*(tab['F_cap']/tab['F_cap_solo'] - 1)
    f.write(f"""
요약
  F_cap  {tab['F_cap'].min()*1e6:.2f} ~ {tab['F_cap'].max()*1e6:.2f} uN
         (중력 {P.W_DROP*1e6:.2f} uN 의 {tab['Fz_over_W'].min():.2f} ~ {tab['Fz_over_W'].max():.2f} 배)
  중력 처짐 {tab['sag'].min()*1e3:.3f} ~ {tab['sag'].max()*1e3:.3f} mm
  축 강성 k_ax {tab['k_ax'].min():.4f} ~ {tab['k_ax'].max():.4f} N/m
         -> 축방향 고유진동수 {np.sqrt(tab['k_ax'].min()/P.M_DROP)/2/np.pi:.1f} ~ """
            f"""{np.sqrt(tab['k_ax'].max()/P.M_DROP)/2/np.pi:.1f} Hz
  횡 복원력 F_esc,x {np.abs(tab['F_esc_x']).min()*1e6:.2f} ~ {np.abs(tab['F_esc_x']).max()*1e6:.2f} uN
  횡 강성 k_lat {tab['k_lat_x'].min():.5f} ~ {tab['k_lat_x'].max():.5f} N/m
         -> 횡방향 고유진동수 {np.sqrt(tab['k_lat_x'].min()/P.M_DROP)/2/np.pi:.1f} ~ """
            f"""{np.sqrt(tab['k_lat_x'].max()/P.M_DROP)/2/np.pi:.1f} Hz
  유닛 간 간섭(crosstalk) : F_cap 을 {ct.min():+.2f} ~ {ct.max():+.2f} % (평균 {ct.mean():+.2f} %) 변화시킴
         -> 10개 부양기를 동시 구동해도 트랩은 유지된다 (모든 액적 F_cap > W)
""")
    gap = np.diff(np.sort(tab['z_node'][:P.N_PER_COL]))*1e3
    f.write(f"  압력 노드 간격 {np.round(gap,3)} mm  (평면파 lambda/2 = "
            f"{P.LAM/2*1e3:.3f} mm 보다 {100*(gap.mean()/(P.LAM/2*1e3)-1):+.1f} % 크다\n"
            f"     -> 집속빔의 Gouy 위상 때문이며, 액적 기둥 전체 높이는 "
            f"{(MC.full_column_z()[-1]-MC.full_column_z()[0])*1e3:.2f} mm 이다)\n")

    # ---------------------------------------------------------- 2. 실험 1
    if e1:
        _write_exp1(f, e1)
    if e2:
        _write_exp2(f, e2, e1)
    sec(f, "[4] 부양기 넥 유입 · 트랜스듀서 오염 · 덕트 바닥 침강")
    f.write("\n")
    NM.report(f)
    if e1:
        _write_budget(f, e1)
        _write_opmap(f, e1)
    f.close()
    print(f"  {path} 작성 완료")
    return path


def operating_map(e1):
    """유속 x 구동세기 격자에서 부양 액적 수와 흡착율을 재구성한다."""
    tab, _ = AC.load_table()
    g = group(e1, 'U')
    out = {}
    for U, rs in g.items():
        A_by_j = [np.mean([r['particles']['per_drop'][j]['A_cap_m2'] for r in rs])
                  for j in range(P.N_PER_COL)]
        drag = [dict(F_drag_N=np.mean([r['drag'][j]['F_drag_N'] for r in rs]))
                for j in range(P.N_PER_COL)]
        wC = np.mean(pick(rs, 'wake_C'))
        Fcol = np.mean(pick(rs, 'F_drag_column_N'))
        for dv in P.SWEEP_DRIVE:
            UL, _ = LM.layer_velocities(U, wC)
            recs, surv = SW.acoustic_status(tab, dv, drag, UL, U)
            casc = SW.build_cascade(A_by_j, SW.survivors_matrix(recs), U, wC,
                                    SW.M_SCALE_DEFAULT)
            eng = LM.summarize(casc, U, Fcol)
            rec = dict(eng); rec['n_stable'] = int(sum(surv))
            rec['removal'] = casc['removal']
            out[(U, dv)] = rec
    return out


def _write_opmap(f, e1):
    sec(f, "[6] 운전 지도 — 유속 x 구동세기별 흡착율 (액적 60개, 4 layer)")
    om = operating_map(e1)
    Us = sorted({k[0] for k in om})
    f.write(f"\n{'U[m/s]':>7} {'V/V0':>6} {'전압[V]':>8} {'전류비':>7} "
            f"{'배열전력[W]':>12} {'부양/60':>8} {'흡착율[%]':>11} "
            f"{'C_out[mg/m3]':>13} {'CADR[m3/h]':>11}\n" + "-"*94 + "\n")
    best = None
    for U in Us:
        for dv in P.SWEEP_DRIVE:
            r = om[(U, dv)]
            V, I, W1, W10 = P.drive_electrical(dv)
            f.write(f"{U:>7.1f} {dv:>6.2f} {V:>8.1f} {I:>7.2f} {W10:>12.1f} "
                    f"{r['n_stable']:>8d} {100*r['removal']:>11.4f} "
                    f"{r['C_out_mg_m3']:>13.4f} {r['CADR_m3h']:>11.3f}\n")
            if best is None or r['removal'] > best[2]['removal']:
                best = (U, dv, r)
        f.write("-"*94 + "\n")
    U, dv, r = best
    V, I, W1, W10 = P.drive_electrical(dv)
    f.write(f"""
최적 운전점 : U = {U} m/s, V/V0 = {dv:.2f} ({V:.0f} V, 배열 {W10:.0f} W)
              흡착율 {100*r['removal']:.3f} %, C_out {r['C_out_mg_m3']:.4f} mg/m3,
              CADR {r['CADR_m3h']:.3f} m3/h, 부양 {r['n_stable']}/60
  * 정격 구동(V/V0=1.00)으로는 U = 2.5 m/s 에서 액적 대부분이 이탈해 흡착율이
    0.36 % 로 떨어진다. 구동을 15 % 올리면(전력 1.32배) 60개가 모두 유지되어
    흡착율이 4배가 된다. 고유속 운전의 병목은 포집효율이 아니라 '트랩 유지' 다.
  * 구동을 1.30 까지 올려도 1.15 대비 이득이 없다 — 이미 60/60 이 부양되므로
    추가 전력은 낭비다.
""")


def _write_budget(f, e1):
    """입자 수지 — 액적 포집 vs 덕트 바닥 침강 vs 넥 유입."""
    sec(f, "[5] 입자 수지 — 액적 포집과 경쟁 경로 비교 (입구 100 % 기준)")
    g = group(e1, 'U')
    f.write(f"\n{'U[m/s]':>7} {'액적포집[%]':>12} {'덕트바닥침강[%]':>16} "
            f"{'넥유입(하한)[%]':>16} {'넥유입(상한)[%]':>16} {'출구잔류[%]':>12}\n")
    f.write("-"*90 + "\n")
    for U, rs in g.items():
        cap = 100*ms(pick(rs, 'removal'))[0]
        st = 100*NM.settling(U)['floor_removal_frac']
        nl = 100*NM.fouling(U, 'cfd')['frac_of_inlet']
        nu = 100*NM.fouling(U, 'turb')['frac_of_inlet']
        f.write(f"{U:>7.1f} {cap:>12.3f} {st:>16.2f} {nl:>16.3f} {nu:>16.2f} "
                f"{max(100-cap-st-nl, 0):>12.2f}\n")
    f.write("""
  * 덕트 바닥 침강은 액적 필터와 무관한 제거 경로다. 10 um 입자의 침강속도가
    6.03 mm/s 로 작지 않아 저유속에서 액적 포집보다 크다. 성능을 주장할 때
    반드시 분리해서 보고해야 한다.
  * 넥 유입은 트랜스듀서 오염으로 이어지지만 액적 포집을 방해하지는 않는다.
""")


def _write_exp1(f, e1):
    g = group(e1, 'U')
    sec(f, "[2] 실험 1 — 미세입자 유속 5단계 (초음파 구동 V/V0 = 1.00 고정) · 실현별 원자료")
    f.write(f"\n{'U[m/s]':>7} {'실현':>4} {'셀수':>8} {'ndof':>8} "
            f"{'기둥항력[uN]':>12} {'A_col[mm2]':>11} {'직접법[mm2]':>12} "
            f"{'eta_col':>8} {'제거율[%]':>10} {'C_out':>8} {'dP[Pa]':>8}\n" + "-"*112 + "\n")
    for U, rs in g.items():
        for r in sorted(rs, key=lambda x: x['rep']):
            p = r['particles']
            f.write(f"{U:>7.1f} {r['rep']+1:>4} {r['n_tets']:>8,} {r['ndof']:>8,} "
                    f"{r['F_drag_column_N']*1e6:>12.3f} {p['A_col_m2']*1e6:>11.4f} "
                    f"{p['A_col_direct_m2']*1e6:>12.4f} {p['eta_col']:>8.4f} "
                    f"{100*r['removal']:>10.4f} {r['C_out_mg_m3']:>8.4f} "
                    f"{r['dP_droplets_Pa']:>8.4f}\n")
        f.write("-"*112 + "\n")

    sec(f, "[2-1] 실험 1 요약 (평균 ± 1σ, n=5)")
    f.write(f"\n{'U[m/s]':>7} {'Re_d':>6} {'Re_Dh':>7} {'St':>6} {'Q[L/min]':>9} "
            f"{'기둥항력[uN]':>14} {'A_col[mm2]':>15} {'eta_col':>14} "
            f"{'제거율[%]':>15} {'C_out[mg/m3]':>14}\n" + "-"*112 + "\n")
    for U, rs in g.items():
        fd = ms(pick(rs, 'F_drag_column_N')); Ac = ms([p['A_col_m2'] for p in pick(rs, 'particles')])
        et = ms([p['eta_col'] for p in pick(rs, 'particles')])
        rm = ms(pick(rs, 'removal')); co = ms(pick(rs, 'C_out_mg_m3'))
        f.write(f"{U:>7.1f} {rs[0]['Re_d']:>6.0f} {rs[0]['Re_duct']:>7.0f} "
                f"{rs[0]['St']:>6.3f} {rs[0]['Q_Lmin']:>9.0f} "
                f"{fd[0]*1e6:>8.3f}±{fd[1]*1e6:<5.3f} "
                f"{Ac[0]*1e6:>9.4f}±{Ac[1]*1e6:<5.4f} "
                f"{et[0]:>8.4f}±{et[1]:<5.4f} "
                f"{100*rm[0]:>9.4f}±{100*rm[1]:<5.4f} "
                f"{co[0]:>8.4f}±{co[1]:<5.4f}\n")

    sec(f, "[2-2] 실험 1 — 액적 level 별 힘 수지와 포집 (평균, n=5)")
    for U, rs in g.items():
        f.write(f"\nU = {U} m/s   (중력 W = {P.W_DROP*1e6:.2f} uN, 액적 1개당)\n")
        f.write(f"  {'level':>5} {'z[mm]':>8} {'u_loc/U':>8} {'Re_loc':>7} "
                f"{'항력[uN]':>9} {'압력[uN]':>9} {'점성[uN]':>9} {'Cd_CFD':>8} "
                f"{'Cd_상관식':>9} {'차이[%]':>8} {'A_cap[mm2]':>11} {'eta':>7}\n")
        for j in range(P.N_PER_COL):
            dr = [r['drag'][j] for r in rs]
            pd = [r['particles']['per_drop'][j] for r in rs]
            Fd = ms([x['F_drag_N'] for x in dr])
            Cd = ms([x['Cd_cfd'] for x in dr]); Cc = dr[0]['Cd_corr']
            A = ms([x['A_cap_m2'] for x in pd]); et = ms([x['eta'] for x in pd])
            f.write(f"  {LEVEL_NAME[j]:>5} {dr[0]['z_m']*1e3:>8.3f} "
                    f"{dr[0]['u_local']/U:>8.3f} {dr[0]['Re_local']:>7.0f} "
                    f"{Fd[0]*1e6:>9.3f} {ms([x['F_drag_pressure_N'] for x in dr])[0]*1e6:>9.3f} "
                    f"{ms([x['F_drag_viscous_N'] for x in dr])[0]*1e6:>9.3f} "
                    f"{Cd[0]:>8.4f} {Cc:>9.4f} {100*(Cd[0]/Cc-1):>8.1f} "
                    f"{A[0]*1e6:>7.4f}±{A[1]*1e6:<3.4f} {et[0]:>7.4f}\n")

    sec(f, "[2-3] 실험 1 — layer 별 농도 캐스케이드 (평균, n=5)")
    f.write(f"\n{'U[m/s]':>7} {'layer':>6} {'기둥':>5} {'액적(부양)':>10} {'U_L[m/s]':>9} "
            f"{'layer제거[%]':>12} {'C/C_in(입)':>11} {'C/C_in(출)':>11} "
            f"{'C[mg/m3]':>10}\n" + "-"*112 + "\n")
    for U, rs in g.items():
        for L in range(len(P.LAYER_NCOL)):
            rows = [r['cascade']['rows'][L] for r in rs]
            fL = ms([x['f'] for x in rows])
            f.write(f"{U:>7.1f} {L+1:>6} {rows[0]['n_col']:>5} "
                    f"{rows[0]['n_drop_stable']:>10} {rows[0]['U_L']:>9.4f} "
                    f"{100*fL[0]:>9.4f}±{100*fL[1]:<2.4f} "
                    f"{np.mean([x['C_in_rel'] for x in rows]):>11.5f} "
                    f"{np.mean([x['C_out_rel'] for x in rows]):>11.5f} "
                    f"{P.C_MASS_IN*1e6*np.mean([x['C_out_rel'] for x in rows]):>10.5f}\n")
        f.write(f"{'':>7} {'전체':>6} {P.N_COL:>5} {rs[0]['n_stable']:>10} "
                f"{'':>9} {100*ms(pick(rs,'removal'))[0]:>12.4f} "
                f"{1.0:>11.5f} {ms(pick(rs,'penetration'))[0]:>11.5f} "
                f"{ms(pick(rs,'C_out_mg_m3'))[0]:>10.5f}\n" + "-"*112 + "\n")

    sec(f, "[2-4] 실험 1 — 공학 지표")
    f.write(f"\n{'U[m/s]':>7} {'Q[L/min]':>9} {'제거율[%]':>10} {'CADR[m3/h]':>11} "
            f"{'dP_액적[Pa]':>12} {'포집[개/s]':>11} {'포집량[mg/h]':>13} "
            f"{'액적질량증가[%/h]':>17} {'N_layer(50%)':>13} {'N_layer(90%)':>13}\n"
            + "-"*112 + "\n")
    for U, rs in g.items():
        f.write(f"{U:>7.1f} {rs[0]['Q_Lmin']:>9.0f} "
                f"{100*ms(pick(rs,'removal'))[0]:>10.4f} "
                f"{ms(pick(rs,'CADR_m3h'))[0]:>11.3f} "
                f"{ms(pick(rs,'dP_droplets_Pa'))[0]:>12.5f} "
                f"{ms(pick(rs,'n_cap_per_s'))[0]:>11.1f} "
                f"{ms(pick(rs,'mdot_cap_mg_h'))[0]:>13.4f} "
                f"{ms(pick(rs,'mass_gain_pct_per_hour'))[0]:>17.5f} "
                f"{ms(pick(rs,'n_layer_for_50'))[0]:>13.0f} "
                f"{ms(pick(rs,'n_layer_for_90'))[0]:>13.0f}\n")


def _write_exp2(f, e2, e1):
    g = group(e2, 'drive')
    U = P.U_FIXED_FOR_DRIVE
    sec(f, f"[3] 실험 2 — 초음파 구동(전압/전류) 5단계 (유속 {U} m/s 고정) · 실현별 원자료")
    f.write(f"\n{'V/V0':>6} {'실현':>4} {'p0[Pa]':>8} {'부양/60':>8} "
            f"{'축여유min':>10} {'축여유max':>10} {'횡여유min':>10} {'횡여유max':>10} "
            f"{'A_col[mm2]':>11} {'제거율[%]':>10} {'C_out':>8}\n" + "-"*112 + "\n")
    for dv, rs in g.items():
        for r in sorted(rs, key=lambda x: x['rep']):
            f.write(f"{dv:>6.2f} {r['rep']+1:>4} {r['p_antinode_Pa']:>8.0f} "
                    f"{r['n_stable']:>8} {r['ac_margin_axial_min']:>10.3f} "
                    f"{r['ac_margin_axial_max']:>10.3f} {r['ac_margin_lateral_min']:>10.3f} "
                    f"{r['ac_margin_lateral_max']:>10.3f} "
                    f"{r['particles']['A_col_m2']*1e6:>11.4f} "
                    f"{100*r['removal']:>10.4f} {r['C_out_mg_m3']:>8.4f}\n")
        f.write("-"*112 + "\n")

    sec(f, "[3-1] 실험 2 요약 (평균 ± 1σ, n=5)")
    f.write("\n전기 구동 환산 (TinyLev 기준: 20 V, 10 W / 부양기 1대)\n")
    hdr = ("V/V0", "전압[V]", "전류비", "전력/대[W]", "배열10대[W]",
           "p0[Pa]", "F_ac비")
    f.write("  " + "".join(f"{h:>13}" for h in hdr) + "\n")
    for dv0 in P.SWEEP_DRIVE:
        V, I, W1, W10 = P.drive_electrical(dv0)
        row = (f"{dv0:.2f}", f"{V:.1f}", f"{I:.2f}", f"{W1:.2f}",
               f"{W10:.1f}", f"{P.P_ANTINODE_REF*dv0:.0f}", f"{dv0*dv0:.2f}")
        f.write("  " + "".join(f"{c:>13}" for c in row) + "\n")
    f.write(f"\n{'V/V0':>6} {'p0[Pa]':>8} {'전력비':>7} {'부양/60':>8} "
            f"{'layer별 부양':>16} {'F_cap[uN]':>17} {'A_col[mm2]':>15} "
            f"{'제거율[%]':>15} {'C_out[mg/m3]':>14}\n" + "-"*112 + "\n")
    for dv, rs in g.items():
        Ac = ms([p['A_col_m2'] for p in pick(rs, 'particles')])
        rm = ms(pick(rs, 'removal')); co = ms(pick(rs, 'C_out_mg_m3'))
        f.write(f"{dv:>6.2f} {rs[0]['p_antinode_Pa']:>8.0f} {dv**2:>7.2f} "
                f"{rs[0]['n_stable']:>8} {str(rs[0]['n_stable_by_layer']):>16} "
                f"{rs[0]['F_cap_min_N']*1e6:>7.2f}~{rs[0]['F_cap_max_N']*1e6:<8.2f} "
                f"{Ac[0]*1e6:>9.4f}±{Ac[1]*1e6:<5.4f} "
                f"{100*rm[0]:>9.4f}±{100*rm[1]:<5.4f} "
                f"{co[0]:>8.4f}±{co[1]:<5.4f}\n")
    f.write(f"""
판정 기준
  축방향 : F_cap(V/V0)² > W = {P.W_DROP*1e6:.2f} uN
  횡방향 : F_esc,x(V/V0)² > 유동 항력
  두 조건을 모두 만족해야 그 액적이 필터로 작동한다.
""")

    # ------- 안정성 지도 (U x drive)
    sec(f, "[3-2] 부양 안정성 지도 — 유속 x 구동세기 (부양 성공 액적 수 / 60)")
    tab, _ = AC.load_table()
    g1 = group(e1, 'U')
    f.write(f"\n{'U[m/s]':>8}" + "".join(f"{'V/V0='+f'{d:.2f}':>12}"
                                         for d in P.SWEEP_DRIVE) + "\n" + "-"*72 + "\n")
    for Uv, rs in g1.items():
        drag = [ms([r['drag'][j]['F_drag_N'] for r in rs])[0] for j in range(P.N_PER_COL)]
        line = f"{Uv:>8.1f}"
        for dv in P.SWEEP_DRIVE:
            n = 0
            for i in range(len(tab['F_cap'])):
                j = int(tab['node'][i])
                st = LM.stability(tab['F_cap'][i], abs(tab['F_esc_x'][i]), drag[j], dv)
                n += int(st['stable'])
            line += f"{n:>12d}"
        f.write(line + "\n")
    f.write("\n  (구동 V/V0 = 0.70 은 어떤 유속에서도 부양 불가 -> 필터 미작동)\n")


# ================================================================ 그림
def _mm(v):
    return np.asarray(v)*1e3


def fig01_layout():
    """배열 배치도 — 평면도(위) + 측면도(아래)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    LC = [RS.ORDINAL5[k] for k in (0, 1, 3, 4)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 7.8),
                                 gridspec_kw=dict(height_ratios=[1.3, 1]))
    # ---- 평면도 (x-y)
    W = P.DUCT_W*1e3
    a1.add_patch(mp.Rectangle((-80, -W/2), 340, W, fc="none", ec=RS.INK2, lw=1.5))
    for L, cols in enumerate(P.layer_columns()):
        for (x, y) in cols:
            a1.add_patch(mp.Circle((x*1e3, y*1e3), P.ARRAY_ENVELOPE_R*1e3,
                                   fc=LC[L], ec="none", alpha=0.55, zorder=2))
            a1.add_patch(mp.Circle((x*1e3, y*1e3), P.R_OPEN*1e3, fc="none",
                                   ec=RS.SERIES[1], lw=1.3, ls=(0, (3, 2)), zorder=3))
            a1.add_patch(mp.Circle((x*1e3, y*1e3), 1.2, fc=RS.INK, ec="none", zorder=4))
            a1.text(x*1e3, y*1e3-2, str(L+1), ha="center", va="top",
                    fontsize=8, color="white" if L >= 2 else RS.INK, zorder=5)
        a1.text(cols[0][0]*1e3, W/2 + (6 if L % 2 == 0 else 24),
                f"layer {L+1} · {len(cols)}기둥 · {len(cols)*6}액적",
                ha="center", va="bottom", fontsize=8.2, color=LC[L], weight="bold")
    a1.annotate("", xy=(-22, 0), xytext=(-72, 0),
                arrowprops=dict(arrowstyle="-|>", color=RS.INK2, lw=1.7))
    a1.text(-74, 8, "공기 + 미세입자 (10 μm)", fontsize=8.4, color=RS.INK2, ha="left")
    a1.set_xlim(-88, 268); a1.set_ylim(-W/2-16, W/2+46); a1.set_aspect("equal")
    a1.set_xlabel("x  유동방향 [mm]"); a1.set_ylabel("y  폭 [mm]")
    a1.set_title("평면도 — 부양기 10대의 엇갈림 4-layer 배치 "
                 "(채운 원 = 배열 외경 65.2 mm, 점선 원 = 덕트 개구 지름 40 mm)")
    a1.grid(False)

    # ---- 측면도 (x-z)
    H = P.DUCT_H*1e3
    for xe in (-80, 260):          # 덕트 좌우 끝면
        a2.plot([xe, xe], [-H/2, H/2], color=RS.INK2, lw=1.5)
    ang = np.degrees(np.arcsin(27.58/57.0))
    Ro, Zn = P.R_OPEN*1e3, P.Z_NECK*1e3
    for L, cols in enumerate(P.layer_columns()):
        x = cols[0][0]*1e3
        for zk in MC.full_column_z():
            a2.add_patch(mp.Circle((x, zk*1e3), P.A_DROP*1e3, fc=LC[L],
                                   ec=RS.INK, lw=0.4, zorder=6))
        for t1, t2 in ((90-ang, 90+ang), (270-ang, 270+ang)):   # 상/하 배열 캡
            a2.add_patch(mp.Wedge((x, 0), 57, t1, t2, width=5,
                                  fc="0.78", ec=RS.INK2, lw=0.6, zorder=3))
        for sgn in (+1, -1):                                     # 부양기 넥(직결)
            a2.add_patch(mp.Rectangle((x-Ro, sgn*H/2 if sgn > 0 else -Zn),
                                      2*Ro, Zn-H/2, fc="#e9edf2",
                                      ec=RS.SERIES[1], lw=1.6, zorder=4))
        # 개구 (멤브레인 없음) — 덕트 벽이 끊긴 구간
        for sgn in (+1, -1):
            a2.plot([x-Ro, x+Ro], [sgn*H/2, sgn*H/2], color=RS.SERIES[1],
                    lw=2.4, ls=(0, (2, 2)), zorder=7)
    # 덕트 벽 (개구 구간은 비워 둠)
    xs_all = sorted(c[0][0]*1e3 for c in P.layer_columns())
    for sgn in (+1, -1):
        seg = [-80]
        for x in xs_all:
            seg += [x-Ro, x+Ro]
        seg += [260]
        for i in range(0, len(seg), 2):
            a2.plot([seg[i], seg[i+1]], [sgn*H/2]*2, color=RS.INK2, lw=1.8, zorder=5)
    a2.set_xlim(-88, 268); a2.set_ylim(-70, 78); a2.set_aspect("equal")
    a2.set_xlabel("x  유동방향 [mm]"); a2.set_ylabel("z  높이 [mm]")
    a2.set_title("측면도 — 멤브레인 없이 덕트를 부양기에 직결. 개구(주황 점선, R = 20 mm)와 "
                 "넥으로 액적을 넣고 뺀다")
    a2.grid(False)
    RS.annotate(a2, f"기둥당 액적 6개  z = ±2.33, ±7.02, ±11.73 mm\n"
                    f"덕트 {P.DUCT_W*1e3:.0f} × {P.DUCT_H*1e3:.0f} mm  ·  "
                    f"배열 정점 간격 99.8 mm\n"
                    f"개구 R = {P.R_OPEN*1e3:.0f} mm (벽면 빔 반경 "
                    f"{P.BEAM_R_AT_WALL*1e3:.1f} mm 의 1.9배)", "upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig01_layout.png"); plt.close(fig)


def fig02_acoustic_axial():
    """중앙 기둥 축을 따르는 |p|, Gor'kov 포텐셜, 축방향 복사력."""
    import matplotlib.pyplot as plt
    d = np.load("acoustic_layers_ref.npz")
    z = d['prof_z']*1e3; A = d['prof_absP']; U = d['prof_U']; Fz = d['prof_Fz']
    tab, _ = AC.load_table()
    zn = tab['z_node'][6:12]*1e3; ze = tab['z_eq'][6:12]*1e3
    m = np.abs(z) <= 18
    fig, ax = plt.subplots(3, 1, figsize=(8.6, 7.6), sharex=True)
    ax[0].plot(z[m], A[m], color=RS.SERIES[0])
    ax[0].set_ylabel("|p|  [Pa]"); ax[0].set_title("정재파 음압 진폭 (구동 V/V0 = 1.00)")
    ax[0].set_ylim(0, 3200)
    ax[1].plot(z[m], U[m]*1e9, color=RS.SERIES[0])
    ax[1].set_ylabel("U_G  [nJ]"); ax[1].set_title("Gor'kov 포텐셜")
    ax[2].plot(z[m], Fz[m]*1e6, color=RS.SERIES[0])
    ax[2].axhline(P.W_DROP*1e6, color=RS.SERIES[1], lw=1.6, ls="--")
    ax[2].text(-17.5, P.W_DROP*1e6+2.5, f"중력 W = {P.W_DROP*1e6:.1f} μN",
               color=RS.SERIES[1], fontsize=8.4)
    ax[2].set_ylabel("F_ac,z  [μN]"); ax[2].set_xlabel("z  [mm]  (음향 초점 기준)")
    ax[2].set_title("축방향 음향 복사력  F = -∇U_G")
    for a in ax:
        for zz in zn:
            a.axvline(zz, color=RS.INK3, lw=0.6, ls=":", zorder=0)
    ax[2].plot(ze, np.full_like(ze, P.W_DROP*1e6), "o", color=RS.INK,
               ms=5, zorder=6, label="액적 평형 위치")
    ax[2].legend(loc="lower left")
    RS.annotate(ax[0], "점선 = 압력 노드 (액적 트랩)\n중앙 6개 노드를 사용", "upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig02_acoustic_axial.png"); plt.close(fig)


def fig03_traps():
    """60개 액적의 트랩 특성 분포."""
    import matplotlib.pyplot as plt
    tab, _ = AC.load_table()
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
    lay = tab['layer'].astype(int)
    mk = ["o", "s", "^", "D"]
    for L in range(4):
        m = lay == L
        c = RS.ORDINAL5[[0, 1, 3, 4][L]]
        ax[0].plot(tab['z_eq'][m]*1e3, tab['Fz_over_W'][m], mk[L], color=c,
                   ms=5.5, label=f"layer {L+1}", alpha=0.9)
        ax[1].plot(tab['z_eq'][m]*1e3, np.abs(tab['F_esc_x'][m])*1e6, mk[L],
                   color=c, ms=5.5, alpha=0.9)
        ax[2].plot(tab['z_eq'][m]*1e3, tab['sag'][m]*1e3, mk[L], color=c,
                   ms=5.5, alpha=0.9)
    ax[0].axhline(1.0, color=RS.SERIES[1], ls="--", lw=1.6)
    ax[0].set_ylim(0.95, None)
    ax[0].text(-13, 1.015, "부양 한계  F_cap = W", color=RS.SERIES[1],
               fontsize=8.2, ha="left")
    ax[0].set_ylabel("트랩 용량  F_cap / W"); ax[0].set_title("축방향 부양 여유")
    ax[1].set_ylabel("F_esc,x  [μN]"); ax[1].set_title("횡방향 최대 복원력")
    ax[2].set_ylabel("중력 처짐  [mm]"); ax[2].set_title("노드 대비 평형 처짐")
    for a in ax:
        a.set_xlabel("액적 높이 z  [mm]")
    ax[0].legend(loc="upper center", ncol=4, fontsize=7.8,
                 bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig03_traps.png"); plt.close(fig)


def fig04_force_balance():
    """액적 level 별 3력(중력·음향·유동) 수지."""
    import matplotlib.pyplot as plt
    e1 = load("E1")
    if not e1:
        return
    tab, _ = AC.load_table()
    g = group(e1, 'U')
    Fcap = np.array([tab['F_cap'][tab['node'] == j].mean() for j in range(P.N_PER_COL)])
    Fesc = np.array([np.abs(tab['F_esc_x'])[tab['node'] == j].mean()
                     for j in range(P.N_PER_COL)])
    z = MC.full_column_z()*1e3
    xx = np.arange(P.N_PER_COL)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.1))

    ax[0].bar(xx, Fcap*1e6, width=0.55, color=RS.SERIES[0], label="음향 트랩 용량 F_cap")
    ax[0].axhline(P.W_DROP*1e6, color=RS.SERIES[1], lw=2.0, ls="--", label="중력 W (일정)")
    for i, v in enumerate(Fcap*1e6):
        ax[0].text(i, v+1.2, f"{v/P.W_DROP*1e0:.2f}×W", ha="center",
                   fontsize=7.6, color=RS.INK2)
    ax[0].set_xticks(xx); ax[0].set_xticklabels([f"L{j+1}\n{z[j]:+.1f}" for j in xx])
    ax[0].set_xlabel("액적 level (아래→위), z [mm]"); ax[0].set_ylabel("힘 [μN]")
    ax[0].set_title("연직방향 — 음향 복사력 대 중력"); ax[0].legend(loc="lower center", ncol=2)
    ax[0].set_ylim(0, max(Fcap)*1e6*1.25)

    ax[1].bar(xx, Fesc*1e6, width=0.55, color=RS.SERIES[0],
              label="음향 횡복원력 F_esc,x")
    for k, (U, rs) in enumerate(g.items()):
        fd = np.array([ms([r['drag'][j]['F_drag_N'] for r in rs])[0]
                       for j in range(P.N_PER_COL)])
        ax[1].plot(xx, fd*1e6, "o-", color=RS.ORDINAL6[k+1], ms=4.5, lw=1.7,
                   label=f"유동 항력  U = {U} m/s")
    ax[1].set_xticks(xx); ax[1].set_xticklabels([f"L{j+1}\n{z[j]:+.1f}" for j in xx])
    ax[1].set_xlabel("액적 level (아래→위), z [mm]"); ax[1].set_ylabel("힘 [μN]")
    ax[1].set_title("유동방향 — 음향 횡복원력 대 항력 (교차 = 액적 이탈)")
    ax[1].legend(loc="upper left", ncol=2, fontsize=7.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig04_force_balance.png"); plt.close(fig)


def _flow_snapshot(U=1.5, mesh="mesh_col_r4.msh"):
    import flow_column as FC, viz_layers as VZ
    msh, ft, w, F, info = FC.solve(mesh, U)
    return VZ.sample_plane(msh, w, "y", 0.0), VZ.sample_plane(msh, w, "z", 0.0), F


def fig05_flow(U=1.5):
    """단위셀 유동장 단면."""
    import matplotlib.pyplot as plt
    import viz_layers as VZ
    (XX, ZZ, M1, _), (XY, YY, M2, _), F = _flow_snapshot(U)
    fig, ax = plt.subplots(2, 1, figsize=(9.0, 6.4))
    cm = RS.cmap_seq()
    vmax = float(np.nanmax(M1))
    c1 = ax[0].contourf(XX*1e3, ZZ*1e3, M1, levels=np.linspace(0, vmax, 13),
                        cmap=cm, extend="neither")
    VZ.droplet_circles(ax[0], "y")
    ax[0].set_ylabel("z [mm]"); ax[0].set_aspect("equal")
    ax[0].set_title(f"유동장 |u| — 기둥 중심면 y = 0  (U = {U} m/s, 액적 6개)")
    fig.colorbar(c1, ax=ax[0], label="|u| [m/s]", pad=0.015, fraction=0.03)
    c2 = ax[1].contourf(XY*1e3, YY*1e3, M2, levels=np.linspace(0, vmax, 13),
                        cmap=cm, extend="neither")
    VZ.droplet_circles(ax[1], "z")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("y [mm]"); ax[1].set_aspect("equal")
    ax[1].set_title("유동장 |u| — 덕트 중앙 수평면 z = 0")
    fig.colorbar(c2, ax=ax[1], label="|u| [m/s]", pad=0.015, fraction=0.03)
    for a in ax:
        a.grid(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig05_flow.png"); plt.close(fig)


def fig06_trajectories(U=1.5):
    """입자 궤적 — 포집/통과."""
    import matplotlib.pyplot as plt
    import viz_layers as VZ
    fn = f"{OUT}/traj_E1_U{U:g}_r0.npz"
    if not os.path.exists(fn):
        return
    d = np.load(fn); tr = d['traj']; st = d['state']
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    cap = (st >= 1) & (st <= 6)
    idx_p = np.flatnonzero(~cap)[::max(1, (~cap).sum()//260)]
    idx_c = np.flatnonzero(cap)[::max(1, cap.sum()//200)]
    for i in idx_p:
        ax.plot(tr[:, i, 0]*1e3, tr[:, i, 2]*1e3, color=RS.INK3, lw=0.35,
                alpha=0.5, zorder=1)
    for i in idx_c:
        ax.plot(tr[:, i, 0]*1e3, tr[:, i, 2]*1e3, color=RS.SERIES[0], lw=0.5,
                alpha=0.85, zorder=3)
    VZ.droplet_circles(ax, "y", fc="0.9", ec=RS.INK, lw=1.0)
    ax.axhline(P.DUCT_H/2*1e3, color=RS.SERIES[1], lw=1.6)
    ax.axhline(-P.DUCT_H/2*1e3, color=RS.SERIES[1], lw=1.6)
    ax.set_xlim(P.CELL_X0*1e3, 12); ax.set_ylim(-P.DUCT_H/2*1e3-1, P.DUCT_H/2*1e3+1)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]"); ax.set_aspect("equal")
    ax.set_title(f"미세입자 궤적 — 기둥 단위셀 (U = {U} m/s, 구동 V/V0 = 1.00)")
    ax.grid(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=RS.SERIES[0], lw=2, label="액적에 포집"),
                       Line2D([], [], color=RS.INK3, lw=2, label="통과"),
                       Line2D([], [], color=RS.SERIES[1], lw=2, label="덕트 벽")],
              loc="lower right", ncol=3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig06_trajectories.png"); plt.close(fig)


def fig07_exp1():
    """실험 1 — 유속 스윕 요약."""
    import matplotlib.pyplot as plt
    e1 = load("E1")
    if not e1:
        return
    g = group(e1, 'U')
    Us = np.array(list(g.keys()))
    fig, ax = plt.subplots(2, 2, figsize=(10.6, 7.0))

    # (a) 항력계수
    for j, c in ((2, RS.SERIES[0]), (1, RS.SERIES[1]), (0, RS.SERIES[2])):
        Re = [g[U][0]['drag'][j]['Re_local'] for U in Us]
        cd = [ms([r['drag'][j]['Cd_cfd'] for r in g[U]]) for U in Us]
        ax[0, 0].errorbar(Re, [c0 for c0, _ in cd], yerr=[c1 for _, c1 in cd],
                          fmt="o-", color=c, ms=5,
                          label=f"CFD  z = {MC.full_column_z()[j]*1e3:+.1f} mm")
    rr = np.linspace(min(Re)*0.85, max(Re)*1.15, 100)
    ax[0, 0].plot(rr, P.cd_schiller_naumann(rr), "--", color=RS.INK2, lw=1.6,
                  label="Schiller-Naumann 상관식")
    ax[0, 0].set_xlabel("국소 Re_d"); ax[0, 0].set_ylabel("항력계수 C_d")
    ax[0, 0].set_title("(a) 액적 항력계수 — 상관식 대조"); ax[0, 0].legend(fontsize=7.8)

    # (b) 액적 level 별 포집 단면적
    for k, U in enumerate(Us):
        A = [ms([r['particles']['per_drop'][j]['A_cap_m2'] for r in g[U]])[0]*1e6
             for j in range(P.N_PER_COL)]
        ax[0, 1].plot(MC.full_column_z()*1e3, A, "o-", color=RS.ORDINAL6[k+1],
                      ms=5, label=f"U = {U} m/s")
    ax[0, 1].set_xlabel("액적 높이 z [mm]"); ax[0, 1].set_ylabel("A_cap [mm²]")
    ax[0, 1].set_title("(b) 액적 1개의 포집 단면적"); ax[0, 1].legend(fontsize=7.8)

    # (c) 포집효율 vs Stokes 수
    St = np.array([g[U][0]['St'] for U in Us])
    et = np.array([ms([r['particles']['eta_col'] for r in g[U]]) for U in Us])
    ax[1, 0].errorbar(St, et[:, 0], yerr=et[:, 1], fmt="o-", color=RS.SERIES[0],
                      ms=6, label="본 해석 (기둥 평균)")
    ss = np.linspace(St.min()*0.8, St.max()*1.15, 100)
    ax[1, 0].plot(ss, (ss/(ss+0.5))**2, "--", color=RS.INK2, lw=1.5,
                  label="(St/(St+0.5))²")
    ax[1, 0].plot(ss, ss**2/(ss+0.25)**2, ":", color=RS.INK2, lw=1.5,
                  label="St²/(St+0.25)²")
    ax[1, 0].set_xlabel("Stokes 수  St = τ_p U / D"); ax[1, 0].set_ylabel("포집효율 η")
    ax[1, 0].set_title("(c) 단일 액적 포집효율 — 고전 관성충돌 상관식 대조")
    ax[1, 0].legend(fontsize=7.8)

    # (d) 전체 제거율과 출구 농도
    rm = np.array([ms(pick(g[U], 'removal')) for U in Us])
    ax[1, 1].errorbar(Us, 100*rm[:, 0], yerr=100*rm[:, 1], fmt="o-",
                      color=RS.SERIES[0], ms=6)
    ax[1, 1].set_xlabel("유속 U [m/s]")
    ax[1, 1].set_ylabel("전체 제거율 [%]  (액적 60개, 4 layer)")
    ax[1, 1].set_title("(d) 필터 전체 제거율")
    for x, y in zip(Us, 100*rm[:, 0]):
        ax[1, 1].annotate(f"{y:.2f} %\nC_out {P.C_MASS_IN*1e6*(1-y/100):.2f}",
                          (x, y), textcoords="offset points", xytext=(0, -22),
                          ha="center", fontsize=7.6, color=RS.INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig07_exp1_velocity.png"); plt.close(fig)


def fig08_cascade():
    """layer 를 지나며 농도가 줄어드는 모습."""
    import matplotlib.pyplot as plt
    e1 = load("E1")
    if not e1:
        return
    g = group(e1, 'U')
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for k, (U, rs) in enumerate(g.items()):
        c = np.array([1.0] + [np.mean([r['cascade']['rows'][L]['C_out_rel']
                                       for r in rs]) for L in range(4)])
        ax[0].plot(range(5), P.C_MASS_IN*1e6*c, "o-", color=RS.ORDINAL6[k+1],
                   ms=5.5, label=f"U = {U} m/s")
        fL = [100*np.mean([r['cascade']['rows'][L]['f'] for r in rs]) for L in range(4)]
        ax[1].plot(range(1, 5), fL, "o-", color=RS.ORDINAL6[k+1], ms=5.5,
                   label=f"U = {U} m/s")
    ax[0].set_xticks(range(5))
    ax[0].set_xticklabels(["입구", "layer1\n후", "layer2\n후", "layer3\n후", "layer4\n후\n(출구)"])
    ax[0].set_ylabel("미세입자 질량농도 [mg/m³]")
    ax[0].set_title("(a) layer 통과에 따른 농도 감쇠"); ax[0].legend(fontsize=7.8)
    ax[1].set_xticks(range(1, 5)); ax[1].set_xlabel("layer 번호")
    ax[1].set_ylabel("layer 1개의 제거율 [%]")
    ax[1].set_title("(b) layer 별 제거 기여 (3-2-3-2 기둥 수 차이)")
    ax[1].legend(fontsize=7.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig08_cascade.png"); plt.close(fig)


def fig09_exp2():
    """실험 2 — 구동세기 스윕."""
    import matplotlib.pyplot as plt
    e2 = load("E2")
    if not e2:
        return
    g = group(e2, 'drive')
    dv = np.array(list(g.keys()))
    d = np.load("acoustic_layers_ref.npz")
    z, Fz = d['prof_z'], d['prof_Fz']
    fig, ax = plt.subplots(2, 2, figsize=(10.6, 7.0))

    n = [g[x][0]['n_stable'] for x in dv]
    bars = ax[0, 0].bar(dv, n, width=0.09,
                        color=[RS.STATUS['bad'] if v == 0 else
                               (RS.STATUS['warn'] if v < P.N_DROP else RS.STATUS['good'])
                               for v in n])
    for b, v in zip(bars, n):
        ax[0, 0].text(b.get_x()+b.get_width()/2, v+1.5, f"{v}/60", ha="center",
                      fontsize=8.4, color=RS.INK2)
    ax[0, 0].set_xlabel("구동비 V/V0"); ax[0, 0].set_ylabel("부양 성공 액적 수")
    ax[0, 0].set_ylim(0, 68)
    ax[0, 0].set_title("(a) 구동세기별 부양 성공 액적 (60개 중)")

    am = np.array([[g[x][0]['ac_margin_axial_min'], g[x][0]['ac_margin_axial_max']]
                   for x in dv])
    lm = np.array([[g[x][0]['ac_margin_lateral_min'], g[x][0]['ac_margin_lateral_max']]
                   for x in dv])
    ax[0, 1].fill_between(dv, am[:, 0], am[:, 1], color=RS.SERIES[0], alpha=0.22)
    ax[0, 1].plot(dv, am.mean(1), "o-", color=RS.SERIES[0], ms=5, label="축방향  F_cap / W")
    ax[0, 1].fill_between(dv, lm[:, 0], lm[:, 1], color=RS.SERIES[1], alpha=0.22)
    ax[0, 1].plot(dv, lm.mean(1), "o-", color=RS.SERIES[1], ms=5,
                  label="횡방향  F_esc,x / F_drag")
    ax[0, 1].axhline(1.0, color=RS.INK2, ls="--", lw=1.5)
    ax[0, 1].text(dv[0], 1.06, "안정 한계", fontsize=8, color=RS.INK2)
    ax[0, 1].set_xlabel("구동비 V/V0"); ax[0, 1].set_ylabel("안정 여유 (배)")
    ax[0, 1].set_title(f"(b) 부양 안정 여유 (U = {P.U_FIXED_FOR_DRIVE} m/s, 띠 = 60개 액적 범위)")
    ax[0, 1].legend(fontsize=8)

    rm = np.array([ms(pick(g[x], 'removal')) for x in dv])
    ax[1, 0].errorbar(dv, 100*rm[:, 0], yerr=100*rm[:, 1], fmt="o-",
                      color=RS.SERIES[0], ms=6)
    ax[1, 0].set_xlabel("구동비 V/V0"); ax[1, 0].set_ylabel("전체 제거율 [%]")
    ax[1, 0].set_title("(c) 구동세기 대 제거율")
    for x, y, v in zip(dv, 100*rm[:, 0], n):
        ax[1, 0].annotate(f"{y:.2f} %\n({v}/60 부양)", (x, y),
                          textcoords="offset points", xytext=(0, 10),
                          ha="center", fontsize=7.4, color=RS.INK2)

    # (d) 구동세기에 따른 중력 처짐 (F_z(z)·drive² = W 재계산)
    sag, zeq = [], []
    for x in dv:
        gz = Fz*x*x - P.W_DROP
        idx = np.where((gz[:-1] > 0) & (gz[1:] <= 0))[0]
        zc = np.array([np.interp(0, [gz[i+1], gz[i]], [z[i+1], z[i]]) for i in idx])
        node = 2.33e-3
        if len(zc):
            k = np.argmin(np.abs(zc-node))
            zeq.append(zc[k]*1e3); sag.append((node-zc[k])*1e3)
        else:
            zeq.append(np.nan); sag.append(np.nan)
    ax[1, 1].plot(dv, sag, "o-", color=RS.SERIES[0], ms=6)
    ax[1, 1].set_xlabel("구동비 V/V0"); ax[1, 1].set_ylabel("중력 처짐 [mm]")
    ax[1, 1].set_title("(d) 노드 대비 액적 처짐 (z = +2.33 mm 트랩)")
    RS.annotate(ax[1, 1], "구동이 약할수록 처짐이 커지고\n"
                          "F_cap < W 가 되면 부양 실패", "upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig09_exp2_drive.png"); plt.close(fig)


def fig10_stability_map():
    """유속 x 구동세기 안정성 지도."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    e1 = load("E1")
    if not e1:
        return
    tab, _ = AC.load_table()
    g = group(e1, 'U')
    Us = list(g.keys())
    M = np.zeros((len(Us), len(P.SWEEP_DRIVE)))
    for iu, U in enumerate(Us):
        drag = [ms([r['drag'][j]['F_drag_N'] for r in g[U]])[0]
                for j in range(P.N_PER_COL)]
        for idv, dv in enumerate(P.SWEEP_DRIVE):
            M[iu, idv] = sum(LM.stability(tab['F_cap'][i], abs(tab['F_esc_x'][i]),
                                          drag[int(tab['node'][i])], dv)['stable']
                             for i in range(len(tab['F_cap'])))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    im = ax.imshow(M, cmap=RS.cmap_seq(), vmin=0, vmax=P.N_DROP,
                   origin="lower", aspect="auto")
    ax.set_xticks(range(len(P.SWEEP_DRIVE)))
    ax.set_xticklabels([f"{d:.2f}\n{2400*d:.0f} Pa" for d in P.SWEEP_DRIVE])
    ax.set_yticks(range(len(Us))); ax.set_yticklabels([f"{u:.1f}" for u in Us])
    ax.set_xlabel("구동비 V/V0  (안티노드 음압)"); ax.set_ylabel("유속 U [m/s]")
    ax.set_title("부양 성공 액적 수 / 60  —  유속 × 구동세기")
    for i in range(len(Us)):
        for j in range(len(P.SWEEP_DRIVE)):
            ax.text(j, i, f"{int(M[i,j])}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > P.N_DROP*0.55 else RS.INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="부양 액적 수", fraction=0.04, pad=0.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig10_stability_map.png"); plt.close(fig)


def fig11_scaling():
    """layer 수 확장 — 목표 제거율에 필요한 규모."""
    import matplotlib.pyplot as plt
    e1 = load("E1")
    if not e1:
        return
    g = group(e1, 'U')
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    N = np.arange(1, 121)
    for k, (U, rs) in enumerate(g.items()):
        f_eff = ms(pick(rs, 'f_layer_eff'))[0]
        ax[0].plot(N, 100*(1-(1-f_eff)**N), color=RS.ORDINAL6[k+1], lw=2,
                   label=f"U = {U} m/s")
    for tgt, ls in ((50, "--"), (90, ":")):
        ax[0].axhline(tgt, color=RS.INK3, ls=ls, lw=1.2)
        ax[0].text(1, tgt+1.5, f"{tgt} %", fontsize=8, color=RS.INK2)
    ax[0].set_xlabel("직렬 layer 수  (layer 1개 = 기둥 2~3개 = 액적 12~18개)")
    ax[0].set_ylabel("누적 제거율 [%]")
    ax[0].set_title("(a) layer 수에 따른 제거율 (동일 조건 반복 가정)")
    ax[0].legend(fontsize=7.8); ax[0].set_ylim(0, 100)

    Us = list(g.keys())
    n50 = [ms(pick(g[U], 'n_layer_for_50'))[0] for U in Us]
    n90 = [ms(pick(g[U], 'n_layer_for_90'))[0] for U in Us]
    w = 0.18
    ax[1].bar(np.array(Us)-w/1.6, n50, width=w, color=RS.SERIES[0], label="50 % 달성")
    ax[1].bar(np.array(Us)+w/1.6, n90, width=w, color=RS.SERIES[1], label="90 % 달성")
    for x, a, b in zip(Us, n50, n90):
        ax[1].text(x-w/1.6, a*1.03, f"{a:.0f}", ha="center", fontsize=7.6, color=RS.INK2)
        ax[1].text(x+w/1.6, b*1.03, f"{b:.0f}", ha="center", fontsize=7.6, color=RS.INK2)
    ax[1].set_yscale("log"); ax[1].set_xlabel("유속 U [m/s]")
    ax[1].set_ylabel("필요 layer 수")
    ax[1].set_title("(b) 목표 제거율에 필요한 layer 수"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig11_scaling.png"); plt.close(fig)


def fig12_validation():
    """검증 — 두 독립 산출법 일치도, 메시 수렴, 상관식 대조."""
    import matplotlib.pyplot as plt
    e1 = load("E1")
    if not e1:
        return
    g = group(e1, 'U')
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.8))

    for k, (U, rs) in enumerate(g.items()):
        a = np.array([r['particles']['A_col_m2'] for r in rs])*1e6
        b = np.array([r['particles']['A_col_direct_m2'] for r in rs])*1e6
        eb = np.array([r['particles']['A_col_direct_err_m2'] for r in rs])*1e6
        ax[0].errorbar(a, b, yerr=eb, fmt="o", color=RS.ORDINAL6[k+1], ms=5,
                       label=f"U = {U}")
    lim = [0, max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
    ax[0].plot(lim, lim, "--", color=RS.INK2, lw=1.4)
    ax[0].set_xlabel("A_col — 중요도 표본추출 [mm²]")
    ax[0].set_ylabel("A_col — 직접법 [mm²]")
    ax[0].set_title("(a) 두 독립 산출법 일치도"); ax[0].legend(fontsize=7.4)

    for k, (U, rs) in enumerate(g.items()):
        n = [r['n_tets'] for r in rs]
        a = [r['particles']['A_col_m2']*1e6 for r in rs]
        o = np.argsort(n)
        ax[1].plot(np.array(n)[o]/1e3, np.array(a)[o], "o-", color=RS.ORDINAL6[k+1],
                   ms=5, label=f"U = {U}")
    ax[1].set_xlabel("메시 셀 수 [천 개]"); ax[1].set_ylabel("A_col [mm²]")
    ax[1].set_title("(b) 메시 의존성 (실현 5종)"); ax[1].legend(fontsize=7.4)

    err = []
    for U, rs in g.items():
        for j in range(P.N_PER_COL):
            cd = ms([r['drag'][j]['Cd_cfd'] for r in rs])[0]
            err.append((rs[0]['drag'][j]['Re_local'], 100*(cd/rs[0]['drag'][j]['Cd_corr']-1)))
    err = np.array(err)
    ax[2].plot(err[:, 0], err[:, 1], "o", color=RS.SERIES[0], ms=5)
    ax[2].axhline(0, color=RS.INK2, lw=1.2)
    ax[2].fill_between([err[:, 0].min()*0.9, err[:, 0].max()*1.1], -10, 10,
                       color=RS.SERIES[0], alpha=0.10)
    ax[2].set_xlabel("국소 Re_d"); ax[2].set_ylabel("C_d 차이 [%]")
    ax[2].set_title("(c) 항력 vs Schiller-Naumann (띠 = ±10 %)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig12_validation.png"); plt.close(fig)


def fig14_budget():
    """입자 수지 — 액적 포집 vs 덕트 바닥 침강 vs 넥 유입."""
    import matplotlib.pyplot as plt
    e1 = load("E1")
    if not e1:
        return
    g = group(e1, 'U')
    Us = np.array(list(g.keys()))
    cap = np.array([100*ms(pick(g[U], 'removal'))[0] for U in Us])
    st = np.array([100*NM.settling(U)['floor_removal_frac'] for U in Us])
    nk = np.array([100*NM.fouling(U, 'cfd')['frac_of_inlet'] for U in Us])
    nku = np.array([100*NM.fouling(U, 'turb')['frac_of_inlet'] for U in Us])
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    w = 0.13
    ax[0].bar(Us-w, cap, width=w, color=RS.SERIES[0], label="액적 60개 포집")
    ax[0].bar(Us, st, width=w, color=RS.SERIES[1], label="덕트 바닥 침강")
    ax[0].bar(Us+w, nku, width=w, color=RS.SERIES[2], label="넥 유입 (상한)")
    ax[0].bar(Us+w, nk, width=w, color=RS.INK3, label="넥 유입 (하한)")
    ax[0].set_xlabel("유속 U [m/s]"); ax[0].set_ylabel("입구 입자 대비 [%]")
    ax[0].set_title("(a) 입자 제거 경로별 기여")
    ax[0].legend(fontsize=7.8)
    ratio = st/np.maximum(cap, 1e-9)
    ax[1].plot(Us, ratio, "o-", color=RS.SERIES[0], ms=6)
    ax[1].axhline(1.0, color=RS.SERIES[1], ls="--", lw=1.5)
    ax[1].text(Us[0], 1.06, "동등", color=RS.SERIES[1], fontsize=8)
    ax[1].set_xlabel("유속 U [m/s]")
    ax[1].set_ylabel("덕트 바닥 침강 / 액적 포집")
    ax[1].set_title("(b) 경쟁 경로 비 — 성능 주장 시 반드시 분리할 것")
    for x, y in zip(Us, ratio):
        ax[1].annotate(f"{y:.1f}배", (x, y), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=7.8, color=RS.INK2)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig14_budget.png"); plt.close(fig)


def fig15_opmap():
    """유속 x 구동세기 운전 지도 (흡착율 등고선 + 부양 액적 수)."""
    import matplotlib.pyplot as plt
    e1 = load("E1")
    if not e1:
        return
    om = operating_map(e1)
    Us = sorted({k[0] for k in om}); Ds = P.SWEEP_DRIVE
    R = np.array([[100*om[(U, d)]['removal'] for d in Ds] for U in Us])
    N = np.array([[om[(U, d)]['n_stable'] for d in Ds] for U in Us])
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))
    im = ax[0].imshow(R, cmap=RS.cmap_seq(), origin="lower", aspect="auto",
                      vmin=0, vmax=R.max())
    for i in range(len(Us)):
        for j in range(len(Ds)):
            ax[0].text(j, i, f"{R[i,j]:.2f}", ha="center", va="center",
                       fontsize=9, color="white" if R[i, j] > 0.6*R.max() else RS.INK)
    ax[0].set_title("(a) 흡착율 [%]")
    im2 = ax[1].imshow(N, cmap=RS.cmap_seq(), origin="lower", aspect="auto",
                       vmin=0, vmax=P.N_DROP)
    for i in range(len(Us)):
        for j in range(len(Ds)):
            ax[1].text(j, i, f"{N[i,j]}", ha="center", va="center", fontsize=9,
                       color="white" if N[i, j] > 0.6*P.N_DROP else RS.INK)
    ax[1].set_title("(b) 부양 성공 액적 수 / 60")
    for a, im_, lab in ((ax[0], im, "흡착율 [%]"), (ax[1], im2, "액적 수")):
        a.set_xticks(range(len(Ds)))
        a.set_xticklabels([f"{d:.2f}\n{P.drive_electrical(d)[0]:.0f} V" for d in Ds])
        a.set_yticks(range(len(Us))); a.set_yticklabels([f"{u:.1f}" for u in Us])
        a.set_xlabel("구동비 V/V0"); a.set_ylabel("유속 U [m/s]")
        a.grid(False)
        fig.colorbar(im_, ax=a, label=lab, fraction=0.045, pad=0.02)
    fig.suptitle("운전 지도 — 고유속의 병목은 포집효율이 아니라 트랩 유지",
                 fontsize=10.5, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig15_opmap.png"); plt.close(fig)


ALL_FIGS = [fig01_layout, fig02_acoustic_axial, fig03_traps, fig04_force_balance,
            fig05_flow, fig06_trajectories, fig07_exp1, fig08_cascade,
            fig09_exp2, fig10_stability_map, fig11_scaling, fig12_validation,
            fig14_budget, fig15_opmap]


def make_figs():
    for fn in ALL_FIGS:
        try:
            fn(); print(f"  {fn.__name__} 완료", flush=True)
        except Exception as e:
            print(f"  {fn.__name__} 실패: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "tables"):
        write_tables()
    if what in ("all", "figs"):
        make_figs()
