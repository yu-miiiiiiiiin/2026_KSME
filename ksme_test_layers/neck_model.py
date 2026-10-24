"""
부양기 넥(막다른 공동) 유입과 트랜스듀서 오염 추정
==================================================
멤브레인 없이 덕트를 부양기에 직접 연결하면, 덕트 천장/바닥의 개구
(반경 R_OPEN = 20 mm, 상/하 각 10개)를 통해 미세입자가 넥으로 들어가
트랜스듀서 면에 쌓일 수 있다. 세 경로를 나누어 정량화한다.

  [A] 중력 침강 (아래쪽 배열) — 유속과 무관한 주 경로. 가장 확실하다.
      10 um 입자의 종말침강속도 v_s = tau_p g = 6.03 mm/s.
  [B] 전단층 교환 (상/하 공통) — 개구 위 전단층에 의한 왕복 교환.
      개구 지름 기준 Re_D = 1.3e3 ~ 6.6e3 로 비정상이라 정상해가 없다.
      (b1) 정상 CFD 하한 : 넥을 실제로 뚫은 형상에서 측정한 개구 통과유량
                           (U=1.5 m/s 에서 직관 유량의 0.21 %)
      (b2) 전단층 상한   : 개방형 공동의 유입계수 C_e = 0.02 (v_e = C_e U)
      실제 부착은 이 상한보다 훨씬 작다. 넥 길이/지름 = 0.94 로 교환 유체가
      배열면까지 닿기 전에 대부분 되돌아 나가고, 위쪽 넥은 중력이 되돌린다.
  [C] 비교 — 덕트 바닥 전체 침강. 액적 필터와 경쟁하는 제거 경로이므로
      성능을 보고할 때 반드시 분리해야 한다.
"""
import numpy as np
import params_layers as P

C_E_TURB = 0.02          # 개방형 공동 전단층 유입계수 (문헌 0.02~0.05 의 하단)
EXCH_RATIO_CFD = 0.0021  # 정상 CFD 로 측정한 개구 교환유량 / 직관 유량 (U=1.5)
A_ARRAY = 36*np.pi*P.A_TRANS**2      # 배열 1면의 트랜스듀서 유효 면적 [m^2]
DUCT_LEN = 5*P.S_X                   # 덕트 길이 (layer 4개 + 입출구 여유) [m]


def opening_area():
    return np.pi*P.R_OPEN**2


# ---------------------------------------------------------------- [A] 침강
def settling(U_bulk, C_mass=P.C_MASS_IN):
    v_s = P.TAU_PART*P.G
    A_open_bot = opening_area()*P.N_COL
    A_floor = P.DUCT_W*DUCT_LEN
    n_dot = P.C_NUM_IN*v_s*A_open_bot
    mdot = C_mass*v_s*A_open_bot
    areal = mdot/(P.N_COL*A_ARRAY)
    n_in = P.C_NUM_IN*U_bulk*P.A_DUCT
    return dict(U=U_bulk, v_settle_mm_s=v_s*1e3,
                A_open_bot_cm2=A_open_bot*1e4, A_floor_cm2=A_floor*1e4,
                open_frac_of_floor=A_open_bot/A_floor,
                n_dot_per_s=n_dot, mdot_mg_h=mdot*3.6e9,
                areal_g_m2_h=areal*3.6e6,
                hours_to_1g_m2=1.0/max(areal*3.6e6, 1e-30),
                frac_of_inlet=n_dot/n_in,
                floor_removal_frac=P.C_NUM_IN*v_s*A_floor/n_in)


# ---------------------------------------------------------------- [B] 교환
def exchange(U_bulk, mode="turb"):
    """개구 1개의 교환 체적유량 [m^3/s]."""
    if mode == "cfd":
        return EXCH_RATIO_CFD*U_bulk*(2*P.CELL_YMAX)*P.DUCT_H
    return C_E_TURB*U_bulk*opening_area()


def fouling(U_bulk, mode="turb", C_mass=P.C_MASS_IN, n_open=2*P.N_COL):
    Qe = exchange(U_bulk, mode)
    mdot = C_mass*Qe*n_open
    n_dot = P.C_NUM_IN*Qe*n_open
    areal = mdot/(n_open*A_ARRAY)
    return dict(mode=mode, U=U_bulk, Q_exchange_Lmin=Qe*6e4,
                v_exchange_mm_s=Qe/opening_area()*1e3,
                mdot_mg_h=mdot*3.6e9, n_dot_per_s=n_dot,
                areal_g_m2_h=areal*3.6e6,
                hours_to_1g_m2=1.0/max(areal*3.6e6, 1e-30),
                frac_of_inlet=n_dot/max(P.C_NUM_IN*U_bulk*P.A_DUCT, 1e-30))


def report(fh=None):
    w = (fh.write if fh else (lambda s: print(s, end="")))
    st = settling(P.U_FIXED_FOR_DRIVE)
    w("="*104 + "\n")
    w(f"부양기 넥 유입과 트랜스듀서 오염 (개구 {P.R_OPEN*1e3:.0f} mm, "
      f"상/하 {2*P.N_COL}개, 입구 {P.C_MASS_IN*1e6:.0f} mg/m3)\n")
    w("="*104 + "\n\n")
    w("[A] 중력 침강 — 아래쪽 배열 (유속 무관, 주 경로)\n")
    w(f"    침강속도 v_s = tau_p g = {st['v_settle_mm_s']:.2f} mm/s\n")
    w(f"    아래 개구 총면적 {st['A_open_bot_cm2']:.1f} cm2 "
      f"(덕트 바닥 {st['A_floor_cm2']:.0f} cm2 의 {100*st['open_frac_of_floor']:.1f} %)\n")
    w(f"    유입 {st['n_dot_per_s']:.0f} 개/s = {st['mdot_mg_h']:.3f} mg/h\n")
    w(f"    트랜스듀서 면 오염 {st['areal_g_m2_h']:.4f} g/(m2 h) "
      f"-> 1 g/m2 까지 {st['hours_to_1g_m2']:.0f} 시간\n\n")
    w("[B] 전단층 교환 — 위/아래 공통 (하한/상한)\n")
    w(f"{'U[m/s]':>7} {'Re_D':>7} {'v_e CFD하한[mm/s]':>19} {'v_e 전단층[mm/s]':>18} "
      f"{'유입 하한[개/s]':>16} {'상한[개/s]':>13} {'상한 오염[g/m2/h]':>18}\n")
    w("-"*104 + "\n")
    for U in P.SWEEP_U:
        a, b = fouling(U, "cfd"), fouling(U, "turb")
        ReD = P.RHO_A*U*2*P.R_OPEN/P.MU_A
        w(f"{U:7.1f} {ReD:7.0f} {a['v_exchange_mm_s']:19.3f} "
          f"{b['v_exchange_mm_s']:18.2f} {a['n_dot_per_s']:16.1f} "
          f"{b['n_dot_per_s']:13.1f} {b['areal_g_m2_h']:18.3f}\n")
    w("\n[C] 비교 — 덕트 바닥 전체 침강 (액적 필터와 경쟁하는 제거 경로)\n")
    for U in P.SWEEP_U:
        s2 = settling(U)
        w(f"    U = {U:.1f} m/s : 입구 입자의 {100*s2['floor_removal_frac']:.2f} % 가 "
          f"덕트 바닥에 침강 (그 중 {100*s2['open_frac_of_floor']:.1f} % 가 아래 배열로)\n")
    w("\n설계 시사점\n")
    w("  * 아래쪽 개구에는 음향 투과 금속망(개구율 높은 미세망)을 대어 침강 입자를 받고\n")
    w("    주기 청소하는 편이 좋다. 위쪽 개구는 중력이 입자를 되돌리므로 덜 급하다.\n")
    w("  * 넥 유입은 액적 포집을 방해하지 않는다 — 포집은 개구보다 3.3 mm 아래,\n")
    w("    액적 표면 1 mm 이내에서 끝난다.\n")
    w("  * 덕트 바닥 침강이 액적 포집과 같은 자릿수이므로 성능은 반드시 분리 보고한다.\n")


if __name__ == "__main__":
    report()
