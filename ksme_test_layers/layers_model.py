"""
4 layer x 10 기둥 x 6 액적 = 60 액적 필터의 농도 캐스케이드
============================================================
단위셀(기둥 1개) 해석으로 얻은 '포집 단면적' A_col 을 배열 전체로 확장한다.

  layer L 제거율      f_L = n_L · A_col(U_L) / A_duct
  layer 통과 후 농도  C_{L+1} = C_L (1 − f_L)
  전체 통과율         C_out/C_in = Π_L (1 − f_L)
  전체 제거율(흡착율) E = 1 − C_out/C_in

A_col 은 덕트 단면적과 무관한 물리량이므로 배열/덕트 설계를 바꿔도 이전 가능하다.

횡방향 혼합 가정 (두 경계값을 모두 계산)
  (a) 완전혼합  : 각 layer 입구에서 농도가 단면에 균일. 난류덕트(Re_Dh~5x10^3)
                  에서 layer 간격 60 mm 를 지나는 동안 난류확산 폭이
                  포집 유관(r~1 mm) 보다 훨씬 크므로 타당하다.
  (b) 비혼합    : 같은 유관이 정렬된 하류 layer(1↔3, 2↔4)를 다시 만난다고 보는
                  하한. 정렬 layer 의 유효 포집 단면적을 (1−f_up) 로 감쇠시킨다.

후류 속도결손
  단위셀 해에서 x = 5..25 mm 의 결손을 측정하고 층류 후류 상사해 δ ∝ x^-1 로
  하류 layer 위치(60, 120 mm)까지 외삽하여 정렬된 layer 의 접근속도를 낮춘다.
"""
import numpy as np
import params_layers as P


def fit_wake_decay(wake):
    """
    wake: {x[m]: [결손_0, 결손_1, 결손_2]} -> 액적 평균 결손에 대해 δ = C/x 적합.
    반환 (C, 함수 δ(x))
    """
    xs = np.array(sorted(wake.keys()))
    d = np.array([np.nanmean(wake[float(x)]) for x in xs])
    ok = np.isfinite(d) & (d > 0) & (xs >= 8e-3)
    if ok.sum() < 2:
        return 0.0, (lambda x: np.zeros_like(np.asarray(x, float)))
    C = float(np.mean(d[ok]*xs[ok]))
    return C, (lambda x: C/np.asarray(x, float))


def layer_velocities(U_bulk, wake_C, aligned_gap=2*P.S_X):
    """
    layer 별 접근 벌크유속.
    엇갈림 배열이므로 layer 1↔3, 2↔4 만 정렬(같은 y)이고 간격은 2·S_x = 120 mm.
    layer 2 는 layer 1 로부터 y 로 S_y/2 = 35 mm (17.5 D) 벗어나 후류 밖이다.
    """
    dz = wake_C/aligned_gap if wake_C > 0 else 0.0
    return np.array([U_bulk, U_bulk, U_bulk*(1-dz), U_bulk*(1-dz)]), dz


def cascade(A_col_of_U, U_bulk, wake_C=0.0, mixed=True):
    """
    A_col_of_U : 접근 벌크유속 U 에서 기둥 1개의 총 포집 단면적 [m^2] 을 주는 함수
    반환: layer 별/전체 제거율
    """
    UL, dz = layer_velocities(U_bulk, wake_C)
    rows = []
    pen = 1.0
    for L, n in enumerate(P.LAYER_NCOL):
        A = A_col_of_U(UL[L])
        f = n*A/P.A_DUCT
        if not mixed and L >= 2:
            f *= (1.0 - rows[L-2]['f'])      # 정렬된 상류 layer 가 이미 청소한 몫
        f = min(max(f, 0.0), 1.0)
        rows.append(dict(layer=L+1, n_col=n, n_drop=n*P.N_PER_COL, U_L=UL[L],
                         A_col_m2=A, f=f, C_in_rel=pen, C_out_rel=pen*(1-f)))
        pen *= (1-f)
    return dict(rows=rows, penetration=pen, removal=1.0-pen,
                wake_deficit=dz, U_layers=UL)


def summarize(casc, U_bulk, F_drag_col, C_in=P.C_MASS_IN):
    """농도·질량유량·압력손실 등 공학 지표."""
    E = casc['removal']
    Q = U_bulk*P.A_DUCT                       # 체적유량 [m^3/s]
    mdot_cap = C_in*Q*E                       # 포집 질량유량 [kg/s]
    dP = P.N_COL*F_drag_col/P.A_DUCT          # 액적 항력에 의한 압력손실 [Pa]
    out = dict(U_bulk=U_bulk, Q_m3s=Q, Q_Lmin=Q*6e4,
               removal=E, penetration=casc['penetration'],
               C_in_mg_m3=C_in*1e6, C_out_mg_m3=C_in*(1-E)*1e6,
               C_in_num_m3=P.C_NUM_IN, C_out_num_m3=P.C_NUM_IN*(1-E),
               mdot_cap_kg_s=mdot_cap, mdot_cap_mg_h=mdot_cap*3.6e9,
               n_cap_per_s=P.C_NUM_IN*Q*E,
               mass_gain_pct_per_hour=100*mdot_cap*3600/(P.N_DROP*P.M_DROP),
               dP_droplets_Pa=dP,
               CADR_m3h=Q*3600*E,
               quality_Pa_inv=(-np.log(max(casc['penetration'], 1e-12))/dP
                               if dP > 0 else np.nan))
    # 목표 제거율에 필요한 layer 수 (같은 조건 반복)
    f_eff = 1.0 - casc['penetration']**(1/len(P.LAYER_NCOL))   # layer 평균
    for tgt in (0.5, 0.9, 0.99):
        out[f"n_layer_for_{int(tgt*100)}"] = (np.log(1-tgt)/np.log(1-f_eff)
                                              if 0 < f_eff < 1 else np.inf)
    out['f_layer_eff'] = f_eff
    return out


def stability(F_cap, F_esc_x, F_drag_drop, drive=1.0):
    """
    액적 1개의 부양 안정성.
      축방향 : F_cap·drive² > W          (중력 지지)
      횡방향 : F_esc_x·drive² > F_drag   (유동에 의한 이탈 방지)
    """
    s = drive*drive
    ax = F_cap*s/P.W_DROP
    lat = F_esc_x*s/F_drag_drop if F_drag_drop > 0 else np.inf
    # 합성: 중력과 항력이 직교하므로 필요한 복원력의 크기로 판정
    need = np.hypot(P.W_DROP, F_drag_drop)
    have = min(F_cap*s, F_esc_x*s)
    return dict(margin_axial=ax, margin_lateral=lat,
                margin_combined=have/need, stable=bool(ax > 1 and lat > 1))


if __name__ == "__main__":
    # 데모: A_col 이 U 에 약하게 의존한다고 두고 캐스케이드 구조 확인
    A_of_U = lambda U: 12.0e-6*(U/1.5)**0.25
    c = cascade(A_of_U, 1.5, wake_C=2e-3)
    print(f"후류 결손(120 mm) = {c['wake_deficit']*100:.2f} %")
    for r in c['rows']:
        print(f"  layer {r['layer']}: 기둥 {r['n_col']} 액적 {r['n_drop']:2d} "
              f"U={r['U_L']:.3f} A_col={r['A_col_m2']*1e6:.2f} mm2 "
              f"f={100*r['f']:.3f}%  C/C_in {r['C_in_rel']:.4f} -> {r['C_out_rel']:.4f}")
    print(f"전체 제거율 {100*c['removal']:.3f} %")
    print(summarize(c, 1.5, 3.3e-6*6)['CADR_m3h'])
