"""
다층 액적 필터 — 공통 물리 파라미터 및 실험계획(DOE)
====================================================
초음파 부양 액적 60개(= 10 기둥 x 6 액적, 4 layer 엇갈림 배열)를 필터로 사용하여
10 um 미세입자를 포함한 공기 유동에서 입자를 흡착시키는 시스템.

기준 장치: TinyLev (Marzo et al., Rev. Sci. Instrum. 88, 085105, 2017)
  - 40 kHz, 직경 10 mm 트랜스듀서 72개 (면당 36개 = 6/12/18 3링, 중앙 제외)
  - 구면 캡 곡률반경 R = 57 mm  -> 두 배열 정점 간격 99.8 mm (논문 ~10 cm 와 일치)
  - 저전압(~20 V) 구동, 소비전력 10 W

좌표계 : 유동 +x , 폭 y , 중력/음향축 z (중력은 -z)
"""
import numpy as np

# ================================================================ 유체 (공기)
RHO_A = 1.2              # 밀도 [kg/m^3]
MU_A  = 1.81e-5          # 점성계수 [Pa s]
NU_A  = MU_A/RHO_A
C0    = 343.0            # 음속 [m/s]

# ================================================================ 액적 (물)
D_DROP = 2.0e-3          # 직경 [m]
A_DROP = D_DROP/2
RHO_W  = 1000.0
C_W    = 1480.0
SIGMA  = 0.072           # 표면장력 [N/m]
G      = 9.81

V_DROP  = np.pi/6*D_DROP**3
M_DROP  = RHO_W*V_DROP
W_DROP  = M_DROP*G                  # 액적 1개 중력 [N]
A_FRONT = np.pi/4*D_DROP**2         # 정면 투영 면적 [m^2]

# ================================================================ 미세입자
D_PART    = 10.0e-6      # 직경 [m]
RHO_PART  = 2000.0       # 밀도 [kg/m^3]
M_PART    = RHO_PART*np.pi/6*D_PART**3
TAU_PART  = RHO_PART*D_PART**2/(18*MU_A)      # 완화시간 [s]
C_MASS_IN = 10.0e-6      # 입구 질량농도 [kg/m^3] = 10 mg/m^3
C_NUM_IN  = C_MASS_IN/M_PART                  # 입구 수농도 [1/m^3]

# ================================================================ 초음파 (TinyLev)
FREQ    = 40.0e3
LAM     = C0/FREQ                   # 8.575 mm
K_WAVE  = 2*np.pi/LAM
OMEGA   = 2*np.pi*FREQ
R_CURV  = 57.0e-3                   # 구면 캡 곡률반경
A_TRANS = 5.0e-3                    # 트랜스듀서 반경 (직경 10 mm)
N_RING  = (6, 12, 18)               # 면당 3링 = 36개, 양면 72개
P_ANTINODE_REF = 2400.0             # 기준 구동(V/V0=1)에서 안티노드 압력 [Pa]
# 전기 구동 (TinyLev 논문: 저전압 ~20 V, 소비전력 10 W / 부양기 1대)
DRIVE_V0_V = 20.0                   # 기준 구동 전압 [V]
DRIVE_P0_W = 10.0                   # 기준 구동 전력 [W] (부양기 1대)
#   음압 p0 ~ V,  전류 I ~ V (정합 부하),  전력 P ~ V*I ~ V^2,  음향복사력 ~ p0^2 ~ V^2
def drive_electrical(dv):
    '''구동비 V/V0 -> (전압[V], 전류비 I/I0, 부양기 1대 전력[W], 배열 10대 전력[W])'''
    return (DRIVE_V0_V*dv, dv, DRIVE_P0_W*dv*dv, 10*DRIVE_P0_W*dv*dv)
ARRAY_ENVELOPE_R = 32.6e-3          # 배열 외곽 반경(트랜스듀서 포함) [m] -> 외경 65.2 mm

# ================================================================ 다층 배열 설계
# --- 액적 기둥 1개 = 연직 정재파의 중앙 6개 압력 노드에 액적 6개
N_PER_COL = 6
# --- 기둥 배치 : 4 layer 엇갈림 (3-2-3-2 = 10 기둥)
LAYER_NCOL = (3, 2, 3, 2)
N_COL      = sum(LAYER_NCOL)        # 10
N_DROP     = N_COL*N_PER_COL        # 60
S_X = 60.0e-3            # layer 간 유동방향 간격 [m]
S_Y = 70.0e-3            # 같은 layer 내 기둥 간 폭방향 간격 [m]
#   최소 배열 중심간 거리 = sqrt(S_X^2+(S_Y/2)^2) = 69.5 mm > 65.2 mm (간섭 없음)

# --- 덕트
DUCT_W = 3*S_Y           # 폭 210 mm  (기둥 1개당 폭 S_Y 담당)
DUCT_H = 32.0e-3         # 높이 32 mm
A_DUCT = DUCT_W*DUCT_H   # 6720 mm^2
DUCT_L = 3*S_X + 2*S_X   # 전체 길이(입구/출구 여유 포함) [m]

def layer_columns():
    """각 layer 의 (x, y) 기둥 중심 목록. 엇갈림 배열."""
    out = []
    for L, n in enumerate(LAYER_NCOL):
        x = L*S_X
        ys = (np.arange(n) - (n-1)/2)*S_Y
        out.append([(x, float(y)) for y in ys])
    return out

# 기둥 1개가 담당하는 덕트 단면적 [m^2] (layer 내 기둥 수로 분할)
def area_per_column(layer):
    return A_DUCT/LAYER_NCOL[layer]

# ================================================================ 부양기 직결 넥
# 액적 교체가 가능해야 하므로 개구를 멤브레인으로 덮지 않고, 덕트를 부양기에 직접
# 연결한다. 각 기둥 위치에서 덕트 천장/바닥에 원형 개구를 내고, 그 위로 부양기
# 하우징(넥)이 트랜스듀서 배열까지 이어진다.
#
#   개구/넥  : 반경 R_OPEN 의 원통, z = DUCT_H/2 -> 배열 구면까지
#   넥 천장  : 트랜스듀서 배열면 = 음향 초점 중심 반경 R_CURV 구면
#
# 개구 반경 결정 : 두 배열은 초점을 향해 집속하므로 덕트 벽면(z=16mm)에서 빔
# 반경은 (rho_t + a_t)*z_w/z_t = 10.45 mm 뿐이다. R_OPEN = 20 mm 로 두면 빔 반경의
# 1.9배 여유가 있어 음향장이 사실상 그대로 유지되고, 2 mm 액적의 투입/회수도 쉽다.
# 개구를 배열 외경(32.6mm)만큼 크게 뚫을 이유가 없다.
R_OPEN   = 20.0e-3                                # 개구/넥 반경
Z_NECK   = np.sqrt(R_CURV**2 - R_OPEN**2)         # 넥이 배열 구면과 만나는 높이 53.4 mm
Z_APEX   = R_CURV                                 # 배열 정점 57.0 mm
NECK_H   = Z_NECK - DUCT_H/2                      # 넥 길이 37.4 mm
BEAM_R_AT_WALL = 10.45e-3                         # 덕트 벽면에서의 빔 반경

def neck_volume():
    """넥+돔 1개 부피 [m^3] (한쪽 면)."""
    from scipy.integrate import quad
    cyl = np.pi*R_OPEN**2*NECK_H
    dome = quad(lambda r: 2*np.pi*r*(np.sqrt(R_CURV**2-r**2)-Z_NECK), 0, R_OPEN)[0]
    return cyl + dome

# ================================================================ CFD 단위셀
# 기둥 1개 주위 유동을 해상. y=0(기둥 축 포함면)과 z=0(기둥 중앙면)이 거울대칭면
# 이므로 1/4 도메인만 계산한다. 측면은 원방(far-field) 처방.
#
# 넥(막다른 공동)의 처리
#   개구 지름 기준 Re_D = rho U (2 R_OPEN)/mu = 1.3x10^3 ~ 6.6x10^3 로, 공동 위
#   전단층이 본질적으로 비정상이라 정상 Navier-Stokes 해가 존재하지 않는다
#   (Newton 이 Re_d >~ 100 에서 잔차 10^-2 수준에서 정체하는 것을 확인).
#   막다른 공동은 순 질량교환이 0 이므로, 시간평균 효과로서 개구면을
#   '자유활동 평면(u_z = 0, 전단응력 0)' 으로 두고 주유동을 푼다.
#   넥으로의 유입/트랜스듀서 오염은 전단층 유입 상관식으로 따로 추정한다
#   (neck_model.py).
CELL_X0, CELL_X1 =  -8.0e-3, 24.0e-3   # 유동방향
CELL_YMAX        =   7.0e-3            # 측면 원방 (= 3.5 D)
CELL_ZMAX        = DUCT_H/2            # 직관부 덕트 벽
CELL_ZTOP        = R_CURV              # 배열 정점

# ================================================================ 실험계획
SWEEP_U     = [0.5, 1.0, 1.5, 2.0, 2.5]        # 실험1: 유속 [m/s]
U_FIXED_FOR_DRIVE = 1.5                        # 실험2 고정 유속
SWEEP_DRIVE = [0.70, 0.85, 1.00, 1.15, 1.30]   # 실험2: 구동비 V/V0

NREP       = 5
H_DROP_REP = [1.24e-4, 1.32e-4, 1.40e-4, 1.48e-4, 1.56e-4]   # 실현별 메시 크기
SEED_REP   = [1000, 2000, 3000, 4000, 5000]

N_SEED_IMP = 7200        # 중요도 표본추출 (액적당 1200개)
N_SEED_DIR = 8000        # 직접법 (입구 단면 균일)
Y_SEED_DIR = 8.0e-3      # 직접법 시딩 폭 |y| <= 8 mm (그 밖에서는 포집 불가)
R_SEED_FAC = 1.75

# ================================================================ 유동 프로파일
# 덕트 Re_Dh = rho U Dh / mu 는 5x10^3 급(천이/난류)이므로 단위셀 입구에는
# 1/7 승 법칙 난류 코어 프로파일을 부여한다. 액적 스케일(Re_d ~ 10^2)은 층류.
POWER_LAW_N = 7.0

def duct_Dh():
    return 4*A_DUCT/(2*(DUCT_W+DUCT_H))

def duct_Re(U):
    return RHO_A*U*duct_Dh()/MU_A

def inlet_profile(z_from_center, U_bulk):
    """
    덕트 중심에서 잰 높이 z 에서의 유속 [m/s].
    1/n 승 법칙  u = u_c (1-|z|/h)^(1/n) 이고, 단면평균이 U_bulk 가 되도록
        <u> = u_c * n/(n+1) = U_bulk   ->   u_c = U_bulk (n+1)/n
    n=7 이면 중심속도는 벌크의 1.143 배이다.
    """
    h, n = DUCT_H/2, POWER_LAW_N
    eta = np.clip(1.0 - np.abs(np.asarray(z_from_center))/h, 0.0, 1.0)
    return U_bulk*(n+1)/n*eta**(1.0/n)


# ================================================================ 무차원수
def reynolds(U, D=D_DROP):  return RHO_A*U*D/MU_A
def stokes(U, D=D_DROP):    return TAU_PART*U/D
def weber(U, D=D_DROP):     return RHO_A*U**2*D/SIGMA

def cd_schiller_naumann(Re):
    Re = np.maximum(Re, 1e-9)
    return 24/Re*(1 + 0.15*Re**0.687)

def drag_correlation(U, D=D_DROP):
    return 0.5*RHO_A*U**2*cd_schiller_naumann(reynolds(U, D))*(np.pi/4*D**2)


if __name__ == "__main__":
    print("="*78)
    print("다층 초음파 부양 액적 필터 — 설계 제원")
    print("="*78)
    print(f"액적    D = {D_DROP*1e3:.1f} mm, m = {M_DROP*1e6:.2f} mg, "
          f"중력 W = {W_DROP*1e6:.2f} uN, ka = {K_WAVE*A_DROP:.3f}")
    print(f"미세입자 d = {D_PART*1e6:.0f} um, rho = {RHO_PART:.0f} kg/m3, "
          f"tau = {TAU_PART*1e3:.3f} ms")
    print(f"입구농도 {C_MASS_IN*1e6:.1f} mg/m3 = {C_NUM_IN:.3e} 개/m3")
    print()
    print(f"배열   기둥 {N_COL}개 x 액적 {N_PER_COL}개 = {N_DROP}개, "
          f"layer {len(LAYER_NCOL)}층 {LAYER_NCOL}")
    print(f"       layer 간격 S_x = {S_X*1e3:.0f} mm, 기둥 간격 S_y = {S_Y*1e3:.0f} mm")
    print(f"       배열 외경 {2*ARRAY_ENVELOPE_R*1e3:.1f} mm, "
          f"최소 중심간 거리 {np.hypot(S_X, S_Y/2)*1e3:.1f} mm")
    for L, cols in enumerate(layer_columns()):
        print(f"       layer {L+1}: x={cols[0][0]*1e3:5.0f} mm, "
              f"y={[f'{y*1e3:+.0f}' for _, y in cols]} mm")
    print(f"덕트   {DUCT_W*1e3:.0f} x {DUCT_H*1e3:.0f} mm, "
          f"A = {A_DUCT*1e6:.0f} mm2, Dh = {duct_Dh()*1e3:.1f} mm")
    print(f"       액적 총 정면면적 {N_DROP*A_FRONT*1e6:.1f} mm2 "
          f"(덕트의 {100*N_DROP*A_FRONT/A_DUCT:.2f} %)")
    print()
    print(f"{'U[m/s]':>7} {'Q[L/min]':>9} {'Re_Dh':>7} {'Re_d':>6} {'St':>6} "
          f"{'We':>7} {'항력상관식[uN]':>14}")
    for U in SWEEP_U:
        print(f"{U:7.1f} {U*A_DUCT*6e4:9.0f} {duct_Re(U):7.0f} {reynolds(U):6.0f} "
              f"{stokes(U):6.3f} {weber(U):7.4f} {drag_correlation(U)*1e6:14.3f}")
