"""
그림 공통 스타일 — 검증된 기본 팔레트 (dataviz 스킬 reference instance)
=======================================================================
* 범주형(categorical) : 슬롯 1~3 만 사용. 이 3색은 두 모드 모두 all-pairs 게이트를
  통과한다(문서화된 검증값). 4번째 슬롯부터는 all-pairs 기준을 통과하지 못하므로
  계열이 4개 이상이면 순서형 램프나 소형다중(small multiple)으로 바꾼다.
* 순서형(ordinal)     : 파랑 단일 램프. 밝은 면에서는 step 250 보다 밝게 쓰지 않는다.
* 발산형(diverging)   : 파랑 <-> 빨강, 중간은 회색.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
INK3      = "#8a8880"
GRID      = "#e3e2dd"

SERIES    = ["#2a78d6", "#eb6834", "#1baf7a"]          # 범주형 슬롯 1~3
ORDINAL6  = ["#86b6ef", "#5598e7", "#2a78d6",
             "#1c5cab", "#104281", "#0d366b"]           # 파랑 램프 250~700
ORDINAL5  = ["#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
SEQ       = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
DIVERGE   = ("#2a78d6", "#f0efec", "#e34948")
STATUS    = dict(good="#1baf7a", warn="#eda100", bad="#e34948")

KFONT_CANDIDATES = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
                    "Malgun Gothic", "DejaVu Sans"]


def use():
    avail = {f.name for f in mpl.font_manager.fontManager.ttflist}
    fam = next((f for f in KFONT_CANDIDATES if f in avail), "DejaVu Sans")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": fam, "axes.unicode_minus": False,
        "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.titleweight": "bold", "axes.titlecolor": INK,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.grid": True,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.markersize": 5.0,
        "figure.dpi": 140, "savefig.dpi": 140,
        "savefig.bbox": "tight", "errorbar.capsize": 2.5,
    })
    return fam


def cmap_seq():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seqblue", SEQ)


def annotate(ax, text, loc="upper left"):
    xy = {"upper left": (0.02, 0.97), "upper right": (0.98, 0.97),
          "lower left": (0.02, 0.04), "lower right": (0.98, 0.04)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8.2, color=INK2, linespacing=1.5)
