"""问题 3（碳化硅复核）：按多光束模型复核附件 1、附件 2。

与问题 2 使用相同的滤波、峰谷定位、包络与厚度组合流程，仅将反射率模型
由双光束（问题 2 已算）替换为多光束模型，判断碳化硅数据是否受多光束
干涉影响，并给出逆方差加权合并的最终结果。
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from common import (
    ROOT,
    centered_moving_average,
    cubic_spline_extrema,
    extract_envelope,
    invert_refractive_index,
    invert_refractive_index_multibeam,
    load_spectrum,
    thickness_pairs,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

W_MIN = 1000.0
THICK_LO, THICK_HI = 2000.0, 3800.0
TABLE_WS = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]
THETA = {1: 10.0, 2: 15.0}
PROM = 5e-4


def theoretical_multibeam(
    w: np.ndarray,
    d_cm: float,
    n_spl: CubicSpline,
    A_spl: CubicSpline,
    B_spl: CubicSpline,
    theta1_deg: float,
    phi: float,
) -> np.ndarray:
    s2 = np.sin(np.radians(theta1_deg)) ** 2
    delta = 2.0 * d_cm * np.sqrt(n_spl(w) ** 2 - s2)
    cosd = np.cos(2.0 * np.pi * delta * w + phi)
    I1 = A_spl(w) ** 2
    I2 = B_spl(w) ** 2
    x = np.sqrt(I1 * I2)
    return (I1 + I2 + 2.0 * x * cosd) / (1.0 + I1 * I2 + 2.0 * x * cosd)


def process_attachment(att: int) -> dict:
    theta1 = THETA[att]
    df = load_spectrum(att)
    work = df.loc[df["omega"] >= W_MIN].reset_index(drop=True)
    w = work["omega"].to_numpy()
    r_raw = work["R"].to_numpy()
    r_filt = centered_moving_average(r_raw, 101)
    extrema = cubic_spline_extrema(w, r_filt, prominence_threshold=PROM)
    x_env, rmax, rmin = extract_envelope(w, extrema)

    A2, B2, n2, ns2 = invert_refractive_index(rmax, rmin)
    Am, Bm, nm, nsm = invert_refractive_index_multibeam(rmax, rmin)
    n2_spl = CubicSpline(x_env, n2)
    nm_spl = CubicSpline(x_env, nm)
    Am_spl = CubicSpline(x_env, Am)
    Bm_spl = CubicSpline(x_env, Bm)

    ext_thick = extrema.loc[
        (extrema["omega"] >= THICK_LO) & (extrema["omega"] <= THICK_HI)
    ].reset_index(drop=True)
    pairs2 = thickness_pairs(ext_thick, n2_spl, theta1)
    pairsm = thickness_pairs(ext_thick, nm_spl, theta1)

    stats = {
        "two": {
            "n_pairs": len(pairs2),
            "mean_um": float(pairs2["d_um"].mean()),
            "std_um": float(pairs2["d_um"].std(ddof=1)),
        },
        "multi": {
            "n_pairs": len(pairsm),
            "mean_um": float(pairsm["d_um"].mean()),
            "std_um": float(pairsm["d_um"].std(ddof=1)),
        },
    }

    # 多光束理论光谱拟合
    d_final_cm = stats["multi"]["mean_um"] * 1e-4
    fit_mask = (
        (w >= THICK_LO)
        & (w <= THICK_HI)
        & (w >= x_env.min())
        & (w <= x_env.max())
    )

    def obj(phi: float) -> float:
        r_the = theoretical_multibeam(
            w[fit_mask], d_final_cm, nm_spl, Am_spl, Bm_spl, theta1, phi
        )
        return float(np.mean((r_the - r_filt[fit_mask]) ** 2))

    res = minimize_scalar(obj, bounds=(0.0, 2.0 * np.pi), method="bounded")
    phi_fit = float(res.x)
    r_the_fit = theoretical_multibeam(
        w[fit_mask], d_final_cm, nm_spl, Am_spl, Bm_spl, theta1, phi_fit
    )
    rmse = float(np.sqrt(np.mean((r_the_fit - r_filt[fit_mask]) ** 2)))

    return {
        "att": att,
        "theta1": theta1,
        "w": w,
        "r_raw": r_raw,
        "r_filt": r_filt,
        "x_env": x_env,
        "n2": n2,
        "nm": nm,
        "n2_spl": n2_spl,
        "nm_spl": nm_spl,
        "Am_spl": Am_spl,
        "Bm_spl": Bm_spl,
        "n2_table": n2_spl(np.array(TABLE_WS)),
        "nm_table": nm_spl(np.array(TABLE_WS)),
        "ns2_table": CubicSpline(x_env, ns2)(np.array(TABLE_WS)),
        "nsm_table": CubicSpline(x_env, nsm)(np.array(TABLE_WS)),
        "pairs2": pairs2,
        "pairsm": pairsm,
        "stats": stats,
        "phi": phi_fit,
        "rmse": rmse,
        "d_final_cm": d_final_cm,
    }


def make_figures(r1: dict, r2: dict) -> None:
    # 图：两种模型的折射率对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, r in zip(axes, (r1, r2)):
        ax.plot(r["x_env"], r["n2"], lw=1.2, label="双光束模型")
        ax.plot(r["x_env"], r["nm"], lw=1.2, ls="--", label="多光束模型")
        ax.set_title(f"附件{r['att']}（{r['theta1']:.0f}°）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("折射率 n")
    fig.suptitle("碳化硅外延层折射率：双光束与多光束模型对比")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_sic_refractive_index.png", dpi=150)
    plt.close(fig)

    # 图：多光束理论光谱与实测对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, r in zip(axes, (r1, r2)):
        m = (r["w"] >= THICK_LO) & (r["w"] <= THICK_HI)
        r_the = theoretical_multibeam(
            r["w"][m], r["d_final_cm"], r["nm_spl"], r["Am_spl"], r["Bm_spl"], r["theta1"], r["phi"]
        )
        ax.plot(r["w"][m], r["r_filt"][m] * 100, lw=0.5, color="k", alpha=0.7, label="实测（滤波）")
        ax.plot(r["w"][m], r_the * 100, lw=1.1, color="tab:red", label="多光束理论")
        ax.set_title(f"附件{r['att']}（d={r['stats']['multi']['mean_um']:.3f} μm）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("反射率 (%)")
    fig.suptitle("碳化硅数据：实测与多光束理论光谱对比（2000–3800 cm-1）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_sic_theory_compare.png", dpi=150)
    plt.close(fig)


def save_results(r1: dict, r2: dict) -> None:
    n_rows = []
    for r in (r1, r2):
        n_rows.append(
            pd.DataFrame(
                {
                    "波数_cm-1": TABLE_WS,
                    f"附件{r['att']}_双光束_n": np.round(r["n2_table"], 4),
                    f"附件{r['att']}_双光束_ns": np.round(r["ns2_table"], 4),
                    f"附件{r['att']}_多光束_n": np.round(r["nm_table"], 4),
                    f"附件{r['att']}_多光束_ns": np.round(r["nsm_table"], 4),
                }
            )
        )
    pd.concat(n_rows, axis=1).to_csv(
        RES_DIR / "q3_sic_refractive_index.csv", index=False, encoding="utf-8-sig"
    )

    q2 = pd.read_csv(RES_DIR / "q2_summary.csv")
    summary = []
    for r in (r1, r2):
        q2row = q2[(q2["附件"] == f"附件{r['att']}") & (q2["口径"] == "峰+谷")].iloc[0]
        summary.append(
            {
                "附件": f"附件{r['att']}",
                "入射角": f"{r['theta1']:.0f}°",
                "双光束_均值_um": q2row["厚度均值_um"],
                "双光束_标准差_um": q2row["标准差_um"],
                "多光束_均值_um": r["stats"]["multi"]["mean_um"],
                "多光束_标准差_um": r["stats"]["multi"]["std_um"],
                "均值绝对差_um": abs(q2row["厚度均值_um"] - r["stats"]["multi"]["mean_um"]),
                "理论光谱RMSE_%": r["rmse"] * 100,
            }
        )
    pd.DataFrame(summary).to_csv(
        RES_DIR / "q3_sic_summary.csv", index=False, encoding="utf-8-sig"
    )


def combine(results: list[dict]) -> dict:
    w = np.array([1.0 / r["stats"]["multi"]["std_um"] ** 2 for r in results])
    x = np.array([r["stats"]["multi"]["mean_um"] for r in results])
    mean = float(np.sum(w * x) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    half = 1.96 * se
    return {"mean_um": mean, "se_um": se, "ci_low": mean - half, "ci_high": mean + half}


def print_summary(r1: dict, r2: dict, comb: dict) -> None:
    print("=" * 66)
    for r in (r1, r2):
        print(f"附件{r['att']}（入射角 {r['theta1']:.0f}°）")
        print(f"  双光束：{r['stats']['two']['mean_um']:.3f} ± {r['stats']['two']['std_um']:.4f} μm")
        print(f"  多光束：{r['stats']['multi']['mean_um']:.3f} ± {r['stats']['multi']['std_um']:.4f} μm")
        print(f"  均值绝对差：{abs(r['stats']['two']['mean_um'] - r['stats']['multi']['mean_um']):.4f} μm")
        print(f"  理论光谱 RMSE={r['rmse']*100:.3f}%（反射率百分点）")
        print("  折射率（1500~3500 cm-1）双光束：", " ".join(f"{v:.2f}" for v in r["n2_table"]))
        print("  折射率（1500~3500 cm-1）多光束：", " ".join(f"{v:.2f}" for v in r["nm_table"]))
    print(f"碳化硅最终合并（逆方差加权）：{comb['mean_um']:.3f} μm，"
          f"SE={comb['se_um']:.4f}，95% CI [{comb['ci_low']:.3f}, {comb['ci_high']:.3f}]")
    print("=" * 66)


def main() -> None:
    r1 = process_attachment(1)
    r2 = process_attachment(2)
    make_figures(r1, r2)
    save_results(r1, r2)
    comb = combine([r1, r2])
    print_summary(r1, r2, comb)
    print(f"\n图表已保存到 {FIG_DIR}")
    print(f"数值结果已保存到 {RES_DIR}")


if __name__ == "__main__":
    sys.exit(main())
