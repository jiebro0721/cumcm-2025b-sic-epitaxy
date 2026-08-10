"""问题 3：硅外延层厚度的多光束干涉分析（附件 3、附件 4）。

流程：包络构造覆盖 ω>=500（避免低波数极值点外推）-> 滤波 -> 峰谷定位 ->
双光束/多光束两种模型反演折射率 -> 必要条件判定 -> 全组合计算厚度 ->
多光束理论光谱可靠性验证 -> 两组数据逆方差加权合并。
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

W_ENV = 500.0       # 包络/峰谷构造下限（覆盖低波数极值，避免外推）
W_MIN = 1000.0      # 厚度计算所用极值的波数下限
FIT_LO = 1500.0     # 多光束理论光谱拟合下限（避开 1000-1500 强变化区）
THICK_HI = 3800.0   # 厚度计算上限
TABLE_WS = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]
THETA = {3: 10.0, 4: 15.0}
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
    """多光束干涉理论光谱（式 31）。"""
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
    work = df.loc[df["omega"] >= W_ENV].reset_index(drop=True)
    w = work["omega"].to_numpy()
    r_raw = work["R"].to_numpy()

    r_filt = centered_moving_average(r_raw, 101)
    extrema = cubic_spline_extrema(w, r_filt, prominence_threshold=PROM)
    x_env, rmax, rmin = extract_envelope(w, extrema)

    A2, B2, n2, ns2 = invert_refractive_index(rmax, rmin)
    Am, Bm, nm, nsm = invert_refractive_index_multibeam(rmax, rmin)
    n2_spl = CubicSpline(x_env, n2)
    nm_spl = CubicSpline(x_env, nm)
    A2_spl = CubicSpline(x_env, A2)
    B2_spl = CubicSpline(x_env, B2)
    Am_spl = CubicSpline(x_env, Am)
    Bm_spl = CubicSpline(x_env, Bm)

    # 必要条件判定：500-1500 cm-1 内的折射率比值与峰谷幅度
    m_low = (x_env >= 500.0) & (x_env <= 1500.0)
    ratio2 = np.maximum(ns2 / n2, n2 / ns2)
    ratio_low_max = float(ratio2[m_low].max()) if m_low.any() else np.nan
    ratio_low_arg = (
        float(x_env[m_low][np.argmax(ratio2[m_low])]) if m_low.any() else np.nan
    )
    swing_raw = float(
        r_raw[(w >= 500.0) & (w <= 1500.0)].max()
        - r_raw[(w >= 500.0) & (w <= 1500.0)].min()
    )
    swing_env = float((rmax - rmin)[m_low].max()) if m_low.any() else np.nan

    # 厚度：1000-3800 cm-1 内全部峰谷组合（排除 N=0.5）
    ext_thick = extrema.loc[
        (extrema["omega"] >= W_MIN) & (extrema["omega"] <= THICK_HI)
    ].reset_index(drop=True)
    pairs2 = thickness_pairs(ext_thick, n2_spl, theta1)
    pairsm = thickness_pairs(ext_thick, nm_spl, theta1)
    pairs2_pk = thickness_pairs(
        ext_thick.loc[ext_thick["kind"] == "peak"].reset_index(drop=True),
        n2_spl,
        theta1,
    )
    pairsm_pk = thickness_pairs(
        ext_thick.loc[ext_thick["kind"] == "peak"].reset_index(drop=True),
        nm_spl,
        theta1,
    )

    stats = {}
    for tag, dfp in [
        ("two_peak_trough", pairs2),
        ("two_peak_only", pairs2_pk),
        ("multi_peak_trough", pairsm),
        ("multi_peak_only", pairsm_pk),
    ]:
        stats[tag] = {
            "n_pairs": len(dfp),
            "mean_um": float(dfp["d_um"].mean()),
            "std_um": float(dfp["d_um"].std(ddof=1)),
        }

    # 多光束理论光谱拟合（可靠性）
    d_final_cm = stats["multi_peak_trough"]["mean_um"] * 1e-4
    fit_mask = (
        (w >= FIT_LO)
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
        "extrema": extrema,
        "ext_thick": ext_thick,
        "x_env": x_env,
        "rmax": rmax,
        "rmin": rmin,
        "n2": n2,
        "nm": nm,
        "n2_spl": n2_spl,
        "nm_spl": nm_spl,
        "A2_spl": A2_spl,
        "B2_spl": B2_spl,
        "Am_spl": Am_spl,
        "Bm_spl": Bm_spl,
        "n2_table": n2_spl(np.array(TABLE_WS)),
        "nm_table": nm_spl(np.array(TABLE_WS)),
        "ns2_table": CubicSpline(x_env, ns2)(np.array(TABLE_WS)),
        "nsm_table": CubicSpline(x_env, nsm)(np.array(TABLE_WS)),
        "pairs2": pairs2,
        "pairsm": pairsm,
        "stats": stats,
        "ratio_low_max": ratio_low_max,
        "ratio_low_arg": ratio_low_arg,
        "swing_raw": swing_raw,
        "swing_env": swing_env,
        "phi": phi_fit,
        "rmse": rmse,
        "d_final_cm": d_final_cm,
    }


def make_figures(r3: dict, r4: dict) -> None:
    # 图 1：原始光谱与包络
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, r in zip(axes, (r3, r4)):
        ax.plot(r["w"], r["r_raw"] * 100, lw=0.5, color="k", alpha=0.6, label="实测光谱")
        ax.plot(r["x_env"], r["rmax"] * 100, lw=1.4, color="tab:blue", label="上包络")
        ax.plot(r["x_env"], r["rmin"] * 100, lw=1.4, color="tab:orange", label="下包络")
        ax.set_title(f"附件{r['att']}（{r['theta1']:.0f}°）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("反射率 (%)")
    fig.suptitle("硅数据的反射光谱与包络（多光束干涉区域 500–1500 cm-1）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_si_spectrum.png", dpi=150)
    plt.close(fig)

    # 图 2：两种模型的折射率
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, r in zip(axes, (r3, r4)):
        ax.plot(r["x_env"], r["n2"], lw=1.2, label="双光束模型")
        ax.plot(r["x_env"], r["nm"], lw=1.2, label="多光束模型")
        ax.set_title(f"附件{r['att']}（{r['theta1']:.0f}°）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("折射率 n")
    fig.suptitle("硅外延层折射率：双光束与多光束模型对比")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_si_refractive_index.png", dpi=150)
    plt.close(fig)

    # 图 3：多光束理论光谱与实测对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, r in zip(axes, (r3, r4)):
        m = (r["w"] >= FIT_LO) & (r["w"] <= THICK_HI)
        r_the = theoretical_multibeam(
            r["w"][m], r["d_final_cm"], r["nm_spl"], r["Am_spl"], r["Bm_spl"], r["theta1"], r["phi"]
        )
        ax.plot(r["w"][m], r["r_filt"][m] * 100, lw=0.5, color="k", alpha=0.7, label="实测（滤波）")
        ax.plot(r["w"][m], r_the * 100, lw=1.1, color="tab:red", label="多光束理论")
        ax.set_title(f"附件{r['att']}（d={r['stats']['multi_peak_trough']['mean_um']:.3f} μm）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("反射率 (%)")
    fig.suptitle("硅数据：实测与多光束理论光谱对比（1500–3800 cm-1）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_si_theory_compare.png", dpi=150)
    plt.close(fig)


def save_results(r3: dict, r4: dict) -> None:
    n_rows = []
    for r in (r3, r4):
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
        RES_DIR / "q3_si_refractive_index.csv", index=False, encoding="utf-8-sig"
    )

    for r in (r3, r4):
        r["pairs2"].to_csv(
            RES_DIR / f"q3_att{r['att']}_thickness_twobeam.csv",
            index=False,
            encoding="utf-8-sig",
        )
        r["pairsm"].to_csv(
            RES_DIR / f"q3_att{r['att']}_thickness_multibeam.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = []
    for r in (r3, r4):
        for tag, label in [
            ("two_peak_trough", "双光束/峰+谷"),
            ("two_peak_only", "双光束/仅波峰"),
            ("multi_peak_trough", "多光束/峰+谷"),
            ("multi_peak_only", "多光束/仅波峰"),
        ]:
            st = r["stats"][tag]
            summary.append(
                {
                    "附件": f"附件{r['att']}",
                    "入射角": f"{r['theta1']:.0f}°",
                    "口径": label,
                    "组合数": st["n_pairs"],
                    "厚度均值_um": st["mean_um"],
                    "标准差_um": st["std_um"],
                    "判定_500_1500最大比值": r["ratio_low_max"],
                    "判定_峰值比值波数": r["ratio_low_arg"],
                    "500_1500原始峰谷差_%": r["swing_raw"] * 100,
                    "500_1500包络峰谷差_%": r["swing_env"] * 100,
                    "理论光谱RMSE_%": r["rmse"] * 100,
                }
            )
    pd.DataFrame(summary).to_csv(
        RES_DIR / "q3_si_summary.csv", index=False, encoding="utf-8-sig"
    )


def combine(results: list[dict]) -> dict:
    """逆方差加权合并同一晶圆片两组入射角的结果，正态近似 95% 置信区间。"""
    w = np.array([1.0 / r["stats"]["multi_peak_trough"]["std_um"] ** 2 for r in results])
    x = np.array([r["stats"]["multi_peak_trough"]["mean_um"] for r in results])
    mean = float(np.sum(w * x) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    half = 1.96 * se
    return {"mean_um": mean, "se_um": se, "ci_low": mean - half, "ci_high": mean + half}


def print_summary(r3: dict, r4: dict, comb: dict) -> None:
    print("=" * 66)
    for r in (r3, r4):
        print(f"附件{r['att']}（入射角 {r['theta1']:.0f}°）")
        print(f"  判定：500-1500 cm-1 内最大折射率比值 {r['ratio_low_max']:.2f}"
              f"（@ {r['ratio_low_arg']:.0f} cm-1），原始峰谷差 {r['swing_raw']*100:.1f}%")
        print(f"  双光束：峰+谷 {r['stats']['two_peak_trough']['mean_um']:.3f} ± "
              f"{r['stats']['two_peak_trough']['std_um']:.4f} μm；仅波峰 "
              f"{r['stats']['two_peak_only']['mean_um']:.3f} ± {r['stats']['two_peak_only']['std_um']:.4f} μm")
        print(f"  多光束：峰+谷 {r['stats']['multi_peak_trough']['mean_um']:.3f} ± "
              f"{r['stats']['multi_peak_trough']['std_um']:.4f} μm；仅波峰 "
              f"{r['stats']['multi_peak_only']['mean_um']:.3f} ± {r['stats']['multi_peak_only']['std_um']:.4f} μm")
        print(f"  理论光谱 RMSE={r['rmse']*100:.3f}%（反射率百分点）")
        print("  折射率（1500~3500 cm-1）双光束：", " ".join(f"{v:.2f}" for v in r["n2_table"]))
        print("  折射率（1500~3500 cm-1）多光束：", " ".join(f"{v:.2f}" for v in r["nm_table"]))
    print(f"硅最终合并（逆方差加权）：{comb['mean_um']:.3f} μm，"
          f"SE={comb['se_um']:.4f}，95% CI [{comb['ci_low']:.3f}, {comb['ci_high']:.3f}]")
    print("=" * 66)


def main() -> None:
    r3 = process_attachment(3)
    r4 = process_attachment(4)
    make_figures(r3, r4)
    save_results(r3, r4)
    comb = combine([r3, r4])
    print_summary(r3, r4, comb)
    print(f"\n图表已保存到 {FIG_DIR}")
    print(f"数值结果已保存到 {RES_DIR}")


if __name__ == "__main__":
    sys.exit(main())
