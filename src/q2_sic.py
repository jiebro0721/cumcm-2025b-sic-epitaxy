"""问题 2：根据问题 1 的模型计算碳化硅外延层厚度（附件 1、附件 2）。

流程：剔除剩余射线带 -> 101 点中心移动平均 -> 三次样条 + 优化求峰谷 ->
包络反演折射率 -> 2000~3800 cm⁻¹ 全部峰/谷组合按式 (11) 计算厚度 ->
统计均值/标准差 -> 可靠性分析（理论光谱对比、两种口径对照）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from common import (
    ROOT,
    centered_moving_average,
    cubic_spline_extrema,
    extract_envelope,
    fit_delta_phi,
    invert_refractive_index,
    load_spectrum,
    theoretical_reflectance,
    thickness_pairs,
)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

W_MIN = 1000.0          # 剔除剩余射线带的波数下限
THICK_LO, THICK_HI = 2000.0, 3800.0  # 厚度计算稳定区间
TABLE_WS = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]
THETA = {1: 10.0, 2: 15.0}


def process_attachment(att: int) -> dict:
    """处理单个附件，返回结果字典。"""
    theta1 = THETA[att]
    df = load_spectrum(att)

    # 1) 剔除剩余射线带，只保留 omega >= 1000 cm⁻¹
    work = df.loc[df["omega"] >= W_MIN].reset_index(drop=True)
    w = work["omega"].to_numpy()
    r_raw = work["R"].to_numpy()

    # 2) 101 点中心移动平均滤波
    r_filt = centered_moving_average(r_raw, 101)

    # 3) 三次样条插值 + 试探区间 + 单变量优化求峰谷
    #    显著性阈值 0.05% 反射率用于剔除残余噪声产生的伪极值
    extrema = cubic_spline_extrema(w, r_filt)

    # 4) 包络提取（全域 omega >= 1000）
    x_env, rmax, rmin = extract_envelope(w, extrema)

    # 5) 包络反演折射率
    A, B, n, ns = invert_refractive_index(rmax, rmin)
    n_spl = CubicSpline(x_env, n)
    ns_spl = CubicSpline(x_env, ns)

    # 6) 厚度计算：稳定区间内全部峰/谷组合（排除 N = 0.5）
    ext_thick = extrema.loc[
        (extrema["omega"] >= THICK_LO) & (extrema["omega"] <= THICK_HI)
    ].reset_index(drop=True)
    pairs = thickness_pairs(ext_thick, n_spl, theta1)
    pairs_peak = thickness_pairs(
        ext_thick.loc[ext_thick["kind"] == "peak"].reset_index(drop=True),
        n_spl,
        theta1,
    )

    stats = {}
    for tag, dfp in [("peak_trough", pairs), ("peak_only", pairs_peak)]:
        stats[tag] = {
            "n_pairs": len(dfp),
            "mean_um": float(dfp["d_um"].mean()),
            "std_um": float(dfp["d_um"].std(ddof=1)),
        }

    # 7) 可靠性：理论光谱对比（d = 最终均值）
    peaks = extrema.loc[extrema["kind"] == "peak"]
    troughs = extrema.loc[extrema["kind"] == "trough"]
    rmax_spl = CubicSpline(peaks["omega"], peaks["value"])
    rmin_spl = CubicSpline(troughs["omega"], troughs["value"])
    d_final_cm = stats["peak_trough"]["mean_um"] * 1e-4

    fit_mask = (w >= THICK_LO) & (w <= THICK_HI) & (w >= x_env.min()) & (w <= x_env.max())
    delta_phi, rmse = fit_delta_phi(
        w[fit_mask], r_filt[fit_mask], d_final_cm, n_spl, theta1, rmax_spl, rmin_spl
    )

    # 折射率表（1500~3500 cm⁻¹）
    n_table = n_spl(np.array(TABLE_WS, dtype=float))

    return {
        "att": att,
        "theta1": theta1,
        "w": w,
        "r_raw": r_raw,
        "r_filt": r_filt,
        "extrema": extrema,
        "x_env": x_env,
        "rmax": rmax,
        "rmin": rmin,
        "n": n,
        "ns": ns,
        "n_spl": n_spl,
        "ns_spl": ns_spl,
        "ext_thick": ext_thick,
        "pairs": pairs,
        "pairs_peak": pairs_peak,
        "stats": stats,
        "delta_phi": delta_phi,
        "rmse": rmse,
        "n_table": n_table,
        "rmax_spl": rmax_spl,
        "rmin_spl": rmin_spl,
        "d_final_cm": d_final_cm,
    }


def make_figures(res1: dict, res2: dict) -> None:
    """生成问题 2 全部插图。"""
    # 图 1：原始光谱与剩余射线带
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in (res1, res2):
        ax.plot(r["w"], r["r_raw"] * 100, lw=0.6, label=f"附件{r['att']}（{r['theta1']:.0f}°）")
    ax.axvspan(400, W_MIN, color="gray", alpha=0.25, label="剩余射线带")
    ax.axvspan(THICK_LO, THICK_HI, color="orange", alpha=0.12, label="厚度计算区间")
    ax.set_xlabel("波数 (cm-1)")
    ax.set_ylabel("反射率 (%)")
    ax.set_title("原始反射光谱")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_raw_spectrum.png", dpi=150)
    plt.close(fig)

    # 图 2：滤波前后对比（局部放大）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, r in zip(axes, (res1, res2)):
        m = (r["w"] >= 1800) & (r["w"] <= 3400)
        ax.plot(r["w"][m], r["r_raw"][m] * 100, lw=0.4, alpha=0.55, label="原始数据")
        ax.plot(r["w"][m], r["r_filt"][m] * 100, lw=1.0, color="crimson", label="101点中心平均")
        ax.set_title(f"附件{r['att']}（{r['theta1']:.0f}°）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("反射率 (%)")
    fig.suptitle("滤波前后对比（1800–3400 cm-1）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_filter_compare.png", dpi=150)
    plt.close(fig)

    # 图 3：包络曲线
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, r in zip(axes, (res1, res2)):
        ax.plot(r["w"], r["r_filt"] * 100, lw=0.5, color="k", alpha=0.6, label="滤波数据")
        ax.plot(r["x_env"], r["rmax"] * 100, lw=1.4, color="tab:blue", label="上包络 Rmax")
        ax.plot(r["x_env"], r["rmin"] * 100, lw=1.4, color="tab:orange", label="下包络 Rmin")
        ax.set_title(f"附件{r['att']}（{r['theta1']:.0f}°）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("反射率 (%)")
    fig.suptitle("干涉光谱的上下包络")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_envelope.png", dpi=150)
    plt.close(fig)

    # 图 4：反演折射率
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in (res1, res2):
        ax.plot(r["x_env"], r["n"], lw=1.2, label=f"附件{r['att']}（{r['theta1']:.0f}°）")
    ax.set_xlabel("波数 (cm-1)")
    ax.set_ylabel("折射率 n")
    ax.set_title("包络反演的外延层折射率")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_refractive_index.png", dpi=150)
    plt.close(fig)

    # 图 5：理论光谱对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, r in zip(axes, (res1, res2)):
        m = (r["w"] >= THICK_LO) & (r["w"] <= THICK_HI)
        r_the = theoretical_reflectance(
            r["w"][m],
            r["d_final_cm"],
            r["n_spl"],
            r["theta1"],
            r["delta_phi"],
            r["rmax_spl"],
            r["rmin_spl"],
        )
        ax.plot(r["w"][m], r["r_filt"][m] * 100, lw=0.6, color="k", alpha=0.7, label="实测（滤波）")
        ax.plot(r["w"][m], r_the * 100, lw=1.1, color="tab:red", label="理论光谱")
        ax.set_title(f"附件{r['att']}（d={r['stats']['peak_trough']['mean_um']:.3f} μm）")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("反射率 (%)")
    fig.suptitle("实测与理论干涉光谱对比（2000–3800 cm-1）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q2_theory_compare.png", dpi=150)
    plt.close(fig)


def save_results(res1: dict, res2: dict) -> None:
    """保存数值结果到 results/ 目录。"""
    # 折射率表
    n_rows = []
    for r in (res1, res2):
        n_rows.append(
            pd.DataFrame(
                {
                    "波数_cm-1": TABLE_WS,
                    f"附件{r['att']}_折射率_n": np.round(r["n_table"], 4),
                    f"附件{r['att']}_衬底折射率_ns": np.round(r["ns_spl"](np.array(TABLE_WS)), 4),
                }
            )
        )
    pd.concat(n_rows, axis=1).to_csv(RES_DIR / "q2_refractive_index.csv", index=False, encoding="utf-8-sig")

    # 厚度组合
    for r in (res1, res2):
        r["pairs"].to_csv(RES_DIR / f"q2_att{r['att']}_thickness_pairs.csv", index=False, encoding="utf-8-sig")
        r["pairs_peak"].to_csv(RES_DIR / f"q2_att{r['att']}_thickness_pairs_peak.csv", index=False, encoding="utf-8-sig")

    # 汇总
    summary = []
    for r in (res1, res2):
        for tag, label in [("peak_trough", "峰+谷"), ("peak_only", "仅波峰")]:
            st = r["stats"][tag]
            summary.append(
                {
                    "附件": f"附件{r['att']}",
                    "入射角": f"{r['theta1']:.0f}°",
                    "口径": label,
                    "组合数": st["n_pairs"],
                    "厚度均值_um": st["mean_um"],
                    "标准差_um": st["std_um"],
                    "delta_phi": r["delta_phi"],
                    "理论光谱RMSE_%": r["rmse"] * 100,
                }
            )
    pd.DataFrame(summary).to_csv(RES_DIR / "q2_summary.csv", index=False, encoding="utf-8-sig")


def print_summary(res1: dict, res2: dict) -> None:
    print("=" * 62)
    for r in (res1, res2):
        print(f"附件{r['att']}（入射角 {r['theta1']:.0f}°）")
        print(f"  波峰+波谷：组合数 {r['stats']['peak_trough']['n_pairs']}，"
              f"厚度 {r['stats']['peak_trough']['mean_um']:.3f} ± {r['stats']['peak_trough']['std_um']:.4f} μm")
        print(f"  仅波峰  ：组合数 {r['stats']['peak_only']['n_pairs']}，"
              f"厚度 {r['stats']['peak_only']['mean_um']:.3f} ± {r['stats']['peak_only']['std_um']:.4f} μm")
        print(f"  理论光谱 delta_phi={r['delta_phi']:.4f}，RMSE={r['rmse'] * 100:.3f}%（反射率百分点）")
        print("  折射率表（1500~3500 cm⁻¹）：")
        print("   " + " ".join(f"{w:.0f}" for w in TABLE_WS))
        print("   " + " ".join(f"{v:.2f}" for v in r["n_table"]))
    print("=" * 62)


def main() -> None:
    res1 = process_attachment(1)
    res2 = process_attachment(2)
    make_figures(res1, res2)
    save_results(res1, res2)
    print_summary(res1, res2)
    print(f"\n图表已保存到 {FIG_DIR}")
    print(f"数值结果已保存到 {RES_DIR}")


if __name__ == "__main__":
    sys.exit(main())
