"""改进四（综合流程）：同时采用改进一+二+三。

流水线：
1) 包络反演 n(ω)、ns(ω)（复用主项目流程）；
2) 拟合数据驱动柯西色散 n(λ)=a+b/λ²+c/λ⁴（改进二阶段 B）；
3) 在色散平滑折射率上做厚度线性回归（改进一），得 d_reg；
4) 以 d_reg 为初值做全谱多光束 NLS（改进二阶段 C2），得 d_nls；
5) 对回归残差做移动块自助（改进三），得 d_reg 的诚实置信区间；
6) 对 NLS 残差做线性化移动块自助，得 d_nls 的诚实置信区间；
7) 与 main 分支主流程（组合法）逐附件对比并逆方差合并。
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from _helpers import ROOT, load_pipeline, thickness_pairs
from improvement2_nls import cauchy_eval, cauchy_fit, forward_multi
from improvement3_bootstrap import residual_mbb, regression_arrays


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

RES_DIR = ROOT / "explore" / "results"
FIG_DIR = ROOT / "explore" / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 综合流程统一使用稳定区间：SiC 2000-3800，硅 1500-3800
RANGE_MAIN = {1: (2000.0, 3800.0), 2: (2000.0, 3800.0), 3: (1500.0, 3800.0), 4: (1500.0, 3800.0)}
B = 2000
BLOCK_LENS = (2, 4, 6)
RNG = np.random.default_rng(20260812)


def run_combined(att: int) -> dict:
    r = load_pipeline(att)
    theta1 = r["cfg"]["theta1"]
    lo, hi = RANGE_MAIN[att]

    # 步骤 2：在拟合区间上拟合色散参数
    emask = (r["x_env"] >= lo) & (r["x_env"] <= hi)
    npar = cauchy_fit(r["x_env"][emask], r["n"][emask])
    nspar = cauchy_fit(r["x_env"][emask], r["ns"][emask])

    # 综合流程的极值区间（与色散拟合区间一致）
    ext = r["extrema"].loc[
        (r["extrema"]["omega"] >= lo) & (r["extrema"]["omega"] <= hi)
    ].reset_index(drop=True)

    # 步骤 3：色散平滑折射率下的厚度回归
    w_ext = ext["omega"].to_numpy()
    n_disp = cauchy_eval(w_ext, npar)
    s2 = np.sin(np.radians(theta1)) ** 2
    x = 2.0 * w_ext * np.sqrt(n_disp**2 - s2)
    y = ext["index"].to_numpy() / 2.0
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted_reg = X @ beta
    resid_reg = y - fitted_reg
    dof = len(y) - 2
    sigma2 = resid_reg @ resid_reg / dof
    cov_reg = sigma2 * np.linalg.inv(X.T @ X)
    d_reg = beta[0] * 1e4
    se_reg = np.sqrt(cov_reg[0, 0]) * 1e4

    # 步骤 4：全谱 NLS（自由色散），初值来自回归与色散拟合
    mask = (r["w"] >= lo) & (r["w"] <= hi)
    w = r["w"][mask]
    y_obs = r["r_filt"][mask]

    def resid_fn(theta):
        d_cm, phi = theta[0], theta[1]
        np_ = theta[2:5]
        nsp_ = theta[5:8]
        return forward_multi(w, d_cm, theta1, phi, np_, nsp_) - y_obs

    lb = np.array([1e-5, 0.0, 0.5, -500.0, -500.0, 0.5, -500.0, -500.0])
    ub = np.array([1e-2, 2.0 * np.pi, 6.0, 500.0, 500.0, 6.0, 500.0, 500.0])
    best = None
    for phi0 in (0.0, np.pi / 2.0, np.pi):
        x0 = np.array([d_reg * 1e-4, phi0] + list(npar) + list(nspar))
        x0 = np.clip(x0, lb + 1e-12, ub - 1e-12)
        res_try = least_squares(resid_fn, x0, bounds=(lb, ub), max_nfev=5000)
        if best is None or res_try.cost < best.cost:
            best = res_try
    res = best
    d_nls = res.x[0] * 1e4
    m = len(w)
    p = len(res.x)
    sigma2_nls = 2.0 * res.cost / (m - p)
    jac = res.jac
    cov_nls = sigma2_nls * np.linalg.inv(jac.T @ jac)
    se_nls = np.sqrt(cov_nls[0, 0]) * 1e4
    rmse = np.sqrt(2.0 * res.cost / m) * 100.0
    fitted_nls = y_obs + res.fun

    # 步骤 5：回归残差 MBB
    arr = {
        "x": x,
        "y": y,
        "fitted": fitted_reg,
        "resid": resid_reg,
        "n": len(y),
    }
    mbb_reg = {}
    for L in BLOCK_LENS:
        sd, lo_b, hi_b = residual_mbb(arr, L, B, RNG)
        mbb_reg[L] = (sd, lo_b, hi_b)

    # 步骤 6：NLS 线性化残差 MBB（一阶高斯-牛顿步）
    resid_nls = -res.fun
    resid_nls = resid_nls - resid_nls.mean()
    n_pts = len(y_obs)
    JTJ = jac.T @ jac
    JTJ_inv = np.linalg.inv(JTJ)
    mbb_nls = {}
    for L in BLOCK_LENS:
        nblocks = int(np.ceil(n_pts / L))
        slopes = np.empty(B)
        for b in range(B):
            starts = RNG.integers(0, n_pts - L + 1, size=nblocks)
            idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n_pts]
            delta = JTJ_inv @ (jac.T @ resid_nls[idx])
            slopes[b] = (res.x[0] + delta[0]) * 1e4
        lo_, hi_ = np.percentile(slopes, [2.5, 97.5])
        mbb_nls[L] = (float(slopes.std(ddof=1)), float(lo_), float(hi_))

    # main 主流程同区间对照（组合法，包络反演折射率）
    pairs_same = thickness_pairs(ext, r["n_spl"], theta1)

    return {
        "att": att,
        "theta1": theta1,
        "range": f"{lo:.0f}-{hi:.0f}",
        "ext_n": len(ext),
        "pairs_same": pairs_same,
        "d_reg": d_reg,
        "se_reg": se_reg,
        "mbb_reg": mbb_reg,
        "d_nls": d_nls,
        "se_nls": se_nls,
        "rmse": rmse,
        "mbb_nls": mbb_nls,
        "phi": res.x[1],
        "npar": res.x[2:5],
        "nspar": res.x[5:8],
    }


def main_combined(reference: dict) -> dict:
    """逆方差合并（用块长 4 的 MBB 标准差）。"""
    w = np.array([1.0 / reference[a]["mbb_nls"][4][0] ** 2 for a in (1, 2)])
    x = np.array([reference[a]["d_nls"] for a in (1, 2)])
    mean = float(np.sum(w * x) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return {"mean_um": mean, "se_um": se, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}


def make_figure(ref: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    for ax, att in zip(axes.ravel(), (1, 2, 3, 4)):
        r = ref[att]
        pair_mean = r["pairs_same"]["d_um"].mean()
        pair_std = r["pairs_same"]["d_um"].std(ddof=1)
        methods = ["主流程组合法", "回归+MBB", "NLS+MBB"]
        vals = [pair_mean, r["d_reg"], r["d_nls"]]
        errs = [
            [pair_std, pair_std],
            [r["d_reg"] - r["mbb_reg"][4][1], r["mbb_reg"][4][2] - r["d_reg"]],
            [r["d_nls"] - r["mbb_nls"][4][1], r["mbb_nls"][4][2] - r["d_nls"]],
        ]
        ax.errorbar(methods, vals, yerr=np.array(errs).T, fmt="o", capsize=4, ms=6)
        ax.set_title(f"附件{att}（{r['theta1']:.0f}°），区间 {r['range']} cm-1")
        ax.set_ylabel("厚度 (μm)")
        ax.grid(alpha=0.3)
    fig.suptitle("综合流程 vs 主流程：厚度估计与置信区间（主流程为±标准差，分支为MBB 95% CI）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imp4_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ref = {}
    rows = []
    print("=" * 104)
    print("改进四：综合流程（色散平滑回归 → 全谱NLS → 残差MBB）")
    print("=" * 104)
    for att in (1, 2, 3, 4):
        r = run_combined(att)
        ref[att] = r
        ps = r["pairs_same"]
        rows.append(
            {
                "附件": att,
                "入射角": r["theta1"],
                "区间": r["range"],
                "极值数": r["ext_n"],
                "主流程均值_um": ps["d_um"].mean(),
                "主流程标准差_um": ps["d_um"].std(ddof=1),
                "回归d_um": r["d_reg"],
                "回归OLS_SE_um": r["se_reg"],
                "回归MBB_CI低": r["mbb_reg"][4][1],
                "回归MBB_CI高": r["mbb_reg"][4][2],
                "NLS_d_um": r["d_nls"],
                "NLS_雅可比SE_um": r["se_nls"],
                "NLS_RMSE_%": r["rmse"],
                "NLS_MBB_CI低": r["mbb_nls"][4][1],
                "NLS_MBB_CI高": r["mbb_nls"][4][2],
            }
        )
        print(
            f"附件{att} [{r['range']} cm-1]（{r['theta1']:.0f}°）: "
            f"主流程 {ps['d_um'].mean():.3f}±{ps['d_um'].std(ddof=1):.4f} | "
            f"回归 {r['d_reg']:.3f} [MBB {r['mbb_reg'][4][1]:.3f},{r['mbb_reg'][4][2]:.3f}] | "
            f"NLS {r['d_nls']:.3f} [MBB {r['mbb_nls'][4][1]:.3f},{r['mbb_nls'][4][2]:.3f}], RMSE={r['rmse']:.3f}%"
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "imp4_combined.csv", index=False, encoding="utf-8-sig")

    comb_sic = main_combined(ref)
    comb_si = main_combined(ref)
    # 硅合并用附件 3、4
    w_si = np.array([1.0 / ref[a]["mbb_nls"][4][0] ** 2 for a in (3, 4)])
    x_si = np.array([ref[a]["d_nls"] for a in (3, 4)])
    mean_si = float(np.sum(w_si * x_si) / np.sum(w_si))
    se_si = float(np.sqrt(1.0 / np.sum(w_si)))
    print(
        f"\n综合流程合并（NLS + MBB 标准差，逆方差加权）：\n"
        f"  碳化硅 {comb_sic['mean_um']:.3f} μm, 95% CI [{comb_sic['ci_low']:.3f}, {comb_sic['ci_high']:.3f}]\n"
        f"  硅     {mean_si:.3f} μm, 95% CI [{mean_si-1.96*se_si:.3f}, {mean_si+1.96*se_si:.3f}]"
    )
    make_figure(ref)
    print(f"\n结果: {RES_DIR / 'imp4_combined.csv'}")
    print(f"图形: {FIG_DIR / 'imp4_comparison.png'}")


if __name__ == "__main__":
    sys.exit(main())
