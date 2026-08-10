"""改进二：全谱非线性最小二乘 + 数据驱动色散模型。

思路：
1) 用包络反演的 n(ω)、ns(ω) 在拟合区间上拟合柯西色散模型
   n(λ)=a+b/λ²+c/λ⁴（参数由附件数据估计，而非文献值）；
2) 以多光束干涉模型重构理论光谱，先固定色散参数拟合 (d, φ)；
3) 再联合拟合 (d, φ, 色散参数)，用最小二乘雅可比给出 d 的标准误。
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from _helpers import ROOT, load_pipeline


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

RES_DIR = ROOT / "explore" / "results"
FIG_DIR = ROOT / "explore" / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 每个附件的主拟合区间；硅另试全区间 1000-3800 作对照
RANGES = {
    1: [(2000.0, 3800.0)],
    2: [(2000.0, 3800.0)],
    3: [(1500.0, 3800.0), (1000.0, 3800.0)],
    4: [(1500.0, 3800.0), (1000.0, 3800.0)],
}


def cauchy_fit(w: np.ndarray, values: np.ndarray) -> np.ndarray:
    """柯西色散最小二乘拟合：n(λ)=a+b/λ²+c/λ⁴，λ 单位 μm。"""
    lam = 1e4 / w
    u = 1.0 / lam**2
    X = np.column_stack([np.ones_like(u), u, u**2])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    return beta


def cauchy_eval(w: np.ndarray, beta: np.ndarray) -> np.ndarray:
    lam = 1e4 / w
    u = 1.0 / lam**2
    return beta[0] + beta[1] * u + beta[2] * u**2


def forward_multi(
    w: np.ndarray,
    d_cm: float,
    theta1_deg: float,
    phi: float,
    npar: np.ndarray,
    nspar: np.ndarray,
) -> np.ndarray:
    """多光束干涉理论光谱（色散模型版）。"""
    n = cauchy_eval(w, npar)
    ns = cauchy_eval(w, nspar)
    I1 = ((1.0 - n) / (1.0 + n)) ** 2
    I2 = (1.0 - I1) ** 2 * ((ns - n) / (ns + n)) ** 2
    s2 = np.sin(np.radians(theta1_deg)) ** 2
    delta = 2.0 * d_cm * np.sqrt(n**2 - s2)
    cosd = np.cos(2.0 * np.pi * delta * w + phi)
    x = np.sqrt(I1 * I2)
    return (I1 + I2 + 2.0 * x * cosd) / (1.0 + I1 * I2 + 2.0 * x * cosd)


def nls_fit(
    att: int,
    lo: float,
    hi: float,
    free_dispersion: bool,
) -> dict:
    r = load_pipeline(att)
    theta1 = r["cfg"]["theta1"]
    mask = (r["w"] >= lo) & (r["w"] <= hi)
    w = r["w"][mask]
    y_obs = r["r_filt"][mask]
    emask = (r["x_env"] >= lo) & (r["x_env"] <= hi)

    # 阶段 B：由包络反演曲线拟合色散参数
    npar0 = cauchy_fit(r["x_env"][emask], r["n"][emask])
    nspar0 = cauchy_fit(r["x_env"][emask], r["ns"][emask])

    # 初值：d 取组合法均值；φ 多起点
    d0_cm = r["pairs"]["d_um"].mean() * 1e-4
    best = None
    for phi0 in (0.0, np.pi / 2.0, np.pi):
        x0 = [d0_cm, phi0]
        lb = [1e-5, 0.0]
        ub = [1e-2, 2.0 * np.pi]
        if free_dispersion:
            x0 = x0 + list(npar0) + list(nspar0)
            lb = lb + [0.5, -500.0, -500.0] * 2
            ub = ub + [6.0, 500.0, 500.0] * 2

        x0 = np.clip(x0, np.asarray(lb) + 1e-12, np.asarray(ub) - 1e-12)

        def resid(x):
            d, phi = x[0], x[1]
            if free_dispersion:
                npar = np.array(x[2:5])
                nspar = np.array(x[5:8])
            else:
                npar, nspar = npar0, nspar0
            return forward_multi(w, d, theta1, phi, npar, nspar) - y_obs

        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=5000)
        if best is None or res.cost < best["cost"]:
            best = {"cost": res.cost, "res": res, "npar": npar0, "nspar": nspar0}

    res = best["res"]
    d_um = res.x[0] * 1e4
    m = len(w)
    p = len(res.x)
    sigma2 = 2.0 * res.cost / (m - p)
    jac = res.jac
    cov = sigma2 * np.linalg.inv(jac.T @ jac)
    se_um = np.sqrt(cov[0, 0]) * 1e4
    rmse = np.sqrt(2.0 * res.cost / m) * 100.0  # 反射率百分点
    return {
        "att": att,
        "range": f"{lo:.0f}-{hi:.0f}",
        "free_dispersion": free_dispersion,
        "d_um": d_um,
        "se_um": se_um,
        "rmse_pct": rmse,
        "n_points": m,
        "n_params": p,
        "d0_um": d0_cm * 1e4,
        "phi": res.x[1],
        "npar": np.array(res.x[2:5]) if free_dispersion else best["npar"],
        "nspar": np.array(res.x[5:8]) if free_dispersion else best["nspar"],
    }


def combine(results: list[dict]) -> dict:
    w = np.array([1.0 / r["se_um"] ** 2 for r in results])
    x = np.array([r["d_um"] for r in results])
    mean = float(np.sum(w * x) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return {"mean_um": mean, "se_um": se, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}


def make_figure(fits: dict[int, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, att in zip(axes.ravel(), (1, 2, 3, 4)):
        r = load_pipeline(att)
        f = fits[att]
        lo, hi = f["range"].split("-")
        m = (r["w"] >= float(lo)) & (r["w"] <= float(hi))
        r_the = forward_multi(
            r["w"][m], f["d_um"] * 1e-4, r["cfg"]["theta1"], f["phi"], f["npar"], f["nspar"]
        )
        ax.plot(r["w"][m], r["r_filt"][m] * 100, lw=0.5, color="k", alpha=0.7, label="实测（滤波）")
        ax.plot(r["w"][m], r_the * 100, lw=1.0, color="tab:red", label="NLS拟合")
        ax.set_title(f"附件{att}（{r['cfg']['theta1']:.0f}°）d={f['d_um']:.3f}±{f['se_um']:.4f} μm")
        ax.set_xlabel("波数 (cm-1)")
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("反射率 (%)")
    axes[1, 0].set_ylabel("反射率 (%)")
    fig.suptitle("改进二：全谱非线性最小二乘拟合（多光束模型+柯西色散）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imp2_nls_fit.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows = []
    fits = {}
    print("=" * 92)
    print("改进二：全谱 NLS + 数据驱动柯西色散（多光束模型）")
    print("=" * 92)
    for att in (1, 2, 3, 4):
        fits[att] = None
        for lo, hi in RANGES[att]:
            for free_disp in (False, True):
                f = nls_fit(att, lo, hi, free_disp)
                rows.append(
                    {
                        "附件": att,
                        "拟合区间": f["range"],
                        "自由色散参数": free_disp,
                        "厚度_um": f["d_um"],
                        "标准误_um": f["se_um"],
                        "RMSE_%": f["rmse_pct"],
                        "点数": f["n_points"],
                        "组合法初值_um": f["d0_um"],
                    }
                )
                tag = "C2(自由色散)" if free_disp else "C1(固定色散)"
                print(
                    f"附件{att} [{f['range']} cm-1] {tag}: "
                    f"d={f['d_um']:.3f}±{f['se_um']:.4f} μm, RMSE={f['rmse_pct']:.3f}%"
                )
                if free_disp and fits[att] is None:
                    fits[att] = f
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "imp2_nls.csv", index=False, encoding="utf-8-sig")

    # 主区间（自由色散）逆方差加权合并
    for att, f in fits.items():
        print(f"  附件{att} 主拟合: {f['range']} cm-1, d={f['d_um']:.3f}±{f['se_um']:.4f} μm")
    comb_sic = combine([fits[1], fits[2]])
    comb_si = combine([fits[3], fits[4]])
    print(f"碳化硅合并: {comb_sic['mean_um']:.3f} μm, 95% CI [{comb_sic['ci_low']:.3f}, {comb_sic['ci_high']:.3f}]")
    print(f"硅合并: {comb_si['mean_um']:.3f} μm, 95% CI [{comb_si['ci_low']:.3f}, {comb_si['ci_high']:.3f}]")
    make_figure(fits)
    print(f"\n结果: {RES_DIR / 'imp2_nls.csv'}")
    print(f"图形: {FIG_DIR / 'imp2_nls_fit.png'}")


if __name__ == "__main__":
    sys.exit(main())
