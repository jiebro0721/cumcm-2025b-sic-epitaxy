"""改进三：移动块自助法（MBB）给出诚实的置信区间。

两两组合的厚度估计共享极值点，样本并不独立，朴素“均值 ± 1.96·std/√n”
不是合法的置信区间；线性回归的残差也存在序列相关。本程序采用标准做法：
对式 (38)(39) 回归的残差按连续块重抽（moving block bootstrap），
重拟合斜率得到厚度的经验分布，给出 2.5%–97.5% 百分位区间，
并与朴素（独立假设）区间对比。
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _helpers import ROOT, load_pipeline, regression_thickness


RES_DIR = ROOT / "explore" / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

B = 2000
BLOCK_LENS = (2, 4, 6)
RNG = np.random.default_rng(20260811)


def regression_arrays(ext_thick, n_spl, theta1_deg):
    """返回回归所需的 x、y、拟合值、残差与 OLS 结果。"""
    w = ext_thick["omega"].to_numpy()
    n = n_spl(w)
    s2 = np.sin(np.radians(theta1_deg)) ** 2
    x = 2.0 * w * np.sqrt(n**2 - s2)
    y = ext_thick["index"].to_numpy() / 2.0
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = len(y) - 2
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return {
        "x": x,
        "y": y,
        "fitted": fitted,
        "resid": resid,
        "d_um": beta[0] * 1e4,
        "se_um": np.sqrt(cov[0, 0]) * 1e4,
        "n": len(y),
    }


def residual_mbb(
    arr: dict, block_len: int, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """残差移动块自助：重抽残差块并加到拟合值上，重拟合斜率。"""
    n = arr["n"]
    nblocks = int(np.ceil(n / block_len))
    X = np.column_stack([arr["x"], np.ones_like(arr["x"])])
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=nblocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        y_star = arr["fitted"] + arr["resid"][idx]
        beta_star, *_ = np.linalg.lstsq(X, y_star, rcond=None)
        slopes[b] = beta_star[0] * 1e4
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(slopes.std(ddof=1)), float(lo), float(hi)


def main() -> None:
    rows = []
    print("=" * 100)
    print("改进三：残差移动块自助法（MBB）置信区间（B=2000，块长 2/4/6）")
    print("=" * 100)
    for att in (1, 2, 3, 4):
        r = load_pipeline(att)
        theta1 = r["cfg"]["theta1"]
        n_spl = r["n_spl"]
        ext = r["ext_thick"]
        arr = regression_arrays(ext, n_spl, theta1)
        pair_mean = r["pairs"]["d_um"].mean()
        pair_std = r["pairs"]["d_um"].std(ddof=1)
        reg = regression_thickness(ext, n_spl, theta1)

        print(f"\n附件{att}（{theta1:.0f}°，极值 {len(ext)} 个）")
        print(f"  组合法：均值 {pair_mean:.3f} μm，标准差 {pair_std:.4f} μm"
              f"（共享极值点，不能作为标准误）")
        print(f"  回归法：d={reg['d_um']:.3f} μm，OLS SE={reg['se_um']:.4f} μm"
              f"（假设残差独立，偏乐观）")
        for L in BLOCK_LENS:
            bsd, blo, bhi = residual_mbb(arr, L, B, RNG)
            naive_lo = reg["d_um"] - 1.96 * reg["se_um"]
            naive_hi = reg["d_um"] + 1.96 * reg["se_um"]
            rows.append(
                {
                    "附件": att,
                    "块长": L,
                    "组合法均值_um": pair_mean,
                    "组合法标准差_um": pair_std,
                    "回归斜率_um": reg["d_um"],
                    "OLS_SE_um": reg["se_um"],
                    "MBB_SD_um": bsd,
                    "MBB_CI低": blo,
                    "MBB_CI高": bhi,
                    "朴素CI低": naive_lo,
                    "朴素CI高": naive_hi,
                    "CI半宽比(MBB/朴素)": (bhi - blo) / (naive_hi - naive_lo),
                }
            )
            print(
                f"  块长 {L}: MBB CI [{blo:.3f}, {bhi:.3f}]（SD {bsd:.4f}）"
                f" vs 朴素 [{naive_lo:.3f}, {naive_hi:.3f}]，"
                f"半宽比 {(bhi-blo)/(naive_hi-naive_lo):.2f}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "imp3_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {RES_DIR / 'imp3_bootstrap.csv'}")


if __name__ == "__main__":
    sys.exit(main())
