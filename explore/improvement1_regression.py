"""改进一：用厚度线性回归代替两两组合取平均。

对每个附件，把全部峰谷极值点代入式 (38)(39)：
    index/2 = d·[2ω√(n²(ω)-sin²θ)] + c，
OLS 斜率即厚度 d，并给出标准误、R² 与残差标准差。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from _helpers import ROOT, load_pipeline, regression_thickness


RES_DIR = ROOT / "explore" / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = []
    print("=" * 78)
    print("改进一：厚度线性回归（斜率 = 厚度） vs 两两组合取平均")
    print("=" * 78)
    for att in (1, 2, 3, 4):
        r = load_pipeline(att)
        reg = regression_thickness(r["ext_thick"], r["n_spl"], r["cfg"]["theta1"])
        pair_mean = r["pairs"]["d_um"].mean()
        pair_std = r["pairs"]["d_um"].std(ddof=1)
        rows.append(
            {
                "附件": att,
                "入射角": r["cfg"]["theta1"],
                "极值点数": reg["n"],
                "回归厚度_um": reg["d_um"],
                "回归SE_um": reg["se_um"],
                "回归R2": reg["r2"],
                "回归残差SD_半级次": reg["resid_sd"],
                "组合均值_um": pair_mean,
                "组合标准差_um": pair_std,
                "两法差_um": reg["d_um"] - pair_mean,
            }
        )
        print(
            f"附件{att}（{r['cfg']['theta1']:.0f}°）: "
            f"回归 d={reg['d_um']:.3f}±{reg['se_um']:.4f} μm (R²={reg['r2']:.4f}, "
            f"残差SD={reg['resid_sd']:.4f} 半级次) | "
            f"组合法 {pair_mean:.3f}±{pair_std:.4f} μm | 差 {reg['d_um']-pair_mean:+.4f} μm"
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "imp1_regression.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {RES_DIR / 'imp1_regression.csv'}")


if __name__ == "__main__":
    sys.exit(main())
