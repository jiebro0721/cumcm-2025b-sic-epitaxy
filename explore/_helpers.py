"""探索分支公共工具：复用主项目 src/common.py 的处理流程。

每个附件统一返回：滤波光谱、极值点、包络、多光束反演折射率曲线、
厚度组合结果等，供三个改进实验共用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    centered_moving_average,
    cubic_spline_extrema,
    extract_envelope,
    invert_refractive_index_multibeam,
    load_spectrum,
    thickness_pairs,
)


# 与主项目完全一致的配置：SiC 包络从 1000 起、厚度区间 2000-3800；
# Si 包络从 500 起（避免低波数外推）、厚度区间 1000-3800。
CONFIG = {
    1: dict(theta1=10.0, w_env=1000.0, lo=2000.0, hi=3800.0),
    2: dict(theta1=15.0, w_env=1000.0, lo=2000.0, hi=3800.0),
    3: dict(theta1=10.0, w_env=500.0, lo=1000.0, hi=3800.0),
    4: dict(theta1=15.0, w_env=500.0, lo=1000.0, hi=3800.0),
}

PROM = 5e-4  # 峰谷显著性阈值（与主项目一致）


def load_pipeline(att: int) -> dict:
    """运行主项目同款处理流程，返回各阶段结果。"""
    cfg = CONFIG[att]
    df = load_spectrum(att)
    work = df.loc[df["omega"] >= cfg["w_env"]].reset_index(drop=True)
    w = work["omega"].to_numpy()
    r_raw = work["R"].to_numpy()
    r_filt = centered_moving_average(r_raw, 101)
    extrema = cubic_spline_extrema(w, r_filt, prominence_threshold=PROM)
    x_env, rmax, rmin = extract_envelope(w, extrema)
    A, B, n, ns = invert_refractive_index_multibeam(rmax, rmin)
    n_spl = CubicSpline(x_env, n)
    ns_spl = CubicSpline(x_env, ns)

    ext_thick = extrema.loc[
        (extrema["omega"] >= cfg["lo"]) & (extrema["omega"] <= cfg["hi"])
    ].reset_index(drop=True)
    pairs = thickness_pairs(ext_thick, n_spl, cfg["theta1"])

    return {
        "att": att,
        "cfg": cfg,
        "w": w,
        "r_raw": r_raw,
        "r_filt": r_filt,
        "extrema": extrema,
        "ext_thick": ext_thick,
        "x_env": x_env,
        "rmax": rmax,
        "rmin": rmin,
        "n": n,
        "ns": ns,
        "n_spl": n_spl,
        "ns_spl": ns_spl,
        "pairs": pairs,
    }


def regression_thickness(
    ext_thick, n_spl: CubicSpline, theta1_deg: float
) -> dict:
    """改进一：厚度线性回归（讲评式 (38)(39)）。

    m_i = 2d·ω_i·√(n²(ω_i)-sin²θ1) + 1/2，令 m_i = m0 + index/2，
    则 index/2 = d·[2ω_i√(n²(ω_i)-sin²θ1)] + (1/2 - m0)。
    因此 y = index/2 对 x = 2ω√(n²-sin²θ) 回归，斜率即厚度 d。
    """
    w = ext_thick["omega"].to_numpy()
    n = n_spl(w)
    s2 = np.sin(np.radians(theta1_deg)) ** 2
    x = 2.0 * w * np.sqrt(n**2 - s2)
    y = ext_thick["index"].to_numpy() / 2.0
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[0, 0])
    r2 = 1.0 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return {
        "d_um": beta[0] * 1e4,
        "se_um": se * 1e4,
        "intercept": beta[1],
        "r2": r2,
        "resid_sd": np.sqrt(sigma2),  # y 单位（半个干涉级次），无量纲
        "n": len(y),
    }


def block_resample(ext_thick, block_len: int, rng: np.random.Generator):
    """移动块自助法：按块重抽极值序列并保持顺序。"""
    E = len(ext_thick)
    nblocks = int(np.ceil(E / block_len))
    starts = rng.integers(0, E - block_len + 1, size=nblocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:E]
    out = ext_thick.iloc[idx].reset_index(drop=True)
    out["index"] = np.arange(E)  # 重抽后顺序号必须重算
    return out
