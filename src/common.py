"""公共工具模块（问题 2、问题 3 共用）。

提供：光谱读取、中心移动平均滤波、三次样条极值定位、包络提取、
折射率反演、条纹组合厚度计算与理论光谱重构。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, peak_prominences


ROOT = Path(__file__).resolve().parent.parent


def load_spectrum(attachment: int) -> pd.DataFrame:
    """读取附件光谱，返回列名 omega（cm⁻¹）与 R（反射率，0~1 小数）。"""
    df = pd.read_excel(ROOT / f"附件{attachment}.xlsx")
    df.columns = ["omega", "R"]
    df["R"] = df["R"] / 100.0
    return df


def centered_moving_average(y: np.ndarray, window: int = 101) -> np.ndarray:
    """window 点中心移动平均；边界处使用较短的对称窗口。"""
    s = pd.Series(np.asarray(y, dtype=float))
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


def cubic_spline_extrema(
    x: np.ndarray,
    y: np.ndarray,
    bracket: int = 5,
    min_distance: int = 50,
    prominence_threshold: float = 5e-4,
) -> pd.DataFrame:
    """三次样条插值 + 试探区间 + 单变量优化，精确求波峰/波谷。

    返回按波数排序的 DataFrame，列为 omega、value、kind（peak/trough）、index。
    index 为极值点在全序列中的顺序号，用于计算条纹间隔数。
    min_distance 与 prominence_threshold 用于在滤波数据上剔除残余噪声产生的
    微小伪极值（阈值为反射率小数，例如 5e-4 对应 0.05%）。
    """
    cs = CubicSpline(x, y)
    pk_idx, _ = find_peaks(
        y, distance=min_distance, prominence=prominence_threshold
    )
    tr_idx, _ = find_peaks(
        -y, distance=min_distance, prominence=prominence_threshold
    )

    records = []
    for i in pk_idx:
        lo, hi = max(0, i - bracket), min(len(x) - 1, i + bracket)
        res = minimize_scalar(lambda w: -cs(w), bounds=(x[lo], x[hi]), method="bounded")
        records.append((float(res.x), float(cs(res.x)), "peak"))
    for i in tr_idx:
        lo, hi = max(0, i - bracket), min(len(x) - 1, i + bracket)
        res = minimize_scalar(cs, bounds=(x[lo], x[hi]), method="bounded")
        records.append((float(res.x), float(cs(res.x)), "trough"))

    df = pd.DataFrame(records, columns=["omega", "value", "kind"])
    df = df.sort_values("omega").reset_index(drop=True)
    df["index"] = np.arange(len(df))
    return df


def extract_envelope(
    x: np.ndarray, extrema: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由波峰/波谷点三次样条插值构造上下包络。

    返回 (x_env, rmax, rmin)，其中 x_env 为上下包络共同的有效波数区间。
    若样条过冲导致局部 rmax <= rmin，该段回退为线性包络并做下限保护。
    """
    peaks = extrema.loc[extrema["kind"] == "peak"]
    troughs = extrema.loc[extrema["kind"] == "trough"]
    if len(peaks) < 3 or len(troughs) < 3:
        raise ValueError("极值点数量不足，无法构造包络。")

    rmax_spl = CubicSpline(peaks["omega"].to_numpy(), peaks["value"].to_numpy())
    rmin_spl = CubicSpline(troughs["omega"].to_numpy(), troughs["value"].to_numpy())

    lo = max(peaks["omega"].min(), troughs["omega"].min())
    hi = min(peaks["omega"].max(), troughs["omega"].max())
    mask = (x >= lo) & (x <= hi)
    x_env = x[mask]
    rmax = rmax_spl(x_env)
    rmin = rmin_spl(x_env)

    bad = rmax <= rmin
    if bad.any():
        rmax_lin = np.interp(x_env, peaks["omega"], peaks["value"])
        rmin_lin = np.interp(x_env, troughs["omega"], troughs["value"])
        rmax = np.where(bad, np.maximum(rmax_lin, rmin_lin + 1e-9), rmax)
        rmin = np.where(bad, rmin_lin, rmin)
    return x_env, rmax, rmin


def invert_refractive_index(
    rmax: np.ndarray, rmin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由包络反演折射率（正入射简化）：返回 A、B、n、ns。"""
    C = np.sqrt(rmax)
    D = np.sqrt(rmin)
    A = (C + D) / 2.0
    B = (C - D) / 2.0
    n = (1.0 + A) / (1.0 - A)
    ns = n * (1.0 - A**2 - B) / (1.0 - A**2 + B)
    return A, B, n, ns


def invert_refractive_index_multibeam(
    rmax: np.ndarray, rmin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由包络反演折射率（多光束干涉模型，式 35-36）：返回 A、B、n、ns。"""
    C = np.sqrt(rmax)
    D = np.sqrt(rmin)
    rad = np.clip((1.0 - C**2) * (1.0 - D**2), 0.0, None)
    A = (1.0 + C * D - np.sqrt(rad)) / (C + D)
    B = (1.0 - C * D - np.sqrt(rad)) / (C - D)
    n = (1.0 + A) / (1.0 - A)
    ns = n * (1.0 - A**2 - B) / (1.0 - A**2 + B)
    return A, B, n, ns


def thickness_pairs(
    extrema: pd.DataFrame,
    n_spl: CubicSpline,
    theta1_deg: float,
    n0: float = 1.0,
) -> pd.DataFrame:
    """对给定极值点序列计算全部组合的厚度估计值。

    条纹间隔数 N = (index_j - index_i) / 2；排除 N = 0.5（相邻峰谷）的组合。
    厚度单位：μm。
    """
    w = extrema["omega"].to_numpy()
    kinds = extrema["kind"].to_numpy()
    idx = (
        extrema["index"].to_numpy()
        if "index" in extrema.columns
        else np.arange(len(extrema))
    )
    n_vals = n_spl(w)
    s2 = np.sin(np.radians(theta1_deg)) ** 2

    rows = []
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            N = (idx[j] - idx[i]) / 2.0
            if N < 1.0:
                continue
            denom = 2.0 * (
                w[j] * np.sqrt(n_vals[j] ** 2 - n0**2 * s2)
                - w[i] * np.sqrt(n_vals[i] ** 2 - n0**2 * s2)
            )
            if denom <= 0:
                continue
            d_cm = N / denom
            rows.append(
                {
                    "omega_i": w[i],
                    "kind_i": kinds[i],
                    "n_i": n_vals[i],
                    "omega_j": w[j],
                    "kind_j": kinds[j],
                    "n_j": n_vals[j],
                    "N": N,
                    "d_um": d_cm * 1e4,
                }
            )
    return pd.DataFrame(rows)


def theoretical_reflectance(
    w: np.ndarray,
    d_cm: float,
    n_spl: CubicSpline,
    theta1_deg: float,
    delta_phi: float,
    rmax_spl: CubicSpline,
    rmin_spl: CubicSpline,
    n0: float = 1.0,
) -> np.ndarray:
    """按式 (27) 重构理论干涉光谱。"""
    rmax = rmax_spl(w)
    rmin = rmin_spl(w)
    s2 = np.sin(np.radians(theta1_deg)) ** 2
    delta = 2.0 * d_cm * np.sqrt(n_spl(w) ** 2 - n0**2 * s2)
    return (rmin + rmax) / 2.0 + (rmax - rmin) * (
        np.cos(delta * np.pi * w + delta_phi) ** 2 - 0.5
    )


def fit_delta_phi(
    w: np.ndarray,
    y_obs: np.ndarray,
    d_cm: float,
    n_spl: CubicSpline,
    theta1_deg: float,
    rmax_spl: CubicSpline,
    rmin_spl: CubicSpline,
) -> tuple[float, float]:
    """拟合理论光谱的相位参数 delta_phi，返回 (delta_phi, RMSE)。"""

    def obj(phi: float) -> float:
        r_the = theoretical_reflectance(
            w, d_cm, n_spl, theta1_deg, phi, rmax_spl, rmin_spl
        )
        return float(np.mean((r_the - y_obs) ** 2))

    res = minimize_scalar(obj, bounds=(0.0, np.pi), method="bounded")
    r_the = theoretical_reflectance(
        w, d_cm, n_spl, theta1_deg, float(res.x), rmax_spl, rmin_spl
    )
    rmse = float(np.sqrt(np.mean((r_the - y_obs) ** 2)))
    return float(res.x), rmse
