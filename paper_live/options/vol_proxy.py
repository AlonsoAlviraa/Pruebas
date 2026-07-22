"""Realized vol and crude IV proxy from OHLCV (no options chain required)."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def historical_vol(
    closes: Union[pd.Series, np.ndarray],
    *,
    window: int = 20,
    trading_days: float = 252.0,
) -> float:
    """Annualized stdev of log returns over last ``window`` bars."""
    s = pd.Series(closes).astype(float).dropna()
    if len(s) < window + 1:
        return float("nan")
    rets = np.log(s / s.shift(1)).dropna().tail(window)
    if rets.empty:
        return float("nan")
    return float(rets.std(ddof=1) * math_sqrt(trading_days))


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    *,
    window: int = 20,
    trading_days: float = 252.0,
) -> float:
    """Parkinson range estimator (annualized)."""
    h = high.astype(float)
    l = low.astype(float)
    n = min(len(h), len(l), window)
    if n < 5:
        return float("nan")
    rs = np.log(h.tail(n).values / l.tail(n).values) ** 2
    # (1/(4 n ln2)) * sum
    var = rs.mean() / (4.0 * np.log(2.0))
    return float(np.sqrt(var * trading_days))


def iv_proxy_from_hv(
    hv: float,
    *,
    premium_mult: float = 1.15,
    floor: float = 0.08,
    cap: float = 1.5,
) -> float:
    """
    Crude IV proxy: inflate HV by a VRP-like multiplier.

    Real chains needed for true IV; this encodes the *hypothesis* that
    market prices options ~15% richer than recent HV (configurable).
    """
    if hv is None or not np.isfinite(hv) or hv <= 0:
        return float("nan")
    iv = float(hv) * float(premium_mult)
    return float(min(max(iv, floor), cap))
