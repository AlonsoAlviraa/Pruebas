"""ORB + HTF daily proxy (Sistema A) — pure signal helpers.

data_label=eod_proxy — NOT a true 15m session ORB.
See docs/design/2026-07-27_orb_htf_falsification.md.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd

BiasMode = Literal["dual_ma", "sma200_only"]


def compute_orb_htf_signals(
    df: pd.DataFrame,
    *,
    bias_mode: BiasMode = "dual_ma",
    eps: float = 1e-6,
) -> Tuple[pd.Series, pd.Series]:
    """Causal long signals from OHLCV + SMA features.

    Requires columns: open, high, low, close, sma_50 (if dual), sma_200, atr_norm optional.
    orb_high/low use prior-day high/low only (shift 1) — no look-ahead.
    """
    need = ["open", "high", "low", "close"]
    for c in need:
        if c not in df.columns:
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")

    orb_high = high.shift(1)
    # orb_low reserved for future exact stop wiring
    _orb_low = low.shift(1)

    if bias_mode == "sma200_only":
        if "sma_200" not in df.columns:
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        sma200 = pd.to_numeric(df["sma_200"], errors="coerce")
        bias = close > sma200
    else:
        if "sma_50" not in df.columns or "sma_200" not in df.columns:
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        sma50 = pd.to_numeric(df["sma_50"], errors="coerce")
        sma200 = pd.to_numeric(df["sma_200"], errors="coerce")
        bias = (close > sma50) & (close > sma200)

    breakout = close > orb_high
    bull_day = close > open_
    sig = bias & breakout & bull_day & orb_high.notna()

    if "atr_norm" in df.columns:
        atr_n = pd.to_numeric(df["atr_norm"], errors="coerce").fillna(0.02).clip(lower=eps)
    else:
        atr_n = pd.Series(0.02, index=df.index)

    raw = (close / orb_high.replace(0.0, np.nan) - 1.0) / atr_n
    score = raw.where(sig, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return sig.fillna(False), score.astype(float)
