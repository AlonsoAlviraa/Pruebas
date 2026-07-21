"""Triple-barrier labels for supervised training (targets only — never as features)."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from trad_research.config import DEFAULT_LABEL_CONFIG, LabelConfig


def triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    k_tp: float = 2.5,
    k_sl: float = 1.5,
    max_horizon: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Long-side triple barrier.
    Returns:
      y: 0=SL first (SELL), 1=time/neutral (HOLD), 2=TP first (BUY)
      meta: 1 if barrier resolved with profit (TP), else 0 (for meta-labeling)
    """
    n = len(close)
    y = np.ones(n, dtype=np.int32)  # default HOLD
    meta = np.zeros(n, dtype=np.int32)

    for i in range(n - 1):
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0:
            continue
        entry = float(close[i])
        tp = entry + k_tp * a
        sl = entry - k_sl * a
        end = min(n - 1, i + max_horizon)
        label = 1
        m = 0
        for j in range(i + 1, end + 1):
            hi = float(high[j])
            lo = float(low[j])
            hit_tp = hi >= tp
            hit_sl = lo <= sl
            if hit_tp and hit_sl:
                # Conservative: SL first same bar
                label = 0
                m = 0
                break
            if hit_sl:
                label = 0
                m = 0
                break
            if hit_tp:
                label = 2
                m = 1
                break
        y[i] = label
        meta[i] = m
    return y, meta


def attach_labels(
    df: pd.DataFrame,
    k_tp: float = 2.5,
    k_sl: float = 1.5,
    max_horizon: int = 20,
    config: Optional[LabelConfig] = None,
) -> pd.DataFrame:
    """Attach y_side (0/1/2) and y_meta to a featured OHLCV frame."""
    cfg = config or LabelConfig(k_tp=k_tp, k_sl=k_sl, max_horizon=max_horizon)
    out = df.copy()
    y, meta = triple_barrier_labels(
        out["close"].to_numpy(dtype=float),
        out["high"].to_numpy(dtype=float),
        out["low"].to_numpy(dtype=float),
        out["atr"].to_numpy(dtype=float),
        k_tp=cfg.k_tp,
        k_sl=cfg.k_sl,
        max_horizon=cfg.max_horizon,
    )
    out["y_side"] = y
    out["y_meta"] = meta
    return out


def to_lopez_de_prado_side(y_side: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Map 0/1/2 → -1/0/+1 (SELL/HOLD/BUY) for legacy scripts."""
    y = np.asarray(y_side, dtype=int)
    out = np.zeros_like(y)
    out[y == 0] = -1
    out[y == 1] = 0
    out[y == 2] = 1
    return out


def label_one_event(
    close: float,
    high: np.ndarray,
    low: np.ndarray,
    atr: float,
    config: Optional[LabelConfig] = None,
) -> Tuple[int, int, float]:
    """
    Single-event label (legacy triple_barrier_labeling API).
    high/low arrays are the path AFTER entry (not including entry bar).
    Returns: (ldp_label ∈ {-1,0,1}, holding_days, return_pct)
    """
    cfg = config or DEFAULT_LABEL_CONFIG
    if not np.isfinite(atr) or atr <= 0 or close <= 0:
        return 0, 0, 0.0
    tp = close + cfg.k_tp * atr
    sl = close - cfg.k_sl * atr
    n = min(len(high), len(low), cfg.max_horizon)
    for i in range(n):
        hi = float(high[i])
        lo = float(low[i])
        days = i + 1
        if hi >= tp and lo <= sl:
            return -1, days, (sl - close) / close
        if lo <= sl:
            return -1, days, (sl - close) / close
        if hi >= tp:
            return 1, days, (tp - close) / close
    # time barrier
    if n > 0:
        # use last mid as proxy if no close path; return 0
        return 0, cfg.max_horizon, 0.0
    return 0, 0, 0.0
