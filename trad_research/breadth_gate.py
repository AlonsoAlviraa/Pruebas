"""Causal universe breadth gate for research entry filters.

Breadth = fraction of names with close > SMA(sma_period) on date t,
using only prices ≤ t (rolling SMA is causal).

Fail-closed: if fewer than ``min_names`` have a valid SMA that day → risk-off.

Causality class (stack convention): same-bar ``close[t] > SMA[t]`` then mega ANDs
into ``regime_ok`` for same-day entry filled at the **same close** — matches index
regime maps in ``regime.py``. Strict EOD→next-open would lag risk-on by 1 bar
(optional future sensitivity: shift risk-on map by +1 session). Not look-ahead into
future bars.

Research only. Not financial advice. No look-ahead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreadthGateConfig:
    """Pre-registered breadth overlay knobs (do not retune on OOS)."""

    enabled: bool = True
    sma_period: int = 50
    min_breadth: float = 0.40  # require ≥40% of names above SMA
    min_names: int = 8  # fail-closed if fewer names with valid SMA
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_dates(index: pd.DatetimeIndex | pd.Index) -> pd.DatetimeIndex:
    di = pd.to_datetime(index, utc=True)
    try:
        di = di.normalize()
    except Exception:
        pass
    return di


def compute_breadth_series(
    close_by_ticker: Mapping[str, pd.Series],
    *,
    sma_period: int = 50,
    min_names: int = 8,
) -> pd.Series:
    """Return daily breadth fraction in [0, 1] (NaN when insufficient names).

    Each series is aligned independently; SMA uses only past/current bars.
    """
    if not close_by_ticker:
        return pd.Series(dtype=float)

    flags: list[pd.Series] = []
    for _t, s in close_by_ticker.items():
        if s is None or len(s) < max(5, sma_period // 2):
            continue
        c = pd.to_numeric(s, errors="coerce").astype(float)
        c.index = _normalize_dates(c.index)
        c = c[~c.index.duplicated(keep="last")].sort_index()
        sma = c.rolling(int(sma_period), min_periods=max(5, int(sma_period) // 2)).mean()
        above = (c > sma).astype(float)
        above[sma.isna() | c.isna()] = np.nan
        flags.append(above)

    if not flags:
        return pd.Series(dtype=float)

    mat = pd.concat(flags, axis=1)
    n_valid = mat.notna().sum(axis=1)
    breadth = mat.mean(axis=1, skipna=True)
    breadth = breadth.where(n_valid >= int(min_names), other=np.nan)
    return breadth.sort_index()


def breadth_to_risk_on_map(
    breadth: pd.Series,
    *,
    min_breadth: float = 0.40,
) -> Dict[pd.Timestamp, bool]:
    """Map date → True when breadth ≥ min_breadth (NaN → False, fail-closed)."""
    out: Dict[pd.Timestamp, bool] = {}
    if breadth is None or breadth.empty:
        return out
    thr = float(min_breadth)
    for d, v in breadth.items():
        ts = pd.Timestamp(d)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        try:
            ts = ts.normalize()
        except Exception:
            pass
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out[ts] = False
        else:
            out[ts] = bool(float(v) >= thr)
    return out


def build_breadth_risk_on_map(
    close_by_ticker: Mapping[str, pd.Series],
    cfg: BreadthGateConfig | None = None,
) -> tuple[Dict[pd.Timestamp, bool], pd.Series, Dict[str, Any]]:
    """Build causal risk-on map from universe closes.

    Returns (risk_on_map, breadth_series, meta).
    """
    cfg = cfg or BreadthGateConfig()
    if not cfg.enabled:
        return {}, pd.Series(dtype=float), {"enabled": False, **cfg.to_dict()}

    breadth = compute_breadth_series(
        close_by_ticker,
        sma_period=int(cfg.sma_period),
        min_names=int(cfg.min_names),
    )
    risk_on = breadth_to_risk_on_map(breadth, min_breadth=float(cfg.min_breadth))
    valid = breadth.dropna()
    meta: Dict[str, Any] = {
        **cfg.to_dict(),
        "n_days": int(len(breadth)),
        "n_days_valid": int(len(valid)),
        "mean_breadth": float(valid.mean()) if len(valid) else float("nan"),
        "frac_risk_on": (
            float(sum(1 for v in risk_on.values() if v) / max(len(risk_on), 1))
            if risk_on
            else float("nan")
        ),
    }
    return risk_on, breadth, meta


def and_regime_maps(
    a: Optional[Dict[pd.Timestamp, bool]],
    b: Optional[Dict[pd.Timestamp, bool]],
    *,
    default_if_empty: bool = True,
) -> Dict[pd.Timestamp, bool]:
    """Causal AND of two date→bool maps. Missing key in either → fail-closed False
    when the other is present; if both empty, return {}.
    """
    if not a and not b:
        return {}
    if not a:
        return dict(b or {})
    if not b:
        return dict(a or {})
    keys = set(a) | set(b)
    out: Dict[pd.Timestamp, bool] = {}
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va is None and vb is None:
            out[k] = bool(default_if_empty)
        elif va is None or vb is None:
            out[k] = False  # fail-closed when only one calendar has the day
        else:
            out[k] = bool(va) and bool(vb)
    return out


def closes_from_panels(
    panels: Mapping[str, pd.DataFrame],
    *,
    close_col: str = "close",
) -> Dict[str, pd.Series]:
    """Extract close series from panel DataFrames (date index or date column)."""
    out: Dict[str, pd.Series] = {}
    for t, df in panels.items():
        if df is None or df.empty:
            continue
        if close_col not in df.columns:
            continue
        if "date" in df.columns:
            s = df.set_index("date")[close_col]
        else:
            s = df[close_col]
        out[str(t)] = s
    return out
