"""Index regime filters — industry-standard + market-specific overlays.

Common practice (Faber GTAA, dual momentum risk-on, CTA trend):
only take new long risk when the broad index is above key moving averages
or has positive absolute momentum.

IBEX-specific filters (`ibex_*`) are calibrated on the **design window
2010-01 → 2017-12** (pre Spain OOS). They avoid US-style dual/strict MA
gates that over-block on range-bound European indices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _load_index_close(
    data_root: Path,
    preferred: Optional[Tuple[str, ...]] = None,
) -> Optional[pd.DataFrame]:
    """Load first available index history. preferred overrides default US order."""
    names = preferred or ("IBEX", "QQQ", "SPY", "IVV")
    for name in names:
        p = data_root / f"{name}_history.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        df["index"] = name
        return df
    return None


def _to_map(dates: pd.Series, flags: pd.Series) -> Dict[pd.Timestamp, bool]:
    return {d: bool(f) for d, f in zip(dates, flags.fillna(False))}


def hysteresis_above(
    close: pd.Series,
    level: pd.Series,
    *,
    exit_band: float = 0.03,
) -> pd.Series:
    """Stateful risk-on: enter when close > level; exit when close < level*(1-band).

    Reduces SMA whipsaw vs a hard close>SMA rule. exit_band=0.03 → 3% cushion.
    No look-ahead: state only uses current and past bars.
    """
    on = False
    out = []
    c_vals = close.to_numpy(dtype=float)
    l_vals = level.to_numpy(dtype=float)
    for px, lv in zip(c_vals, l_vals):
        if np.isnan(px) or np.isnan(lv) or lv <= 0:
            out.append(False)
            continue
        if not on and px > lv:
            on = True
        elif on and px < lv * (1.0 - exit_band):
            on = False
        out.append(on)
    return pd.Series(out, index=close.index, dtype=bool)


def build_all_regime_maps(
    data_root: Path,
    preferred_index: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[Dict[pd.Timestamp, bool], Dict[pd.Timestamp, bool], str]]:
    """
    Returns name -> (hard_risk_on, soft_full_size, description).

    hard: block new entries when False
    soft: scale size when False (risk_off_scale)
    preferred_index: e.g. ("IBEX",) for Spain OOS tests
    """
    pref = tuple(preferred_index) if preferred_index else None
    df = _load_index_close(Path(data_root), preferred=pref)
    if df is None or df.empty:
        empty: Dict[pd.Timestamp, bool] = {}
        return {
            "none": (empty, empty, "No index file; no regime filter"),
        }
    index_name = str(df["index"].iloc[0]) if "index" in df.columns else "INDEX"

    close = df["close"].reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)
    sma20 = close.rolling(20, min_periods=10).mean()
    sma50 = close.rolling(50, min_periods=25).mean()
    sma100 = close.rolling(100, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=100).mean()
    ret_12m = close.pct_change(252)
    ret_6m = close.pct_change(126)

    # --- Industry / research standards (US-friendly; often too tight on IBEX) ---
    hard_200 = close > sma200
    hard_50 = close > sma50
    hard_dual = (close > sma50) & (close > sma200)
    hard_golden = sma50 > sma200
    hard_strict = hard_dual & hard_golden
    hard_abs = ret_12m > 0
    hard_abs6 = ret_6m > 0
    soft_100 = close > sma100
    hard_legacy = (close > sma50) | (close > sma20)
    # Level-band (stateless): close > SMA200 * 0.98
    hard_200_hyst = close > (sma200 * 0.98)

    # --- IBEX-specific (design window 2010–2017; not retuned on 2018–2025) ---
    # Design findings on IBEX:
    # - dual / strict MA: no positive ON-vs-OFF edge (often worse)
    # - abs mom 12m: best simple long filter pre-2018
    # - SMA200 with 3% *stateful* hysteresis: second best, less whipsaw
    # - deep-bear only: only block when BOTH below SMA200 AND mom12 < 0
    #   (lets stock-picker run in shallow/sideways bears like parts of 2018)
    hard_ibex_mom12 = ret_12m > 0
    hard_ibex_hyst = hysteresis_above(close, sma200, exit_band=0.03)
    hard_ibex_not_bear = ~((close < sma200) & (ret_12m < 0))
    # Union risk-on: either long-term mom or above 200 (higher occupancy for
    # European stock-picking sleeves that still work when index is flat)
    hard_ibex_or = (ret_12m > 0) | (close > sma200)
    # Soft block: only severe drawdown regimes (mom12 < -10% and below SMA200)
    hard_ibex_soft = ~((ret_12m < -0.10) & (close < sma200))
    # IBEX soft size map: full size if above SMA100 OR 6m mom not deeply red
    soft_ibex = (close > sma100) | (ret_6m > -0.05)

    idx = dates
    soft_map = _to_map(idx, soft_100)
    soft_ibex_map = _to_map(idx, soft_ibex)

    return {
        "none": (
            {d: True for d in dates},
            {d: True for d in dates},
            f"No index gate (always risk-on); index_available={index_name}",
        ),
        "legacy_sma50": (
            _to_map(idx, hard_legacy),
            soft_map,
            f"Legacy: {index_name}>SMA50 or SMA20; soft SMA100",
        ),
        "sma200": (
            _to_map(idx, hard_200),
            soft_map,
            f"Industry Faber GTAA: {index_name} close > SMA200",
        ),
        "sma50": (
            _to_map(idx, hard_50),
            soft_map,
            f"Intermediate: {index_name} close > SMA50",
        ),
        "dual_50_200": (
            _to_map(idx, hard_dual),
            soft_map,
            f"Industry dual MA: {index_name} close > SMA50 AND SMA200",
        ),
        "golden_cross": (
            _to_map(idx, hard_golden),
            soft_map,
            f"Golden cross: {index_name} SMA50 > SMA200",
        ),
        "strict_dual_golden": (
            _to_map(idx, hard_strict),
            soft_map,
            f"Strict: {index_name} dual MA + golden cross",
        ),
        "abs_mom_12m": (
            _to_map(idx, hard_abs),
            soft_map,
            f"Absolute momentum 12m: {index_name} return > 0",
        ),
        "abs_mom_6m": (
            _to_map(idx, hard_abs6),
            soft_map,
            f"Absolute momentum 6m: {index_name} return > 0",
        ),
        "sma200_hysteresis": (
            _to_map(idx, hard_200_hyst),
            soft_map,
            f"SMA200 with 2% level band on {index_name} (stateless)",
        ),
        # ---- IBEX family (use with preferred_index=IBEX) ----
        "ibex_mom12": (
            _to_map(idx, hard_ibex_mom12),
            soft_ibex_map,
            f"IBEX-design: {index_name} abs mom 12m > 0; soft SMA100|mom6>-5%",
        ),
        "ibex_sma200_hyst3": (
            _to_map(idx, hard_ibex_hyst),
            soft_ibex_map,
            f"IBEX-design: {index_name} SMA200 enter/exit ±3% hysteresis (stateful)",
        ),
        "ibex_not_bear": (
            _to_map(idx, hard_ibex_not_bear),
            soft_ibex_map,
            f"IBEX-design: block only deep bear ({index_name}<SMA200 AND mom12<0)",
        ),
        "ibex_or_trend": (
            _to_map(idx, hard_ibex_or),
            soft_ibex_map,
            f"IBEX-design: risk-on if mom12>0 OR {index_name}>SMA200",
        ),
        "ibex_soft_block": (
            _to_map(idx, hard_ibex_soft),
            soft_ibex_map,
            f"IBEX-design: block only severe (mom12<-10% AND {index_name}<SMA200)",
        ),
        # Portable aliases (RSK-02): economic priors. Where logic matches ibex_*,
        # informed by IBEX design window 2010–2017; constants frozen; never re-ranked
        # on 2018–2025 OOS. Not "zero Spain-market information" and not IBEX ML.
        "portable_abs_mom_12m": (
            _to_map(idx, hard_abs),
            soft_ibex_map,
            f"Portable Antonacci: {index_name} abs mom 12m > 0",
        ),
        "portable_sma200": (
            _to_map(idx, hard_200),
            soft_ibex_map,
            f"Portable Faber GTAA: {index_name} close > SMA200",
        ),
        "portable_not_deep_bear": (
            _to_map(idx, hard_ibex_not_bear),
            soft_ibex_map,
            f"Portable not-deep-bear: block only {index_name}<SMA200 AND mom12<0 "
            f"(logic shared with ibex_not_bear; design-informed 2010–17, frozen)",
        ),
    }
