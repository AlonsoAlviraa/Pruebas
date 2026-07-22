"""Causal TA / volume gates for options paper strategies.

All gates use feed history **through** the evaluation day only (no look-ahead).
Option marks remain ``proxy_bs``; these gates only decide whether to open new risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from paper_live.signals.daily_pipeline import _f, _last_row


@dataclass(frozen=True)
class TaGateResult:
    """Outcome of evaluating meta TA gates for an options open."""

    allow: bool
    reason: str = ""
    features: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow": self.allow,
            "reason": self.reason,
            "features": dict(self.features or {}),
        }


def _featured_row(feed: Any, ticker: str, day: Union[str, date]) -> Optional[pd.Series]:
    try:
        feat = feed.featured(ticker, through=day)
    except Exception:
        return None
    return _last_row(feat)


def _atr_percentile(feed: Any, ticker: str, day: Union[str, date], window: int = 60) -> Optional[float]:
    """Causal rank of current atr_norm vs trailing window (0–1)."""
    try:
        feat = feed.featured(ticker, through=day)
    except Exception:
        return None
    if feat is None or feat.empty or "atr_norm" not in feat.columns:
        return None
    series = pd.to_numeric(feat["atr_norm"], errors="coerce").dropna()
    if len(series) < max(10, window // 3):
        return None
    tail = series.iloc[-window:] if len(series) >= window else series
    cur = float(tail.iloc[-1])
    if not np.isfinite(cur):
        return None
    # fraction of past days with atr_norm <= current (inclusive rank)
    return float((tail <= cur).mean())


def _recent_volume_elevated(
    feed: Any,
    ticker: str,
    day: Union[str, date],
    *,
    lookback: int = 5,
    min_ratio: float = 1.3,
) -> bool:
    """True if any of the last ``lookback`` bars (≤ day) had elevated volume_ratio."""
    try:
        feat = feed.featured(ticker, through=day)
    except Exception:
        return False
    if feat is None or feat.empty or "volume_ratio" not in feat.columns:
        return False
    vr = pd.to_numeric(feat["volume_ratio"], errors="coerce").dropna()
    if vr.empty:
        return False
    tail = vr.iloc[-lookback:]
    return bool((tail >= min_ratio).any())


def evaluate_ta_gates(
    feed: Any,
    ticker: str,
    day: Union[str, date],
    meta: Optional[Mapping[str, Any]] = None,
) -> TaGateResult:
    """Evaluate all TA/volume meta keys; fail-closed if a required gate fails.

    Known meta keys (all optional, causal):

    * ``require_uptrend`` — close > sma50 and close > sma200
    * ``require_sma200`` — close > sma200 only
    * ``require_volume_confirm`` — volume_ratio ≥ min_volume_ratio OR z ≥ min_volume_z
    * ``require_volume_dryup`` — volume_ratio ≤ max_volume_ratio OR z ≤ max_volume_z
    * ``require_rsi_oversold`` — rsi_14 ≤ max_rsi (default 35)
    * ``require_rsi_overbought`` — rsi_14 ≥ min_rsi (default 70)
    * ``require_low_atr`` — atr_norm percentile ≤ max_atr_pctile (default 0.40)
    * ``require_range_regime`` — low ATR + |dist_sma_50| ≤ max_dist (default 0.04)
    * ``require_vol_climax`` — volume_zscore ≥ min_z (default 1.5) or ratio ≥ 1.8
    * ``require_compression_after_vol`` — recent elevated volume then current dry + low ATR
    * ``require_pullback_uptrend`` — close > sma200 and (near/under sma50 or soft RSI)
    * ``min_volume_ratio``, ``max_volume_ratio``, ``min_volume_z``, ``max_volume_z``
    * ``max_rsi``, ``min_rsi``, ``max_atr_pctile``, ``max_dist_sma50``
    """
    meta = dict(meta or {})
    # Legacy HV gate is handled in replay_options; skip if no TA keys
    ta_keys = {
        "require_uptrend",
        "require_sma200",
        "require_volume_confirm",
        "require_volume_dryup",
        "require_rsi_oversold",
        "require_rsi_overbought",
        "require_low_atr",
        "require_range_regime",
        "require_vol_climax",
        "require_compression_after_vol",
        "require_pullback_uptrend",
    }
    if not any(meta.get(k) for k in ta_keys):
        return TaGateResult(allow=True, reason="no_ta_gates")

    row = _featured_row(feed, ticker, day)
    if row is None:
        return TaGateResult(allow=False, reason="no_features")

    close = _f(row, "close")
    sma50 = _f(row, "sma_50")
    sma200 = _f(row, "sma_200")
    atr_n = _f(row, "atr_norm")
    rsi = _f(row, "rsi_14")
    vr = _f(row, "volume_ratio")
    vz = _f(row, "volume_zscore")
    dist = _f(row, "dist_sma_50")
    if not np.isfinite(dist) and np.isfinite(close) and np.isfinite(sma50) and sma50 > 0:
        dist = close / sma50 - 1.0

    feats: Dict[str, Any] = {
        "close": close,
        "sma_50": sma50,
        "sma_200": sma200,
        "atr_norm": atr_n,
        "rsi_14": rsi,
        "volume_ratio": vr,
        "volume_zscore": vz,
        "dist_sma_50": dist,
    }

    min_vr = float(meta.get("min_volume_ratio") or 1.25)
    max_vr = float(meta.get("max_volume_ratio") or 0.80)
    min_vz = float(meta.get("min_volume_z") or 0.75)
    max_vz = float(meta.get("max_volume_z") or -0.4)
    max_rsi = float(meta.get("max_rsi") or 35.0)
    min_rsi = float(meta.get("min_rsi") or 70.0)
    max_atr_pctile = float(meta.get("max_atr_pctile") or 0.40)
    max_dist = float(meta.get("max_dist_sma50") or 0.04)

    if meta.get("require_uptrend"):
        # Fail closed: both SMAs must be finite (SMA200 needs long history).
        if not np.isfinite(close):
            return TaGateResult(False, "uptrend_no_close", feats)
        if not np.isfinite(sma50) or not np.isfinite(sma200):
            return TaGateResult(False, "uptrend_missing_sma", feats)
        if close <= sma50:
            return TaGateResult(False, "uptrend_below_sma50", feats)
        if close <= sma200:
            return TaGateResult(False, "uptrend_below_sma200", feats)

    if meta.get("require_sma200"):
        if not np.isfinite(close) or not np.isfinite(sma200) or close <= sma200:
            return TaGateResult(False, "below_sma200", feats)

    if meta.get("require_volume_confirm"):
        ok = (np.isfinite(vr) and vr >= min_vr) or (np.isfinite(vz) and vz >= min_vz)
        if not ok:
            return TaGateResult(False, "volume_not_elevated", feats)

    if meta.get("require_volume_dryup"):
        ok = (np.isfinite(vr) and vr <= max_vr) or (np.isfinite(vz) and vz <= max_vz)
        if not ok:
            return TaGateResult(False, "volume_not_dry", feats)

    if meta.get("require_rsi_oversold"):
        if not np.isfinite(rsi) or rsi > max_rsi:
            return TaGateResult(False, "rsi_not_oversold", feats)

    if meta.get("require_rsi_overbought"):
        if not np.isfinite(rsi) or rsi < min_rsi:
            return TaGateResult(False, "rsi_not_overbought", feats)

    if meta.get("require_vol_climax"):
        climax_z = float(meta.get("climax_volume_z") or 1.5)
        climax_r = float(meta.get("climax_volume_ratio") or 1.8)
        ok = (np.isfinite(vz) and vz >= climax_z) or (np.isfinite(vr) and vr >= climax_r)
        if not ok:
            return TaGateResult(False, "no_volume_climax", feats)

    if meta.get("require_low_atr") or meta.get("require_range_regime"):
        pctile = _atr_percentile(feed, ticker, day)
        feats["atr_pctile"] = pctile
        if pctile is None or pctile > max_atr_pctile:
            return TaGateResult(False, "atr_not_low", feats)

    if meta.get("require_range_regime"):
        if not np.isfinite(dist) or abs(dist) > max_dist:
            return TaGateResult(False, "not_in_range", feats)

    if meta.get("require_pullback_uptrend"):
        if not np.isfinite(close) or not np.isfinite(sma200) or close <= sma200:
            return TaGateResult(False, "pullback_no_uptrend", feats)
        near_sma50 = np.isfinite(sma50) and close <= sma50 * 1.02
        soft_rsi = np.isfinite(rsi) and rsi <= 48.0
        soft_dist = np.isfinite(dist) and dist <= 0.015
        if not (near_sma50 or soft_rsi or soft_dist):
            return TaGateResult(False, "not_pullback", feats)

    if meta.get("require_compression_after_vol"):
        # Recent participation then quiet + low ATR (classic premium-selling setup)
        lookback = int(meta.get("vol_lookback") or 5)
        elev_ratio = float(meta.get("elevated_volume_ratio") or 1.3)
        recent = _recent_volume_elevated(
            feed, ticker, day, lookback=lookback, min_ratio=elev_ratio
        )
        dry_now = (np.isfinite(vr) and vr <= max_vr) or (np.isfinite(vz) and vz <= max_vz)
        pctile = feats.get("atr_pctile")
        if pctile is None:
            pctile = _atr_percentile(feed, ticker, day)
            feats["atr_pctile"] = pctile
        low_atr = pctile is not None and pctile <= max_atr_pctile
        if not (recent and dry_now and low_atr):
            return TaGateResult(False, "no_compression_after_vol", feats)

    return TaGateResult(allow=True, reason="ta_gates_pass", features=feats)


def should_skip_new_from_meta(
    feed: Any,
    ticker: str,
    day: Union[str, date],
    meta: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Convenience: True ⇒ do not open new option risk today."""
    return not evaluate_ta_gates(feed, ticker, day, meta).allow
