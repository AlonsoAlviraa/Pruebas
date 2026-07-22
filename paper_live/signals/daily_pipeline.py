"""Daily causal signal pipeline (post-close D → candidates for entry on D+1).

LIV-04: rule-based paper signals with explicit A/B ``signal_mode`` variants
(AUD-B). Full XGB retrain can plug in later via ``signal_fn`` override.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed

logger = logging.getLogger(__name__)
_UNKNOWN_MODE_WARNED: set[str] = set()

# Modes handled by score_row_for_mode (pipeline specials like qqq_hold are separate).
KNOWN_SIGNAL_MODES = frozenset(
    {
        "trend_mom",
        "topk_mom",
        "qqq_gate",
        "baseline",
        "no_extension",
        "combined_v1",
        "pullback",
        "combined_v2",
        "combined_v3",
        "vol_confirm",
        "volume_breakout",
        "vol_breakout",
        "rsi_mr",
        "rsi_mean_reversion",
        "rsi_oversold",
        "vol_dryup",
        "volume_dryup",
        "vol_expand",
        "volume_expansion",
        "rvol_trend",
        "rel_volume_trend",
        "vol_pullback",
        "volume_pullback",
        "combined_ta_v1",
    }
)

# Index names used for regime only — never trade by default when exclude_index
INDEX_TICKERS = frozenset({"SPY", "QQQ", "IWM", "DIA"})


@dataclass(frozen=True)
class EntryCandidate:
    ticker: str
    signal_date: date  # as-of close used for features
    score: float
    p_buy: float
    close: float
    atr: float
    atr_norm: float
    reason: str = "rule_trend_mom"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "signal_date": self.signal_date.isoformat(),
            "score": self.score,
            "p_buy": self.p_buy,
            "close": self.close,
            "atr": self.atr,
            "atr_norm": self.atr_norm,
            "reason": self.reason,
            "meta": self.meta,
        }


@dataclass
class SignalBatch:
    signal_date: date
    regime_on: bool
    candidates: List[EntryCandidate]
    n_scanned: int = 0
    n_rejected: int = 0

    def top(self, n: int) -> List[EntryCandidate]:
        return sorted(self.candidates, key=lambda c: c.score, reverse=True)[:n]


SignalFn = Callable[[DailyReplayFeed, date, Sequence[str]], SignalBatch]


def _last_row(feat: pd.DataFrame) -> Optional[pd.Series]:
    if feat is None or feat.empty:
        return None
    return feat.iloc[-1]


def _f(row: pd.Series, key: str, default: float = np.nan) -> float:
    try:
        v = row.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def default_rule_signal_row(row: pd.Series) -> Optional[tuple[float, float, str]]:
    """Baseline: above SMA50/200 + ret_1m>0 + ATR band (legacy — often buys extension)."""
    close = _f(row, "close")
    sma50 = _f(row, "sma_50")
    sma200 = _f(row, "sma_200")
    ret_1m = _f(row, "ret_1m")
    atr_n = _f(row, "atr_norm")
    vol20 = _f(row, "volatility_20", 0.0)

    if not np.isfinite(close) or close <= 0:
        return None
    if np.isfinite(sma50) and close <= sma50:
        return None
    if np.isfinite(sma200) and close <= sma200:
        return None
    if not np.isfinite(ret_1m) or ret_1m <= 0:
        return None
    if not np.isfinite(atr_n) or atr_n < 0.005 or atr_n > 0.25:
        return None

    score = float(ret_1m) * (1.0 + min(max(atr_n, 0.0), 0.15) / 0.05)
    score *= 1.0 + min(max(vol20, 0.0), 1.0) * 0.25
    p_buy = float(np.clip(0.55 + ret_1m * 2.0 + atr_n, 0.0, 0.99))
    return score, p_buy, "rule_trend_mom_atr"


def no_extension_signal_row(
    row: pd.Series,
    *,
    max_dist_sma50: float = 0.05,
    max_rsi: float = 68.0,
    max_ret_1m: float = 0.18,
) -> Optional[tuple[float, float, str]]:
    """Trend + mom but reject parabolic extension (audit root cause)."""
    base = default_rule_signal_row(row)
    if base is None:
        return None
    dist = _f(row, "dist_sma_50")
    if not np.isfinite(dist):
        close, sma50 = _f(row, "close"), _f(row, "sma_50")
        if np.isfinite(close) and np.isfinite(sma50) and sma50 > 0:
            dist = close / sma50 - 1.0
    if np.isfinite(dist) and dist > max_dist_sma50:
        return None
    rsi = _f(row, "rsi_14")
    if np.isfinite(rsi) and rsi > max_rsi:
        return None
    ret_1m = _f(row, "ret_1m")
    if np.isfinite(ret_1m) and ret_1m > max_ret_1m:
        return None
    score, p_buy, _ = base
    # Prefer milder extension
    if np.isfinite(dist):
        score = score / (1.0 + max(dist, 0.0) * 8.0)
    return score, p_buy, "rule_no_extension"


def pullback_signal_row(row: pd.Series) -> Optional[tuple[float, float, str]]:
    """Long-term trend OK, intermediate weakness (buy dip in uptrend)."""
    close = _f(row, "close")
    sma50 = _f(row, "sma_50")
    sma200 = _f(row, "sma_200")
    ret_1m = _f(row, "ret_1m")
    atr_n = _f(row, "atr_norm")
    rsi = _f(row, "rsi_14")
    dist = _f(row, "dist_sma_50")

    if not np.isfinite(close) or close <= 0:
        return None
    if not np.isfinite(sma200) or close <= sma200:
        return None
    # Prefer at/under SMA50 or RSI soft
    near_sma50 = np.isfinite(sma50) and close <= sma50 * 1.01
    soft_rsi = np.isfinite(rsi) and rsi <= 48.0
    soft_dist = np.isfinite(dist) and dist <= 0.01
    if not (near_sma50 or soft_rsi or soft_dist):
        return None
    # Avoid free-fall
    if np.isfinite(ret_1m) and ret_1m < -0.12:
        return None
    if not np.isfinite(atr_n) or atr_n < 0.006 or atr_n > 0.28:
        return None

    # Higher score when more oversold vs SMA50 but still above 200
    stretch = 0.0
    if np.isfinite(dist):
        stretch = max(-dist, 0.0)  # below SMA50 → positive
    if np.isfinite(rsi):
        stretch += max(50.0 - rsi, 0.0) / 100.0
    score = 0.05 + stretch + max(ret_1m, 0.0) * 0.5
    p_buy = float(np.clip(0.52 + stretch + (0.0 if not np.isfinite(ret_1m) else ret_1m), 0.0, 0.95))
    return float(score), p_buy, "rule_pullback"


# ---------------------------------------------------------------------------
# TA / volume signal modes (causal features only — as-of row through signal day)
# ---------------------------------------------------------------------------


def volume_breakout_signal_row(
    row: pd.Series,
    *,
    min_volume_ratio: float = 1.3,
    min_volume_z: float = 0.75,
) -> Optional[tuple[float, float, str]]:
    """Trend + momentum only when volume confirms (ratio or z-score elevated)."""
    base = default_rule_signal_row(row)
    if base is None:
        return None
    vr = _f(row, "volume_ratio")
    vz = _f(row, "volume_zscore")
    vol_ok = (np.isfinite(vr) and vr >= min_volume_ratio) or (
        np.isfinite(vz) and vz >= min_volume_z
    )
    if not vol_ok:
        return None
    score, p_buy, _ = base
    # Prefer stronger relative volume
    boost = 0.0
    if np.isfinite(vr):
        boost += min(max(vr - 1.0, 0.0), 2.0) * 0.15
    if np.isfinite(vz):
        boost += min(max(vz, 0.0), 3.0) * 0.05
    score = float(score) * (1.0 + boost)
    p_buy = float(np.clip(p_buy + boost * 0.05, 0.0, 0.99))
    return score, p_buy, "rule_volume_breakout"


def rsi_mean_reversion_signal_row(
    row: pd.Series,
    *,
    rsi_max: float = 32.0,
    require_above_sma200: bool = True,
) -> Optional[tuple[float, float, str]]:
    """Oversold RSI long with causal long-term trend filter (above SMA200)."""
    close = _f(row, "close")
    sma200 = _f(row, "sma_200")
    rsi = _f(row, "rsi_14")
    atr_n = _f(row, "atr_norm")
    ret_1m = _f(row, "ret_1m")
    dist = _f(row, "dist_sma_50")

    if not np.isfinite(close) or close <= 0:
        return None
    if not np.isfinite(rsi) or rsi > rsi_max:
        return None
    if require_above_sma200:
        if not np.isfinite(sma200) or close < sma200 * 0.98:
            return None
    # Avoid free-fall / crash bars
    if np.isfinite(ret_1m) and ret_1m < -0.18:
        return None
    if not np.isfinite(atr_n) or atr_n < 0.005 or atr_n > 0.30:
        return None

    depth = max(rsi_max - rsi, 0.0) / max(rsi_max, 1.0)
    stretch = 0.0
    if np.isfinite(dist):
        stretch = max(-dist, 0.0)
    score = 0.08 + depth * 0.6 + stretch * 0.5
    p_buy = float(np.clip(0.50 + depth * 0.35 + stretch, 0.0, 0.95))
    return float(score), p_buy, "rule_rsi_mean_reversion"


def volume_dryup_signal_row(
    row: pd.Series,
    *,
    max_volume_ratio: float = 0.75,
    max_volume_z: float = -0.5,
) -> Optional[tuple[float, float, str]]:
    """Pullback-in-uptrend with quiet volume (dry-up = less selling pressure)."""
    pb = pullback_signal_row(row)
    if pb is None:
        return None
    vr = _f(row, "volume_ratio")
    vz = _f(row, "volume_zscore")
    dry = (np.isfinite(vr) and vr <= max_volume_ratio) or (
        np.isfinite(vz) and vz <= max_volume_z
    )
    if not dry:
        return None
    score, p_buy, _ = pb
    quiet_boost = 0.0
    if np.isfinite(vr):
        quiet_boost += max(1.0 - vr, 0.0) * 0.2
    score = float(score) * (1.0 + quiet_boost)
    return score, float(np.clip(p_buy + quiet_boost * 0.1, 0.0, 0.95)), "rule_volume_dryup"


def volume_expansion_signal_row(
    row: pd.Series,
    *,
    min_volume_ratio: float = 1.5,
    max_rsi: float = 70.0,
) -> Optional[tuple[float, float, str]]:
    """Uptrend + expanding volume without extreme RSI (participation, not climax)."""
    close = _f(row, "close")
    sma50 = _f(row, "sma_50")
    sma200 = _f(row, "sma_200")
    ret_1m = _f(row, "ret_1m")
    atr_n = _f(row, "atr_norm")
    rsi = _f(row, "rsi_14")
    vr = _f(row, "volume_ratio")
    vz = _f(row, "volume_zscore")

    if not np.isfinite(close) or close <= 0:
        return None
    if np.isfinite(sma50) and close <= sma50:
        return None
    if np.isfinite(sma200) and close <= sma200:
        return None
    if not np.isfinite(ret_1m) or ret_1m <= 0:
        return None
    if not np.isfinite(vr) or vr < min_volume_ratio:
        # allow z-score substitute
        if not (np.isfinite(vz) and vz >= 1.0):
            return None
    if np.isfinite(rsi) and rsi > max_rsi:
        return None
    if not np.isfinite(atr_n) or atr_n < 0.006 or atr_n > 0.25:
        return None

    vol_part = 0.0
    if np.isfinite(vr):
        vol_part = min(max(vr - 1.0, 0.0), 2.5)
    elif np.isfinite(vz):
        vol_part = min(max(vz, 0.0), 3.0) * 0.4
    score = float(ret_1m) * (1.0 + vol_part * 0.35) + vol_part * 0.05
    p_buy = float(np.clip(0.55 + ret_1m * 1.5 + vol_part * 0.05, 0.0, 0.98))
    return score, p_buy, "rule_volume_expansion"


def rvol_trend_signal_row(
    row: pd.Series,
    *,
    min_volume_ratio: float = 1.15,
    max_dist_sma50: float = 0.06,
    max_rsi: float = 68.0,
) -> Optional[tuple[float, float, str]]:
    """Relative-volume + mild trend hybrid (no_extension + volume confirm)."""
    mild = no_extension_signal_row(
        row, max_dist_sma50=max_dist_sma50, max_rsi=max_rsi
    )
    if mild is None:
        return None
    vr = _f(row, "volume_ratio")
    vz = _f(row, "volume_zscore")
    if not (
        (np.isfinite(vr) and vr >= min_volume_ratio)
        or (np.isfinite(vz) and vz >= 0.5)
    ):
        return None
    score, p_buy, _ = mild
    rvol = vr if np.isfinite(vr) else 1.0 + max(vz, 0.0) * 0.25
    score = float(score) * (0.85 + 0.25 * min(max(rvol, 0.5), 2.5))
    return score, p_buy, "rule_rvol_trend"


def vol_pullback_signal_row(row: pd.Series) -> Optional[tuple[float, float, str]]:
    """Pullback + dry volume; fallback to RSI MR if no pullback soft signal."""
    dry = volume_dryup_signal_row(row)
    if dry is not None:
        s, p, _ = dry
        return s, p, "rule_vol_pullback"
    # Mild RSI oversold with dry volume still acceptable
    rsi = _f(row, "rsi_14")
    vr = _f(row, "volume_ratio")
    if np.isfinite(rsi) and rsi <= 38.0 and np.isfinite(vr) and vr <= 0.85:
        mr = rsi_mean_reversion_signal_row(row, rsi_max=38.0)
        if mr is not None:
            s, p, _ = mr
            return s, p, "rule_vol_pullback"
    return None


def score_row_for_mode(row: pd.Series, mode: str) -> Optional[tuple[float, float, str]]:
    """Score a feature row for ``mode``.

    Unknown modes fail closed (return ``None``) instead of silently trading
    as ``trend_mom`` — zoo typos must not become baseline entries.
    """
    m = (mode or "trend_mom").lower().strip()
    if m not in KNOWN_SIGNAL_MODES:
        if m not in _UNKNOWN_MODE_WARNED:
            _UNKNOWN_MODE_WARNED.add(m)
            logger.warning(
                "Unknown signal_mode=%r — rejecting (fail-closed). Known: %s",
                mode,
                sorted(KNOWN_SIGNAL_MODES),
            )
        return None
    if m in ("trend_mom", "topk_mom", "qqq_gate", "baseline"):
        return default_rule_signal_row(row)
    if m in ("no_extension", "combined_v1"):
        return no_extension_signal_row(row)
    if m in ("pullback", "combined_v2"):
        return pullback_signal_row(row)
    if m == "combined_v3":
        # Pullback preferred; fall back to mild no-extension
        pb = pullback_signal_row(row)
        if pb is not None:
            return pb
        return no_extension_signal_row(row, max_dist_sma50=0.04, max_rsi=65.0)
    # --- TA / volume modes ---
    if m in ("vol_confirm", "volume_breakout", "vol_breakout"):
        return volume_breakout_signal_row(row)
    if m in ("rsi_mr", "rsi_mean_reversion", "rsi_oversold"):
        return rsi_mean_reversion_signal_row(row)
    if m in ("vol_dryup", "volume_dryup"):
        return volume_dryup_signal_row(row)
    if m in ("vol_expand", "volume_expansion"):
        return volume_expansion_signal_row(row)
    if m in ("rvol_trend", "rel_volume_trend"):
        return rvol_trend_signal_row(row)
    if m in ("vol_pullback", "volume_pullback"):
        return vol_pullback_signal_row(row)
    if m == "combined_ta_v1":
        # Prefer dry-up pullback, then volume-confirmed mild trend
        a = volume_dryup_signal_row(row)
        if a is not None:
            return a
        return rvol_trend_signal_row(row)
    # Defensive: known set should have been fully dispatched above
    logger.warning("signal_mode=%r in KNOWN_SIGNAL_MODES but unhandled", m)
    return None


class DailySignalPipeline:
    """Build entry candidates as-of signal_date close (no look-ahead)."""

    def __init__(
        self,
        feed: DailyReplayFeed,
        *,
        universe: Optional[Sequence[str]] = None,
        min_price: float = 5.0,
        max_atr_pct: float = 0.22,
        min_atr_norm: float = 0.008,
        regime_symbol: str = "QQQ",
        require_regime: bool = True,
        signal_fn: Optional[SignalFn] = None,
        signal_mode: str = "trend_mom",
        top_k: Optional[int] = None,
        exclude_index: bool = True,
        qqq_mom_gate: bool = False,
        qqq_min_ret_1m: float = 0.0,
    ):
        self.feed = feed
        self.universe = [t.upper() for t in (universe or feed.tickers)]
        self.min_price = float(min_price)
        self.max_atr_pct = float(max_atr_pct)
        self.min_atr_norm = float(min_atr_norm)
        self.regime_symbol = regime_symbol.upper()
        self.require_regime = bool(require_regime)
        self.signal_fn = signal_fn
        self.signal_mode = str(signal_mode or "trend_mom")
        self.top_k = int(top_k) if top_k is not None else None
        self.exclude_index = bool(exclude_index)
        # Auto-enable gates from mode name
        mode = self.signal_mode.lower()
        self.qqq_mom_gate = bool(qqq_mom_gate) or mode in (
            "qqq_gate",
            "combined_v1",
            "combined_v2",
            "combined_v3",
        )
        self.qqq_min_ret_1m = float(qqq_min_ret_1m)
        if mode in ("topk_mom", "combined_v1", "combined_v2", "combined_v3") and self.top_k is None:
            self.top_k = 3
        if mode in ("combined_v1", "combined_v2", "combined_v3"):
            self.exclude_index = True

    def regime_on(self, signal_date: Union[str, date]) -> bool:
        """Simple dual-MA style regime on index (causal)."""
        feat = self.feed.featured(self.regime_symbol, through=signal_date)
        row = _last_row(feat)
        if row is None:
            return not self.require_regime
        close = float(row["close"])
        sma50 = row.get("sma_50")
        sma200 = row.get("sma_200")
        ok = True
        if pd.notna(sma50):
            ok = ok and close > float(sma50)
        if pd.notna(sma200):
            ok = ok and close > float(sma200)
        if pd.notna(sma50) and pd.notna(sma200):
            ok = ok and float(sma50) >= float(sma200) * 0.98
        return bool(ok)

    def qqq_momentum_ok(self, signal_date: Union[str, date]) -> bool:
        feat = self.feed.featured(self.regime_symbol, through=signal_date)
        row = _last_row(feat)
        if row is None:
            return not self.qqq_mom_gate
        ret = _f(row, "ret_1m")
        if not np.isfinite(ret):
            return not self.qqq_mom_gate
        return ret >= self.qqq_min_ret_1m

    def generate(self, signal_date: Union[str, date]) -> SignalBatch:
        d = pd.Timestamp(signal_date).date()
        if self.signal_fn is not None:
            return self.signal_fn(self.feed, d, self.universe)

        regime = self.regime_on(d)
        cands: List[EntryCandidate] = []
        scanned = 0
        rejected = 0

        if self.require_regime and not regime:
            return SignalBatch(
                signal_date=d,
                regime_on=False,
                candidates=[],
                n_scanned=0,
                n_rejected=len(self.universe),
            )
        if self.qqq_mom_gate and not self.qqq_momentum_ok(d):
            return SignalBatch(
                signal_date=d,
                regime_on=regime,
                candidates=[],
                n_scanned=0,
                n_rejected=len(self.universe),
            )

        # Passive-hold style: single QQQ (or regime index) when regime on
        mode_l = self.signal_mode.lower()
        if mode_l in ("qqq_hold", "index_hold"):
            idx = self.regime_symbol if self.regime_symbol in self.feed.tickers else (
                "QQQ" if "QQQ" in self.feed.tickers else ("SPY" if "SPY" in self.feed.tickers else None)
            )
            if idx is None:
                return SignalBatch(d, regime, [], 0, 1)
            feat = self.feed.featured(idx, through=d)
            row = _last_row(feat)
            if row is None:
                return SignalBatch(d, regime, [], 1, 1)
            close = float(row["close"])
            atr = float(row["atr"]) if pd.notna(row.get("atr")) else close * 0.015
            atr_n = float(row["atr_norm"]) if pd.notna(row.get("atr_norm")) else atr / close
            return SignalBatch(
                signal_date=d,
                regime_on=regime,
                candidates=[
                    EntryCandidate(
                        ticker=idx,
                        signal_date=d,
                        score=1.0,
                        p_buy=0.7,
                        close=close,
                        atr=atr,
                        atr_norm=atr_n,
                        reason="rule_qqq_hold",
                        meta={"regime_on": regime, "signal_mode": self.signal_mode},
                    )
                ],
                n_scanned=1,
                n_rejected=0,
            )

        trade_univ = [
            t
            for t in self.universe
            if not (self.exclude_index and t in INDEX_TICKERS)
        ]

        for t in trade_univ:
            feat = self.feed.featured(t, through=d)
            row = _last_row(feat)
            scanned += 1
            if row is None:
                rejected += 1
                continue
            close = float(row["close"])
            if close < self.min_price:
                rejected += 1
                continue
            atr = float(row["atr"]) if pd.notna(row.get("atr")) else close * 0.02
            atr_n = float(row["atr_norm"]) if pd.notna(row.get("atr_norm")) else atr / close
            if atr_n > self.max_atr_pct or atr_n < self.min_atr_norm:
                rejected += 1
                continue
            sig = score_row_for_mode(row, self.signal_mode)
            if sig is None:
                rejected += 1
                continue
            score, p_buy, reason = sig
            cands.append(
                EntryCandidate(
                    ticker=t,
                    signal_date=d,
                    score=score,
                    p_buy=p_buy,
                    close=close,
                    atr=atr,
                    atr_norm=atr_n,
                    reason=reason,
                    meta={
                        "regime_on": regime,
                        "signal_mode": self.signal_mode,
                    },
                )
            )

        if self.top_k is not None and self.top_k > 0 and cands:
            cands = sorted(cands, key=lambda c: c.score, reverse=True)[: self.top_k]

        return SignalBatch(
            signal_date=d,
            regime_on=regime,
            candidates=cands,
            n_scanned=scanned,
            n_rejected=rejected,
        )
