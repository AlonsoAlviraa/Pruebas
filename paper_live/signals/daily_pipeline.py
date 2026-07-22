"""Daily causal signal pipeline (post-close D → candidates for entry on D+1).

LIV-04: rule-based paper signals with explicit A/B ``signal_mode`` variants
(AUD-B). Full XGB retrain can plug in later via ``signal_fn`` override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed

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


def score_row_for_mode(row: pd.Series, mode: str) -> Optional[tuple[float, float, str]]:
    m = (mode or "trend_mom").lower().strip()
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
    return default_rule_signal_row(row)


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
