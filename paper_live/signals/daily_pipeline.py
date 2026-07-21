"""Daily causal signal pipeline (post-close D → candidates for entry on D+1).

LIV-04 v1 uses a **rule-based** paper signal aligned with highvol/minalloc filters
(trend + momentum + ATR band). Full XGB retrain can plug in later via
``signal_fn`` override without changing the session loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed


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


def default_rule_signal_row(row: pd.Series) -> Optional[tuple[float, float, str]]:
    """Return (score, p_buy, reason) or None if not a buy candidate."""
    try:
        close = float(row["close"])
        sma50 = float(row["sma_50"]) if pd.notna(row.get("sma_50")) else np.nan
        sma200 = float(row["sma_200"]) if pd.notna(row.get("sma_200")) else np.nan
        ret_1m = float(row["ret_1m"]) if pd.notna(row.get("ret_1m")) else np.nan
        atr_n = float(row["atr_norm"]) if pd.notna(row.get("atr_norm")) else np.nan
        vol20 = float(row["volatility_20"]) if pd.notna(row.get("volatility_20")) else 0.0
    except Exception:
        return None

    if not np.isfinite(close) or close <= 0:
        return None
    # Trend: above intermediate and long MAs when available
    if np.isfinite(sma50) and close <= sma50:
        return None
    if np.isfinite(sma200) and close <= sma200:
        return None
    # Momentum
    if not np.isfinite(ret_1m) or ret_1m <= 0:
        return None
    # Vol band (skip dead quiet; soft cap extreme)
    if not np.isfinite(atr_n) or atr_n < 0.005:
        return None
    if atr_n > 0.25:
        return None

    score = float(ret_1m) * (1.0 + min(max(atr_n, 0.0), 0.15) / 0.05)
    score *= 1.0 + min(max(vol20, 0.0), 1.0) * 0.25
    p_buy = float(np.clip(0.55 + ret_1m * 2.0 + atr_n, 0.0, 0.99))
    return score, p_buy, "rule_trend_mom_atr"


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
    ):
        self.feed = feed
        self.universe = [t.upper() for t in (universe or feed.tickers)]
        self.min_price = float(min_price)
        self.max_atr_pct = float(max_atr_pct)
        self.min_atr_norm = float(min_atr_norm)
        self.regime_symbol = regime_symbol.upper()
        self.require_regime = bool(require_regime)
        self.signal_fn = signal_fn

    def regime_on(self, signal_date: Union[str, date]) -> bool:
        """Simple dual-MA style regime on index (causal)."""
        feat = self.feed.featured(self.regime_symbol, through=signal_date)
        row = _last_row(feat)
        if row is None:
            # No index → allow if not required
            return not self.require_regime
        close = float(row["close"])
        sma50 = row.get("sma_50")
        sma200 = row.get("sma_200")
        ok = True
        if pd.notna(sma50):
            ok = ok and close > float(sma50)
        if pd.notna(sma200):
            ok = ok and close > float(sma200)
        # golden-ish: sma50 > sma200 when both present
        if pd.notna(sma50) and pd.notna(sma200):
            ok = ok and float(sma50) >= float(sma200) * 0.98
        return bool(ok)

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

        for t in self.universe:
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
            sig = default_rule_signal_row(row)
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
                    meta={"regime_on": regime},
                )
            )
        return SignalBatch(
            signal_date=d,
            regime_on=regime,
            candidates=cands,
            n_scanned=scanned,
            n_rejected=rejected,
        )
