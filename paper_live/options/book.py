"""Book-level risk helpers: beta-weighted delta + sleeve portfolio (paper).

Labels:
  - ``approx_bs_delta_book`` — raw share-equivalent BS deltas
  - ``beta_weighted_delta`` — delta × rolling beta to SPY (causal)
  - sleeve portfolio is paper capital split, not live multi-account
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def rolling_beta_to_spy(
    feed: Any,
    underlying: str,
    day: date,
    *,
    window: int = 60,
    spy_ticker: str = "SPY",
) -> Optional[float]:
    """Causal OLS beta of underlying daily returns vs SPY through ``day``."""
    try:
        u = feed.history(underlying, through=day, include_through=True)
        s = feed.history(spy_ticker, through=day, include_through=True)
    except Exception:
        return None
    if u is None or s is None or u.empty or s.empty:
        return None
    ud = u.set_index("date")["close"].astype(float).pct_change().dropna()
    sd = s.set_index("date")["close"].astype(float).pct_change().dropna()
    both = pd.concat([ud.rename("u"), sd.rename("s")], axis=1, join="inner").dropna()
    if len(both) < max(20, window // 3):
        return None
    tail = both.iloc[-window:] if len(both) >= window else both
    var_s = float(tail["s"].var())
    if var_s <= 1e-16:
        return None
    cov = float(tail["u"].cov(tail["s"]))
    beta = cov / var_s
    if not np.isfinite(beta):
        return None
    return float(beta)


def beta_weighted_delta(
    approx_delta: float,
    beta: Optional[float],
    *,
    default_beta: float = 1.0,
) -> float:
    """Share-equivalent delta scaled by beta to SPY."""
    b = float(beta) if beta is not None and np.isfinite(beta) else float(default_beta)
    return float(approx_delta) * b


def book_delta_report_beta(
    results: Sequence[Any],
    feed: Any,
    day: date,
    *,
    window: int = 60,
) -> Dict[str, Any]:
    """Aggregate beta-weighted book delta across strategy results."""
    end_sum = 0.0
    raw_sum = 0.0
    n = 0
    per: List[Dict[str, Any]] = []
    for r in results:
        de = getattr(r, "approx_delta_end", None)
        if de is None and isinstance(r, Mapping):
            de = r.get("approx_delta_end")
        und = getattr(r, "underlying", None) or (r.get("underlying") if isinstance(r, Mapping) else None)
        sid = getattr(r, "strategy_id", None) or (r.get("strategy_id") if isinstance(r, Mapping) else None)
        if de is None:
            continue
        beta = rolling_beta_to_spy(feed, str(und or "SPY"), day, window=window)
        bwd = beta_weighted_delta(float(de), beta)
        end_sum += bwd
        raw_sum += float(de)
        n += 1
        per.append(
            {
                "strategy_id": sid,
                "underlying": und,
                "approx_delta_end": float(de),
                "beta_spy": beta,
                "beta_weighted_delta": bwd,
            }
        )
    return {
        "n_strategies": n,
        "sum_raw_delta_end": raw_sum,
        "sum_beta_weighted_delta": end_sum,
        "mean_beta_weighted_delta": end_sum / n if n else 0.0,
        "label": "beta_weighted_delta",
        "note": "approx_bs_delta × rolling 60d OLS beta to SPY (causal). Research proxy.",
        "strategies": per,
    }


DEFAULT_SLEEVE_WEIGHTS: Dict[str, float] = {
    "covered_call": 0.40,
    "put_credit_spread": 0.30,
    "cash": 0.20,
    "protective_put": 0.10,
}


@dataclass
class SleevePortfolioResult:
    """Paper sleeve portfolio from strategy equity curves."""

    final_equity: float
    total_return: float
    max_dd: float
    weights: Dict[str, float] = field(default_factory=dict)
    members: List[str] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    label: str = "sleeve_portfolio_paper"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "max_dd": self.max_dd,
            "weights": self.weights,
            "members": self.members,
            "equity_curve": self.equity_curve,
            "label": self.label,
            "notes": self.notes,
        }


def _curve_to_series(curve: Sequence[Mapping[str, Any]]) -> pd.Series:
    if not curve:
        return pd.Series(dtype=float)
    idx = [c.get("date") for c in curve]
    vals = [float(c.get("equity") or 0.0) for c in curve]
    s = pd.Series(vals, index=pd.to_datetime(idx, utc=True, errors="coerce"))
    s = s[~s.index.isna()]
    return s.sort_index()


def build_sleeve_portfolio(
    results: Sequence[Any],
    *,
    capital0: float = 100_000.0,
    weights_by_kind: Optional[Mapping[str, float]] = None,
    rebalance: str = "monthly",
) -> SleevePortfolioResult:
    """
    Combine strategy equity curves into a fixed-weight sleeve book.

    Uses daily returns of each member; rebalances to target weights on month starts
    (paper approximation — not simultaneous multi-strategy execution costs).
    """
    wmap = dict(weights_by_kind or DEFAULT_SLEEVE_WEIGHTS)
    # pick one strategy per kind (highest final equity among that kind)
    by_kind: Dict[str, Any] = {}
    for r in results:
        kind = getattr(r, "kind", None) or (r.get("kind") if isinstance(r, Mapping) else None)
        if kind is None or kind not in wmap:
            continue
        curve = getattr(r, "equity_curve", None) or (r.get("equity_curve") if isinstance(r, Mapping) else None)
        if not curve:
            continue
        prev = by_kind.get(str(kind))
        fe = float(getattr(r, "final_equity", 0) or (r.get("final_equity") if isinstance(r, Mapping) else 0) or 0)
        if prev is None:
            by_kind[str(kind)] = r
        else:
            pfe = float(getattr(prev, "final_equity", 0) or 0)
            if fe > pfe:
                by_kind[str(kind)] = r

    if not by_kind:
        return SleevePortfolioResult(
            final_equity=float(capital0),
            total_return=0.0,
            max_dd=0.0,
            weights=dict(wmap),
            notes=["no_member_curves"],
        )

    # Normalize weights over available kinds
    avail_w = {k: float(wmap[k]) for k in by_kind if wmap.get(k, 0) > 0}
    s = sum(avail_w.values()) or 1.0
    avail_w = {k: v / s for k, v in avail_w.items()}

    series_map: Dict[str, pd.Series] = {}
    members: List[str] = []
    for kind, r in by_kind.items():
        curve = getattr(r, "equity_curve", None) or r.get("equity_curve")  # type: ignore[union-attr]
        ser = _curve_to_series(curve or [])
        if ser.empty:
            continue
        # convert to daily returns
        rets = ser.pct_change().fillna(0.0)
        series_map[kind] = rets
        sid = getattr(r, "strategy_id", None) or (r.get("strategy_id") if isinstance(r, Mapping) else kind)
        members.append(f"{kind}:{sid}")

    if not series_map:
        return SleevePortfolioResult(
            final_equity=float(capital0),
            total_return=0.0,
            max_dd=0.0,
            weights=avail_w,
            notes=["empty_series"],
        )

    aligned = pd.DataFrame(series_map).fillna(0.0)
    # monthly rebalance: hold fixed weights within month
    eq = float(capital0)
    peak = eq
    max_dd = 0.0
    curve_out: List[Dict[str, Any]] = []
    weights = np.array([avail_w.get(c, 0.0) for c in aligned.columns], dtype=float)
    last_month: Optional[Tuple[int, int]] = None

    for ts, row in aligned.iterrows():
        month_key = (int(ts.year), int(ts.month))
        if rebalance == "monthly" and month_key != last_month:
            # rebalance at month start (weights already target)
            last_month = month_key
        day_ret = float(np.dot(weights, row.values.astype(float)))
        eq *= 1.0 + day_ret
        peak = max(peak, eq)
        dd = eq / peak - 1.0 if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
        curve_out.append({"date": ts.date().isoformat() if hasattr(ts, "date") else str(ts), "equity": eq})

    return SleevePortfolioResult(
        final_equity=eq,
        total_return=eq / float(capital0) - 1.0 if capital0 > 0 else 0.0,
        max_dd=max_dd,
        weights=avail_w,
        members=members,
        equity_curve=curve_out,
        notes=[f"rebalance={rebalance}", "paper sleeve — not simultaneous live multi-leg book"],
    )
