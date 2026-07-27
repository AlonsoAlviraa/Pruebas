"""Extended risk / performance metrics (MET-01).

Complements trad_research.metrics.equity_metrics with Sortino MAR variants,
ulcer, tail ratio, CVaR, consecutive losses, expectancy helpers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


def _as_float_array(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def downside_deviation(returns: ArrayLike, mar: float = 0.0) -> float:
    """Std of returns strictly below MAR (population ddof=0 on downside sample)."""
    r = _as_float_array(returns)
    if r.size == 0:
        return 0.0
    down = r[r < mar]
    if down.size == 0:
        return 0.0
    return float(np.std(down, ddof=0))


def sortino_ratio(
    returns: ArrayLike,
    *,
    mar: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized Sortino: (mean - mar) / downside_dev * sqrt(N)."""
    r = _as_float_array(returns)
    if r.size < 2:
        return 0.0
    mean = float(np.mean(r))
    dd = downside_deviation(r, mar=mar)
    if dd < 1e-12:
        # No downside: large positive if edge above MAR, else 0
        return 100.0 if mean > mar else 0.0
    return float((mean - mar) / dd * np.sqrt(periods_per_year))


def ulcer_index(equity: ArrayLike) -> float:
    """RMS of percentage drawdowns from peak."""
    eq = _as_float_array(equity)
    if eq.size < 2 or np.any(eq <= 0):
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd_pct = (eq - peak) / peak * 100.0
    return float(np.sqrt(np.mean(dd_pct**2)))


def max_drawdown(equity: ArrayLike) -> float:
    eq = _as_float_array(equity)
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(np.min(dd))


def tail_ratio(returns: ArrayLike, q: float = 0.05) -> float:
    """p(1-q) / |p(q)| of returns (default p95 / |p5|)."""
    r = _as_float_array(returns)
    if r.size < 10:
        return 0.0
    lo = float(np.quantile(r, q))
    hi = float(np.quantile(r, 1.0 - q))
    if abs(lo) < 1e-12:
        return 0.0
    return float(hi / abs(lo))


def cvar(returns: ArrayLike, alpha: float = 0.05) -> float:
    """Expected shortfall: mean of returns at or below alpha quantile."""
    r = _as_float_array(returns)
    if r.size < 5:
        return 0.0
    thr = float(np.quantile(r, alpha))
    tail = r[r <= thr]
    if tail.size == 0:
        return thr
    return float(np.mean(tail))


def max_consecutive_losses(trade_pnls: ArrayLike) -> int:
    """Longest streak of strictly negative trade PnLs."""
    p = _as_float_array(trade_pnls)
    if p.size == 0:
        return 0
    best = cur = 0
    for x in p:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def expectancy(trade_pnls: ArrayLike) -> float:
    p = _as_float_array(trade_pnls)
    if p.size == 0:
        return 0.0
    return float(np.mean(p))


def profit_factor_from_pnls(trade_pnls: ArrayLike) -> float:
    p = _as_float_array(trade_pnls)
    if p.size == 0:
        return 0.0
    gp = float(p[p > 0].sum()) if np.any(p > 0) else 0.0
    gl = float(-p[p < 0].sum()) if np.any(p < 0) else 0.0
    if gl < 1e-12:
        return 999.0 if gp > 0 else 0.0
    return gp / gl


def omega_ratio(returns: ArrayLike, mar: float = 0.0) -> float:
    r = _as_float_array(returns)
    if r.size == 0:
        return 0.0
    gains = r[r > mar] - mar
    losses = mar - r[r < mar]
    den = float(losses.sum()) if losses.size else 0.0
    num = float(gains.sum()) if gains.size else 0.0
    if den < 1e-12:
        return 999.0 if num > 0 else 0.0
    return num / den


def cagr_from_equity(equity: ArrayLike, years: Optional[float] = None) -> float:
    eq = _as_float_array(equity)
    if eq.size < 2 or eq[0] <= 0 or eq[-1] <= 0:
        return 0.0
    if years is None or years <= 0:
        years = max(len(eq) - 1, 1) / 252.0
    years = max(float(years), 1.0 / 365.25)
    return float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)


def sharpe_from_returns(
    returns: ArrayLike, *, periods_per_year: float = 252.0
) -> float:
    r = _as_float_array(returns)
    if r.size < 2:
        return 0.0
    vol = float(np.std(r, ddof=0))
    if vol < 1e-12:
        return 0.0
    return float(np.mean(r) / vol * np.sqrt(periods_per_year))


@dataclass
class ExtendedRiskReport:
    sortino: float
    sortino_mar0: float
    downside_dev: float
    ulcer: float
    tail_ratio: float
    cvar_5: float
    max_drawdown: float
    calmar: float
    cagr: float
    sharpe: float
    omega: float
    max_consecutive_losses: int
    expectancy: float
    profit_factor: float
    n_obs: int
    n_trades: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extended_risk_from_equity(
    equity: ArrayLike,
    *,
    trade_pnls: Optional[ArrayLike] = None,
    years: Optional[float] = None,
    mar: float = 0.0,
) -> ExtendedRiskReport:
    eq = _as_float_array(equity)
    if eq.size < 2:
        return ExtendedRiskReport(
            sortino=0.0,
            sortino_mar0=0.0,
            downside_dev=0.0,
            ulcer=0.0,
            tail_ratio=0.0,
            cvar_5=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            cagr=0.0,
            sharpe=0.0,
            omega=0.0,
            max_consecutive_losses=0,
            expectancy=0.0,
            profit_factor=0.0,
            n_obs=0,
            n_trades=0,
        )
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    cagr = cagr_from_equity(eq, years=years)
    mdd = max_drawdown(eq)
    sh = sharpe_from_returns(rets)
    so = sortino_ratio(rets, mar=mar)
    so0 = sortino_ratio(rets, mar=0.0)
    cal = (cagr / abs(mdd)) if mdd < -1e-12 else 0.0
    pnls = _as_float_array(trade_pnls) if trade_pnls is not None else np.array([])
    return ExtendedRiskReport(
        sortino=so,
        sortino_mar0=so0,
        downside_dev=downside_deviation(rets, mar=mar),
        ulcer=ulcer_index(eq),
        tail_ratio=tail_ratio(rets),
        cvar_5=cvar(rets, 0.05),
        max_drawdown=mdd,
        calmar=cal,
        cagr=cagr,
        sharpe=sh,
        omega=omega_ratio(rets, mar=mar),
        max_consecutive_losses=max_consecutive_losses(pnls) if pnls.size else 0,
        expectancy=expectancy(pnls) if pnls.size else 0.0,
        profit_factor=profit_factor_from_pnls(pnls) if pnls.size else 0.0,
        n_obs=int(rets.size),
        n_trades=int(pnls.size),
    )
