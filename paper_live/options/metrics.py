"""Portfolio metrics for paper options equity curves (research only)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def equity_from_curve(curve: Sequence[Dict[str, Any]]) -> List[float]:
    """Extract equity series from curve dicts ``{date, equity}``."""
    out: List[float] = []
    for row in curve:
        try:
            out.append(float(row["equity"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def daily_returns(equity: Sequence[float]) -> List[float]:
    """Simple daily returns from equity marks (skip non-positive)."""
    rets: List[float] = []
    for i in range(1, len(equity)):
        prev = float(equity[i - 1])
        cur = float(equity[i])
        if prev <= 0:
            continue
        rets.append(cur / prev - 1.0)
    return rets


def session_returns_from_curve(curve: Sequence[Dict[str, Any]]) -> List[float]:
    """
    Consecutive-session returns for tail metrics.

    Skips returns into a row marked ``session_gap=True`` so multi-session jumps
    (missing bars) are not treated as a single daily return for CVaR / worst_day.
    """
    rets: List[float] = []
    prev_eq: Optional[float] = None
    for row in curve:
        try:
            eq = float(row["equity"])
        except (KeyError, TypeError, ValueError):
            continue
        gap = bool(row.get("session_gap"))
        if prev_eq is not None and prev_eq > 0 and not gap:
            rets.append(eq / prev_eq - 1.0)
        prev_eq = eq
    return rets


def max_drawdown(equity: Sequence[float]) -> float:
    """Max drawdown as a negative fraction (e.g. -0.12). Empty → 0.0."""
    if not equity:
        return 0.0
    peak = float(equity[0])
    worst = 0.0
    for x in equity:
        v = float(x)
        peak = max(peak, v)
        if peak > 0:
            dd = v / peak - 1.0
            worst = min(worst, dd)
    return float(worst)


def cvar(
    returns: Sequence[float],
    *,
    alpha: float = 0.05,
) -> Optional[float]:
    """
    Conditional Value-at-Risk (expected shortfall) of the left tail.

    ``alpha`` is the tail mass (e.g. 0.05 = average of worst 5% daily returns).
    Returns a **negative** number when losses dominate; None if insufficient data
    or ``alpha`` not in (0, 1).

    Estimator: sort returns ascending; take the ``k = max(1, ceil(n * alpha))``
    worst observations and average them (conservative small-sample ES).
    """
    if not returns or alpha <= 0 or alpha >= 1:
        return None
    arr = sorted(float(r) for r in returns)
    n = len(arr)
    # ceil for conservative tail mass on small samples
    k = max(1, int(-(-n * alpha // 1)))  # ceil without numpy
    k = min(k, n)
    tail = arr[:k]
    if not tail:
        return None
    return float(sum(tail) / len(tail))


def calmar_like(
    total_return: float,
    max_dd: float,
    *,
    years: Optional[float] = None,
) -> Optional[float]:
    """
    Calmar-ish ratio: annualized_return / |max_dd|.

    If ``years`` is None, uses total_return (not annualized) / |max_dd|.
    """
    dd = abs(float(max_dd))
    if dd < 1e-12:
        return None
    if years is not None and years > 0:
        tr = float(total_return)
        ann = (1.0 + tr) ** (1.0 / years) - 1.0 if tr > -1.0 else tr
        return float(ann / dd)
    return float(total_return) / dd


def years_from_curve(curve: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Estimate years from first/last ISO dates in curve."""
    if len(curve) < 2:
        return None
    try:
        import pandas as pd

        d0 = pd.Timestamp(curve[0]["date"])
        d1 = pd.Timestamp(curve[-1]["date"])
        days = max((d1 - d0).days, 1)
        return float(days) / 365.25
    except Exception:
        return None


def worst_month_return(curve: Sequence[Dict[str, Any]]) -> Optional[float]:
    """
    Worst calendar-month simple return from equity curve dates.

    Uses last equity of each (year, month) vs previous month's last equity.
    """
    if len(curve) < 2:
        return None
    try:
        import pandas as pd

        rows = []
        for row in curve:
            try:
                rows.append((pd.Timestamp(row["date"]), float(row["equity"])))
            except (KeyError, TypeError, ValueError):
                continue
        if len(rows) < 2:
            return None
        s = pd.Series(
            [e for _, e in rows],
            index=pd.DatetimeIndex([d for d, _ in rows]),
        ).sort_index()
        monthly = s.resample("ME").last().dropna()
        if len(monthly) < 2:
            monthly = s.resample("MS").last().dropna()
        if len(monthly) < 2:
            return None
        mrets = monthly.pct_change().dropna()
        if mrets.empty:
            return None
        return float(mrets.min())
    except Exception:
        return None


def metrics_from_curve(
    curve: Sequence[Dict[str, Any]],
    *,
    capital0: float,
    cvar_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Aggregate return / DD / CVaR / Calmar / worst month from an equity curve."""
    eq = equity_from_curve(curve)
    if not eq:
        return {
            "final_equity": float(capital0),
            "total_return": 0.0,
            "max_dd": 0.0,
            "cvar_5pct": None,
            "calmar_like": None,
            "n_days": 0,
            "worst_day": None,
            "worst_month": None,
        }
    # Prefer gap-aware session returns for tail stats when curve carries flags
    rets = session_returns_from_curve(curve)
    if not rets and len(eq) > 1:
        rets = daily_returns(eq)
    final = float(eq[-1])
    tr = final / float(capital0) - 1.0 if capital0 > 0 else 0.0
    mdd = max_drawdown(eq)
    yrs = years_from_curve(curve)
    return {
        "final_equity": final,
        "total_return": float(tr),
        "max_dd": float(mdd),
        "cvar_5pct": cvar(rets, alpha=cvar_alpha),
        "calmar_like": calmar_like(tr, mdd, years=yrs),
        "n_days": len(eq),
        "worst_day": float(min(rets)) if rets else None,
        "worst_month": worst_month_return(curve),
    }
