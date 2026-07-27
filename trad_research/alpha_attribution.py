"""Alpha / residual attribution helpers (STR-01, STR-04).

Layers:
  A0 — strategy metrics vs SPY/QQQ (vanity / secondary)
  A1 — cash-aware invested benchmark w·bench + (1−w)·cash
  A2 — residual vs style clone (primary promotion gate)
  A3 — residual vs PIT EW (honesty / survivorship)

Does not claim financial advice; pure research math on equity series.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.metrics import PerformanceReport, equity_metrics


# Pre-registered diagnostic thresholds (design 2026-07-23)
P1_STYLE_CAPTURE_MIN = 0.60  # fraction of excess vs SPY explained by best clone
P1_RESIDUAL_SHARPE_MAX = 0.15  # baseline sharpe − clone sharpe ≤ this → P1✓
P2_PIT_EW_EXCESS_MAX = 0.0  # excess vs PIT EW < 0 confirms P2
# Clone path metrics beyond these are treated as data/engine pathology (not P1 evidence)
P1_CLONE_CAGR_ABS_MAX = 1.0  # |clone CAGR| > 100% → pathology_suspect
P1_CAPTURE_ABS_MAX = 5.0  # |style capture| > 5× baseline excess → not interpretable


@dataclass
class ResidualReport:
    """Compare strategy equity to a style / fair benchmark series."""

    strategy_cagr: float
    strategy_sharpe: float
    strategy_mdd: float
    bench_cagr: float
    bench_sharpe: float
    bench_mdd: float
    excess_cagr: float
    residual_sharpe: float  # Sharpe of (strat_ret − bench_ret) daily
    style_capture_of_spy_excess: Optional[float] = None
    label: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def align_equity_pair(
    strategy: pd.Series,
    benchmark: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Inner-join on **calendar day** (UTC normalize), dropna, rebase to 1.0."""
    s = strategy.dropna().astype(float)
    b = benchmark.dropna().astype(float)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.to_datetime(b.index, utc=True, errors="coerce")
    try:
        s.index = s.index.tz_convert("UTC") if s.index.tz is not None else s.index.tz_localize("UTC")
    except Exception:
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    try:
        b.index = b.index.tz_convert("UTC") if b.index.tz is not None else b.index.tz_localize("UTC")
    except Exception:
        b.index = pd.to_datetime(b.index, utc=True, errors="coerce")
    # Collapse to date so 00:00 vs 14:30 same-day series join
    s.index = pd.DatetimeIndex(s.index).normalize()
    b.index = pd.DatetimeIndex(b.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    b = b[~b.index.duplicated(keep="last")].sort_index()
    joined = pd.concat([s.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
    if joined.empty:
        return s.iloc[0:0], b.iloc[0:0]
    s2 = joined["s"] / float(joined["s"].iloc[0])
    b2 = joined["b"] / float(joined["b"].iloc[0])
    return s2, b2


def daily_returns(equity: pd.Series) -> pd.Series:
    eq = equity.dropna().astype(float)
    return eq.pct_change().dropna()


def residual_sharpe(strategy: pd.Series, benchmark: pd.Series) -> float:
    s, b = align_equity_pair(strategy, benchmark)
    if len(s) < 5:
        return 0.0
    rs = daily_returns(s)
    rb = daily_returns(b)
    common = rs.index.intersection(rb.index)
    if len(common) < 5:
        return 0.0
    resid = rs.reindex(common) - rb.reindex(common)
    vol = float(resid.std())
    if vol < 1e-12:
        return 0.0
    return float(resid.mean() / vol * np.sqrt(252))


def mean_invested_weight(position_notional: pd.Series, equity: pd.Series) -> float:
    """Average fraction of equity invested (w). Clipped to [0, 2]."""
    eq = equity.replace(0, np.nan)
    w = (position_notional / eq).replace([np.inf, -np.inf], np.nan).dropna()
    if w.empty:
        return 0.0
    return float(np.clip(w.mean(), 0.0, 2.0))


def cash_aware_benchmark(
    bench_equity: pd.Series,
    w: float,
    *,
    start_value: float = 1.0,
) -> pd.Series:
    """w · bench + (1−w) · cash (cash return 0). Causal on same dates as bench."""
    b = bench_equity.dropna().astype(float)
    if b.empty:
        return b
    rets = b.pct_change().fillna(0.0)
    w = float(np.clip(w, 0.0, 2.0))
    blend_rets = w * rets  # cash leg 0
    out = (1.0 + blend_rets).cumprod() * start_value
    out.iloc[0] = start_value
    return out


def compare_to_benchmark(
    strategy_equity: pd.Series,
    bench_equity: pd.Series,
    *,
    start_equity: float = 100_000.0,
    label: str = "",
    spy_excess_strategy: Optional[float] = None,
    spy_excess_bench: Optional[float] = None,
) -> ResidualReport:
    s, b = align_equity_pair(strategy_equity, bench_equity)
    if s.empty:
        return ResidualReport(
            strategy_cagr=0.0,
            strategy_sharpe=0.0,
            strategy_mdd=0.0,
            bench_cagr=0.0,
            bench_sharpe=0.0,
            bench_mdd=0.0,
            excess_cagr=0.0,
            residual_sharpe=0.0,
            label=label,
        )
    # Scale to start_equity for metrics helper
    s_eq = s * start_equity
    b_eq = b * start_equity
    rep_s = equity_metrics(s_eq, start_equity)
    rep_b = equity_metrics(b_eq, start_equity)
    excess = float(rep_s.cagr - rep_b.cagr)
    r_sh = residual_sharpe(s_eq, b_eq)
    capture = None
    if (
        spy_excess_strategy is not None
        and spy_excess_bench is not None
        and abs(spy_excess_strategy) > 1e-9
    ):
        # How much of strategy's excess-vs-SPY is matched by the style bench excess-vs-SPY
        capture = float(np.clip(spy_excess_bench / spy_excess_strategy, -2.0, 2.0))
    return ResidualReport(
        strategy_cagr=float(rep_s.cagr),
        strategy_sharpe=float(rep_s.sharpe),
        strategy_mdd=float(rep_s.max_drawdown),
        bench_cagr=float(rep_b.cagr),
        bench_sharpe=float(rep_b.sharpe),
        bench_mdd=float(rep_b.max_drawdown),
        excess_cagr=excess,
        residual_sharpe=r_sh,
        style_capture_of_spy_excess=capture,
        label=label,
    )


def clone_metrics_pathology(
    *,
    clone_cagr: Optional[float] = None,
    clone_excess_vs_spy: Optional[float] = None,
    style_capture: Optional[float] = None,
    clone_cagr_abs_max: float = P1_CLONE_CAGR_ABS_MAX,
    capture_abs_max: float = P1_CAPTURE_ABS_MAX,
) -> Dict[str, Any]:
    """Flag absurd clone metrics that must not drive P1 confirmation.

    E.g. style_ew early-window CAGR ≫ 100% with multi-hundred-percent annual
    legs is data/engine pathology, not "style explains the edge."
    """
    reasons: List[str] = []
    if clone_cagr is not None and abs(float(clone_cagr)) > float(clone_cagr_abs_max):
        reasons.append(f"|clone_cagr|={abs(float(clone_cagr)):.2f}>{clone_cagr_abs_max}")
    # Only flag *positive* absurd excess (collapsing styles with large negative excess are OK)
    if clone_excess_vs_spy is not None and float(clone_excess_vs_spy) > float(clone_cagr_abs_max):
        reasons.append(
            f"clone_excess_vs_spy={float(clone_excess_vs_spy):.2f}>{clone_cagr_abs_max}"
        )
    # Capture only pathology when absurdly large *positive* (not large negative ratios)
    if style_capture is not None and float(style_capture) > float(capture_abs_max):
        reasons.append(f"style_capture={float(style_capture):.2f}>{capture_abs_max}")
    return {
        "pathology_suspect": bool(reasons),
        "pathology_reasons": reasons,
        "thresholds": {
            "clone_cagr_abs_max": clone_cagr_abs_max,
            "capture_abs_max": capture_abs_max,
            "pathology_positive_capture_only": True,
        },
    }


def confirm_p1_style_confusion(
    *,
    baseline_excess_vs_spy: float,
    clone_excess_vs_spy: float,
    baseline_sharpe: float,
    clone_sharpe: float,
    capture_min: float = P1_STYLE_CAPTURE_MIN,
    residual_sharpe_max: float = P1_RESIDUAL_SHARPE_MAX,
    clone_cagr: Optional[float] = None,
) -> Dict[str, Any]:
    """P1✓ if clone captures most *positive* SPY excess or residual sharpe gap is tiny.

    Capture rule is **degenerate** when baseline_excess_vs_spy ≤ 0 (both series can
    underperform SPY and yield a large ratio that is not "style explains alpha").
    In that case by_capture is forced False and ``capture_degenerate=True``.

    Pathological clone CAGR/capture (e.g. multi-hundred % style equity) never
    confirms P1 via capture or residual-gap on that clone.
    """
    capture = None
    capture_degenerate = float(baseline_excess_vs_spy) <= 0.0
    if not capture_degenerate and abs(baseline_excess_vs_spy) > 1e-9:
        capture = clone_excess_vs_spy / baseline_excess_vs_spy
    elif abs(baseline_excess_vs_spy) > 1e-9:
        # Still report the raw ratio for diagnostics, but do not use for confirmation
        capture = clone_excess_vs_spy / baseline_excess_vs_spy
    path = clone_metrics_pathology(
        clone_cagr=clone_cagr,
        clone_excess_vs_spy=clone_excess_vs_spy,
        style_capture=capture,
    )
    residual_sh_gap = baseline_sharpe - clone_sharpe
    by_capture = (
        (not capture_degenerate)
        and (not path["pathology_suspect"])
        and capture is not None
        and capture >= capture_min
    )
    by_gap = (not path["pathology_suspect"]) and residual_sh_gap <= residual_sharpe_max
    return {
        "problem": "P1",
        "confirmed": bool(by_capture or by_gap),
        "style_capture": capture,
        "capture_degenerate": capture_degenerate,
        "pathology_suspect": path["pathology_suspect"],
        "pathology_reasons": path["pathology_reasons"],
        "residual_sharpe_gap": residual_sh_gap,
        "by_capture": by_capture,
        "by_gap": by_gap,
        "thresholds": {
            "capture_min": capture_min,
            "residual_sharpe_max": residual_sharpe_max,
            "capture_requires_positive_baseline_excess": True,
            "clone_cagr_abs_max": P1_CLONE_CAGR_ABS_MAX,
            "capture_abs_max": P1_CAPTURE_ABS_MAX,
        },
    }


def confirm_p2_unfair_spy_bench(
    excess_vs_pit_ew: float,
    *,
    max_excess: float = P2_PIT_EW_EXCESS_MAX,
) -> Dict[str, Any]:
    """P2✓ if strategy does not beat PIT EW of its own universe."""
    return {
        "problem": "P2",
        "confirmed": bool(excess_vs_pit_ew < max_excess),
        "excess_vs_pit_ew": excess_vs_pit_ew,
        "threshold": max_excess,
    }


def factor_proxy_regression(
    strategy_returns: pd.Series,
    factor_returns: Mapping[str, pd.Series],
) -> Dict[str, Any]:
    """OLS residual alpha using numpy (no statsmodels dependency).

    factor_returns keys e.g. mkt, qqq_mkt, iwm_mkt (already excess or raw daily).
    Returns alpha (daily), annualized alpha, betas, r2, n.
    """
    y = strategy_returns.dropna().astype(float)
    cols: List[str] = []
    mats: List[pd.Series] = []
    for k, s in factor_returns.items():
        cols.append(k)
        mats.append(s.astype(float).rename(k))
    if not mats:
        return {"alpha_daily": 0.0, "alpha_ann": 0.0, "betas": {}, "r2": 0.0, "n": 0}
    Xdf = pd.concat(mats, axis=1)
    joined = pd.concat([y.rename("y"), Xdf], axis=1, join="inner").dropna()
    n = len(joined)
    if n < len(cols) + 5:
        return {"alpha_daily": 0.0, "alpha_ann": 0.0, "betas": {}, "r2": 0.0, "n": n}
    yv = joined["y"].to_numpy(dtype=float)
    X = joined[cols].to_numpy(dtype=float)
    Xd = np.column_stack([np.ones(n), X])
    try:
        beta_hat, _, _, _ = np.linalg.lstsq(Xd, yv, rcond=None)
    except np.linalg.LinAlgError:
        return {"alpha_daily": 0.0, "alpha_ann": 0.0, "betas": {}, "r2": 0.0, "n": n}
    alpha = float(beta_hat[0])
    betas = {c: float(beta_hat[i + 1]) for i, c in enumerate(cols)}
    yhat = Xd @ beta_hat
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    return {
        "alpha_daily": alpha,
        "alpha_ann": alpha * 252.0,
        "betas": betas,
        "r2": r2,
        "n": n,
    }


def rank_problems_by_false_alpha(
    confirmations: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Order confirmed problems; attach heuristic severity weights."""
    weights = {
        "P1": 0.30,
        "P2": 0.20,
        "P3": 0.20,
        "P4": 0.10,
        "P5": 0.10,
        "P6": 0.05,
        "P7": 0.05,
        "P8": 0.05,
    }
    rows: List[Dict[str, Any]] = []
    for c in confirmations:
        pid = str(c.get("problem", "?"))
        conf = bool(c.get("confirmed", False))
        w = weights.get(pid, 0.05)
        rows.append(
            {
                "problem": pid,
                "confirmed": conf,
                "severity_weight": w if conf else 0.0,
                "detail": {k: v for k, v in c.items() if k not in ("problem", "confirmed")},
            }
        )
    rows.sort(key=lambda r: (-r["severity_weight"], r["problem"]))
    return rows


def promotion_gates_residual(
    residual_vs_style: ResidualReport,
    residual_vs_pit_ew: Optional[ResidualReport] = None,
    *,
    require_early_and_modern: bool = False,
    early_excess: Optional[float] = None,
    modern_excess: Optional[float] = None,
    engine_matched: bool = True,
    diagnostic_only: bool = False,
) -> Dict[str, Any]:
    """R1/R2-style gates for ALPHA-PORTABLE promotion (not STYLE-US).

    R2 defaults to **not_evaluated** (not True) when PIT EW residual is missing —
    pass_core cannot succeed without an evaluated R2.
    When ``engine_matched`` is False or ``diagnostic_only`` is True, pass_core is
    forced False (cross-engine residual is diagnostic only).
    """
    r1 = residual_vs_style.excess_cagr > 0.0
    r1_dual = None
    if require_early_and_modern:
        if early_excess is not None and modern_excess is not None:
            r1_dual = early_excess > 0.0 and modern_excess > 0.0
            r1 = r1 and r1_dual
        else:
            r1_dual = False
            r1 = False  # dual-window required but incomplete
    if residual_vs_pit_ew is not None:
        r2: Any = residual_vs_pit_ew.excess_cagr >= -0.01
        r2_status = "evaluated"
    else:
        r2 = False
        r2_status = "not_evaluated"
    incomplete = residual_vs_pit_ew is None or diagnostic_only or (not engine_matched)
    if require_early_and_modern and (
        early_excess is None or modern_excess is None
    ):
        incomplete = True
    pass_core = bool(r1 and r2 and not incomplete and not diagnostic_only and engine_matched)
    return {
        "R1_residual_style": r1,
        "R1_dual_window": r1_dual,
        "R2_pit_ew": r2,
        "R2_status": r2_status,
        "pass_core": pass_core,
        "incomplete": incomplete,
        "diagnostic_only": bool(diagnostic_only),
        "engine_matched": bool(engine_matched),
        "excess_vs_style": residual_vs_style.excess_cagr,
        "excess_vs_pit_ew": None
        if residual_vs_pit_ew is None
        else residual_vs_pit_ew.excess_cagr,
        "note": (
            "R1 modern-only unless require_early_and_modern=True with both excesses; "
            "R2 missing → not_evaluated, pass_core False"
        ),
    }
