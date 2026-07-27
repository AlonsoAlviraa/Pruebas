"""Pre-registered universe_limit screen/confirm robustness study.

Strategy: turbo_highvol_minalloc only.
Grid (fixed, do not expand after seeing results): {40, 50, 60, 80} — no 54.
Screen OOS 2010–2017; confirm OOS 2018–2025; full path = stitch both.
Gates (path metrics): CAGR > 10%, max_drawdown >= −65%, n_trades >= 50.
Banlists: none. Paper freeze: unchanged.

Research only. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

# --- Pre-registered constants (do not change after results) ---
COMMISSION = 0.001
SLIPPAGE = 0.0005
GATE_CAGR = 0.10
GATE_MDD = -0.65  # max drawdown must be >= this (not deeper)
GATE_TRADES = 50
DEFAULT_LIMITS: Tuple[int, ...] = (40, 50, 60, 80)
DEFAULT_STRATEGY = "turbo_highvol_minalloc"
DEFAULT_SCREEN = (2010, 2017)
DEFAULT_CONFIRM = (2018, 2025)
DEFAULT_MIN_TRAIN_ROWS = 1500
PREREG_NOTE = (
    "Pre-registered grid {40,50,60,80}; no limit=54; banlists none; "
    "paper freeze not auto-changed."
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def parse_limits(s: str) -> List[int]:
    """Parse comma-separated positive ints; preserve order, drop duplicates."""
    out: List[int] = []
    seen = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError(f"universe_limit must be positive, got {v}")
        if v not in seen:
            seen.add(v)
            out.append(v)
    if not out:
        raise ValueError("empty limits list")
    return out


def validate_limits(
    limits: Sequence[int],
    *,
    allow_unregistered: bool = False,
    prereg: Sequence[int] = DEFAULT_LIMITS,
    banned: Sequence[int] = (54,),
) -> List[int]:
    """Enforce pre-registered grid; permanently ban accidental limit=54.

    Non-prereg limits require ``allow_unregistered=True``. Limit 54 is always
    rejected (longpath artifact that motivated this study).
    """
    lims = [int(x) for x in limits]
    banned_set = {int(x) for x in banned}
    hit_ban = sorted({x for x in lims if x in banned_set})
    if hit_ban:
        raise ValueError(
            f"Banned universe_limit(s) {hit_ban} (accidental longpath artifact). "
            f"Pre-registered grid is {list(prereg)}. Do not re-introduce 54."
        )
    allowed = {int(x) for x in prereg}
    extra = sorted({x for x in lims if x not in allowed})
    if extra and not allow_unregistered:
        raise ValueError(
            f"limits {extra} not in pre-registered grid {list(prereg)}. "
            "Pass --allow-unregistered-limits for ad-hoc research only."
        )
    return lims


def resolve_ticker_file(
    preferred: Optional[Path],
    *,
    root: Path,
    min_n: int = 40,
) -> Path:
    """Prefer universe_longhist2010_pass.txt if exists and n>=min_n, else longhist100.

    If ``preferred`` is an explicit non-default path, use it as-is.
    """
    default_longhist = (root / "universe_longhist100.txt").resolve()
    passers = root / "universe_longhist2010_pass.txt"

    if preferred is None:
        preferred = root / "universe_longhist100.txt"
    preferred = Path(preferred)
    if not preferred.is_absolute():
        preferred = root / preferred

    try:
        pref_res = preferred.resolve()
    except Exception:
        pref_res = preferred

    # Only auto-swap when user asked for default longhist100 (or None)
    if pref_res == default_longhist or preferred.name == "universe_longhist100.txt":
        if passers.is_file():
            n_pass = count_tickers(passers)
            if n_pass >= min_n:
                return passers
    return preferred


def count_tickers(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(
        1
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )


def _metric_float(metrics: Dict[str, Any], key: str, default: float) -> float:
    """Read float metric with explicit None-default (0.0 is a valid value)."""
    v = metrics.get(key, None)
    if v is None:
        return float(default)
    return float(v)


def _metric_int(metrics: Dict[str, Any], key: str, default: int) -> int:
    v = metrics.get(key, None)
    if v is None:
        return int(default)
    return int(v)


def apply_path_gates(
    metrics: Dict[str, Any],
    *,
    gate_cagr: float = GATE_CAGR,
    gate_mdd: float = GATE_MDD,
    gate_trades: int = GATE_TRADES,
) -> Dict[str, Any]:
    """Apply pre-registered path gates to a metrics dict.

    Uses explicit None checks so max_drawdown=0.0 (no drawdown) is not treated
    as missing (``0.0 or -1.0`` would incorrectly fail the MDD gate).
    """
    cagr = _metric_float(metrics, "cagr", 0.0)
    mdd = _metric_float(metrics, "max_drawdown", -1.0)
    n = _metric_int(metrics, "n_trades", 0)
    g_cagr = cagr > gate_cagr
    g_mdd = mdd >= gate_mdd
    g_tr = n >= gate_trades
    return {
        "cagr_ok": g_cagr,
        "mdd_ok": g_mdd,
        "trades_ok": g_tr,
        "pass": bool(g_cagr and g_mdd and g_tr),
        "thresholds": {
            "cagr_gt": gate_cagr,
            "mdd_ge": gate_mdd,
            "n_trades_ge": gate_trades,
        },
    }


def stitch_equity(seg_a: pd.Series, seg_b: pd.Series) -> pd.Series:
    """Stitch two equity segments with capital continuity (screen then confirm)."""
    parts: List[pd.Series] = []
    prev_end: Optional[float] = None
    for seg in (seg_a, seg_b):
        if seg is None:
            continue
        s = seg.dropna().astype(float)
        if s.empty:
            continue
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
        s = s[~s.index.duplicated(keep="last")].dropna().sort_index()
        if s.empty:
            continue
        if prev_end is not None and float(s.iloc[0]) != 0:
            s = s * (prev_end / float(s.iloc[0]))
        parts.append(s)
        prev_end = float(s.iloc[-1])
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def capital_continuity_scale(eq_a: pd.Series, eq_b: pd.Series) -> float:
    """Scale factor so eq_b starts at eq_a end capital (for trade PnL alignment)."""
    a = eq_a.dropna().astype(float) if eq_a is not None else pd.Series(dtype=float)
    b = eq_b.dropna().astype(float) if eq_b is not None else pd.Series(dtype=float)
    if a.empty or b.empty:
        return 1.0
    start_b = float(b.iloc[0])
    if start_b == 0.0:
        return 1.0
    return float(a.iloc[-1]) / start_b


def stitch_trades(
    tr_a: pd.DataFrame,
    tr_b: pd.DataFrame,
    *,
    scale_b: float = 1.0,
) -> pd.DataFrame:
    """Concat trades; scale confirm net_profit to screen ending capital.

    Full-path **gates** use equity CAGR/MDD and n_trades count. Scaled
    ``net_profit`` keeps secondary trade-dollar stats capital-comparable
    after independent WF restarts.
    """
    parts: List[pd.DataFrame] = []
    if isinstance(tr_a, pd.DataFrame) and not tr_a.empty:
        parts.append(tr_a.copy())
    if isinstance(tr_b, pd.DataFrame) and not tr_b.empty:
        b = tr_b.copy()
        if "net_profit" in b.columns and float(scale_b) != 1.0:
            b["net_profit"] = (
                pd.to_numeric(b["net_profit"], errors="coerce") * float(scale_b)
            )
            b["pnl_capital_scale"] = float(scale_b)
        parts.append(b)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def missing_oos_years(
    equity: Optional[pd.Series],
    first: int,
    last: int,
    *,
    year_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[int]:
    """Return expected OOS years missing from equity index or year_results."""
    expected = set(range(int(first), int(last) + 1))
    present: set = set()
    if year_results is not None:
        for r in year_results:
            if isinstance(r, dict) and r.get("year") is not None:
                present.add(int(r["year"]))
    elif equity is not None and isinstance(equity, pd.Series) and not equity.empty:
        eq = equity.dropna()
        if not eq.empty:
            idx = pd.to_datetime(eq.index, utc=True, errors="coerce")
            present = {int(y) for y in pd.Index(idx).year.dropna().unique()}
    return sorted(expected - present)


@dataclass
class ArmWindowResult:
    """Metrics + gates for one window (screen / confirm / full)."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    gates: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    missing_oos_years: List[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.gates.get("pass")) and self.error is None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "gates": self.gates,
            "error": self.error,
            "pass": self.passed,
            "missing_oos_years": list(self.missing_oos_years),
            "coverage_ok": len(self.missing_oos_years) == 0 and self.error is None,
        }


@dataclass
class LimitArm:
    """One universe_limit arm with screen/confirm/full results."""

    universe_limit: int
    screen: ArmWindowResult = field(default_factory=ArmWindowResult)
    confirm: ArmWindowResult = field(default_factory=ArmWindowResult)
    full: ArmWindowResult = field(default_factory=ArmWindowResult)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_limit": self.universe_limit,
            "screen": self.screen.to_public_dict(),
            "confirm": self.confirm.to_public_dict(),
            "full": self.full.to_public_dict(),
        }


def _confirm_sort_key(arm: LimitArm) -> Tuple[float, float, int]:
    """Higher confirm CAGR, then higher (less negative) MDD, then more trades."""
    m = arm.confirm.metrics or {}
    cagr = _metric_float(m, "cagr", -999.0)
    mdd = _metric_float(m, "max_drawdown", -999.0)
    n = _metric_int(m, "n_trades", 0)
    return (cagr, mdd, n)


def rank_arms(arms: Sequence[LimitArm]) -> List[LimitArm]:
    """Rank by confirm passers first, then confirm CAGR, MDD, trades.

    Pre-registered ranking: among arms that pass **confirm** gates, sort by
    confirm CAGR desc; ties by better (higher) confirm MDD, then more trades.
    Non-passers are ordered after passers with the same secondary sort.
    """
    passers = [a for a in arms if a.confirm.passed]
    fails = [a for a in arms if not a.confirm.passed]
    passers_s = sorted(passers, key=_confirm_sort_key, reverse=True)
    fails_s = sorted(fails, key=_confirm_sort_key, reverse=True)
    return passers_s + fails_s


def build_decision(
    arms: Sequence[LimitArm],
    *,
    ranked: Optional[Sequence[LimitArm]] = None,
) -> Dict[str, Any]:
    """Decision language for pre-registered screen/confirm study.

    - Confirm pass + full path pass → research PASS candidate (no freeze change)
    - Screen good but confirm fail → FAIL (overfit to screen window)
    - 50/60 pass confirm but 80 fails → capacity sensitivity (not magic 54)
    """
    ranked_list = list(ranked) if ranked is not None else rank_arms(arms)
    by_limit = {int(a.universe_limit): a for a in arms}

    confirm_passers = [a for a in ranked_list if a.confirm.passed]
    full_passers = [a for a in ranked_list if a.full.passed]
    # Research PASS requires confirm AND full (pre-registered decision rule)
    full_and_confirm = [a for a in confirm_passers if a.full.passed]
    screen_only_passers = [
        a
        for a in ranked_list
        if a.screen.passed and not a.confirm.passed
    ]

    best_confirm = confirm_passers[0] if confirm_passers else None
    best_research = full_and_confirm[0] if full_and_confirm else None
    # Best stitched full among full-path passers (not used for research PASS)
    best_full_path_only = (
        sorted(
            full_passers,
            key=lambda a: (
                _metric_float(a.full.metrics or {}, "cagr", -999.0),
                _metric_float(a.full.metrics or {}, "max_drawdown", -999.0),
                _metric_int(a.full.metrics or {}, "n_trades", 0),
            ),
            reverse=True,
        )[0]
        if full_passers
        else None
    )

    research_pass = best_research is not None
    verdict = "RESEARCH_PASS_CANDIDATE" if research_pass else "FAIL"

    # Overfit signal: any screen pass with no confirm pass
    overfit_screen = bool(screen_only_passers) and not confirm_passers

    # Capacity sensitivity: mid limits pass confirm, 80 fails
    mid_pass = any(
        by_limit[L].confirm.passed
        for L in (50, 60)
        if L in by_limit
    )
    lim80 = by_limit.get(80)
    capacity_sensitivity = bool(
        mid_pass and lim80 is not None and not lim80.confirm.passed
    )

    notes: List[str] = []
    if research_pass and best_research is not None:
        notes.append(
            f"limit={best_research.universe_limit} passes confirm + full path gates "
            "→ research PASS candidate (paper freeze unchanged)."
        )
    if overfit_screen:
        lims = ", ".join(str(a.universe_limit) for a in screen_only_passers)
        notes.append(
            f"Screen-only pass (limits {lims}) with confirm FAIL → overfit to 2010–17."
        )
    if capacity_sensitivity:
        notes.append(
            "Limits 50/60 pass confirm but 80 fails → capacity sensitivity "
            "(not a magic-54 artifact claim)."
        )
    if not confirm_passers and not overfit_screen:
        notes.append("No arm passed confirm gates.")
    if confirm_passers and not full_and_confirm:
        notes.append(
            "Confirm passers exist but full stitched path failed gates "
            "(or full not computed)."
        )
    full_only = [a for a in full_passers if not a.confirm.passed]
    if full_only and not research_pass:
        lims = ", ".join(str(a.universe_limit) for a in full_only)
        notes.append(
            f"Full path gates pass for limits {lims} but confirm FAIL "
            "→ not a research PASS (confirm is the hold-out)."
        )

    notes.append("Paper freeze: turbo_highvol_minalloc — NOT auto-changed.")
    notes.append(PREREG_NOTE)

    best_confirm_and_full = (
        int(best_research.universe_limit) if best_research else None
    )
    best_full_only = (
        int(best_full_path_only.universe_limit) if best_full_path_only else None
    )
    return {
        "verdict": verdict,
        "research_pass_candidate": research_pass,
        "paper_freeze_unchanged": True,
        "paper_freeze": "turbo_highvol_minalloc",
        "best_confirm_limit": (
            int(best_confirm.universe_limit) if best_confirm else None
        ),
        # Confirm ∩ full (research selection). Alias best_full_limit kept for
        # older readers; DECISION labels use the explicit name.
        "best_confirm_and_full_limit": best_confirm_and_full,
        "best_full_limit": best_confirm_and_full,  # alias → confirm∩full
        "best_full_path_only_limit": best_full_only,
        "confirm_pass_limits": [int(a.universe_limit) for a in confirm_passers],
        "full_path_pass_limits": [int(a.universe_limit) for a in full_passers],
        "full_pass_limits": [int(a.universe_limit) for a in full_and_confirm],
        "screen_only_pass_limits": [
            int(a.universe_limit) for a in screen_only_passers
        ],
        "overfit_to_screen": overfit_screen,
        "capacity_sensitivity": capacity_sensitivity,
        "ranked_limits": [int(a.universe_limit) for a in ranked_list],
        "notes": notes,
        "disclaimer": "Research only. Not financial advice.",
    }


# ---------------------------------------------------------------------------
# Metrics / path runner (reuse longpath patterns)
# ---------------------------------------------------------------------------


def _eq_norm(s: pd.Series) -> pd.Series:
    out = s.dropna().astype(float)
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].dropna().sort_index()
    return out


def compute_metrics(
    eq: pd.Series, trades: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    eq = _eq_norm(eq)
    if eq.empty:
        return {"error": "empty"}
    start = float(eq.iloc[0])
    tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    rep = equity_metrics(eq, start_equity=start, trades=tdf if not tdf.empty else None)
    risk = extended_risk_from_equity(
        eq.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy()
        if not tdf.empty and "net_profit" in tdf.columns
        else None,
    )
    return {
        "cagr": float(rep.cagr),
        "sharpe": float(rep.sharpe),
        "sortino": float(risk.sortino),
        "max_drawdown": float(rep.max_drawdown),
        "n_trades": int(rep.n_trades),
        "win_rate": float(rep.win_rate) if rep.win_rate is not None else None,
        "total_return": float(eq.iloc[-1] / start - 1.0),
        "start": str(eq.index.min()),
        "end": str(eq.index.max()),
        "n_bars": int(len(eq)),
    }


def _spy_excess(eq: pd.Series, data_root: Path) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        if b is None or b.empty:
            return None
        eq2 = _eq_norm(eq)
        b = _eq_norm(b)
        j = pd.concat([eq2.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        if len(j) < 5:
            return None
        return float(j["s"].iloc[-1] / j["s"].iloc[0] - j["b"].iloc[-1] / j["b"].iloc[0])
    except Exception:
        return None


def _year_table(eq: pd.Series, trades: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    eq = _eq_norm(eq)
    rows: List[Dict[str, Any]] = []
    for y, seg in eq.groupby(eq.index.year):
        if len(seg) < 3:
            continue
        ret = float(seg.iloc[-1] / float(seg.iloc[0]) - 1.0)
        peak = seg.cummax()
        dd = float((seg / peak - 1.0).min())
        n_tr = 0
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            t = trades.copy()
            if "oos_year" in t.columns:
                n_tr = int((pd.to_numeric(t["oos_year"], errors="coerce") == int(y)).sum())
            elif "entry_date" in t.columns:
                ed = pd.to_datetime(t["entry_date"], utc=True, errors="coerce")
                n_tr = int((ed.dt.year == int(y)).sum())
        rows.append(
            {
                "year": int(y),
                "return": ret,
                "max_drawdown": dd,
                "n_trades": n_tr,
                "green": ret > 0,
            }
        )
    return rows


def run_path(
    name: str,
    *,
    first: int,
    last: int,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    min_train_rows: int,
) -> Dict[str, Any]:
    """Walk-forward annual retrain for one window and one universe_limit."""
    strat = get_strategy(name)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(ticker_file)
    base_ov = strat.backtest_overrides() if hasattr(strat, "backtest_overrides") else {}
    merged = {**base_ov, "commission": COMMISSION, "slippage": SLIPPAGE}

    def _ov() -> Dict[str, Any]:
        return dict(merged)

    orig = getattr(strat, "backtest_overrides", None)
    if orig is not None:
        strat.backtest_overrides = _ov  # type: ignore[method-assign]
    try:
        res = run_strategy_walk_forward(
            strat,
            data_root=data_root,
            ticker_file=ticker_file,
            universe_limit=universe_limit,
            first_oos_year=int(first),
            last_oos_year=int(last),
            min_train_rows=int(min_train_rows),
            preferred_index=["SPY", "QQQ"],
            base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
        )
    finally:
        if orig is not None:
            strat.backtest_overrides = orig  # type: ignore[method-assign]
    return res


def _tag_trades(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty:
        return tdf
    out = tdf.copy()
    if "oos_year" not in out.columns and "entry_date" in out.columns:
        out["oos_year"] = pd.to_datetime(
            out["entry_date"], utc=True, errors="coerce"
        ).dt.year
    return out


def _window_from_run(
    res: Dict[str, Any],
    data_root: Path,
    *,
    first: Optional[int] = None,
    last: Optional[int] = None,
) -> Tuple[ArmWindowResult, Optional[pd.Series], pd.DataFrame]:
    eq = res.get("equity")
    tr = res.get("trades")
    year_results = res.get("year_results")
    if not isinstance(eq, pd.Series) or eq.empty:
        miss: List[int] = []
        if first is not None and last is not None:
            miss = missing_oos_years(
                None, int(first), int(last), year_results=year_results
            )
        arm = ArmWindowResult(
            error="empty_equity",
            gates=apply_path_gates({}),
            missing_oos_years=miss,
        )
        return arm, None, pd.DataFrame()
    eq = _eq_norm(eq)
    tdf = _tag_trades(tr if isinstance(tr, pd.DataFrame) else pd.DataFrame())
    m = compute_metrics(eq, tdf)
    m["excess_spy_total"] = _spy_excess(eq, data_root)
    miss = []
    err = None
    if first is not None and last is not None:
        miss = missing_oos_years(
            eq, int(first), int(last), year_results=year_results
        )
        m["missing_oos_years"] = miss
        m["coverage_ok"] = len(miss) == 0
        if miss:
            err = f"missing_oos_years={miss}"
    gates = apply_path_gates(m)
    return (
        ArmWindowResult(metrics=m, gates=gates, error=err, missing_oos_years=miss),
        eq,
        tdf,
    )


def _fmt_pct(x: Any, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(x):.{digits}f}%"
    except Exception:
        return "n/a"


def _write_reports(
    out: Path,
    *,
    arms: List[LimitArm],
    ranked: List[LimitArm],
    decision: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "decision": decision,
        "arms": [a.to_dict() for a in ranked],
        "disclaimer": "Research only. Not financial advice.",
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Universe limit screen/confirm robustness",
        "",
        "> **Research only.** Not financial advice. Paper freeze unchanged.",
        "",
        f"- Strategy: `{meta.get('strategy')}`",
        f"- Universe: `{meta.get('ticker_file')}` (n={meta.get('n_tickers')})",
        f"- Limits (pre-reg): **{meta.get('limits')}**",
        f"- Screen OOS: **{meta.get('screen_first')}–{meta.get('screen_last')}**",
        f"- Confirm OOS: **{meta.get('confirm_first')}–{meta.get('confirm_last')}**",
        f"- Full path: stitched screen+confirm",
        f"- Costs: commission {COMMISSION} + slippage {SLIPPAGE}",
        f"- Gates: CAGR > {100*GATE_CAGR:.0f}%, MDD ≥ {100*GATE_MDD:.0f}%, "
        f"n_trades ≥ {GATE_TRADES}",
        f"- Banlists: **none**",
        f"- Generated: {payload['generated']}",
        "",
        "## Ranking (confirm passers first, by confirm CAGR)",
        "",
        "| rank | limit | screen CAGR | screen MDD | screen n | screen | "
        "confirm CAGR | confirm MDD | confirm n | confirm | "
        "full CAGR | full MDD | full n | full |",
        "|------|-------|-------------|------------|----------|--------|"
        "--------------|-------------|-----------|---------|"
        "-----------|----------|--------|------|",
    ]
    for i, a in enumerate(ranked, 1):
        sm, cm, fm = a.screen.metrics, a.confirm.metrics, a.full.metrics
        lines.append(
            f"| {i} | **{a.universe_limit}** | "
            f"{_fmt_pct(sm.get('cagr'))} | {_fmt_pct(sm.get('max_drawdown'))} | "
            f"{sm.get('n_trades', '—')} | "
            f"{'**PASS**' if a.screen.passed else 'FAIL'} | "
            f"{_fmt_pct(cm.get('cagr'))} | {_fmt_pct(cm.get('max_drawdown'))} | "
            f"{cm.get('n_trades', '—')} | "
            f"{'**PASS**' if a.confirm.passed else 'FAIL'} | "
            f"{_fmt_pct(fm.get('cagr'))} | {_fmt_pct(fm.get('max_drawdown'))} | "
            f"{fm.get('n_trades', '—')} | "
            f"{'**PASS**' if a.full.passed else 'FAIL'} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        f"- Verdict: **{decision.get('verdict')}**",
        f"- Research PASS candidate: **{decision.get('research_pass_candidate')}**",
        f"- Best confirm limit: **{decision.get('best_confirm_limit')}**",
        f"- Best confirm∩full limit: "
        f"**{decision.get('best_confirm_and_full_limit')}**",
        f"- Best full-path-only limit (not research PASS): "
        f"**{decision.get('best_full_path_only_limit')}**",
        f"- Confirm pass limits: {decision.get('confirm_pass_limits')}",
        f"- Full path pass limits (stitched only): "
        f"{decision.get('full_path_pass_limits')}",
        f"- Confirm∩full pass limits: {decision.get('full_pass_limits')}",
        f"- Screen-only pass limits: {decision.get('screen_only_pass_limits')}",
        f"- Overfit to screen: **{decision.get('overfit_to_screen')}**",
        f"- Capacity sensitivity (50/60 pass, 80 fail confirm): "
        f"**{decision.get('capacity_sensitivity')}**",
        f"- Paper freeze: **unchanged** (`turbo_highvol_minalloc`)",
        f"- Full-path gates: equity CAGR/MDD + n_trades; confirm trade $ "
        f"scaled by capital continuity for secondary stats",
        "",
        "### Notes",
        "",
    ]
    for n in decision.get("notes") or []:
        lines.append(f"- {n}")

    lines += [
        "",
        "## Why this study",
        "",
        "Prior accidental limit=54 passed gates while pre-reg limit=80 failed. "
        "This grid separates luck vs real capacity sensitivity without re-introducing 54.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    dlines = [
        "# Universe limit screen/confirm — Decision",
        "",
        f"**Verdict:** `{decision.get('verdict')}`",
        "",
        f"- Research PASS candidate: **{decision.get('research_pass_candidate')}**",
        f"- Best confirm limit: **{decision.get('best_confirm_limit')}**",
        f"- Best confirm∩full limit: "
        f"**{decision.get('best_confirm_and_full_limit')}**",
        f"- Best full-path-only limit (stitched; not research PASS): "
        f"**{decision.get('best_full_path_only_limit')}**",
        f"- Confirm passers: {decision.get('confirm_pass_limits')}",
        f"- Full path passers (stitched): {decision.get('full_path_pass_limits')}",
        f"- Confirm∩full passers: {decision.get('full_pass_limits')}",
        f"- Overfit to 2010–17 screen: **{decision.get('overfit_to_screen')}**",
        f"- Capacity sensitivity (mid pass / 80 fail): "
        f"**{decision.get('capacity_sensitivity')}**",
        "",
        "**Paper freeze unchanged** (`turbo_highvol_minalloc`). No auto-promotion.",
        "",
    ]
    for n in decision.get("notes") or []:
        dlines.append(f"- {n}")
    dlines += ["", "Research only. Not financial advice.", ""]
    (out / "DECISION.md").write_text("\n".join(dlines), encoding="utf-8")


def run_study(
    *,
    strategy: str,
    limits: Sequence[int],
    screen_first: int,
    screen_last: int,
    confirm_first: int,
    confirm_last: int,
    ticker_file: Path,
    data_root: Path,
    out: Path,
    min_train_rows: int,
) -> Dict[str, Any]:
    """Run full pre-registered screen/confirm grid and write reports."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arms_dir = out / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    n_tickers = count_tickers(ticker_file)
    meta = {
        "strategy": strategy,
        "limits": list(limits),
        "screen_first": screen_first,
        "screen_last": screen_last,
        "confirm_first": confirm_first,
        "confirm_last": confirm_last,
        "ticker_file": str(ticker_file),
        "n_tickers": n_tickers,
        "data_root": str(data_root),
        "min_train_rows": min_train_rows,
        "commission": COMMISSION,
        "slippage": SLIPPAGE,
        "gates": {
            "cagr_gt": GATE_CAGR,
            "mdd_ge": GATE_MDD,
            "n_trades_ge": GATE_TRADES,
        },
        "banlists": None,
        "prereg": PREREG_NOTE,
    }

    arms: List[LimitArm] = []

    for lim in limits:
        print(
            f"[limit={lim}] screen {screen_first}-{screen_last} …",
            flush=True,
        )
        arm = LimitArm(universe_limit=int(lim))
        arm_out = arms_dir / f"limit_{lim}"
        arm_out.mkdir(parents=True, exist_ok=True)

        # --- Screen ---
        screen_run_exc: Optional[str] = None
        try:
            res_s = run_path(
                strategy,
                first=screen_first,
                last=screen_last,
                data_root=data_root,
                ticker_file=ticker_file,
                universe_limit=int(lim),
                min_train_rows=min_train_rows,
            )
        except Exception as exc:
            screen_run_exc = str(exc)
            res_s = {}
            print(f"  screen ERROR: {exc}", flush=True)

        screen_win, eq_s, tr_s = (
            _window_from_run(
                res_s, data_root, first=screen_first, last=screen_last
            )
            if res_s
            else (
                ArmWindowResult(
                    error=screen_run_exc or "no_run",
                    gates=apply_path_gates({}),
                    missing_oos_years=list(
                        range(screen_first, screen_last + 1)
                    ),
                ),
                None,
                pd.DataFrame(),
            )
        )
        if screen_run_exc:
            arm.screen = ArmWindowResult(
                error=screen_run_exc,
                gates=screen_win.gates,
                metrics=screen_win.metrics,
                missing_oos_years=screen_win.missing_oos_years,
            )
        else:
            arm.screen = screen_win

        if eq_s is not None:
            eq_s.to_csv(arm_out / "equity_screen.csv", header=["equity"])
        if not tr_s.empty:
            tr_s.to_csv(arm_out / "trades_screen.csv", index=False)
        sm = arm.screen.metrics
        miss_s = arm.screen.missing_oos_years
        print(
            f"  screen CAGR={_fmt_pct(sm.get('cagr'))} "
            f"MDD={_fmt_pct(sm.get('max_drawdown'))} "
            f"n={sm.get('n_trades')} pass={arm.screen.passed}"
            f"{f' missing_years={miss_s}' if miss_s else ''}",
            flush=True,
        )

        # --- Confirm ---
        print(
            f"[limit={lim}] confirm {confirm_first}-{confirm_last} …",
            flush=True,
        )
        confirm_run_exc: Optional[str] = None
        try:
            res_c = run_path(
                strategy,
                first=confirm_first,
                last=confirm_last,
                data_root=data_root,
                ticker_file=ticker_file,
                universe_limit=int(lim),
                min_train_rows=min_train_rows,
            )
        except Exception as exc:
            confirm_run_exc = str(exc)
            res_c = {}
            print(f"  confirm ERROR: {exc}", flush=True)

        conf_win, eq_c, tr_c = (
            _window_from_run(
                res_c, data_root, first=confirm_first, last=confirm_last
            )
            if res_c
            else (
                ArmWindowResult(
                    error=confirm_run_exc or "no_run",
                    gates=apply_path_gates({}),
                    missing_oos_years=list(
                        range(confirm_first, confirm_last + 1)
                    ),
                ),
                None,
                pd.DataFrame(),
            )
        )
        if confirm_run_exc:
            arm.confirm = ArmWindowResult(
                error=confirm_run_exc,
                gates=conf_win.gates,
                metrics=conf_win.metrics,
                missing_oos_years=conf_win.missing_oos_years,
            )
        else:
            arm.confirm = conf_win

        if eq_c is not None:
            eq_c.to_csv(arm_out / "equity_confirm.csv", header=["equity"])
        if not tr_c.empty:
            tr_c.to_csv(arm_out / "trades_confirm.csv", index=False)
        cm = arm.confirm.metrics
        miss_c = arm.confirm.missing_oos_years
        print(
            f"  confirm CAGR={_fmt_pct(cm.get('cagr'))} "
            f"MDD={_fmt_pct(cm.get('max_drawdown'))} "
            f"n={cm.get('n_trades')} pass={arm.confirm.passed}"
            f"{f' missing_years={miss_c}' if miss_c else ''}",
            flush=True,
        )

        # --- Full stitch ---
        # Equity stitch is capital-continuous; trade $ scaled by same factor so
        # secondary trade stats are comparable. Gates use equity CAGR/MDD + n.
        if eq_s is not None and eq_c is not None:
            eq_full = stitch_equity(eq_s, eq_c)
            scale_b = capital_continuity_scale(eq_s, eq_c)
            tr_full = stitch_trades(tr_s, tr_c, scale_b=scale_b)
            if not tr_full.empty:
                tr_full = _tag_trades(tr_full)
            if not eq_full.empty:
                m_f = compute_metrics(eq_full, tr_full)
                m_f["excess_spy_total"] = _spy_excess(eq_full, data_root)
                m_f["trade_pnl_capital_scale"] = float(scale_b)
                m_f["full_gates_equity_primary"] = True
                miss_full = sorted(
                    set(arm.screen.missing_oos_years)
                    | set(arm.confirm.missing_oos_years)
                )
                m_f["missing_oos_years"] = miss_full
                m_f["coverage_ok"] = len(miss_full) == 0
                err_f = (
                    f"missing_oos_years={miss_full}" if miss_full else None
                )
                g_f = apply_path_gates(m_f)
                arm.full = ArmWindowResult(
                    metrics=m_f,
                    gates=g_f,
                    error=err_f,
                    missing_oos_years=miss_full,
                )
                eq_full.to_csv(arm_out / "equity_full.csv", header=["equity"])
                if not tr_full.empty:
                    tr_full.to_csv(arm_out / "trades_full.csv", index=False)
                years = _year_table(eq_full, tr_full)
                (arm_out / "years_full.json").write_text(
                    json.dumps(years, indent=2, default=str), encoding="utf-8"
                )
            else:
                arm.full = ArmWindowResult(
                    error="empty_stitch", gates=apply_path_gates({})
                )
        else:
            arm.full = ArmWindowResult(
                error="missing_segment", gates=apply_path_gates({})
            )

        fm = arm.full.metrics
        print(
            f"  full CAGR={_fmt_pct(fm.get('cagr'))} "
            f"MDD={_fmt_pct(fm.get('max_drawdown'))} "
            f"n={fm.get('n_trades')} pass={arm.full.passed}",
            flush=True,
        )

        (arm_out / "arm.json").write_text(
            json.dumps(arm.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        arms.append(arm)

    ranked = rank_arms(arms)
    decision = build_decision(arms, ranked=ranked)
    _write_reports(out, arms=arms, ranked=ranked, decision=decision, meta=meta)

    print(
        f"Wrote {out / 'SUMMARY.md'} verdict={decision.get('verdict')}",
        flush=True,
    )
    return {
        "meta": meta,
        "decision": decision,
        "arms": [a.to_dict() for a in ranked],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-registered universe_limit screen/confirm study"
    )
    ap.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    ap.add_argument(
        "--limits",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LIMITS),
        help="Comma-separated universe_limit grid (pre-reg: 40,50,60,80)",
    )
    ap.add_argument("--screen-first", type=int, default=DEFAULT_SCREEN[0])
    ap.add_argument("--screen-last", type=int, default=DEFAULT_SCREEN[1])
    ap.add_argument("--confirm-first", type=int, default=DEFAULT_CONFIRM[0])
    ap.add_argument("--confirm-last", type=int, default=DEFAULT_CONFIRM[1])
    ap.add_argument(
        "--ticker-file",
        type=Path,
        default=None,
        help="Universe file (default: longhist2010_pass if n>=40 else longhist100)",
    )
    ap.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "universe_limit_sc",
    )
    ap.add_argument(
        "--allow-unregistered-limits",
        action="store_true",
        help=(
            "Allow limits outside pre-registered {40,50,60,80}. "
            "Limit 54 is always banned."
        ),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    limits = validate_limits(
        parse_limits(args.limits),
        allow_unregistered=bool(args.allow_unregistered_limits),
    )
    if tuple(limits) != DEFAULT_LIMITS:
        print(
            f"NOTE: limits {limits} differ from default pre-reg order "
            f"{list(DEFAULT_LIMITS)} (allowed via flag or subset).",
            flush=True,
        )

    ticker_file = resolve_ticker_file(args.ticker_file, root=ROOT, min_n=40)
    n = count_tickers(ticker_file)
    print(f"Universe: {ticker_file} n={n}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    run_study(
        strategy=str(args.strategy).strip(),
        limits=limits,
        screen_first=int(args.screen_first),
        screen_last=int(args.screen_last),
        confirm_first=int(args.confirm_first),
        confirm_last=int(args.confirm_last),
        ticker_file=ticker_file,
        data_root=data_root,
        out=out,
        min_train_rows=int(args.min_train_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
