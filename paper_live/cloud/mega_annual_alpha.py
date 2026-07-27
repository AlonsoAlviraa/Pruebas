"""Mega annual alpha study — beat all index B&H by +3pp each calendar year.

Honesty labels:
  - Capital: VIRTUAL
  - Equity fills: paper
  - Options marks: proxy_bs | vix_surface (never OPRA)
  - Features causal ≤ t only

This module separates pure evaluation / ranking (unit-testable, no network)
from the heavy multi-year runner that reuses cloud equity + options batches.
"""
from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from paper_live.options.metrics import max_drawdown as _max_drawdown_curve

logger = logging.getLogger(__name__)

DEFAULT_ZOO = Path(__file__).resolve().parent / "zoo_mega_alpha.json"
DEFAULT_OUT = Path("reports/mega_annual_alpha")
EXCESS_MARGIN = 0.03  # +3 percentage points over best index

# Calendar study windows (2025_study = YTD as available)
DEFAULT_WINDOWS: List[Tuple[str, str, str]] = [
    ("2022", "2022-01-03", "2022-12-30"),
    ("2023", "2023-01-03", "2023-12-29"),
    ("2024", "2024-01-02", "2024-12-31"),
    ("2025_study", "2025-01-02", "2099-12-31"),
]

BENCH_TICKERS = ("SPY", "QQQ", "IWM")


# ---------------------------------------------------------------------------
# Pure evaluation helpers (no I/O, no network)
# ---------------------------------------------------------------------------


def clamp_calendar_window(
    days: Sequence[date],
    start: date,
    end: date,
) -> Tuple[date, date, bool]:
    """Clamp [start, end] to available session days. Returns (s, e, clamped)."""
    if not days:
        raise ValueError("No session days available")
    s = next((d for d in days if d >= start), days[0])
    e = next((d for d in reversed(days) if d <= end), days[-1])
    if s > e:
        s, e = days[0], days[-1]
    clamped = s != start or e != end
    return s, e, clamped


def best_index_return(
    benchmarks: Mapping[str, Optional[float]],
) -> Optional[float]:
    """Max of available index B&H returns (SPY/QQQ/IWM)."""
    vals = [float(v) for v in benchmarks.values() if v is not None]
    if not vals:
        return None
    return max(vals)


def excess_over_best_index(
    strategy_return: float,
    benchmarks: Mapping[str, Optional[float]],
) -> Optional[float]:
    """strategy_return − max(available index B&H)."""
    best = best_index_return(benchmarks)
    if best is None:
        return None
    return float(strategy_return) - float(best)


def beat_all_indices_by_3pp(
    strategy_return: float,
    benchmarks: Mapping[str, Optional[float]],
    *,
    margin: float = EXCESS_MARGIN,
) -> bool:
    """True iff strategy_return ≥ max(available index B&H) + margin.

    Requires at least one benchmark. Missing indices are ignored (not treated as 0).
    """
    best = best_index_return(benchmarks)
    if best is None:
        return False
    return float(strategy_return) >= float(best) + float(margin)


def max_drawdown_from_equity(equity: Sequence[float]) -> float:
    """Max DD as negative fraction (reuse options metrics helper)."""
    return float(_max_drawdown_curve(equity))


@dataclass
class YearEval:
    """One strategy × one calendar year evaluation."""

    strategy_id: str
    year: str
    total_return: float
    max_dd: Optional[float] = None
    spy_bh: Optional[float] = None
    qqq_bh: Optional[float] = None
    iwm_bh: Optional[float] = None
    vs_spy: Optional[float] = None
    vs_qqq: Optional[float] = None
    vs_iwm: Optional[float] = None
    excess_vs_best: Optional[float] = None
    beat_all_indices_by_3pp: bool = False
    n_opens: int = 0
    n_closed_trades: int = 0
    hard_kill: bool = False
    error: Optional[str] = None
    asset_class: str = "equity"  # equity | options
    signal_mode: str = ""
    kind: str = ""
    window_start: str = ""
    window_end: str = ""
    clamped: bool = False
    data_label: str = "paper_equity"

    def benchmarks_map(self) -> Dict[str, Optional[float]]:
        return {"spy_bh": self.spy_bh, "qqq_bh": self.qqq_bh, "iwm_bh": self.iwm_bh}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "year": self.year,
            "total_return": self.total_return,
            "max_dd": self.max_dd,
            "spy_bh": self.spy_bh,
            "qqq_bh": self.qqq_bh,
            "iwm_bh": self.iwm_bh,
            "vs_spy": self.vs_spy,
            "vs_qqq": self.vs_qqq,
            "vs_iwm": self.vs_iwm,
            "excess_vs_best": self.excess_vs_best,
            "beat_all_indices_by_3pp": self.beat_all_indices_by_3pp,
            "n_opens": self.n_opens,
            "n_closed_trades": self.n_closed_trades,
            "hard_kill": self.hard_kill,
            "error": self.error,
            "asset_class": self.asset_class,
            "signal_mode": self.signal_mode,
            "kind": self.kind,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "clamped": self.clamped,
            "data_label": self.data_label,
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }


def build_year_eval(
    *,
    strategy_id: str,
    year: str,
    total_return: float,
    benchmarks: Mapping[str, Optional[float]],
    max_dd: Optional[float] = None,
    n_opens: int = 0,
    n_closed_trades: int = 0,
    hard_kill: bool = False,
    error: Optional[str] = None,
    asset_class: str = "equity",
    signal_mode: str = "",
    kind: str = "",
    window_start: str = "",
    window_end: str = "",
    clamped: bool = False,
    data_label: str = "paper_equity",
    margin: float = EXCESS_MARGIN,
) -> YearEval:
    """Construct a YearEval with derived excess / beat flags."""
    spy = benchmarks.get("spy_bh")
    qqq = benchmarks.get("qqq_bh")
    iwm = benchmarks.get("iwm_bh")
    if error:
        return YearEval(
            strategy_id=strategy_id,
            year=year,
            total_return=float(total_return),
            max_dd=max_dd,
            spy_bh=spy,
            qqq_bh=qqq,
            iwm_bh=iwm,
            error=error,
            asset_class=asset_class,
            signal_mode=signal_mode,
            kind=kind,
            window_start=window_start,
            window_end=window_end,
            clamped=clamped,
            data_label=data_label,
            hard_kill=hard_kill,
            n_opens=n_opens,
            n_closed_trades=n_closed_trades,
        )
    excess = excess_over_best_index(total_return, benchmarks)
    beat = beat_all_indices_by_3pp(total_return, benchmarks, margin=margin)
    return YearEval(
        strategy_id=strategy_id,
        year=year,
        total_return=float(total_return),
        max_dd=max_dd,
        spy_bh=spy,
        qqq_bh=qqq,
        iwm_bh=iwm,
        vs_spy=(float(total_return) - float(spy)) if spy is not None else None,
        vs_qqq=(float(total_return) - float(qqq)) if qqq is not None else None,
        vs_iwm=(float(total_return) - float(iwm)) if iwm is not None else None,
        excess_vs_best=excess,
        beat_all_indices_by_3pp=beat,
        n_opens=n_opens,
        n_closed_trades=n_closed_trades,
        hard_kill=hard_kill,
        asset_class=asset_class,
        signal_mode=signal_mode,
        kind=kind,
        window_start=window_start,
        window_end=window_end,
        clamped=clamped,
        data_label=data_label,
    )


@dataclass
class MultiYearSummary:
    strategy_id: str
    label: str = ""
    asset_class: str = "equity"
    years_evaluated: int = 0
    years_passed: int = 0
    years_failed: int = 0
    years_error: int = 0
    mean_excess_vs_best: Optional[float] = None
    mean_total_return: Optional[float] = None
    worst_max_dd: Optional[float] = None
    total_opens: int = 0
    total_closed_trades: int = 0
    hard_kill_years: int = 0
    beat_flags: Dict[str, bool] = field(default_factory=dict)
    year_returns: Dict[str, float] = field(default_factory=dict)
    year_excess: Dict[str, Optional[float]] = field(default_factory=dict)
    tier: str = ""  # e.g. "4/4", "3/4"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_strategy_years(
    year_evals: Sequence[YearEval],
    *,
    strategy_id: str,
    label: str = "",
    asset_class: str = "equity",
    n_study_years: Optional[int] = None,
) -> MultiYearSummary:
    """Aggregate year rows for one strategy into multi-year summary + tier."""
    rows = [y for y in year_evals if y.strategy_id == strategy_id]
    ok_rows = [y for y in rows if not y.error]
    err_rows = [y for y in rows if y.error]
    passed = [y for y in ok_rows if y.beat_all_indices_by_3pp]
    failed = [y for y in ok_rows if not y.beat_all_indices_by_3pp]
    excesses = [y.excess_vs_best for y in ok_rows if y.excess_vs_best is not None]
    rets = [y.total_return for y in ok_rows]
    dds = [y.max_dd for y in ok_rows if y.max_dd is not None]
    n_years = n_study_years if n_study_years is not None else max(len(rows), 1)
    tier = f"{len(passed)}/{n_years}"
    return MultiYearSummary(
        strategy_id=strategy_id,
        label=label or strategy_id,
        asset_class=asset_class,
        years_evaluated=len(ok_rows),
        years_passed=len(passed),
        years_failed=len(failed),
        years_error=len(err_rows),
        mean_excess_vs_best=(sum(excesses) / len(excesses)) if excesses else None,
        mean_total_return=(sum(rets) / len(rets)) if rets else None,
        worst_max_dd=min(dds) if dds else None,
        total_opens=sum(y.n_opens for y in ok_rows),
        total_closed_trades=sum(y.n_closed_trades for y in ok_rows),
        hard_kill_years=sum(1 for y in ok_rows if y.hard_kill),
        beat_flags={y.year: y.beat_all_indices_by_3pp for y in ok_rows},
        year_returns={y.year: y.total_return for y in ok_rows},
        year_excess={y.year: y.excess_vs_best for y in ok_rows},
        tier=tier,
    )


def filter_winners(
    summaries: Sequence[MultiYearSummary],
    *,
    min_years_passed: int,
    n_study_years: int,
    min_opens: int = 0,
    allow_hard_kill: bool = False,
) -> List[MultiYearSummary]:
    """Strict/tiered winner filter.

    Primary strict gate uses min_years_passed == n_study_years (e.g. 4/4).
    Secondary tiers: 3/4, 2/4, etc.
    """
    out: List[MultiYearSummary] = []
    for s in summaries:
        if s.years_passed < min_years_passed:
            continue
        if min_opens > 0 and s.total_opens < min_opens and s.total_closed_trades < min_opens:
            continue
        if not allow_hard_kill and s.hard_kill_years > 0:
            # soft: allow if they still passed enough years? task says no hard-kill spam
            # skip strategies that hard-killed in any evaluated year
            continue
        out.append(s)
    return rank_by_mean_excess(out)


def rank_by_mean_excess(
    summaries: Sequence[MultiYearSummary],
) -> List[MultiYearSummary]:
    """Rank by years_passed desc, then mean excess vs best index desc."""

    def key(s: MultiYearSummary) -> Tuple[int, float, float]:
        me = s.mean_excess_vs_best if s.mean_excess_vs_best is not None else -1e9
        mr = s.mean_total_return if s.mean_total_return is not None else -1e9
        return (s.years_passed, me, mr)

    return sorted(summaries, key=key, reverse=True)


def build_tier_tables(
    summaries: Sequence[MultiYearSummary],
    *,
    n_study_years: int,
    min_opens: int = 0,
) -> Dict[str, List[MultiYearSummary]]:
    """Build 4/4, 3/4, 2/4 … tier lists (allow hard-kill for transparency)."""
    tiers: Dict[str, List[MultiYearSummary]] = {}
    for k in range(n_study_years, 0, -1):
        name = f"{k}/{n_study_years}"
        tiers[name] = filter_winners(
            summaries,
            min_years_passed=k,
            n_study_years=n_study_years,
            min_opens=min_opens,
            allow_hard_kill=True,
        )
    return tiers


def window_bh_return(feed: Any, ticker: str, start_d: date, end_d: date) -> Optional[float]:
    """Buy&hold total return using session closes on/inside window."""
    try:
        b0 = feed.bar(ticker, start_d)
        b1 = feed.bar(ticker, end_d)
        if b0 is None or b1 is None:
            days = feed.session_days(start_d, end_d)
            if not days:
                return None
            if b0 is None:
                for d in days:
                    b0 = feed.bar(ticker, d)
                    if b0 is not None:
                        break
            if b1 is None:
                for d in reversed(list(days)):
                    b1 = feed.bar(ticker, d)
                    if b1 is not None:
                        break
        if b0 is None or b1 is None:
            return None
        c0, c1 = float(b0.close), float(b1.close)
        if c0 <= 0:
            return None
        return c1 / c0 - 1.0
    except Exception:
        return None


def compute_benchmarks(
    feed: Any, start_d: date, end_d: date, tickers: Sequence[str] = BENCH_TICKERS
) -> Dict[str, Optional[float]]:
    """SPY/QQQ/IWM B&H for window (None if ticker missing)."""
    out: Dict[str, Optional[float]] = {}
    feed_tickers = set(getattr(feed, "tickers", []) or [])
    for t in tickers:
        key = f"{t.lower()}_bh"
        if t.upper() not in {x.upper() for x in feed_tickers} and t.upper() not in feed_tickers:
            # still try bar — feed may store differently
            pass
        out[key] = window_bh_return(feed, t.upper(), start_d, end_d)
    # normalize keys used throughout
    return {
        "spy_bh": out.get("spy_bh"),
        "qqq_bh": out.get("qqq_bh"),
        "iwm_bh": out.get("iwm_bh"),
    }


# ---------------------------------------------------------------------------
# Zoo loading
# ---------------------------------------------------------------------------


def load_mega_zoo(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_ZOO
    return json.loads(p.read_text(encoding="utf-8"))


def equity_strategies_from_zoo(zoo: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if "equity_strategies" in zoo:
        return list(zoo["equity_strategies"] or [])
    if "strategies" in zoo and zoo.get("asset_class", "equity") != "options":
        # plain equity zoo format
        return list(zoo.get("strategies") or [])
    return list(zoo.get("equity", {}).get("strategies") or [])


def options_strategies_from_zoo(zoo: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if "options_strategies" in zoo:
        return list(zoo["options_strategies"] or [])
    return list(zoo.get("options", {}).get("strategies") or [])


def merge_strategy_lists(
    *lists: Sequence[Dict[str, Any]],
    max_strategies: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Deduplicate by id, preserve order, optional cap."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for lst in lists:
        for s in lst:
            sid = str(s.get("id") or "")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            out.append(dict(s))
            if max_strategies is not None and len(out) >= max_strategies:
                return out
    return out


# ---------------------------------------------------------------------------
# Equity single-strategy run (reuses freeze/session)
# ---------------------------------------------------------------------------


def run_one_equity_strategy(
    feed: Any,
    strat: Mapping[str, Any],
    *,
    start_d: date,
    end_d: date,
    capital0: float,
    ledger_root: Path,
    lean_ledger: bool = True,
) -> Dict[str, Any]:
    """Run one equity zoo member over a window; return metrics dict.

    ``lean_ledger=True`` (default) uses in-memory SQLite + no JSONL audit so
    multi-year matrices do not write thousands of audit files.
    """
    from paper_live.cloud.batch import _override_freeze, _trade_stats
    from paper_live.freeze import load_freeze
    from paper_live.ledger import PaperLedger
    from paper_live.replay_session import ReplaySession
    from paper_live.signals.daily_pipeline import DailySignalPipeline

    sid = str(strat["id"])
    signal_mode = str(strat.get("signal_mode") or "trend_mom")
    freeze = _override_freeze(load_freeze(), dict(strat), capital0)
    if ledger_root.exists():
        shutil.rmtree(ledger_root, ignore_errors=True)
    ledger = PaperLedger.create_run(
        ledger_root,
        freeze,
        meta={"mega_annual": True, "signal_mode": signal_mode, "strategy_id": sid},
        lean=bool(lean_ledger),
    )
    univ = [t for t in feed.tickers if t.upper() not in ("SPY",)]
    if not bool(strat.get("exclude_index", True)):
        univ = list(feed.tickers)
    top_k = strat.get("top_k", None)
    regime_sym = (
        "QQQ"
        if "QQQ" in feed.tickers
        else ("SPY" if "SPY" in feed.tickers else feed.tickers[0])
    )
    pipe = DailySignalPipeline(
        feed,
        universe=univ,
        min_price=float(strat.get("min_price", 5.0)),
        max_atr_pct=float(strat.get("max_atr_pct", 0.22)),
        min_atr_norm=float(strat.get("min_atr_norm", 0.01)),
        require_regime=bool(strat.get("require_regime", True)),
        regime_symbol=regime_sym,
        signal_mode=signal_mode,
        top_k=int(top_k) if top_k is not None else None,
        exclude_index=bool(strat.get("exclude_index", True)),
        qqq_mom_gate=bool(strat.get("qqq_mom_gate", False)),
        qqq_min_ret_1m=float(strat.get("qqq_min_ret_1m", 0.0)),
    )
    max_pos_pct = float(
        strat.get(
            "max_position_pct",
            max(0.25, float(strat.get("min_alloc_pct", 0.015)) * 1.1),
        )
    )
    session = ReplaySession(
        feed,
        freeze,
        ledger=ledger,
        pipeline=pipe,
        max_positions=int(strat.get("max_positions", 8)),
        max_horizon=int(strat.get("max_horizon", 20)),
        k_atr=float(strat.get("k_atr", 3.0)),
        hard_stop_pct=float(strat.get("hard_stop_pct", 0.07)),
        min_alloc_pct=float(strat.get("min_alloc_pct", 0.015)),
        max_position_pct=max_pos_pct,
        max_entries_per_day=int(
            strat.get(
                "max_entries_per_day",
                freeze.strategy.risk_paper.get("max_daily_new_entries", 5),
            )
        ),
        enable_risk=bool(strat.get("enable_risk", True)),
        trail_stops=bool(strat["trail_stops"]) if "trail_stops" in strat else None,
        use_hard_stop=bool(strat["use_hard_stop"]) if "use_hard_stop" in strat else None,
    )
    if "QQQ" not in feed.tickers and "SPY" not in feed.tickers:
        session.pipeline.require_regime = False

    result = session.run(start_d, end_d)
    trades = list(result.closed_trades or [])
    ts = _trade_stats(trades)
    # Prefer in-session daily_nav (works with lean memory ledger; no disk audit).
    equity: List[float] = []
    for r in result.daily_nav or []:
        if isinstance(r, dict) and r.get("equity") is not None:
            equity.append(float(r["equity"]))
    if not equity:
        try:
            ledger._commit(force=True)  # type: ignore[attr-defined]
            nav = ledger.list_nav()
            equity = [float(r["equity"]) for r in nav]
        except Exception:
            equity = []
    mdd = max_drawdown_from_equity(equity) if equity else 0.0
    tr = result.final_equity / capital0 - 1.0
    out = {
        "strategy_id": sid,
        "label": str(strat.get("label") or sid),
        "signal_mode": signal_mode,
        "total_return": tr,
        "max_dd": mdd,
        "n_entries": result.n_entries,
        "n_exits": result.n_exits,
        "n_closed_trades": int(ts["n"]),
        "n_opens": int(result.n_entries),
        "hard_kill": bool(result.hard_kill),
        "final_equity": result.final_equity,
        "error": None,
        "lean_ledger": bool(lean_ledger),
    }
    ledger.close()
    # cleanup heavy ledger artifacts (full-audit mode only leaves these)
    if not lean_ledger:
        try:
            db = ledger_root / "paper_year.db"
            if db.is_file():
                db.unlink(missing_ok=True)
            audit = ledger_root / "audit"
            if audit.is_dir():
                shutil.rmtree(audit, ignore_errors=True)
        except Exception:
            pass
    return out


def options_specs_from_dicts(strategies: Sequence[Mapping[str, Any]]):
    from paper_live.options.strategies import OptionStrategySpec

    out = []
    for s in strategies:
        out.append(
            OptionStrategySpec(
                id=str(s["id"]),
                label=str(s.get("label") or s["id"]),
                kind=str(s["kind"]),
                underlying=str(s.get("underlying") or "SPY"),
                dte_days=int(s.get("dte_days") or 30),
                otm_pct=float(s.get("otm_pct") or 0.05),
                wing_otm_pct=float(s.get("wing_otm_pct") or 0.15),
                premium_mult=float(s.get("premium_mult") or 1.15),
                contracts=int(s.get("contracts") or 1),
                max_portfolio_dd=(
                    float(s["max_portfolio_dd"]) if s.get("max_portfolio_dd") is not None else None
                ),
                max_single_day_drop=(
                    float(s["max_single_day_drop"])
                    if s.get("max_single_day_drop") is not None
                    else None
                ),
                max_margin_fraction=(
                    float(s["max_margin_fraction"])
                    if s.get("max_margin_fraction") is not None
                    else None
                ),
                hard_kill_enabled=(
                    bool(s["hard_kill_enabled"]) if s.get("hard_kill_enabled") is not None else None
                ),
                meta=dict(s.get("meta") or {}),
                notes=str(s.get("notes") or ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2%}"


def write_report_pack(
    *,
    out_root: Path,
    year_evals: Sequence[YearEval],
    summaries: Sequence[MultiYearSummary],
    tiers: Mapping[str, Sequence[MultiYearSummary]],
    windows_meta: Sequence[Dict[str, Any]],
    data_sources: Mapping[str, str],
    capital0: float,
    n_study_years: int,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write latest/SUMMARY.md, winners.json, by_year tables, full JSON."""
    out_root = Path(out_root)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    by_year_dir = latest / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)

    years = sorted({y.year for y in year_evals}, key=lambda y: (y != "2025_study", y))

    # by_year JSON + MD
    for year in years:
        rows = [y for y in year_evals if y.year == year]
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r.beat_all_indices_by_3pp,
                r.excess_vs_best if r.excess_vs_best is not None else -1e9,
            ),
            reverse=True,
        )
        payload = {
            "year": year,
            "n": len(rows_sorted),
            "rows": [r.to_dict() for r in rows_sorted],
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }
        (by_year_dir / f"{year}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        lines = [
            f"# Year `{year}`",
            "",
            "| Strategy | Class | Return | MaxDD | SPY | QQQ | IWM | Excess vs best | Beat+3pp | Kill |",
            "|----------|-------|--------|-------|-----|-----|-----|----------------|----------|------|",
        ]
        for r in rows_sorted:
            lines.append(
                f"| `{r.strategy_id}` | {r.asset_class} | {_fmt_pct(r.total_return)} | "
                f"{_fmt_pct(r.max_dd)} | {_fmt_pct(r.spy_bh)} | {_fmt_pct(r.qqq_bh)} | "
                f"{_fmt_pct(r.iwm_bh)} | {_fmt_pct(r.excess_vs_best)} | "
                f"{'YES' if r.beat_all_indices_by_3pp else 'no'} | "
                f"{'YES' if r.hard_kill else 'no'} |"
            )
        (by_year_dir / f"{year}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    strict = list(tiers.get(f"{n_study_years}/{n_study_years}") or [])
    near_miss = list(tiers.get(f"{max(n_study_years - 1, 1)}/{n_study_years}") or [])
    # near-misses that are not strict winners
    strict_ids = {s.strategy_id for s in strict}
    near_only = [s for s in near_miss if s.strategy_id not in strict_ids]

    winners_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_study_years": n_study_years,
        "margin_pp": EXCESS_MARGIN,
        "capital0": capital0,
        "mode": "paper",
        "capital_label": "VIRTUAL",
        "options_label": "proxy_bs|vix_surface (never OPRA)",
        "strict_winners": [s.to_dict() for s in strict],
        "tiers": {k: [s.to_dict() for s in v] for k, v in tiers.items()},
        "near_misses": [s.to_dict() for s in near_only[:25]],
        "all_summaries": [s.to_dict() for s in rank_by_mean_excess(list(summaries))],
        "windows": list(windows_meta),
        "data_sources": dict(data_sources),
        "meta": meta or {},
    }
    winners_path = latest / "winners.json"
    winners_path.write_text(json.dumps(winners_payload, indent=2, default=str), encoding="utf-8")

    full_path = latest / "full_results.json"
    full_path.write_text(
        json.dumps(
            {
                **winners_payload,
                "year_evals": [y.to_dict() for y in year_evals],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # SUMMARY.md
    md: List[str] = [
        "# Mega annual alpha study",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Capital:** VIRTUAL ${capital0:,.0f} · **mode:** paper",
        "",
        f"**Rule:** strategy_return ≥ max(SPY, QQQ, IWM B&H) + **{EXCESS_MARGIN:.0%}** per calendar year.",
        "",
        "**Options marks:** `proxy_bs` / `vix_surface` — never OPRA / exchange fills.",
        "",
        "Not financial advice. Past paper ≠ future results.",
        "",
        "## Study windows",
        "",
    ]
    for w in windows_meta:
        md.append(
            f"- `{w.get('name')}`: {w.get('start')} → {w.get('end')}"
            f"{' (clamped)' if w.get('clamped') else ''}"
            f" · SPY={_fmt_pct(w.get('spy_bh'))} QQQ={_fmt_pct(w.get('qqq_bh'))} "
            f"IWM={_fmt_pct(w.get('iwm_bh'))}"
        )
    # Exact-tier counts (not cumulative)
    exact_counts: Dict[int, int] = {}
    for s in summaries:
        exact_counts[s.years_passed] = exact_counts.get(s.years_passed, 0) + 1
    n_strict = len(strict)
    n_near = len(near_only)
    md += [
        "",
        "## Headline",
        "",
        f"- **Strict 4/4 (every year beat max(index)+3pp):** **{n_strict}**",
        f"- **Near-miss 3/4 only:** **{n_near}**",
        f"- **Exact tier counts:** "
        + ", ".join(
            f"{k}/{n_study_years}={exact_counts.get(k, 0)}"
            for k in range(n_study_years, -1, -1)
        ),
        f"- **Strategies evaluated:** {len(summaries)} "
        f"({sum(1 for s in summaries if s.asset_class == 'equity')} equity / "
        f"{sum(1 for s in summaries if s.asset_class == 'options')} options)",
        "",
    ]
    if n_strict == 0:
        md += [
            "> **Zero strategies cleared the strict gate every year.** "
            "This is a valid scientific result — not a failed run.",
            "",
        ]

    md += ["## Strict winners (beat all indices by +3pp every year)", ""]
    if not strict:
        md += [
            "**Zero strategies cleared the strict gate.** This is a valid scientific result.",
            "",
            "See near-miss tiers below (e.g. 3/4, 2/4 years).",
            "",
        ]
    else:
        md += [
            "| Strategy | Class | Years passed | Mean excess | Mean return | Tier |",
            "|----------|-------|--------------|-------------|-------------|------|",
        ]
        for s in strict:
            md.append(
                f"| `{s.strategy_id}` | {s.asset_class} | {s.years_passed}/{n_study_years} | "
                f"{_fmt_pct(s.mean_excess_vs_best)} | {_fmt_pct(s.mean_total_return)} | {s.tier} |"
            )
        md.append("")

    md += ["## Tiers (years beating max index +3pp)", ""]
    for k in sorted(tiers.keys(), reverse=True):
        group = list(tiers[k])
        # for non-strict, show only those that land exactly in this tier (not higher)
        if k != f"{n_study_years}/{n_study_years}":
            try:
                need = int(k.split("/")[0])
            except Exception:
                need = 0
            group = [s for s in group if s.years_passed == need]
        md.append(f"### Tier {k} (n={len(group)})")
        md.append("")
        if not group:
            md.append("_None._")
            md.append("")
            continue
        md += [
            "| Rank | Strategy | Class | Mean excess | Mean ret | Year flags | Opens | Kill yrs |",
            "|------|----------|-------|-------------|----------|------------|-------|----------|",
        ]
        for i, s in enumerate(group[:30], 1):
            flags = ",".join(
                f"{y}:{'Y' if s.beat_flags.get(y) else 'n'}"
                for y in sorted(s.beat_flags.keys(), key=lambda x: (x != "2025_study", x))
            )
            md.append(
                f"| {i} | `{s.strategy_id}` | {s.asset_class} | "
                f"{_fmt_pct(s.mean_excess_vs_best)} | {_fmt_pct(s.mean_total_return)} | "
                f"{flags} | {s.total_opens} | {s.hard_kill_years} |"
            )
        md.append("")

    if near_only:
        md += [
            "## Best near-misses (3/4, not 4/4)",
            "",
            "| Strategy | Class | Mean excess | Mean ret | Beat flags |",
            "|----------|-------|-------------|----------|------------|",
        ]
        for s in near_only[:15]:
            flags = ",".join(
                f"{y}:{'Y' if s.beat_flags.get(y) else 'n'}"
                for y in sorted(s.beat_flags.keys(), key=lambda x: (x != "2025_study", x))
            )
            md.append(
                f"| `{s.strategy_id}` | {s.asset_class} | {_fmt_pct(s.mean_excess_vs_best)} | "
                f"{_fmt_pct(s.mean_total_return)} | {flags} |"
            )
        md.append("")

    md += [
        "## Top 15 by mean excess vs best index (any tier)",
        "",
        "| Rank | Strategy | Class | Years passed | Mean excess | Mean ret | Worst DD |",
        "|------|----------|-------|--------------|-------------|----------|----------|",
    ]
    ranked = rank_by_mean_excess(list(summaries))
    for i, s in enumerate(ranked[:15], 1):
        md.append(
            f"| {i} | `{s.strategy_id}` | {s.asset_class} | {s.years_passed}/{n_study_years} | "
            f"{_fmt_pct(s.mean_excess_vs_best)} | {_fmt_pct(s.mean_total_return)} | "
            f"{_fmt_pct(s.worst_max_dd)} |"
        )

    md += [
        "",
        "## Data sources",
        "",
    ]
    for t, src in sorted(data_sources.items()):
        md.append(f"- `{t}`: {src}")
    md += [
        "",
        "## Artifacts",
        "",
        "- `winners.json` — strict + tiers + near-misses",
        "- `full_results.json` — all year evals",
        "- `by_year/<year>.md|json` — per-year tables",
        "",
        "---",
        f"_paper · VIRTUAL · margin={EXCESS_MARGIN:.0%} vs max(SPY,QQQ,IWM)_",
        "",
    ]
    summary_path = latest / "SUMMARY.md"
    summary_path.write_text("\n".join(md), encoding="utf-8")

    # also stamp history
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hist = out_root / "history" / stamp
    hist.mkdir(parents=True, exist_ok=True)
    for name in ("SUMMARY.md", "winners.json", "full_results.json"):
        src = latest / name
        if src.is_file():
            shutil.copy2(src, hist / name)
    if by_year_dir.is_dir():
        hby = hist / "by_year"
        if hby.exists():
            shutil.rmtree(hby, ignore_errors=True)
        shutil.copytree(by_year_dir, hby)

    return {
        "summary": summary_path,
        "winners": winners_path,
        "full": full_path,
        "latest": latest,
        "history": hist,
    }


# ---------------------------------------------------------------------------
# Full study runner
# ---------------------------------------------------------------------------


@dataclass
class MegaStudyResult:
    year_evals: List[YearEval] = field(default_factory=list)
    summaries: List[MultiYearSummary] = field(default_factory=list)
    tiers: Dict[str, List[MultiYearSummary]] = field(default_factory=dict)
    windows_meta: List[Dict[str, Any]] = field(default_factory=list)
    data_sources: Dict[str, str] = field(default_factory=dict)
    out_paths: Dict[str, str] = field(default_factory=dict)
    n_equity: int = 0
    n_options: int = 0
    capital0: float = 100_000.0
    force_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_year_evals": len(self.year_evals),
            "n_summaries": len(self.summaries),
            "n_equity": self.n_equity,
            "n_options": self.n_options,
            "capital0": self.capital0,
            "force_synthetic": self.force_synthetic,
            "windows": self.windows_meta,
            "data_sources": self.data_sources,
            "out_paths": self.out_paths,
            "strict_winners": [
                s.to_dict()
                for s in self.tiers.get(
                    f"{len(self.windows_meta)}/{len(self.windows_meta)}", []
                )
            ],
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }


def run_mega_annual_alpha_study(
    *,
    out_root: Union[str, Path] = DEFAULT_OUT,
    zoo_path: Optional[Union[str, Path]] = None,
    force_synthetic: bool = False,
    lookback_days: int = 1800,
    max_equity: Optional[int] = None,
    max_options: Optional[int] = None,
    skip_options: bool = False,
    skip_equity: bool = False,
    min_opens: int = 0,
    windows: Optional[Sequence[Tuple[str, str, str]]] = None,
    keep_ledgers: bool = False,
    lean_ledger: bool = True,
    min_real_tickers: int = 5,
) -> MegaStudyResult:
    """Run equity + options zoo annually; write report pack under out_root.

    ``lean_ledger`` (default True) avoids per-day JSONL audit files that made
    the full Yahoo matrix hang for multi-hour runs.
    """
    import pandas as pd

    from paper_live.cloud.free_data import SEED_DIR, build_cloud_feed
    from paper_live.options.replay_options import run_options_batch
    from paper_live.options.risk import OptionsRiskConfig

    zoo = load_mega_zoo(zoo_path)
    capital0 = float(zoo.get("capital0") or 100_000.0)
    eq_strats = equity_strategies_from_zoo(zoo)
    opt_strats = options_strategies_from_zoo(zoo)
    if max_equity is not None:
        eq_strats = eq_strats[: max(0, int(max_equity))]
    if max_options is not None:
        opt_strats = opt_strats[: max(0, int(max_options))]
    if skip_equity:
        eq_strats = []
    if skip_options:
        opt_strats = []

    universe = list(
        zoo.get("universe")
        or ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM", "QQQ", "SPY", "IWM"]
    )
    # ensure benchmarks + VIX for options surface
    for t in ("SPY", "QQQ", "IWM", "VIX", "VIX3M"):
        if t not in universe:
            universe.append(t)
    if opt_strats:
        for s in opt_strats:
            u = str(s.get("underlying") or "SPY").upper()
            if u not in universe:
                universe.append(u)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "data_cache"
    ledger_base = out_root / "_ledgers"
    # Clean incomplete prior ledgers unless keep_ledgers (avoid stale audit piles)
    if ledger_base.exists() and not keep_ledgers:
        shutil.rmtree(ledger_base, ignore_errors=True)
    if not lean_ledger or keep_ledgers:
        ledger_base.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Mega annual alpha: equity=%d options=%d lookback=%d synthetic=%s lean_ledger=%s",
        len(eq_strats),
        len(opt_strats),
        lookback_days,
        force_synthetic,
        lean_ledger,
    )
    feed, sources = build_cloud_feed(
        universe,
        cache_dir=cache_dir,
        seed_dir=SEED_DIR,
        lookback_calendar_days=int(lookback_days),
        force_synthetic=force_synthetic,
        require_real=not force_synthetic,
        min_real_tickers=int(min_real_tickers) if not force_synthetic else 0,
    )
    days = list(feed.days)
    if not days:
        raise RuntimeError("No trading days in feed")

    win_defs = list(windows) if windows else list(DEFAULT_WINDOWS)
    year_evals: List[YearEval] = []
    windows_meta: List[Dict[str, Any]] = []
    labels: Dict[str, Tuple[str, str]] = {}  # id -> (label, asset_class)

    risk_map = zoo.get("options_risk") or zoo.get("options", {}).get("risk") or {}
    global_risk = OptionsRiskConfig.from_mapping(risk_map) if risk_map else OptionsRiskConfig()

    for name, ws, we in win_defs:
        req_s = pd.Timestamp(ws).date()
        req_e = pd.Timestamp(we).date() if we != "2099-12-31" else days[-1]
        start_d, end_d, clamped = clamp_calendar_window(days, req_s, req_e)
        ben = compute_benchmarks(feed, start_d, end_d)
        wmeta = {
            "name": name,
            "start": start_d.isoformat(),
            "end": end_d.isoformat(),
            "requested_start": req_s.isoformat(),
            "requested_end": req_e.isoformat(),
            "clamped": clamped,
            **ben,
        }
        windows_meta.append(wmeta)
        logger.info(
            "Window %s %s→%s SPY=%s QQQ=%s IWM=%s",
            name,
            start_d,
            end_d,
            ben.get("spy_bh"),
            ben.get("qqq_bh"),
            ben.get("iwm_bh"),
        )

        # --- equity ---
        for i_eq, strat in enumerate(eq_strats, 1):
            sid = str(strat["id"])
            labels[sid] = (str(strat.get("label") or sid), "equity")
            led = ledger_base / name / sid
            logger.info(
                "Equity [%d/%d] %s @ %s",
                i_eq,
                len(eq_strats),
                sid,
                name,
            )
            try:
                raw = run_one_equity_strategy(
                    feed,
                    strat,
                    start_d=start_d,
                    end_d=end_d,
                    capital0=capital0,
                    ledger_root=led,
                    lean_ledger=lean_ledger,
                )
                ye = build_year_eval(
                    strategy_id=sid,
                    year=name,
                    total_return=float(raw["total_return"]),
                    benchmarks=ben,
                    max_dd=raw.get("max_dd"),
                    n_opens=int(raw.get("n_opens") or 0),
                    n_closed_trades=int(raw.get("n_closed_trades") or 0),
                    hard_kill=bool(raw.get("hard_kill")),
                    asset_class="equity",
                    signal_mode=str(raw.get("signal_mode") or ""),
                    window_start=start_d.isoformat(),
                    window_end=end_d.isoformat(),
                    clamped=clamped,
                    data_label="paper_equity",
                )
            except Exception as e:
                logger.exception("Equity %s @ %s failed", sid, name)
                ye = build_year_eval(
                    strategy_id=sid,
                    year=name,
                    total_return=0.0,
                    benchmarks=ben,
                    error=str(e),
                    asset_class="equity",
                    signal_mode=str(strat.get("signal_mode") or ""),
                    window_start=start_d.isoformat(),
                    window_end=end_d.isoformat(),
                    clamped=clamped,
                    data_label="paper_equity",
                )
            year_evals.append(ye)
            logger.info(
                "  → ret=%.2f%% beat+3pp=%s opens=%s err=%s",
                100.0 * float(ye.total_return),
                ye.beat_all_indices_by_3pp,
                ye.n_opens,
                ye.error,
            )
            if (not keep_ledgers) and led.exists():
                shutil.rmtree(led, ignore_errors=True)

        # --- options batch for this year ---
        if opt_strats:
            specs = options_specs_from_dicts(opt_strats)
            for sp in specs:
                labels[sp.id] = (sp.label, "options")
            try:
                results = run_options_batch(
                    feed,
                    specs,
                    start=start_d,
                    end=end_d,
                    capital0=capital0,
                    risk=global_risk,
                    data_label="proxy_bs",
                    spy_bh=ben.get("spy_bh"),
                    qqq_bh=ben.get("qqq_bh"),
                )
                by_id = {r.strategy_id: r for r in results}
                for sp in specs:
                    r = by_id.get(sp.id)
                    if r is None:
                        ye = build_year_eval(
                            strategy_id=sp.id,
                            year=name,
                            total_return=0.0,
                            benchmarks=ben,
                            error="missing_result",
                            asset_class="options",
                            kind=sp.kind,
                            window_start=start_d.isoformat(),
                            window_end=end_d.isoformat(),
                            clamped=clamped,
                            data_label="proxy_bs",
                        )
                    else:
                        ye = build_year_eval(
                            strategy_id=sp.id,
                            year=name,
                            total_return=float(r.total_return),
                            benchmarks=ben,
                            max_dd=float(r.max_dd),
                            n_opens=int(getattr(r, "n_opens", r.n_rolls) or 0),
                            n_closed_trades=int(getattr(r, "n_opens", r.n_rolls) or 0),
                            hard_kill=bool(r.hard_kill),
                            asset_class="options",
                            kind=sp.kind,
                            window_start=start_d.isoformat(),
                            window_end=end_d.isoformat(),
                            clamped=clamped,
                            data_label=str(r.data_label or "proxy_bs"),
                        )
                    year_evals.append(ye)
            except Exception as e:
                logger.exception("Options batch @ %s failed", name)
                for sp in specs:
                    year_evals.append(
                        build_year_eval(
                            strategy_id=sp.id,
                            year=name,
                            total_return=0.0,
                            benchmarks=ben,
                            error=str(e),
                            asset_class="options",
                            kind=sp.kind,
                            window_start=start_d.isoformat(),
                            window_end=end_d.isoformat(),
                            clamped=clamped,
                            data_label="proxy_bs",
                        )
                    )

    n_study = len(windows_meta)
    # summaries
    all_ids = sorted({y.strategy_id for y in year_evals})
    summaries: List[MultiYearSummary] = []
    for sid in all_ids:
        lab, aclass = labels.get(sid, (sid, "equity"))
        summaries.append(
            summarize_strategy_years(
                year_evals,
                strategy_id=sid,
                label=lab,
                asset_class=aclass,
                n_study_years=n_study,
            )
        )
    tiers = build_tier_tables(summaries, n_study_years=n_study, min_opens=min_opens)
    paths = write_report_pack(
        out_root=out_root,
        year_evals=year_evals,
        summaries=summaries,
        tiers=tiers,
        windows_meta=windows_meta,
        data_sources=sources,
        capital0=capital0,
        n_study_years=n_study,
        meta={
            "n_equity": len(eq_strats),
            "n_options": len(opt_strats),
            "force_synthetic": force_synthetic,
            "lookback_days": lookback_days,
            "lean_ledger": lean_ledger,
            "zoo_path": str(zoo_path or DEFAULT_ZOO),
        },
    )
    if not keep_ledgers and ledger_base.exists():
        shutil.rmtree(ledger_base, ignore_errors=True)

    return MegaStudyResult(
        year_evals=year_evals,
        summaries=summaries,
        tiers={k: list(v) for k, v in tiers.items()},
        windows_meta=windows_meta,
        data_sources=sources,
        out_paths={k: str(v) for k, v in paths.items()},
        n_equity=len(eq_strats),
        n_options=len(opt_strats),
        capital0=capital0,
        force_synthetic=force_synthetic,
    )
