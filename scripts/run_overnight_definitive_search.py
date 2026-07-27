"""Overnight definitive research search — screen/confirm/full stitch.

Integrates redesign_v2 mega zoo + controls + small pre-registered notches.
Resumes via PROGRESS.json; can seed incomplete work from redesign_v2 PROGRESS.

Protocol (pre-registered):
  - Screen OOS 2010–2017 (rank only)
  - Confirm OOS 2018–2025 (gates)
  - Full stitch 2010–2025
  - Gates: CAGR > 10%, MDD ≥ −65%, n_trades ≥ 80 on confirm (and full for research PASS)
  - honest_score ranking on confirm
  - No soft-ban tickers, no limit=54 as pre-reg winner, no paper freeze auto-change

Research only. No strategy is “definitive live”.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.backtest import BacktestConfig  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.redesign_v2.graph_math import (  # noqa: E402
    graph_summary_dict,
    graph_to_html,
    hub_scores,
    trade_cooccurrence_graph,
)
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

COMMISSION = 0.001
SLIPPAGE = 0.0005
GATE_CAGR = 0.10
GATE_MDD = -0.65
GATE_TRADES = 80

DEFAULT_SCREEN = (2010, 2017)
DEFAULT_CONFIRM = (2018, 2025)
DEFAULT_HOURS = 10.0
DEFAULT_MIN_TRAIN_ROWS = 1500
DEFAULT_OUT = ROOT / "reports" / "redesign" / "overnight_definitive"
DEFAULT_SEED_PROGRESS = ROOT / "reports" / "redesign" / "redesign_v2" / "PROGRESS.json"
DEFAULT_SEED_ARMS = ROOT / "reports" / "redesign" / "redesign_v2" / "arms"

# Base zoo (structurally distinct) — same as redesign_v2 mega
BASE_STRATEGIES: Tuple[str, ...] = (
    "turbo_highvol_minalloc",  # control / paper freeze
    "turbo_strict",
    "champion_ml",
    "r2_residual_mom",
    "r2_mom_sharpe",
    "r2_trend_stack",
    "r2_defensive_vt",
    "r2_rsi_reclaim",
)

UNIVERSE_ARMS: Tuple[Tuple[str, Path, int], ...] = (
    ("longhist_L50", ROOT / "universe_longhist2010_pass.txt", 50),
    ("longhist_L80", ROOT / "universe_longhist2010_pass.txt", 80),
    ("highvol2010_L50", ROOT / "universe_highvol80_2010_pass.txt", 50),
)

# Pre-registered residual_mom notches (0.03 = default already in base zoo)
RESID_MOM_NOTCHES: Tuple[float, ...] = (0.02, 0.05)

# Forced first arm when still incomplete
SEED_FORCE_ARM_ID = "turbo_strict__longhist_L80"

PAPER_FREEZE = "turbo_highvol_minalloc"
DISCLAIMER = "Research only. Not financial advice. No strategy is definitive live."


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def metric_float(metrics: Dict[str, Any], key: str, default: float) -> float:
    """Read float with explicit None-default (0.0 is a valid value)."""
    v = metrics.get(key, None)
    if v is None:
        return float(default)
    return float(v)


def metric_int(metrics: Dict[str, Any], key: str, default: int) -> int:
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
    """Apply research path gates. MDD=0.0 is valid (not missing)."""
    cagr = metric_float(metrics, "cagr", 0.0)
    mdd = metric_float(metrics, "max_drawdown", -1.0)
    n = metric_int(metrics, "n_trades", 0)
    ok_c = cagr > gate_cagr
    ok_m = mdd >= gate_mdd
    ok_t = n >= gate_trades
    return {
        "cagr_ok": ok_c,
        "mdd_ok": ok_m,
        "trades_ok": ok_t,
        "pass": bool(ok_c and ok_m and ok_t),
        "thresholds": {
            "cagr_gt": gate_cagr,
            "mdd_ge": gate_mdd,
            "n_trades_ge": gate_trades,
        },
    }


def honest_score(metrics: Dict[str, Any], excess_spy: Optional[float]) -> float:
    """Confirm ranking score (design SSOT).

    score = 2·cagr + 1·sortino + 0.5·max(0, excess_spy) − 2·max(0, −0.50 − mdd)
    """
    cagr = metric_float(metrics, "cagr", 0.0)
    sortino = metric_float(metrics, "sortino", 0.0)
    mdd = metric_float(metrics, "max_drawdown", -1.0)
    score = 2.0 * cagr + 1.0 * sortino
    if excess_spy is not None:
        score += 0.5 * max(0.0, float(excess_spy))
    if mdd < -0.50:
        score -= 2.0 * ((-0.50) - mdd)
    return float(score)


def research_pass_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Arms that pass confirm gates AND full-path gates."""
    out: List[str] = []
    for r in rows:
        if r.get("error"):
            continue
        cg = (r.get("confirm") or {}).get("gates") or {}
        fg = (r.get("full") or {}).get("gates") or {}
        if cg.get("pass") and fg.get("pass"):
            out.append(str(r.get("arm_id")))
    return out


def confirm_pass_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for r in rows:
        if r.get("error"):
            continue
        cg = (r.get("confirm") or {}).get("gates") or {}
        if cg.get("pass"):
            out.append(str(r.get("arm_id")))
    return out


def rank_by_honest_score(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort clean rows by honest_score descending."""
    clean = [r for r in rows if not r.get("error") and r.get("honest_score") is not None]
    return sorted(clean, key=lambda r: float(r.get("honest_score") or -999.0), reverse=True)


def decide_verdict(
    rows: Sequence[Dict[str, Any]],
    *,
    complete: bool,
) -> Dict[str, Any]:
    """Honest research decision. PASS only if confirm∩full; never claims live."""
    ranked = rank_by_honest_score(rows)
    c_pass = confirm_pass_ids(rows)
    r_pass = research_pass_ids(rows)
    if r_pass:
        status = "PASS"
        message = (
            "Research PASS: ≥1 arm with confirm∩full gates. "
            "Not definitive live; paper freeze unchanged."
        )
    elif c_pass:
        status = "HOLD"
        message = (
            "Confirm passers exist but full stitch failed gates (or incomplete). "
            "Not research PASS."
        )
    else:
        status = "FAIL"
        message = "No confirm∩full research PASS. Honest FAIL."
    if not complete:
        status = f"PARTIAL_{status}"
        message = "Run incomplete (hours/resume). " + message
    best = ranked[0] if ranked else None
    return {
        "status": status,
        "message": message,
        "confirm_passers": c_pass,
        "research_pass": r_pass,
        "best_arm_id": (best or {}).get("arm_id"),
        "best_honest_score": (best or {}).get("honest_score"),
        "complete": bool(complete),
        "live_claim": False,
        "paper_freeze": PAPER_FREEZE,
        "disclaimer": DISCLAIMER,
    }


def prioritize_arm_ids(
    arm_ids: Sequence[str],
    *,
    done: Sequence[str],
    force_first: str = SEED_FORCE_ARM_ID,
) -> List[str]:
    """Order pending arms: force seed first if incomplete, else preserve zoo order.

    ``done`` must be *successful* completions only. Failed arms are omitted from
    done so they re-queue automatically on resume.
    """
    done_set = set(done)
    pending = [a for a in arm_ids if a not in done_set]
    if not pending:
        return []
    if force_first in pending:
        rest = [a for a in pending if a != force_first]
        return [force_first] + rest
    return pending


def row_is_success(row: Dict[str, Any]) -> bool:
    """True when arm finished without exception and has confirm metrics."""
    if row.get("error"):
        return False
    confirm = row.get("confirm")
    if not isinstance(confirm, dict) or not confirm:
        return False
    if confirm.get("error"):
        return False
    # Confirm window produced metrics (cagr key present, including 0.0 / negative)
    return "cagr" in confirm or "gates" in confirm


def zoo_complete(arm_ids: Sequence[str], done: Sequence[str]) -> bool:
    """True iff every planned arm_id is in successful done (set coverage)."""
    planned = {str(a) for a in arm_ids}
    if not planned:
        return True
    return planned <= {str(d) for d in done}


def partition_done_failed(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Split rows into successful done ids vs failed/error ids (order preserved).

    Last row wins per arm_id (local-wins when iterating local after seed).
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in rows:
        aid = str(r.get("arm_id") or "")
        if not aid:
            continue
        if aid not in by_id:
            order.append(aid)
        by_id[aid] = r
    done: List[str] = []
    failed: List[str] = []
    for aid in order:
        if row_is_success(by_id[aid]):
            done.append(aid)
        else:
            failed.append(aid)
    return done, failed


def finalize_stop_reason(
    arm_ids: Sequence[str],
    done: Sequence[str],
    *,
    hours_exhausted: bool = False,
    accept_errors: bool = False,
    failed: Sequence[str] = (),
    prior_stop: Optional[str] = None,
) -> str:
    """Compute stop_reason from zoo coverage; seed_partial never blocks complete.

    Completion is set equality / subset: ``set(arm_ids) <= set(effective_done)``.
    ``prior_stop`` values like ``seeded_partial`` are ignored for completion.
    """
    effective = set(str(x) for x in done)
    if accept_errors:
        effective |= {str(x) for x in failed}
    if zoo_complete(arm_ids, list(effective)):
        return "complete"
    # hours cut only if not fully covered
    if hours_exhausted or prior_stop == "hours_exhausted":
        return "hours_exhausted"
    if prior_stop == "seeded_partial":
        return "incomplete"
    if prior_stop in ("incomplete", None, "seeded_partial"):
        return "incomplete"
    # preserve unknown prior only when still incomplete
    if prior_stop and prior_stop not in ("complete",):
        return str(prior_stop)
    return "incomplete"


def is_run_complete(
    arm_ids: Sequence[str],
    done: Sequence[str],
    *,
    accept_errors: bool = False,
    failed: Sequence[str] = (),
    stop_reason: Optional[str] = None,
) -> bool:
    """Research-complete if zoo covered; stop_reason alone never decides.

    ``stop_reason='seeded_partial'`` does **not** force incomplete when done
    already covers the zoo (e.g. seed imported full set).
    """
    effective = list(done)
    if accept_errors:
        effective = list(set(done) | set(failed))
    covered = zoo_complete(arm_ids, effective)
    if not covered:
        return False
    # hours_exhausted with full coverage still complete
    _ = stop_reason  # documented: ignored when coverage holds
    return True


def record_arm_outcome(
    state: Dict[str, Any],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """Update rows/done/failed after one arm attempt. Errors stay out of done."""
    aid = str(row.get("arm_id") or "")
    rows = [r for r in (state.get("rows") or []) if str(r.get("arm_id")) != aid]
    rows.append(row)
    state["rows"] = rows
    done = [d for d in (state.get("done") or []) if d != aid]
    failed = [f for f in (state.get("failed") or []) if f != aid]
    if row_is_success(row):
        done.append(aid)
    else:
        failed.append(aid)
    state["done"] = done
    state["failed"] = failed
    return state


def fixup_full_trade_count(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure full.n_trades reflects screen+confirm when full path omitted trades.

    redesign_v2 mega wrote full metrics with trades=None → n_trades=0, which
    falsely fails GATE_TRADES. Prefer sum of window trade counts.
    """
    out = dict(row)
    full = dict(out.get("full") or {})
    if full.get("error"):
        return out
    screen = out.get("screen") or {}
    confirm = out.get("confirm") or {}
    n_s = metric_int(screen, "n_trades", 0)
    n_c = metric_int(confirm, "n_trades", 0)
    n_f = metric_int(full, "n_trades", 0)
    n_sum = n_s + n_c
    if n_sum > n_f:
        full["n_trades"] = int(n_sum)
        full["gates"] = apply_path_gates(full)
        out["full"] = full
    elif "gates" not in full or full.get("gates") is None:
        full["gates"] = apply_path_gates(full)
        out["full"] = full
    # Re-apply confirm gates if missing thresholds (seed hygiene)
    if confirm and "gates" in confirm:
        cg = apply_path_gates(confirm)
        conf = dict(confirm)
        conf["gates"] = cg
        out["confirm"] = conf
        xs = conf.get("excess_spy_total")
        out["honest_score"] = honest_score(conf, xs if xs is not None else None)
    return out


def merge_progress_rows(
    local_rows: Sequence[Dict[str, Any]],
    seed_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Merge seed rows under local; local wins on arm_id conflict.

    Returns ``(done, failed, rows)`` where done is successful arms only.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in seed_rows:
        aid = str(r.get("arm_id") or "")
        if not aid:
            continue
        if aid not in by_id:
            order.append(aid)
        by_id[aid] = fixup_full_trade_count(dict(r))
    for r in local_rows:
        aid = str(r.get("arm_id") or "")
        if not aid:
            continue
        if aid not in by_id:
            order.append(aid)
        by_id[aid] = fixup_full_trade_count(dict(r))
    rows = [by_id[k] for k in order]
    done, failed = partition_done_failed(rows)
    return done, failed, rows


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------


@dataclass
class Arm:
    arm_id: str
    strategy: str
    universe_label: str
    ticker_file: Path
    universe_limit: int
    param_tag: str = ""
    param_overrides: Dict[str, Any] = field(default_factory=dict)


def resolve_universe_path(path: Path, *, repo_root: Optional[Path] = None) -> Optional[Path]:
    if path.is_file():
        return path
    roots = [repo_root, ROOT] if repo_root is not None else [ROOT]
    name = path.name
    for r in roots:
        if r is None:
            continue
        cand = Path(r) / name
        if cand.is_file():
            return cand
        cand2 = Path(r) / path
        if cand2.is_file():
            return cand2
    fb_roots = roots
    for r in fb_roots:
        if r is None:
            continue
        fb = Path(r) / "universe_longhist100.txt"
        if fb.is_file():
            return fb
    return None


def universe_arms_for_root(repo_root: Path) -> List[Tuple[str, Path, int]]:
    """Universe files relative to repo_root (Kaggle input or local ROOT)."""
    rr = Path(repo_root)
    return [
        ("longhist_L50", rr / "universe_longhist2010_pass.txt", 50),
        ("longhist_L80", rr / "universe_longhist2010_pass.txt", 80),
        ("highvol2010_L50", rr / "universe_highvol80_2010_pass.txt", 50),
    ]


def build_arms(
    *,
    strategies: Sequence[str] = BASE_STRATEGIES,
    universe_arms: Optional[Sequence[Tuple[str, Path, int]]] = None,
    resid_notches: Sequence[float] = RESID_MOM_NOTCHES,
    include_notches: bool = True,
    repo_root: Optional[Path] = None,
) -> List[Arm]:
    """Build pre-registered arm list: base zoo + optional residual_mom notches."""
    arms: List[Arm] = []
    seen: set = set()
    uarms = list(universe_arms) if universe_arms is not None else list(UNIVERSE_ARMS)
    if repo_root is not None:
        uarms = universe_arms_for_root(Path(repo_root))

    for ulab, upath, lim in uarms:
        resolved = resolve_universe_path(Path(upath), repo_root=repo_root)
        if resolved is None:
            continue
        for s in strategies:
            aid = f"{s}__{ulab}"
            if aid in seen:
                continue
            seen.add(aid)
            arms.append(
                Arm(
                    arm_id=aid,
                    strategy=s,
                    universe_label=ulab,
                    ticker_file=resolved,
                    universe_limit=int(lim),
                )
            )

    if include_notches:
        for ulab, upath, lim in uarms:
            resolved = resolve_universe_path(Path(upath), repo_root=repo_root)
            if resolved is None:
                continue
            for mr in resid_notches:
                # Skip 0.03 — identical to base r2_residual_mom
                if abs(float(mr) - 0.03) < 1e-12:
                    continue
                tag = f"mr{mr:g}".replace(".", "p")
                aid = f"r2_residual_mom_{tag}__{ulab}"
                if aid in seen:
                    continue
                seen.add(aid)
                arms.append(
                    Arm(
                        arm_id=aid,
                        strategy="r2_residual_mom",
                        universe_label=ulab,
                        ticker_file=resolved,
                        universe_limit=int(lim),
                        param_tag=tag,
                        param_overrides={"min_resid": float(mr)},
                    )
                )
    return arms


# ---------------------------------------------------------------------------
# Metrics / run path
# ---------------------------------------------------------------------------


def _eq_norm(s: pd.Series) -> pd.Series:
    out = s.dropna().astype(float)
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    return out[~out.index.duplicated(keep="last")].dropna().sort_index()


def stitch_equity(seg_a: pd.Series, seg_b: pd.Series) -> pd.Series:
    """Stitch two equity segments with capital continuity."""
    segs: List[pd.Series] = []
    prev: Optional[float] = None
    for seg in (seg_a, seg_b):
        s = _eq_norm(seg)
        if s.empty:
            continue
        if prev is not None and float(s.iloc[0]) != 0:
            s = s * (prev / float(s.iloc[0]))
        segs.append(s)
        prev = float(s.iloc[-1])
    if not segs:
        return pd.Series(dtype=float)
    out = pd.concat(segs)
    return out[~out.index.duplicated(keep="last")].sort_index()


def path_metrics(eq: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    eq = _eq_norm(eq)
    if eq.empty:
        return {"error": "empty", "cagr": 0.0, "max_drawdown": -1.0, "n_trades": 0, "sortino": 0.0}
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
    }


def spy_excess(eq: pd.Series, data_root: Path) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, eq.index.min(), eq.index.max(), preferred=["SPY"]
        )
        if b is None or b.empty:
            return None
        j = pd.concat(
            [_eq_norm(eq).rename("s"), _eq_norm(b).rename("b")], axis=1, join="inner"
        ).dropna()
        if len(j) < 5:
            return None
        return float(j["s"].iloc[-1] / j["s"].iloc[0] - j["b"].iloc[-1] / j["b"].iloc[0])
    except Exception:
        return None


def run_window(
    strategy: str,
    *,
    first: int,
    last: int,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    min_train_rows: int,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    strat = get_strategy(strategy)
    if param_overrides:
        for k, v in param_overrides.items():
            if hasattr(strat, k):
                setattr(strat, k, v)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(ticker_file)
    base = strat.backtest_overrides() if hasattr(strat, "backtest_overrides") else {}
    merged = {**base, "commission": COMMISSION, "slippage": SLIPPAGE}

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
            universe_limit=int(universe_limit),
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


def save_progress(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def equity_chart_html(series_map: Dict[str, pd.Series], *, title: str) -> str:
    colors = ["#4cc9f0", "#f72585", "#ffb020", "#80ed99", "#c77dff", "#ff6b6b", "#90e0ef"]
    paths = []
    legend = []
    w, h, pad = 900, 360, 40
    for i, (name, eq) in enumerate(series_map.items()):
        s = _eq_norm(eq)
        if s.empty or len(s) < 2:
            continue
        s = s / float(s.iloc[0])
        xs = np.linspace(pad, w - pad, len(s))
        ymin, ymax = float(s.min()), float(s.max())
        if ymax <= ymin:
            ymax = ymin + 1e-6
        ys = pad + (1.0 - (s.to_numpy(dtype=float) - ymin) / (ymax - ymin)) * (h - 2 * pad)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        col = colors[i % len(colors)]
        paths.append(
            f"<polyline fill='none' stroke='{col}' stroke-width='2' points='{pts}' />"
        )
        legend.append(f"<span style='color:{col}'>■</span> {name} &nbsp;")
    return (
        f"<div><h3>{title}</h3><div>{''.join(legend)}</div>"
        f"<svg width='{w}' height='{h}' style='background:#121a2f;border-radius:8px'>"
        f"{''.join(paths)}</svg></div>"
    )


def _copy_seed_arm_artifacts(
    arm_id: str,
    *,
    seed_arms: Path,
    dest_arms: Path,
) -> None:
    src = seed_arms / arm_id.replace("/", "_")
    dst = dest_arms / arm_id.replace("/", "_")
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            target = dst / p.name
            if not target.is_file():
                shutil.copy2(p, target)


def seed_from_redesign_v2(
    *,
    seed_progress: Path,
    seed_arms: Path,
    dest_arms: Path,
    known_arm_ids: Sequence[str],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Import redesign_v2 rows that match zoo arm_ids.

    Only successful rows enter ``done``; error / incomplete rows go to ``failed``
    (retry on resume). Unknown arm_ids are skipped.
    """
    if not seed_progress.is_file():
        return [], [], []
    try:
        prev = json.loads(seed_progress.read_text(encoding="utf-8"))
    except Exception:
        return [], [], []
    known = set(known_arm_ids)
    rows: List[Dict[str, Any]] = []
    for r in prev.get("rows") or []:
        aid = str(r.get("arm_id") or "")
        if aid not in known:
            continue
        rr = fixup_full_trade_count(dict(r))
        rows.append(rr)
        _copy_seed_arm_artifacts(aid, seed_arms=seed_arms, dest_arms=dest_arms)
    done, failed = partition_done_failed(rows)
    return done, failed, rows


def write_reports(
    *,
    out: Path,
    arms: Sequence[Arm],
    state: Dict[str, Any],
    screen: Tuple[int, int],
    confirm: Tuple[int, int],
) -> Dict[str, Any]:
    arms_dir = out / "arms"
    graphs_dir = out / "graphs"
    rows = list(state.get("rows") or [])
    ranked = rank_by_honest_score(rows)
    c_pass = confirm_pass_ids(rows)
    r_pass = research_pass_ids(rows)
    arm_ids = [a.arm_id for a in arms]
    n_arms = len(arms)
    n_done = len(state.get("done") or [])
    n_failed = len(state.get("failed") or [])
    accept_errors = bool(state.get("accept_errors"))
    complete = is_run_complete(
        arm_ids,
        state.get("done") or [],
        accept_errors=accept_errors,
        failed=state.get("failed") or [],
        stop_reason=state.get("stop_reason"),
    )
    decision = decide_verdict(rows, complete=complete)

    chart_map: Dict[str, pd.Series] = {}
    for r in ranked[:5]:
        p = arms_dir / str(r["arm_id"]).replace("/", "_") / "equity_confirm.csv"
        if p.is_file():
            eq = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            chart_map[str(r["arm_id"])] = eq

    dash = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Overnight definitive</title>",
        "<style>body{font-family:system-ui;background:#0b1020;color:#e8ecf5;padding:24px}",
        "table{border-collapse:collapse} td,th{border:1px solid #334;padding:6px 10px}",
        "a{color:#8ecae6}.pass{color:#80ed99}.fail{color:#ff6b6b}</style></head><body>",
        "<h1>Overnight definitive search — screen/confirm/full</h1>",
        f"<p>Generated {datetime.now(timezone.utc).isoformat()} · "
        f"stop={state.get('stop_reason')} · done={n_done}/{n_arms} · "
        f"failed={n_failed} · complete={complete} · "
        f"verdict=<strong>{decision['status']}</strong></p>",
        f"<p>{DISCLAIMER}</p>",
        equity_chart_html(chart_map, title="Confirm equity (top honest_score)"),
        "<h2>Leaderboard (confirm honest_score)</h2>",
        "<table><tr><th>arm</th><th>confirm CAGR</th><th>MDD</th><th>n</th>"
        "<th>pass</th><th>screen CAGR</th><th>full CAGR</th><th>full pass</th>"
        "<th>score</th></tr>",
    ]
    for r in ranked:
        c = r.get("confirm") or {}
        s = r.get("screen") or {}
        f = r.get("full") or {}
        cp = (c.get("gates") or {}).get("pass")
        fp = (f.get("gates") or {}).get("pass")
        dash.append(
            f"<tr><td><code>{r.get('arm_id')}</code></td>"
            f"<td>{100 * float(c.get('cagr') or 0):.1f}%</td>"
            f"<td>{100 * float(c.get('max_drawdown') or 0):.1f}%</td>"
            f"<td>{c.get('n_trades')}</td>"
            f"<td class='{'pass' if cp else 'fail'}'>{cp}</td>"
            f"<td>{100 * float(s.get('cagr') or 0):.1f}%</td>"
            f"<td>{100 * float(f.get('cagr') or 0):.1f}%</td>"
            f"<td class='{'pass' if fp else 'fail'}'>{fp}</td>"
            f"<td>{float(r.get('honest_score') or 0):.3f}</td></tr>"
        )
    dash.append("</table>")
    dash.append("<h2>Graphs</h2><ul>")
    if graphs_dir.is_dir():
        for g in sorted(graphs_dir.glob("*.html")):
            dash.append(f"<li><a href='graphs/{g.name}'>{g.name}</a></li>")
    dash.append(f"</ul><p>{DISCLAIMER}</p></body></html>")
    (out / "dashboard.html").write_text("\n".join(dash), encoding="utf-8")

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "stop_reason": state.get("stop_reason"),
        "n_done": n_done,
        "n_failed": n_failed,
        "n_arms": n_arms,
        "zoo_complete": complete,
        "screen": f"{screen[0]}-{screen[1]}",
        "confirm": f"{confirm[0]}-{confirm[1]}",
        "gates": {
            "cagr_gt": GATE_CAGR,
            "mdd_ge": GATE_MDD,
            "n_trades_ge": GATE_TRADES,
        },
        "confirm_passers": c_pass,
        "research_pass": r_pass,
        "decision": decision,
        "ranked": ranked,
        "done": list(state.get("done") or []),
        "failed": list(state.get("failed") or []),
        "seeded_from": state.get("seeded_from"),
        "paper_freeze": PAPER_FREEZE,
        "live_claim": False,
        "disclaimer": DISCLAIMER,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Overnight definitive search — SUMMARY",
        "",
        f"> **{DISCLAIMER}**",
        "",
        f"- Stop: **{state.get('stop_reason')}** · done **{n_done}/{n_arms}** · "
        f"failed **{n_failed}** · zoo_complete **{complete}**",
        f"- Screen **{screen[0]}–{screen[1]}** · Confirm **{confirm[0]}–{confirm[1]}**",
        f"- Gates: CAGR>{GATE_CAGR:.0%} · MDD≥{GATE_MDD:.0%} · n≥{GATE_TRADES}",
        f"- Confirm passers: `{', '.join(c_pass) or 'none'}`",
        f"- Research PASS (confirm∩full): `{', '.join(r_pass) or 'none'}`",
        f"- Verdict: **{decision['status']}** — {decision['message']}",
        f"- Paper freeze: `{PAPER_FREEZE}` **unchanged**",
        f"- Seeded from: `{state.get('seeded_from') or 'none'}`",
        f"- Failed (retry on resume): `{', '.join(state.get('failed') or []) or 'none'}`",
        "",
        "## Leaderboard (honest_score on confirm)",
        "",
        "| arm | confirm CAGR | MDD | n | pass | screen CAGR | full CAGR | full pass | score |",
        "|-----|--------------|-----|---|------|-------------|-----------|-----------|-------|",
    ]
    for r in ranked:
        c = r.get("confirm") or {}
        s = r.get("screen") or {}
        f = r.get("full") or {}
        lines.append(
            f"| `{r.get('arm_id')}` | {100 * float(c.get('cagr') or 0):.1f}% | "
            f"{100 * float(c.get('max_drawdown') or 0):.1f}% | "
            f"{c.get('n_trades')} | {(c.get('gates') or {}).get('pass')} | "
            f"{100 * float(s.get('cagr') or 0):.1f}% | "
            f"{100 * float(f.get('cagr') or 0):.1f}% | "
            f"{(f.get('gates') or {}).get('pass')} | "
            f"{float(r.get('honest_score') or 0):.3f} |"
        )
    lines += [
        "",
        f"[Dashboard](dashboard.html) · [DECISION](DECISION.md)",
        "",
        "## Resume",
        "",
        "```powershell",
        f"python scripts/run_overnight_definitive_search.py --hours 10 --out {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}",
        "```",
        "",
        DISCLAIMER,
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    dlines = [
        "# Overnight definitive search — DECISION",
        "",
        f"**Status:** `{decision['status']}`",
        "",
        decision["message"],
        "",
        f"- Complete run: **{complete}**",
        f"- Arms done: **{n_done}/{n_arms}**",
        f"- Confirm passers: `{', '.join(c_pass) or 'none'}`",
        f"- Research PASS (confirm∩full): `{', '.join(r_pass) or 'none'}`",
        f"- Best honest_score arm: `{decision.get('best_arm_id') or 'n/a'}` "
        f"(score={decision.get('best_honest_score')})",
        "",
        "## Rules",
        "",
        "- Research winner **only** if confirm∩full PASS.",
        "- **No strategy is definitive live.**",
        f"- Paper freeze stays **`{PAPER_FREEZE}`** (not auto-changed).",
        "- No soft-ban; no limit=54 pre-reg winner.",
        "",
        "## Gates",
        "",
        f"- CAGR > {GATE_CAGR:.0%}",
        f"- MDD ≥ {GATE_MDD:.0%}",
        f"- n_trades ≥ {GATE_TRADES} (confirm; full uses same path gates)",
        "",
        f"**Live claim:** {decision['live_claim']}",
        "",
        DISCLAIMER,
        "",
    ]
    (out / "DECISION.md").write_text("\n".join(dlines), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Root with universe_*.txt (Kaggle input root). Default: script parents.",
    )
    ap.add_argument(
        "--shard",
        type=str,
        default="",
        help="Optional shard i/n for multi-GPU workers, e.g. 0/2 and 1/2 on T4x2",
    )
    ap.add_argument("--screen-first", type=int, default=DEFAULT_SCREEN[0])
    ap.add_argument("--screen-last", type=int, default=DEFAULT_SCREEN[1])
    ap.add_argument("--confirm-first", type=int, default=DEFAULT_CONFIRM[0])
    ap.add_argument("--confirm-last", type=int, default=DEFAULT_CONFIRM[1])
    ap.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--strategies",
        type=str,
        default=",".join(BASE_STRATEGIES),
        help="Comma base strategies",
    )
    ap.add_argument(
        "--no-notches",
        action="store_true",
        help="Skip residual_mom min_resid notches",
    )
    ap.add_argument(
        "--seed-progress",
        type=Path,
        default=DEFAULT_SEED_PROGRESS,
        help="redesign_v2 PROGRESS.json to import if local empty",
    )
    ap.add_argument(
        "--seed-arms",
        type=Path,
        default=DEFAULT_SEED_ARMS,
        help="redesign_v2 arms/ dir to copy artifacts from",
    )
    ap.add_argument("--no-seed", action="store_true", help="Do not import redesign_v2")
    ap.add_argument("--max-arms", type=int, default=0, help="0=all (debug cap)")
    ap.add_argument(
        "--force-first",
        type=str,
        default=SEED_FORCE_ARM_ID,
        help="Arm id to run first if incomplete",
    )
    ap.add_argument(
        "--no-retry-errors",
        action="store_true",
        help="Do not re-queue failed arms (default retries errors on resume)",
    )
    ap.add_argument(
        "--accept-errors",
        action="store_true",
        help="Treat failed arms as zoo coverage for complete (not research success)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    arms_dir = out / "arms"
    graphs_dir = out / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    arms_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)
    prog_path = out / "PROGRESS.json"

    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    repo_root = Path(args.repo_root) if args.repo_root else ROOT
    arms = build_arms(
        strategies=strategies,
        include_notches=not bool(args.no_notches),
        repo_root=repo_root,
    )
    if int(args.max_arms) > 0:
        arms = arms[: int(args.max_arms)]
    # Multi-GPU shard: keep even/odd indices (stable order)
    shard_s = str(args.shard or "").strip()
    if shard_s and "/" in shard_s:
        try:
            si, sn = shard_s.split("/", 1)
            si_i, sn_i = int(si), int(sn)
            if sn_i > 1:
                arms = [a for j, a in enumerate(arms) if j % sn_i == si_i]
                print(f"Shard {si_i}/{sn_i} → {len(arms)} arms", flush=True)
        except ValueError:
            print(f"WARN bad --shard {shard_s!r}, ignoring", flush=True)
    arm_by_id = {a.arm_id: a for a in arms}
    arm_ids = [a.arm_id for a in arms]

    t0 = time.time()
    deadline = t0 + float(args.hours) * 3600.0
    retry_errors = not bool(args.no_retry_errors)
    accept_errors = bool(args.accept_errors)
    state: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "hours": float(args.hours),
        "n_arms": len(arms),
        "arm_ids": arm_ids,
        "done": [],
        "failed": [],
        "rows": [],
        "stop_reason": None,
        "seeded_from": None,
        "accept_errors": accept_errors,
        "retry_errors": retry_errors,
        "protocol": {
            "screen": f"{args.screen_first}-{args.screen_last}",
            "confirm": f"{args.confirm_first}-{args.confirm_last}",
            "gates": {
                "cagr_gt": GATE_CAGR,
                "mdd_ge": GATE_MDD,
                "n_trades_ge": GATE_TRADES,
            },
            "force_first": args.force_first,
            "resid_notches": list(RESID_MOM_NOTCHES),
        },
    }

    if prog_path.is_file():
        try:
            prev = json.loads(prog_path.read_text(encoding="utf-8"))
            raw_rows = [fixup_full_trade_count(dict(r)) for r in (prev.get("rows") or [])]
            state["rows"] = raw_rows
            # Re-partition: error arms never sticky-success (fixes old PROGRESS)
            done_ids, failed_ids = partition_done_failed(raw_rows)
            state["done"] = done_ids
            state["failed"] = failed_ids
            state["seeded_from"] = prev.get("seeded_from")
            if prev.get("started"):
                state["started"] = prev["started"]
            # Do not restore stop_reason=seeded_partial — blocks false complete
            print(
                f"Resume local done={len(state['done'])} failed={len(state['failed'])}",
                flush=True,
            )
        except Exception as e:
            print(f"WARN could not load PROGRESS: {e}", flush=True)

    if not state["done"] and not state["failed"] and not args.no_seed:
        seed_prog = Path(args.seed_progress)
        if not seed_prog.is_absolute():
            seed_prog = ROOT / seed_prog
        seed_arms = Path(args.seed_arms)
        if not seed_arms.is_absolute():
            seed_arms = ROOT / seed_arms
        _s_done, _s_failed, s_rows = seed_from_redesign_v2(
            seed_progress=seed_prog,
            seed_arms=seed_arms,
            dest_arms=arms_dir,
            known_arm_ids=arm_ids,
        )
        if s_rows:
            state["done"], state["failed"], state["rows"] = merge_progress_rows(
                [], s_rows
            )
            state["seeded_from"] = str(seed_prog)
            # Membership-based stop only — never sticky seeded_partial for completion
            state["stop_reason"] = finalize_stop_reason(
                arm_ids,
                state["done"],
                hours_exhausted=False,
                accept_errors=accept_errors,
                failed=state["failed"],
                prior_stop="seeded_partial",
            )
            save_progress(prog_path, state)
            write_reports(
                out=out,
                arms=arms,
                state=state,
                screen=(int(args.screen_first), int(args.screen_last)),
                confirm=(int(args.confirm_first), int(args.confirm_last)),
            )
            print(
                f"Seeded done={len(state['done'])} failed={len(state['failed'])} "
                f"from redesign_v2 → {out}",
                flush=True,
            )

    # Clear non-terminal stop markers before arm loop so finish can set complete
    if state.get("stop_reason") in ("seeded_partial", "incomplete", None):
        state["stop_reason"] = None

    skip_set = set(state["done"])
    if not retry_errors:
        # Sticky failures: do not re-run, but they remain out of successful done
        skip_set |= set(state.get("failed") or [])
    pending = prioritize_arm_ids(
        arm_ids, done=list(skip_set), force_first=str(args.force_first)
    )
    print(
        f"Overnight definitive arms={len(arms)} pending={len(pending)} "
        f"done={len(state['done'])} failed={len(state.get('failed') or [])} "
        f"hours={args.hours} screen={args.screen_first}-{args.screen_last} "
        f"confirm={args.confirm_first}-{args.confirm_last}",
        flush=True,
    )

    hours_exhausted = False
    for aid in pending:
        if time.time() > deadline:
            hours_exhausted = True
            state["stop_reason"] = "hours_exhausted"
            break
        arm = arm_by_id[aid]
        print(f"[arm] {arm.arm_id} …", flush=True)
        row: Dict[str, Any] = {
            "arm_id": arm.arm_id,
            "strategy": arm.strategy,
            "universe": arm.universe_label,
            "limit": arm.universe_limit,
            "param_tag": arm.param_tag or None,
            "param_overrides": arm.param_overrides or None,
        }
        adir = arms_dir / arm.arm_id.replace("/", "_")
        adir.mkdir(parents=True, exist_ok=True)
        try:
            rs = run_window(
                arm.strategy,
                first=int(args.screen_first),
                last=int(args.screen_last),
                data_root=Path(args.data_root),
                ticker_file=arm.ticker_file,
                universe_limit=arm.universe_limit,
                min_train_rows=int(args.min_train_rows),
                param_overrides=arm.param_overrides or None,
            )
            eq_s = rs.get("equity")
            tr_s = (
                rs.get("trades")
                if isinstance(rs.get("trades"), pd.DataFrame)
                else pd.DataFrame()
            )
            m_s = path_metrics(eq_s, tr_s) if isinstance(eq_s, pd.Series) else {"error": "empty"}
            g_s = apply_path_gates(m_s)
            row["screen"] = {**m_s, "gates": g_s}
            if isinstance(eq_s, pd.Series):
                _eq_norm(eq_s).to_csv(adir / "equity_screen.csv", header=["equity"])
            if isinstance(tr_s, pd.DataFrame) and not tr_s.empty:
                tr_s.to_csv(adir / "trades_screen.csv", index=False)

            rc = run_window(
                arm.strategy,
                first=int(args.confirm_first),
                last=int(args.confirm_last),
                data_root=Path(args.data_root),
                ticker_file=arm.ticker_file,
                universe_limit=arm.universe_limit,
                min_train_rows=int(args.min_train_rows),
                param_overrides=arm.param_overrides or None,
            )
            eq_c = rc.get("equity")
            tr_c = (
                rc.get("trades")
                if isinstance(rc.get("trades"), pd.DataFrame)
                else pd.DataFrame()
            )
            m_c = path_metrics(eq_c, tr_c) if isinstance(eq_c, pd.Series) else {"error": "empty"}
            g_c = apply_path_gates(m_c)
            xs = spy_excess(eq_c, Path(args.data_root)) if isinstance(eq_c, pd.Series) else None
            row["confirm"] = {**m_c, "gates": g_c, "excess_spy_total": xs}
            row["honest_score"] = honest_score(m_c, xs)
            if isinstance(eq_c, pd.Series):
                _eq_norm(eq_c).to_csv(adir / "equity_confirm.csv", header=["equity"])
            if isinstance(tr_c, pd.DataFrame) and not tr_c.empty:
                tr_c.to_csv(adir / "trades_confirm.csv", index=False)
                edges = trade_cooccurrence_graph(tr_c)
                hubs = hub_scores(edges)
                (graphs_dir / f"{arm.arm_id}_cooccur.html").write_text(
                    graph_to_html(
                        edges,
                        title=f"Trade co-occurrence confirm — {arm.arm_id}",
                        hubs=hubs,
                    ),
                    encoding="utf-8",
                )
                row["graph"] = graph_summary_dict(edges)

            if isinstance(eq_s, pd.Series) and isinstance(eq_c, pd.Series):
                eq_f = stitch_equity(eq_s, eq_c)
                # Full n_trades = screen + confirm when both available
                tr_f = None
                if isinstance(tr_s, pd.DataFrame) and isinstance(tr_c, pd.DataFrame):
                    if not tr_s.empty or not tr_c.empty:
                        tr_f = pd.concat([tr_s, tr_c], ignore_index=True)
                m_f = path_metrics(eq_f, tr_f)
                g_f = apply_path_gates(m_f)
                row["full"] = {**m_f, "gates": g_f}
                eq_f.to_csv(adir / "equity_full.csv", header=["equity"])
            else:
                row["full"] = {"error": "missing_segment"}

            print(
                f"  screen_cagr={m_s.get('cagr')} confirm_cagr={m_c.get('cagr')} "
                f"confirm_pass={g_c.get('pass')} score={row.get('honest_score')}",
                flush=True,
            )
        except Exception as e:
            row["error"] = f"{type(e).__name__}:{e}"
            print(f"  ERROR {row['error']}", flush=True)

        (adir / "metrics.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )
        record_arm_outcome(state, row)
        state["elapsed_sec"] = time.time() - t0
        state["n_arms"] = len(arms)
        # Intermediate stop_reason from membership (never leave seeded_partial sticky)
        state["stop_reason"] = finalize_stop_reason(
            arm_ids,
            state["done"],
            hours_exhausted=False,
            accept_errors=accept_errors,
            failed=state.get("failed") or [],
            prior_stop=None,
        )
        save_progress(prog_path, state)
        # Refresh partial reports after each arm
        write_reports(
            out=out,
            arms=arms,
            state=state,
            screen=(int(args.screen_first), int(args.screen_last)),
            confirm=(int(args.confirm_first), int(args.confirm_last)),
        )

    state["stop_reason"] = finalize_stop_reason(
        arm_ids,
        state["done"],
        hours_exhausted=hours_exhausted
        or state.get("stop_reason") == "hours_exhausted",
        accept_errors=accept_errors,
        failed=state.get("failed") or [],
        prior_stop=None,
    )
    state["finished"] = datetime.now(timezone.utc).isoformat()
    state["elapsed_sec"] = time.time() - t0
    save_progress(prog_path, state)

    summary = write_reports(
        out=out,
        arms=arms,
        state=state,
        screen=(int(args.screen_first), int(args.screen_last)),
        confirm=(int(args.confirm_first), int(args.confirm_last)),
    )
    print(
        f"Wrote {out / 'SUMMARY.md'} status={summary['decision']['status']} "
        f"stop={state['stop_reason']} research_pass={summary['research_pass']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
