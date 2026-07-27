#!/usr/bin/env python
"""Overnight universe generalization + EU geo transfer for 3 frozen winners.

Protocol (research only)
------------------------
1. Freeze strategies: turbo_strict (L50/L80), turbo_highvol_minalloc (L50).
2. US: Monte Carlo random books from longhist2010_pass + PREFIX/FULL controls.
3. EU: transfer same knobs to ES/DE/FR/UK with local index + random local books.
4. Aggregate mean/median/std/pass_rate — never cherry-pick best seed.
5. Paper freeze turbo_highvol_minalloc is NOT auto-changed.

Usage::

    $env:PYTHONPATH = (Get-Location).Path
    python scripts/run_universe_generalization_overnight.py --hours 14 --workers 2

    # Smoke (tiny)
    python scripts/run_universe_generalization_overnight.py --smoke --hours 2

Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402
from trad_research.strategies import get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.universe_sampling import (  # noqa: E402
    MARKET_SPECS,
    aggregate_numeric,
    draw_seed,
    geo_verdict,
    market_specs,
    materialize_draw,
    pool_coverage,
    prefix_tickers,
    read_tickers,
    sample_without_replacement,
    us_verdict,
    write_decade_pool,
    write_tickers,
)
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402

logger = logging.getLogger("univ_gen")

PAPER_FREEZE = "turbo_highvol_minalloc"
DISCLAIMER = "Research only. Not financial advice. Past backtests ≠ future results."
COMMISSION = 0.001
SLIPPAGE = 0.0005
GATE_CAGR = 0.10
GATE_MDD = -0.65
GATE_TRADES = 80
GATE_TRADES_SMALL = 40
BASE_SEED_DEFAULT = 20260726


@dataclass
class Job:
    arm_id: str
    market: str
    series: str
    strategy: str
    seed: int
    draw_size: int
    universe_limit: int
    ticker_file: Path
    data_root: Path
    preferred_index: Tuple[str, ...]
    screen_first: int
    screen_last: int
    confirm_first: int
    confirm_last: int
    run_screen: bool
    gate_trades: int
    min_train_rows: int
    metrics_only: bool


def _eq_norm(s: pd.Series) -> pd.Series:
    out = s.dropna().astype(float)
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    return out[~out.index.duplicated(keep="last")].dropna().sort_index()


def _stitch(a: pd.Series, b: pd.Series) -> pd.Series:
    segs = []
    prev = None
    for seg in (a, b):
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


def _metrics(eq: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    eq = _eq_norm(eq)
    if eq.empty:
        return {"error": "empty", "cagr": 0.0, "max_drawdown": -1.0, "n_trades": 0}
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


def _gates(m: Dict[str, Any], gate_trades: int = GATE_TRADES) -> Dict[str, Any]:
    cagr = float(m.get("cagr") if m.get("cagr") is not None else 0.0)
    mdd = float(m.get("max_drawdown") if m.get("max_drawdown") is not None else -1.0)
    n = int(m.get("n_trades") if m.get("n_trades") is not None else 0)
    ok_c = cagr > GATE_CAGR
    ok_m = mdd >= GATE_MDD
    ok_t = n >= int(gate_trades)
    return {
        "cagr_ok": ok_c,
        "mdd_ok": ok_m,
        "trades_ok": ok_t,
        "pass": bool(ok_c and ok_m and ok_t),
        "gate_trades": int(gate_trades),
    }


def _bench_excess(
    eq: pd.Series,
    data_root: Path,
    preferred: Sequence[str],
) -> Optional[float]:
    try:
        b = load_benchmark_equity(
            data_root, eq.index.min(), eq.index.max(), preferred=list(preferred)
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


def _honest_score(m: Dict[str, Any], xs: Optional[float]) -> float:
    cagr = float(m.get("cagr") or 0.0)
    sortino = float(m.get("sortino") or 0.0)
    mdd = float(m.get("max_drawdown") or -1.0)
    score = 2.0 * cagr + 1.0 * sortino
    if xs is not None:
        score += 0.5 * max(0.0, float(xs))
    if mdd < -0.50:
        score -= 2.0 * ((-0.50) - mdd)
    return float(score)


def run_window(
    strategy: str,
    *,
    first: int,
    last: int,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    min_train_rows: int,
    preferred_index: Sequence[str],
) -> Dict[str, Any]:
    strat = get_strategy(strategy)
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
            preferred_index=list(preferred_index),
            base_bt=BacktestConfig(commission=COMMISSION, slippage=SLIPPAGE),
        )
    finally:
        if orig is not None:
            strat.backtest_overrides = orig  # type: ignore[method-assign]
    return res


def run_job(job: Job, arms_dir: Path) -> Dict[str, Any]:
    adir = arms_dir / job.arm_id.replace("/", "_")
    adir.mkdir(parents=True, exist_ok=True)
    row: Dict[str, Any] = {
        "arm_id": job.arm_id,
        "market": job.market,
        "series": job.series,
        "strategy": job.strategy,
        "seed": job.seed,
        "draw_size": job.draw_size,
        "universe_limit": job.universe_limit,
        "ticker_file": str(job.ticker_file),
        "data_root": str(job.data_root),
        "run_screen": job.run_screen,
        "gate_trades": job.gate_trades,
        "preferred_index": list(job.preferred_index),
    }
    t0 = time.time()
    try:
        eq_s: Optional[pd.Series] = None
        tr_s = pd.DataFrame()
        if job.run_screen:
            rs = run_window(
                job.strategy,
                first=job.screen_first,
                last=job.screen_last,
                data_root=job.data_root,
                ticker_file=job.ticker_file,
                universe_limit=job.universe_limit,
                min_train_rows=job.min_train_rows,
                preferred_index=job.preferred_index,
            )
            eq_s = rs.get("equity") if isinstance(rs.get("equity"), pd.Series) else None
            tr_s = rs.get("trades") if isinstance(rs.get("trades"), pd.DataFrame) else pd.DataFrame()
            m_s = _metrics(eq_s, tr_s) if eq_s is not None else {"error": "empty"}
            row["screen"] = {**m_s, "gates": _gates(m_s, job.gate_trades)}
            if eq_s is not None and not job.metrics_only:
                _eq_norm(eq_s).to_csv(adir / "equity_screen.csv", header=["equity"])
            if not tr_s.empty and not job.metrics_only:
                tr_s.to_csv(adir / "trades_screen.csv", index=False)
        else:
            row["screen"] = {"skipped": True}

        rc = run_window(
            job.strategy,
            first=job.confirm_first,
            last=job.confirm_last,
            data_root=job.data_root,
            ticker_file=job.ticker_file,
            universe_limit=job.universe_limit,
            min_train_rows=job.min_train_rows,
            preferred_index=job.preferred_index,
        )
        eq_c = rc.get("equity") if isinstance(rc.get("equity"), pd.Series) else None
        tr_c = rc.get("trades") if isinstance(rc.get("trades"), pd.DataFrame) else pd.DataFrame()
        m_c = _metrics(eq_c, tr_c) if eq_c is not None else {"error": "empty"}
        g_c = _gates(m_c, job.gate_trades)
        xs = (
            _bench_excess(eq_c, job.data_root, job.preferred_index)
            if eq_c is not None
            else None
        )
        row["confirm"] = {**m_c, "gates": g_c, "excess_index_total": xs}
        row["honest_score"] = _honest_score(m_c, xs)
        if eq_c is not None and not job.metrics_only:
            _eq_norm(eq_c).to_csv(adir / "equity_confirm.csv", header=["equity"])
        if not tr_c.empty and not job.metrics_only:
            tr_c.to_csv(adir / "trades_confirm.csv", index=False)

        if job.run_screen and eq_s is not None and eq_c is not None:
            eq_f = _stitch(eq_s, eq_c)
            m_f = _metrics(eq_f, None)
            row["full"] = {**m_f, "gates": _gates(m_f, job.gate_trades)}
            if not job.metrics_only:
                eq_f.to_csv(adir / "equity_full.csv", header=["equity"])
        else:
            # confirm-only markets: full = confirm for reporting transparency
            row["full"] = {
                "skipped": not job.run_screen,
                "note": "confirm_only" if not job.run_screen else "missing_segment",
                "gates": {"pass": False},
            }
            if not job.run_screen and eq_c is not None:
                row["full"] = {
                    **m_c,
                    "gates": g_c,
                    "note": "confirm_only_proxy_not_decade_full",
                    "research_pass_eligible": False,
                }

        cpass = bool((row.get("confirm") or {}).get("gates", {}).get("pass"))
        fpass = bool((row.get("full") or {}).get("gates", {}).get("pass"))
        # Research PASS only when screen+full path is honest decade stitch
        row["confirm_pass"] = cpass
        row["full_pass"] = fpass and job.run_screen
        row["research_pass"] = bool(cpass and fpass and job.run_screen)
    except Exception as e:
        row["error"] = f"{type(e).__name__}:{e}"
        row["traceback"] = traceback.format_exc(limit=8)
        logger.error("Job %s failed: %s", job.arm_id, row["error"])

    row["elapsed_sec"] = round(time.time() - t0, 2)
    (adir / "metrics.json").write_text(
        json.dumps(row, indent=2, default=str), encoding="utf-8"
    )
    return row


def _gate_trades_for(m: int) -> int:
    return GATE_TRADES_SMALL if int(m) < 40 else GATE_TRADES


def build_us_jobs(
    out: Path,
    *,
    base_seed: int,
    k50: int,
    k60: int,
    k80: int,
    strategies_l50: Sequence[str],
    include_l80: bool,
    metrics_only: bool,
    min_train_rows: int,
    pool_b: Optional[List[str]] = None,
    k_b50: int = 0,
    specs: Optional[Dict[str, Dict[str, Any]]] = None,
    repo_root: Optional[Path] = None,
) -> List[Job]:
    specs = specs or MARKET_SPECS
    repo = Path(repo_root) if repo_root is not None else ROOT
    spec = specs["US"]
    data_root = Path(spec["data_root"])
    pool = read_tickers(Path(spec["universe_file"]))
    if not pool:
        # fallback
        pool = read_tickers(repo / "universe_longhist100.txt")
    draws_dir = out / "draws" / "US"
    draws_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[Job] = []
    pref = tuple(spec["preferred_index"])

    def add_random(series: str, m: int, k: int, strategies: Sequence[str], limit: int) -> None:
        offset = {"R50": 0, "R60": 500, "R80": 1000, "B50": 3000}.get(series, 9000)
        src = pool_b if series == "B50" and pool_b else pool
        for i in range(int(k)):
            seed = draw_seed(base_seed, "US", offset + i)
            tfile = draws_dir / f"{series}_m{m}_s{seed}.txt"
            if not tfile.is_file():
                materialize_draw(src, series=series, m=m, seed=seed, out_path=tfile)
            for strat in strategies:
                arm = f"US__{strat}__{series}_s{seed}"
                jobs.append(
                    Job(
                        arm_id=arm,
                        market="US",
                        series=series,
                        strategy=strat,
                        seed=seed,
                        draw_size=m,
                        universe_limit=limit,
                        ticker_file=tfile,
                        data_root=data_root,
                        preferred_index=pref,
                        screen_first=2010,
                        screen_last=2017,
                        confirm_first=2018,
                        confirm_last=2025,
                        run_screen=True,
                        gate_trades=_gate_trades_for(m),
                        min_train_rows=min_train_rows,
                        metrics_only=metrics_only,
                    )
                )

    add_random("R50", 50, k50, strategies_l50, 50)
    if k60 > 0:
        add_random("R60", 60, k60, strategies_l50, 60)
    if include_l80 and k80 > 0:
        add_random("R80", 80, k80, ["turbo_strict"], 80)

    # FULL100 controls
    full_path = draws_dir / "FULL100.txt"
    write_tickers(full_path, pool)
    full_specs: List[Tuple[str, int]] = []
    for strat in strategies_l50:
        full_specs.append((strat, 50))
    if include_l80:
        full_specs.append(("turbo_strict", 80))
    # unique
    seen_full = set()
    for strat, lim in full_specs:
        key = (strat, lim)
        if key in seen_full:
            continue
        if lim == 80 and strat != "turbo_strict":
            continue
        if lim == 80 and not include_l80:
            continue
        seen_full.add(key)
        arm = f"US__{strat}__FULL100_lim{lim}"
        jobs.append(
            Job(
                arm_id=arm,
                market="US",
                series="FULL100",
                strategy=strat,
                seed=base_seed,
                draw_size=len(pool),
                universe_limit=lim,
                ticker_file=full_path,
                data_root=data_root,
                preferred_index=pref,
                screen_first=2010,
                screen_last=2017,
                confirm_first=2018,
                confirm_last=2025,
                run_screen=True,
                gate_trades=GATE_TRADES,
                min_train_rows=min_train_rows,
                metrics_only=metrics_only,
            )
        )

    # PREFIX (Kaggle repro)
    prefix_specs: List[Tuple[str, int, int]] = []
    for strat in strategies_l50:
        prefix_specs.append((strat, 50, 50))
    if include_l80:
        prefix_specs.append(("turbo_strict", 80, 80))
    seen_pref = set()
    for strat, m, lim in prefix_specs:
        key = (strat, m, lim)
        if key in seen_pref:
            continue
        seen_pref.add(key)
        tfile = draws_dir / f"PREFIX_m{m}.txt"
        write_tickers(tfile, prefix_tickers(pool, m))
        jobs.append(
            Job(
                arm_id=f"US__{strat}__PREFIX_L{m}",
                market="US",
                series="PREFIX",
                strategy=strat,
                seed=base_seed,
                draw_size=m,
                universe_limit=lim,
                ticker_file=tfile,
                data_root=data_root,
                preferred_index=pref,
                screen_first=2010,
                screen_last=2017,
                confirm_first=2018,
                confirm_last=2025,
                run_screen=True,
                gate_trades=GATE_TRADES,
                min_train_rows=min_train_rows,
                metrics_only=metrics_only,
            )
        )

    if pool_b and k_b50 > 0 and len(pool_b) >= 50:
        # temporary swap for B50 materialize
        for i in range(k_b50):
            seed = draw_seed(base_seed, "US", 3000 + i)
            tfile = draws_dir / f"B50_m50_s{seed}.txt"
            if not tfile.is_file():
                materialize_draw(pool_b, series="B50", m=50, seed=seed, out_path=tfile)
            for strat in strategies_l50:
                jobs.append(
                    Job(
                        arm_id=f"US__{strat}__B50_s{seed}",
                        market="US",
                        series="B50",
                        strategy=strat,
                        seed=seed,
                        draw_size=50,
                        universe_limit=50,
                        ticker_file=tfile,
                        data_root=data_root,
                        preferred_index=pref,
                        screen_first=2010,
                        screen_last=2017,
                        confirm_first=2018,
                        confirm_last=2025,
                        run_screen=True,
                        gate_trades=GATE_TRADES,
                        min_train_rows=min_train_rows,
                        metrics_only=metrics_only,
                    )
                )
    return jobs


def build_eu_jobs(
    out: Path,
    *,
    base_seed: int,
    markets: Sequence[str],
    strategies: Sequence[str],
    k_es: int,
    k_fr: int,
    k_de: int,
    k_uk: int,
    metrics_only: bool,
    min_train_rows: int,
    specs: Optional[Dict[str, Dict[str, Any]]] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[List[Job], Dict[str, Any]]:
    specs = specs or MARKET_SPECS
    repo = Path(repo_root) if repo_root is not None else ROOT
    eu_dir = out / "eu_pools"
    eu_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {}
    jobs: List[Job] = []
    k_map = {"ES": k_es, "FR": k_fr, "DE": k_de, "UK": k_uk}

    for mid in markets:
        mid = mid.upper()
        if mid == "US":
            continue
        if mid not in specs:
            continue
        spec = specs[mid]
        data_root = Path(spec["data_root"])
        ufile = Path(spec["universe_file"])
        pref = tuple(spec["preferred_index"])
        cov = pool_coverage(mid, repo_root=repo)
        meta[mid] = cov

        decade_path = eu_dir / f"{mid}_decade2010_pass.txt"
        decade = write_decade_pool(mid, decade_path, max_start_year=2010, repo_root=repo)
        modern_path = eu_dir / f"{mid}_modern_all.txt"
        all_tickers = [t for t in read_tickers(ufile) if (data_root / f"{t}_history.csv").is_file()]
        write_tickers(modern_path, all_tickers)

        # UK: confirm-only; DE decade if n>=20 else confirm-only
        if mid == "UK":
            pool = all_tickers
            run_screen = False
            m_draw = min(30, len(pool))
            k = int(k_map.get(mid, 10))
            series_base = "R30"
            screen = (2018, 2017)  # unused
        elif mid == "DE" and len(decade) < 20:
            pool = all_tickers
            run_screen = False
            m_draw = min(25, len(pool))
            k = int(k_map.get(mid, 10))
            series_base = "R25"
            screen = (2010, 2017)
        else:
            pool = decade if decade else all_tickers
            run_screen = True
            if mid == "ES":
                m_draw = min(40, len(pool))
                series_base = "R40"
            elif mid == "FR":
                m_draw = min(30, len(pool))
                series_base = "R30"
            else:
                m_draw = min(20, len(pool))
                series_base = "R20"
            k = int(k_map.get(mid, 10))
            screen = (2010, 2017)

        if m_draw < 10 or len(pool) < m_draw:
            logger.warning("%s pool too small n=%d m=%d — skip random", mid, len(pool), m_draw)
            meta[mid]["skipped_random"] = True
            continue

        draws_dir = out / "draws" / mid
        draws_dir.mkdir(parents=True, exist_ok=True)
        gt = _gate_trades_for(m_draw)

        for i in range(k):
            seed = draw_seed(base_seed, mid, i)
            tfile = draws_dir / f"{series_base}_m{m_draw}_s{seed}.txt"
            if not tfile.is_file():
                materialize_draw(pool, series=series_base, m=m_draw, seed=seed, out_path=tfile)
            for strat in strategies:
                jobs.append(
                    Job(
                        arm_id=f"{mid}__{strat}__{series_base}_s{seed}",
                        market=mid,
                        series=series_base,
                        strategy=strat,
                        seed=seed,
                        draw_size=m_draw,
                        universe_limit=m_draw,
                        ticker_file=tfile,
                        data_root=data_root,
                        preferred_index=pref,
                        screen_first=screen[0],
                        screen_last=screen[1],
                        confirm_first=2018,
                        confirm_last=2025,
                        run_screen=run_screen,
                        gate_trades=gt,
                        min_train_rows=min_train_rows,
                        metrics_only=metrics_only,
                    )
                )

        # FULL control
        full_p = draws_dir / "FULL.txt"
        write_tickers(full_p, pool)
        lim = min(50, len(pool)) if mid != "DE" else min(20, len(pool))
        for strat in strategies:
            jobs.append(
                Job(
                    arm_id=f"{mid}__{strat}__FULL_lim{lim}",
                    market=mid,
                    series="FULL",
                    strategy=strat,
                    seed=base_seed,
                    draw_size=len(pool),
                    universe_limit=lim,
                    ticker_file=full_p,
                    data_root=data_root,
                    preferred_index=pref,
                    screen_first=2010 if run_screen else 2018,
                    screen_last=2017 if run_screen else 2017,
                    confirm_first=2018,
                    confirm_last=2025,
                    run_screen=run_screen,
                    gate_trades=_gate_trades_for(lim),
                    min_train_rows=min_train_rows,
                    metrics_only=metrics_only,
                )
            )

        # ES extra R50 for S1 only
        if mid == "ES" and len(pool) >= 50:
            for i in range(min(15, k)):
                seed = draw_seed(base_seed, mid, 500 + i)
                tfile = draws_dir / f"R50_m50_s{seed}.txt"
                if not tfile.is_file():
                    materialize_draw(pool, series="R50", m=50, seed=seed, out_path=tfile)
                jobs.append(
                    Job(
                        arm_id=f"ES__turbo_strict__R50_s{seed}",
                        market="ES",
                        series="R50",
                        strategy="turbo_strict",
                        seed=seed,
                        draw_size=50,
                        universe_limit=50,
                        ticker_file=tfile,
                        data_root=data_root,
                        preferred_index=pref,
                        screen_first=2010,
                        screen_last=2017,
                        confirm_first=2018,
                        confirm_last=2025,
                        run_screen=True,
                        gate_trades=GATE_TRADES,
                        min_train_rows=min_train_rows,
                        metrics_only=metrics_only,
                    )
                )

        meta[mid]["pool_used_n"] = len(pool)
        meta[mid]["run_screen"] = run_screen
        meta[mid]["m_draw"] = m_draw
        meta[mid]["k"] = k

    return jobs, meta


def prioritize_jobs(jobs: List[Job]) -> List[Job]:
    """PREFIX US strict first, then other PREFIX, US R50, ES, rest."""

    def prio(j: Job) -> Tuple[int, int, str]:
        if j.market == "US" and j.series == "PREFIX" and j.strategy == "turbo_strict" and j.draw_size == 50:
            return (0, 0, j.arm_id)
        if j.market == "US" and j.series == "PREFIX":
            return (0, 1, j.arm_id)
        if j.market == "US" and j.series == "R50":
            return (1, 0, j.arm_id)
        if j.market == "ES":
            return (2, 0, j.arm_id)
        if j.market == "US" and j.series in ("R60", "R80", "FULL100"):
            return (3, 0, j.arm_id)
        if j.market == "FR":
            return (4, 0, j.arm_id)
        if j.market == "DE":
            return (5, 0, j.arm_id)
        if j.market == "UK":
            return (6, 0, j.arm_id)
        return (9, 0, j.arm_id)

    return sorted(jobs, key=prio)


def flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    c = row.get("confirm") or {}
    f = row.get("full") or {}
    s = row.get("screen") or {}
    return {
        "arm_id": row.get("arm_id"),
        "market": row.get("market"),
        "series": row.get("series"),
        "strategy": row.get("strategy"),
        "seed": row.get("seed"),
        "draw_size": row.get("draw_size"),
        "universe_limit": row.get("universe_limit"),
        "confirm_cagr": c.get("cagr"),
        "confirm_mdd": c.get("max_drawdown"),
        "confirm_sharpe": c.get("sharpe"),
        "confirm_n_trades": c.get("n_trades"),
        "confirm_pass": row.get("confirm_pass"),
        "full_cagr": f.get("cagr"),
        "full_mdd": f.get("max_drawdown"),
        "full_pass": row.get("full_pass"),
        "research_pass": row.get("research_pass"),
        "excess_index": c.get("excess_index_total"),
        "honest_score": row.get("honest_score"),
        "screen_cagr": s.get("cagr"),
        "error": row.get("error"),
        "elapsed_sec": row.get("elapsed_sec"),
    }


def aggregate_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("error"):
            continue
        key = f"{r.get('market')}|{r.get('strategy')}|{r.get('series')}"
        by.setdefault(key, []).append(r)

    agg: Dict[str, Any] = {}
    for key, group in by.items():
        market, strategy, series = key.split("|", 2)
        cagrs = [(g.get("confirm") or {}).get("cagr") for g in group]
        mdds = [(g.get("confirm") or {}).get("max_drawdown") for g in group]
        sharpes = [(g.get("confirm") or {}).get("sharpe") for g in group]
        n = len(group)
        n_cpass = sum(1 for g in group if g.get("confirm_pass"))
        n_rpass = sum(1 for g in group if g.get("research_pass"))
        agg[key] = {
            "market": market,
            "strategy": strategy,
            "series": series,
            "n": n,
            "confirm_cagr": aggregate_numeric(cagrs),
            "confirm_mdd": aggregate_numeric(mdds),
            "confirm_sharpe": aggregate_numeric(sharpes),
            "confirm_pass_rate": n_cpass / n if n else 0.0,
            "research_pass_rate": n_rpass / n if n else 0.0,
            "median_confirm_pass": bool(
                (aggregate_numeric(cagrs).get("median") or 0) > GATE_CAGR
                and (aggregate_numeric(mdds).get("median") or -1) >= GATE_MDD
            ),
        }
    return agg


def write_reports(
    out: Path,
    rows: Sequence[Dict[str, Any]],
    agg: Dict[str, Any],
    *,
    complete: bool,
    eu_meta: Dict[str, Any],
) -> None:
    flat = [flatten_row(r) for r in rows]
    pd.DataFrame(flat).to_csv(out / "all_runs.csv", index=False)
    (out / "aggregate_by_market.json").write_text(
        json.dumps(agg, indent=2, default=str), encoding="utf-8"
    )

    # US S1 R50 verdict
    us_key = "US|turbo_strict|R50"
    us_a = agg.get(us_key) or {}
    prefix_rows = [
        r
        for r in rows
        if r.get("market") == "US"
        and r.get("strategy") == "turbo_strict"
        and r.get("series") == "PREFIX"
        and "L50" in str(r.get("arm_id"))
    ]
    prefix_pass = any(r.get("research_pass") or r.get("confirm_pass") for r in prefix_rows)
    pr = float(us_a.get("research_pass_rate") or us_a.get("confirm_pass_rate") or 0.0)
    med_c = (us_a.get("confirm_cagr") or {}).get("median")
    med_m = (us_a.get("confirm_mdd") or {}).get("median")
    # Prefer research_pass_rate for verdict; if series incomplete use confirm
    v_us = us_verdict(
        pass_rate=float(us_a.get("research_pass_rate") or 0.0) if us_a else 0.0,
        median_cagr=med_c,
        median_mdd=med_m,
        prefix_pass=prefix_pass,
    )
    if us_a and float(us_a.get("research_pass_rate") or 0) == 0 and float(us_a.get("confirm_pass_rate") or 0) > 0:
        # still use research rate for GENERALIZES; note in decision
        pass

    # Geo: median_confirm_pass for ES/FR/DE on primary series
    med_pass: Dict[str, bool] = {}
    for mid in ("ES", "FR", "DE"):
        # pick R40/R30/R20 keys for turbo_strict
        hits = [
            a
            for k, a in agg.items()
            if a.get("market") == mid
            and a.get("strategy") == "turbo_strict"
            and str(a.get("series", "")).startswith("R")
        ]
        if hits:
            # prefer largest n
            hits = sorted(hits, key=lambda x: -int(x.get("n") or 0))
            med_pass[mid] = bool(hits[0].get("median_confirm_pass"))
    uk_hits = [
        a
        for k, a in agg.items()
        if a.get("market") == "UK" and a.get("strategy") == "turbo_strict"
    ]
    uk_ok = None
    if uk_hits:
        uk_ok = bool((uk_hits[0].get("confirm_cagr") or {}).get("median", -1) > 0)

    v_geo = geo_verdict(med_pass, uk_ok=uk_ok)

    # DISTRIBUTION.md
    lines = [
        "# Universe generalization — DISTRIBUTION",
        "",
        DISCLAIMER,
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Complete: {complete}",
        f"Rows: {len(rows)}",
        "",
        "## Aggregates (confirm window)",
        "",
        "| key | n | mean CAGR | median CAGR | mean MDD | pass_rate confirm | research_pass_rate |",
        "|-----|--:|----------:|------------:|---------:|------------------:|-------------------:|",
    ]
    for k, a in sorted(agg.items()):
        cc = a.get("confirm_cagr") or {}
        mm = a.get("confirm_mdd") or {}
        lines.append(
            f"| `{k}` | {a.get('n')} | {_pct(cc.get('mean'))} | {_pct(cc.get('median'))} | "
            f"{_pct(mm.get('mean'))} | {_pct(a.get('confirm_pass_rate'))} | {_pct(a.get('research_pass_rate'))} |"
        )
    lines.extend(["", f"**US verdict (S1·R50):** `{v_us}`", f"**GEO verdict:** `{v_geo}`", ""])
    (out / "DISTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")

    # GEO_TRANSFER.md
    geo_lines = [
        "# Geo transfer",
        "",
        DISCLAIMER,
        "",
        f"Verdict: **{v_geo}**",
        "",
        "## Median confirm pass by market (turbo_strict random series)",
        "",
    ]
    for mid, ok in med_pass.items():
        geo_lines.append(f"- **{mid}**: median_confirm_pass={ok}")
    geo_lines.append(f"- **UK** (confirm-only, median CAGR>0): {uk_ok}")
    geo_lines.append("")
    geo_lines.append("## EU pool meta")
    geo_lines.append("```json")
    geo_lines.append(json.dumps(eu_meta, indent=2, default=str)[:8000])
    geo_lines.append("```")
    (out / "GEO_TRANSFER.md").write_text("\n".join(geo_lines), encoding="utf-8")

    # DECISION.md
    dlines = [
        "# Universe generalization overnight — DECISION",
        "",
        f"- Complete: **{complete}**",
        f"- US verdict (turbo_strict · R50): **{v_us}**",
        f"- GEO verdict: **{v_geo}**",
        f"- Paper freeze: **`{PAPER_FREEZE}`** unchanged (no auto-ADVANCE)",
        f"- PREFIX turbo_strict L50 confirm/research pass: **{prefix_pass}**",
        f"- US R50 research_pass_rate: **{_pct(us_a.get('research_pass_rate') if us_a else None)}** "
        f"(n={us_a.get('n') if us_a else 0})",
        f"- US R50 median confirm CAGR/MDD: **{_pct(med_c)}** / **{_pct(med_m)}**",
        "",
        DISCLAIMER,
    ]
    (out / "DECISION.md").write_text("\n".join(dlines), encoding="utf-8")

    summary = {
        "complete": complete,
        "us_verdict": v_us,
        "geo_verdict": v_geo,
        "paper_freeze": PAPER_FREEZE,
        "n_rows": len(rows),
        "prefix_pass": prefix_pass,
        "us_r50": us_a,
        "disclaimer": DISCLAIMER,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.1f}%"
    except (TypeError, ValueError):
        return "—"


def save_progress(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Universe generalization overnight")
    ap.add_argument("--hours", type=float, default=14.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED_DEFAULT)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "redesign" / "universe_gen_overnight")
    ap.add_argument("--us-k50", type=int, default=40)
    ap.add_argument("--us-k60", type=int, default=15)
    ap.add_argument("--us-k80", type=int, default=25)
    ap.add_argument("--eu-k-es", type=int, default=25)
    ap.add_argument("--eu-k-fr", type=int, default=20)
    ap.add_argument("--eu-k-de", type=int, default=15)
    ap.add_argument("--eu-k-uk", type=int, default=15)
    ap.add_argument("--markets", type=str, default="US,ES,FR,DE,UK")
    ap.add_argument(
        "--strategies",
        type=str,
        default="turbo_strict,turbo_highvol_minalloc",
    )
    ap.add_argument("--include-strict-l80-us", action="store_true", default=True)
    ap.add_argument("--no-strict-l80-us", action="store_true")
    ap.add_argument("--us-only", action="store_true")
    ap.add_argument("--eu-only", action="store_true")
    ap.add_argument("--metrics-only", action="store_true", default=True)
    ap.add_argument("--save-equity", action="store_true", help="Write equity CSVs (more disk)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--min-train-rows", type=int, default=1500)
    ap.add_argument("--min-train-rows-eu", type=int, default=800)
    ap.add_argument("--max-jobs", type=int, default=0, help="0=all")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Dataset/repo root (Kaggle: path with data/, data_es/, universes)",
    )
    ap.add_argument(
        "--shard",
        type=str,
        default="",
        help="Optional shard index/count e.g. 0/2 or 1/2 for multi-GPU",
    )
    ap.add_argument(
        "--kaggle-dense",
        action="store_true",
        help="Large K for ~8h on GPU T4x2 (overrides us/eu K defaults if still default-ish)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(args.repo_root) if args.repo_root else ROOT
    if not repo_root.is_absolute():
        repo_root = (ROOT / repo_root).resolve()
    specs = market_specs(repo_root)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    arms_dir = out / "runs"
    arms_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "run.log", encoding="utf-8"),
        ],
    )

    if args.kaggle_dense and not args.smoke:
        # Target ~500–600 jobs for ~8h wall with 2 GPU workers (~90–120s/job GPU)
        args.us_k50 = max(int(args.us_k50), 80)
        args.us_k60 = max(int(args.us_k60), 30)
        args.us_k80 = max(int(args.us_k80), 50)
        args.eu_k_es = max(int(args.eu_k_es), 40)
        args.eu_k_fr = max(int(args.eu_k_fr), 35)
        args.eu_k_de = max(int(args.eu_k_de), 25)
        args.eu_k_uk = max(int(args.eu_k_uk), 25)
        logger.info(
            "KAGGLE-DENSE K: us50=%s us60=%s us80=%s es=%s fr=%s de=%s uk=%s",
            args.us_k50,
            args.us_k60,
            args.us_k80,
            args.eu_k_es,
            args.eu_k_fr,
            args.eu_k_de,
            args.eu_k_uk,
        )

    if args.smoke:
        args.us_k50 = 2
        args.us_k60 = 0
        args.us_k80 = 1
        args.eu_k_es = 1
        args.eu_k_fr = 1
        args.eu_k_de = 1
        args.eu_k_uk = 1
        args.hours = min(float(args.hours), 3.0)
        logger.info("SMOKE mode: tiny K, hours=%s", args.hours)

    metrics_only = bool(args.metrics_only) and not bool(args.save_equity)
    include_l80 = bool(args.include_strict_l80_us) and not bool(args.no_strict_l80_us)
    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    markets = [m.strip().upper() for m in str(args.markets).split(",") if m.strip()]

    # Coverage
    cov_rows = []
    for m in markets:
        if m in specs:
            cov_rows.append(pool_coverage(m, repo_root=repo_root))
    (out / "pool_coverage.json").write_text(
        json.dumps({"markets": cov_rows, "generated": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    logger.info("Coverage written for %d markets repo_root=%s", len(cov_rows), repo_root)

    jobs: List[Job] = []
    eu_meta: Dict[str, Any] = {}
    if not args.eu_only:
        jobs.extend(
            build_us_jobs(
                out,
                base_seed=int(args.base_seed),
                k50=int(args.us_k50),
                k60=int(args.us_k60),
                k80=int(args.us_k80),
                strategies_l50=strategies,
                include_l80=include_l80,
                metrics_only=metrics_only,
                min_train_rows=int(args.min_train_rows),
                specs=specs,
                repo_root=repo_root,
            )
        )
    if not args.us_only:
        eu_jobs, eu_meta = build_eu_jobs(
            out,
            base_seed=int(args.base_seed),
            markets=markets,
            strategies=strategies,
            k_es=int(args.eu_k_es),
            k_fr=int(args.eu_k_fr),
            k_de=int(args.eu_k_de),
            k_uk=int(args.eu_k_uk),
            metrics_only=metrics_only,
            min_train_rows=int(args.min_train_rows_eu),
            specs=specs,
            repo_root=repo_root,
        )
        jobs.extend(eu_jobs)
        (out / "eu_pools" / "meta.json").write_text(
            json.dumps(eu_meta, indent=2, default=str), encoding="utf-8"
        )

    jobs = prioritize_jobs(jobs)
    # de-dupe arm_id
    seen = set()
    uniq: List[Job] = []
    for j in jobs:
        if j.arm_id in seen:
            continue
        seen.add(j.arm_id)
        uniq.append(j)
    jobs = uniq

    # Optional multi-GPU shard: stable index split after prioritize
    shard_s = str(args.shard or "").strip()
    if shard_s and "/" in shard_s:
        try:
            si, sn = shard_s.split("/", 1)
            si_i, sn_i = int(si), int(sn)
            if sn_i > 1 and 0 <= si_i < sn_i:
                before = len(jobs)
                jobs = [j for idx, j in enumerate(jobs) if idx % sn_i == si_i]
                logger.info("Shard %s → %d / %d jobs", shard_s, len(jobs), before)
            else:
                logger.warning("Invalid shard %s — running all jobs", shard_s)
        except ValueError:
            logger.warning("Could not parse --shard %s", shard_s)

    if int(args.max_jobs) > 0:
        jobs = jobs[: int(args.max_jobs)]

    logger.info(
        "Planned jobs=%d workers=%d hours=%s shard=%s",
        len(jobs),
        args.workers,
        args.hours,
        shard_s or "none",
    )

    prog_path = out / "PROGRESS.json"
    state: Dict[str, Any] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "hours": float(args.hours),
        "n_jobs": len(jobs),
        "done": [],
        "rows": [],
        "stop_reason": None,
        "paper_freeze": PAPER_FREEZE,
    }
    if prog_path.is_file():
        try:
            prev = json.loads(prog_path.read_text(encoding="utf-8"))
            state["done"] = list(prev.get("done") or [])
            state["rows"] = list(prev.get("rows") or [])
            logger.info("Resume done=%d", len(state["done"]))
        except Exception:
            pass

    done_set = set(state["done"])
    t_run0 = time.time()
    deadline = t_run0 + float(args.hours) * 3600.0
    pending = [j for j in jobs if j.arm_id not in done_set]
    logger.info("Pending=%d", len(pending))

    workers = max(1, int(args.workers))

    def _run_one(j: Job) -> Dict[str, Any]:
        logger.info("[job] %s …", j.arm_id)
        row = run_job(j, arms_dir)
        logger.info(
            "  done %s confirm_cagr=%s pass=%s err=%s t=%.1fs",
            j.arm_id,
            (row.get("confirm") or {}).get("cagr"),
            row.get("confirm_pass"),
            row.get("error"),
            row.get("elapsed_sec") or 0,
        )
        return row

    if workers == 1:
        for j in pending:
            if time.time() > deadline:
                state["stop_reason"] = "hours_exhausted"
                break
            row = _run_one(j)
            state["rows"].append(row)
            state["done"].append(j.arm_id)
            done_set.add(j.arm_id)
            state["elapsed_sec"] = round(time.time() - t_run0, 1)
            state["n_done"] = len(state["done"])
            save_progress(prog_path, state)
    else:
        # Thread pool with deadline checks between submissions batches
        batch_i = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for j in pending:
                if time.time() > deadline:
                    state["stop_reason"] = "hours_exhausted"
                    break
                futures[ex.submit(_run_one, j)] = j
                batch_i += 1
                # throttle submit if many — still submit all until deadline at submit time
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "arm_id": j.arm_id,
                        "market": j.market,
                        "series": j.series,
                        "strategy": j.strategy,
                        "error": f"{type(e).__name__}:{e}",
                    }
                state["rows"].append(row)
                state["done"].append(j.arm_id)
                state["n_done"] = len(state["done"])
                save_progress(prog_path, state)
                if time.time() > deadline and state.get("stop_reason") is None:
                    state["stop_reason"] = "hours_exhausted"
                    # cannot cancel running easily; mark and continue collecting

    if state.get("stop_reason") is None:
        state["stop_reason"] = "complete" if len(state["done"]) >= len(jobs) else "partial"
    state["finished"] = datetime.now(timezone.utc).isoformat()
    complete = state["stop_reason"] == "complete"
    save_progress(prog_path, state)

    rows = list(state["rows"])
    agg = aggregate_results(rows)
    write_reports(out, rows, agg, complete=complete, eu_meta=eu_meta)
    logger.info(
        "Finished stop=%s done=%d/%d → %s",
        state["stop_reason"],
        len(state["done"]),
        len(jobs),
        out,
    )
    print(f"DECISION: {(out / 'DECISION.md').read_text(encoding='utf-8')[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
