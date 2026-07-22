"""Multi-strategy paper cloud batch — 10 strategies, free data, study-ready outputs."""
from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from paper_live.cloud.free_data import build_cloud_feed
from paper_live.freeze import PaperFreeze, load_freeze
from paper_live.ledger import PaperLedger
from paper_live.replay_session import ReplaySession
from paper_live.reports.pipeline import generate_reports_for_run
from paper_live.signals.daily_pipeline import DailySignalPipeline

logger = logging.getLogger(__name__)

DEFAULT_ZOO = Path(__file__).resolve().parent / "strategy_zoo.json"


def load_zoo(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_ZOO
    return json.loads(p.read_text(encoding="utf-8"))


def _trade_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_ret": None,
            "exit_reasons": {},
        }
    rets = [float(t.get("ret") or 0.0) for t in trades]
    wins = [t for t in trades if float(t.get("pnl") or 0.0) > 0]
    losses = [t for t in trades if float(t.get("pnl") or 0.0) <= 0]
    gp = sum(float(t.get("pnl") or 0.0) for t in wins)
    gl = abs(sum(float(t.get("pnl") or 0.0) for t in losses))
    er: Dict[str, int] = {}
    for t in trades:
        r = str(t.get("exit_reason") or "unknown")
        er[r] = er.get(r, 0) + 1
    pf = (gp / gl) if gl > 1e-9 else (float("inf") if gp > 0 else 0.0)
    if pf == float("inf"):
        pf = 99.0
    return {
        "n": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": pf,
        "avg_ret": sum(rets) / len(rets),
        "exit_reasons": er,
    }


def _window_bh_return(feed: Any, ticker: str, start_d: date, end_d: date) -> Optional[float]:
    """Buy&hold total return using session closes on/inside window."""
    try:
        b0 = feed.bar(ticker, start_d)
        b1 = feed.bar(ticker, end_d)
        # walk to first/last available bar in window if missing
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
                for d in reversed(days):
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


def _eq_weight_bh(
    feed: Any, tickers: Sequence[str], start_d: date, end_d: date
) -> Optional[float]:
    rets = []
    for t in tickers:
        if t.upper() in ("SPY",):
            continue
        r = _window_bh_return(feed, t, start_d, end_d)
        if r is not None:
            rets.append(r)
    if not rets:
        return None
    return float(sum(rets) / len(rets))


def _write_trades_csv(path: Path, trades: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "ticker",
        "entry_day",
        "exit_day",
        "entry_px",
        "exit_px",
        "qty",
        "ret",
        "pnl",
        "bars_held",
        "exit_reason",
        "is_stop",
    ]
    lines = [",".join(cols)]
    for t in trades:
        row = []
        for c in cols:
            v = t.get(c, "")
            row.append(str(v))
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _override_freeze(base: PaperFreeze, strat: Dict[str, Any], capital0: float) -> PaperFreeze:
    """Shallow override of strategy knobs / risk for one zoo member."""
    # PaperFreeze is frozen dataclass — rebuild via dict
    s = base.strategy.to_dict()
    s["strategy_id"] = str(strat["id"])
    s["description"] = str(strat.get("label") or strat["id"])
    s["capital0"] = float(capital0)
    kn = dict(s.get("knobs") or {})
    kn["min_alloc_pct"] = float(strat.get("min_alloc_pct", kn.get("min_alloc_pct", 0.015)))
    kn["max_positions"] = int(strat.get("max_positions", kn.get("max_positions", 8)))
    kn["max_horizon"] = int(strat.get("max_horizon", kn.get("max_horizon", 20)))
    kn["hard_stop_pct"] = float(strat.get("hard_stop_pct", kn.get("hard_stop_pct", 0.07)))
    kn["k_atr"] = float(strat.get("k_atr", kn.get("k_atr", 3.0)))
    kn["require_regime"] = bool(strat.get("require_regime", kn.get("require_regime", True)))
    if "trail_stops" in strat:
        kn["trail_stops"] = bool(strat["trail_stops"])
    if "use_hard_stop" in strat:
        kn["use_hard_stop"] = bool(strat["use_hard_stop"])
    s["knobs"] = kn
    rp = dict(s.get("risk_paper") or {})
    # Cloud study defaults: real tape, avoid false sharpe hard-kills mid-sample
    rp["enable_sharpe_kill"] = bool(strat.get("enable_sharpe_kill", False))
    rp["min_returns_for_sharpe_kill"] = int(strat.get("min_returns_for_sharpe_kill", 60))
    rp["max_portfolio_dd"] = float(strat.get("max_portfolio_dd", rp.get("max_portfolio_dd", 0.22)))
    rp["kill_dd_from_start"] = float(strat.get("kill_dd_from_start", rp.get("kill_dd_from_start", 0.25)))
    if "max_entries_per_day" in strat:
        rp["max_daily_new_entries"] = int(strat["max_entries_per_day"])
    if "ticker_max_capital_pct" in strat:
        rp["ticker_max_capital_pct"] = float(strat["ticker_max_capital_pct"])
    elif float(strat.get("min_alloc_pct", 0) or 0) >= 0.5:
        # concentrated / index-hold styles need a higher single-name cap
        rp["ticker_max_capital_pct"] = max(
            float(rp.get("ticker_max_capital_pct", 0.12)),
            float(strat.get("min_alloc_pct", 0.5)),
        )
    s["risk_paper"] = rp

    from paper_live.freeze import (
        CostModel,
        ScheduleConfig,
        StrategyFreeze,
        UniverseConfig,
        compute_config_hash,
    )

    strategy = StrategyFreeze.from_dict(s)
    cost = base.cost
    schedule = base.schedule
    universe = base.universe
    bundle = {
        "strategy": strategy.to_dict(),
        "cost": cost.to_dict(),
        "schedule": schedule.to_dict(),
        "universe": universe.to_dict(),
    }
    return PaperFreeze(
        strategy=strategy,
        cost=cost,
        schedule=schedule,
        universe=universe,
        config_hash=compute_config_hash(bundle),
        source_dir=base.source_dir,
    )


@dataclass
class StrategyRunSummary:
    strategy_id: str
    label: str
    days_run: int
    n_entries: int
    n_exits: int
    n_signals: int
    final_equity: float
    total_return: float
    total_commission: float
    total_fees: float
    hard_kill: bool
    kill_trips: int
    run_id: str
    error: Optional[str] = None
    signal_mode: str = "trend_mom"
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_trade_ret: Optional[float] = None
    n_closed_trades: int = 0
    exit_reasons: Dict[str, int] = field(default_factory=dict)
    vs_spy: Optional[float] = None  # strategy return - spy return

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "label": self.label,
            "days_run": self.days_run,
            "n_entries": self.n_entries,
            "n_exits": self.n_exits,
            "n_signals": self.n_signals,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_commission": self.total_commission,
            "total_fees": self.total_fees,
            "hard_kill": self.hard_kill,
            "kill_trips": self.kill_trips,
            "run_id": self.run_id,
            "error": self.error,
            "signal_mode": self.signal_mode,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_ret": self.avg_trade_ret,
            "n_closed_trades": self.n_closed_trades,
            "exit_reasons": self.exit_reasons,
            "vs_spy": self.vs_spy,
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }


@dataclass
class CloudBatchResult:
    as_of: str
    n_strategies: int
    data_sources: Dict[str, str]
    strategies: List[StrategyRunSummary] = field(default_factory=list)
    out_dir: str = ""
    mode: str = "paper"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "n_strategies": self.n_strategies,
            "data_sources": self.data_sources,
            "strategies": [s.to_dict() for s in self.strategies],
            "out_dir": self.out_dir,
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }

    def ranking(self) -> List[StrategyRunSummary]:
        ok = [s for s in self.strategies if not s.error]
        return sorted(ok, key=lambda s: s.total_return, reverse=True)


def run_cloud_batch(
    *,
    out_root: Union[str, Path] = "reports/paper_cloud",
    zoo_path: Optional[Union[str, Path]] = None,
    force_synthetic: bool = False,
    lookback_days: Optional[int] = None,
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
    keep_ledgers: bool = False,
) -> CloudBatchResult:
    """Run all zoo strategies on free data; write study artifacts under out_root."""
    zoo = load_zoo(zoo_path)
    capital0 = float(zoo.get("capital0") or 100_000.0)
    tickers = list(zoo.get("universe") or ["AAPL", "MSFT", "QQQ", "SPY"])
    lb = int(lookback_days or zoo.get("lookback_calendar_days") or 400)
    strategies = list(zoo.get("strategies") or [])
    if not strategies:
        raise ValueError("strategy zoo is empty")

    as_of = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_root = Path(out_root)
    day_dir = out_root / "history" / as_of
    latest_dir = out_root / "latest"
    day_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = out_root / "data_cache"
    from paper_live.cloud.free_data import SEED_DIR

    feed, sources = build_cloud_feed(
        tickers,
        cache_dir=cache_dir,
        seed_dir=SEED_DIR,
        lookback_calendar_days=lb,
        force_synthetic=force_synthetic,
        require_real=not force_synthetic,
        min_real_tickers=5 if not force_synthetic else 0,
    )

    days = feed.days
    if not days:
        raise RuntimeError("No trading days in feed")

    # Default window: last ~180 sessions (~9m) ending at as_of / last available
    end_d = days[-1]
    if end_date:
        # clamp to available
        ed = __import__("pandas").Timestamp(end_date).date()
        avail = [d for d in days if d <= ed]
        end_d = avail[-1] if avail else days[-1]
    if start_date:
        sd = __import__("pandas").Timestamp(start_date).date()
        start_d = next((d for d in days if d >= sd), days[0])
    else:
        start_d = days[max(0, days.index(end_d) - 180)]

    # Sanity: window must not be in the future relative to market data
    logger.info(
        "Cloud window %s → %s | sources=%s | n_days_avail=%d",
        start_d,
        end_d,
        sources,
        len(days),
    )
    real_n = sum(
        1
        for s in sources.values()
        if not str(s).startswith("synthetic") and s != "missing"
    )
    if not force_synthetic and real_n < 5:
        raise RuntimeError(f"Refusing synthetic cloud study pack: sources={sources}")

    base_freeze = load_freeze()
    summaries: List[StrategyRunSummary] = []
    equity_series: Dict[str, List[Dict[str, Any]]] = {}

    spy_bh = _window_bh_return(feed, "SPY", start_d, end_d)
    if spy_bh is None and "QQQ" in feed.tickers:
        spy_bh = _window_bh_return(feed, "QQQ", start_d, end_d)
    eq_bh = _eq_weight_bh(feed, feed.tickers, start_d, end_d)
    logger.info("Benchmarks window %s→%s SPY_BH=%s EQ_BH=%s", start_d, end_d, spy_bh, eq_bh)

    for strat in strategies:
        sid = str(strat["id"])
        label = str(strat.get("label") or sid)
        signal_mode = str(strat.get("signal_mode") or "trend_mom")
        logger.info("Cloud paper strategy %s mode=%s …", sid, signal_mode)
        try:
            freeze = _override_freeze(base_freeze, strat, capital0)
            ledger_root = day_dir / "ledgers" / sid
            if ledger_root.exists():
                shutil.rmtree(ledger_root, ignore_errors=True)
            ledger = PaperLedger.create_run(
                ledger_root,
                freeze,
                meta={
                    "cloud": True,
                    "as_of": as_of,
                    "label": label,
                    "signal_mode": signal_mode,
                },
            )
            univ = [t for t in feed.tickers if t not in ("SPY",)]
            exclude_index = bool(strat.get("exclude_index", True))
            top_k = strat.get("top_k", None)
            pipe = DailySignalPipeline(
                feed,
                universe=univ,
                min_price=float(strat.get("min_price", 5.0)),
                max_atr_pct=float(strat.get("max_atr_pct", 0.22)),
                min_atr_norm=float(strat.get("min_atr_norm", 0.01)),
                require_regime=bool(strat.get("require_regime", True)),
                regime_symbol="QQQ"
                if "QQQ" in feed.tickers
                else ("SPY" if "SPY" in feed.tickers else feed.tickers[0]),
                signal_mode=signal_mode,
                top_k=int(top_k) if top_k is not None else None,
                exclude_index=exclude_index,
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
            rep_dir = day_dir / "strategies" / sid
            generate_reports_for_run(ledger, rep_dir, write_html=True)

            trades = list(result.closed_trades or [])
            _write_trades_csv(rep_dir / "closed_trades.csv", trades)
            ts = _trade_stats(trades)

            nav = ledger.list_nav()
            equity_series[sid] = [
                {"date": r["date"], "equity": float(r["equity"])} for r in nav
            ]

            tr = result.final_equity / capital0 - 1.0
            vs = (tr - spy_bh) if spy_bh is not None else None
            summaries.append(
                StrategyRunSummary(
                    strategy_id=sid,
                    label=label,
                    days_run=result.days_run,
                    n_entries=result.n_entries,
                    n_exits=result.n_exits,
                    n_signals=result.n_signals,
                    final_equity=result.final_equity,
                    total_return=tr,
                    total_commission=result.total_commission,
                    total_fees=result.total_fees,
                    hard_kill=result.hard_kill,
                    kill_trips=result.kill_trips,
                    run_id=result.run_id or ledger.run_id,
                    signal_mode=signal_mode,
                    win_rate=ts["win_rate"],
                    profit_factor=ts["profit_factor"],
                    avg_trade_ret=ts["avg_ret"],
                    n_closed_trades=int(ts["n"]),
                    exit_reasons=dict(ts["exit_reasons"]),
                    vs_spy=vs,
                )
            )
            ledger.write_snapshot("cloud_end")
            ledger.close()
            if not keep_ledgers:
                db = ledger_root / "paper_year.db"
                if db.is_file():
                    db.unlink(missing_ok=True)
                audit = ledger_root / "audit"
                if audit.is_dir():
                    shutil.rmtree(audit, ignore_errors=True)
        except Exception as e:
            logger.exception("Strategy %s failed", sid)
            summaries.append(
                StrategyRunSummary(
                    strategy_id=sid,
                    label=label,
                    days_run=0,
                    n_entries=0,
                    n_exits=0,
                    n_signals=0,
                    final_equity=capital0,
                    total_return=0.0,
                    total_commission=0.0,
                    total_fees=0.0,
                    hard_kill=False,
                    kill_trips=0,
                    run_id="",
                    error=str(e),
                    signal_mode=signal_mode,
                )
            )

    batch = CloudBatchResult(
        as_of=as_of,
        n_strategies=len(summaries),
        data_sources=sources,
        strategies=summaries,
        out_dir=str(day_dir),
    )
    _write_master_reports(
        batch,
        day_dir,
        latest_dir,
        equity_series,
        capital0,
        start_d,
        end_d,
        benchmarks={"spy_bh": spy_bh, "eq_weight_bh": eq_bh},
        feed=feed,
    )
    return batch


def _write_master_reports(
    batch: CloudBatchResult,
    day_dir: Path,
    latest_dir: Path,
    equity_series: Dict[str, List[Dict[str, Any]]],
    capital0: float,
    start_d: date,
    end_d: date,
    benchmarks: Optional[Dict[str, Any]] = None,
    feed: Any = None,
) -> None:
    benchmarks = benchmarks or {}
    spy_bh = benchmarks.get("spy_bh")
    eq_bh = benchmarks.get("eq_weight_bh")
    # SPY equity curve for comparison
    if feed is not None and "SPY" in getattr(feed, "tickers", []):
        spy_curve = []
        days = feed.session_days(start_d, end_d)
        c0 = None
        for d in days:
            b = feed.bar("SPY", d)
            if b is None:
                continue
            c = float(b.close)
            if c0 is None:
                c0 = c
            spy_curve.append(
                {
                    "date": d.isoformat(),
                    "equity": capital0 * (c / c0) if c0 else capital0,
                }
            )
        if spy_curve:
            equity_series["SPY_BH"] = spy_curve

    payload = batch.to_dict()
    payload["window"] = {"start": start_d.isoformat(), "end": end_d.isoformat()}
    payload["capital0"] = capital0
    payload["benchmarks"] = {
        "spy_bh": spy_bh,
        "eq_weight_bh": eq_bh,
    }
    payload["ranking"] = [s.to_dict() for s in batch.ranking()]

    (day_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (day_dir / "equity_curves.json").write_text(
        json.dumps(equity_series, indent=2), encoding="utf-8"
    )

    real_src = {
        k: v
        for k, v in batch.data_sources.items()
        if not str(v).startswith("synthetic") and v != "missing"
    }
    data_note = (
        f"**Data:** REAL free market ({len(real_src)}/{len(batch.data_sources)} tickers) — "
        f"`{', '.join(sorted(set(real_src.values())) or ['none'])}`"
        if real_src
        else "**Data:** WARNING non-real / missing sources"
    )
    spy_s = f"{spy_bh:.2%}" if spy_bh is not None else "n/a"
    eq_s = f"{eq_bh:.2%}" if eq_bh is not None else "n/a"
    lines = [
        f"# Paper cloud multi-strategy — `{batch.as_of}`",
        "",
        f"**Window:** {start_d} → {end_d} · **Capital:** VIRTUAL ${capital0:,.0f} · **mode:** paper",
        "",
        data_note,
        "",
        f"**Benchmarks:** SPY B&H **{spy_s}** · Equal-weight names B&H **{eq_s}**",
        "",
        "Free cloud batch (GitHub Actions). Not financial advice.",
        "",
        "## Ranking by total return",
        "",
        "| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |",
        "|------|----------|------|--------|--------|----|----|--------|---------|------|",
    ]
    for i, s in enumerate(batch.ranking(), 1):
        wr = f"{s.win_rate:.1%}" if s.win_rate is not None else "n/a"
        pf = f"{s.profit_factor:.2f}" if s.profit_factor is not None else "n/a"
        vs = f"{s.vs_spy:+.2%}" if s.vs_spy is not None else "n/a"
        lines.append(
            f"| {i} | `{s.strategy_id}` | `{s.signal_mode}` | {s.total_return:.2%} | "
            f"{vs} | {wr} | {pf} | {s.n_closed_trades} | {s.n_entries} | "
            f"{'YES' if s.hard_kill else 'no'} |"
        )
    failed = [s for s in batch.strategies if s.error]
    if failed:
        lines += ["", "## Failures", ""]
        for s in failed:
            lines.append(f"- `{s.strategy_id}`: {s.error}")
    lines += [
        "",
        "## Exit reasons (per strategy)",
        "",
    ]
    for s in batch.ranking():
        if s.exit_reasons:
            er = ", ".join(f"{k}={v}" for k, v in sorted(s.exit_reasons.items()))
            lines.append(f"- `{s.strategy_id}`: {er}")
    lines += [
        "",
        "## Data sources",
        "",
    ]
    for t, src in sorted(batch.data_sources.items()):
        lines.append(f"- `{t}`: {src}")
    lines += [
        "",
        "## Per-strategy digests",
        "",
        "See `strategies/<id>/dashboard.html`, `daily/`, and `closed_trades.csv`.",
        "",
        "---",
        f"_Generated {datetime.now(timezone.utc).isoformat()} · paper only_",
        "",
    ]
    md = "\n".join(lines)
    (day_dir / "SUMMARY.md").write_text(md, encoding="utf-8")

    html = _comparison_html(batch, equity_series, capital0, start_d, end_d, spy_bh=spy_bh)
    (day_dir / "dashboard.html").write_text(html, encoding="utf-8")

    for name in ("summary.json", "SUMMARY.md", "dashboard.html", "equity_curves.json"):
        src = day_dir / name
        if src.is_file():
            shutil.copy2(src, latest_dir / name)

    # Append history index
    index_path = day_dir.parent.parent / "INDEX.md"
    _update_history_index(index_path, batch)


def _update_history_index(index_path: Path, batch: CloudBatchResult) -> None:
    import re

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if batch.ranking():
        top = batch.ranking()[0]
        line = (
            f"| {batch.as_of} | {batch.n_strategies} | `{top.strategy_id}` | "
            f"{top.total_return:.2%} | [open](history/{batch.as_of}/SUMMARY.md) |"
        )
    else:
        line = (
            f"| {batch.as_of} | {batch.n_strategies} | — | — | "
            f"[open](history/{batch.as_of}/SUMMARY.md) |"
        )

    header = (
        "# Paper cloud history\n\n"
        "Free multi-strategy paper runs (GitHub Actions). Virtual capital only.\n\n"
        "| Date | N strats | Top strategy | Return | Link |\n"
        "|------|----------|--------------|--------|------|\n"
    )
    if index_path.is_file():
        text = index_path.read_text(encoding="utf-8")
        if re.search(rf"\| {re.escape(batch.as_of)} \|", text):
            text = re.sub(
                rf"\| {re.escape(batch.as_of)} \|.*(\n|$)",
                line + "\n",
                text,
            )
            index_path.write_text(text, encoding="utf-8")
            return
        if "| Date |" in text:
            parts = text.split("\n")
            out: List[str] = []
            inserted = False
            for p in parts:
                out.append(p)
                if (not inserted) and p.startswith("|------"):
                    out.append(line)
                    inserted = True
            if not inserted:
                out.append(line)
            index_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
            return
    index_path.write_text(header + line + "\n", encoding="utf-8")


def _comparison_html(
    batch: CloudBatchResult,
    equity_series: Dict[str, List[Dict[str, Any]]],
    capital0: float,
    start_d: date,
    end_d: date,
    spy_bh: Optional[float] = None,
) -> str:
    rows = []
    for i, s in enumerate(batch.ranking(), 1):
        wr = f"{s.win_rate:.1%}" if s.win_rate is not None else "n/a"
        pf = f"{s.profit_factor:.2f}" if s.profit_factor is not None else "n/a"
        vs = f"{s.vs_spy:+.2%}" if s.vs_spy is not None else "n/a"
        rows.append(
            f"<tr><td>{i}</td><td><code>{s.strategy_id}</code></td><td>{s.signal_mode}</td>"
            f"<td>{s.total_return:.2%}</td><td>{vs}</td><td>{wr}</td><td>{pf}</td>"
            f"<td>{s.n_closed_trades}</td><td>{'KILL' if s.hard_kill else 'ok'}</td></tr>"
        )
    spy_s = f"{spy_bh:.2%}" if spy_bh is not None else "n/a"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Paper cloud {batch.as_of}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f1419;color:#e7ecf3;padding:1.5rem}}
h1{{font-size:1.3rem}}.badge{{background:#2a3f5f;color:#9ecbff;padding:.15rem .5rem;border-radius:999px;font-size:.75rem;margin-right:.3rem}}
table{{width:100%;border-collapse:collapse;margin-top:1rem}} th,td{{padding:.4rem;border-bottom:1px solid #2a3548;text-align:left;font-size:.9rem}}
th{{color:#8b9bb4}} a{{color:#3d9cf0}} .card{{background:#1a2332;border-radius:12px;padding:1rem;margin:1rem 0}}
</style></head><body>
<h1>Paper cloud multi-strategy — {batch.as_of}</h1>
<div>
<span class="badge">PAPER</span>
<span class="badge">VIRTUAL ${capital0:,.0f}</span>
<span class="badge">SPY B&amp;H {spy_s}</span>
</div>
<p>Window {start_d} → {end_d}. closed_trades.csv under each strategy.</p>
<div class="card"><h2>Ranking</h2>
<table><thead><tr><th>#</th><th>ID</th><th>Mode</th><th>Return</th><th>vs SPY</th><th>WR</th><th>PF</th><th>Closed</th><th>Kill</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">Not financial advice. Past paper ≠ future results.</p>
</body></html>"""
