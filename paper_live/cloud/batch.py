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
    s["knobs"] = kn
    rp = dict(s.get("risk_paper") or {})
    if "max_portfolio_dd" in strat:
        rp["max_portfolio_dd"] = float(strat["max_portfolio_dd"])
    if "kill_dd_from_start" in strat:
        rp["kill_dd_from_start"] = float(strat["kill_dd_from_start"])
    if "max_entries_per_day" in strat:
        rp["max_daily_new_entries"] = int(strat["max_entries_per_day"])
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
    feed, sources = build_cloud_feed(
        tickers,
        cache_dir=cache_dir,
        lookback_calendar_days=lb,
        force_synthetic=force_synthetic,
    )

    days = feed.days
    if not days:
        raise RuntimeError("No trading days in feed")

    # Default window: last ~120 sessions ending at as_of / last available
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
        # ~6 months of sessions
        start_d = days[max(0, days.index(end_d) - 130)]

    base_freeze = load_freeze()
    summaries: List[StrategyRunSummary] = []
    equity_series: Dict[str, List[Dict[str, Any]]] = {}

    for strat in strategies:
        sid = str(strat["id"])
        label = str(strat.get("label") or sid)
        logger.info("Cloud paper strategy %s …", sid)
        try:
            freeze = _override_freeze(base_freeze, strat, capital0)
            ledger_root = day_dir / "ledgers" / sid
            if ledger_root.exists():
                shutil.rmtree(ledger_root, ignore_errors=True)
            ledger = PaperLedger.create_run(
                ledger_root,
                freeze,
                meta={"cloud": True, "as_of": as_of, "label": label},
            )
            univ = [t for t in feed.tickers if t not in ("SPY",)]
            # keep QQQ in universe for signals only if listed — pipeline uses regime separately
            pipe = DailySignalPipeline(
                feed,
                universe=univ,
                min_price=float(strat.get("min_price", 5.0)),
                max_atr_pct=float(strat.get("max_atr_pct", 0.22)),
                min_atr_norm=float(strat.get("min_atr_norm", 0.01)),
                require_regime=bool(strat.get("require_regime", True)),
                regime_symbol="QQQ" if "QQQ" in feed.tickers else (
                    "SPY" if "SPY" in feed.tickers else feed.tickers[0]
                ),
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
                max_entries_per_day=int(
                    strat.get("max_entries_per_day", freeze.strategy.risk_paper.get("max_daily_new_entries", 5))
                ),
                enable_risk=True,
            )
            # if no QQQ, disable regime
            if "QQQ" not in feed.tickers and "SPY" not in feed.tickers:
                session.pipeline.require_regime = False

            result = session.run(start_d, end_d)
            rep_dir = day_dir / "strategies" / sid
            generate_reports_for_run(ledger, rep_dir, write_html=True)

            # equity curve from nav
            nav = ledger.list_nav()
            equity_series[sid] = [
                {"date": r["date"], "equity": float(r["equity"])} for r in nav
            ]

            tr = result.final_equity / capital0 - 1.0
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
                )
            )
            ledger.write_snapshot("cloud_end")
            ledger.close()
            if not keep_ledgers:
                # keep digests; drop bulky sqlite to save git size
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
                )
            )

    batch = CloudBatchResult(
        as_of=as_of,
        n_strategies=len(summaries),
        data_sources=sources,
        strategies=summaries,
        out_dir=str(day_dir),
    )
    _write_master_reports(batch, day_dir, latest_dir, equity_series, capital0, start_d, end_d)
    return batch


def _write_master_reports(
    batch: CloudBatchResult,
    day_dir: Path,
    latest_dir: Path,
    equity_series: Dict[str, List[Dict[str, Any]]],
    capital0: float,
    start_d: date,
    end_d: date,
) -> None:
    payload = batch.to_dict()
    payload["window"] = {"start": start_d.isoformat(), "end": end_d.isoformat()}
    payload["capital0"] = capital0
    payload["ranking"] = [s.to_dict() for s in batch.ranking()]

    (day_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (day_dir / "equity_curves.json").write_text(
        json.dumps(equity_series, indent=2), encoding="utf-8"
    )

    # Markdown comparison
    lines = [
        f"# Paper cloud multi-strategy — `{batch.as_of}`",
        "",
        f"**Window:** {start_d} → {end_d} · **Capital:** VIRTUAL ${capital0:,.0f} · **mode:** paper",
        "",
        "Free cloud batch (GitHub Actions). Not financial advice.",
        "",
        "## Ranking by total return",
        "",
        "| Rank | Strategy | Label | Return | Final $ | Entries | Exits | Commission | Kill |",
        "|------|----------|-------|--------|---------|---------|-------|------------|------|",
    ]
    for i, s in enumerate(batch.ranking(), 1):
        lines.append(
            f"| {i} | `{s.strategy_id}` | {s.label} | {s.total_return:.2%} | "
            f"${s.final_equity:,.0f} | {s.n_entries} | {s.n_exits} | "
            f"${s.total_commission:.2f} | {'YES' if s.hard_kill else 'no'} |"
        )
    failed = [s for s in batch.strategies if s.error]
    if failed:
        lines += ["", "## Failures", ""]
        for s in failed:
            lines.append(f"- `{s.strategy_id}`: {s.error}")
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
        "See `strategies/<id>/dashboard.html` and `daily/`.",
        "",
        "---",
        f"_Generated {datetime.now(timezone.utc).isoformat()} · paper only_",
        "",
    ]
    md = "\n".join(lines)
    (day_dir / "SUMMARY.md").write_text(md, encoding="utf-8")

    # Comparison HTML
    html = _comparison_html(batch, equity_series, capital0, start_d, end_d)
    (day_dir / "dashboard.html").write_text(html, encoding="utf-8")

    # Copy to latest/
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
) -> str:
    rows = []
    for i, s in enumerate(batch.ranking(), 1):
        rows.append(
            f"<tr><td>{i}</td><td><code>{s.strategy_id}</code></td><td>{s.label}</td>"
            f"<td>{s.total_return:.2%}</td><td>${s.final_equity:,.0f}</td>"
            f"<td>{s.n_entries}</td><td>${s.total_commission:.2f}</td>"
            f"<td>{'KILL' if s.hard_kill else 'ok'}</td></tr>"
        )
    # multi sparkline simplified: table only + note
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
<span class="badge">FREE GITHUB ACTIONS</span>
</div>
<p>Window {start_d} → {end_d}. Study digests under strategies/*/dashboard.html</p>
<div class="card"><h2>Ranking</h2>
<table><thead><tr><th>#</th><th>ID</th><th>Label</th><th>Return</th><th>Final</th><th>Entries</th><th>Comm</th><th>Kill</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">Not financial advice. Past paper ≠ future results.</p>
</body></html>"""
