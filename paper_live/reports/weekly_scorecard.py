"""Weekly paper scorecard from ledger NAV/fills/costs."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from paper_live.ledger import EventType, PaperLedger
from paper_live.ledger.events import utc_now
from paper_live.reports.daily_digest import DailyDigest, build_daily_digest


@dataclass
class WeeklyScorecard:
    run_id: str
    week_start: str
    week_end: str
    strategy_id: str
    mode: str = "paper"
    capital_label: str = "VIRTUAL"
    days: int = 0
    start_equity: Optional[float] = None
    end_equity: Optional[float] = None
    week_return: Optional[float] = None
    max_dd: Optional[float] = None
    rolling_sharpe_approx: Optional[float] = None
    n_fills: int = 0
    n_buys: int = 0
    n_sells: int = 0
    commission: float = 0.0
    fees: float = 0.0
    slippage_est: float = 0.0
    turnover: float = 0.0
    cost_drag_bps: Optional[float] = None
    n_kill_events: int = 0
    n_rejects: int = 0
    micro_trade_pct: float = 0.0  # fills with notional < 2k
    flags: List[str] = field(default_factory=list)
    daily: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Paper weekly scorecard — `{self.week_start}` → `{self.week_end}`",
            "",
            f"**Run:** `{self.run_id}` · **strategy:** `{self.strategy_id}` · "
            f"**mode:** {self.mode} · **capital:** {self.capital_label}",
            "",
            "## Performance",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Session days | {self.days} |",
            f"| Start equity | {_m(self.start_equity)} |",
            f"| End equity | {_m(self.end_equity)} |",
            f"| Week return | {_p(self.week_return)} |",
            f"| Max DD (week path) | {_p(self.max_dd)} |",
            f"| Sharpe approx (daily) | "
            f"{f'{self.rolling_sharpe_approx:.2f}' if self.rolling_sharpe_approx is not None else '—'} |",
            "",
            "## Costs & activity",
            "",
            "| Metric | Value | Soft gate |",
            "|--------|-------|-----------|",
            f"| Fills | {self.n_fills} (B{self.n_buys}/S{self.n_sells}) | |",
            f"| Commission | {_m(self.commission)} | |",
            f"| Fees | {_m(self.fees)} | |",
            f"| Slippage est. | {_m(self.slippage_est)} | |",
            f"| Turnover | {_m(self.turnover)} | flag if >> research |",
            f"| Cost drag (bps of start eq) | {self.cost_drag_bps if self.cost_drag_bps is not None else '—'} | flag if elevated |",
            f"| Micro fills (&lt;$2k notional) | {self.micro_trade_pct:.1%} | should stay ~0% |",
            f"| Rejects | {self.n_rejects} | |",
            f"| Kill events | {self.n_kill_events} | |",
            "",
        ]
        if self.flags:
            lines += ["## Flags", ""]
            for f in self.flags:
                lines.append(f"- ⚠ {f}")
            lines.append("")
        if self.daily:
            lines += [
                "## Daily NAV",
                "",
                "| Date | Equity | DD | Fills | Comm |",
                "|------|--------|-----|-------|------|",
            ]
            for d in self.daily:
                lines.append(
                    f"| {d.get('day')} | {_m(d.get('equity'))} | {_p(d.get('dd_from_peak'))} | "
                    f"{d.get('n_fills', 0)} | {_m(d.get('commission'))} |"
                )
            lines.append("")
        lines += [
            "---",
            f"_Generated {self.generated_at} · research paper only · not financial advice._",
            "",
        ]
        return "\n".join(lines)


def _m(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"${float(x):,.2f}"


def _p(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{float(x):.2%}"


def _week_bounds(day: Union[str, date]) -> tuple[str, str]:
    d = date.fromisoformat(str(day)[:10])
    # ISO week: Monday start
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()


def build_weekly_scorecard(
    ledger: PaperLedger,
    *,
    week_of: Optional[Union[str, date]] = None,
    week_start: Optional[Union[str, date]] = None,
    week_end: Optional[Union[str, date]] = None,
) -> WeeklyScorecard:
    """Build scorecard for ISO week containing ``week_of``, or explicit range."""
    if week_start and week_end:
        ws, we = str(week_start)[:10], str(week_end)[:10]
    else:
        ref = week_of or utc_now().date().isoformat()
        ws, we = _week_bounds(ref)

    run = ledger.get_run()
    nav = ledger.list_nav(start=ws, end=we)
    fills = ledger.list_fills(start=ws, end=we)
    costs = ledger.list_costs(start=ws, end=we)
    orders = ledger.list_orders(start=ws, end=we)

    # Per-day digests for table
    days_set = sorted({r["date"] for r in nav} | {str(f.get("ts", ""))[:10] for f in fills})
    daily_rows: List[Dict[str, Any]] = []
    for d in days_set:
        if not d or d < ws or d > we:
            continue
        dig = build_daily_digest(ledger, d)
        daily_rows.append(
            {
                "day": dig.day,
                "equity": dig.equity,
                "dd_from_peak": dig.dd_from_peak,
                "n_fills": dig.n_fills,
                "commission": dig.commission,
            }
        )

    start_eq = float(nav[0]["equity"]) if nav else None
    end_eq = float(nav[-1]["equity"]) if nav else None
    week_ret = None
    if start_eq and start_eq > 0 and end_eq is not None:
        week_ret = end_eq / start_eq - 1.0

    # max dd along week equity path
    max_dd = None
    if nav:
        peak = float(nav[0]["equity"])
        mdd = 0.0
        for r in nav:
            eq = float(r["equity"])
            peak = max(peak, eq)
            if peak > 0:
                mdd = min(mdd, eq / peak - 1.0)
        max_dd = mdd

    rets: List[float] = []
    for i in range(1, len(nav)):
        a, b = float(nav[i - 1]["equity"]), float(nav[i]["equity"])
        if a > 0:
            rets.append(b / a - 1.0)
    sharpe = None
    if len(rets) >= 3:
        arr = np.asarray(rets, dtype=float)
        sd = float(np.std(arr, ddof=1))
        if sd > 1e-12:
            sharpe = float(np.mean(arr) / sd * math.sqrt(252.0))
        else:
            sharpe = 0.0

    commission = sum(float(c.get("commission") or 0) for c in costs) if costs else sum(
        float(f.get("commission") or 0) for f in fills
    )
    fees = sum(float(c.get("fees") or 0) for c in costs) if costs else sum(
        float(f.get("fees") or 0) for f in fills
    )
    slip = sum(float(c.get("slippage_est") or 0) for c in costs)
    turnover = sum(float(c.get("turnover") or 0) for c in costs)
    if not turnover:
        turnover = sum(abs(float(f.get("qty") or 0) * float(f.get("price") or 0)) for f in fills)

    cost_drag_bps = None
    if start_eq and start_eq > 0:
        cost_drag_bps = round((commission + fees + slip) / start_eq * 10_000.0, 2)

    micro = 0
    for f in fills:
        notional = abs(float(f.get("qty") or 0) * float(f.get("price") or 0))
        if 0 < notional < 2000:
            micro += 1
    micro_pct = (micro / len(fills)) if fills else 0.0

    n_rejects = sum(
        1 for o in orders if str(o.get("status") or "").lower() in ("rejected", "cancelled")
    )
    n_kill = ledger.count_events(EventType.KILL_SWITCH, start=ws, end=we)

    flags: List[str] = []
    if micro_pct > 0.05:
        flags.append(f"micro fills elevated: {micro_pct:.1%}")
    if cost_drag_bps is not None and cost_drag_bps > 50:
        flags.append(f"cost drag {cost_drag_bps} bps — review slippage/commission")
    if n_kill:
        flags.append(f"kill switch fired {n_kill} time(s) this week")
    if max_dd is not None and max_dd <= -0.15:
        flags.append(f"week path max DD {max_dd:.1%} severe")
    if sharpe is not None and sharpe < -1.0:
        flags.append(f"week daily Sharpe approx {sharpe:.2f} < -1")

    return WeeklyScorecard(
        run_id=ledger.run_id,
        week_start=ws,
        week_end=we,
        strategy_id=str(run.get("strategy") or ledger.strategy_id),
        days=len(nav) or len(days_set),
        start_equity=start_eq,
        end_equity=end_eq,
        week_return=week_ret,
        max_dd=max_dd,
        rolling_sharpe_approx=sharpe,
        n_fills=len(fills),
        n_buys=sum(1 for f in fills if str(f.get("side")).lower() == "buy"),
        n_sells=sum(1 for f in fills if str(f.get("side")).lower() == "sell"),
        commission=float(commission),
        fees=float(fees),
        slippage_est=float(slip),
        turnover=float(turnover),
        cost_drag_bps=cost_drag_bps,
        n_kill_events=n_kill,
        n_rejects=n_rejects,
        micro_trade_pct=float(micro_pct),
        flags=flags,
        daily=daily_rows,
        generated_at=utc_now().isoformat(),
    )


def write_weekly_scorecard(
    card: WeeklyScorecard,
    out_dir: Union[str, Path],
    *,
    write_json: bool = True,
    write_md: bool = True,
) -> Dict[str, Path]:
    out = Path(out_dir)
    weekly = out / "weekly"
    weekly.mkdir(parents=True, exist_ok=True)
    tag = f"{card.week_start}_{card.week_end}"
    paths: Dict[str, Path] = {}
    if write_json:
        p = weekly / f"{tag}.json"
        p.write_text(json.dumps(card.to_dict(), indent=2, default=str), encoding="utf-8")
        paths["json"] = p
    if write_md:
        p = weekly / f"{tag}.md"
        p.write_text(card.to_markdown(), encoding="utf-8")
        paths["md"] = p
    return paths
