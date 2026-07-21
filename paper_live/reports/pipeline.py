"""End-to-end: digests + weekly + HTML for a paper run."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from paper_live.ledger import PaperLedger
from paper_live.reports.daily_digest import DailyDigest, build_daily_digest, write_daily_digest
from paper_live.reports.html_report import write_html_dashboard
from paper_live.reports.weekly_scorecard import (
    WeeklyScorecard,
    build_weekly_scorecard,
    write_weekly_scorecard,
)


@dataclass
class DigestBundle:
    daily: List[DailyDigest]
    weekly: Optional[WeeklyScorecard]
    paths: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_daily": len(self.daily),
            "weekly": self.weekly.to_dict() if self.weekly else None,
            "paths": self.paths,
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }


def generate_reports_for_run(
    ledger: PaperLedger,
    out_dir: Union[str, Path],
    *,
    days: Optional[List[str]] = None,
    week_of: Optional[Union[str, date]] = None,
    write_html: bool = True,
) -> DigestBundle:
    """Build daily digests for all NAV days (or provided list) + weekly + HTML."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if days is None:
        nav = ledger.list_nav()
        days = [r["date"] for r in nav]
        if not days:
            # fall back to fill dates
            fills = ledger.list_fills()
            days = sorted({str(f.get("ts", ""))[:10] for f in fills if f.get("ts")})

    daily_list: List[DailyDigest] = []
    paths: Dict[str, str] = {}
    for d in days:
        if not d:
            continue
        dig = build_daily_digest(ledger, d)
        daily_list.append(dig)
        written = write_daily_digest(dig, out)
        for k, p in written.items():
            paths[f"daily_{d}_{k}"] = str(p)

    weekly = None
    if days:
        ref = week_of or days[-1]
        weekly = build_weekly_scorecard(ledger, week_of=ref)
        wpaths = write_weekly_scorecard(weekly, out)
        for k, p in wpaths.items():
            paths[f"weekly_{k}"] = str(p)

    if write_html and daily_list:
        html_path = out / "dashboard.html"
        write_html_dashboard(
            html_path,
            title=f"Paper Live — {ledger.strategy_id}",
            run_id=ledger.run_id,
            strategy_id=ledger.strategy_id,
            daily=daily_list,
            weekly=weekly,
        )
        paths["html"] = str(html_path)

    # index markdown
    index_lines = [
        f"# Paper digests — `{ledger.run_id}`",
        "",
        f"Strategy: `{ledger.strategy_id}` · mode=paper · VIRTUAL capital",
        "",
        f"- Days: {len(daily_list)}",
        f"- HTML: `{paths.get('html', '—')}`",
        "",
    ]
    if weekly:
        ret_s = f"{weekly.week_return:.2%}" if weekly.week_return is not None else "—"
        index_lines += [
            f"## Latest week {weekly.week_start} → {weekly.week_end}",
            "",
            f"- Return: {ret_s}",
            f"- Commission: ${weekly.commission:,.2f}",
            f"- Kill events: {weekly.n_kill_events}",
            "",
        ]
    index_path = out / "INDEX.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    paths["index"] = str(index_path)

    return DigestBundle(daily=daily_list, weekly=weekly, paths=paths)
