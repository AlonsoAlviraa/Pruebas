"""LIV-08: paper digests and HTML reports (virtual capital only)."""
from __future__ import annotations

from paper_live.reports.daily_digest import DailyDigest, build_daily_digest, write_daily_digest
from paper_live.reports.html_report import write_html_dashboard
from paper_live.reports.weekly_scorecard import (
    WeeklyScorecard,
    build_weekly_scorecard,
    write_weekly_scorecard,
)

__all__ = [
    "DailyDigest",
    "WeeklyScorecard",
    "build_daily_digest",
    "build_weekly_scorecard",
    "write_daily_digest",
    "write_html_dashboard",
    "write_weekly_scorecard",
]
