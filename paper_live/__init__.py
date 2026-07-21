"""Paper Live Year — virtual capital only (no real money).

LIV-01: config freeze (strategy + costs + schedule + universe)
LIV-02: append-only ledger (SQLite + JSONL audit)
LIV-05: paper OMS + fill model + commissions
LIV-03/04: daily replay datafeed + signal → entry session
LIV-06/07: risk kill switch + RTH paper runner
"""
from __future__ import annotations

__version__ = "0.5.0"

from paper_live.freeze import (
    CostModel,
    PaperFreeze,
    ScheduleConfig,
    StrategyFreeze,
    UniverseConfig,
    assert_paper_only,
    compute_config_hash,
    load_freeze,
)
from paper_live.ledger import EventType, PaperLedger, new_run_id
from paper_live.oms import (
    FillModel,
    FillQuote,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperBroker,
    PaperOrder,
    PortfolioState,
    SimulatedFill,
)
from paper_live.datafeed import DailyReplayFeed
from paper_live.replay_session import ReplaySession, ReplaySessionResult
from paper_live.signals import DailySignalPipeline, EntryCandidate, confirm_entry
from paper_live.risk import KillSwitch, PortfolioRisk, RiskConfig
from paper_live.runner import PaperRunner, build_runner
from paper_live.schedule_clock import ScheduleClock, SessionPhase
from paper_live.reports import (
    DailyDigest,
    WeeklyScorecard,
    build_daily_digest,
    build_weekly_scorecard,
    write_daily_digest,
    write_html_dashboard,
    write_weekly_scorecard,
)
from paper_live.reports.pipeline import generate_reports_for_run

__all__ = [
    "CostModel",
    "DailyDigest",
    "DailyReplayFeed",
    "DailySignalPipeline",
    "EntryCandidate",
    "EventType",
    "FillModel",
    "FillQuote",
    "KillSwitch",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "PaperFreeze",
    "PaperLedger",
    "PaperOrder",
    "PaperRunner",
    "PortfolioRisk",
    "PortfolioState",
    "ReplaySession",
    "ReplaySessionResult",
    "RiskConfig",
    "ScheduleClock",
    "ScheduleConfig",
    "SessionPhase",
    "SimulatedFill",
    "StrategyFreeze",
    "UniverseConfig",
    "WeeklyScorecard",
    "assert_paper_only",
    "build_daily_digest",
    "build_runner",
    "build_weekly_scorecard",
    "compute_config_hash",
    "confirm_entry",
    "generate_reports_for_run",
    "load_freeze",
    "new_run_id",
    "write_daily_digest",
    "write_html_dashboard",
    "write_weekly_scorecard",
    "__version__",
]
