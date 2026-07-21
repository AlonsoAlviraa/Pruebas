"""LIV-07: Paper RTH runner — replay or guarded live loop (virtual capital only)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.freeze import PaperFreeze, assert_paper_only, load_freeze
from paper_live.ledger import EventType, PaperLedger
from paper_live.ledger.events import utc_now
from paper_live.replay_session import ReplaySession, ReplaySessionResult
from paper_live.risk.kill_switch import KillSwitch
from paper_live.risk.portfolio_risk import PortfolioRisk, RiskConfig
from paper_live.schedule_clock import ScheduleClock, SessionPhase


@dataclass
class RunnerResult:
    mode: str
    result: Optional[ReplaySessionResult] = None
    phases_executed: List[str] = field(default_factory=list)
    kill_state: Optional[Dict[str, Any]] = None
    risk_last: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mode": self.mode,
            "capital_label": "VIRTUAL",
            "phases_executed": list(self.phases_executed),
            "kill_state": self.kill_state,
            "risk_last": self.risk_last,
        }
        if self.result is not None:
            out["session"] = self.result.to_dict()
        return out


class PaperRunner:
    """Orchestrates paper sessions with schedule awareness + kill switch.

    Modes:
    - ``replay``: historical daily loop (no TRAD_PAPER_ONLY env required)
    - ``live`` / ``live_day_stub``: requires TRAD_PAPER_ONLY=1
    """

    def __init__(
        self,
        freeze: Optional[PaperFreeze] = None,
        *,
        feed: Optional[DailyReplayFeed] = None,
        ledger: Optional[PaperLedger] = None,
        session: Optional[ReplaySession] = None,
    ):
        self.freeze = freeze or load_freeze()
        self.feed = feed
        self.ledger = ledger
        self.clock = ScheduleClock.from_schedule(self.freeze.schedule)
        self.session = session
        self.risk: Optional[PortfolioRisk] = None
        self.kill: Optional[KillSwitch] = None
        self.phases_executed: List[str] = []

    def _ensure_session(self) -> ReplaySession:
        if self.session is not None:
            return self.session
        if self.feed is None:
            raise ValueError("PaperRunner requires feed or session")
        self.session = ReplaySession(
            self.feed,
            self.freeze,
            ledger=self.ledger,
            enable_risk=True,
        )
        self.risk = self.session.risk
        self.kill = self.session.kill
        return self.session

    def _sync_risk_refs(self, session: ReplaySession) -> KillSwitch:
        if session.kill is None or session.risk is None:
            session.attach_risk()
        self.risk = session.risk
        self.kill = session.kill
        assert self.kill is not None
        return self.kill

    def run_replay(
        self,
        start: Union[str, date],
        end: Union[str, date],
    ) -> RunnerResult:
        """Run multi-day paper replay with risk kill switch armed."""
        assert_paper_only(require_env=False)
        session = self._ensure_session()
        kill = self._sync_risk_refs(session)
        self.phases_executed = [
            SessionPhase.ENTRY_WINDOW.value,
            SessionPhase.EXIT_CHECK.value,
            SessionPhase.POST_CLOSE.value,
        ]
        if self.ledger is not None:
            self.ledger.append_event(
                EventType.HEARTBEAT,
                {
                    "runner": "replay",
                    "start": str(start),
                    "end": str(end),
                    "risk": RiskConfig.from_risk_paper(
                        self.freeze.strategy.risk_paper
                    ).to_dict(),
                    "mode": "paper",
                },
            )
        result = session.run(start, end)
        snap = kill.evaluate(update_history=False)
        return RunnerResult(
            mode="replay",
            result=result,
            phases_executed=list(self.phases_executed),
            kill_state=kill.state.to_dict(),
            risk_last=snap.to_dict(),
        )

    def tick_live(
        self,
        ts: Optional[datetime] = None,
        *,
        require_env: bool = True,
    ) -> Dict[str, Any]:
        """Single live tick: classify phase, heartbeat, risk check.

        Does not place real orders.
        """
        assert_paper_only(require_env=require_env)
        now = ts or utc_now()
        phase = self.clock.phase(now)
        self.phases_executed.append(phase.value)

        payload: Dict[str, Any] = {
            "ts": now.isoformat(),
            "phase": phase.value,
            "entry_allowed_by_clock": self.clock.is_entry_allowed(now),
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }

        if self.session is not None:
            kill = self._sync_risk_refs(self.session)
            snap = kill.evaluate(update_history=False)
            payload["risk"] = snap.to_dict()
            payload["kill"] = kill.state.to_dict()
            payload["equity"] = self.session.broker.state.equity()

        if self.ledger is not None:
            self.ledger.append_event(EventType.HEARTBEAT, payload, ts=now)
        return payload

    def run_live_day_stub(
        self,
        day: Union[str, date],
        *,
        require_env: bool = True,
    ) -> RunnerResult:
        """Simulate RTH phase sequence for one calendar day (ops smoke)."""
        assert_paper_only(require_env=require_env)
        d = pd.Timestamp(day).date()
        phases = [
            (SessionPhase.PRE_OPEN, 9, 0),
            (SessionPhase.ENTRY_WINDOW, 9, 45),
            (SessionPhase.MIDDAY, 12, 0),
            (SessionPhase.EXIT_CHECK, 15, 40),
            (SessionPhase.POST_CLOSE, 16, 15),
        ]
        executed: List[str] = []
        res: Optional[ReplaySessionResult] = None
        if self.feed is not None:
            session = self._ensure_session()
            self._sync_risk_refs(session)
            res = session.run(d, d)

        for ph, hh, mm in phases:
            local = datetime(d.year, d.month, d.day, hh, mm, tzinfo=self.clock.tz)
            ts_utc = local.astimezone(timezone.utc)
            executed.append(ph.value)
            if self.ledger is not None:
                self.ledger.append_event(
                    EventType.HEARTBEAT,
                    {
                        "runner": "live_day_stub",
                        "phase": ph.value,
                        "day": d.isoformat(),
                        "mode": "paper",
                    },
                    ts=ts_utc,
                )

        kill_state = self.kill.state.to_dict() if self.kill else None
        risk_last = None
        if self.kill is not None:
            risk_last = self.kill.evaluate(update_history=False).to_dict()

        return RunnerResult(
            mode="live_day_stub",
            result=res,
            phases_executed=executed,
            kill_state=kill_state,
            risk_last=risk_last,
        )


def build_runner(
    *,
    config_dir: Optional[Union[str, Path]] = None,
    ledger_root: Optional[Union[str, Path]] = None,
    data_root: Union[str, Path] = "data",
    tickers: Optional[Sequence[str]] = None,
    synthetic: bool = False,
    run_id: Optional[str] = None,
) -> PaperRunner:
    """Factory: freeze + optional ledger + feed."""
    freeze = load_freeze(config_dir)
    ledger = None
    if ledger_root is not None:
        ledger = PaperLedger.create_run(
            Path(ledger_root),
            freeze,
            run_id=run_id,
            meta={"builder": "build_runner"},
        )
    feed = None
    if tickers:
        if synthetic:
            feed = DailyReplayFeed.from_synthetic(list(tickers), n_days=400, seed=42)
        else:
            feed = DailyReplayFeed.from_data_root(data_root, list(tickers))
    return PaperRunner(freeze, feed=feed, ledger=ledger)
