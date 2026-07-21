"""RTH schedule clock — phase classification for paper runner (LIV-07)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Optional, Tuple, Union
from zoneinfo import ZoneInfo

from paper_live.freeze import ScheduleConfig


class SessionPhase(str, Enum):
    PRE_OPEN = "pre_open"
    OPEN_AUCTION = "open_auction"  # first N minutes after open
    ENTRY_WINDOW = "entry_window"
    MIDDAY = "midday"
    INTRADAY = "intraday"
    EXIT_CHECK = "exit_check"
    FORCE_FLATTEN = "force_flatten"
    POST_CLOSE = "post_close"
    NIGHT = "night"
    CLOSED = "closed"


def _parse_hhmm(hhmm: str) -> time:
    parts = str(hhmm).strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return time(h, m)


def _mins(t: time) -> int:
    return t.hour * 60 + t.minute


@dataclass(frozen=True)
class ScheduleClock:
    """Map a timestamp to a session phase using freeze schedule.json."""

    schedule: ScheduleConfig
    tz_name: str = "America/New_York"

    @classmethod
    def from_schedule(cls, schedule: ScheduleConfig) -> "ScheduleClock":
        return cls(schedule=schedule, tz_name=schedule.timezone or "America/New_York")

    @property
    def tz(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.tz_name)
        except Exception:
            return ZoneInfo("America/New_York")

    def localize(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc).astimezone(self.tz)
        return ts.astimezone(self.tz)

    def phase(self, ts: datetime) -> SessionPhase:
        local = self.localize(ts)
        # weekdays only for equity RTH
        if local.weekday() >= 5:
            return SessionPhase.CLOSED

        sch = self.schedule
        m = local.hour * 60 + local.minute
        pre = _mins(_parse_hhmm(sch.pre_open_hhmm))
        rth = _mins(_parse_hhmm(sch.rth_open_hhmm))
        entry_s = _mins(_parse_hhmm(sch.entry_window_start_hhmm))
        entry_e = _mins(_parse_hhmm(sch.entry_window_end_hhmm))
        mid = _mins(_parse_hhmm(sch.midday_rescan_hhmm))
        exit_s = _mins(_parse_hhmm(sch.exit_check_start_hhmm))
        exit_e = _mins(_parse_hhmm(sch.exit_check_end_hhmm))
        flat = _mins(_parse_hhmm(sch.force_flatten_hhmm))
        post = _mins(_parse_hhmm(sch.post_close_hhmm))
        night = _mins(_parse_hhmm(sch.night_job_hhmm))
        skip = int(sch.skip_first_minutes_after_open)

        if m < pre:
            return SessionPhase.NIGHT if m < 4 * 60 else SessionPhase.CLOSED
        if pre <= m < rth:
            return SessionPhase.PRE_OPEN
        if rth <= m < rth + skip:
            return SessionPhase.OPEN_AUCTION
        if entry_s <= m <= entry_e:
            return SessionPhase.ENTRY_WINDOW
        if abs(m - mid) <= 2:
            return SessionPhase.MIDDAY
        if exit_s <= m <= exit_e:
            return SessionPhase.EXIT_CHECK
        if flat <= m < post:
            return SessionPhase.FORCE_FLATTEN
        if post <= m < night:
            return SessionPhase.POST_CLOSE
        if m >= night:
            return SessionPhase.NIGHT
        if rth + skip <= m < exit_s:
            return SessionPhase.INTRADAY
        return SessionPhase.CLOSED

    def is_entry_allowed(self, ts: datetime) -> bool:
        return self.phase(ts) == SessionPhase.ENTRY_WINDOW

    def is_exit_check(self, ts: datetime) -> bool:
        return self.phase(ts) in (SessionPhase.EXIT_CHECK, SessionPhase.FORCE_FLATTEN)

    def phase_for_replay_step(self, step: str) -> SessionPhase:
        """Map logical daily replay steps to phases (no wall clock)."""
        s = step.lower()
        if s in ("open", "entry"):
            return SessionPhase.ENTRY_WINDOW
        if s in ("exit", "stops"):
            return SessionPhase.EXIT_CHECK
        if s in ("close", "nav", "signal"):
            return SessionPhase.POST_CLOSE
        if s == "pre":
            return SessionPhase.PRE_OPEN
        return SessionPhase.INTRADAY
