"""Event types and id helpers for the paper ledger."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import FrozenSet


class EventType(str, Enum):
    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    BAR = "bar"
    SIGNAL_COMPUTED = "signal_computed"
    ENTRY_CANDIDATE = "entry_candidate"
    ENTRY_REJECTED = "entry_rejected"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACK = "order_ack"
    ORDER_REJECT = "order_reject"
    FILL = "fill"
    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"
    STOP_UPDATED = "stop_updated"
    RISK_BLOCK = "risk_block"
    RETRAIN_START = "retrain_start"
    RETRAIN_END = "retrain_end"
    DAILY_NAV = "daily_nav"
    KILL_SWITCH = "kill_switch"
    HEARTBEAT = "heartbeat"
    CORRECTION = "correction"
    RUN_INIT = "run_init"
    SNAPSHOT = "snapshot"


EVENT_TYPES: FrozenSet[str] = frozenset(e.value for e in EventType)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    """Time-sortable unique run id."""
    ts = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"paper_{ts}_{uuid.uuid4().hex[:10]}"


def new_event_id() -> str:
    return uuid.uuid4().hex


def new_order_id() -> str:
    return f"ord_{uuid.uuid4().hex[:16]}"


def new_fill_id() -> str:
    return f"fill_{uuid.uuid4().hex[:16]}"


def new_decision_id() -> str:
    return f"dec_{uuid.uuid4().hex[:16]}"
