"""Kill switch: block new entries; hard trip is sticky (LIV-06)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from paper_live.ledger import EventType, PaperLedger
from paper_live.ledger.events import utc_now
from paper_live.oms.paper_broker import PaperBroker
from paper_live.risk.portfolio_risk import PortfolioRisk, RiskSnapshot


@dataclass
class KillSwitchState:
    entries_blocked: bool = False
    hard_kill: bool = False
    last_reasons: List[str] = field(default_factory=list)
    trip_count: int = 0
    last_trip_ts: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_blocked": self.entries_blocked,
            "hard_kill": self.hard_kill,
            "last_reasons": list(self.last_reasons),
            "trip_count": self.trip_count,
            "last_trip_ts": self.last_trip_ts,
            "mode": "paper",
        }


class KillSwitch:
    """Apply PortfolioRisk snapshots to PaperBroker.entries_blocked.

    Hard kill (DD peak/start or rolling Sharpe breach) is **sticky** for the run:
    new entries stay blocked; exits/stops may continue.
    """

    def __init__(
        self,
        risk: PortfolioRisk,
        broker: PaperBroker,
        *,
        ledger: Optional[PaperLedger] = None,
    ):
        self.risk = risk
        self.broker = broker
        self.ledger = ledger
        self.state = KillSwitchState()

    def evaluate(
        self,
        *,
        equity: Optional[float] = None,
        cash: Optional[float] = None,
        gross_exposure: Optional[float] = None,
        update_history: bool = True,
        ts: Optional[datetime] = None,
    ) -> RiskSnapshot:
        eq = float(equity if equity is not None else self.broker.state.equity())
        ca = float(cash if cash is not None else self.broker.state.cash)
        ge = float(
            gross_exposure
            if gross_exposure is not None
            else self.broker.state.gross_exposure()
        )
        if update_history:
            self.risk.update_equity(eq)
            if eq > self.broker.state.peak_equity:
                self.broker.state.peak_equity = eq

        snap = self.risk.snapshot(equity=eq, cash=ca, gross_exposure=ge)
        self._apply(snap, ts=ts)
        return snap

    def _apply(self, snap: RiskSnapshot, *, ts: Optional[datetime] = None) -> None:
        self.state.last_reasons = list(snap.reasons)

        # Sticky hard kill: once set, remains for the run
        if self.state.hard_kill:
            self.state.entries_blocked = True
            if not self.broker.state.entries_blocked:
                self.broker.set_entries_blocked(
                    True, reason="hard_kill_sticky", emit_event=False
                )
            return

        if snap.hard_kill or snap.block_new_entries:
            was = self.state.entries_blocked
            self.state.entries_blocked = True
            if snap.hard_kill:
                self.state.hard_kill = True
            self.broker.set_entries_blocked(
                True, reason=";".join(snap.reasons) or "kill", emit_event=False
            )
            if not was:
                self.state.trip_count += 1
                self.state.last_trip_ts = (ts or utc_now()).isoformat()
                if self.ledger is not None:
                    self.ledger.append_event(
                        EventType.KILL_SWITCH,
                        {
                            "action": "trip",
                            "hard_kill": bool(snap.hard_kill),
                            "reasons": list(snap.reasons),
                            "dd_from_peak": snap.dd_from_peak,
                            "dd_from_start": snap.dd_from_start,
                            "rolling_sharpe_20d": snap.rolling_sharpe_20d,
                            "trip_count": self.state.trip_count,
                            "capital_label": "VIRTUAL",
                        },
                        ts=ts,
                    )
        else:
            if self.state.entries_blocked and not self.state.hard_kill:
                self.state.entries_blocked = False
                self.broker.set_entries_blocked(
                    False, reason="risk_clear", emit_event=False
                )
                if self.ledger is not None:
                    self.ledger.append_event(
                        EventType.RISK_BLOCK,
                        {"action": "clear", "reasons": []},
                        ts=ts,
                    )

    def force_reset(self, *, reason: str = "manual_reset") -> None:
        """Operator override (still paper-only)."""
        self.state = KillSwitchState()
        self.broker.set_entries_blocked(False, reason=reason, emit_event=False)
        if self.ledger is not None:
            self.ledger.append_event(
                EventType.RISK_BLOCK,
                {"action": "manual_reset", "reason": reason},
            )

    def size_scale(self) -> float:
        eq = self.broker.state.equity()
        peak = max(self.risk.peak_equity, eq, self.risk.capital0)
        dd = eq / peak - 1.0 if peak > 0 else 0.0
        return self.risk.soft_size_scale(dd)
