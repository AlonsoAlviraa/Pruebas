"""Orchestrate daily replay: signal D-1 → confirm/entry D open → exits (LIV-03/04)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.freeze import PaperFreeze, assert_paper_only, load_freeze
from paper_live.ledger import EventType, PaperLedger
from paper_live.oms.fill_model import FillQuote
from paper_live.oms.order_types import OrderSide
from paper_live.oms.paper_broker import PaperBroker
from paper_live.signals.daily_pipeline import DailySignalPipeline, EntryCandidate, SignalBatch
from paper_live.signals.entry_confirm import confirm_entry
from paper_live.risk.kill_switch import KillSwitch
from paper_live.risk.portfolio_risk import PortfolioRisk, RiskConfig


@dataclass
class ManagedPosition:
    ticker: str
    qty: float
    entry_px: float
    entry_day: date
    stop: float
    hard_stop: float
    bars_held: int = 0
    atr: float = 0.0


@dataclass
class ReplaySessionResult:
    days_run: int
    n_signals: int
    n_entries: int
    n_entry_rejects: int
    n_exits: int
    final_equity: float
    final_cash: float
    total_commission: float
    total_fees: float
    run_id: Optional[str] = None
    daily_nav: List[Dict[str, Any]] = field(default_factory=list)
    kill_trips: int = 0
    hard_kill: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "days_run": self.days_run,
            "n_signals": self.n_signals,
            "n_entries": self.n_entries,
            "n_entry_rejects": self.n_entry_rejects,
            "n_exits": self.n_exits,
            "final_equity": self.final_equity,
            "final_cash": self.final_cash,
            "total_commission": self.total_commission,
            "total_fees": self.total_fees,
            "run_id": self.run_id,
            "kill_trips": self.kill_trips,
            "hard_kill": self.hard_kill,
            "capital_label": "VIRTUAL",
            "mode": "paper",
        }


class ReplaySession:
    """Daily paper replay loop (virtual capital only).

    Flow per calendar session ``D``:
    1. Open: confirm + enter candidates generated on ``prev(D)`` close.
    2. Intraday (daily proxy): stop/hard-stop vs bar low; update marks.
    3. Close: time-stop; generate candidates for next session; record NAV.
    """

    def __init__(
        self,
        feed: DailyReplayFeed,
        freeze: Optional[PaperFreeze] = None,
        *,
        ledger: Optional[PaperLedger] = None,
        broker: Optional[PaperBroker] = None,
        pipeline: Optional[DailySignalPipeline] = None,
        max_positions: Optional[int] = None,
        max_horizon: Optional[int] = None,
        k_atr: Optional[float] = None,
        hard_stop_pct: Optional[float] = None,
        min_alloc_pct: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        max_entries_per_day: Optional[int] = None,
        enable_risk: bool = True,
    ):
        assert_paper_only(require_env=False)
        self.feed = feed
        self.freeze = freeze or load_freeze()
        kn = self.freeze.strategy.knobs
        rp = self.freeze.strategy.risk_paper
        self.max_positions = int(max_positions if max_positions is not None else kn.get("max_positions", 10))
        self.max_horizon = int(max_horizon if max_horizon is not None else kn.get("max_horizon", 20))
        self.k_atr = float(k_atr if k_atr is not None else kn.get("k_atr", 3.0))
        self.hard_stop_pct = float(
            hard_stop_pct if hard_stop_pct is not None else kn.get("hard_stop_pct", 0.07)
        )
        self.min_alloc_pct = float(
            min_alloc_pct if min_alloc_pct is not None else kn.get("min_alloc_pct", 0.015)
        )
        self.max_position_pct = float(
            max_position_pct if max_position_pct is not None else kn.get("max_position_pct", 0.25)
        )
        self.max_entries_per_day = int(
            max_entries_per_day
            if max_entries_per_day is not None
            else rp.get("max_daily_new_entries", 5)
        )

        self.ledger = ledger
        self.broker = broker or PaperBroker(
            self.freeze.cost,
            capital0=self.freeze.strategy.capital0,
            ledger=ledger,
            long_only=self.freeze.strategy.long_only,
        )
        univ = [t for t in feed.tickers if t not in ("SPY",)]
        self.pipeline = pipeline or DailySignalPipeline(
            feed,
            universe=univ,
            min_price=float(self.freeze.universe.min_price),
            require_regime=bool(kn.get("require_regime", True)),
            regime_symbol=(
                (kn.get("preferred_index") or ["QQQ", "SPY"])[0]
                if isinstance(kn.get("preferred_index"), list)
                else "QQQ"
            ),
        )
        self.managed: Dict[str, ManagedPosition] = {}
        self.pending: SignalBatch = SignalBatch(
            signal_date=date(1970, 1, 1), regime_on=False, candidates=[]
        )
        self.stats = {
            "n_signals": 0,
            "n_entries": 0,
            "n_entry_rejects": 0,
            "n_exits": 0,
        }
        self.daily_nav: List[Dict[str, Any]] = []
        self.risk: Optional[PortfolioRisk] = None
        self.kill: Optional[KillSwitch] = None
        if enable_risk:
            self.attach_risk()

    def attach_risk(
        self,
        risk: Optional[PortfolioRisk] = None,
        kill: Optional[KillSwitch] = None,
    ) -> KillSwitch:
        """Attach or replace portfolio risk + kill switch."""
        if risk is None:
            cfg = RiskConfig.from_risk_paper(
                self.freeze.strategy.risk_paper,
                max_leverage=self.freeze.strategy.max_leverage,
            )
            risk = PortfolioRisk(cfg, capital0=self.freeze.strategy.capital0)
            risk.peak_equity = max(
                self.broker.state.peak_equity, self.freeze.strategy.capital0
            )
        if kill is None:
            kill = KillSwitch(risk, self.broker, ledger=self.ledger)
        self.risk = risk
        self.kill = kill
        return kill

    def _ts(self, day: date, hh: int = 9, mm: int = 45) -> datetime:
        return datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc)

    def _adv_shares(self, ticker: str, through: date) -> Optional[float]:
        hist = self.feed.history(ticker, through=through)
        if hist.empty or "volume" not in hist.columns:
            return None
        v = hist["volume"].tail(20)
        if v.empty:
            return None
        return float(v.mean())

    def _size_shares(self, entry_px: float) -> int:
        eq = self.broker.state.equity()
        if entry_px <= 0 or eq <= 0:
            return 0
        scale = 1.0
        if self.kill is not None:
            scale = float(self.kill.size_scale())
        alloc = max(self.min_alloc_pct, 0.0) * eq
        # target between min_alloc and max_position, then soft de-risk scale
        target = min(alloc * 2.0, self.max_position_pct * eq)
        target = max(target, self.min_alloc_pct * eq) * scale
        # ticker capital cap from risk_paper
        cap_pct = float(self.freeze.strategy.risk_paper.get("ticker_max_capital_pct", 0.12))
        target = min(target, cap_pct * eq)
        shares = int(target / entry_px)
        if shares <= 0:
            return 0
        # enforce min_alloc after costs roughly (skip if soft-scaled below floor)
        if scale >= 0.99 and shares * entry_px < self.min_alloc_pct * eq * 0.95:
            return 0
        return shares

    def _process_entries(self, day: date) -> None:
        if not self.pending.candidates:
            return
        if self.broker.state.entries_blocked:
            return
        slots = self.max_positions - self.broker.state.n_positions()
        if slots <= 0:
            return

        ranked = self.pending.top(self.max_positions * 2)
        entries = 0
        for cand in ranked:
            if entries >= self.max_entries_per_day or slots <= 0:
                break
            if cand.ticker in self.managed:
                continue
            bar = self.feed.bar(cand.ticker, day)
            conf = confirm_entry(
                cand,
                bar,
                min_price=float(self.freeze.universe.min_price),
            )
            if self.ledger is not None:
                self.ledger.record_decision(
                    ticker=cand.ticker,
                    action="enter" if conf.ok else "reject",
                    p_buy=cand.p_buy,
                    score=cand.score,
                    filters={
                        "confirm": conf.to_dict(),
                        "signal": cand.to_dict(),
                    },
                    ts=self._ts(day, 9, 45),
                )
            if not conf.ok:
                self.stats["n_entry_rejects"] += 1
                continue

            shares = self._size_shares(conf.entry_px_ref)
            if shares <= 0:
                self.stats["n_entry_rejects"] += 1
                if self.ledger is not None:
                    self.ledger.record_decision(
                        ticker=cand.ticker,
                        action="reject",
                        p_buy=cand.p_buy,
                        score=cand.score,
                        filters={"reason": "size_zero", "confirm": conf.to_dict()},
                        ts=self._ts(day, 9, 45),
                    )
                continue

            adv = self._adv_shares(cand.ticker, day)
            quote = FillQuote(
                mid=conf.entry_px_ref,
                adv_shares=adv,
                last=conf.entry_px_ref,
            )
            order, fills = self.broker.submit_and_execute(
                cand.ticker,
                OrderSide.BUY,
                shares,
                quote,
                ts=self._ts(day, 9, 45),
                meta={"signal_date": cand.signal_date.isoformat(), "score": cand.score},
            )
            if not fills or not fills[0].ok:
                self.stats["n_entry_rejects"] += 1
                continue

            fill = fills[0]
            stop = fill.price * (1.0 - self.hard_stop_pct)
            # chandelier-ish: entry - k*atr
            if cand.atr > 0:
                stop = max(stop, fill.price - self.k_atr * cand.atr)
            hard = fill.price * (1.0 - self.hard_stop_pct)
            self.managed[cand.ticker] = ManagedPosition(
                ticker=cand.ticker,
                qty=fill.qty,
                entry_px=fill.price,
                entry_day=day,
                stop=float(stop),
                hard_stop=float(hard),
                bars_held=0,
                atr=cand.atr,
            )
            if self.ledger is not None:
                self.ledger.upsert_position(
                    ticker=cand.ticker,
                    qty=fill.qty,
                    avg_px=fill.price,
                    stop=stop,
                    hard_stop=hard,
                    opened_at=self._ts(day, 9, 45),
                    bars_held=0,
                )
            self.stats["n_entries"] += 1
            entries += 1
            slots -= 1

    def _exit_ticker(
        self,
        day: date,
        pos: ManagedPosition,
        px: float,
        *,
        is_stop: bool,
        reason: str,
    ) -> None:
        quote = FillQuote(mid=px, last=px, adv_shares=self._adv_shares(pos.ticker, day))
        order, fills = self.broker.submit_and_execute(
            pos.ticker,
            OrderSide.SELL,
            pos.qty,
            quote,
            is_stop=is_stop,
            ts=self._ts(day, 15, 45),
            meta={"exit_reason": reason},
        )
        if fills and fills[0].ok:
            self.managed.pop(pos.ticker, None)
            self.stats["n_exits"] += 1
            if self.ledger is not None:
                self.ledger.append_event(
                    EventType.POSITION_CLOSED,
                    {
                        "ticker": pos.ticker,
                        "exit_reason": reason,
                        "day": day.isoformat(),
                        "exit_px": fills[0].price,
                    },
                    ts=self._ts(day, 15, 45),
                )

    def _process_exits(self, day: date) -> None:
        for t, pos in list(self.managed.items()):
            bar = self.feed.bar(t, day)
            if bar is None:
                continue
            pos.bars_held += 1
            # stop vs low of day
            if bar.low <= pos.hard_stop or bar.low <= pos.stop:
                stop_px = min(pos.hard_stop, pos.stop)
                # fill at worse of stop and open if gapped through
                px = min(float(bar.open), stop_px) if bar.open < stop_px else stop_px
                self._exit_ticker(day, pos, px, is_stop=True, reason="stop")
                continue
            # trail stop up using high
            if pos.atr > 0:
                trail = float(bar.high) - self.k_atr * pos.atr
                pos.stop = max(pos.stop, trail)
            # time stop at close
            if pos.bars_held >= self.max_horizon:
                self._exit_ticker(day, pos, float(bar.close), is_stop=False, reason="time_stop")
                continue
            # mark
            self.broker.update_marks({t: float(bar.close)})
            if self.ledger is not None:
                self.ledger.upsert_position(
                    ticker=t,
                    qty=pos.qty,
                    avg_px=pos.entry_px,
                    stop=pos.stop,
                    hard_stop=pos.hard_stop,
                    bars_held=pos.bars_held,
                )

    def _generate_signals(self, day: date) -> None:
        batch = self.pipeline.generate(day)
        self.pending = batch
        self.stats["n_signals"] += 1
        if self.ledger is not None:
            self.ledger.append_event(
                EventType.SIGNAL_COMPUTED,
                {
                    "signal_date": day.isoformat(),
                    "regime_on": batch.regime_on,
                    "n_candidates": len(batch.candidates),
                    "n_scanned": batch.n_scanned,
                    "n_rejected": batch.n_rejected,
                    "top": [c.to_dict() for c in batch.top(5)],
                },
                ts=self._ts(day, 16, 15),
            )
            for c in batch.candidates:
                self.ledger.record_decision(
                    ticker=c.ticker,
                    action="candidate",
                    p_buy=c.p_buy,
                    score=c.score,
                    filters={"signal_date": day.isoformat(), "reason": c.reason},
                    ts=self._ts(day, 16, 15),
                )

    def run(
        self,
        start: Union[str, date],
        end: Union[str, date],
        *,
        warmup_signal_days: int = 1,
    ) -> ReplaySessionResult:
        days = self.feed.session_days(start, end)
        if not days:
            return ReplaySessionResult(
                days_run=0,
                n_signals=0,
                n_entries=0,
                n_entry_rejects=0,
                n_exits=0,
                final_equity=self.broker.state.equity(),
                final_cash=self.broker.state.cash,
                total_commission=self.broker.state.total_commission,
                total_fees=self.broker.state.total_fees,
                run_id=self.ledger.run_id if self.ledger else None,
                kill_trips=int(self.kill.state.trip_count) if self.kill else 0,
                hard_kill=bool(self.kill.state.hard_kill) if self.kill else False,
            )

        # Seed pending signals from previous session before start if possible
        first = days[0]
        prev = self.feed.prev_session(first)
        if prev is not None:
            self._generate_signals(prev)

        for i, day in enumerate(days):
            if self.ledger is not None:
                self.ledger.append_event(
                    EventType.SESSION_OPEN,
                    {"day": day.isoformat()},
                    ts=self._ts(day, 9, 30),
                )

            self._process_entries(day)
            self._process_exits(day)

            # EOD marks for open positions
            marks = {}
            for t in list(self.managed.keys()):
                b = self.feed.bar(t, day)
                if b:
                    marks[t] = float(b.close)
            if marks:
                self.broker.update_marks(marks)

            nav = self.broker.record_nav(day.isoformat())
            risk_snap = None
            if self.kill is not None:
                risk_snap = self.kill.evaluate(
                    equity=self.broker.state.equity(),
                    cash=self.broker.state.cash,
                    gross_exposure=self.broker.state.gross_exposure(),
                    update_history=True,
                    ts=self._ts(day, 16, 10),
                )
                nav = {**nav, "risk": risk_snap.to_dict(), "kill": self.kill.state.to_dict()}
            self.daily_nav.append({"date": day.isoformat(), **nav})

            # Signals for next session (post-close)
            self._generate_signals(day)

            if self.ledger is not None:
                self.ledger.append_event(
                    EventType.SESSION_CLOSE,
                    {
                        "day": day.isoformat(),
                        "equity": self.broker.state.equity(),
                        "n_positions": self.broker.state.n_positions(),
                        "entries_blocked": self.broker.state.entries_blocked,
                        "hard_kill": bool(self.kill.state.hard_kill) if self.kill else False,
                    },
                    ts=self._ts(day, 16, 15),
                )

        kill_trips = int(self.kill.state.trip_count) if self.kill else 0
        hard_kill = bool(self.kill.state.hard_kill) if self.kill else False
        return ReplaySessionResult(
            days_run=len(days),
            n_signals=int(self.stats["n_signals"]),
            n_entries=int(self.stats["n_entries"]),
            n_entry_rejects=int(self.stats["n_entry_rejects"]),
            n_exits=int(self.stats["n_exits"]),
            final_equity=self.broker.state.equity(),
            final_cash=self.broker.state.cash,
            total_commission=self.broker.state.total_commission,
            total_fees=self.broker.state.total_fees,
            run_id=self.ledger.run_id if self.ledger else None,
            daily_nav=list(self.daily_nav),
            kill_trips=kill_trips,
            hard_kill=hard_kill,
        )
