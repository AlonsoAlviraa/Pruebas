"""Paper broker: cash/positions, submit/cancel/fill — virtual capital only."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from paper_live.freeze import CostModel, assert_paper_only
from paper_live.ledger import EventType, PaperLedger
from paper_live.ledger.events import new_order_id, utc_now
from paper_live.oms.fill_model import FillModel, FillQuote, SimulatedFill
from paper_live.oms.order_types import (
    OrderSide,
    OrderStatus,
    OrderType,
    PaperOrder,
    RejectReason,
)


@dataclass
class PortfolioState:
    """Mark-to-market portfolio (virtual)."""

    cash: float
    capital0: float
    positions: Dict[str, float] = field(default_factory=dict)  # ticker -> shares
    avg_px: Dict[str, float] = field(default_factory=dict)
    marks: Dict[str, float] = field(default_factory=dict)  # last mid
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0
    peak_equity: float = 0.0
    entries_blocked: bool = False  # kill switch

    def equity(self) -> float:
        mv = 0.0
        for t, q in self.positions.items():
            px = self.marks.get(t) or self.avg_px.get(t) or 0.0
            mv += float(q) * float(px)
        return float(self.cash) + mv

    def gross_exposure(self) -> float:
        s = 0.0
        for t, q in self.positions.items():
            px = self.marks.get(t) or self.avg_px.get(t) or 0.0
            s += abs(float(q) * float(px))
        return s

    def dd_from_peak(self) -> float:
        eq = self.equity()
        if self.peak_equity <= 0:
            return 0.0
        return float(eq / self.peak_equity - 1.0)

    def n_positions(self) -> int:
        return sum(1 for q in self.positions.values() if abs(q) > 1e-12)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "capital0": self.capital0,
            "equity": self.equity(),
            "gross_exposure": self.gross_exposure(),
            "realized_pnl": self.realized_pnl,
            "total_commission": self.total_commission,
            "total_fees": self.total_fees,
            "total_slippage_cost": self.total_slippage_cost,
            "peak_equity": self.peak_equity,
            "dd_from_peak": self.dd_from_peak(),
            "n_positions": self.n_positions(),
            "entries_blocked": self.entries_blocked,
            "capital_label": "VIRTUAL",
            "positions": dict(self.positions),
            "avg_px": dict(self.avg_px),
        }


class PaperBroker:
    """Simulated broker wired to CostModel + PaperLedger.

    Never places real orders. mode is always paper.
    """

    def __init__(
        self,
        cost: CostModel,
        *,
        capital0: float = 100_000.0,
        ledger: Optional[PaperLedger] = None,
        long_only: bool = True,
        fill_model: Optional[FillModel] = None,
        n_clips: int = 3,
    ):
        assert_paper_only(require_env=False)
        self.cost = cost
        self.long_only = bool(long_only)
        self.ledger = ledger
        self.fill_model = fill_model or FillModel(cost, n_clips=n_clips, long_only=long_only)
        self.state = PortfolioState(
            cash=float(capital0),
            capital0=float(capital0),
            peak_equity=float(capital0),
        )
        self.open_orders: Dict[str, PaperOrder] = {}
        self._mode = "paper"

    # --- risk gates used by OMS ---

    def set_entries_blocked(
        self,
        blocked: bool,
        *,
        reason: str = "kill_switch",
        emit_event: bool = True,
    ) -> None:
        self.state.entries_blocked = bool(blocked)
        if emit_event and self.ledger is not None:
            self.ledger.append_event(
                EventType.KILL_SWITCH if blocked else EventType.RISK_BLOCK,
                {"entries_blocked": blocked, "reason": reason, "mode": "paper"},
            )

    def update_marks(self, marks: Mapping[str, float]) -> None:
        for t, px in marks.items():
            if px and float(px) > 0:
                self.state.marks[t.upper()] = float(px)
        eq = self.state.equity()
        if eq > self.state.peak_equity:
            self.state.peak_equity = eq

    # --- order lifecycle ---

    def submit(
        self,
        ticker: str,
        side: Union[OrderSide, str],
        qty: float,
        *,
        order_type: Union[OrderType, str] = OrderType.MARKET,
        limit_px: Optional[float] = None,
        is_stop: bool = False,
        ts: Optional[datetime] = None,
        meta: Optional[Mapping[str, Any]] = None,
        order_id: Optional[str] = None,
    ) -> PaperOrder:
        """Accept or reject an order into the paper book (not yet filled)."""
        assert_paper_only(require_env=False)
        side_e = OrderSide(side.value if isinstance(side, OrderSide) else str(side).lower())
        type_e = OrderType(
            order_type.value if isinstance(order_type, OrderType) else str(order_type).lower()
        )
        t = ticker.upper()
        q = float(int(qty))  # whole shares
        oid = order_id or new_order_id()
        order = PaperOrder(
            order_id=oid,
            ticker=t,
            side=side_e,
            qty=q,
            order_type=type_e,
            limit_px=limit_px,
            status=OrderStatus.PENDING,
            is_stop=is_stop,
            submitted_at=ts or utc_now(),
            meta=dict(meta or {}),
        )

        reject: Optional[RejectReason] = None
        detail = ""
        if q <= 0:
            reject, detail = RejectReason.INVALID_QTY, "qty <= 0"
        elif self.state.entries_blocked and side_e == OrderSide.BUY:
            reject, detail = RejectReason.KILL_SWITCH, "entries blocked"
        elif self.long_only and side_e == OrderSide.SELL:
            held = float(self.state.positions.get(t, 0.0))
            if q > held + 1e-9:
                reject, detail = RejectReason.SHORT_NOT_ALLOWED, f"held={held} sell={q}"

        if reject is not None:
            order.status = OrderStatus.REJECTED
            order.reason = f"{reject.value}:{detail}"
            if self.ledger is not None:
                self.ledger.record_order(
                    ticker=t,
                    side=side_e.value,
                    qty=q,
                    order_type=type_e.value,
                    limit_px=limit_px,
                    status=OrderStatus.REJECTED.value,
                    reason=order.reason,
                    order_id=oid,
                    ts=ts,
                    meta=order.meta,
                )
                self.ledger.update_order_status(
                    oid,
                    OrderStatus.REJECTED.value,
                    reason=order.reason,
                    event=EventType.ORDER_REJECT,
                    ts=ts,
                )
            return order

        order.status = OrderStatus.SUBMITTED
        self.open_orders[oid] = order
        if self.ledger is not None:
            self.ledger.record_order(
                ticker=t,
                side=side_e.value,
                qty=q,
                order_type=type_e.value,
                limit_px=limit_px,
                status=OrderStatus.SUBMITTED.value,
                reason=None,
                order_id=oid,
                ts=ts,
                meta={**order.meta, "is_stop": is_stop},
            )
            self.ledger.update_order_status(
                oid,
                OrderStatus.ACK.value,
                event=EventType.ORDER_ACK,
                ts=ts,
            )
        order.status = OrderStatus.ACK
        return order

    def cancel(self, order_id: str, *, reason: str = "user_cancel", ts: Optional[datetime] = None) -> bool:
        order = self.open_orders.get(order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return False
        order.status = OrderStatus.CANCELLED
        order.reason = reason
        del self.open_orders[order_id]
        if self.ledger is not None:
            self.ledger.update_order_status(
                order_id,
                OrderStatus.CANCELLED.value,
                reason=reason,
                event=EventType.ORDER_REJECT,  # reuse reject channel for cancel audit
                ts=ts,
                meta={"cancelled": True},
            )
        return True

    def execute(
        self,
        order_id: str,
        quote: FillQuote,
        *,
        ts: Optional[datetime] = None,
        use_twap: bool = False,
        twap_quotes: Optional[Sequence[FillQuote]] = None,
    ) -> List[SimulatedFill]:
        """Fill an open order against quote(s); update cash/positions + ledger."""
        order = self.open_orders.get(order_id)
        if order is None:
            return []

        fills_out: List[SimulatedFill] = []
        if use_twap:
            sim_list = self.fill_model.simulate_twap(
                order, list(twap_quotes) if twap_quotes else [quote]
            )
        else:
            sim_list = [self.fill_model.simulate_fill(order, quote)]

        for sim in sim_list:
            if not sim.ok:
                if order.filled_qty <= 0:
                    order.status = OrderStatus.REJECTED
                    order.reason = (
                        f"{sim.reject.value}:{sim.reject_detail}" if sim.reject else sim.reject_detail
                    )
                    if self.ledger is not None:
                        self.ledger.update_order_status(
                            order_id,
                            OrderStatus.REJECTED.value,
                            reason=order.reason,
                            event=EventType.ORDER_REJECT,
                            ts=ts,
                        )
                    self.open_orders.pop(order_id, None)
                fills_out.append(sim)
                break

            applied = self._apply_fill(order, sim, ts=ts)
            if applied is None:
                fills_out.append(sim)  # still report attempt
                break
            fills_out.append(applied)

        if order.order_id in self.open_orders and order.remaining_qty <= 0:
            order.status = OrderStatus.FILLED
            del self.open_orders[order_id]
        elif order.order_id in self.open_orders and order.filled_qty > 0:
            order.status = OrderStatus.PARTIAL

        return fills_out

    def submit_and_execute(
        self,
        ticker: str,
        side: Union[OrderSide, str],
        qty: float,
        quote: FillQuote,
        *,
        order_type: Union[OrderType, str] = OrderType.MARKET,
        limit_px: Optional[float] = None,
        is_stop: bool = False,
        ts: Optional[datetime] = None,
        meta: Optional[Mapping[str, Any]] = None,
        use_twap: bool = False,
    ) -> tuple[PaperOrder, List[SimulatedFill]]:
        """Convenience: submit then immediately execute (marketable paper path)."""
        order = self.submit(
            ticker,
            side,
            qty,
            order_type=order_type,
            limit_px=limit_px,
            is_stop=is_stop,
            ts=ts,
            meta=meta,
        )
        if order.status == OrderStatus.REJECTED:
            return order, []
        fills = self.execute(order.order_id, quote, ts=ts, use_twap=use_twap)
        return order, fills

    def _apply_fill(
        self,
        order: PaperOrder,
        sim: SimulatedFill,
        *,
        ts: Optional[datetime] = None,
    ) -> Optional[SimulatedFill]:
        """Apply cash/position effects; may reject for cash/shares."""
        t = order.ticker
        q = float(sim.qty)
        px = float(sim.price)

        if order.side == OrderSide.BUY:
            need = q * px + sim.commission + sim.fees
            if need > self.state.cash + 1e-9:
                # try downsize to affordable whole shares
                if px <= 0:
                    return self._reject_open(order, RejectReason.INSUFFICIENT_CASH, "no price", ts=ts)
                afford = self._max_affordable_buy_qty(sim.mid if sim.mid > 0 else px)
                if afford <= 0:
                    return self._reject_open(
                        order, RejectReason.INSUFFICIENT_CASH, f"need={need:.2f}", ts=ts
                    )
                if afford < q:
                    sim = self.fill_model.simulate_fill(
                        order,
                        FillQuote(mid=sim.mid, adv_shares=None),
                        qty=float(afford),
                    )
                    if not sim.ok:
                        return self._reject_open(
                            order, RejectReason.INSUFFICIENT_CASH, "resize failed", ts=ts
                        )
                    q = float(sim.qty)
                    px = float(sim.price)
                    need = q * px + sim.commission + sim.fees
                    if need > self.state.cash + 1e-9 or q <= 0:
                        return self._reject_open(
                            order, RejectReason.INSUFFICIENT_CASH, "cannot afford", ts=ts
                        )

            self.state.cash -= need
            prev_q = float(self.state.positions.get(t, 0.0))
            prev_avg = float(self.state.avg_px.get(t, 0.0))
            new_q = prev_q + q
            if new_q > 0:
                self.state.avg_px[t] = (prev_avg * prev_q + px * q) / new_q
            self.state.positions[t] = new_q
            self.state.marks[t] = sim.mid
            self._book_costs(sim)
            self._ledger_fill(order, sim, ts=ts, order_status=self._status_after(order, q))
            order.filled_qty += q
            order.avg_fill_px = self._avg_fill(order, q, px)
            if self.ledger is not None:
                self.ledger.upsert_position(
                    ticker=t,
                    qty=new_q,
                    avg_px=self.state.avg_px[t],
                    event=EventType.POSITION_OPENED if prev_q <= 0 else EventType.POSITION_UPDATED,
                )
            return sim

        # SELL
        held = float(self.state.positions.get(t, 0.0))
        if q > held + 1e-9:
            if self.long_only:
                if held <= 0:
                    return self._reject_open(
                        order, RejectReason.INSUFFICIENT_SHARES, f"held={held}", ts=ts
                    )
                sim = self.fill_model.simulate_fill(
                    order, FillQuote(mid=sim.mid), qty=float(int(held))
                )
                if not sim.ok:
                    return self._reject_open(
                        order, RejectReason.INSUFFICIENT_SHARES, "resize failed", ts=ts
                    )
                q = float(sim.qty)
                px = float(sim.price)

        proceeds = q * px
        self.state.cash += proceeds - sim.commission - sim.fees
        prev_q = float(self.state.positions.get(t, 0.0))
        prev_avg = float(self.state.avg_px.get(t, 0.0))
        # realized pnl on sold shares
        self.state.realized_pnl += (px - prev_avg) * q - sim.commission - sim.fees
        new_q = prev_q - q
        if new_q <= 1e-12:
            self.state.positions.pop(t, None)
            self.state.avg_px.pop(t, None)
            if self.ledger is not None:
                self.ledger.close_position(t, meta={"reason": "sold"})
        else:
            self.state.positions[t] = new_q
            if self.ledger is not None:
                self.ledger.upsert_position(
                    ticker=t,
                    qty=new_q,
                    avg_px=prev_avg,
                    event=EventType.POSITION_UPDATED,
                )
        self.state.marks[t] = sim.mid
        self._book_costs(sim)
        self._ledger_fill(order, sim, ts=ts, order_status=self._status_after(order, q))
        order.filled_qty += q
        order.avg_fill_px = self._avg_fill(order, q, px)
        return sim

    def _status_after(self, order: PaperOrder, fill_qty: float) -> str:
        if order.filled_qty + fill_qty + 1e-9 >= order.qty:
            return OrderStatus.FILLED.value
        return OrderStatus.PARTIAL.value

    def _avg_fill(self, order: PaperOrder, q: float, px: float) -> float:
        prev_f = order.filled_qty
        if prev_f <= 0:
            return px
        return (order.avg_fill_px * prev_f + px * q) / (prev_f + q)

    def _book_costs(self, sim: SimulatedFill) -> None:
        self.state.total_commission += sim.commission
        self.state.total_fees += sim.fees
        self.state.total_slippage_cost += sim.slippage_cost
        eq = self.state.equity()
        if eq > self.state.peak_equity:
            self.state.peak_equity = eq

    def _ledger_fill(
        self,
        order: PaperOrder,
        sim: SimulatedFill,
        *,
        ts: Optional[datetime],
        order_status: str,
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.record_fill(
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side.value,
            qty=sim.qty,
            price=sim.price,
            commission=sim.commission,
            fees=sim.fees,
            slippage_bps=sim.slippage_bps,
            liquidity=sim.liquidity,
            ts=ts,
            order_status=order_status,
            meta={
                "mid": sim.mid,
                "slippage_cost": sim.slippage_cost,
                "net_cash_delta": sim.net_cash_delta,
                "participation_pct": sim.participation_pct,
                "gross_notional": sim.gross_notional,
                "mode": "paper",
            },
        )

    def _reject_open(
        self,
        order: PaperOrder,
        reason: RejectReason,
        detail: str,
        *,
        ts: Optional[datetime] = None,
    ) -> SimulatedFill:
        order.status = OrderStatus.REJECTED
        order.reason = f"{reason.value}:{detail}"
        self.open_orders.pop(order.order_id, None)
        if self.ledger is not None:
            self.ledger.update_order_status(
                order.order_id,
                OrderStatus.REJECTED.value,
                reason=order.reason,
                event=EventType.ORDER_REJECT,
                ts=ts,
            )
        return SimulatedFill(
            qty=0,
            price=0.0,
            mid=0.0,
            commission=0.0,
            fees=0.0,
            slippage_bps=0.0,
            slippage_cost=0.0,
            gross_notional=0.0,
            net_cash_delta=0.0,
            participation_pct=0.0,
            reject=reason,
            reject_detail=detail,
        )

    def _max_affordable_buy_qty(self, price: float) -> int:
        """Largest whole-share buy such that price*q + commission + fees <= cash."""
        if price <= 0 or self.state.cash <= 0:
            return 0
        # upper bound ignore min commission
        rough = int(self.state.cash / price)
        for q in range(rough, 0, -1):
            comm = self.cost.estimate_commission(q, price)
            need = q * price + comm
            if need <= self.state.cash + 1e-9:
                return q
        return 0

    def record_nav(self, day: str) -> Dict[str, Any]:
        """Persist daily NAV + costs to ledger if attached."""
        st = self.state
        eq = st.equity()
        if self.ledger is not None:
            self.ledger.record_nav_daily(
                day,
                equity=eq,
                cash=st.cash,
                gross_exposure=st.gross_exposure(),
                dd_from_peak=st.dd_from_peak(),
                n_positions=st.n_positions(),
                peak_equity=st.peak_equity,
            )
            self.ledger.record_costs_daily(
                day,
                commission=st.total_commission,
                fees=st.total_fees,
                slippage_est=st.total_slippage_cost,
                turnover=st.gross_exposure(),
            )
        return st.to_dict()
