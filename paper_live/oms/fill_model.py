"""Fill model: slippage, commission, fees, ADV participation, partial clips."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from paper_live.freeze import CostModel
from paper_live.oms.order_types import OrderSide, OrderType, PaperOrder, RejectReason


@dataclass(frozen=True)
class FillQuote:
    """Market snapshot used for a paper fill attempt."""

    mid: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    adv_shares: Optional[float] = None  # average daily volume (shares)
    halted: bool = False
    last: Optional[float] = None


@dataclass(frozen=True)
class SimulatedFill:
    """One simulated fill clip (may be partial)."""

    qty: float
    price: float
    mid: float
    commission: float
    fees: float
    slippage_bps: float
    slippage_cost: float
    gross_notional: float
    net_cash_delta: float  # negative = cash out (buy), positive = cash in (sell)
    participation_pct: float
    liquidity: str = "paper_sim"
    reject: Optional[RejectReason] = None
    reject_detail: str = ""

    @property
    def ok(self) -> bool:
        return self.reject is None and self.qty > 0


class FillModel:
    """Cost-aware fill simulator (always virtual; no real broker)."""

    def __init__(
        self,
        cost: CostModel,
        *,
        n_clips: int = 3,
        long_only: bool = True,
    ):
        if n_clips < 1:
            raise ValueError("n_clips must be >= 1")
        self.cost = cost
        self.n_clips = int(n_clips)
        self.long_only = bool(long_only)

    def participation_pct(self, qty: float, adv_shares: Optional[float]) -> float:
        if adv_shares is None or adv_shares <= 0:
            return 0.0
        return float(qty) / float(adv_shares) * 100.0  # percent of ADV

    def max_qty_by_adv(self, adv_shares: Optional[float]) -> Optional[float]:
        """Max shares allowed by max_participation_rate (fraction of ADV)."""
        if adv_shares is None or adv_shares <= 0:
            return None
        return float(adv_shares) * float(self.cost.max_participation_rate)

    def _ref_mid(self, quote: FillQuote, side: OrderSide) -> Optional[float]:
        if quote.halted:
            return None
        if quote.mid and quote.mid > 0:
            mid = float(quote.mid)
        elif quote.last and quote.last > 0:
            mid = float(quote.last)
        elif side == OrderSide.BUY and quote.ask and quote.ask > 0:
            mid = float(quote.ask)
        elif side == OrderSide.SELL and quote.bid and quote.bid > 0:
            mid = float(quote.bid)
        else:
            return None
        # optional: tighten mid with bid/ask if present
        if (
            self.cost.spread.get("use_quote_if_available")
            and quote.bid
            and quote.ask
            and quote.bid > 0
            and quote.ask > quote.bid
        ):
            mid = 0.5 * (float(quote.bid) + float(quote.ask))
        return mid

    def _slippage_bps_used(
        self,
        side: OrderSide,
        *,
        is_stop: bool,
        participation_pct: float,
    ) -> float:
        bps = float(
            self.cost.slippage.get(
                "entry_bps" if side == OrderSide.BUY else "exit_bps", 5.0
            )
        )
        if is_stop:
            bps += float(self.cost.slippage.get("stop_extra_bps", 10.0))
        bps += float(self.cost.slippage.get("impact_bps_per_adv_pct", 0.0)) * max(
            0.0, participation_pct
        )
        bps += float(self.cost.spread.get("fallback_bps", 0.0)) / 2.0
        return bps

    def simulate_fill(
        self,
        order: PaperOrder,
        quote: FillQuote,
        *,
        qty: Optional[float] = None,
    ) -> SimulatedFill:
        """Simulate one fill for up to ``qty`` (default: full remaining)."""
        q = float(qty if qty is not None else order.remaining_qty)
        if q <= 0:
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
                reject=RejectReason.INVALID_QTY,
                reject_detail="qty <= 0",
            )

        mid = self._ref_mid(quote, order.side)
        if mid is None:
            reason = RejectReason.HALT if quote.halted else RejectReason.NO_QUOTE
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
                reject_detail="halted or missing quote",
            )

        if mid < float(self.cost.min_price):
            return SimulatedFill(
                qty=0,
                price=0.0,
                mid=mid,
                commission=0.0,
                fees=0.0,
                slippage_bps=0.0,
                slippage_cost=0.0,
                gross_notional=0.0,
                net_cash_delta=0.0,
                participation_pct=0.0,
                reject=RejectReason.MIN_PRICE,
                reject_detail=f"mid {mid} < min_price {self.cost.min_price}",
            )

        # ADV participation cap: truncate qty
        max_q = self.max_qty_by_adv(quote.adv_shares)
        if max_q is not None and max_q <= 0:
            return SimulatedFill(
                qty=0,
                price=0.0,
                mid=mid,
                commission=0.0,
                fees=0.0,
                slippage_bps=0.0,
                slippage_cost=0.0,
                gross_notional=0.0,
                net_cash_delta=0.0,
                participation_pct=0.0,
                reject=RejectReason.ADV_CAP,
                reject_detail="ADV cap is zero",
            )
        if max_q is not None and q > max_q:
            q = float(int(max_q))  # whole shares
            if q <= 0:
                return SimulatedFill(
                    qty=0,
                    price=0.0,
                    mid=mid,
                    commission=0.0,
                    fees=0.0,
                    slippage_bps=0.0,
                    slippage_cost=0.0,
                    gross_notional=0.0,
                    net_cash_delta=0.0,
                    participation_pct=0.0,
                    reject=RejectReason.ADV_CAP,
                    reject_detail="qty below one share after ADV cap",
                )

        # whole shares only
        q = float(int(q))
        if q <= 0:
            return SimulatedFill(
                qty=0,
                price=0.0,
                mid=mid,
                commission=0.0,
                fees=0.0,
                slippage_bps=0.0,
                slippage_cost=0.0,
                gross_notional=0.0,
                net_cash_delta=0.0,
                participation_pct=0.0,
                reject=RejectReason.INVALID_QTY,
                reject_detail="fractional only",
            )

        part = self.participation_pct(q, quote.adv_shares)
        is_stop = bool(order.is_stop or order.order_type == OrderType.STOP)
        fill_px = self.cost.slip_price(
            order.side.value,
            mid,
            is_stop=is_stop,
            participation_pct=part,
        )

        # Limit: buy only if fill <= limit; sell only if fill >= limit
        if order.order_type == OrderType.LIMIT and order.limit_px is not None:
            lim = float(order.limit_px)
            if order.side == OrderSide.BUY and fill_px > lim:
                # improve to limit if mid allows (conservative: no fill through limit)
                return SimulatedFill(
                    qty=0,
                    price=fill_px,
                    mid=mid,
                    commission=0.0,
                    fees=0.0,
                    slippage_bps=0.0,
                    slippage_cost=0.0,
                    gross_notional=0.0,
                    net_cash_delta=0.0,
                    participation_pct=part,
                    reject=RejectReason.OTHER,
                    reject_detail="limit not marketable",
                )
            if order.side == OrderSide.SELL and fill_px < lim:
                return SimulatedFill(
                    qty=0,
                    price=fill_px,
                    mid=mid,
                    commission=0.0,
                    fees=0.0,
                    slippage_bps=0.0,
                    slippage_cost=0.0,
                    gross_notional=0.0,
                    net_cash_delta=0.0,
                    participation_pct=part,
                    reject=RejectReason.OTHER,
                    reject_detail="limit not marketable",
                )

        bps = self._slippage_bps_used(order.side, is_stop=is_stop, participation_pct=part)
        gross = q * fill_px
        commission = self.cost.estimate_commission(int(q), fill_px)
        fees = (
            self.cost.estimate_sell_fees(int(q), fill_px)
            if order.side == OrderSide.SELL
            else 0.0
        )
        slip_cost = abs(fill_px - mid) * q
        if order.side == OrderSide.BUY:
            net_cash = -(gross + commission + fees)
        else:
            net_cash = gross - commission - fees

        return SimulatedFill(
            qty=q,
            price=fill_px,
            mid=mid,
            commission=float(commission),
            fees=float(fees),
            slippage_bps=float(bps),
            slippage_cost=float(slip_cost),
            gross_notional=float(gross),
            net_cash_delta=float(net_cash),
            participation_pct=float(part),
            liquidity="paper_sim",
        )

    def clip_qtys(self, total_qty: float) -> List[float]:
        """Split total into n_clips whole-share clips (TWAP-lite)."""
        total = int(total_qty)
        if total <= 0:
            return []
        n = min(self.n_clips, total)
        base = total // n
        rem = total % n
        clips = []
        for i in range(n):
            q = base + (1 if i < rem else 0)
            if q > 0:
                clips.append(float(q))
        return clips

    def simulate_twap(
        self,
        order: PaperOrder,
        quotes: Sequence[FillQuote],
    ) -> List[SimulatedFill]:
        """Fill remaining qty across clips using successive quotes (or last quote)."""
        remaining = order.remaining_qty
        clips = self.clip_qtys(remaining)
        if not clips:
            return []
        fills: List[SimulatedFill] = []
        for i, cq in enumerate(clips):
            quote = quotes[min(i, len(quotes) - 1)] if quotes else FillQuote(mid=0.0)
            # temporary order slice
            slice_order = PaperOrder(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                qty=cq,
                order_type=order.order_type,
                limit_px=order.limit_px,
                status=order.status,
                is_stop=order.is_stop,
                filled_qty=0.0,
            )
            fills.append(self.simulate_fill(slice_order, quote, qty=cq))
        return fills
