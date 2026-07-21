"""Paper order enums and request/result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"  # marketable stop (treated as market with stop slip)


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACK = "ack"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RejectReason(str, Enum):
    HALT = "halt"
    NO_QUOTE = "no_quote"
    MIN_PRICE = "min_price"
    INSUFFICIENT_CASH = "insufficient_cash"
    INSUFFICIENT_SHARES = "insufficient_shares"
    ADV_CAP = "adv_cap_zero"
    INVALID_QTY = "invalid_qty"
    SHORT_NOT_ALLOWED = "short_not_allowed"
    KILL_SWITCH = "kill_switch"
    DUPLICATE = "duplicate"
    OTHER = "other"


@dataclass
class PaperOrder:
    """In-memory paper order (also mirrored to ledger)."""

    order_id: str
    ticker: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_px: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    reason: Optional[str] = None
    is_stop: bool = False
    filled_qty: float = 0.0
    avg_fill_px: float = 0.0
    submitted_at: Optional[datetime] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_qty(self) -> float:
        return max(0.0, float(self.qty) - float(self.filled_qty))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "ticker": self.ticker,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "limit_px": self.limit_px,
            "status": self.status.value,
            "reason": self.reason,
            "is_stop": self.is_stop,
            "filled_qty": self.filled_qty,
            "avg_fill_px": self.avg_fill_px,
            "remaining_qty": self.remaining_qty,
            "meta": self.meta,
        }
