"""Premium-seller management rules + bid haircut + assignment proxy.

All marks remain model-based (proxy_bs / vix_surface). Virtual capital only.

Config keys (spec.meta or defaults for short-premium kinds):
  - take_profit_credit_frac (default 0.50): close when credit captured ≥ frac
  - stop_loss_credit_mult (default 2.0): close when loss ≥ mult × initial credit
  - max_rolls (default 1): max DTE rolls per structure lifetime
  - bid_haircut (default 0.05): sell premium at mid×(1−h)
  - enable_assignment_proxy (default True)
  - deep_itm_assign_pct (default 0.08): |S−K|/K threshold for early assign stub
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


SHORT_PREMIUM_KINDS = frozenset(
    {
        "cash_secured_put",
        "put_credit_spread",
        "call_credit_spread",
        "iron_condor",
        "covered_call",  # short call side managed on credit
    }
)


@dataclass(frozen=True)
class ManagementConfig:
    take_profit_credit_frac: float = 0.50
    stop_loss_credit_mult: float = 2.0
    max_rolls: int = 1
    bid_haircut: float = 0.05
    enable_assignment_proxy: bool = True
    deep_itm_assign_pct: float = 0.08
    manage_long_premium: bool = False  # protective put / long wings not TP'd as sellers
    # Mgmt 2.0 — time exit (premium seller discipline)
    time_exit_dte: int = 0  # 0 = disabled; else close when DTE ≤ N and residual low
    time_exit_residual_credit_frac: float = 0.25  # residual credit / initial ≤ this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "take_profit_credit_frac": self.take_profit_credit_frac,
            "stop_loss_credit_mult": self.stop_loss_credit_mult,
            "max_rolls": self.max_rolls,
            "bid_haircut": self.bid_haircut,
            "enable_assignment_proxy": self.enable_assignment_proxy,
            "deep_itm_assign_pct": self.deep_itm_assign_pct,
            "manage_long_premium": self.manage_long_premium,
            "time_exit_dte": self.time_exit_dte,
            "time_exit_residual_credit_frac": self.time_exit_residual_credit_frac,
        }


def management_from_meta(
    meta: Optional[Mapping[str, Any]],
    *,
    kind: str = "",
    defaults: Optional[ManagementConfig] = None,
) -> ManagementConfig:
    """Build management config from strategy meta with kind-aware defaults."""
    d = defaults or ManagementConfig()
    m = dict(meta or {})
    # Long-premium structures: no short-seller TP/SL by default
    is_short = kind in SHORT_PREMIUM_KINDS and kind != "protective_put"
    return ManagementConfig(
        take_profit_credit_frac=float(
            m.get("take_profit_credit_frac", d.take_profit_credit_frac if is_short else 1.01)
        ),
        stop_loss_credit_mult=float(
            m.get("stop_loss_credit_mult", d.stop_loss_credit_mult if is_short else 99.0)
        ),
        max_rolls=int(m.get("max_rolls", d.max_rolls)),
        bid_haircut=float(m.get("bid_haircut", d.bid_haircut)),
        enable_assignment_proxy=bool(
            m.get("enable_assignment_proxy", d.enable_assignment_proxy)
        ),
        deep_itm_assign_pct=float(m.get("deep_itm_assign_pct", d.deep_itm_assign_pct)),
        manage_long_premium=bool(m.get("manage_long_premium", False)),
        time_exit_dte=int(m.get("time_exit_dte", d.time_exit_dte)),
        time_exit_residual_credit_frac=float(
            m.get("time_exit_residual_credit_frac", d.time_exit_residual_credit_frac)
        ),
    )


def apply_bid_haircut(
    mid_premium: float,
    *,
    side: str,
    haircut: float,
) -> float:
    """
    Adjust mid mark for realistic short-sale fills.

    - side='sell': credit received = mid × (1 − haircut)  (worse for seller)
    - side='buy':  debit paid = mid × (1 + haircut)       (worse for buyer)

    ``haircut`` is a fraction of mid (e.g. 0.05 = 5%). Documented as bid/ask
    proxy — not NBBO.
    """
    h = max(float(haircut), 0.0)
    px = max(float(mid_premium), 0.0)
    s = (side or "sell").lower()
    if s == "sell":
        return px * (1.0 - h)
    if s == "buy":
        return px * (1.0 + h)
    return px


def credit_captured_frac(initial_credit: float, mark_to_close: float) -> float:
    """
    Fraction of initial credit captured for a short-premium book.

    initial_credit > 0 cash received at open.
    mark_to_close = signed debit to flatten (can be negative if longs dominate).
    Captured = (credit − mark) / credit; may exceed 1.0 when mark < 0.
    """
    c = float(initial_credit)
    m = float(mark_to_close)
    if c <= 1e-12:
        return 0.0
    return (c - m) / c


def should_take_profit(
    initial_credit: float,
    mark_to_close: float,
    *,
    frac: float = 0.50,
) -> bool:
    if initial_credit <= 1e-12:
        return False
    return credit_captured_frac(initial_credit, mark_to_close) + 1e-12 >= float(frac)


def should_stop_loss(
    initial_credit: float,
    mark_to_close: float,
    *,
    mult: float = 2.0,
) -> bool:
    """
    Stop when unrealized loss ≥ mult × initial credit.

    loss = mark_to_close − initial_credit  (for short premium).
    Trigger when mark_to_close ≥ initial_credit × (1 + mult).
    Example: credit=1, mult=2 → stop when mark ≥ 3 (loss of 2).
    """
    c = float(initial_credit)
    if c <= 1e-12:
        return False
    return float(mark_to_close) + 1e-12 >= c * (1.0 + float(mult))


def can_roll(rolls_done: int, max_rolls: int) -> bool:
    """True if another DTE roll is allowed (rolls_done counts completed rolls)."""
    return int(rolls_done) < int(max_rolls)


@dataclass
class AssignmentEvent:
    leg: str  # short_put | short_call
    strike: float
    contracts: int
    shares_delta: float  # +100*n put assign; −100*n call assign
    cash_delta: float  # −K*100*n put; +K*100*n call
    reason: str
    label: str = "assignment_proxy"


def check_assignment(
    *,
    spot: float,
    short_put_k: Optional[float],
    short_call_k: Optional[float],
    contracts: int,
    stock_qty: float,
    at_expiry: bool,
    deep_itm_pct: float = 0.08,
    enabled: bool = True,
) -> list[AssignmentEvent]:
    """
    Approximate American assignment for short equity options.

    Rules (simple, labeled ``assignment_proxy``):
      - At expiry: short put ITM (S < K) → long stock @ K
      - At expiry: short call ITM (S > K) → deliver stock @ K (if stock held)
      - Deep ITM before expiry (optional stub): same if |S−K|/K ≥ deep_itm_pct
        and (for calls) stock is held; early put assign only when deep ITM.
    """
    if not enabled or contracts <= 0:
        return []
    events: list[AssignmentEvent] = []
    n = int(contracts)
    s = float(spot)

    def _deep(k: float) -> bool:
        if k <= 0:
            return False
        return abs(s - k) / k + 1e-12 >= float(deep_itm_pct)

    if short_put_k is not None and s < float(short_put_k):
        if at_expiry or _deep(float(short_put_k)):
            k = float(short_put_k)
            events.append(
                AssignmentEvent(
                    leg="short_put",
                    strike=k,
                    contracts=n,
                    shares_delta=100.0 * n,
                    cash_delta=-k * 100.0 * n,
                    reason="expiry_itm" if at_expiry else "deep_itm",
                )
            )
    if short_call_k is not None and s > float(short_call_k):
        if at_expiry or _deep(float(short_call_k)):
            k = float(short_call_k)
            # Can only deliver if we have shares (covered); naked call → cash settle intrinsic
            if stock_qty + 1e-9 >= 100.0 * n:
                events.append(
                    AssignmentEvent(
                        leg="short_call",
                        strike=k,
                        contracts=n,
                        shares_delta=-100.0 * n,
                        cash_delta=k * 100.0 * n,
                        reason="expiry_itm" if at_expiry else "deep_itm",
                    )
                )
            else:
                # cash-settle intrinsic as proxy (no stock)
                intrinsic = max(s - k, 0.0) * 100.0 * n
                events.append(
                    AssignmentEvent(
                        leg="short_call",
                        strike=k,
                        contracts=n,
                        shares_delta=0.0,
                        cash_delta=-intrinsic,
                        reason=("expiry_itm_cash" if at_expiry else "deep_itm_cash"),
                    )
                )
    return events


def structure_mark_to_close(
    *,
    short_call_mid: float = 0.0,
    short_put_mid: float = 0.0,
    long_call_mid: float = 0.0,
    long_put_mid: float = 0.0,
    contracts: int = 0,
) -> float:
    """
    Net debit to flatten option legs (short pays mid, long receives mid) × 100 × n.

    **Signed** (no floor at 0): if long wings dominate, result can be negative
    (net credit to close). TP uses capture = (initial_credit − mark) / credit;
    a negative mark correctly implies >100% capture without a spurious zero-floor TP.
    """
    n = max(int(contracts), 0)
    if n <= 0:
        return 0.0
    # Cost to close = buy back shorts − sell longs (may be negative)
    per = (short_call_mid + short_put_mid) - (long_call_mid + long_put_mid)
    return float(per) * 100.0 * n


def residual_credit_frac(initial_credit: float, mark_to_close: float) -> float:
    """Fraction of initial credit still 'in the market' (1 − capture)."""
    cap = credit_captured_frac(initial_credit, mark_to_close)
    return max(1.0 - cap, 0.0)


def should_time_exit(
    *,
    dte: int,
    initial_credit: float,
    mark_to_close: float,
    time_exit_dte: int = 7,
    residual_frac: float = 0.25,
) -> bool:
    """
    Close short premium when DTE ≤ N and residual credit ≤ residual_frac of initial.

    Example: DTE≤7 and only 25% of credit left to capture → time exit (theta decay
    no longer worth gamma risk near expiry).
    """
    if int(time_exit_dte) <= 0:
        return False
    if int(dte) > int(time_exit_dte):
        return False
    if initial_credit <= 1e-12:
        return False
    return residual_credit_frac(initial_credit, mark_to_close) <= float(residual_frac) + 1e-12


def management_action(
    *,
    kind: str,
    initial_credit: float,
    mark_to_close: float,
    cfg: ManagementConfig,
    dte: Optional[int] = None,
) -> Optional[str]:
    """
    Return 'take_profit' | 'stop_loss' | 'time_exit' | None for short-premium kinds.
    """
    if kind not in SHORT_PREMIUM_KINDS:
        return None
    if kind == "covered_call":
        # Manage only on short call credit if initial_credit tracked
        pass
    if initial_credit <= 1e-12:
        return None
    if should_take_profit(
        initial_credit, mark_to_close, frac=cfg.take_profit_credit_frac
    ):
        return "take_profit"
    if should_stop_loss(
        initial_credit, mark_to_close, mult=cfg.stop_loss_credit_mult
    ):
        return "stop_loss"
    if dte is not None and should_time_exit(
        dte=int(dte),
        initial_credit=initial_credit,
        mark_to_close=mark_to_close,
        time_exit_dte=cfg.time_exit_dte,
        residual_frac=cfg.time_exit_residual_credit_frac,
    ):
        return "time_exit"
    return None
