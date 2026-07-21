"""Entry confirmation at next-session open (gap / min price / halt)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from paper_live.datafeed.base import Bar
from paper_live.signals.daily_pipeline import EntryCandidate


@dataclass(frozen=True)
class ConfirmationResult:
    ok: bool
    ticker: str
    entry_px_ref: float
    gap_pct: float
    reason: str = ""

    def to_dict(self):
        return {
            "ok": self.ok,
            "ticker": self.ticker,
            "entry_px_ref": self.entry_px_ref,
            "gap_pct": self.gap_pct,
            "reason": self.reason,
        }


def confirm_entry(
    candidate: EntryCandidate,
    open_bar: Optional[Bar],
    *,
    min_price: float = 5.0,
    max_gap_pct: float = 0.08,
    max_adverse_gap_pct: float = 0.05,
) -> ConfirmationResult:
    """Confirm candidate at entry-session open.

    Rejects:
    - missing bar / halt-like missing open
    - open below min_price
    - gap up too large (chase) or gap down beyond adverse threshold
    """
    t = candidate.ticker
    if open_bar is None:
        return ConfirmationResult(False, t, 0.0, 0.0, "no_open_bar")
    op = float(open_bar.open)
    if op <= 0 or not (op == op):  # NaN
        return ConfirmationResult(False, t, 0.0, 0.0, "bad_open")
    if op < min_price:
        return ConfirmationResult(False, t, op, 0.0, "min_price")

    prev = float(candidate.close)
    gap = (op / prev - 1.0) if prev > 0 else 0.0
    if gap > max_gap_pct:
        return ConfirmationResult(False, t, op, gap, "gap_up_chase")
    if gap < -max_adverse_gap_pct:
        return ConfirmationResult(False, t, op, gap, "gap_down")

    return ConfirmationResult(True, t, op, gap, "ok")
