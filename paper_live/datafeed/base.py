"""Bar / day snapshot types for paper datafeeds."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional


@dataclass(frozen=True)
class Bar:
    """Single OHLCV bar (daily or lower)."""

    ticker: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def day(self) -> date:
        t = self.ts
        if getattr(t, "tzinfo", None) is not None:
            return t.date()
        return t.date() if hasattr(t, "date") else t  # type: ignore[return-value]

    def mid(self) -> float:
        return float(self.close)


@dataclass(frozen=True)
class DayBars:
    """All available bars for one calendar session date."""

    day: date
    bars: Dict[str, Bar]

    def get(self, ticker: str) -> Optional[Bar]:
        return self.bars.get(ticker.upper())

    def tickers(self):
        return list(self.bars.keys())
