"""Daily OHLCV replay from in-memory frames or data/*_history.csv (LIV-03)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from paper_live.datafeed.base import Bar, DayBars
from trad_research.features import engineer_m2_features, load_history


def _to_utc_ts(d: Union[str, date, datetime, pd.Timestamp]) -> pd.Timestamp:
    t = pd.Timestamp(d)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _row_to_bar(ticker: str, row: Mapping[str, object]) -> Bar:
    ts = _to_utc_ts(row["date"])
    return Bar(
        ticker=ticker.upper(),
        ts=ts.to_pydatetime(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume") or 0.0),
    )


class DailyReplayFeed:
    """Causal daily bar store for paper replay.

    - ``asof(day)`` returns bars with date == day (session).
    - ``history(ticker, through=day)`` returns OHLCV rows with date <= day only.
    - ``featured(ticker, through=day)`` engineers M2 features on causal history.
    """

    def __init__(
        self,
        panels: Mapping[str, pd.DataFrame],
        *,
        min_history: int = 60,
    ):
        self.min_history = int(min_history)
        self._raw: Dict[str, pd.DataFrame] = {}
        self._by_day: Dict[date, Dict[str, Bar]] = {}
        self._days: List[date] = []

        all_days = set()
        for t, df in panels.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            d.columns = [str(c).lower().strip() for c in d.columns]
            if "date" not in d.columns or "close" not in d.columns:
                continue
            d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
            for col in ("open", "high", "low", "close", "volume"):
                if col not in d.columns:
                    if col == "volume":
                        d["volume"] = 0.0
                    elif col in ("open", "high", "low"):
                        d[col] = d["close"]
                d[col] = pd.to_numeric(d[col], errors="coerce")
            d = d.dropna(subset=["date", "close"]).sort_values("date")
            d = d[~d["date"].dt.normalize().duplicated(keep="last")].reset_index(drop=True)
            if d.empty:
                continue
            key = t.upper()
            self._raw[key] = d
            for _, row in d.iterrows():
                bar = _row_to_bar(key, row)
                day = bar.day
                self._by_day.setdefault(day, {})[key] = bar
                all_days.add(day)

        self._days = sorted(all_days)

    @classmethod
    def from_data_root(
        cls,
        data_root: Union[str, Path],
        tickers: Sequence[str],
        *,
        min_history: int = 60,
    ) -> "DailyReplayFeed":
        root = Path(data_root)
        panels: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            hist = load_history(t.upper(), root)
            if not hist.empty:
                panels[t.upper()] = hist
        return cls(panels, min_history=min_history)

    @classmethod
    def from_synthetic(
        cls,
        tickers: Sequence[str],
        *,
        n_days: int = 300,
        start: str = "2020-01-02",
        seed: int = 0,
    ) -> "DailyReplayFeed":
        """Deterministic geometric Brownian-ish synthetic OHLCV for unit tests."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start=start, periods=n_days, tz="UTC")
        panels: Dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            ret = rng.normal(0.0005 + i * 0.0001, 0.02, size=n_days)
            close = 50.0 * (1.0 + i) * np.cumprod(1.0 + ret)
            open_ = np.roll(close, 1)
            open_[0] = close[0]
            high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, n_days))
            low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, n_days))
            vol = rng.integers(100_000, 2_000_000, size=n_days).astype(float)
            panels[t.upper()] = pd.DataFrame(
                {
                    "date": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                }
            )
        return cls(panels, min_history=50)

    @property
    def tickers(self) -> List[str]:
        return sorted(self._raw.keys())

    @property
    def days(self) -> List[date]:
        return list(self._days)

    def raw_panels(self) -> Dict[str, pd.DataFrame]:
        """Public copy of per-ticker OHLCV panels (for stress / export)."""
        return {k: v.copy() for k, v in self._raw.items()}

    def session_days(
        self,
        start: Optional[Union[str, date]] = None,
        end: Optional[Union[str, date]] = None,
    ) -> List[date]:
        days = self._days
        if start is not None:
            s = pd.Timestamp(start).date()
            days = [d for d in days if d >= s]
        if end is not None:
            e = pd.Timestamp(end).date()
            days = [d for d in days if d <= e]
        return days

    def asof(self, day: Union[str, date]) -> DayBars:
        d = pd.Timestamp(day).date()
        return DayBars(day=d, bars=dict(self._by_day.get(d, {})))

    def bar(self, ticker: str, day: Union[str, date]) -> Optional[Bar]:
        return self.asof(day).get(ticker)

    def history(
        self,
        ticker: str,
        *,
        through: Union[str, date],
        include_through: bool = True,
    ) -> pd.DataFrame:
        """Causal OHLCV: only bars on or before ``through``."""
        t = ticker.upper()
        raw = self._raw.get(t)
        if raw is None or raw.empty:
            return pd.DataFrame()
        end = _to_utc_ts(through).normalize()
        d = raw.copy()
        if include_through:
            d = d[d["date"].dt.normalize() <= end]
        else:
            d = d[d["date"].dt.normalize() < end]
        return d.reset_index(drop=True)

    def featured(
        self,
        ticker: str,
        *,
        through: Union[str, date],
        min_history: Optional[int] = None,
    ) -> pd.DataFrame:
        """Feature frame causal through ``through`` (inclusive)."""
        hist = self.history(ticker, through=through, include_through=True)
        need = int(min_history if min_history is not None else self.min_history)
        if hist.empty or len(hist) < need:
            return pd.DataFrame()
        feat = engineer_m2_features(hist)
        # Keep rows with usable signal fields when possible (still causal)
        if "ret_1m" in feat.columns and "sma_50" in feat.columns:
            usable = feat.dropna(subset=["close", "ret_1m", "sma_50"], how="any")
            if len(usable) >= max(30, need // 3):
                feat = usable
        return feat.reset_index(drop=True)

    def next_session(self, day: Union[str, date]) -> Optional[date]:
        d = pd.Timestamp(day).date()
        for x in self._days:
            if x > d:
                return x
        return None

    def prev_session(self, day: Union[str, date]) -> Optional[date]:
        d = pd.Timestamp(day).date()
        prev = None
        for x in self._days:
            if x >= d:
                return prev
            prev = x
        return prev

    def iter_days(
        self,
        start: Optional[Union[str, date]] = None,
        end: Optional[Union[str, date]] = None,
    ) -> Iterable[DayBars]:
        for d in self.session_days(start, end):
            yield self.asof(d)
