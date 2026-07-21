"""Point-in-time membership, delisting dates, ISIN chains, survivorship-free benchmarks.

Uses EODHD-style catalogs + inferred first/last EOD bars as listing window
(CRSP-grade listing calendars not available on typical EODHD plans).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from trad_research.features import load_history

logger = logging.getLogger(__name__)

DEFAULT_MEMBERSHIP_PATH = Path("data/pit/membership_index.json")
DEFAULT_CATALOG_ACTIVE = Path("data/pit/catalog_active_us_common.json")
DEFAULT_CATALOG_DELISTED = Path("data/pit/catalog_delisted_us_common.json")


@dataclass
class MembershipRow:
    ticker: str
    isin: Optional[str] = None
    name: str = ""
    exchange: str = "US"
    first_date: Optional[str] = None  # ISO date
    last_date: Optional[str] = None
    source: str = "inferred_eod"  # catalog + eod
    is_delisted_catalog: bool = False

    def first_ts(self) -> Optional[pd.Timestamp]:
        if not self.first_date:
            return None
        return pd.Timestamp(self.first_date, tz="UTC")

    def last_ts(self) -> Optional[pd.Timestamp]:
        if not self.last_date:
            return None
        return pd.Timestamp(self.last_date, tz="UTC")


class MembershipIndex:
    """ticker -> MembershipRow with as-of queries."""

    def __init__(self, rows: Optional[Dict[str, MembershipRow]] = None):
        self.rows: Dict[str, MembershipRow] = rows or {}

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, ticker: str) -> Optional[MembershipRow]:
        return self.rows.get(ticker.upper())

    def add(self, row: MembershipRow) -> None:
        self.rows[row.ticker.upper()] = row

    def is_listed(self, ticker: str, as_of: pd.Timestamp) -> bool:
        row = self.get(ticker)
        if row is None:
            return False
        a = _utc(as_of)
        f, l = row.first_ts(), row.last_ts()
        if f is None or l is None:
            return False
        # listed inclusive of last trade day
        return f.normalize() <= a.normalize() <= l.normalize()

    def members_as_of(
        self,
        as_of: pd.Timestamp,
        *,
        tickers: Optional[Sequence[str]] = None,
    ) -> List[str]:
        a = _utc(as_of)
        pool = list(tickers) if tickers is not None else list(self.rows.keys())
        out = []
        for t in pool:
            if self.is_listed(t, a):
                out.append(t.upper())
        return sorted(set(out))

    def delist_date(self, ticker: str) -> Optional[pd.Timestamp]:
        row = self.get(ticker)
        return row.last_ts() if row else None

    def isin_chains(self) -> Dict[str, List[MembershipRow]]:
        """ISIN -> membership rows sorted by first_date."""
        chains: Dict[str, List[MembershipRow]] = {}
        for row in self.rows.values():
            if not row.isin or row.isin in ("", "null", "None"):
                continue
            chains.setdefault(row.isin, []).append(row)
        for isin, lst in chains.items():
            lst.sort(key=lambda r: r.first_date or "9999")
        return chains

    def successor_after_delist(
        self,
        ticker: str,
        *,
        gap_days: int = 10,
    ) -> Optional[str]:
        """If same ISIN has a later ticker starting within gap_days of delist, return it."""
        row = self.get(ticker)
        if not row or not row.isin or not row.last_date:
            return None
        chain = self.isin_chains().get(row.isin) or []
        last = row.last_ts()
        if last is None:
            return None
        for other in chain:
            if other.ticker.upper() == ticker.upper():
                continue
            f = other.first_ts()
            if f is None:
                continue
            # successor starts after or slightly overlapping delist
            delta = (f.normalize() - last.normalize()).days
            if -2 <= delta <= gap_days:
                return other.ticker.upper()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "pit-membership-v1",
            "n": len(self.rows),
            "rows": {k: asdict(v) for k, v in self.rows.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MembershipIndex":
        idx = cls()
        for k, v in (data.get("rows") or {}).items():
            idx.add(MembershipRow(**v))
        return idx

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Wrote membership index n=%d -> %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> "MembershipIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def infer_eod_span(data_root: Path, ticker: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    hist = load_history(ticker, data_root)
    if hist.empty:
        return None, None
    d = pd.to_datetime(hist["date"], utc=True)
    return d.min(), d.max()


def build_membership_from_catalogs_and_eod(
    data_root: Path,
    catalog_active: Sequence[Dict[str, Any]],
    catalog_delisted: Sequence[Dict[str, Any]],
    *,
    only_tickers: Optional[Set[str]] = None,
) -> MembershipIndex:
    """Merge catalogs + inferred EOD spans for tickers that have history on disk."""
    idx = MembershipIndex()
    catalogs = []
    for row in catalog_active:
        catalogs.append((row, False))
    for row in catalog_delisted:
        catalogs.append((row, True))

    by_code: Dict[str, Tuple[Dict[str, Any], bool]] = {}
    for row, delisted in catalogs:
        code = str(row.get("Code") or row.get("code") or "").upper()
        if not code:
            continue
        by_code[code] = (row, delisted)

    # Prefer scanning data_root histories so we only index what we can trade
    hist_tickers = sorted(
        {
            p.name.replace("_history.csv", "").upper()
            for p in Path(data_root).glob("*_history.csv")
        }
    )
    if only_tickers is not None:
        hist_tickers = [t for t in hist_tickers if t in only_tickers]

    for t in hist_tickers:
        first, last = infer_eod_span(data_root, t)
        if first is None or last is None:
            continue
        meta, is_del = by_code.get(t, ({}, False))
        isin = meta.get("Isin") or meta.get("isin")
        if isin is not None:
            isin = str(isin) if str(isin).lower() not in ("none", "null", "nan") else None
        idx.add(
            MembershipRow(
                ticker=t,
                isin=isin,
                name=str(meta.get("Name") or meta.get("name") or ""),
                exchange=str(meta.get("Exchange") or meta.get("exchange") or "US"),
                first_date=first.strftime("%Y-%m-%d"),
                last_date=last.strftime("%Y-%m-%d"),
                source="catalog+eod",
                is_delisted_catalog=bool(is_del)
                or (last < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)),
            )
        )
    return idx


def filter_panels_pit(
    panels: Dict[str, pd.DataFrame],
    membership: MembershipIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Keep only tickers listed at some point in [start, end]; clip bars to membership window."""
    out: Dict[str, pd.DataFrame] = {}
    start, end = _utc(start), _utc(end)
    for t, df in panels.items():
        row = membership.get(t)
        if row is None:
            # no membership → exclude under strict PIT (survivorship-safe mode)
            continue
        f, l = row.first_ts(), row.last_ts()
        if f is None or l is None:
            continue
        # overlap with window?
        if l < start or f > end:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], utc=True)
        d = d[(d["date"] >= f) & (d["date"] <= l) & (d["date"] >= start) & (d["date"] <= end)]
        if len(d) < 30:
            continue
        out[t.upper()] = d.reset_index(drop=True)
    return out


def _clip_ret(r: float, cap: float = 0.35) -> Optional[float]:
    """Drop non-finite / extreme single-name daily returns (splits/bad prints)."""
    if not np.isfinite(r):
        return None
    if abs(r) > cap:
        return None
    return float(r)


def build_equal_weight_benchmark(
    panels: Dict[str, pd.DataFrame],
    membership: MembershipIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    min_members: int = 5,
    min_price: float = 2.0,
    max_daily_ret: float = 0.35,
) -> pd.Series:
    """Daily equal-weight return of PIT members with valid close; equity starts at 1.0.

    Extreme single-name daily moves (|r|>max_daily_ret) are dropped (corporate-action
    / bad print protection). min_price defaults to $2 to limit penny noise.
    """
    start, end = _utc(start), _utc(end)
    closes: Dict[str, pd.Series] = {}
    for t, df in panels.items():
        if membership.get(t) is None:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], utc=True)
        s = d.set_index("date")["close"].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        closes[t.upper()] = s

    if not closes:
        return pd.Series(dtype=float)

    all_dates = sorted(
        {d for s in closes.values() for d in s.index if start <= d <= end}
    )
    if not all_dates:
        return pd.Series(dtype=float)

    equity = []
    eq = 1.0
    prev_prices: Dict[str, float] = {}
    for i, day in enumerate(all_dates):
        prices = {}
        for t, s in closes.items():
            if not membership.is_listed(t, day):
                continue
            if day not in s.index:
                continue
            px = float(s.loc[day])
            if not np.isfinite(px) or px < min_price:
                continue
            prices[t] = px
        if len(prices) < min_members:
            equity.append((day, eq))
            prev_prices = prices
            continue
        if i == 0 or not prev_prices:
            equity.append((day, eq))
            prev_prices = prices
            continue
        rets = []
        for t, px in prices.items():
            if t in prev_prices and prev_prices[t] > 0:
                rr = _clip_ret(px / prev_prices[t] - 1.0, max_daily_ret)
                if rr is not None:
                    rets.append(rr)
        if rets:
            eq *= 1.0 + float(np.mean(rets))
        equity.append((day, eq))
        prev_prices = prices

    ser = pd.Series({d: e for d, e in equity}).sort_index()
    ser.name = "pit_equal_weight"
    return ser


def build_dollar_volume_weight_benchmark(
    panels: Dict[str, pd.DataFrame],
    membership: MembershipIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    min_members: int = 5,
    min_price: float = 2.0,
    lookback: int = 20,
    max_daily_ret: float = 0.35,
) -> pd.Series:
    """Dollar-volume weighted proxy for cap-weight (mcap often unavailable)."""
    start, end = _utc(start), _utc(end)
    closes: Dict[str, pd.Series] = {}
    dvols: Dict[str, pd.Series] = {}
    for t, df in panels.items():
        if membership.get(t) is None:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], utc=True)
        d = d.set_index("date").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        px = d["close"].astype(float)
        vol = d["volume"].astype(float) if "volume" in d.columns else pd.Series(0.0, index=d.index)
        dv = (px * vol).rolling(lookback, min_periods=max(5, lookback // 2)).mean()
        closes[t.upper()] = px
        dvols[t.upper()] = dv

    all_dates = sorted({d for s in closes.values() for d in s.index if start <= d <= end})
    eq = 1.0
    out = []
    prev_px: Dict[str, float] = {}
    for i, day in enumerate(all_dates):
        prices = {}
        weights = {}
        for t, s in closes.items():
            if not membership.is_listed(t, day) or day not in s.index:
                continue
            px = float(s.loc[day])
            if not np.isfinite(px) or px < min_price:
                continue
            w = float(dvols[t].loc[day]) if day in dvols[t].index else 0.0
            if not np.isfinite(w) or w <= 0:
                w = 1.0  # fallback equal
            prices[t] = px
            weights[t] = w
        if len(prices) < min_members:
            out.append((day, eq))
            prev_px = prices
            continue
        if i == 0 or not prev_px:
            out.append((day, eq))
            prev_px = prices
            continue
        # recompute weights only on names with valid clipped returns
        contrib = []
        for t, px in prices.items():
            if t not in prev_px or prev_px[t] <= 0:
                continue
            rr = _clip_ret(px / prev_px[t] - 1.0, max_daily_ret)
            if rr is None:
                continue
            contrib.append((weights[t], rr))
        if contrib:
            wsum = sum(w for w, _ in contrib)
            if wsum > 0:
                r = sum((w / wsum) * rr for w, rr in contrib)
                eq *= 1.0 + r
        out.append((day, eq))
        prev_px = prices
    ser = pd.Series({d: e for d, e in out}).sort_index()
    ser.name = "pit_dv_weight"
    return ser


def attach_delist_dates_to_config(
    membership: MembershipIndex,
    tickers: Sequence[str],
    *,
    only_terminal: bool = True,
    active_asof: Optional[pd.Timestamp] = None,
    min_days_since_last: int = 45,
) -> Dict[str, pd.Timestamp]:
    """ticker -> last trade date for backtest delisting exits.

    When only_terminal=True (default), only names that look truly delisted get a
    force-exit date: catalog delisted flag OR last bar older than
    min_days_since_last vs active_asof (default: now UTC). Surviving names with
    recent last bars are omitted so end-of-CSV is not treated as a delisting.
    """
    asof = _utc(active_asof) if active_asof is not None else pd.Timestamp.now(tz="UTC")
    out: Dict[str, pd.Timestamp] = {}
    for t in tickers:
        row = membership.get(t)
        if row is None:
            continue
        ld = row.last_ts()
        if ld is None:
            continue
        if only_terminal:
            stale = (asof.normalize() - ld.normalize()).days >= min_days_since_last
            if not (row.is_delisted_catalog or stale):
                continue
        out[t.upper()] = ld
    return out


def build_research_universe(
    membership: MembershipIndex,
    *,
    max_n: int = 120,
    prefer_tickers: Optional[Sequence[str]] = None,
    window_start: str = "2005-01-01",
    window_end: str = "2020-12-31",
    min_bars_proxy_days: int = 400,
    delisted_frac: float = 0.25,
) -> List[str]:
    """Balanced survivor + delisted set that overlaps a research window.

    Reserves ~delisted_frac slots for terminal names whose last bar falls inside
    the research window (so delisting exits can fire in bakeoffs), then fills
    with prefer_tickers and remaining active/delisted members.
    """
    w0 = pd.Timestamp(window_start, tz="UTC")
    w1 = pd.Timestamp(window_end, tz="UTC")
    chosen: List[str] = []
    seen: Set[str] = set()
    n_del_target = max(5, int(max_n * delisted_frac))

    def _ok(row: MembershipRow) -> bool:
        f, l = row.first_ts(), row.last_ts()
        if f is None or l is None:
            return False
        if l < w0 or f > w1:
            return False
        span = (l - f).days
        return span >= min_bars_proxy_days

    def _add(t: str) -> bool:
        t = t.upper()
        if t in seen or t in ("SPY", "QQQ"):
            return False
        row = membership.get(t)
        if row is None or not _ok(row):
            return False
        seen.add(t)
        chosen.append(t)
        return True

    delisted = [r for r in membership.rows.values() if r.is_delisted_catalog and _ok(r)]
    # Prefer delists that end inside the research window (realized terminal events)
    delisted.sort(
        key=lambda r: (
            0 if w0 <= (r.last_ts() or w1) <= w1 else 1,
            -(r.last_ts() - r.first_ts()).days if r.last_ts() and r.first_ts() else 0,
            r.last_date or "",
        )
    )
    n_del_added = 0
    for r in delisted:
        if n_del_added >= n_del_target or len(chosen) >= max_n:
            break
        if _add(r.ticker):
            n_del_added += 1

    if prefer_tickers:
        for t in prefer_tickers:
            if len(chosen) >= max_n:
                break
            _add(t)

    for r in delisted:
        if len(chosen) >= max_n:
            break
        _add(r.ticker)

    active = [r for r in membership.rows.values() if not r.is_delisted_catalog and _ok(r)]
    active.sort(key=lambda r: r.first_date or "9999")
    for r in active:
        if len(chosen) >= max_n:
            break
        _add(r.ticker)

    return chosen


def write_trade_universe_file(
    tickers: Sequence[str],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(t.upper() for t in tickers) + "\n", encoding="utf-8")
    return path
