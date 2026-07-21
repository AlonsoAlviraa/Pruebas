"""Free market data for cloud paper (no paid API required).

Primary: Stooq daily CSV (no key).
Fallback: deterministic synthetic panels if download fails.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed

logger = logging.getLogger(__name__)

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"


def _stooq_symbol(ticker: str) -> str:
    t = ticker.upper().strip()
    # US equities / ETFs on Stooq use .us suffix
    if "." in t:
        return t.lower()
    return f"{t.lower()}.us"


def fetch_stooq_daily(ticker: str, *, timeout: int = 30) -> pd.DataFrame:
    """Download full daily history from Stooq. Empty on failure."""
    sym = _stooq_symbol(ticker)
    url = STOOQ_URL.format(symbol=sym)
    req = Request(url, headers={"User-Agent": "trad-paper-cloud/1.0 (research)"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, TimeoutError, OSError) as e:
        logger.warning("Stooq fetch failed %s: %s", ticker, e)
        return pd.DataFrame()

    if not raw or "Date" not in raw[:200] and "date" not in raw[:200].lower():
        # Stooq sometimes returns HTML error
        if "<html" in raw.lower() or len(raw) < 50:
            logger.warning("Stooq empty/HTML for %s", ticker)
            return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception as e:
        logger.warning("Stooq parse fail %s: %s", ticker, e)
        return pd.DataFrame()

    df.columns = [str(c).lower().strip() for c in df.columns]
    # stooq: date,open,high,low,close,volume
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = df["close"] if col != "volume" else 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df
    # Stooq often newest-first; ensure ascending
    if len(df) > 1 and df["date"].iloc[0] > df["date"].iloc[-1]:
        df = df.iloc[::-1].reset_index(drop=True)
    df["ticker"] = ticker.upper()
    return df


def load_free_panels(
    tickers: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
    min_rows: int = 120,
    lookback_calendar_days: int = 500,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Fetch free OHLCV for tickers. Returns (panels, source_by_ticker)."""
    panels: Dict[str, pd.DataFrame] = {}
    sources: Dict[str, str] = {}
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(lookback_calendar_days))

    for t in tickers:
        t = t.upper()
        df = pd.DataFrame()
        cache_path = cache_dir / f"{t}_history.csv" if cache_dir else None

        # Prefer fresh Stooq; fall back to cache
        df = fetch_stooq_daily(t)
        if not df.empty and cache_path is not None:
            try:
                df.to_csv(cache_path, index=False)
            except Exception:
                pass
        if df.empty and cache_path is not None and cache_path.is_file():
            try:
                df = pd.read_csv(cache_path)
                df.columns = [c.lower().strip() for c in df.columns]
                df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                df = df.dropna(subset=["date", "close"]).sort_values("date")
                sources[t] = "cache"
            except Exception:
                df = pd.DataFrame()

        if not df.empty:
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            if len(df) >= min_rows:
                panels[t] = df
                sources.setdefault(t, "stooq")
                continue

        sources[t] = "missing"

    return panels, sources


def build_cloud_feed(
    tickers: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
    lookback_calendar_days: int = 500,
    force_synthetic: bool = False,
    synthetic_seed: int = 42,
) -> tuple[DailyReplayFeed, Dict[str, str]]:
    """Build DailyReplayFeed from free data or synthetic fallback."""
    meta: Dict[str, str] = {}
    if force_synthetic:
        feed = DailyReplayFeed.from_synthetic(
            list(tickers),
            n_days=max(260, lookback_calendar_days // 2),
            start=(datetime.now(timezone.utc) - timedelta(days=lookback_calendar_days)).strftime(
                "%Y-%m-%d"
            ),
            seed=synthetic_seed,
        )
        return feed, {t.upper(): "synthetic" for t in tickers}

    panels, sources = load_free_panels(
        tickers,
        cache_dir=cache_dir,
        lookback_calendar_days=lookback_calendar_days,
    )
    # Ensure QQQ/SPY for regime if missing → synthetic index
    need = [t.upper() for t in tickers]
    missing = [t for t in need if t not in panels]
    if len(panels) < 3 or missing:
        logger.warning(
            "Free data incomplete (%d panels, missing=%s) — filling gaps with synthetic",
            len(panels),
            missing,
        )
        syn = DailyReplayFeed.from_synthetic(
            need,
            n_days=320,
            start=(datetime.now(timezone.utc) - timedelta(days=500)).strftime("%Y-%m-%d"),
            seed=synthetic_seed,
        )
        for t in need:
            if t not in panels and t in syn._raw:
                panels[t] = syn._raw[t]
                sources[t] = "synthetic_gapfill"
        meta["fallback"] = "partial_or_full_synthetic"

    if not panels:
        feed = DailyReplayFeed.from_synthetic(
            list(tickers), n_days=320, seed=synthetic_seed
        )
        return feed, {t.upper(): "synthetic" for t in tickers}

    feed = DailyReplayFeed(panels, min_history=50)
    return feed, sources
