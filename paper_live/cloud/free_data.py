"""Free *real* market data for cloud paper (no paid API required).

Primary: Yahoo Finance chart API (no key; works on GitHub Actions).
Secondary: Stooq CSV (often JS-blocked).
Tertiary: on-disk seed/cache CSVs of previously downloaded real bars.
Synthetic only when force_synthetic=True (tests / explicit opt-in).
"""
from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
YAHOO_CHART = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?interval=1d&range={range_}"
)
# Alternate host sometimes less rate-limited
YAHOO_CHART_ALT = (
    "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?interval=1d&range={range_}"
)

SEED_DIR = Path(__file__).resolve().parent / "seed_ohlcv"


def _http_get(url: str, *, timeout: int = 45, retries: int = 3) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": DEFAULT_UA,
                    "Accept": "application/json,text/csv,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (URLError, HTTPError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"HTTP GET failed {url}: {last_err}")


def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d.columns = [str(c).lower().strip() for c in d.columns]
    # adjclose optional
    if "date" not in d.columns or "close" not in d.columns:
        return pd.DataFrame()
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in d.columns:
            d[col] = d["close"] if col != "volume" else 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date")
    d = d[~d["date"].dt.normalize().duplicated(keep="last")].reset_index(drop=True)
    if d.empty:
        return d
    if len(d) > 1 and d["date"].iloc[0] > d["date"].iloc[-1]:
        d = d.iloc[::-1].reset_index(drop=True)
    d["ticker"] = ticker.upper()
    return d


def fetch_yahoo_daily(ticker: str, *, range_: str = "5y", timeout: int = 45) -> pd.DataFrame:
    """Download daily OHLCV via Yahoo chart API (real market data)."""
    t = ticker.upper().strip()
    for template in (YAHOO_CHART, YAHOO_CHART_ALT):
        url = template.format(ticker=t, range_=range_)
        try:
            raw = _http_get(url, timeout=timeout)
            j = json.loads(raw.decode("utf-8", errors="replace"))
            result = (j.get("chart") or {}).get("result") or []
            if not result:
                err = (j.get("chart") or {}).get("error")
                logger.warning("Yahoo empty result %s: %s", t, err)
                continue
            res = result[0]
            ts = res.get("timestamp") or []
            q = (res.get("indicators") or {}).get("quote") or [{}]
            q0 = q[0] if q else {}
            if not ts:
                continue
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(ts, unit="s", utc=True),
                    "open": q0.get("open"),
                    "high": q0.get("high"),
                    "low": q0.get("low"),
                    "close": q0.get("close"),
                    "volume": q0.get("volume"),
                }
            )
            out = _normalize_ohlcv(df, t)
            if len(out) >= 30:
                logger.info("Yahoo OK %s bars=%d", t, len(out))
                return out
        except Exception as e:
            logger.warning("Yahoo fetch fail %s: %s", t, e)
            time.sleep(0.4)
    return pd.DataFrame()


def fetch_stooq_daily(ticker: str, *, timeout: int = 30) -> pd.DataFrame:
    """Stooq CSV (often blocked by JS challenge; kept as secondary)."""
    t = ticker.upper().strip()
    sym = t.lower() if "." in t else f"{t.lower()}.us"
    url = STOOQ_URL.format(symbol=sym)
    try:
        raw = _http_get(url, timeout=timeout).decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Stooq fetch failed %s: %s", t, e)
        return pd.DataFrame()
    if not raw or ("Date" not in raw[:300] and "date" not in raw[:300].lower()):
        if "<html" in raw.lower() or len(raw) < 50:
            logger.warning("Stooq empty/HTML for %s", t)
            return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception as e:
        logger.warning("Stooq parse fail %s: %s", t, e)
        return pd.DataFrame()
    return _normalize_ohlcv(df, t)


def _load_csv_panel(path: Path, ticker: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return _normalize_ohlcv(df, ticker)
    except Exception as e:
        logger.warning("CSV load fail %s: %s", path, e)
        return pd.DataFrame()


def fetch_real_daily(
    ticker: str,
    *,
    cache_dir: Optional[Path] = None,
    seed_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str]:
    """Cascade: Yahoo → Stooq → cache → seed. Returns (df, source)."""
    t = ticker.upper()
    df = fetch_yahoo_daily(t)
    if not df.empty:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                df.to_csv(cache_dir / f"{t}_history.csv", index=False)
            except Exception:
                pass
        return df, "yahoo"

    time.sleep(0.3)
    df = fetch_stooq_daily(t)
    if not df.empty:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                df.to_csv(cache_dir / f"{t}_history.csv", index=False)
            except Exception:
                pass
        return df, "stooq"

    if cache_dir is not None:
        df = _load_csv_panel(cache_dir / f"{t}_history.csv", t)
        if not df.empty:
            return df, "cache"

    seed = Path(seed_dir) if seed_dir else SEED_DIR
    df = _load_csv_panel(seed / f"{t}_history.csv", t)
    if not df.empty:
        return df, "seed"

    return pd.DataFrame(), "missing"


def load_free_panels(
    tickers: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
    seed_dir: Optional[Path] = None,
    min_rows: int = 120,
    lookback_calendar_days: int = 500,
    allow_synthetic: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Fetch real OHLCV for tickers. Synthetic only if allow_synthetic."""
    panels: Dict[str, pd.DataFrame] = {}
    sources: Dict[str, str] = {}
    cache_dir = Path(cache_dir) if cache_dir else None
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(lookback_calendar_days))

    for i, t in enumerate(tickers):
        t = t.upper()
        if i:
            time.sleep(0.35)  # be polite to free APIs
        df, src = fetch_real_daily(t, cache_dir=cache_dir, seed_dir=seed_dir)
        if not df.empty:
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            if len(df) >= min_rows:
                panels[t] = df
                sources[t] = src
                continue
            if len(df) >= 60:
                # accept thinner history rather than synthetic
                panels[t] = df
                sources[t] = f"{src}_short"
                continue
        sources[t] = "missing"
        logger.warning("No real data for %s (src tried yahoo/stooq/cache/seed)", t)

    return panels, sources


def build_cloud_feed(
    tickers: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
    seed_dir: Optional[Path] = None,
    lookback_calendar_days: int = 500,
    force_synthetic: bool = False,
    synthetic_seed: int = 42,
    require_real: bool = True,
    min_real_tickers: int = 5,
) -> Tuple[DailyReplayFeed, Dict[str, str]]:
    """Build feed from real free data.

    If require_real and fewer than min_real_tickers succeed, raises RuntimeError
    (unless force_synthetic).
    """
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
        seed_dir=seed_dir,
        lookback_calendar_days=lookback_calendar_days,
        allow_synthetic=False,
    )
    n_real = sum(1 for s in sources.values() if s not in ("missing", "synthetic", "synthetic_gapfill"))
    if require_real and n_real < min_real_tickers:
        raise RuntimeError(
            f"Insufficient real market data: {n_real}/{len(tickers)} tickers "
            f"(need ≥{min_real_tickers}). sources={sources}"
        )
    if not panels:
        raise RuntimeError(f"No panels loaded. sources={sources}")

    # Prefer real only — do not silent-fill missing with synthetic when require_real
    feed = DailyReplayFeed(panels, min_history=50)
    logger.info(
        "Cloud feed ready: %d tickers, sources=%s, days=%s..%s",
        len(panels),
        sources,
        feed.days[0] if feed.days else None,
        feed.days[-1] if feed.days else None,
    )
    return feed, sources


def download_seed_ohlcv(
    tickers: Sequence[str],
    seed_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Download real bars into seed_ohlcv/ for CI offline fallback."""
    seed = Path(seed_dir) if seed_dir else SEED_DIR
    seed.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}
    for i, t in enumerate(tickers):
        if i:
            time.sleep(0.4)
        df, src = fetch_real_daily(t.upper())
        if df.empty:
            out[t.upper()] = "fail"
            continue
        path = seed / f"{t.upper()}_history.csv"
        df.to_csv(path, index=False)
        out[t.upper()] = f"{src}:{len(df)}"
        logger.info("Seeded %s (%s, %d bars) -> %s", t, src, len(df), path)
    return out
