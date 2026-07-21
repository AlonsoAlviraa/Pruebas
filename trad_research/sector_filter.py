"""Sector-ETF trend gate: skip entries when the stock's sector ETF is below MAs.

Causal: only uses sector ETF OHLCV ≤ decision date (rolling SMA on past+today).
If a ticker has no sector map, behaviour is configurable (default: allow).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.features import load_history

logger = logging.getLogger(__name__)

# SPDR sector ETFs (US). Used as sector proxies.
SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Communication": "XLC",
}

DEFAULT_SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC")


def normalize_sector_name(raw: str) -> Optional[str]:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s in SECTOR_ETFS:
        return s
    # fuzzy light
    low = s.lower()
    for k in SECTOR_ETFS:
        if k.lower() == low or k.lower() in low or low in k.lower():
            return k
    return s  # unknown — may still map if we add later


def sector_to_etf(sector: str) -> Optional[str]:
    key = normalize_sector_name(sector)
    if key is None:
        return None
    return SECTOR_ETFS.get(key)


def load_ticker_sector_map(path: Path) -> Dict[str, str]:
    """ticker -> sector name. CSV: ticker,sector  or JSON {ticker: sector}."""
    if not path.is_file():
        return {}
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k).upper(): str(v) for k, v in raw.items() if v}
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol") or list(df.columns)[0]
    scol = cols.get("sector") or cols.get("gics_sector") or list(df.columns)[1]
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).strip().upper()
        s = str(row[scol]).strip() if pd.notna(row[scol]) else ""
        if t and s and s.lower() not in ("nan", "none", ""):
            out[t] = s
    return out


def build_etf_trend_map(
    data_root: Path,
    etf: str,
    *,
    ma: int = 50,
    require_sma200: bool = False,
) -> Dict[pd.Timestamp, bool]:
    """date -> True if ETF close > SMA(ma) [and optionally > SMA200]."""
    hist = load_history(etf, data_root)
    if hist.empty or len(hist) < max(ma, 60):
        logger.warning("Sector ETF %s missing/short — no sector gate for it", etf)
        return {}
    close = hist["close"].astype(float)
    sma = close.rolling(ma, min_periods=max(10, ma // 2)).mean()
    ok = close > sma
    if require_sma200:
        sma200 = close.rolling(200, min_periods=100).mean()
        ok = ok & (close > sma200)
    dates = pd.to_datetime(hist["date"], utc=True)
    return {d: bool(f) for d, f in zip(dates, ok.fillna(False))}


def build_all_sector_etf_maps(
    data_root: Path,
    *,
    ma: int = 50,
    require_sma200: bool = False,
    etfs: Sequence[str] = DEFAULT_SECTOR_ETFS,
) -> Dict[str, Dict[pd.Timestamp, bool]]:
    out: Dict[str, Dict[pd.Timestamp, bool]] = {}
    for e in etfs:
        m = build_etf_trend_map(data_root, e, ma=ma, require_sma200=require_sma200)
        if m:
            out[e] = m
    return out


def _flag_on_day(day: pd.Timestamp, m: Dict[pd.Timestamp, bool], default: bool = True) -> bool:
    if not m:
        return default
    if day in m:
        return bool(m[day])
    prior = [d for d in m if d <= day]
    if not prior:
        return default
    return bool(m[max(prior)])


def sector_allows_entry(
    ticker: str,
    day: pd.Timestamp,
    *,
    ticker_sector: Dict[str, str],
    etf_maps: Dict[str, Dict[pd.Timestamp, bool]],
    allow_if_unmapped: bool = True,
) -> bool:
    """True if ticker may enter on day given sector ETF trend."""
    sec = ticker_sector.get(ticker.upper()) or ticker_sector.get(ticker)
    if not sec:
        return allow_if_unmapped
    etf = sector_to_etf(sec)
    if not etf:
        return allow_if_unmapped
    m = etf_maps.get(etf)
    if not m:
        return allow_if_unmapped
    return _flag_on_day(day, m, default=True)


def summarize_sector_coverage(
    tickers: Sequence[str],
    ticker_sector: Dict[str, str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tickers:
        sec = ticker_sector.get(t.upper(), "UNMAPPED")
        etf = sector_to_etf(sec) if sec != "UNMAPPED" else "UNMAPPED"
        key = f"{sec}->{etf}" if etf else f"{sec}->?"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def assign_sector_by_etf_correlation(
    data_root: Path,
    tickers: Sequence[str],
    *,
    etfs: Sequence[str] = DEFAULT_SECTOR_ETFS,
    lookback: int = 252,
    min_overlap: int = 120,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, str]:
    """Map ticker → sector ETF via max corr of daily returns (causal if as_of set).

    Used when fundamentals API is unavailable. Sector label stored as the ETF
    code itself (XLK, XLF, …); sector_to_etf identity-maps these.
    """
    # Load ETF returns
    etf_rets: Dict[str, pd.Series] = {}
    for e in etfs:
        h = load_history(e, data_root)
        if h.empty:
            continue
        h = h.copy()
        h["date"] = pd.to_datetime(h["date"], utc=True)
        if as_of is not None:
            a = as_of if as_of.tzinfo else pd.Timestamp(as_of, tz="UTC")
            h = h[h["date"] <= a]
        if len(h) < min_overlap:
            continue
        s = h.set_index("date")["close"].astype(float).pct_change().dropna()
        if len(s) >= min_overlap:
            etf_rets[e] = s.iloc[-lookback:]
    if not etf_rets:
        logger.warning("No sector ETF returns for correlation map")
        return {}

    out: Dict[str, str] = {}
    for i, t in enumerate(tickers, 1):
        h = load_history(t, data_root)
        if h.empty:
            continue
        h = h.copy()
        h["date"] = pd.to_datetime(h["date"], utc=True)
        if as_of is not None:
            a = as_of if as_of.tzinfo else pd.Timestamp(as_of, tz="UTC")
            h = h[h["date"] <= a]
        r = h.set_index("date")["close"].astype(float).pct_change().dropna()
        if len(r) < min_overlap:
            continue
        r = r.iloc[-lookback:]
        best_e, best_c = None, -1.0
        for e, er in etf_rets.items():
            aligned = pd.concat([r, er], axis=1, join="inner").dropna()
            if len(aligned) < min_overlap // 2:
                continue
            c = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if np.isfinite(c) and c > best_c:
                best_c, best_e = c, e
        if best_e is not None and best_c > 0.15:
            out[t.upper()] = best_e  # sector label = ETF code
        if i % 200 == 0:
            logger.info("corr-sector map %d/%d assigned=%d", i, len(tickers), len(out))
    return out


# Allow sector label to be the ETF ticker itself (corr-based map)
SECTOR_ETFS.update({e: e for e in DEFAULT_SECTOR_ETFS})
