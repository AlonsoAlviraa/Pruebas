"""Universe construction: high-vol and fundamental quality screens.

All as-of cutoffs use only information available at/before the cutoff
(fundamentals use available_at lag — no look-ahead).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.features import list_tickers, load_history

logger = logging.getLogger(__name__)


@dataclass
class UniverseRow:
    ticker: str
    vol: float = np.nan
    rev_yoy: float = np.nan
    eps_yoy: float = np.nan
    quality_score: float = 0.0  # fundamental if available, else 0
    price_quality: float = 0.0  # causal multi-year sharpe/calmar proxy
    ret_36m: float = np.nan
    max_dd_3y: float = np.nan
    avg_dollar_vol: float = np.nan
    last_close: float = np.nan


def _realized_vol(close: pd.Series, window: int = 252) -> float:
    rets = close.pct_change().dropna()
    if len(rets) < max(60, window // 3):
        return float("nan")
    w = min(window, len(rets))
    return float(rets.iloc[-w:].std() * np.sqrt(252))


def _avg_dollar_vol(df: pd.DataFrame, window: int = 60) -> float:
    if df.empty or "close" not in df.columns or "volume" not in df.columns:
        return float("nan")
    d = df.tail(window)
    return float((d["close"] * d["volume"]).mean())


def _price_quality_metrics(close: pd.Series) -> Tuple[float, float, float]:
    """Causal 3y-ish quality: sharpe proxy, ret_36m, max_dd.

    Used when fundamental history is too short (common in local cache).
    """
    rets = close.pct_change().dropna()
    if len(rets) < 252:
        return 0.0, float("nan"), float("nan")
    w = min(len(rets), 756)  # ~3y trading days
    r = rets.iloc[-w:]
    mu = float(r.mean() * 252)
    sig = float(r.std() * np.sqrt(252))
    sharpe = mu / sig if sig > 1e-8 else 0.0
    ret_36m = float(close.iloc[-1] / close.iloc[-w] - 1.0)
    equity = (1.0 + r).cumprod()
    dd = float((equity / equity.cummax() - 1.0).min())
    # Combined: prefer high sharpe, positive multi-year return, shallow DD
    pq = sharpe + 0.5 * float(np.clip(ret_36m, -1, 2)) + abs(dd) * (-1.0)
    return float(pq), ret_36m, dd


def load_fundamentals_pit(ticker: str, data_root: Path) -> pd.DataFrame:
    p = data_root / f"{ticker}_fundamentals.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [c.lower().strip() for c in df.columns]
    if "available_at" not in df.columns:
        return pd.DataFrame()
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    if "as_of" in df.columns:
        df["as_of"] = pd.to_datetime(df["as_of"], utc=True, errors="coerce")
    for col in ("eps", "revenue"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["available_at"]).sort_values("available_at")


def fundamental_yoy_at(
    fund: pd.DataFrame,
    as_of: pd.Timestamp,
) -> Tuple[float, float]:
    """YoY revenue and EPS using only rows with available_at <= as_of.

    Compares latest row to the row ~4 quarters earlier (or nearest prior year).
    """
    if fund.empty:
        return float("nan"), float("nan")
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    hist = fund[fund["available_at"] <= as_of].copy()
    if len(hist) < 2:
        return float("nan"), float("nan")
    latest = hist.iloc[-1]
    # Prefer ~365d lookback on as_of if present else on available_at
    ref_col = "as_of" if "as_of" in hist.columns and hist["as_of"].notna().any() else "available_at"
    hist[ref_col] = pd.to_datetime(hist[ref_col], utc=True, errors="coerce")
    t1 = latest[ref_col]
    if pd.isna(t1):
        t1 = pd.to_datetime(latest["available_at"], utc=True)
    else:
        t1 = pd.to_datetime(t1, utc=True)
    target = t1 - pd.Timedelta(days=365)
    prior = hist[hist[ref_col] <= target + pd.Timedelta(days=45)]
    if prior.empty:
        # fall back: 4 rows earlier if quarterly
        if len(hist) >= 5:
            prior_row = hist.iloc[-5]
        else:
            return float("nan"), float("nan")
    else:
        prior_row = prior.iloc[-1]

    def yoy(cur, old):
        if pd.isna(cur) or pd.isna(old) or abs(float(old)) < 1e-12:
            return float("nan")
        return float(cur) / float(old) - 1.0

    rev = yoy(latest.get("revenue"), prior_row.get("revenue"))
    eps_cur, eps_old = latest.get("eps"), prior_row.get("eps")
    # EPS: if both negative, improvement if less negative
    if pd.notna(eps_cur) and pd.notna(eps_old):
        if float(eps_old) <= 0 and float(eps_cur) > float(eps_old):
            eps_g = 0.5  # synthetic positive for turnaround
        elif float(eps_old) > 0:
            eps_g = float(eps_cur) / float(eps_old) - 1.0
        else:
            eps_g = float("nan")
    else:
        eps_g = float("nan")
    return rev, eps_g


def quality_score(rev_yoy: float, eps_yoy: float) -> float:
    s = 0.0
    if pd.notna(rev_yoy):
        s += 1.0 if rev_yoy > 0 else (-0.5 if rev_yoy < -0.05 else 0.0)
        s += float(np.clip(rev_yoy, -0.5, 1.0))
    if pd.notna(eps_yoy):
        s += 1.0 if eps_yoy > 0 else (-0.5 if eps_yoy < -0.05 else 0.0)
        s += float(np.clip(eps_yoy, -0.5, 1.0))
    return s


def score_ticker(
    ticker: str,
    data_root: Path,
    as_of: pd.Timestamp,
    min_history: int = 400,
) -> Optional[UniverseRow]:
    hist = load_history(ticker, data_root)
    if hist.empty or len(hist) < min_history:
        return None
    if as_of.tzinfo is None:
        as_of = pd.Timestamp(as_of, tz="UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    h = hist[hist["date"] <= as_of]
    if len(h) < min_history:
        return None
    close = h["close"].astype(float)
    vol = _realized_vol(close, 252)
    adv = _avg_dollar_vol(h)
    last = float(close.iloc[-1])
    pq, ret36, mdd = _price_quality_metrics(close)
    fund = load_fundamentals_pit(ticker, data_root)
    rev_yoy, eps_yoy = fundamental_yoy_at(fund, as_of)
    q = quality_score(rev_yoy, eps_yoy)
    return UniverseRow(
        ticker=ticker,
        vol=vol,
        rev_yoy=rev_yoy,
        eps_yoy=eps_yoy,
        quality_score=q,
        price_quality=pq,
        ret_36m=ret36,
        max_dd_3y=mdd,
        avg_dollar_vol=adv,
        last_close=last,
    )


def build_scored_universe(
    data_root: Path,
    ticker_file: Path,
    as_of: str | pd.Timestamp = "2017-12-31",
    limit_scan: Optional[int] = None,
    min_price: float = 5.0,
    min_dollar_vol: float = 1_000_000.0,
) -> List[UniverseRow]:
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")
    tickers = list_tickers(ticker_file, data_root, limit=limit_scan)
    rows: List[UniverseRow] = []
    for i, t in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info("scoring %d/%d", i + 1, len(tickers))
        r = score_ticker(t, data_root, as_of_ts)
        if r is None:
            continue
        if r.last_close < min_price:
            continue
        if pd.notna(r.avg_dollar_vol) and r.avg_dollar_vol < min_dollar_vol:
            continue
        if pd.isna(r.vol) or r.vol <= 0:
            continue
        rows.append(r)
    return rows


def select_high_vol(rows: Sequence[UniverseRow], n: int = 80) -> List[str]:
    ranked = sorted(rows, key=lambda r: r.vol, reverse=True)
    return [r.ticker for r in ranked[:n]]


def select_quality_growth(rows: Sequence[UniverseRow], n: int = 80) -> List[str]:
    """Best quality: fundamentals YoY if PIT history exists, else price quality.

    Local fund CSVs often only cover ~2024+; for pre-OOS (2017) cutoffs we use
    multi-year price quality (sharpe/calmar/ret) as the 'better businesses' proxy.
    """
    with_fund = [
        r
        for r in rows
        if r.quality_score > 0 and (pd.notna(r.rev_yoy) or pd.notna(r.eps_yoy))
    ]
    if len(with_fund) >= max(n // 2, 20):
        ranked = sorted(with_fund, key=lambda r: r.quality_score, reverse=True)
        return [r.ticker for r in ranked[:n]]
    # Price quality: require positive multi-year return and not-terrible DD
    pool = [
        r
        for r in rows
        if pd.notna(r.ret_36m)
        and r.ret_36m > 0
        and (pd.isna(r.max_dd_3y) or r.max_dd_3y > -0.70)
    ]
    if len(pool) < n // 2:
        pool = list(rows)
    ranked = sorted(pool, key=lambda r: r.price_quality, reverse=True)
    return [r.ticker for r in ranked[:n]]


def select_quality_highvol_blend(rows: Sequence[UniverseRow], n: int = 80) -> List[str]:
    """Quality (fund or price) crossed with elevated vol."""
    q_names = set(select_quality_growth(rows, n=max(n * 2, 100)))
    pool = [r for r in rows if r.ticker in q_names]
    if len(pool) < n // 2:
        pool = list(rows)
    med_vol = float(np.median([r.vol for r in pool if pd.notna(r.vol)])) if pool else 0.0
    pool = [r for r in pool if pd.notna(r.vol) and r.vol >= med_vol * 0.9]
    ranked = sorted(
        pool,
        key=lambda r: r.vol * (1.0 + max(r.price_quality, 0) + max(r.quality_score, 0)),
        reverse=True,
    )
    return [r.ticker for r in ranked[:n]]


def write_ticker_file(path: Path, tickers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tickers) + "\n", encoding="utf-8")


def attach_fundamental_flags(
    df: pd.DataFrame,
    ticker: str,
    data_root: Path,
) -> pd.DataFrame:
    """Add causal fund flags/columns for strategy filters (per bar date)."""
    out = df.copy()
    fund = load_fundamentals_pit(ticker, data_root)
    out["fund_rev_yoy"] = np.nan
    out["fund_eps_yoy"] = np.nan
    out["fund_quality"] = 0.0
    if fund.empty or "date" not in out.columns:
        return out
    # For each unique year-month or daily: use latest available fund before date
    # Efficient: forward-fill YoY computed on fund available_at calendar
    rows = []
    for _, fr in fund.iterrows():
        rev, eps = fundamental_yoy_at(fund, fr["available_at"])
        rows.append(
            {
                "available_at": fr["available_at"],
                "fund_rev_yoy": rev,
                "fund_eps_yoy": eps,
                "fund_quality": quality_score(rev, eps),
            }
        )
    fdf = pd.DataFrame(rows).drop_duplicates("available_at").sort_values("available_at")
    if fdf.empty:
        return out
    dates = pd.to_datetime(out["date"], utc=True)
    # merge_asof
    left = pd.DataFrame({"date": dates}).sort_values("date")
    merged = pd.merge_asof(
        left,
        fdf.rename(columns={"available_at": "date"}),
        on="date",
        direction="backward",
    )
    out["fund_rev_yoy"] = merged["fund_rev_yoy"].to_numpy()
    out["fund_eps_yoy"] = merged["fund_eps_yoy"].to_numpy()
    out["fund_quality"] = merged["fund_quality"].fillna(0.0).to_numpy()
    return out
