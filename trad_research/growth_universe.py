"""PIT growth universe: double-digit quarterly EPS + 15%+ annual growth.

Design: docs/design/2026-07-24_eodhd_growth_universe_strategies.md
Gates G-Q / G-A are fail-closed; rank prefers highest growers among passers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.features import list_tickers, load_history
from trad_research.universe import load_fundamentals_pit, write_ticker_file

logger = logging.getLogger(__name__)


@dataclass
class GrowthGateConfig:
    min_eps_q_yoy: float = 0.10
    min_eps_ttm_yoy: float = 0.15
    min_rev_ttm_yoy: float = 0.15
    require_eps_ttm: bool = False
    # When TTM/annual missing (Yahoo free ~5Q), allow G-A via Q YoY ≥ min_eps_ttm_yoy
    allow_q_as_annual_fallback: bool = True
    allow_negative_base: bool = False
    min_price: float = 5.0
    min_adv: float = 2_000_000.0
    min_history_bars: int = 400
    min_quarters: int = 4
    lag_days: int = 45
    top_n: int = 80
    # rank weights
    w_eps_ttm: float = 0.50
    w_eps_q: float = 0.30
    w_rev_ttm: float = 0.20


@dataclass
class GrowthMetrics:
    ticker: str
    eps_q_yoy: float = float("nan")
    eps_ttm_yoy: float = float("nan")
    rev_ttm_yoy: float = float("nan")
    rev_q_yoy: float = float("nan")
    last_close: float = float("nan")
    avg_dollar_vol: float = float("nan")
    n_quarters: int = 0
    pass_gq: bool = False
    pass_ga: bool = False
    pass_liq: bool = False
    pass_all: bool = False
    growth_rank_score: float = float("nan")
    fail_reason: str = ""


def _asof_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _yoy(cur: float, old: float, *, allow_neg_base: bool = False) -> float:
    if cur != cur or old != old:
        return float("nan")
    if abs(float(old)) < 1e-12:
        return float("nan")
    if float(old) <= 0 and not allow_neg_base:
        return float("nan")
    if float(old) <= 0 and allow_neg_base:
        if float(cur) > float(old):
            return 0.5
        return float("nan")
    return float(cur) / float(old) - 1.0


def _ttm_sum(series: pd.Series, end_idx: int) -> float:
    """Sum last 4 available values ending at end_idx (inclusive)."""
    if end_idx < 3:
        return float("nan")
    window = series.iloc[end_idx - 3 : end_idx + 1]
    vals = pd.to_numeric(window, errors="coerce")
    if vals.isna().any():
        return float("nan")
    return float(vals.sum())


def growth_metrics_from_fund(
    fund: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    cfg: Optional[GrowthGateConfig] = None,
) -> Dict[str, float]:
    """Compute Q and TTM/annual YoY metrics using only rows with available_at <= as_of."""
    cfg = cfg or GrowthGateConfig()
    as_of = _asof_utc(as_of)
    empty = {
        "eps_q_yoy": float("nan"),
        "eps_ttm_yoy": float("nan"),
        "rev_ttm_yoy": float("nan"),
        "rev_q_yoy": float("nan"),
        "n_quarters": 0,
    }
    if fund is None or fund.empty:
        return empty
    df = fund.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    if "available_at" not in df.columns:
        return empty
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    if "as_of" in df.columns:
        df["as_of"] = pd.to_datetime(df["as_of"], utc=True, errors="coerce")
    for col in ("eps", "revenue", "net_income"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    hist_all = df[df["available_at"] <= as_of].copy()
    if hist_all.empty:
        return empty

    # Split quarterly vs annual (period column optional; default treat all as Q)
    if "period" in hist_all.columns:
        period = hist_all["period"].astype(str).str.upper()
        hist = hist_all[period.str.startswith("Q") | (period == "nan") | (period == "NONE")].copy()
        annual = hist_all[period.str.startswith("A")].copy()
    else:
        hist = hist_all
        annual = hist_all.iloc[0:0].copy()

    sort_col = "as_of" if "as_of" in hist.columns else "available_at"
    hist = hist.sort_values(sort_col)
    out = dict(empty)
    nq = len(hist)
    out["n_quarters"] = nq
    if nq < 2 and annual.empty:
        return out

    allow = bool(cfg.allow_negative_base)

    if nq >= 2:
        latest = hist.iloc[-1]
        if nq >= 5:
            prior_q = hist.iloc[-5]
        else:
            ref = "as_of" if "as_of" in hist.columns else "available_at"
            t1 = pd.to_datetime(latest[ref], utc=True, errors="coerce")
            if pd.isna(t1):
                prior_q = None
            else:
                target = t1 - pd.Timedelta(days=365)
                prior = hist[pd.to_datetime(hist[ref], utc=True) <= target + pd.Timedelta(days=45)]
                prior_q = prior.iloc[-1] if not prior.empty else None
        if prior_q is not None:
            if "eps" in hist.columns:
                out["eps_q_yoy"] = _yoy(
                    float(latest.get("eps", np.nan)),
                    float(prior_q.get("eps", np.nan)),
                    allow_neg_base=allow,
                )
            if "revenue" in hist.columns:
                out["rev_q_yoy"] = _yoy(
                    float(latest.get("revenue", np.nan)),
                    float(prior_q.get("revenue", np.nan)),
                    allow_neg_base=True,
                )

        # TTM YoY needs 8 quarters ideally
        if nq >= 8 and "eps" in hist.columns:
            ttm_now = _ttm_sum(hist["eps"], nq - 1)
            ttm_old = _ttm_sum(hist["eps"], nq - 5)
            out["eps_ttm_yoy"] = _yoy(ttm_now, ttm_old, allow_neg_base=allow)
        if nq >= 8 and "revenue" in hist.columns:
            ttm_now = _ttm_sum(hist["revenue"], nq - 1)
            ttm_old = _ttm_sum(hist["revenue"], nq - 5)
            out["rev_ttm_yoy"] = _yoy(ttm_now, ttm_old, allow_neg_base=True)

    # Annual YoY fallback for G-A when TTM missing (Yahoo free depth ~4 annuals)
    if annual is not None and len(annual) >= 2:
        annual = annual.sort_values(sort_col if sort_col in annual.columns else "available_at")
        a_last = annual.iloc[-1]
        a_prev = annual.iloc[-2]
        if out["eps_ttm_yoy"] != out["eps_ttm_yoy"] and "eps" in annual.columns:
            out["eps_ttm_yoy"] = _yoy(
                float(a_last.get("eps", np.nan)),
                float(a_prev.get("eps", np.nan)),
                allow_neg_base=allow,
            )
        if out["rev_ttm_yoy"] != out["rev_ttm_yoy"] and "revenue" in annual.columns:
            out["rev_ttm_yoy"] = _yoy(
                float(a_last.get("revenue", np.nan)),
                float(a_prev.get("revenue", np.nan)),
                allow_neg_base=True,
            )
    return out


def passes_growth_gates(
    metrics: Dict[str, float],
    *,
    cfg: Optional[GrowthGateConfig] = None,
) -> Tuple[bool, bool, str]:
    """Return (pass_gq, pass_ga, fail_reason)."""
    cfg = cfg or GrowthGateConfig()
    eps_q = metrics.get("eps_q_yoy", float("nan"))
    eps_ttm = metrics.get("eps_ttm_yoy", float("nan"))
    rev_ttm = metrics.get("rev_ttm_yoy", float("nan"))

    if eps_q != eps_q:
        return False, False, "missing_eps_q_yoy"
    if float(eps_q) < float(cfg.min_eps_q_yoy):
        return False, False, "gq_fail"

    pass_gq = True
    # G-A: prefer EPS TTM; fallback revenue TTM
    if eps_ttm == eps_ttm:
        pass_ga = float(eps_ttm) >= float(cfg.min_eps_ttm_yoy)
        if not pass_ga:
            return pass_gq, False, "ga_eps_ttm_fail"
        return pass_gq, True, ""
    if cfg.require_eps_ttm:
        return pass_gq, False, "missing_eps_ttm"
    if rev_ttm == rev_ttm:
        pass_ga = float(rev_ttm) >= float(cfg.min_rev_ttm_yoy)
        if not pass_ga:
            return pass_gq, False, "ga_rev_ttm_fail"
        return pass_gq, True, ""
    # Thin-history fallback: quarterly YoY already ≥ annual threshold
    if cfg.allow_q_as_annual_fallback and eps_q == eps_q:
        if float(eps_q) >= float(cfg.min_eps_ttm_yoy):
            return pass_gq, True, ""
        return pass_gq, False, "ga_q_fallback_fail"
    return pass_gq, False, "missing_annual_growth"


def _cs_rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return out
    # average rank → 0..1
    order = arr[mask].argsort().argsort().astype(float)
    if len(order) == 1:
        ranks = np.array([1.0])
    else:
        ranks = order / (len(order) - 1)
    out[mask] = ranks
    return out


def rank_growth_passers(
    rows: Sequence[GrowthMetrics],
    *,
    cfg: Optional[GrowthGateConfig] = None,
) -> List[GrowthMetrics]:
    """Assign growth_rank_score and sort descending among pass_all."""
    cfg = cfg or GrowthGateConfig()
    passers = [r for r in rows if r.pass_all]
    if not passers:
        return []
    r_eps_ttm = _cs_rank([r.eps_ttm_yoy for r in passers])
    r_eps_q = _cs_rank([r.eps_q_yoy for r in passers])
    r_rev = _cs_rank([r.rev_ttm_yoy for r in passers])
    scored: List[GrowthMetrics] = []
    for i, r in enumerate(passers):
        parts = []
        wsum = 0.0
        if r_eps_ttm[i] == r_eps_ttm[i]:
            parts.append(cfg.w_eps_ttm * float(r_eps_ttm[i]))
            wsum += cfg.w_eps_ttm
        if r_eps_q[i] == r_eps_q[i]:
            parts.append(cfg.w_eps_q * float(r_eps_q[i]))
            wsum += cfg.w_eps_q
        if r_rev[i] == r_rev[i]:
            parts.append(cfg.w_rev_ttm * float(r_rev[i]))
            wsum += cfg.w_rev_ttm
        score = float(sum(parts) / wsum) if wsum > 0 else 0.0
        r2 = GrowthMetrics(**{**r.__dict__, "growth_rank_score": score})
        scored.append(r2)
    scored.sort(key=lambda x: float(x.growth_rank_score), reverse=True)
    return scored


def score_growth_ticker(
    ticker: str,
    data_root: Path,
    as_of: pd.Timestamp | str,
    *,
    cfg: Optional[GrowthGateConfig] = None,
    fund_root: Optional[Path] = None,
) -> Optional[GrowthMetrics]:
    cfg = cfg or GrowthGateConfig()
    as_of = _asof_utc(as_of)
    data_root = Path(data_root)
    fund_root = Path(fund_root) if fund_root else data_root

    hist = load_history(ticker, data_root)
    if hist.empty or len(hist) < cfg.min_history_bars:
        return GrowthMetrics(ticker=ticker, fail_reason="thin_price_history")
    h = hist[hist["date"] <= as_of]
    if len(h) < cfg.min_history_bars:
        return GrowthMetrics(ticker=ticker, fail_reason="thin_price_asof")
    close = h["close"].astype(float)
    last = float(close.iloc[-1])
    adv = float((h["close"].astype(float) * h["volume"].astype(float)).tail(60).mean())

    # Prefer EODHD-normalized fund in fund_root or standard path
    fund = load_fundamentals_pit(ticker, fund_root)
    if fund.empty and fund_root != data_root:
        fund = load_fundamentals_pit(ticker, data_root)
    metrics = growth_metrics_from_fund(fund, as_of, cfg=cfg)
    gq, ga, reason = passes_growth_gates(metrics, cfg=cfg)
    liq = last >= cfg.min_price and (adv != adv or adv >= cfg.min_adv)
    pass_all = gq and ga and liq and int(metrics.get("n_quarters", 0)) >= cfg.min_quarters
    fail = ""
    if not liq:
        fail = "liq_fail"
    elif not pass_all:
        fail = reason or "gate_fail"
    return GrowthMetrics(
        ticker=ticker.upper(),
        eps_q_yoy=float(metrics["eps_q_yoy"]),
        eps_ttm_yoy=float(metrics["eps_ttm_yoy"]),
        rev_ttm_yoy=float(metrics["rev_ttm_yoy"]),
        rev_q_yoy=float(metrics["rev_q_yoy"]),
        last_close=last,
        avg_dollar_vol=adv,
        n_quarters=int(metrics["n_quarters"]),
        pass_gq=gq,
        pass_ga=ga,
        pass_liq=liq,
        pass_all=pass_all,
        fail_reason=fail,
    )


def build_growth_universe(
    data_root: Path,
    ticker_file: Path,
    as_of: str | pd.Timestamp,
    *,
    cfg: Optional[GrowthGateConfig] = None,
    limit_scan: Optional[int] = None,
    fund_root: Optional[Path] = None,
) -> Tuple[List[str], List[GrowthMetrics]]:
    """Scan tickers, apply gates, return top-N names + full metric rows."""
    cfg = cfg or GrowthGateConfig()
    data_root = Path(data_root)
    tickers = list_tickers(Path(ticker_file), data_root, limit=limit_scan)
    rows: List[GrowthMetrics] = []
    for i, t in enumerate(tickers):
        if (i + 1) % 100 == 0:
            logger.info("growth score %d/%d", i + 1, len(tickers))
        r = score_growth_ticker(t, data_root, as_of, cfg=cfg, fund_root=fund_root)
        if r is not None:
            rows.append(r)
    ranked = rank_growth_passers(rows, cfg=cfg)
    top = [r.ticker for r in ranked[: int(cfg.top_n)]]
    return top, rows


def build_growth_universe_yearly(
    data_root: Path,
    ticker_file: Path,
    years: Sequence[int],
    *,
    cfg: Optional[GrowthGateConfig] = None,
    out_dir: Optional[Path] = None,
    fund_root: Optional[Path] = None,
    limit_scan: Optional[int] = None,
) -> Dict[int, List[str]]:
    """For each OOS year Y, build L0 as-of (Y-1)-12-31."""
    cfg = cfg or GrowthGateConfig()
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    by_year: Dict[int, List[str]] = {}
    for y in years:
        as_of = f"{int(y) - 1}-12-31"
        top, rows = build_growth_universe(
            data_root,
            ticker_file,
            as_of,
            cfg=cfg,
            limit_scan=limit_scan,
            fund_root=fund_root,
        )
        by_year[int(y)] = top
        n_pass = sum(1 for r in rows if r.pass_all)
        logger.info("growth L0 year=%s as_of=%s pass=%d top=%d", y, as_of, n_pass, len(top))
        if out_dir:
            write_ticker_file(out_dir / f"universe_growth_top{cfg.top_n}_{y}.txt", top)
            # compact metrics dump
            recs = [r.__dict__ for r in rows if r.pass_all]
            if recs:
                pd.DataFrame(recs).to_csv(out_dir / f"growth_passers_{y}.csv", index=False)
    return by_year
