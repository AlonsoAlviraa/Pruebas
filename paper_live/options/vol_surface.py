"""VIX / term-structure IV surface proxy for paper options marks.

LABELS (honest data quality):
  - ``vix_surface`` — IV built from VIX (± VIX3M) + mild tenor/skew
  - ``proxy_hv``    — fallback IV = HV × premium_mult (legacy)

Not exchange IVs. Not OPRA. Research proxy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from paper_live.options.vol_proxy import historical_vol, iv_proxy_from_hv

# Canonical feed keys + Yahoo caret symbols
VIX_TICKERS: tuple[str, ...] = ("VIX", "^VIX")
VIX3M_TICKERS: tuple[str, ...] = ("VIX3M", "^VIX3M")
# Optional short-vol proxy if present (Yahoo ^VXST is sparse)
VXST_TICKERS: tuple[str, ...] = ("VXST", "^VXST")


@dataclass(frozen=True)
class SurfaceIV:
    """Resolved IV for a single option mark."""

    iv: float
    source: str  # "vix_surface" | "proxy_hv"
    vix: Optional[float] = None
    vix3m: Optional[float] = None
    tenor_years: float = 0.0
    moneyness: float = 0.0  # log(K/S)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iv": self.iv,
            "source": self.source,
            "vix": self.vix,
            "vix3m": self.vix3m,
            "tenor_years": self.tenor_years,
            "moneyness": self.moneyness,
            "notes": self.notes,
        }


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def resolve_vix_level(
    feed: Any,
    day: date,
    *,
    aliases: Sequence[str] = VIX_TICKERS,
) -> Optional[float]:
    """Return VIX-like level in index points (e.g. 18.5) as of ``day``, or None."""
    for t in aliases:
        bar = None
        try:
            bar = feed.bar(t, day)
        except Exception:
            bar = None
        if bar is None:
            # try history last available through day (causal)
            try:
                hist = feed.history(t, through=day, include_through=True)
            except Exception:
                hist = None
            if hist is not None and not hist.empty:
                px = float(hist["close"].iloc[-1])
                if math.isfinite(px) and px > 0:
                    return px
            continue
        px = float(bar.close)
        if math.isfinite(px) and px > 0:
            return px
    return None


def vix_series_from_feed(
    feed: Any,
    through: date,
    *,
    aliases: Sequence[str] = VIX_TICKERS,
) -> pd.Series:
    """Causal close series for the first available VIX alias."""
    for t in aliases:
        try:
            hist = feed.history(t, through=through, include_through=True)
        except Exception:
            hist = None
        if hist is None or hist.empty:
            continue
        s = hist.set_index("date")["close"].astype(float)
        if not s.empty:
            return s
    return pd.Series(dtype=float)


def term_structure_base_vol(
    t_years: float,
    vix: float,
    vix3m: Optional[float] = None,
    *,
    vxst: Optional[float] = None,
) -> float:
    """
    Map tenor (years) → base ATM IV from VIX term structure.

    Anchors (index points → decimal vol):
      - ~9d  : VXST if available else slightly elevated VIX
      - ~30d : VIX / 100
      - ~90d : VIX3M / 100 (or mild flatten of VIX if missing)
    """
    v30 = float(vix) / 100.0
    if not math.isfinite(v30) or v30 <= 0:
        return float("nan")
    t_days = max(float(t_years) * 365.0, 0.0)
    v90 = float(vix3m) / 100.0 if vix3m is not None and math.isfinite(vix3m) and vix3m > 0 else None
    v9 = float(vxst) / 100.0 if vxst is not None and math.isfinite(vxst) and vxst > 0 else None

    if v9 is None:
        # short-end stub: slightly richer than 30d (pin / weekend risk proxy)
        v9 = v30 * 1.05

    if v90 is None:
        # mild contango default when VIX3M missing
        v90 = v30 * 0.97

    # Piecewise linear in calendar days between anchors 9 / 30 / 90 / 180+
    if t_days <= 9.0:
        # extrapolate flat-ish at short end
        w = t_days / 9.0 if t_days > 0 else 0.0
        base = v9 * (1.0 + 0.03 * (1.0 - w))
    elif t_days <= 30.0:
        w = (t_days - 9.0) / 21.0
        base = v9 * (1.0 - w) + v30 * w
    elif t_days <= 90.0:
        w = (t_days - 30.0) / 60.0
        base = v30 * (1.0 - w) + v90 * w
    else:
        # beyond 90d: gentle mean-reversion toward mid of VIX/VIX3M
        mid = 0.5 * (v30 + v90)
        w = min((t_days - 90.0) / 180.0, 1.0)
        base = v90 * (1.0 - w) + mid * w
    return float(base)


def apply_mild_skew(
    base_iv: float,
    *,
    spot: float,
    strike: float,
    option_type: str,
    put_slope: float = 0.40,
    call_slope: float = 0.15,
) -> tuple[float, float]:
    """
    Mild sticky-strike skew proxy.

    OTM puts (K < S) get richer IV; OTM calls slightly richer.
    ``put_slope`` / ``call_slope`` are vol points per unit log-moneyness.
    Returns (iv, log_moneyness).
    """
    if spot <= 0 or strike <= 0 or not math.isfinite(base_iv):
        return float(base_iv), 0.0
    m = math.log(float(strike) / float(spot))
    iv = float(base_iv)
    otype = (option_type or "call").lower()
    if otype == "put" and m < 0.0:
        # OTM put
        iv = iv + float(put_slope) * abs(m)
    elif otype == "call" and m > 0.0:
        iv = iv + float(call_slope) * m
    # ITM mild wing (symmetric damp)
    elif otype == "put" and m > 0.0:
        iv = iv + 0.25 * float(put_slope) * m
    elif otype == "call" and m < 0.0:
        iv = iv + 0.25 * float(call_slope) * abs(m)
    return iv, float(m)


def iv_from_surface(
    *,
    t_years: float,
    spot: float,
    strike: float,
    option_type: str = "put",
    vix: Optional[float] = None,
    vix3m: Optional[float] = None,
    vxst: Optional[float] = None,
    hv: Optional[float] = None,
    premium_mult: float = 1.15,
    floor: float = 0.08,
    cap: float = 1.5,
    put_skew_slope: float = 0.40,
    call_skew_slope: float = 0.15,
) -> SurfaceIV:
    """
    Build IV for one leg.

    Prefer VIX surface when ``vix`` is finite; else HV × premium_mult.
    """
    t = max(float(t_years), 0.0)
    if vix is not None and math.isfinite(float(vix)) and float(vix) > 0:
        base = term_structure_base_vol(t, float(vix), vix3m, vxst=vxst)
        iv, m = apply_mild_skew(
            base,
            spot=spot,
            strike=strike,
            option_type=option_type,
            put_slope=put_skew_slope,
            call_slope=call_skew_slope,
        )
        iv = _clamp(iv, floor, cap)
        return SurfaceIV(
            iv=iv,
            source="vix_surface",
            vix=float(vix),
            vix3m=float(vix3m) if vix3m is not None and math.isfinite(float(vix3m)) else None,
            tenor_years=t,
            moneyness=m,
            notes="IV from VIX term structure + mild skew proxy",
        )

    # Fallback: HV-based (explicit label)
    hv_use = float(hv) if hv is not None and math.isfinite(float(hv)) and float(hv) > 0 else float("nan")
    if not math.isfinite(hv_use):
        hv_use = 0.20
    iv = iv_proxy_from_hv(hv_use, premium_mult=premium_mult, floor=floor, cap=cap)
    if not math.isfinite(iv):
        iv = _clamp(hv_use * float(premium_mult), floor, cap)
    m = 0.0
    if spot > 0 and strike > 0:
        m = math.log(float(strike) / float(spot))
    return SurfaceIV(
        iv=float(iv),
        source="proxy_hv",
        vix=None,
        vix3m=None,
        tenor_years=t,
        moneyness=m,
        notes=f"IV fallback HV×{premium_mult} (no VIX)",
    )


def iv_for_mark(
    feed: Any,
    day: date,
    *,
    spot: float,
    strike: float,
    expiry: date,
    option_type: str,
    hv: Optional[float] = None,
    premium_mult: float = 1.15,
    put_skew_slope: float = 0.40,
    call_skew_slope: float = 0.15,
) -> SurfaceIV:
    """Convenience: pull VIX levels from feed (causal) and quote IV."""
    t_years = max((expiry - day).days, 0) / 365.0
    vix = resolve_vix_level(feed, day, aliases=VIX_TICKERS)
    vix3m = resolve_vix_level(feed, day, aliases=VIX3M_TICKERS)
    vxst = resolve_vix_level(feed, day, aliases=VXST_TICKERS)
    return iv_from_surface(
        t_years=t_years,
        spot=spot,
        strike=strike,
        option_type=option_type,
        vix=vix,
        vix3m=vix3m,
        vxst=vxst,
        hv=hv,
        premium_mult=premium_mult,
        put_skew_slope=put_skew_slope,
        call_skew_slope=call_skew_slope,
    )


def aggregate_surface_label(sources: Sequence[str]) -> str:
    """
    Collapse per-leg sources into a run-level data_label.

    - all vix_surface → vix_surface
    - all proxy_hv → proxy_hv
    - mixed → vix_surface_partial
    """
    sset = {str(s) for s in sources if s}
    if not sset:
        return "proxy_hv"
    if sset == {"vix_surface"}:
        return "vix_surface"
    if sset == {"proxy_hv"}:
        return "proxy_hv"
    if "vix_surface" in sset:
        return "vix_surface_partial"
    return "proxy_hv"


def synthetic_vix_path(
    n: int,
    *,
    level: float = 20.0,
    seed: int = 0,
    start: Union[str, date] = "2024-01-02",
) -> pd.DataFrame:
    """Synthetic VIX-like OHLCV for unit tests (no network)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n, tz="UTC")
    # mean-reverting log level
    x = math.log(max(level, 5.0))
    closes = []
    for _ in range(n):
        x = 0.95 * x + 0.05 * math.log(level) + rng.normal(0, 0.04)
        closes.append(math.exp(x))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.02,
            "low": c * 0.98,
            "close": c,
            "volume": np.full(n, 1.0),
        }
    )


# ---------------------------------------------------------------------------
# IV rank / VRP proxy (research labels — not true exchange VRP)
# ---------------------------------------------------------------------------


def series_percentile_rank(series: pd.Series, lookback: int = 252) -> Optional[float]:
    """Causal percentile of last value vs trailing ``lookback`` (0–1)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < max(10, lookback // 10):
        return None
    tail = s.iloc[-lookback:] if len(s) >= lookback else s
    cur = float(tail.iloc[-1])
    if not math.isfinite(cur):
        return None
    return float((tail <= cur).mean())


def iv_rank_from_vix(
    feed: Any,
    day: date,
    *,
    lookback: int = 252,
) -> Optional[float]:
    """
    Approximate IV rank = percentile of VIX close in trailing lookback.

    Label: research proxy from index VIX, **not** single-name IV rank.
    """
    ser = vix_series_from_feed(feed, day)
    if ser is None or ser.empty:
        return None
    # clip series through day
    try:
        ser = ser.loc[: pd.Timestamp(day)]
    except Exception:
        pass
    return series_percentile_rank(ser, lookback=lookback)


def iv_rank_from_hv(
    feed: Any,
    ticker: str,
    day: date,
    *,
    lookback: int = 252,
    hv_window: int = 20,
) -> Optional[float]:
    """Percentile rank of trailing HV20 for ``ticker`` (causal)."""
    try:
        hist = feed.history(ticker, through=day, include_through=True)
    except Exception:
        return None
    if hist is None or hist.empty or len(hist) < hv_window + 5:
        return None
    closes = hist.set_index("date")["close"].astype(float)
    rets = closes.pct_change()
    hv = rets.rolling(hv_window).std() * math.sqrt(252.0)
    hv = hv.dropna()
    return series_percentile_rank(hv, lookback=lookback)


def vrp_proxy(
    iv: float,
    hv: float,
) -> Optional[float]:
    """
    ``vrp_proxy = iv − HV`` (decimal vols).

    Label: ``vrp_proxy`` — **not** true variance risk premium from option markets.
    """
    if not math.isfinite(float(iv)) or not math.isfinite(float(hv)):
        return None
    return float(iv) - float(hv)


def atm_iv_proxy_for_day(
    *,
    vix: Optional[float],
    vix3m: Optional[float] = None,
    hv: Optional[float] = None,
    premium_mult: float = 1.15,
    t_years: float = 30.0 / 365.0,
) -> tuple[float, str]:
    """ATM-ish IV for gate logic (30d default tenor). Returns (iv, source)."""
    siv = iv_from_surface(
        t_years=t_years,
        spot=100.0,
        strike=100.0,
        option_type="put",
        vix=vix,
        vix3m=vix3m,
        hv=hv,
        premium_mult=premium_mult,
    )
    return float(siv.iv), str(siv.source)


def vix_term_contango(
    vix: Optional[float],
    vix3m: Optional[float],
    *,
    min_ratio: float = 1.0,
) -> Optional[bool]:
    """True when VIX3M / VIX ≥ min_ratio (contango). None if missing data."""
    if vix is None or vix3m is None:
        return None
    if not math.isfinite(float(vix)) or float(vix) <= 0:
        return None
    if not math.isfinite(float(vix3m)) or float(vix3m) <= 0:
        return None
    return float(vix3m) / float(vix) + 1e-12 >= float(min_ratio)
