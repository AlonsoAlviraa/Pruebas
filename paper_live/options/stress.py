"""Synthetic crash stress for paper options marks (research only).

Injects a forced equity-path shock (~−30% month) into OHLCV used by proxy_bs
marks. Results are labeled with stress metadata — not real history.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed

# Vol-index panel keys (friendly + Yahoo caret) — stressed separately from equities
VOL_INDEX_KEYS = frozenset({"VIX", "VIX3M", "VXST", "^VIX", "^VIX3M", "^VXST"})


@dataclass
class StressSpec:
    """How to inject a synthetic crash into a replay window."""

    label: str = "crash_minus_30pct_month"
    shock_pct: float = -0.30
    """Total underlying shock over the stress path (e.g. -0.30)."""

    n_days: int = 20
    """Trading days over which to distribute the shock (approx 1 month)."""

    start_offset_frac: float = 0.45
    """Where in the window to start the crash (fraction of session days)."""

    vol_spike_mult: float = 2.5
    """Inflate high-low range during crash (affects HV if used from bars)."""

    vix_spike_mult: float = 2.5
    """Multiply VIX / VIX3M / VXST levels during crash window (vol surface stress)."""

    vix_floor: float = 35.0
    """Minimum VIX-like level during crash after spike (index points)."""

    notes: List[str] = field(
        default_factory=lambda: [
            "Synthetic crash stress on OHLCV; option marks remain proxy_bs.",
            "VIX/VIX3M panels spiked during crash (not left unshocked).",
            "Not a historical scenario replay of a real crisis tape.",
        ]
    )


def _session_days_sorted(feed: DailyReplayFeed, start: date, end: date) -> List[date]:
    return list(feed.session_days(start, end))


def inject_crash_into_panels(
    panels: Mapping[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    tickers: Optional[Sequence[str]] = None,
    stress: Optional[StressSpec] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Return deep-copied panels with a multi-day price path shock.

    Crash is timed inside [start, end]. From crash_start onward (including
    dates **after** ``end``), prices remain at the depressed level so a
    stressed feed reused past the window does not jump back to the original
    path. No recovery path is modeled (intentional severe stress).
    """
    st = stress or StressSpec()
    out: Dict[str, pd.DataFrame] = {}
    meta: Dict[str, Any] = {
        "label": st.label,
        "shock_pct": st.shock_pct,
        "n_days": st.n_days,
        "start_offset_frac": st.start_offset_frac,
        "vol_spike_mult": st.vol_spike_mult,
        "vix_spike_mult": st.vix_spike_mult,
        "vix_floor": st.vix_floor,
        "data_label": "proxy_bs_stress",
        "notes": list(st.notes),
        "tickers": [],
        "vix_tickers_spiked": [],
        "crash_start": None,
        "crash_end": None,
    }

    want = {t.upper() for t in tickers} if tickers else None

    # Build a common calendar from first available panel
    cal: List[date] = []
    for t, df in panels.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
        days = sorted({ts.date() for ts in d["date"].dropna()})
        cal = [x for x in days if start <= x <= end]
        if cal:
            break

    if not cal:
        for t, df in panels.items():
            out[t] = df.copy() if df is not None else df
        meta["error"] = "no calendar in window"
        return out, meta

    i0 = int(max(0, min(len(cal) - 1, round(len(cal) * st.start_offset_frac))))
    crash_days = cal[i0 : i0 + int(st.n_days)]
    if not crash_days:
        crash_days = cal[-min(len(cal), int(st.n_days)) :]
    meta["crash_start"] = crash_days[0].isoformat() if crash_days else None
    meta["crash_end"] = crash_days[-1].isoformat() if crash_days else None

    # Per-day multiplicative factors: compound to (1+shock)
    n = max(len(crash_days), 1)
    # equal log steps
    daily_factor = (1.0 + float(st.shock_pct)) ** (1.0 / n)
    day_mult: Dict[date, float] = {}
    cum = 1.0
    for d in crash_days:
        cum *= daily_factor
        day_mult[d] = cum
    final_mult = cum
    last_crash = crash_days[-1] if crash_days else None

    for t, df in panels.items():
        if df is None or df.empty:
            out[t] = df
            continue
        key = str(t).upper()
        # Always process vol-index panels when present (even if not in want),
        # so VIX surface spikes with equity stress.
        is_vol_index = key in VOL_INDEX_KEYS
        if want is not None and key not in want and not is_vol_index:
            out[key] = df.copy()
            continue

        d = df.copy()
        d.columns = [str(c).lower().strip() for c in d.columns]
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")

        if is_vol_index:
            # Spike VIX surface during crash; keep elevated after (no mean-revert).
            # Do NOT apply equity crash mult (that would drop VIX incorrectly).
            vix_m = max(float(st.vix_spike_mult), 1.0)
            vix_floor = float(st.vix_floor)
            closes = pd.to_numeric(d["close"], errors="coerce").astype(float)
            new_close = closes.copy()
            for i, ts in enumerate(d["date"]):
                if pd.isna(ts):
                    continue
                day = ts.date()
                if day in day_mult or (last_crash is not None and day > last_crash):
                    spiked = float(closes.iloc[i]) * vix_m
                    new_close.iloc[i] = max(spiked, vix_floor)
            d["close"] = new_close
            for col in ("open", "high", "low"):
                if col in d.columns:
                    d[col] = new_close
            if "high" in d.columns and "low" in d.columns:
                d["high"] = new_close * 1.02
                d["low"] = new_close * 0.98
            out[key] = d
            meta["tickers"].append(key)
            meta["vix_tickers_spiked"].append(key)
            continue

        mults = []
        for ts in d["date"]:
            if pd.isna(ts):
                mults.append(1.0)
                continue
            day = ts.date()
            if day in day_mult:
                mults.append(day_mult[day])
            elif last_crash is not None and day > last_crash:
                # Keep depressed level after crash (including beyond `end`)
                mults.append(final_mult)
            else:
                mults.append(1.0)
        m = np.asarray(mults, dtype=float)
        for col in ("open", "high", "low", "close"):
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce") * m
        # widen range during crash for vol proxies
        if st.vol_spike_mult and st.vol_spike_mult > 1.0:
            in_crash = np.array(
                [ts.date() in day_mult if not pd.isna(ts) else False for ts in d["date"]]
            )
            if "high" in d.columns and "low" in d.columns and "close" in d.columns:
                mid = d["close"].astype(float)
                half = (d["high"].astype(float) - d["low"].astype(float)) * 0.5 * float(
                    st.vol_spike_mult
                )
                d.loc[in_crash, "high"] = mid[in_crash] + half[in_crash]
                d.loc[in_crash, "low"] = mid[in_crash] - half[in_crash]
        out[key] = d
        meta["tickers"].append(key)

    return out, meta


def build_stressed_feed(
    base_feed: DailyReplayFeed,
    *,
    start: date,
    end: date,
    tickers: Optional[Sequence[str]] = None,
    stress: Optional[StressSpec] = None,
) -> Tuple[DailyReplayFeed, Dict[str, Any]]:
    """Clone a feed's panels via public ``raw_panels()``, inject crash, return feed."""
    if hasattr(base_feed, "raw_panels"):
        panels = base_feed.raw_panels()
    else:
        panels = getattr(base_feed, "_raw", None)
    if not panels:
        raise ValueError("base_feed has no panels to stress (raw_panels/_raw empty)")
    stressed, meta = inject_crash_into_panels(
        panels, start=start, end=end, tickers=tickers, stress=stress
    )
    meta.setdefault(
        "notes",
        [],
    )
    if isinstance(meta.get("notes"), list):
        meta["notes"].append(
            "Post-crash depression persists for all dates >= crash_end (no recovery jump)."
        )
    feed = DailyReplayFeed(stressed, min_history=getattr(base_feed, "min_history", 60))
    return feed, meta
