"""Unit tests: VIX surface proxy + HV fallback (no network)."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.vol_surface import (
    aggregate_surface_label,
    apply_mild_skew,
    iv_for_mark,
    iv_from_surface,
    resolve_vix_level,
    synthetic_vix_path,
    term_structure_base_vol,
)


def test_term_structure_short_richer_than_long_when_vix_gt_vix3m():
    vix, vix3m = 25.0, 22.0
    short = term_structure_base_vol(10 / 365.0, vix, vix3m)
    mid = term_structure_base_vol(30 / 365.0, vix, vix3m)
    long = term_structure_base_vol(90 / 365.0, vix, vix3m)
    assert short > mid > long * 0.95
    assert abs(mid - 0.25) < 0.03


def test_iv_from_surface_vix_label():
    q = iv_from_surface(
        t_years=30 / 365.0,
        spot=100.0,
        strike=95.0,
        option_type="put",
        vix=20.0,
        vix3m=19.0,
    )
    assert q.source == "vix_surface"
    assert 0.10 < q.iv < 0.40
    assert q.vix == 20.0


def test_iv_from_surface_hv_fallback_label():
    q = iv_from_surface(
        t_years=0.1,
        spot=100.0,
        strike=100.0,
        option_type="call",
        vix=None,
        hv=0.18,
        premium_mult=1.15,
    )
    assert q.source == "proxy_hv"
    assert abs(q.iv - 0.18 * 1.15) < 1e-9


def test_otm_put_richer_than_atm():
    atm, _ = apply_mild_skew(0.20, spot=100, strike=100, option_type="put")
    otm, m = apply_mild_skew(0.20, spot=100, strike=90, option_type="put")
    assert m < 0
    assert otm > atm


def test_aggregate_surface_label():
    assert aggregate_surface_label(["vix_surface", "vix_surface"]) == "vix_surface"
    assert aggregate_surface_label(["proxy_hv"]) == "proxy_hv"
    assert aggregate_surface_label(["vix_surface", "proxy_hv"]) == "vix_surface_partial"


def test_resolve_vix_from_synthetic_feed():
    n = 80
    spy_dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    px = 400 * np.cumprod(1 + np.full(n, 0.0005))
    spy = pd.DataFrame(
        {
            "date": spy_dates,
            "open": px,
            "high": px * 1.01,
            "low": px * 0.99,
            "close": px,
            "volume": 1e6,
        }
    )
    vix = synthetic_vix_path(n, level=18.0, seed=1, start="2024-01-02")
    feed = DailyReplayFeed({"SPY": spy, "VIX": vix})
    day = feed.days[40]
    lvl = resolve_vix_level(feed, day)
    assert lvl is not None and 5 < lvl < 80

    exp = day + timedelta(days=30)
    q = iv_for_mark(
        feed,
        day,
        spot=float(feed.bar("SPY", day).close),
        strike=float(feed.bar("SPY", day).close) * 0.95,
        expiry=exp,
        option_type="put",
        hv=0.15,
    )
    assert q.source == "vix_surface"
    assert math.isfinite(q.iv)


def test_missing_vix_uses_proxy_hv_in_mark():
    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    px = 100 * np.cumprod(1 + np.full(n, 0.001))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": px,
            "high": px * 1.01,
            "low": px * 0.99,
            "close": px,
            "volume": 1e6,
        }
    )
    feed = DailyReplayFeed({"SPY": df})
    day = feed.days[-1]
    q = iv_for_mark(
        feed,
        day,
        spot=float(feed.bar("SPY", day).close),
        strike=100.0,
        expiry=day + timedelta(days=30),
        option_type="call",
        hv=0.22,
        premium_mult=1.15,
    )
    assert q.source == "proxy_hv"
