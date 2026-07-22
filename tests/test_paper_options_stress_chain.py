"""OPT-PR5 stress + OPT-PR6 Yahoo chain (mocked HTTP, no live network required)."""
from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.replay_options import run_options_strategy
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.stress import StressSpec, build_stressed_feed, inject_crash_into_panels
from paper_live.options.yahoo_chain import (
    YahooChainError,
    fetch_yahoo_option_chain,
    summarize_chain_vs_proxy,
)


def _panels(n: int = 80, start: date = date(2024, 1, 2), spot0: float = 400.0):
    rows = []
    px = spot0
    d = start
    for _ in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        px *= 1.001
        rows.append(
            {
                "date": pd.Timestamp(d, tz="UTC"),
                "open": px,
                "high": px * 1.01,
                "low": px * 0.99,
                "close": px,
                "volume": 1e6,
            }
        )
        d += timedelta(days=1)
    df = pd.DataFrame(rows)
    return {"SPY": df, "QQQ": df.copy()}


def test_inject_crash_drops_prices():
    panels = _panels()
    start = date(2024, 1, 2)
    end = date(2024, 4, 30)
    stressed, meta = inject_crash_into_panels(
        panels,
        start=start,
        end=end,
        tickers=["SPY"],
        stress=StressSpec(shock_pct=-0.30, n_days=15, start_offset_frac=0.3),
    )
    assert meta["data_label"] == "proxy_bs_stress"
    assert meta.get("crash_start")
    base = panels["SPY"]
    out = stressed["SPY"]
    # last bar in window should be lower than unstressed path scale
    b_last = float(base["close"].iloc[-1])
    o_last = float(out["close"].iloc[-1])
    assert o_last < b_last * 0.85


def test_inject_crash_spikes_vix_surface():
    """VIX panels must rise in crash (not follow equity mult downward)."""
    from paper_live.options.vol_surface import synthetic_vix_path

    n = 100
    panels = _panels(n=n)
    vix = synthetic_vix_path(n, level=18.0, seed=0, start="2024-01-02")
    panels["VIX"] = vix
    panels["VIX3M"] = vix.copy()
    start = date(2024, 1, 2)
    end = date(2024, 5, 15)
    stressed, meta = inject_crash_into_panels(
        panels,
        start=start,
        end=end,
        tickers=["SPY"],  # equity only in want — VIX still spiked
        stress=StressSpec(
            shock_pct=-0.30,
            n_days=15,
            start_offset_frac=0.3,
            vix_spike_mult=2.5,
            vix_floor=35.0,
        ),
    )
    assert "VIX" in meta.get("vix_tickers_spiked", [])
    # Find a crash day
    cs = meta.get("crash_start")
    assert cs
    crash_day = date.fromisoformat(cs)
    base_v = float(panels["VIX"].loc[panels["VIX"]["date"].dt.date == crash_day, "close"].iloc[0])
    out_v = float(stressed["VIX"].loc[stressed["VIX"]["date"].dt.date == crash_day, "close"].iloc[0])
    assert out_v >= max(base_v * 2.0, 35.0) - 1e-6
    # Equity still dropped
    assert float(stressed["SPY"]["close"].iloc[-1]) < float(panels["SPY"]["close"].iloc[-1]) * 0.9


def test_stressed_feed_replay_reports_label():
    panels = _panels(n=100)
    feed = DailyReplayFeed(panels)
    days = feed.days
    start, end = days[10], days[-1]
    sfeed, meta = build_stressed_feed(
        feed,
        start=start,
        end=end,
        tickers=["SPY"],
        stress=StressSpec(shock_pct=-0.30, n_days=10),
    )
    assert meta["shock_pct"] == -0.30
    spec = OptionStrategySpec(
        id="s_pcs",
        label="stress pcs",
        kind="put_credit_spread",
        underlying="SPY",
        dte_days=21,
        otm_pct=0.05,
        wing_otm_pct=0.12,
        max_portfolio_dd=0.40,
        max_single_day_drop=0.40,
        max_margin_fraction=0.40,
    )
    r = run_options_strategy(
        sfeed, spec, start=start, end=end, capital0=100_000.0, data_label="proxy_bs_stress"
    )
    assert r.data_label.startswith("proxy_bs_stress")
    assert r.iv_source in ("proxy_hv", "vix_surface", "vix_surface_partial")
    assert r.max_dd <= 0
    assert r.cvar_5pct is not None or r.days_run < 5


def _fake_yahoo_payload(spot: float = 500.0) -> bytes:
    # Fixed unix timestamp (UTC) for a far expiry — platform-independent
    exp = 1787270400
    payload = {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": spot},
                    "expirationDates": [exp],
                    "options": [
                        {
                            "expirationDate": exp,
                            "calls": [
                                {
                                    "contractSymbol": "SPY260821C00525000",
                                    "strike": 525.0,
                                    "expiration": exp,
                                    "bid": 2.0,
                                    "ask": 2.2,
                                    "lastPrice": 2.1,
                                    "volume": 10,
                                    "openInterest": 100,
                                    "impliedVolatility": 0.18,
                                    "inTheMoney": False,
                                }
                            ],
                            "puts": [
                                {
                                    "contractSymbol": "SPY260821P00475000",
                                    "strike": 475.0,
                                    "expiration": exp,
                                    "bid": 3.0,
                                    "ask": 3.3,
                                    "lastPrice": 3.1,
                                    "volume": 20,
                                    "openInterest": 200,
                                    "impliedVolatility": 0.22,
                                    "inTheMoney": False,
                                }
                            ],
                        }
                    ],
                }
            ],
            "error": None,
        }
    }
    return json.dumps(payload).encode("utf-8")


def test_yahoo_chain_parse_mocked():
    with patch("paper_live.options.yahoo_chain._http_get", return_value=_fake_yahoo_payload(500.0)):
        snap = fetch_yahoo_option_chain("SPY", raise_on_error=True)
    assert snap.ok
    assert snap.data_label == "yahoo_chain"
    assert snap.spot == pytest.approx(500.0)
    assert len(snap.puts) == 1
    assert snap.puts[0].mid == pytest.approx(3.15)
    summ = summarize_chain_vs_proxy(snap, otm_pct=0.05, side="put")
    assert summ["ok"] is True
    assert summ["data_label"] == "yahoo_chain"
    assert summ["nearest_strike"] == 475.0


def test_yahoo_chain_failure_not_fake():
    with patch("paper_live.options.yahoo_chain._http_get", side_effect=YahooChainError("rate limit")):
        snap = fetch_yahoo_option_chain("SPY", raise_on_error=False)
    assert snap.ok is False
    assert snap.data_label == "yahoo_chain_failed"
    assert not snap.calls and not snap.puts
    assert snap.error

    with patch("paper_live.options.yahoo_chain._http_get", side_effect=YahooChainError("rate limit")):
        with pytest.raises(YahooChainError):
            fetch_yahoo_option_chain("SPY", raise_on_error=True)


def test_yahoo_chain_bad_volume_still_parses():
    """Malformed volume/OI must not abort the whole chain (Issue 4)."""
    exp = 1787270400
    payload = {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": 500.0},
                    "expirationDates": [exp],
                    "options": [
                        {
                            "expirationDate": exp,
                            "calls": [],
                            "puts": [
                                {
                                    "contractSymbol": "SPY260821P00475000",
                                    "strike": 475.0,
                                    "expiration": exp,
                                    "bid": 3.0,
                                    "ask": 3.3,
                                    "lastPrice": 3.1,
                                    "volume": "N/A",
                                    "openInterest": {"bad": True},
                                    "impliedVolatility": 0.22,
                                    "inTheMoney": False,
                                }
                            ],
                        }
                    ],
                }
            ],
            "error": None,
        }
    }
    raw = json.dumps(payload).encode("utf-8")
    with patch("paper_live.options.yahoo_chain._http_get", return_value=raw):
        snap = fetch_yahoo_option_chain("SPY", raise_on_error=True)
    assert snap.ok
    assert snap.data_label == "yahoo_chain"
    assert len(snap.puts) == 1
    assert snap.puts[0].volume is None
    assert snap.puts[0].open_interest is None
    assert snap.puts[0].mid == pytest.approx(3.15)


def test_stress_post_crash_stays_depressed():
    """Bars after end of stress window remain at depressed mult (Issue 11)."""
    panels = _panels(n=100)
    start = date(2024, 1, 2)
    end = date(2024, 3, 29)
    stressed, meta = inject_crash_into_panels(
        panels,
        start=start,
        end=end,
        tickers=["SPY"],
        stress=StressSpec(shock_pct=-0.30, n_days=10, start_offset_frac=0.2),
    )
    base = panels["SPY"]
    out = stressed["SPY"]
    # last panel bar (may be after end) should be depressed vs base
    assert float(out["close"].iloc[-1]) < float(base["close"].iloc[-1]) * 0.85
