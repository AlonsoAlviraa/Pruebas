"""LIV-03/LIV-04: daily replay feed + signal → entry → broker fills."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.freeze import load_freeze
from paper_live.ledger import EventType, PaperLedger
from paper_live.replay_session import ReplaySession
from paper_live.signals.daily_pipeline import DailySignalPipeline, default_rule_signal_row
from paper_live.signals.entry_confirm import confirm_entry
from paper_live.signals.daily_pipeline import EntryCandidate
from paper_live.datafeed.base import Bar
from datetime import datetime, timezone


def _bull_panel(ticker: str, n: int = 280, start: str = "2020-01-02", drift: float = 0.003):
    """Trending OHLCV with enough range for ATR filters to pass."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
    dates = pd.bdate_range(start=start, periods=n, tz="UTC")
    noise = rng.normal(0, 0.008, size=n)
    rets = np.full(n, drift) + noise
    close = 20.0 * np.cumprod(1.0 + rets)
    open_ = np.r_[close[0], close[:-1]]
    # ~2–3% daily range so atr_norm stays above min filter
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.01, 0.025, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.01, 0.025, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def test_replay_feed_causal_history():
    feed = DailyReplayFeed.from_synthetic(["AAA", "QQQ"], n_days=120, seed=1)
    days = feed.days
    assert len(days) >= 100
    mid = days[80]
    hist = feed.history("AAA", through=mid)
    assert hist["date"].max().date() <= mid
    # no future bars
    future = days[90]
    assert hist["date"].max().date() < future
    feat = feed.featured("AAA", through=mid)
    assert not feat.empty
    assert "sma_50" in feat.columns
    bar = feed.bar("AAA", mid)
    assert bar is not None
    assert bar.close > 0
    assert feed.next_session(mid) > mid


def test_rule_signal_and_confirm():
    # Build a clearly trending series
    feed = DailyReplayFeed({"AAA": _bull_panel("AAA"), "QQQ": _bull_panel("QQQ", drift=0.002)})
    pipe = DailySignalPipeline(feed, universe=["AAA"], require_regime=True, regime_symbol="QQQ")
    d = feed.days[-5]
    batch = pipe.generate(d)
    assert batch.signal_date == d
    # Strong bull should produce candidates when regime on
    if batch.regime_on:
        assert len(batch.candidates) >= 1
        c = batch.candidates[0]
        assert c.score > 0
        nxt = feed.next_session(d)
        assert nxt is not None
        conf = confirm_entry(c, feed.bar(c.ticker, nxt), min_price=1.0)
        assert conf.ok
        assert conf.entry_px_ref > 0

    # gap up chase reject
    c = EntryCandidate(
        ticker="AAA",
        signal_date=d,
        score=1.0,
        p_buy=0.8,
        close=100.0,
        atr=2.0,
        atr_norm=0.02,
    )
    gap_bar = Bar(
        ticker="AAA",
        ts=datetime(2021, 1, 4, tzinfo=timezone.utc),
        open=120.0,
        high=121.0,
        low=119.0,
        close=120.0,
    )
    bad = confirm_entry(c, gap_bar, max_gap_pct=0.08)
    assert not bad.ok
    assert bad.reason == "gap_up_chase"


def test_default_rule_filters_downtrend():
    row = pd.Series(
        {
            "close": 10.0,
            "sma_50": 12.0,
            "sma_200": 15.0,
            "ret_1m": -0.1,
            "atr_norm": 0.05,
            "volatility_20": 0.2,
        }
    )
    assert default_rule_signal_row(row) is None


def test_replay_session_end_to_end(tmp_path: Path):
    # Index + 3 names with strong drift so rules fire
    panels = {
        "QQQ": _bull_panel("QQQ", n=300, drift=0.0025),
        "AAA": _bull_panel("AAA", n=300, drift=0.004),
        "BBB": _bull_panel("BBB", n=300, drift=0.0035),
        "CCC": _bull_panel("CCC", n=300, drift=0.003),
    }
    feed = DailyReplayFeed(panels, min_history=50)
    freeze = load_freeze()
    led = PaperLedger.create_run(tmp_path / "replay", freeze, run_id="replay_e2e_1")
    session = ReplaySession(feed, freeze, ledger=led, max_entries_per_day=3)
    session.pipeline.require_regime = True
    # Use middle window with enough warmup features
    days = feed.days
    start, end = days[220], days[260]
    result = session.run(start, end)
    assert result.days_run > 20
    assert result.n_signals >= result.days_run  # signal each day + seed
    # Should trade something in a strong bull tape
    assert result.n_entries >= 1
    assert result.total_commission > 0
    assert result.final_equity > 0
    # Ledger events
    assert led.list_events(event_type=EventType.SESSION_OPEN)
    assert led.list_events(event_type=EventType.SIGNAL_COMPUTED)
    fills = led.list_events(event_type=EventType.FILL)
    assert fills
    assert any(f["payload"].get("commission", 0) > 0 for f in fills)
    led.close()
    d = result.to_dict()
    assert d["mode"] == "paper"
    assert d["capital_label"] == "VIRTUAL"


def test_replay_session_no_lookahead_on_signal_date():
    feed = DailyReplayFeed.from_synthetic(["AAA", "QQQ"], n_days=200, seed=3)
    pipe = DailySignalPipeline(feed, universe=["AAA"], require_regime=False)
    d = feed.days[150]
    # featured through d must not include d+1
    feat = feed.featured("AAA", through=d)
    assert feat["date"].max().date() <= d
    batch = pipe.generate(d)
    for c in batch.candidates:
        assert c.signal_date == d


def test_cli_replay_synthetic_smoke(tmp_path: Path, monkeypatch):
    # Run a short synthetic replay via API (CLI parity)
    freeze = load_freeze()
    feed = DailyReplayFeed.from_synthetic(
        ["AAA", "BBB", "QQQ"], n_days=350, start="2019-06-03", seed=11
    )
    led = PaperLedger.create_run(tmp_path / "cli", freeze)
    session = ReplaySession(feed, freeze, ledger=led)
    session.pipeline.require_regime = False  # more trades in synthetic noise
    days = feed.days
    res = session.run(days[250], days[280])
    assert res.days_run == 31 or res.days_run >= 20
    led.close()
