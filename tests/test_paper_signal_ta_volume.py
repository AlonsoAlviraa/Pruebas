"""Unit + integration tests for TA/volume equity signal modes (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.signals.daily_pipeline import (
    DailySignalPipeline,
    KNOWN_SIGNAL_MODES,
    rvol_trend_signal_row,
    rsi_mean_reversion_signal_row,
    score_row_for_mode,
    vol_pullback_signal_row,
    volume_breakout_signal_row,
    volume_dryup_signal_row,
    volume_expansion_signal_row,
)


def _row(**kwargs):
    base = {
        "close": 100.0,
        "sma_50": 95.0,
        "sma_200": 90.0,
        "ret_1m": 0.05,
        "atr_norm": 0.02,
        "volatility_20": 0.2,
        "dist_sma_50": 0.05,
        "rsi_14": 55.0,
        "volume_ratio": 1.0,
        "volume_zscore": 0.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_volume_breakout_requires_elevated_volume():
    trend = _row(volume_ratio=1.0, volume_zscore=0.0)
    assert volume_breakout_signal_row(trend) is None
    ok = volume_breakout_signal_row(_row(volume_ratio=1.5, volume_zscore=1.0))
    assert ok is not None
    assert ok[2] == "rule_volume_breakout"


def test_rsi_mr_accepts_oversold_above_sma200():
    sig = rsi_mean_reversion_signal_row(
        _row(close=100.0, sma_200=95.0, rsi_14=28.0, ret_1m=-0.02, dist_sma_50=-0.03)
    )
    assert sig is not None
    assert sig[2] == "rule_rsi_mean_reversion"


def test_rsi_mr_rejects_hot_rsi():
    assert (
        rsi_mean_reversion_signal_row(
            _row(close=100.0, sma_200=90.0, rsi_14=45.0)
        )
        is None
    )


def test_rsi_mr_rejects_below_sma200():
    assert (
        rsi_mean_reversion_signal_row(
            _row(close=80.0, sma_200=100.0, rsi_14=25.0)
        )
        is None
    )


def test_volume_dryup_pullback():
    sig = volume_dryup_signal_row(
        _row(
            close=99.0,
            sma_50=100.0,
            sma_200=90.0,
            rsi_14=40.0,
            dist_sma_50=-0.01,
            ret_1m=0.0,
            volume_ratio=0.6,
            volume_zscore=-0.8,
        )
    )
    assert sig is not None
    assert sig[2] == "rule_volume_dryup"


def test_volume_dryup_rejects_high_volume_pullback():
    assert (
        volume_dryup_signal_row(
            _row(
                close=99.0,
                sma_50=100.0,
                sma_200=90.0,
                rsi_14=40.0,
                dist_sma_50=-0.01,
                ret_1m=0.0,
                volume_ratio=1.8,
                volume_zscore=1.5,
            )
        )
        is None
    )


def test_volume_expansion():
    sig = volume_expansion_signal_row(
        _row(volume_ratio=1.8, volume_zscore=1.2, rsi_14=58.0, ret_1m=0.04)
    )
    assert sig is not None
    assert sig[2] == "rule_volume_expansion"


def test_volume_expansion_rejects_rsi_climax():
    assert (
        volume_expansion_signal_row(
            _row(volume_ratio=2.0, rsi_14=78.0, ret_1m=0.05)
        )
        is None
    )


def test_rvol_trend():
    mild = _row(dist_sma_50=0.02, rsi_14=55.0, ret_1m=0.04, volume_ratio=1.3)
    sig = rvol_trend_signal_row(mild)
    assert sig is not None
    assert sig[2] == "rule_rvol_trend"
    assert rvol_trend_signal_row(
        _row(dist_sma_50=0.02, rsi_14=55.0, volume_ratio=0.5, volume_zscore=-1.0)
    ) is None


def test_vol_pullback_mode():
    sig = vol_pullback_signal_row(
        _row(
            close=99.0,
            sma_50=100.0,
            sma_200=90.0,
            rsi_14=40.0,
            dist_sma_50=-0.01,
            ret_1m=0.0,
            volume_ratio=0.55,
        )
    )
    assert sig is not None


def test_score_row_for_mode_ta_dispatch():
    elev = _row(volume_ratio=1.6, volume_zscore=1.0, dist_sma_50=0.02, rsi_14=55.0)
    assert score_row_for_mode(elev, "vol_confirm") is not None
    assert score_row_for_mode(elev, "volume_breakout") is not None
    assert score_row_for_mode(elev, "vol_expand") is not None
    assert score_row_for_mode(elev, "rvol_trend") is not None

    oversold = _row(
        close=100.0, sma_200=95.0, rsi_14=28.0, ret_1m=-0.01, dist_sma_50=-0.02
    )
    assert score_row_for_mode(oversold, "rsi_mr") is not None
    assert score_row_for_mode(oversold, "rsi_mean_reversion") is not None

    dry_pb = _row(
        close=99.0,
        sma_50=100.0,
        sma_200=90.0,
        rsi_14=40.0,
        dist_sma_50=-0.01,
        ret_1m=0.0,
        volume_ratio=0.6,
    )
    assert score_row_for_mode(dry_pb, "vol_dryup") is not None
    assert score_row_for_mode(dry_pb, "vol_pullback") is not None
    assert score_row_for_mode(dry_pb, "combined_ta_v1") is not None


def test_combined_ta_v1_prefers_dryup_over_rvol():
    """When both dry-up and rvol could fire, combined returns dry-up reason path."""
    both = _row(
        close=99.0,
        sma_50=100.0,
        sma_200=90.0,
        rsi_14=40.0,
        dist_sma_50=-0.01,
        ret_1m=0.04,
        volume_ratio=0.55,  # dry for dryup; rvol would fail
        volume_zscore=-0.8,
        atr_norm=0.02,
    )
    sig = score_row_for_mode(both, "combined_ta_v1")
    assert sig is not None
    assert sig[2] == "rule_volume_dryup"


def test_unknown_signal_mode_fail_closed():
    """Issue 3: typo must not silently become trend_mom."""
    mild = _row(volume_ratio=1.5, dist_sma_50=0.02, rsi_14=55.0, ret_1m=0.04)
    assert score_row_for_mode(mild, "trend_mom") is not None
    assert score_row_for_mode(mild, "vol_cofirm") is None  # typo
    assert score_row_for_mode(mild, "not_a_real_mode") is None
    assert "vol_confirm" in KNOWN_SIGNAL_MODES


def test_volume_breakout_rejects_no_trend():
    assert (
        volume_breakout_signal_row(
            _row(close=90.0, sma_50=100.0, volume_ratio=2.0)
        )
        is None
    )


def test_pipeline_ta_modes_causal_generate():
    """Integration: DailySignalPipeline.generate uses featured through day only."""
    feed = DailyReplayFeed.from_synthetic(
        ["AAPL", "MSFT", "QQQ", "SPY"],
        n_days=220,
        start="2020-01-02",
        seed=42,
    )
    days = feed.days
    signal_day = days[-5]
    # Causality smoke: featured last close == history through day last close
    hist = feed.history("AAPL", through=signal_day, include_through=True)
    feat = feed.featured("AAPL", through=signal_day)
    assert not hist.empty and not feat.empty
    assert float(feat.iloc[-1]["close"]) == float(hist.iloc[-1]["close"])
    assert pd.Timestamp(feat.iloc[-1]["date"]).date() <= signal_day

    for mode in ("vol_confirm", "rsi_mr", "rvol_trend", "vol_dryup"):
        pipe = DailySignalPipeline(
            feed,
            universe=["AAPL", "MSFT"],
            require_regime=False,
            exclude_index=True,
            signal_mode=mode,
            min_price=1.0,
            min_atr_norm=0.001,
            max_atr_pct=0.50,
        )
        batch = pipe.generate(signal_day)
        assert batch.signal_date == signal_day
        for c in batch.candidates:
            assert c.signal_date == signal_day
            assert c.ticker in ("AAPL", "MSFT")
            assert c.close > 0
            # candidate close matches causal last bar
            bar = feed.bar(c.ticker, signal_day)
            if bar is not None:
                assert abs(c.close - float(bar.close)) < 1e-6


def test_pipeline_unknown_mode_no_candidates():
    feed = DailyReplayFeed.from_synthetic(
        ["AAPL", "QQQ"], n_days=200, start="2020-01-02", seed=1
    )
    day = feed.days[-1]
    pipe = DailySignalPipeline(
        feed,
        universe=["AAPL"],
        require_regime=False,
        signal_mode="vol_cofirm",  # typo
        min_price=1.0,
    )
    batch = pipe.generate(day)
    assert batch.candidates == []
