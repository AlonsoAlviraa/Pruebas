"""Unit tests for AUD-B signal modes (synthetic rows, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from paper_live.signals.daily_pipeline import (
    default_rule_signal_row,
    no_extension_signal_row,
    pullback_signal_row,
    score_row_for_mode,
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
    }
    base.update(kwargs)
    return pd.Series(base)


def test_baseline_rejects_below_sma50():
    assert default_rule_signal_row(_row(close=90.0, sma_50=95.0)) is None


def test_baseline_accepts_trend_mom():
    sig = default_rule_signal_row(_row())
    assert sig is not None
    assert sig[2] == "rule_trend_mom_atr"


def test_no_extension_rejects_extended():
    assert no_extension_signal_row(_row(dist_sma_50=0.12, rsi_14=80)) is None


def test_no_extension_accepts_mild():
    sig = no_extension_signal_row(_row(dist_sma_50=0.02, rsi_14=55, ret_1m=0.04))
    assert sig is not None
    assert sig[2] == "rule_no_extension"


def test_pullback_accepts_soft_rsi():
    sig = pullback_signal_row(
        _row(close=100.0, sma_50=102.0, sma_200=90.0, rsi_14=40.0, ret_1m=0.01, dist_sma_50=-0.02)
    )
    assert sig is not None
    assert sig[2] == "rule_pullback"


def test_pullback_rejects_far_above_sma50_hot_rsi():
    assert (
        pullback_signal_row(
            _row(close=120.0, sma_50=100.0, sma_200=90.0, rsi_14=72.0, dist_sma_50=0.2, ret_1m=0.1)
        )
        is None
    )


def test_score_mode_dispatch():
    mild = _row(dist_sma_50=0.02, rsi_14=50, ret_1m=0.03)
    assert score_row_for_mode(mild, "trend_mom") is not None
    assert score_row_for_mode(mild, "no_extension") is not None
    assert score_row_for_mode(
        _row(close=99.0, sma_50=100.0, sma_200=90.0, rsi_14=42.0, dist_sma_50=-0.01, ret_1m=0.0),
        "pullback",
    ) is not None
