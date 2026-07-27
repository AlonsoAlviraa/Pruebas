"""Unit tests for style clones (STR-04) — synthetic panels only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trad_research.backtest import BacktestConfig
from trad_research.style_clone import (
    STYLE_CLONE_NAMES,
    StyleEWClone,
    StyleMomClone,
    StyleTrendMomClone,
    StyleTrendSMA50Clone,
    all_style_clones,
    get_style_clone,
)
from trad_research.strategies import get_strategy


def _synth_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "open": close,
            "volume": rng.uniform(1e6, 2e6, n),
            "sma_50": pd.Series(close).rolling(50, min_periods=5).mean().to_numpy(),
            "dist_sma_50": np.nan,
            "ret_1m": pd.Series(close).pct_change(21).to_numpy(),
            "atr_norm": rng.uniform(0.01, 0.08, n),
            "volatility_20": rng.uniform(0.15, 0.45, n),
            "volume_ratio": rng.uniform(0.5, 1.5, n),
        },
        index=idx,
    )
    df["dist_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]
    return df


def test_all_clones_registered():
    names = {s.name for s in all_style_clones()}
    assert names == set(STYLE_CLONE_NAMES)
    for n in STYLE_CLONE_NAMES:
        s = get_style_clone(n)
        assert s.name == n
        assert s.needs_training is False


def test_get_strategy_resolves_style_clone():
    s = get_strategy("style_trend_mom_hv")
    assert s.name == "style_trend_mom_hv"


def test_ew_clone_signals():
    df = _synth_df()
    sig, score = StyleEWClone().generate_signals(df, BacktestConfig())
    assert len(sig) == len(df)
    assert sig.dtype == bool or sig.dtype == np.bool_
    # min atr filter may zero some early — but not all if atr_norm high
    assert score.notna().all()


def test_trend_clone_respects_sma():
    df = _synth_df()
    sig, score = StyleTrendSMA50Clone().generate_signals(df, BacktestConfig())
    # Where sma known and close below, signal false
    below = df["close"] < df["sma_50"]
    # atr filter may also mask; check subset with atr ok
    atr_ok = df["atr_norm"] >= 0.02
    mask = below & atr_ok & df["sma_50"].notna()
    if mask.any():
        assert not bool(sig[mask].any())


def test_mom_and_combo_shapes():
    df = _synth_df()
    for cls in (StyleMomClone, StyleTrendMomClone):
        sig, score = cls().generate_signals(df, BacktestConfig())
        assert len(sig) == len(df)
        assert (score[~sig] == 0).all() or (score[~sig] >= 0).all()


def test_backtest_overrides_minalloc_shell():
    o = StyleTrendMomClone().backtest_overrides()
    assert o["min_alloc_pct"] == 0.015
    assert o["require_trend"] is False
    assert o["min_confidence"] == 0.0
