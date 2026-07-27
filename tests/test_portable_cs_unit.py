"""Unit tests for portable CS features + residual labels (FEA-04 / STR-03)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trad_research.portable.cs_features import (
    ABSOLUTE_BANNED_FEATURES,
    INVARIANT_FEATURE_NAMES,
    assert_no_absolute_prices,
    cross_sectional_ranks,
    feature_set_diff,
    invariant_matrix,
)
from trad_research.portable.residual_labels import (
    beat_style_labels,
    beat_style_meta_frame,
    forward_return,
    panel_beat_style_vs_ew,
    residual_excess_labels,
    residual_excess_series,
)


def test_ban_absolute_features():
    with pytest.raises(ValueError, match="bans absolute"):
        assert_no_absolute_prices(["atr_norm", "close"])
    assert_no_absolute_prices(list(INVARIANT_FEATURE_NAMES))
    assert "close" in ABSOLUTE_BANNED_FEATURES


def test_feature_set_diff():
    assert feature_set_diff(["atr_norm", "open"]) == ["open"]
    assert feature_set_diff(["rsi_14"]) == []


def test_cross_sectional_ranks_unit_interval():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3 + ["2020-01-02"] * 3,
            "ticker": ["A", "B", "C"] * 2,
            "atr_norm": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            "ret_1m": [0.0, 0.1, -0.1, 0.05, 0.0, -0.05],
        }
    )
    out = cross_sectional_ranks(df, feature_cols=["atr_norm", "ret_1m"])
    r = out.loc[out["date"] == "2020-01-01", "atr_norm_csrank"]
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert abs(float(r.max()) - 1.0) < 1e-9 or float(r.max()) <= 1.0


def test_invariant_matrix():
    df = pd.DataFrame({c: np.arange(5, dtype=float) for c in INVARIANT_FEATURE_NAMES})
    X = invariant_matrix(df)
    assert list(X.columns) == list(INVARIANT_FEATURE_NAMES)


def test_forward_return_and_beat_style():
    close = np.array([100.0, 101.0, 102.0, 103.0, 110.0], dtype=float)
    style = np.array([100.0, 100.5, 101.0, 101.5, 102.0], dtype=float)
    fr = forward_return(close, 2)
    assert np.isnan(fr[-1]) and np.isnan(fr[-2])
    assert abs(fr[0] - (102.0 / 100.0 - 1.0)) < 1e-9
    excess, beat = residual_excess_labels(close, style, horizon=2)
    # incomplete horizon → NaN beat (not hard 0)
    assert np.isnan(beat[-1]) and np.isnan(beat[-2])
    assert np.isfinite(beat[0])
    fin = beat_style_labels(close, style, horizon=2)
    assert np.nansum(fin) >= 0


def test_panel_beat_style_vs_ew():
    rows = []
    for t, base in [("AAA", 100.0), ("BBB", 50.0)]:
        for i in range(30):
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
                    "ticker": t,
                    "close": base * (1.01**i) if t == "AAA" else base * (1.001**i),
                }
            )
    panel = pd.DataFrame(rows)
    out = panel_beat_style_vs_ew(panel, horizon=5)
    assert "y_beat_style" in out.columns
    assert "y_excess" in out.columns
    fin = out["y_beat_style"].dropna()
    assert fin.isin([0.0, 1.0]).all()
    assert out["y_beat_style"].isna().any()  # incomplete tail


def test_residual_excess_series_and_meta_api():
    idx = pd.date_range("2021-01-01", periods=10, freq="B", tz="UTC")
    strat = pd.Series(np.cumprod(1 + np.full(10, 0.01)), index=idx) * 100
    style = pd.Series(np.cumprod(1 + np.full(10, 0.005)), index=idx) * 100
    resid = residual_excess_series(strat, style)
    assert (resid > 0).all()
    ex, beat = beat_style_meta_frame(np.array([0.05, -0.01]), np.array([0.0, 0.0]))
    assert beat[0] == 1 and beat[1] == 0
    assert abs(ex[0] - 0.05) < 1e-12


def test_ban_ma_aliases():
    with pytest.raises(ValueError, match="bans absolute"):
        assert_no_absolute_prices(["atr_norm", "ma_50"])
    with pytest.raises(ValueError, match="bans absolute"):
        assert_no_absolute_prices(["ma_200"])


def test_l1_missing_rank_not_filled_half():
    """Missing features must not invent 0.5 rank mass."""
    from trad_research.portable.score_l1 import rule_rank_scores, ScoreL1Config

    df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "ticker": ["A", "B", "C"],
            "ret_1m": [0.1, 0.2, np.nan],
            "dist_sma_50": [0.0, 0.1, 0.2],
        }
    )
    cfg = ScoreL1Config(
        feature_cols=("ret_1m", "dist_sma_50"),
        weights={"ret_1m_csrank": 1.0, "dist_sma_50_csrank": 0.0},
    )
    out = rule_rank_scores(df, config=cfg)
    # C has nan ret_1m → rank nan → l1_score nan (not 0.5 invent)
    row_c = out.loc[out["ticker"] == "C"].iloc[0]
    assert not bool(row_c.get("_l1_valid", True)) or not np.isfinite(row_c["l1_score"])


def test_attach_residual_labels_single_ticker():
    from trad_research.portable.residual_labels import attach_residual_labels

    dates = pd.date_range("2020-01-01", periods=40, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "close": np.linspace(100, 120, 40),
        }
    )
    style = pd.Series(np.linspace(100, 110, 40), index=dates)
    out = attach_residual_labels(df, style, horizon=5)
    assert "y_excess" in out.columns
    fin = out["y_beat_style"].dropna()
    assert fin.isin([0.0, 1.0]).all()
    assert out["y_excess"].notna().sum() >= 30
    # incomplete horizon → NaN, not hard 0
    assert out["y_beat_style"].isna().sum() >= 5


def test_attach_residual_labels_multi_ticker_per_group():
    from trad_research.portable.residual_labels import attach_residual_labels

    dates = pd.date_range("2020-01-01", periods=30, freq="B", tz="UTC")
    rows = []
    for t, base in [("AAA", 100.0), ("BBB", 50.0)]:
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "close": base * (1.01**i)})
    df = pd.DataFrame(rows)
    style = pd.Series(np.linspace(100, 105, 30), index=dates)
    out = attach_residual_labels(df, style, horizon=3)
    assert out["ticker"].nunique() == 2
    # Per-ticker: last horizon rows are nan excess
    for tkr, g in out.groupby("ticker"):
        assert g["y_excess"].isna().sum() >= 3


def test_attach_residual_labels_low_hit_rate_raises():
    from trad_research.portable.residual_labels import attach_residual_labels

    dates = pd.date_range("2020-01-01", periods=20, freq="B", tz="UTC")
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": np.arange(20) + 100.0})
    # Style on completely different dates
    style = pd.Series([1.0, 2.0], index=pd.date_range("2010-01-01", periods=2, freq="D", tz="UTC"))
    with pytest.raises(ValueError, match="hit-rate"):
        attach_residual_labels(df, style, horizon=2, min_style_hit_rate=0.5)
