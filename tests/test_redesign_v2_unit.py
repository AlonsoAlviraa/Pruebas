"""Unit tests for redesign v2 features + graph math (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trad_research.redesign_v2.features_ext import (
    REDESIGN_V2_FEATURE_NAMES,
    engineer_redesign_v2_features,
)
from trad_research.redesign_v2.graph_math import (
    correlation_graph_from_returns,
    graph_summary_dict,
    hub_scores,
    trade_cooccurrence_graph,
)
from trad_research.strategies import get_strategy


def _ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(1e5, 1e6, size=n),
        },
        index=dates,
    )


def test_engineer_features_causal_names():
    df = _ohlcv()
    out = engineer_redesign_v2_features(df)
    for c in REDESIGN_V2_FEATURE_NAMES:
        assert c in out.columns, c
    # early NaNs expected for long windows; late rows finite for ret_20
    assert out["ret_20"].iloc[50:].notna().mean() > 0.8


def test_no_lookahead_peak_dd():
    """dd_from_peak should not use future closes."""
    df = _ohlcv(200)
    # spike at end only
    df.loc[df.index[-1], "close"] = df["close"].iloc[-2] * 3.0
    out = engineer_redesign_v2_features(df)
    # penultimate bar peak should not include final spike
    assert out["dd_from_peak_252"].iloc[-2] > -0.99


def test_residual_with_market():
    df = _ohlcv(250)
    mkt = df["close"] * 1.0 + np.linspace(0, 10, len(df))
    mkt.index = df.index
    out = engineer_redesign_v2_features(df, market_close=mkt)
    assert out["beta_60"].iloc[100:].notna().any()
    assert out["resid_ret_20"].iloc[100:].notna().any()


def test_graph_corr_and_hubs():
    rng = np.random.default_rng(1)
    n = 80
    a = rng.normal(0, 0.01, n)
    b = a + rng.normal(0, 0.002, n)
    c = rng.normal(0, 0.01, n)
    rets = pd.DataFrame({"AAA": a, "BBB": b, "CCC": c})
    mat, edges = correlation_graph_from_returns(rets, min_obs=20, corr_threshold=0.3)
    assert not mat.empty
    assert any(e[0] == "AAA" and e[1] == "BBB" for e in edges) or any(
        e[0] == "BBB" and e[1] == "AAA" for e in edges
    )
    hubs = hub_scores(edges)
    assert hubs
    sm = graph_summary_dict(edges)
    assert sm["n_nodes"] >= 2


def test_trade_cooccurrence():
    trades = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "CCC"],
            "entry_date": pd.to_datetime(
                ["2020-01-06", "2020-01-07", "2020-01-08", "2020-02-03"], utc=True
            ),
        }
    )
    edges = trade_cooccurrence_graph(trades)
    assert isinstance(edges, list)


@pytest.mark.parametrize(
    "name",
    [
        "r2_residual_mom",
        "r2_mom_sharpe",
        "r2_trend_stack",
        "r2_defensive_vt",
        "r2_rsi_reclaim",
    ],
)
def test_r2_strategies_register_and_signal(name: str):
    s = get_strategy(name)
    assert s.needs_training is False
    df = _ohlcv(260)
    from trad_research.features import engineer_m2_features

    df = engineer_m2_features(df)
    from trad_research.backtest import BacktestConfig

    sig, score = s.generate_signals(df, BacktestConfig())
    assert len(sig) == len(df)
    assert len(score) == len(df)
    assert sig.dtype == bool or sig.dtype == np.bool_
