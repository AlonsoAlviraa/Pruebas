"""Unit tests for Sistema A ORB+HTF daily proxy (causal signals + strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trad_research.backtest import BacktestConfig, _chandelier_step, OpenPosition
from trad_research.orb_htf import compute_orb_htf_signals
from trad_research.strategies import get_strategy


def _synth_uptrend(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.15, 0.8, size=n))
    close = np.maximum(close, 10.0)
    high = close + rng.uniform(0.1, 1.5, size=n)
    low = close - rng.uniform(0.1, 1.5, size=n)
    open_ = close + rng.normal(0, 0.3, size=n)
    # force last bars: prior high break + bull day
    close[-1] = high[-2] + 2.0
    open_[-1] = close[-1] - 0.5
    high[-1] = close[-1] + 0.2
    low[-1] = open_[-1] - 0.1
    sma50 = pd.Series(close).rolling(50, min_periods=1).mean().to_numpy()
    sma200 = pd.Series(close).rolling(200, min_periods=1).mean().to_numpy()
    # ensure last close above both SMAs
    close[-1] = max(close[-1], sma50[-1], sma200[-1]) + 1.0
    high[-1] = close[-1] + 0.2
    atr = np.full(n, 1.5)
    atr_norm = atr / close
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "sma_50": sma50,
            "sma_200": sma200,
            "atr": atr,
            "atr_norm": atr_norm,
        }
    )


def test_orb_uses_prior_day_high_only():
    df = _synth_uptrend(220)
    sig, score = compute_orb_htf_signals(df, bias_mode="dual_ma")
    # First bar cannot signal (no prior high)
    assert bool(sig.iloc[0]) is False
    # Last bar engineered to break prior high with bias
    assert bool(sig.iloc[-1]) is True
    assert float(score.iloc[-1]) > 0


def test_no_lookahead_orb_high():
    df = _synth_uptrend(220)
    # Mutate future high — should not change signal at t if only t+1 high changes
    sig1, _ = compute_orb_htf_signals(df, bias_mode="dual_ma")
    df2 = df.copy()
    df2.loc[df2.index[-1], "high"] = 1e9
    sig2, _ = compute_orb_htf_signals(df2, bias_mode="dual_ma")
    # signals through -2 should match (last bar high not used as orb_high until next day)
    pd.testing.assert_series_equal(sig1.iloc[:-1], sig2.iloc[:-1])


def test_sma200_only_mode():
    df = _synth_uptrend(220)
    # Put close below sma50 but above sma200 on last bar
    df.loc[df.index[-1], "sma_50"] = df.loc[df.index[-1], "close"] + 5
    df.loc[df.index[-1], "sma_200"] = df.loc[df.index[-1], "close"] - 5
    df.loc[df.index[-1], "close"] = df.loc[df.index[-2], "high"] + 1.0
    df.loc[df.index[-1], "open"] = df.loc[df.index[-1], "close"] - 0.2
    sig_dual, _ = compute_orb_htf_signals(df, bias_mode="dual_ma")
    sig_200, _ = compute_orb_htf_signals(df, bias_mode="sma200_only")
    assert bool(sig_dual.iloc[-1]) is False
    assert bool(sig_200.iloc[-1]) is True


def test_strategy_registered():
    s0 = get_strategy("orb_htf_daily_proxy")
    s1 = get_strategy("orb_htf_daily_proxy_a1")
    assert s0.needs_training is False
    assert s1.bias_mode == "sma200_only"
    o = s0.backtest_overrides()
    assert o["require_regime"] is False
    assert o["risk_per_trade_pct"] == 0.0075
    assert o["take_profit_r"] == 2.0


def test_take_profit_r_exit():
    cfg = BacktestConfig(k_atr=1.5, take_profit_r=2.0, max_horizon=20)
    pos = OpenPosition(
        ticker="X",
        entry_date=pd.Timestamp("2020-01-01", tz="UTC"),
        entry_idx=0,
        entry_price=100.0,
        shares=10,
        stop=95.0,
        hard_stop=95.0,
        highest_high=100.0,
        bars_held=0,
        capital_used=1000.0,
        horizon_limit=20,
    )
    # 2R = 10; high 111 triggers TP
    should, reason, px = _chandelier_step(pos, 111.0, 99.0, 110.0, 1.0, cfg)
    assert should is True
    assert reason == "take_profit"
    assert px == 110.0
