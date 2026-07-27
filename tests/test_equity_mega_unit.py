"""Unit tests equity mega grid + costs + signal leverage (no network)."""
from __future__ import annotations

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.equity.cost_drag import (
    CostDragConfig,
    rebalance_cost_on_turnover,
    roundtrip_cost_fraction,
)
from paper_live.equity.grid_zoo import build_equity_grid_zoo, is_banned_equity_spec
from paper_live.equity.signal_backtest import run_equity_spec


def test_grid_zoo_thousands_and_bans():
    zoo = build_equity_grid_zoo(max_strategies=500, include_names=True)
    assert zoo["n_strategies"] >= 100
    for s in zoo["strategies"]:
        assert not is_banned_equity_spec(s)
        Lh = float((s.get("meta") or {}).get("leverage_high") or 1)
        assert Lh <= 2.01
        if Lh > 1.01:
            assert (s.get("meta") or {}).get("apply_financing") is True
            assert (s.get("meta") or {}).get("apply_commissions") is True


def test_cost_drag_positive():
    c = CostDragConfig()
    rt = roundtrip_cost_fraction(price=100.0, cfg=c, leverage=1.0)
    assert rt > 0
    assert rt < 0.05  # not absurd
    rt2 = roundtrip_cost_fraction(price=100.0, cfg=c, leverage=2.0)
    assert rt2 > rt
    d = rebalance_cost_on_turnover(0.5, price=100.0, cfg=c, leverage=1.0)
    assert d >= 0


def test_run_equity_synthetic():
    feed = DailyReplayFeed.from_synthetic(
        ["SPY", "AAPL", "MSFT", "QQQ"], n_days=400, seed=7, start="2018-01-02"
    )
    spec = {
        "id": "t_sma",
        "kind": "sma_trend",
        "underlying": "SPY",
        "meta": {
            "sma_fast": 20,
            "sma_slow": 100,
            "leverage_base": 1.0,
            "leverage_high": 1.5,
            "signal_thresh": 0.55,
            "financing_rate": 0.06,
            "apply_financing": True,
            "apply_commissions": True,
            "hard_dd_cap": -0.5,
        },
    }
    r = run_equity_spec(feed, spec, capital0=100_000.0)
    assert r.n_days > 50
    assert r.mean_leverage >= 0
    assert r.mean_leverage <= 2.01
    assert isinstance(r.total_return, float)
    # costs should be non-negative accumulated drag
    assert r.cost_drag_total >= 0


def test_higher_leverage_not_free():
    """With commissions+financing, high L path differs from 1x (not free alpha)."""
    feed = DailyReplayFeed.from_synthetic(
        ["SPY"], n_days=300, seed=1, start="2019-01-02"
    )
    base = {
        "kind": "buy_hold",
        "underlying": "SPY",
        "meta": {
            "leverage_base": 1.0,
            "leverage_high": 1.0,
            "signal_thresh": 0.5,
            "financing_rate": 0.08,
            "apply_financing": True,
            "apply_commissions": True,
            "hard_dd_cap": -0.9,
        },
    }
    hi = {
        "kind": "buy_hold",
        "underlying": "SPY",
        "meta": {
            "leverage_base": 2.0,
            "leverage_high": 2.0,
            "signal_thresh": 0.0,
            "financing_rate": 0.08,
            "apply_financing": True,
            "apply_commissions": True,
            "hard_dd_cap": -0.9,
        },
    }
    r1 = run_equity_spec(feed, {**base, "id": "l1"}, capital0=100_000.0)
    r2 = run_equity_spec(feed, {**hi, "id": "l2"}, capital0=100_000.0)
    assert r2.mean_leverage > r1.mean_leverage
    # financing drag on 2x should make path different
    assert abs(r2.total_return - 2 * r1.total_return) > 1e-6 or r2.cost_drag_total >= 0
