"""Unit tests for leverage path math (no network)."""
from __future__ import annotations

import numpy as np

from paper_live.leverage.models import (
    LeverSpec,
    apply_leverage_to_returns,
    daily_reset_levered_returns,
    financing_daily,
    geometric_mean,
    rank_by_mean_return,
    select_good_levered,
    year_returns_from_daily,
)


def test_financing_zero_at_1x():
    assert financing_daily(1.0, 0.05) == 0.0
    assert financing_daily(2.0, 0.05) > 0


def test_2x_approx_doubles_in_up_trend():
    r = [0.01] * 50  # strong up
    p1 = apply_leverage_to_returns(r, spec=LeverSpec(leverage=1.0, financing_rate=0.0))
    p2 = apply_leverage_to_returns(r, spec=LeverSpec(leverage=2.0, financing_rate=0.0))
    assert p2.total_return > p1.total_return


def test_financing_reduces_return():
    r = [0.001] * 100
    no_fin = apply_leverage_to_returns(r, spec=LeverSpec(leverage=2.0, financing_rate=0.0))
    fin = apply_leverage_to_returns(r, spec=LeverSpec(leverage=2.0, financing_rate=0.10))
    assert fin.total_return < no_fin.total_return


def test_wipe_on_hard_dd():
    # big down day
    r = [0.0] * 10 + [-0.40] + [0.0] * 10
    p = apply_leverage_to_returns(
        r,
        spec=LeverSpec(leverage=2.0, financing_rate=0.0, hard_dd_cap=-0.50),
    )
    # 2x * -40% = -80% day → wipe
    assert p.wiped or p.max_dd <= -0.50


def test_daily_reset_helper():
    r = np.random.default_rng(0).normal(0.0005, 0.01, 80)
    p = daily_reset_levered_returns(r, 3.0, financing_rate=0.05)
    assert p.data_label in ("etf_levered_proxy", "levered_wipe_proxy", "levered_proxy")
    assert len(p.daily_returns) == 80


def test_year_returns_and_rank():
    from datetime import date, timedelta

    dates = [date(2023, 1, 3) + timedelta(days=i) for i in range(10)]
    dates += [date(2024, 1, 3) + timedelta(days=i) for i in range(10)]
    rets = [0.01] * 10 + [0.02] * 10
    yr = year_returns_from_daily(rets, dates)
    assert "2023" in yr and "2024" in yr
    assert yr["2024"] > yr["2023"]

    rows = [
        {"strategy_id": "a", "mean_ret": 0.1, "mean_xs_spy": 0.02, "calmar_like": 1.0, "worst_dd": -0.1},
        {"strategy_id": "b", "mean_ret": 0.3, "mean_xs_spy": 0.05, "calmar_like": 2.0, "worst_dd": -0.2},
    ]
    ranked = rank_by_mean_return(rows)
    assert ranked[0]["strategy_id"] == "b"
    assert ranked[0]["rank_mean_ret"] == 1


def test_select_good():
    rows = [
        {
            "strategy_id": "good",
            "mean_ret": 0.40,
            "mean_xs_spy": 0.10,
            "worst_dd": -0.25,
            "n_positive_years": 3,
            "wipe_years": 0,
            "max_upside_share": 0.4,
        },
        {
            "strategy_id": "lottery",
            "mean_ret": 0.50,
            "mean_xs_spy": 0.12,
            "worst_dd": -0.20,
            "n_positive_years": 2,
            "wipe_years": 0,
            "max_upside_share": 0.85,
        },
        {
            "strategy_id": "bad",
            "mean_ret": -0.05,
            "mean_xs_spy": -0.10,
            "worst_dd": -0.45,
            "n_positive_years": 1,
            "wipe_years": 2,
            "max_upside_share": 0.5,
        },
    ]
    prom, watch, kill = select_good_levered(rows, qqq_bh_mean=0.25, spy_bh_mean=0.15)
    assert any(r["strategy_id"] == "good" for r in prom)
    assert any(r["strategy_id"] == "lottery" for r in watch)
    assert any(r["strategy_id"] == "bad" for r in kill)


def test_geometric_mean():
    g = geometric_mean([0.1, 0.1, 0.1])
    assert g is not None and abs(g - 0.1) < 1e-9
