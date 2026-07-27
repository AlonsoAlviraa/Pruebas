"""Unit tests for multi-market result audit helpers."""
from __future__ import annotations

from scripts.audit_multimarket_results import (
    leave_one_year_out_cagr,
    score_decomposition,
    wealth_path,
)


def test_score_decomp_mdd_ok_dominates_deep_dd():
    deep = {
        "max_drawdown": -0.70,
        "excess_total_vs_spy": 2.0,
        "sharpe": 0.6,
        "cagr": 0.25,
        "n_trades": 100,
    }
    ok = {
        "max_drawdown": -0.42,
        "excess_total_vs_spy": 2.0,
        "sharpe": 0.6,
        "cagr": 0.25,
        "n_trades": 100,
    }
    d0 = score_decomposition(deep)
    d1 = score_decomposition(ok)
    assert d1["mdd_ok_bonus"] == 50.0
    assert d0["mdd_ok_bonus"] == 0.0
    assert d1["total"] > d0["total"]


def test_leave_one_year_out_drops_2020():
    years = [
        {"year": 2018, "year_return": -0.10},
        {"year": 2019, "year_return": -0.10},
        {"year": 2020, "year_return": 1.50},
        {"year": 2021, "year_return": 0.20},
    ]
    cagr_full_n = wealth_path([-0.10, -0.10, 1.50, 0.20]) ** 0.25 - 1
    cagr_loo = leave_one_year_out_cagr(years, 2020)
    assert cagr_loo is not None
    assert cagr_loo < cagr_full_n


def test_wealth_path():
    assert abs(wealth_path([1.0, 0.0]) - 2.0) < 1e-9
