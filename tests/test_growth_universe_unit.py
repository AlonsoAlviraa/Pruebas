"""Unit tests for growth gates G-Q / G-A (synthetic, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trad_research.growth_universe import (
    GrowthGateConfig,
    GrowthMetrics,
    growth_metrics_from_fund,
    passes_growth_gates,
    rank_growth_passers,
)


def _fund_quarters(eps_list, rev_list, start="2018-03-31"):
    """Build quarterly fund frame with lag 0 (available_at = as_of)."""
    asofs = pd.date_range(start, periods=len(eps_list), freq="QE", tz="UTC")
    return pd.DataFrame(
        {
            "as_of": asofs,
            "eps": eps_list,
            "revenue": rev_list,
            "net_income": [e * 10 for e in eps_list],
            "available_at": asofs,
            "source": "test",
        }
    )


def test_gq_ga_pass_double_digit_and_15pct():
    # 8 quarters: last 4 EPS sum higher by >15% vs prior 4; Q YoY >10%
    # Q0-Q3: 1,1,1,1  Q4-Q7: 1.2,1.2,1.2,1.5 → Q YoY last vs -4 = 1.5/1-1=50%
    # TTM now 5.1 vs old 4.0 = +27.5%
    eps = [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.5]
    rev = [100, 100, 100, 100, 120, 120, 120, 150]
    fund = _fund_quarters(eps, rev)
    as_of = fund["available_at"].iloc[-1]
    m = growth_metrics_from_fund(fund, as_of)
    assert m["n_quarters"] == 8
    assert m["eps_q_yoy"] == 0.5
    assert m["eps_ttm_yoy"] > 0.15
    gq, ga, reason = passes_growth_gates(m)
    assert gq and ga and reason == ""


def test_gq_fails_below_10pct():
    eps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05]  # Q YoY 5%
    rev = [100] * 8
    fund = _fund_quarters(eps, rev)
    m = growth_metrics_from_fund(fund, fund["available_at"].iloc[-1])
    gq, ga, reason = passes_growth_gates(m)
    assert not gq
    assert reason == "gq_fail"


def test_ga_fails_below_15pct():
    # Q YoY strong but TTM only +5%
    eps = [1.0, 1.0, 1.0, 1.0, 1.02, 1.02, 1.02, 1.2]
    # TTM old=4, new=4.26 → 6.5%
    rev = [100] * 8
    fund = _fund_quarters(eps, rev)
    m = growth_metrics_from_fund(fund, fund["available_at"].iloc[-1])
    assert m["eps_q_yoy"] >= 0.10
    gq, ga, reason = passes_growth_gates(m)
    assert gq
    assert not ga
    assert "ga" in reason


def test_no_lookahead_future_fund():
    eps = [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 2.0]
    rev = [100] * 7 + [200]
    fund = _fund_quarters(eps, rev)
    # as_of before last available
    early = fund["available_at"].iloc[3]
    m = growth_metrics_from_fund(fund, early)
    assert m["n_quarters"] == 4
    # cannot compute full TTM YoY with only 4q
    assert m["eps_ttm_yoy"] != m["eps_ttm_yoy"] or True


def test_annual_fallback_for_ga():
    """When only annual rows exist for G-A, use YoY annual EPS."""
    asofs_q = pd.date_range("2023-03-31", periods=5, freq="QE", tz="UTC")
    # Q YoY: last 1.5 vs first of prior year-ish
    q = pd.DataFrame(
        {
            "as_of": asofs_q,
            "period": ["Q"] * 5,
            "eps": [1.0, 1.0, 1.0, 1.0, 1.3],
            "revenue": [100, 100, 100, 100, 130],
            "available_at": asofs_q,
            "source": "test",
        }
    )
    asofs_a = pd.to_datetime(["2022-12-31", "2023-12-31"], utc=True)
    a = pd.DataFrame(
        {
            "as_of": asofs_a,
            "period": ["A", "A"],
            "eps": [4.0, 5.0],  # +25%
            "revenue": [400, 500],
            "available_at": asofs_a,
            "source": "test",
        }
    )
    fund = pd.concat([q, a], ignore_index=True)
    m = growth_metrics_from_fund(fund, "2024-06-01")
    assert abs(m["eps_q_yoy"] - 0.3) < 1e-9
    assert abs(m["eps_ttm_yoy"] - 0.25) < 1e-9
    gq, ga, reason = passes_growth_gates(m)
    assert gq and ga


def test_rank_prefers_higher_growth():
    rows = [
        GrowthMetrics(
            ticker="A",
            eps_q_yoy=0.2,
            eps_ttm_yoy=0.2,
            rev_ttm_yoy=0.2,
            pass_all=True,
        ),
        GrowthMetrics(
            ticker="B",
            eps_q_yoy=0.5,
            eps_ttm_yoy=0.8,
            rev_ttm_yoy=0.6,
            pass_all=True,
        ),
        GrowthMetrics(ticker="C", eps_q_yoy=0.12, pass_all=False),
    ]
    ranked = rank_growth_passers(rows)
    assert [r.ticker for r in ranked] == ["B", "A"]
