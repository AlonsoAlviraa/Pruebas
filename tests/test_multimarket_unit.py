"""Unit tests for multi-market global scoring."""
from __future__ import annotations

from trad_research.multimarket import (
    available_markets,
    default_markets,
    global_rank_table,
    market_row_score,
)


def test_default_markets_have_us_and_geo():
    specs = default_markets()
    ids = {m.market_id for m in specs}
    assert "US" in ids
    assert {"ES", "DE", "FR", "UK"} <= ids


def test_available_filters_missing_dirs():
    # Should not crash; US data exists in this repo
    ok = available_markets(default_markets())
    assert any(m.market_id == "US" for m in ok)


def test_global_rank_penalizes_single_market_hero():
    per = {
        "US": [
            {
                "label": "hero_us",
                "cagr": 0.40,
                "sharpe": 1.0,
                "max_drawdown": -0.40,
                "excess_total_vs_spy": 2.0,
                "n_trades": 100,
            },
            {
                "label": "steady",
                "cagr": 0.20,
                "sharpe": 0.7,
                "max_drawdown": -0.45,
                "excess_total_vs_spy": 0.5,
                "n_trades": 100,
            },
        ],
        "ES": [
            {
                "label": "hero_us",
                "cagr": -0.10,
                "sharpe": -0.5,
                "max_drawdown": -0.70,
                "excess_total_vs_spy": -1.0,
                "n_trades": 30,
            },
            {
                "label": "steady",
                "cagr": 0.12,
                "sharpe": 0.5,
                "max_drawdown": -0.48,
                "excess_total_vs_spy": 0.2,
                "n_trades": 80,
            },
        ],
    }
    table = global_rank_table(per)
    assert table[0]["label"] == "steady"
    assert market_row_score(per["US"][0]) > market_row_score(per["US"][1])
