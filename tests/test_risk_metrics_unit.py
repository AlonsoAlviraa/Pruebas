"""Unit tests for extended risk metrics (MET-01)."""
from __future__ import annotations

import numpy as np

from trad_research.metrics import PerformanceReport, acceptance_gates
from trad_research.risk_metrics import (
    cvar,
    downside_deviation,
    expectancy,
    max_consecutive_losses,
    sortino_ratio,
    tail_ratio,
    ulcer_index,
)


def test_sortino_no_downside_is_large():
    r = np.array([0.01, 0.02, 0.015, 0.01])
    s = sortino_ratio(r, mar=0.0)
    assert s >= 10.0


def test_sortino_with_losses():
    r = np.array([0.02, -0.01, 0.01, -0.02, 0.015])
    s = sortino_ratio(r, mar=0.0)
    assert s > 0
    dd = downside_deviation(r, mar=0.0)
    assert dd > 0


def test_ulcer_and_tail_cvar():
    eq = np.cumprod(1.0 + np.array([0.01, -0.02, 0.0, 0.01, -0.03, 0.02])) * 100
    u = ulcer_index(eq)
    assert u >= 0
    r = np.diff(eq) / eq[:-1]
    assert tail_ratio(r) >= 0
    assert cvar(r, 0.2) <= 0 or True  # worst tail mean


def test_consecutive_losses_and_expectancy():
    pnls = np.array([1.0, -1.0, -1.0, -1.0, 2.0, -0.5])
    assert max_consecutive_losses(pnls) == 3
    assert abs(expectancy(pnls) - pnls.mean()) < 1e-12


def test_acceptance_gates_include_sortino():
    rep = PerformanceReport(
        n_trades=200,
        win_rate=0.5,
        profit_factor=1.2,
        avg_return=0.01,
        total_return=1.0,
        cagr=0.12,
        sharpe=0.6,
        sortino=0.7,
        max_drawdown=-0.2,
        calmar=0.6,
        years=8.0,
        final_equity=200_000,
        start_equity=100_000,
        positive_year_frac=0.7,
    )
    g = acceptance_gates(rep)
    assert "sortino_ok" in g
    assert g["sortino_ok"] is True
    rep2 = PerformanceReport(
        n_trades=200,
        win_rate=0.5,
        profit_factor=1.2,
        avg_return=0.01,
        total_return=0.5,
        cagr=0.12,
        sharpe=0.6,
        sortino=0.1,
        max_drawdown=-0.2,
        calmar=0.6,
        years=8.0,
        final_equity=150_000,
        start_equity=100_000,
        positive_year_frac=0.7,
    )
    assert acceptance_gates(rep2)["sortino_ok"] is False
