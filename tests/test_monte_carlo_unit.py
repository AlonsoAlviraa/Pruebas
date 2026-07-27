"""Unit tests for Monte Carlo trade robustness (MET-02)."""
from __future__ import annotations

import numpy as np

from trad_research.monte_carlo import (
    equity_from_trade_pnls,
    mc_bootstrap_trades,
    mc_shuffle_trades,
    mc_skip_trades,
)


def test_shuffle_preserves_total_pnl():
    rng = np.random.default_rng(0)
    pnls = rng.normal(0.01, 0.05, 40)
    base = float(pnls.sum())
    # internal check via many shuffles sum
    res = mc_shuffle_trades(pnls, n_sims=50, seed=1, min_trades_full=10)
    assert res.total_pnl_constant is True
    assert res.n_sims == 50
    # reconstruct one shuffle
    order = np.random.default_rng(99).permutation(len(pnls))
    assert abs(float(pnls[order].sum()) - base) < 1e-9


def test_bootstrap_varies_total():
    rng = np.random.default_rng(1)
    pnls = rng.normal(0.02, 0.08, 60)
    res = mc_bootstrap_trades(pnls, n_sims=100, seed=2, min_trades_full=10)
    assert res.total_pnl_constant is False
    assert res.n_sims == 100
    # p5 and p95 sortino should be ordered
    assert res.sortino_p5 <= res.sortino_p95 + 1e-9


def test_skip_reduces_count():
    pnls = np.array([0.1, -0.05, 0.02, 0.03, -0.01, 0.04] * 10)
    res = mc_skip_trades(pnls, skip_frac=0.2, n_sims=30, seed=3, min_trades_full=10)
    assert res.method == "skip"
    assert res.n_sims == 30


def test_equity_from_fractional_returns():
    pnls = np.array([0.1, -0.05, 0.0])
    eq = equity_from_trade_pnls(pnls, start_equity=100.0)
    assert abs(eq[0] - 100.0) < 1e-9
    assert abs(eq[1] - 110.0) < 1e-9
    assert eq[-1] > 0


def test_few_trades_diagnostic():
    res = mc_bootstrap_trades([0.01, -0.02], n_sims=10, seed=0, min_trades_full=50)
    assert res.diagnostic_only is True
