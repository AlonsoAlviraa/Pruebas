"""Unit tests for residual / alpha attribution (STR-01 / STR-04)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trad_research.alpha_attribution import (
    cash_aware_benchmark,
    compare_to_benchmark,
    confirm_p1_style_confusion,
    confirm_p2_unfair_spy_bench,
    factor_proxy_regression,
    mean_invested_weight,
    promotion_gates_residual,
    rank_problems_by_false_alpha,
    residual_sharpe,
)


def _eq(mu: float, n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, 0.01, n)
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series((1 + rets).cumprod() * 100_000.0, index=idx)


def test_residual_sharpe_identical_is_zero():
    eq = _eq(0.0005)
    assert abs(residual_sharpe(eq, eq)) < 0.05


def test_compare_outperformance():
    idx = pd.date_range("2018-01-01", periods=252, freq="B", tz="UTC")
    # Deterministic: strong compounds faster than weak
    t = np.arange(252, dtype=float)
    strong = pd.Series(100_000.0 * (1.001**t), index=idx)
    weak = pd.Series(100_000.0 * (1.0003**t), index=idx)
    rep = compare_to_benchmark(strong, weak, label="test")
    assert rep.excess_cagr > 0
    assert rep.label == "test"


def test_cash_aware_half_weight():
    b = _eq(0.001, seed=3)
    blend = cash_aware_benchmark(b, w=0.5, start_value=1.0)
    # Half participation → lower terminal than full bench (positive drift)
    b1 = b / float(b.iloc[0])
    assert float(blend.iloc[-1]) < float(b1.iloc[-1]) + 1e-9
    assert float(blend.iloc[0]) == 1.0


def test_mean_invested_weight():
    eq = pd.Series([100.0, 100.0, 100.0])
    pos = pd.Series([50.0, 50.0, 50.0])
    assert abs(mean_invested_weight(pos, eq) - 0.5) < 1e-9


def test_confirm_p1_by_capture():
    r = confirm_p1_style_confusion(
        baseline_excess_vs_spy=0.10,
        clone_excess_vs_spy=0.08,
        baseline_sharpe=0.8,
        clone_sharpe=0.5,
    )
    assert r["confirmed"] is True
    assert r["by_capture"] is True
    assert r["capture_degenerate"] is False


def test_confirm_p1_by_gap():
    r = confirm_p1_style_confusion(
        baseline_excess_vs_spy=0.10,
        clone_excess_vs_spy=0.01,
        baseline_sharpe=0.70,
        clone_sharpe=0.65,
    )
    assert r["by_gap"] is True
    assert r["confirmed"] is True


def test_confirm_p1_capture_degenerate_when_baseline_excess_nonpositive():
    """Early-window case: both underperform SPY — capture must not confirm P1."""
    r = confirm_p1_style_confusion(
        baseline_excess_vs_spy=-0.08,
        clone_excess_vs_spy=-0.45,
        baseline_sharpe=0.9,
        clone_sharpe=-1.9,
    )
    assert r["capture_degenerate"] is True
    assert r["by_capture"] is False
    # residual sharpe gap is large → not by_gap either
    assert r["by_gap"] is False
    assert r["confirmed"] is False


def test_confirm_p1_pathology_blocks_false_confirm():
    """Absurd style CAGR (e.g. 246%) must not confirm P1 even if capture huge."""
    from trad_research.alpha_attribution import clone_metrics_pathology

    path = clone_metrics_pathology(clone_cagr=2.46, clone_excess_vs_spy=2.31)
    assert path["pathology_suspect"] is True
    r = confirm_p1_style_confusion(
        baseline_excess_vs_spy=0.038,
        clone_excess_vs_spy=2.31,
        baseline_sharpe=0.79,
        clone_sharpe=0.71,
        clone_cagr=2.46,
    )
    assert r["pathology_suspect"] is True
    assert r["by_capture"] is False
    assert r["confirmed"] is False


def test_confirm_p2():
    assert confirm_p2_unfair_spy_bench(-0.05)["confirmed"] is True
    assert confirm_p2_unfair_spy_bench(0.02)["confirmed"] is False


def test_factor_proxy_regression_smoke():
    idx = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.0004, 0.01, 200), index=idx)
    y = 0.0001 + 1.2 * mkt + rng.normal(0, 0.002, 200)
    out = factor_proxy_regression(y, {"mkt": mkt})
    assert out["n"] == 200
    assert abs(out["betas"]["mkt"] - 1.2) < 0.15
    assert out["r2"] > 0.5


def test_promotion_gates():
    from trad_research.alpha_attribution import ResidualReport

    good = ResidualReport(
        strategy_cagr=0.12,
        strategy_sharpe=0.6,
        strategy_mdd=-0.2,
        bench_cagr=0.08,
        bench_sharpe=0.4,
        bench_mdd=-0.25,
        excess_cagr=0.04,
        residual_sharpe=0.3,
    )
    pit = ResidualReport(
        strategy_cagr=0.12,
        strategy_sharpe=0.6,
        strategy_mdd=-0.2,
        bench_cagr=0.11,
        bench_sharpe=0.5,
        bench_mdd=-0.3,
        excess_cagr=0.01,
        residual_sharpe=0.1,
    )
    g = promotion_gates_residual(good, pit)
    assert g["pass_core"] is True
    assert g["R2_status"] == "evaluated"


def test_promotion_gates_missing_pit_not_pass():
    from trad_research.alpha_attribution import ResidualReport

    good = ResidualReport(
        strategy_cagr=0.12,
        strategy_sharpe=0.6,
        strategy_mdd=-0.2,
        bench_cagr=0.08,
        bench_sharpe=0.4,
        bench_mdd=-0.25,
        excess_cagr=0.04,
        residual_sharpe=0.3,
    )
    g = promotion_gates_residual(good, None)
    assert g["R2_pit_ew"] is False
    assert g["R2_status"] == "not_evaluated"
    assert g["pass_core"] is False
    assert g["incomplete"] is True


def test_promotion_gates_engine_mismatch_diagnostic():
    from trad_research.alpha_attribution import ResidualReport

    good = ResidualReport(
        strategy_cagr=0.2,
        strategy_sharpe=1.0,
        strategy_mdd=-0.2,
        bench_cagr=0.1,
        bench_sharpe=0.5,
        bench_mdd=-0.3,
        excess_cagr=0.1,
        residual_sharpe=0.4,
    )
    pit = ResidualReport(
        strategy_cagr=0.2,
        strategy_sharpe=1.0,
        strategy_mdd=-0.2,
        bench_cagr=0.15,
        bench_sharpe=0.6,
        bench_mdd=-0.25,
        excess_cagr=0.05,
        residual_sharpe=0.2,
    )
    g = promotion_gates_residual(
        good, pit, engine_matched=False, diagnostic_only=True
    )
    assert g["pass_core"] is False
    assert g["diagnostic_only"] is True


def test_rank_problems():
    rows = rank_problems_by_false_alpha(
        [
            {"problem": "P2", "confirmed": True},
            {"problem": "P1", "confirmed": True},
            {"problem": "P5", "confirmed": False},
        ]
    )
    assert rows[0]["problem"] == "P1"
    assert rows[0]["severity_weight"] > 0
