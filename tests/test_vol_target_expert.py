"""Unit tests for expert vol-target (synthetic; no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.equity.signal_backtest import run_equity_spec
from paper_live.equity.vol_target_expert import (
    PRESETS,
    VolTargetExpertConfig,
    apply_deadband,
    baseline_leverage_series,
    ewma_vol_path,
    expert_feature_gap_report,
    iter_study_specs,
    leverage_series_expert,
    rolling_std_vol_path,
)


def test_ewma_vol_causal_and_positive():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=300)
    # inject vol spike mid-sample
    r[150:160] *= 4.0
    vol = ewma_vol_path(r, lam=0.94, seed_window=20)
    assert np.isnan(vol[:20]).all() or np.isnan(vol[0])
    # after seed, should be finite
    assert np.isfinite(vol[25:]).mean() > 0.9
    # vol should rise after spike (compare mean before vs after)
    pre = float(np.nanmean(vol[100:150]))
    post = float(np.nanmean(vol[160:200]))
    assert post > pre


def test_rolling_std_matches_manual():
    r = np.array([0.01, -0.02, 0.015, -0.01, 0.005] * 20, dtype=float)
    v = rolling_std_vol_path(r, lookback=20)
    i = 25
    manual = float(np.std(r[i - 20 : i], ddof=1) * np.sqrt(252))
    assert abs(v[i] - manual) < 1e-9


def test_deadband_holds():
    assert apply_deadband(1.05, 1.0, 0.10) == 1.0
    assert apply_deadband(1.15, 1.0, 0.10) == 1.15
    assert apply_deadband(0.5, 1.0, 0.0) == 0.5


def test_leverage_series_expert_bounded():
    idx = pd.date_range("2018-01-01", periods=400, freq="B", tz="UTC")
    # geometric random walk
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0004, 0.012, size=len(idx))
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    vix = pd.Series(15 + 5 * np.abs(rng.normal(0, 1, size=len(idx))), index=idx)
    cfg = PRESETS["full_expert"]
    cfg.vol_target = 0.15
    cfg.max_leverage = 2.0
    lev, diag = leverage_series_expert(close, cfg=cfg, vix=vix)
    assert len(lev) == len(close)
    assert float(lev.max()) <= 2.0 + 1e-9
    assert float(lev.min()) >= 0.0
    assert diag["used_vix"] is True
    assert diag["mean_leverage"] >= 0


def test_baseline_vs_expert_differ():
    idx = pd.date_range("2018-01-01", periods=500, freq="B", tz="UTC")
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0005, 0.015, size=len(idx))
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx)
    base = baseline_leverage_series(close, vol_target=0.15, max_leverage=2.0)
    cfg = VolTargetExpertConfig(
        vol_target=0.15,
        max_leverage=2.0,
        vol_estimator="ewma",
        use_trend_filter=True,
        use_circuit_breaker=True,
        use_vix_level=False,
        use_vix_rank=False,
        rebalance_band=0.1,
    )
    exp, _ = leverage_series_expert(close, cfg=cfg, vix=None)
    # paths should not be identical (EWMA + trend + band)
    corr = np.corrcoef(base.values[50:], exp.values[50:])[0, 1]
    assert corr < 0.999 or abs(base.values[50:] - exp.values[50:]).mean() > 1e-4


def test_run_equity_expert_synthetic():
    feed = DailyReplayFeed.from_synthetic(
        ["SPY", "QQQ", "VIX"], n_days=500, seed=9, start="2018-01-02"
    )
    base_spec = {
        "id": "vt_base",
        "kind": "vol_target_hold",
        "underlying": "QQQ",
        "meta": {
            "vol_target": 0.15,
            "vol_lookback": 20,
            "max_leverage": 2.0,
            "leverage_base": 1.0,
            "leverage_high": 2.0,
            "signal_thresh": 0.0,
            "financing_rate": 0.06,
            "apply_financing": True,
            "apply_commissions": True,
            "hard_dd_cap": -0.5,
        },
    }
    exp_spec = {
        "id": "vt_exp",
        "kind": "vol_target_expert",
        "underlying": "QQQ",
        "meta": {
            **PRESETS["full_expert"].to_meta(),
            "vol_target": 0.15,
            "max_leverage": 2.0,
            "leverage_base": 1.0,
            "leverage_high": 2.0,
            "signal_thresh": 0.0,
            "financing_rate": 0.06,
            "apply_financing": True,
            "apply_commissions": True,
            "hard_dd_cap": -0.5,
        },
    }
    r1 = run_equity_spec(feed, base_spec, capital0=100_000.0)
    r2 = run_equity_spec(feed, exp_spec, capital0=100_000.0)
    assert r1.n_days > 100
    assert r2.n_days > 100
    assert r1.mean_leverage <= 2.01
    assert r2.mean_leverage <= 2.01
    assert r1.cost_drag_total >= 0
    assert r2.cost_drag_total >= 0
    # expert should mention notes
    assert any("vol_target_expert" in n for n in r2.notes)


def test_study_specs_nonempty():
    specs = iter_study_specs(
        underlyings=("QQQ",),
        vol_targets=(0.15,),
        max_leverages=(2.0,),
        presets=("ewma_only", "full_expert"),
    )
    assert len(specs) >= 3  # baseline + 2 presets
    kinds = {s["kind"] for s in specs}
    assert "vol_target_hold" in kinds
    assert "vol_target_expert" in kinds


def test_feature_gap_report():
    g = expert_feature_gap_report()
    assert "implied_vol" in g
    assert "VIX" in g["implied_vol"]
