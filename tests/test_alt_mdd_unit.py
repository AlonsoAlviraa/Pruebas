"""Unit tests for alt-loop MDD levers + alt_mdd grid + DD circuit behavior."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trad_research.risk_levers import (
    ALT_PRIMARY_LEVER_ID,
    apply_risk_mdd_lever,
    alt_mdd_extra_bt_for_strategy,
    alt_mdd_lever_ids,
    get_lever,
    list_levers,
    resolve_peak_equity_seed,
    update_peak_equity_state,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_mega():
    path = ROOT / "scripts" / "run_crash_entry_mega_study.py"
    spec = importlib.util.spec_from_file_location("crash_mega_alt", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _AlwaysBuy:
    """Minimal strategy: always long, high score."""

    feature_names = None

    def generate_signals(self, df, cfg):
        sig = pd.Series(True, index=df.index)
        score = pd.Series(0.95, index=df.index)
        return sig, score


def _panel(
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    ticker: str = "AAA",
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
            "atr": np.maximum(close * 0.02, 0.5),
            "sma_50": close,
            "sma_20": close,
            "dist_sma200": 0.05,
            "dist_sma_200": 0.05,
            "momentum_20": 0.05,
            "atr_norm": 0.02,
            "rsi_14": 50.0,
            "volatility_20": 0.02,
            "ret_5": 0.0,
            "ret_20": 0.0,
            "ret_1m": 0.05,
            "volume_z": 0.0,
            "high_low_range": 0.02,
        }
    )


def _loose_cfg(**kwargs):
    from trad_research.backtest import BacktestConfig

    base = dict(
        initial_capital=100_000.0,
        min_confidence=0.1,
        require_trend=False,
        require_momentum=False,
        require_regime=False,
        max_atr_pct=1.0,
        min_dist_sma200=-1.0,
        commission=0.0,
        slippage=0.0,
        volatility_target_pct=0.05,
        max_position_pct=0.50,
        max_positions=4,
        max_horizon=5,
        hard_stop_pct=0.99,
        k_atr=10.0,
        min_alloc_pct=0.0,
    )
    base.update(kwargs)
    return BacktestConfig(**base)


def test_alt_levers_registered():
    ids = list_levers()
    assert "dd25_vt70" in ids
    assert "dd20_vt60" in ids
    assert "dd18_vt70_pos75" in ids
    assert ALT_PRIMARY_LEVER_ID == "dd25_vt70"


def test_dd25_vt70_scales_vol_and_sets_dd():
    base = {"volatility_target_pct": 0.04, "max_position_pct": 0.22}
    out = apply_risk_mdd_lever(base, "dd25_vt70")
    assert out["max_portfolio_dd"] == pytest.approx(0.25)
    assert out["volatility_target_pct"] == pytest.approx(0.028)
    assert out["max_position_pct"] == pytest.approx(0.22)
    assert out["dd_soft_scale"] == pytest.approx(0.50)


def test_dd18_scales_position():
    base = {"volatility_target_pct": 0.04, "max_position_pct": 0.20}
    out = apply_risk_mdd_lever(base, "dd18_vt70_pos75")
    assert out["max_portfolio_dd"] == pytest.approx(0.18)
    assert out["volatility_target_pct"] == pytest.approx(0.028)
    assert out["max_position_pct"] == pytest.approx(0.15)


def test_dd20_vt60():
    base = {"volatility_target_pct": 0.05}
    out = apply_risk_mdd_lever(base, "dd20_vt60")
    assert out["max_portfolio_dd"] == pytest.approx(0.20)
    assert out["volatility_target_pct"] == pytest.approx(0.03)
    assert out["dd_soft_scale"] == pytest.approx(0.45)


def test_apply_clears_stale_dd_breach_size_scale():
    """Hard lever must not inherit soft scale from contaminated base_overrides."""
    base = {
        "volatility_target_pct": 0.04,
        "dd_breach_size_scale": 0.30,
    }
    hard = apply_risk_mdd_lever(base, "dd25_vt70_yr")
    assert hard["dd_breach_size_scale"] is None
    soft = apply_risk_mdd_lever(base, "dd25_vt70_soft")
    assert soft["dd_breach_size_scale"] == pytest.approx(0.30)


def test_alt_mdd_extra_bt_uses_strategy_overrides():
    from trad_research.strategies import HighVolMinAllocStrategy

    ov = HighVolMinAllocStrategy().backtest_overrides()
    out = alt_mdd_extra_bt_for_strategy(ov, "dd25_vt70")
    assert out["max_portfolio_dd"] == pytest.approx(0.25)
    assert "volatility_target_pct" in out
    assert out["volatility_target_pct"] < ov["volatility_target_pct"]


def test_alt_mdd_grid_shape():
    mod = _load_mega()
    cfgs = mod._build_config_grid("alt_mdd")
    ids = [c["id"] for c in cfgs]
    assert "turbo_highvol_minalloc__baseline" in ids
    assert f"turbo_highvol_minalloc__{ALT_PRIMARY_LEVER_ID}" in ids
    assert "turbo_highvol_minalloc__dd20_vt60" in ids
    assert "turbo_highvol_minalloc__dd18_vt70_pos75" in ids
    assert "turbo_highvol_minalloc__breadth40_dd25_vt70" in ids
    assert "turbo_highvol_minalloc__crash_rsi30_wr_dd25" in ids
    assert all(c["base"] == "turbo_highvol_minalloc" for c in cfgs)
    b = next(c for c in cfgs if "breadth" in c["label"])
    assert b["breadth"] is not None
    assert b["breadth"].min_breadth == pytest.approx(0.40)
    for lid in alt_mdd_lever_ids():
        assert any(c["label"] == lid for c in cfgs)


def test_alt_loop_alias():
    mod = _load_mega()
    a = [c["id"] for c in mod._build_config_grid("alt_mdd")]
    b = [c["id"] for c in mod._build_config_grid("alt_loop")]
    assert a == b


def test_primary_lever_description():
    lev = get_lever(ALT_PRIMARY_LEVER_ID)
    assert "0.70" in lev.description or "×0.70" in lev.description
    assert lev.max_portfolio_dd == pytest.approx(0.25)


def test_loop2_soft_breach_and_yearly():
    soft = apply_risk_mdd_lever(
        {"volatility_target_pct": 0.04}, "dd25_vt70_soft"
    )
    assert soft["dd_breach_size_scale"] == pytest.approx(0.30)
    assert soft["max_portfolio_dd"] == pytest.approx(0.25)
    assert soft["_peak_mode"] == "continuous"

    yr = apply_risk_mdd_lever({"volatility_target_pct": 0.04}, "dd25_vt70_yr")
    assert yr["_peak_mode"] == "yearly"
    assert yr.get("dd_breach_size_scale") is None


def test_alt_mdd_v2_grid():
    mod = _load_mega()
    cfgs = mod._build_config_grid("alt_mdd_v2")
    ids = [c["id"] for c in cfgs]
    assert "turbo_highvol_minalloc__baseline" in ids
    assert "turbo_highvol_minalloc__dd25_vt70_yr" in ids
    assert "turbo_highvol_minalloc__dd25_vt70_soft" in ids
    assert "turbo_highvol_minalloc__vt60_only" in ids
    assert "turbo_highvol_minalloc__dd35_vt80_yr" in ids
    assert "turbo_highvol_minalloc__breadth40_dd25_vt70_yr" in ids
    yr = next(c for c in cfgs if c["label"] == "dd25_vt70_yr")
    assert yr["peak_mode"] == "yearly"
    soft = next(c for c in cfgs if c["label"] == "dd25_vt70_soft")
    assert soft["extra_bt"].get("dd_breach_size_scale") == pytest.approx(0.30)
    assert "_peak_mode" not in soft["extra_bt"]


def test_resolve_and_update_peak_mode_helpers():
    """Multi-year peak seed: continuous carries HWM; yearly next seed is None."""
    # year1 continuous
    seed1 = resolve_peak_equity_seed("continuous", None)
    assert seed1 is None
    peak1 = update_peak_equity_state(
        "continuous", None, segment_hi=150_000.0, ending_capital=140_000.0, initial_capital=100_000.0
    )
    assert peak1 == pytest.approx(150_000.0)
    seed2 = resolve_peak_equity_seed("continuous", peak1)
    assert seed2 == pytest.approx(150_000.0)
    peak2 = update_peak_equity_state(
        "continuous",
        peak1,
        segment_hi=160_000.0,
        ending_capital=155_000.0,
        initial_capital=140_000.0,
    )
    assert peak2 == pytest.approx(160_000.0)

    # yearly: always re-seed None even if stored peak is high
    stored = update_peak_equity_state(
        "yearly", None, segment_hi=180_000.0, ending_capital=120_000.0, initial_capital=100_000.0
    )
    assert stored == pytest.approx(180_000.0)
    assert resolve_peak_equity_seed("yearly", stored) is None


def test_dd_breach_soft_allows_entries():
    """Hard breach blocks new entries when deep underwater; soft recovery allows them."""
    from trad_research.backtest import run_portfolio_backtest

    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    # Mild drift so positions can open; circuit is driven by peak seed, not path
    close = 100.0 * np.cumprod(1.0 + np.full(n, 0.0005))
    panels = {"AAA": _panel(close, dates), "BBB": _panel(close * 1.01, dates)}

    common = dict(
        peak_equity_seed=200_000.0,  # already −50% if eq~100k
        max_portfolio_dd=0.25,
        max_horizon=8,
        max_positions=4,
        volatility_target_pct=0.08,
        max_position_pct=0.40,
    )
    cfg_hard = _loose_cfg(dd_breach_size_scale=None, **common)
    cfg_soft = _loose_cfg(dd_breach_size_scale=0.30, **common)

    model = _AlwaysBuy()
    trades_h, eq_h, _ = run_portfolio_backtest(panels, model, cfg_hard)
    trades_s, eq_s, _ = run_portfolio_backtest(panels, model, cfg_soft)

    n_hard = 0 if trades_h is None or trades_h.empty else len(trades_h)
    n_soft = 0 if trades_s is None or trades_s.empty else len(trades_s)
    assert n_hard == 0, f"hard block should prevent entries under deep seed DD, got {n_hard}"
    assert n_soft > 0, "soft recovery must allow reduced-size entries under same DD"
    assert not eq_s.empty


def test_peak_ratchets_on_full_book_days():
    """When max_positions full, rising MTM still lifts peak so later DD circuit uses true HWM.

    Path: enter 1 name early → prices rise while full → crash → exit on horizon →
    re-entry attempted while still deep vs HWM. Hard circuit at 25% must block re-entry
    only if peak ratcheted during the full-book rise (EOD MTM path).
    """
    from trad_research.backtest import run_portfolio_backtest

    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    # Rise hard for ~40 bars then crash ~40%
    rets = np.concatenate(
        [
            np.full(35, 0.012),
            np.full(10, -0.04),
            np.full(n - 45, 0.001),
        ]
    )
    close = 50.0 * np.cumprod(1.0 + rets)
    # Two always-buy names; only 1 slot → full book during the rise
    panels = {
        "AAA": _panel(close, dates),
        "BBB": _panel(close * 1.02, dates),
    }

    cfg = _loose_cfg(
        max_positions=1,
        max_horizon=30,  # hold through rise; exit after crash starts
        max_portfolio_dd=0.25,
        dd_breach_size_scale=None,
        volatility_target_pct=0.15,
        max_position_pct=0.90,
        hard_stop_pct=0.99,
        k_atr=50.0,
    )
    trades, equity, _ = run_portfolio_backtest(panels, _AlwaysBuy(), cfg)
    assert equity is not None and not equity.empty
    # Equity path must show a meaningful rise (full-book MTM) then drawdown
    peak_eq = float(equity.max())
    assert peak_eq > 110_000.0, f"expected equity peak well above capital, got {peak_eq}"
    # After first exit(s), hard circuit should suppress re-entry while deep vs HWM:
    # total trades should be limited (first wave only), not unlimited re-churn.
    n_tr = 0 if trades is None or trades.empty else len(trades)
    assert n_tr >= 1
    # With correct HWM, deep post-crash DD blocks most re-entries
    if "entry_date" in trades.columns:
        late = trades[pd.to_datetime(trades["entry_date"], utc=True) > dates[50]]
        # Late re-entries should be rare/zero when peak was ratcheted during full-book rise
        assert len(late) <= 1, (
            f"peak ratchet should block late re-entries after crash; late={len(late)}"
        )
