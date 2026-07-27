"""Unit tests for week / week_risk config grids + promotion loader (no market data)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_mega_module():
    path = ROOT / "scripts" / "run_crash_entry_mega_study.py"
    spec = importlib.util.spec_from_file_location("crash_mega_study", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_promo_module():
    path = ROOT / "scripts" / "run_promotion_scorecard.py"
    spec = importlib.util.spec_from_file_location("promo_scorecard", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_week_grid_has_five_curated_configs():
    mod = _load_mega_module()
    cfgs = mod._build_config_grid("week")
    ids = [c["id"] for c in cfgs]
    assert len(ids) == 5
    assert "turbo_highvol_minalloc__baseline" in ids
    assert "turbo_highvol_minalloc__crash_rsi30_wr" in ids
    assert "turbo_highvol__wr_pack" in ids
    assert "turbo_highvol__crash_dd15" in ids
    assert "turbo_highvol__crash_rsi_or_dd15" in ids
    # no dual highvol pure baseline in week set
    assert "turbo_highvol__baseline" not in ids


def test_curated_alias_matches_week():
    mod = _load_mega_module()
    a = [c["id"] for c in mod._build_config_grid("week")]
    b = [c["id"] for c in mod._build_config_grid("curated")]
    assert a == b


def test_week_risk_ab_two_arms():
    mod = _load_mega_module()
    cfgs = mod._build_config_grid("week_risk")
    assert len(cfgs) == 2
    ids = {c["id"] for c in cfgs}
    assert "turbo_highvol_minalloc__baseline" in ids
    assert "turbo_highvol_minalloc__dd_circuit_25" in ids
    treat = next(c for c in cfgs if c["label"] == "dd_circuit_25")
    assert treat["extra_bt"]["max_portfolio_dd"] == pytest.approx(0.25)


def test_resolve_universe_limit_zero_is_full():
    mod = _load_mega_module()
    assert mod._resolve_universe_limit(0) is None
    assert mod._resolve_universe_limit(-1) is None
    assert mod._resolve_universe_limit(None) is None
    assert mod._resolve_universe_limit(40) == 40


def test_make_bt_carries_peak_equity_seed():
    mod = _load_mega_module()
    from trad_research.strategies import HighVolMinAllocStrategy

    strat = HighVolMinAllocStrategy()
    bt = mod._make_bt(
        strat,
        120_000.0,
        None,
        None,
        None,
        None,
        None,
        {"max_portfolio_dd": 0.25},
        peak_equity_seed=180_000.0,
    )
    assert bt.initial_capital == pytest.approx(120_000.0)
    assert bt.peak_equity_seed == pytest.approx(180_000.0)
    assert bt.max_portfolio_dd == pytest.approx(0.25)


def test_candidates_from_configs_dir_style_and_trades(tmp_path: Path):
    """Loader picks minalloc baseline as style; style row skips residual; trades attached."""
    promo = _load_promo_module()
    idx = pd.date_range("2020-01-01", periods=30, freq="B", tz="UTC")
    for name, level in [
        ("turbo_highvol_minalloc__baseline", 100_000.0),
        ("turbo_highvol__wr_pack", 110_000.0),
        ("turbo_highvol_minalloc__crash_rsi30_wr", 105_000.0),
    ]:
        d = tmp_path / name
        d.mkdir()
        eq = pd.Series([level * (1.001 ** i) for i in range(len(idx))], index=idx)
        eq.to_csv(d / "equity.csv", header=["equity"])
        pd.DataFrame({"net_profit": [100.0, -50.0, 80.0]}).to_csv(
            d / "trades.csv", index=False
        )

    cands = promo._candidates_from_configs_dir(tmp_path)
    assert len(cands) == 3
    by_name = {c["name"]: c for c in cands}
    assert by_name["turbo_highvol_minalloc__baseline"]["style"] is None  # skip residual
    assert by_name["turbo_highvol__wr_pack"]["style"] is not None
    assert "minalloc__baseline" in str(by_name["turbo_highvol__wr_pack"]["style"])
    assert by_name["turbo_highvol__wr_pack"]["trades"] is not None
    assert by_name["turbo_highvol__wr_pack"]["trades"].name == "trades.csv"
    assert by_name["turbo_highvol__wr_pack"]["style_key"] == (
        "turbo_highvol_minalloc__baseline"
    )


def test_candidates_explicit_style_name(tmp_path: Path):
    promo = _load_promo_module()
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    for name in ("a_style", "b_peer"):
        d = tmp_path / name
        d.mkdir()
        pd.Series(range(10), index=idx).to_csv(d / "equity.csv", header=["equity"])
    cands = promo._candidates_from_configs_dir(tmp_path, style_name="a_style")
    by_name = {c["name"]: c for c in cands}
    assert by_name["a_style"]["style"] is None
    assert by_name["b_peer"]["style"] is not None
    assert by_name["b_peer"]["style"].parent.name == "a_style"
