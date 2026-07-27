"""Unit tests for universe_limit screen/confirm pure logic (no network/backtest)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_mod(*, reload: bool = True):
    path = ROOT / "scripts" / "run_universe_limit_screen_confirm.py"
    name = "universe_limit_sc"
    if reload and name in sys.modules:
        del sys.modules[name]
    if name in sys.modules and not reload:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Required so @dataclass can resolve annotations via sys.modules
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _arm(mod, limit: int, *, screen_cagr, screen_mdd, screen_n,
         confirm_cagr, confirm_mdd, confirm_n,
         full_cagr=None, full_mdd=None, full_n=None):
    def _win(cagr, mdd, n):
        m = {"cagr": cagr, "max_drawdown": mdd, "n_trades": n}
        return mod.ArmWindowResult(metrics=m, gates=mod.apply_path_gates(m))

    full_cagr = confirm_cagr if full_cagr is None else full_cagr
    full_mdd = confirm_mdd if full_mdd is None else full_mdd
    full_n = confirm_n if full_n is None else full_n
    return mod.LimitArm(
        universe_limit=limit,
        screen=_win(screen_cagr, screen_mdd, screen_n),
        confirm=_win(confirm_cagr, confirm_mdd, confirm_n),
        full=_win(full_cagr, full_mdd, full_n),
    )


def test_default_limits_prereg_invariant():
    mod = _load_mod()
    assert mod.DEFAULT_LIMITS == (40, 50, 60, 80)
    assert 54 not in mod.DEFAULT_LIMITS


def test_parse_limits_default_prereg():
    mod = _load_mod()
    assert mod.parse_limits("40,50,60,80") == [40, 50, 60, 80]
    assert mod.parse_limits("40, 50, 60, 80, 40") == [40, 50, 60, 80]


def test_parse_limits_rejects_empty_and_nonpositive():
    mod = _load_mod()
    with pytest.raises(ValueError):
        mod.parse_limits("")
    with pytest.raises(ValueError):
        mod.parse_limits("0,40")
    with pytest.raises(ValueError):
        mod.parse_limits("-1")


def test_validate_limits_blocks_54_and_unregistered():
    mod = _load_mod()
    assert mod.validate_limits([40, 50, 60, 80]) == [40, 50, 60, 80]
    assert mod.validate_limits([50, 40]) == [50, 40]  # subset/reorder OK
    with pytest.raises(ValueError, match="54"):
        mod.validate_limits([40, 54, 80])
    with pytest.raises(ValueError, match="54"):
        # 54 still banned even with allow_unregistered
        mod.validate_limits([54], allow_unregistered=True)
    with pytest.raises(ValueError, match="not in pre-registered"):
        mod.validate_limits([40, 70])
    assert mod.validate_limits([40, 70], allow_unregistered=True) == [40, 70]


def test_apply_path_gates_thresholds():
    mod = _load_mod()
    # Pass: CAGR 12%, MDD -50%, 60 trades
    g = mod.apply_path_gates(
        {"cagr": 0.12, "max_drawdown": -0.50, "n_trades": 60}
    )
    assert g["pass"] is True
    assert g["cagr_ok"] and g["mdd_ok"] and g["trades_ok"]

    # CAGR exactly 10% fails (must be > 10%)
    g2 = mod.apply_path_gates(
        {"cagr": 0.10, "max_drawdown": -0.50, "n_trades": 60}
    )
    assert g2["cagr_ok"] is False
    assert g2["pass"] is False

    # MDD deeper than -65% fails
    g3 = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": -0.70, "n_trades": 100}
    )
    assert g3["mdd_ok"] is False
    assert g3["pass"] is False

    # Trades below 50 fails
    g4 = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": -0.40, "n_trades": 49}
    )
    assert g4["trades_ok"] is False
    assert g4["pass"] is False

    # Boundary: MDD exactly -65% OK; n_trades 50 OK; CAGR 10.01% OK
    g5 = mod.apply_path_gates(
        {"cagr": 0.1001, "max_drawdown": -0.65, "n_trades": 50}
    )
    assert g5["pass"] is True


def test_apply_path_gates_zero_drawdown_is_pass():
    """max_drawdown=0.0 must not be treated as missing via truthiness."""
    mod = _load_mod()
    g = mod.apply_path_gates(
        {"cagr": 0.20, "max_drawdown": 0.0, "n_trades": 100}
    )
    assert g["mdd_ok"] is True
    assert g["pass"] is True

    # Missing MDD still fails (default -1.0)
    g_miss = mod.apply_path_gates({"cagr": 0.20, "n_trades": 100})
    assert g_miss["mdd_ok"] is False
    assert g_miss["pass"] is False


def test_rank_arms_prefer_confirm_passers_by_cagr():
    mod = _load_mod()
    # limit 80: confirm fails low CAGR
    a80 = _arm(
        mod, 80,
        screen_cagr=0.15, screen_mdd=-0.40, screen_n=100,
        confirm_cagr=0.044, confirm_mdd=-0.60, confirm_n=80,
        full_cagr=0.08, full_mdd=-0.55, full_n=180,
    )
    # limit 50: confirm pass high CAGR
    a50 = _arm(
        mod, 50,
        screen_cagr=0.12, screen_mdd=-0.45, screen_n=90,
        confirm_cagr=0.14, confirm_mdd=-0.50, confirm_n=70,
        full_cagr=0.13, full_mdd=-0.52, full_n=160,
    )
    # limit 60: confirm pass lower CAGR than 50
    a60 = _arm(
        mod, 60,
        screen_cagr=0.11, screen_mdd=-0.48, screen_n=85,
        confirm_cagr=0.11, confirm_mdd=-0.40, confirm_n=75,
        full_cagr=0.11, full_mdd=-0.45, full_n=160,
    )
    # limit 40: screen pass confirm fail
    a40 = _arm(
        mod, 40,
        screen_cagr=0.20, screen_mdd=-0.30, screen_n=100,
        confirm_cagr=0.05, confirm_mdd=-0.55, confirm_n=60,
        full_cagr=0.09, full_mdd=-0.50, full_n=160,
    )
    ranked = mod.rank_arms([a80, a50, a60, a40])
    assert [a.universe_limit for a in ranked] == [50, 60, 40, 80]
    assert ranked[0].confirm.passed and ranked[1].confirm.passed
    assert not ranked[2].confirm.passed
    assert not ranked[3].confirm.passed


def test_rank_arms_empty_and_decision_empty():
    mod = _load_mod()
    assert mod.rank_arms([]) == []
    d = mod.build_decision([])
    assert d["verdict"] == "FAIL"
    assert d["research_pass_candidate"] is False
    assert d["paper_freeze_unchanged"] is True
    assert d["best_confirm_and_full_limit"] is None
    assert d["best_full_path_only_limit"] is None
    assert d["confirm_pass_limits"] == []


def test_rank_arms_tie_break_mdd_then_trades():
    mod = _load_mod()
    # Same confirm CAGR; better (higher) MDD wins
    a1 = _arm(
        mod, 50,
        screen_cagr=0.12, screen_mdd=-0.40, screen_n=80,
        confirm_cagr=0.12, confirm_mdd=-0.55, confirm_n=60,
    )
    a2 = _arm(
        mod, 60,
        screen_cagr=0.12, screen_mdd=-0.40, screen_n=80,
        confirm_cagr=0.12, confirm_mdd=-0.45, confirm_n=60,
    )
    ranked = mod.rank_arms([a1, a2])
    assert ranked[0].universe_limit == 60
    assert ranked[1].universe_limit == 50

    # Same CAGR and MDD; more trades wins
    b1 = _arm(
        mod, 40,
        screen_cagr=0.12, screen_mdd=-0.40, screen_n=80,
        confirm_cagr=0.12, confirm_mdd=-0.50, confirm_n=55,
    )
    b2 = _arm(
        mod, 50,
        screen_cagr=0.12, screen_mdd=-0.40, screen_n=80,
        confirm_cagr=0.12, confirm_mdd=-0.50, confirm_n=90,
    )
    ranked2 = mod.rank_arms([b1, b2])
    assert ranked2[0].universe_limit == 50
    assert ranked2[1].universe_limit == 40


def test_build_decision_research_pass():
    mod = _load_mod()
    arms = [
        _arm(
            mod, 80,
            screen_cagr=0.08, screen_mdd=-0.50, screen_n=80,
            confirm_cagr=0.044, confirm_mdd=-0.60, confirm_n=80,
            full_cagr=0.05, full_mdd=-0.55, full_n=160,
        ),
        _arm(
            mod, 50,
            screen_cagr=0.12, screen_mdd=-0.45, screen_n=90,
            confirm_cagr=0.13, confirm_mdd=-0.48, confirm_n=70,
            full_cagr=0.125, full_mdd=-0.50, full_n=160,
        ),
    ]
    d = mod.build_decision(arms)
    assert d["research_pass_candidate"] is True
    assert d["verdict"] == "RESEARCH_PASS_CANDIDATE"
    assert d["best_confirm_limit"] == 50
    assert d["best_confirm_and_full_limit"] == 50
    assert d["best_full_limit"] == 50  # alias
    assert d["best_full_path_only_limit"] == 50
    assert d["paper_freeze_unchanged"] is True
    assert d["overfit_to_screen"] is False


def test_build_decision_overfit_screen():
    mod = _load_mod()
    arms = [
        _arm(
            mod, 40,
            screen_cagr=0.15, screen_mdd=-0.40, screen_n=80,
            confirm_cagr=0.03, confirm_mdd=-0.50, confirm_n=60,
            full_cagr=0.07, full_mdd=-0.50, full_n=140,
        ),
        _arm(
            mod, 80,
            screen_cagr=0.12, screen_mdd=-0.45, screen_n=90,
            confirm_cagr=0.04, confirm_mdd=-0.60, confirm_n=70,
            full_cagr=0.06, full_mdd=-0.55, full_n=160,
        ),
    ]
    d = mod.build_decision(arms)
    assert d["research_pass_candidate"] is False
    assert d["verdict"] == "FAIL"
    assert d["overfit_to_screen"] is True
    assert 40 in d["screen_only_pass_limits"]


def test_build_decision_full_pass_without_confirm_not_research_pass():
    """Stitched full may pass gates while confirm fails — still FAIL."""
    mod = _load_mod()
    arms = [
        _arm(
            mod, 50,
            screen_cagr=0.18, screen_mdd=-0.40, screen_n=500,
            confirm_cagr=0.073, confirm_mdd=-0.599, confirm_n=600,
            full_cagr=0.125, full_mdd=-0.599, full_n=1100,
        ),
    ]
    d = mod.build_decision(arms)
    assert arms[0].full.passed is True
    assert arms[0].confirm.passed is False
    assert d["research_pass_candidate"] is False
    assert 50 in d["full_path_pass_limits"]
    assert d["full_pass_limits"] == []
    assert d["best_confirm_and_full_limit"] is None
    assert d["best_full_path_only_limit"] == 50
    assert d["overfit_to_screen"] is True


def test_build_decision_capacity_sensitivity():
    mod = _load_mod()
    arms = [
        _arm(
            mod, 50,
            screen_cagr=0.12, screen_mdd=-0.45, screen_n=80,
            confirm_cagr=0.12, confirm_mdd=-0.50, confirm_n=70,
            full_cagr=0.12, full_mdd=-0.50, full_n=150,
        ),
        _arm(
            mod, 60,
            screen_cagr=0.11, screen_mdd=-0.48, screen_n=85,
            confirm_cagr=0.11, confirm_mdd=-0.52, confirm_n=75,
            full_cagr=0.11, full_mdd=-0.52, full_n=160,
        ),
        _arm(
            mod, 80,
            screen_cagr=0.10, screen_mdd=-0.50, screen_n=90,
            confirm_cagr=0.044, confirm_mdd=-0.60, confirm_n=80,
            full_cagr=0.06, full_mdd=-0.55, full_n=170,
        ),
    ]
    d = mod.build_decision(arms)
    assert d["capacity_sensitivity"] is True
    assert 50 in d["confirm_pass_limits"]
    assert 80 not in d["confirm_pass_limits"]
    assert d["research_pass_candidate"] is True  # 50/60 also pass full


def test_stitch_equity_capital_continuity():
    mod = _load_mod()
    idx_a = pd.date_range("2010-01-01", periods=5, freq="B", tz="UTC")
    idx_b = pd.date_range("2018-01-01", periods=5, freq="B", tz="UTC")
    # Screen ends at 120; confirm starts at 100 → scale by 1.2
    a = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=idx_a)
    b = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=idx_b)
    out = mod.stitch_equity(a, b)
    assert len(out) == 10
    assert out.iloc[4] == pytest.approx(120.0)
    assert out.iloc[5] == pytest.approx(120.0)  # 100 * 1.2
    assert out.iloc[-1] == pytest.approx(168.0)  # 140 * 1.2


def test_stitch_equity_empty_segment_keeps_other():
    mod = _load_mod()
    idx_a = pd.date_range("2010-01-01", periods=3, freq="B", tz="UTC")
    a = pd.Series([100.0, 110.0, 120.0], index=idx_a)
    empty = pd.Series(dtype=float)
    out = mod.stitch_equity(a, empty)
    assert len(out) == 3
    assert out.iloc[-1] == pytest.approx(120.0)
    out2 = mod.stitch_equity(empty, a)
    assert len(out2) == 3
    assert out2.iloc[0] == pytest.approx(100.0)
    assert mod.stitch_equity(empty, empty).empty


def test_stitch_trades_scales_confirm_pnl():
    mod = _load_mod()
    tr_a = pd.DataFrame({"net_profit": [10.0, -5.0], "oos_year": [2010, 2011]})
    tr_b = pd.DataFrame({"net_profit": [20.0, 30.0], "oos_year": [2018, 2019]})
    out = mod.stitch_trades(tr_a, tr_b, scale_b=1.5)
    assert len(out) == 4
    assert out.iloc[0]["net_profit"] == pytest.approx(10.0)
    assert out.iloc[2]["net_profit"] == pytest.approx(30.0)
    assert out.iloc[3]["net_profit"] == pytest.approx(45.0)
    assert out.iloc[2]["pnl_capital_scale"] == pytest.approx(1.5)


def test_capital_continuity_scale():
    mod = _load_mod()
    idx_a = pd.date_range("2010-01-01", periods=2, freq="B", tz="UTC")
    idx_b = pd.date_range("2018-01-01", periods=2, freq="B", tz="UTC")
    a = pd.Series([100.0, 200.0], index=idx_a)
    b = pd.Series([100.0, 150.0], index=idx_b)
    assert mod.capital_continuity_scale(a, b) == pytest.approx(2.0)


def test_missing_oos_years_from_equity_and_year_results():
    mod = _load_mod()
    idx = pd.DatetimeIndex(
        ["2010-06-01", "2011-06-01", "2013-06-01"], tz="UTC"
    )
    eq = pd.Series([100.0, 110.0, 120.0], index=idx)
    miss = mod.missing_oos_years(eq, 2010, 2013)
    assert miss == [2012]

    miss2 = mod.missing_oos_years(
        eq,
        2010,
        2013,
        year_results=[{"year": 2010}, {"year": 2011}, {"year": 2012}],
    )
    # Prefer year_results when provided
    assert miss2 == [2013]

    miss3 = mod.missing_oos_years(None, 2018, 2020)
    assert miss3 == [2018, 2019, 2020]


def test_resolve_ticker_file_prefers_passers(tmp_path: Path):
    mod = _load_mod()
    # Create fake root with both files; passers has enough names
    passers = tmp_path / "universe_longhist2010_pass.txt"
    longhist = tmp_path / "universe_longhist100.txt"
    passers.write_text("\n".join(f"T{i}" for i in range(50)), encoding="utf-8")
    longhist.write_text("\n".join(f"L{i}" for i in range(100)), encoding="utf-8")
    resolved = mod.resolve_ticker_file(longhist, root=tmp_path, min_n=40)
    assert resolved == passers

    # Explicit custom file is not swapped
    custom = tmp_path / "universe_highvol80.txt"
    custom.write_text("AAA\nBBB\n", encoding="utf-8")
    resolved2 = mod.resolve_ticker_file(custom, root=tmp_path, min_n=40)
    assert resolved2 == custom

    # Passers too small → keep longhist
    passers.write_text("\n".join(f"T{i}" for i in range(10)), encoding="utf-8")
    resolved3 = mod.resolve_ticker_file(longhist, root=tmp_path, min_n=40)
    assert resolved3.resolve() == longhist.resolve()
