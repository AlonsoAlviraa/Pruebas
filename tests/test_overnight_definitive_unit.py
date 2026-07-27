"""Unit tests for overnight definitive search pure helpers (no network/backtest)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_mod(*, reload: bool = True):
    path = ROOT / "scripts" / "run_overnight_definitive_search.py"
    name = "overnight_definitive_search"
    if reload and name in sys.modules:
        del sys.modules[name]
    if name in sys.modules and not reload:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_apply_path_gates_thresholds():
    mod = _load_mod()
    g = mod.apply_path_gates(
        {"cagr": 0.12, "max_drawdown": -0.50, "n_trades": 100}
    )
    assert g["pass"] is True
    assert g["cagr_ok"] and g["mdd_ok"] and g["trades_ok"]

    # CAGR must be strictly > 10%
    g2 = mod.apply_path_gates(
        {"cagr": 0.10, "max_drawdown": -0.50, "n_trades": 100}
    )
    assert g2["cagr_ok"] is False
    assert g2["pass"] is False

    # MDD deeper than -65% fails
    g3 = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": -0.66, "n_trades": 100}
    )
    assert g3["mdd_ok"] is False

    # n_trades boundary: 80 required
    g4 = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": -0.40, "n_trades": 79}
    )
    assert g4["trades_ok"] is False
    g5 = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": -0.40, "n_trades": 80}
    )
    assert g5["trades_ok"] is True

    # MDD=0.0 is valid (not missing)
    g6 = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": 0.0, "n_trades": 80}
    )
    assert g6["mdd_ok"] is True
    assert g6["pass"] is True


def test_honest_score_formula():
    mod = _load_mod()
    # 2*0.10 + 1*0.5 + 0.5*0.2 = 0.2+0.5+0.1 = 0.8; mdd ok
    s = mod.honest_score(
        {"cagr": 0.10, "sortino": 0.5, "max_drawdown": -0.40},
        0.2,
    )
    assert abs(s - 0.8) < 1e-9

    # deep mdd penalty: mdd=-0.60 → penalty 2*(0.10)=0.2
    s2 = mod.honest_score(
        {"cagr": 0.10, "sortino": 0.5, "max_drawdown": -0.60},
        None,
    )
    assert abs(s2 - (0.2 + 0.5 - 0.2)) < 1e-9

    # negative excess ignored
    s3 = mod.honest_score(
        {"cagr": 0.10, "sortino": 0.0, "max_drawdown": -0.20},
        -0.5,
    )
    assert abs(s3 - 0.2) < 1e-9


def test_research_pass_requires_confirm_and_full():
    mod = _load_mod()
    rows = [
        {
            "arm_id": "a",
            "honest_score": 1.0,
            "confirm": {"cagr": 0.15, "gates": {"pass": True}},
            "full": {"cagr": 0.12, "gates": {"pass": True}},
        },
        {
            "arm_id": "b",
            "honest_score": 2.0,
            "confirm": {"cagr": 0.20, "gates": {"pass": True}},
            "full": {"cagr": 0.05, "gates": {"pass": False}},
        },
        {
            "arm_id": "c",
            "error": "boom",
            "confirm": {"gates": {"pass": True}},
            "full": {"gates": {"pass": True}},
        },
    ]
    assert mod.confirm_pass_ids(rows) == ["a", "b"]
    assert mod.research_pass_ids(rows) == ["a"]
    ranked = mod.rank_by_honest_score(rows)
    assert ranked[0]["arm_id"] == "b"


def test_decide_verdict_statuses():
    mod = _load_mod()
    pass_row = {
        "arm_id": "win",
        "honest_score": 1.5,
        "confirm": {"gates": {"pass": True}},
        "full": {"gates": {"pass": True}},
    }
    d = mod.decide_verdict([pass_row], complete=True)
    assert d["status"] == "PASS"
    assert d["live_claim"] is False
    assert d["paper_freeze"] == "turbo_highvol_minalloc"
    assert "win" in d["research_pass"]

    d2 = mod.decide_verdict(
        [
            {
                "arm_id": "c_only",
                "honest_score": 1.0,
                "confirm": {"gates": {"pass": True}},
                "full": {"gates": {"pass": False}},
            }
        ],
        complete=True,
    )
    assert d2["status"] == "HOLD"

    d3 = mod.decide_verdict([], complete=True)
    assert d3["status"] == "FAIL"

    d4 = mod.decide_verdict([], complete=False)
    assert d4["status"].startswith("PARTIAL_")


def test_prioritize_force_first():
    mod = _load_mod()
    ids = ["a", "turbo_strict__longhist_L80", "b"]
    assert mod.prioritize_arm_ids(ids, done=[])[0] == "turbo_strict__longhist_L80"
    assert mod.prioritize_arm_ids(ids, done=["turbo_strict__longhist_L80"]) == ["a", "b"]
    assert mod.prioritize_arm_ids(ids, done=ids) == []


def test_build_arms_includes_notches_no_54():
    mod = _load_mod()
    arms = mod.build_arms()
    ids = [a.arm_id for a in arms]
    assert "turbo_strict__longhist_L80" in ids
    assert "turbo_highvol_minalloc__longhist_L50" in ids
    # notches
    assert any("r2_residual_mom_mr0p02" in i or "r2_residual_mom_mr02" in i for i in ids)
    assert any("mr0p05" in i or "mr05" in i for i in ids)
    # no accidental limit 54
    assert all(a.universe_limit != 54 for a in arms)
    # no duplicate default 0.03 notch arm id collision with base
    assert "r2_residual_mom__longhist_L80" in ids
    notch_only = [a for a in arms if a.param_overrides]
    for a in notch_only:
        assert "min_resid" in a.param_overrides
        assert abs(float(a.param_overrides["min_resid"]) - 0.03) > 1e-9


def test_fixup_full_trade_count_from_windows():
    mod = _load_mod()
    row = {
        "arm_id": "x",
        "screen": {"cagr": 0.2, "max_drawdown": -0.3, "n_trades": 100, "sortino": 0.5},
        "confirm": {
            "cagr": 0.12,
            "max_drawdown": -0.4,
            "n_trades": 90,
            "sortino": 0.4,
            "excess_spy_total": 0.1,
            "gates": {"pass": True},
        },
        "full": {
            "cagr": 0.15,
            "max_drawdown": -0.45,
            "n_trades": 0,  # bug from trades=None
            "sortino": 0.4,
        },
        "honest_score": 0.5,
    }
    fixed = mod.fixup_full_trade_count(row)
    assert fixed["full"]["n_trades"] == 190
    assert fixed["full"]["gates"]["pass"] is True
    assert fixed["full"]["gates"]["trades_ok"] is True


def test_merge_progress_local_wins():
    mod = _load_mod()
    # confirm with cagr so rows count as success
    seed = [
        {
            "arm_id": "a",
            "honest_score": 1.0,
            "screen": {},
            "confirm": {"cagr": 0.1, "gates": {"pass": False}},
            "full": {},
        }
    ]
    local = [
        {
            "arm_id": "a",
            "honest_score": 9.0,
            "screen": {},
            "confirm": {"cagr": 0.2, "gates": {"pass": True}},
            "full": {},
        }
    ]
    done, failed, rows = mod.merge_progress_rows(local, seed)
    assert done == ["a"]
    assert failed == []
    # local wins arm payload (confirm cagr); honest_score is re-derived by fixup
    assert rows[0]["confirm"]["cagr"] == 0.2


def test_mdd_boundary_exactly_minus_65():
    mod = _load_mod()
    g = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": -0.65, "n_trades": 80}
    )
    assert g["mdd_ok"] is True
    assert g["pass"] is True
    g2 = mod.apply_path_gates(
        {"cagr": 0.15, "max_drawdown": -0.6500001, "n_trades": 80}
    )
    assert g2["mdd_ok"] is False


def test_metric_float_none_vs_zero():
    mod = _load_mod()
    assert mod.metric_float({"max_drawdown": None}, "max_drawdown", -1.0) == -1.0
    assert mod.metric_float({"max_drawdown": 0.0}, "max_drawdown", -1.0) == 0.0
    assert mod.metric_float({}, "max_drawdown", -1.0) == -1.0


def test_zoo_complete_set_membership():
    mod = _load_mod()
    assert mod.zoo_complete(["a", "b"], ["a", "b"]) is True
    assert mod.zoo_complete(["a", "b"], ["b", "a", "orphan"]) is True
    assert mod.zoo_complete(["a", "b"], ["a"]) is False
    # len(done) >= n_arms with orphan is NOT enough if missing planned
    assert mod.zoo_complete(["a", "b", "c"], ["x", "y", "z"]) is False
    assert mod.zoo_complete([], []) is True


def test_finalize_stop_reason_seeded_partial_does_not_block_complete():
    mod = _load_mod()
    ids = ["a", "b"]
    # After seed alone: incomplete
    assert (
        mod.finalize_stop_reason(
            ids, ["a"], prior_stop="seeded_partial"
        )
        == "incomplete"
    )
    # Full coverage after seed+run: complete even if prior was seeded_partial
    assert (
        mod.finalize_stop_reason(
            ids, ["a", "b"], prior_stop="seeded_partial"
        )
        == "complete"
    )
    # hours exhausted without coverage
    assert (
        mod.finalize_stop_reason(
            ids, ["a"], hours_exhausted=True
        )
        == "hours_exhausted"
    )
    # hours exhausted but all done → complete
    assert (
        mod.finalize_stop_reason(
            ids, ["a", "b"], hours_exhausted=True
        )
        == "complete"
    )


def test_is_run_complete_ignores_seeded_partial_stop():
    mod = _load_mod()
    ids = ["a", "b"]
    assert (
        mod.is_run_complete(ids, ["a", "b"], stop_reason="seeded_partial") is True
    )
    assert mod.is_run_complete(ids, ["a"], stop_reason="complete") is False
    assert (
        mod.is_run_complete(
            ids, ["a"], accept_errors=True, failed=["b"], stop_reason="incomplete"
        )
        is True
    )


def test_row_is_success_and_partition():
    mod = _load_mod()
    ok = {
        "arm_id": "ok",
        "confirm": {"cagr": 0.11, "max_drawdown": -0.3, "n_trades": 100},
    }
    err = {"arm_id": "bad", "error": "RuntimeError: boom"}
    empty = {"arm_id": "empty", "confirm": {}}
    assert mod.row_is_success(ok) is True
    assert mod.row_is_success(err) is False
    assert mod.row_is_success(empty) is False
    done, failed = mod.partition_done_failed([ok, err, empty])
    assert done == ["ok"]
    assert failed == ["bad", "empty"]


def test_error_not_sticky_in_done_record_outcome():
    mod = _load_mod()
    state: dict = {"done": [], "failed": [], "rows": []}
    bad = {"arm_id": "x", "error": "OSError: disk"}
    mod.record_arm_outcome(state, bad)
    assert "x" not in state["done"]
    assert "x" in state["failed"]
    # pending would include x when done is only success list
    pending = mod.prioritize_arm_ids(["x", "y"], done=state["done"])
    assert "x" in pending
    # success clears failed
    good = {
        "arm_id": "x",
        "confirm": {"cagr": 0.12, "gates": {"pass": True}},
        "honest_score": 1.0,
    }
    mod.record_arm_outcome(state, good)
    assert state["done"] == ["x"]
    assert "x" not in state["failed"]


def test_seed_from_redesign_filters_errors_and_unknown(tmp_path: Path):
    mod = _load_mod()
    seed = {
        "rows": [
            {
                "arm_id": "known_ok",
                "confirm": {"cagr": 0.1, "max_drawdown": -0.2, "n_trades": 90},
            },
            {"arm_id": "known_err", "error": "ValueError: x"},
            {
                "arm_id": "unknown_id",
                "confirm": {"cagr": 0.5, "max_drawdown": -0.1, "n_trades": 200},
            },
        ]
    }
    prog = tmp_path / "PROGRESS.json"
    prog.write_text(__import__("json").dumps(seed), encoding="utf-8")
    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    done, failed, rows = mod.seed_from_redesign_v2(
        seed_progress=prog,
        seed_arms=arms_dir,
        dest_arms=arms_dir,
        known_arm_ids=["known_ok", "known_err"],
    )
    assert done == ["known_ok"]
    assert failed == ["known_err"]
    assert {r["arm_id"] for r in rows} == {"known_ok", "known_err"}
    # fixup applied (trade count path still runs)
    assert all("arm_id" in r for r in rows)
