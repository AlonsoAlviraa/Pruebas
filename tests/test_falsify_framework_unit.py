"""Unit tests for Falsification Framework v1 (FALSIFY-01). Synthetic only."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trad_research.falsify.book_corr import book_correlation
from trad_research.falsify.config import FalsifyConfig
from trad_research.falsify.costs_capacity import capacity_check, cost_stress_grid
from trad_research.falsify.deflated_sharpe import (
    deflated_sharpe_ratio,
    dsr_passes,
    expected_max_sharpe,
)
from trad_research.falsify.feature_store import FeatureStore, register_ohlcv_basics
from trad_research.falsify.leakage import has_high_severity, scan_leakage
from trad_research.falsify.pipeline import FalsifyCandidate, run_falsify_suite
from trad_research.falsify.purged_cv import (
    CombinatorialPurgedCV,
    FoldIndices,
    PurgedKFold,
    embargo_gap_ok,
    purge_and_embargo_train,
    validate_folds_no_leakage,
)
from trad_research.falsify.research_memory import (
    ResearchMemory,
    is_allowed_temp_memory_path,
    resolve_memory_path,
)
from trad_research.falsify.scorecard import FalsifyReport, assemble_report


# ---------------------------------------------------------------------------
# Purged CV
# ---------------------------------------------------------------------------


def test_purged_kfold_no_overlap_and_embargo():
    n = 100
    embargo = 5
    purge = 3
    pk = PurgedKFold(n_splits=5, purge_bars=purge, embargo_bars=embargo)
    folds = list(pk.split(n))
    assert len(folds) == 5
    errs = validate_folds_no_leakage(
        folds, purge_bars=purge, embargo_bars=embargo, all_idx=np.arange(n)
    )
    assert errs == []
    for f in folds:
        tr = set(int(x) for x in f.train)
        te = set(int(x) for x in f.test)
        assert tr.isdisjoint(te)
        assert embargo_gap_ok(f.train, f.test, embargo, all_idx=np.arange(n))
        # Purge band: positional neighbors of test excluded from train
        all_sorted = np.arange(n)
        pos_of = {int(v): i for i, v in enumerate(all_sorted)}
        for t_idx in f.test:
            p = pos_of[int(t_idx)]
            for d in range(1, purge + 1):
                if p - d >= 0:
                    assert int(all_sorted[p - d]) not in tr
                if p + d < n:
                    assert int(all_sorted[p + d]) not in tr


def test_purge_and_embargo_train_explicit():
    all_idx = np.arange(20)
    test = np.array([10, 11, 12])
    train = purge_and_embargo_train(all_idx, test, purge_bars=2, embargo_bars=3)
    tr = set(int(x) for x in train)
    assert 10 not in tr and 11 not in tr and 12 not in tr
    assert 8 not in tr and 9 not in tr and 13 not in tr and 14 not in tr
    assert 15 not in tr
    assert 0 in tr and 19 in tr


def test_purge_embargo_gapped_indices_position_based():
    """Weekend-style gaps: labels 0,1,2,5,6,7,10,11 — ban by position not t±d arithmetic."""
    # Positions: 0:0, 1:1, 2:2, 3:5, 4:6, 5:7, 6:10, 7:11
    gapped = np.array([0, 1, 2, 5, 6, 7, 10, 11])
    test = np.array([5, 6])  # positions 3,4 in ordered stream
    train = purge_and_embargo_train(gapped, test, purge_bars=1, embargo_bars=1)
    tr = set(int(x) for x in train)
    # purge pos neighbors of 5,6 → pos2=2, pos5=7; embargo after block end pos4 → pos5=7
    assert 5 not in tr and 6 not in tr
    assert 2 not in tr  # purge left of 5
    assert 7 not in tr  # purge right of 6 / embargo
    # Arithmetic t-1 for label 5 would ban 4 (absent); must still ban position neighbor 2
    assert 0 in tr and 1 in tr
    assert 10 in tr or 11 in tr  # far side after embargo of 1 from pos4 may ban only 7
    assert embargo_gap_ok(train, test, 1, all_idx=gapped)


def test_combinatorial_purged_cv_no_overlap():
    n = 60
    cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_bars=2, embargo_bars=2)
    folds = list(cpcv.split(n))
    assert len(folds) == cpcv.n_combinations()
    assert (
        validate_folds_no_leakage(
            folds, purge_bars=2, embargo_bars=2, all_idx=np.arange(n)
        )
        == []
    )
    for f in folds:
        assert embargo_gap_ok(f.train, f.test, 2, all_idx=np.arange(n))


def test_validate_folds_detects_purge_band_leak():
    # Manually inject train index in purge band
    fold = FoldIndices(train=np.array([0, 1, 4, 5]), test=np.array([2, 3]), fold_id=0)
    errs = validate_folds_no_leakage(
        [fold], purge_bars=1, embargo_bars=0, all_idx=np.arange(6)
    )
    assert errs  # 1 is purge-adjacent to 2


def test_cv_constructors_value_errors():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=2, n_test_groups=2)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=1, n_test_groups=1)
    pk = PurgedKFold(n_splits=5)
    with pytest.raises(ValueError):
        list(pk.split(3))


# ---------------------------------------------------------------------------
# DSR
# ---------------------------------------------------------------------------


def test_dsr_decreases_as_n_trials_increases():
    sr = 1.2
    n_obs = 8 * 252
    d1 = deflated_sharpe_ratio(sr, n_trials=1, n_obs=n_obs, annualized=True)
    d10 = deflated_sharpe_ratio(sr, n_trials=10, n_obs=n_obs, annualized=True)
    d100 = deflated_sharpe_ratio(sr, n_trials=100, n_obs=n_obs, annualized=True)
    assert d1["dsr"] > d10["dsr"] > d100["dsr"]
    assert expected_max_sharpe(100, sr_std=0.1) > expected_max_sharpe(10, sr_std=0.1)


def test_expected_max_sharpe_n1_is_zero():
    assert expected_max_sharpe(1) == 0.0


def test_dsr_n_trials_required_and_positive():
    with pytest.raises(TypeError):
        deflated_sharpe_ratio(1.0, n_obs=100)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(1.0, n_trials=0, n_obs=100)
    with pytest.raises(ValueError):
        expected_max_sharpe(0)


def test_dsr_passes_threshold():
    good = {"dsr": 0.20}
    bad = {"dsr": 0.01}
    assert dsr_passes(good, dsr_min=0.05) is True
    assert dsr_passes(bad, dsr_min=0.05) is False


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_leakage_catches_future_named_column():
    rng = np.random.default_rng(0)
    n = 80
    fwd = rng.normal(size=n)
    df = pd.DataFrame({"noise": rng.normal(size=n), "fwd_ret_5": fwd})
    findings = scan_leakage(feature_frame=df, future_returns=fwd)
    assert has_high_severity(findings)
    assert any(f.code == "future_named_column" for f in findings)


def test_leakage_catches_exact_future_neutral_name_col1():
    """Exact future returns in column index ≥1 with neutral name → high severity."""
    rng = np.random.default_rng(1)
    n = 80
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    fwd = np.r_[close[5:] / close[:-5] - 1.0, np.full(5, np.nan)]
    # col0 noise, col1 exact forward with neutral name
    df = pd.DataFrame({"x0": rng.normal(size=n), "alpha_signal": fwd})
    findings = scan_leakage(feature_frame=df, future_returns=fwd)
    assert has_high_severity(findings)
    assert any(f.code == "equals_future_return" for f in findings)


def test_leakage_all_columns_vs_computed_forward_close():
    rng = np.random.default_rng(2)
    n = 100
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    h = 5
    leaked = np.full(n, np.nan)
    leaked[: n - h] = close[h:] / close[: n - h] - 1.0
    df = pd.DataFrame({"noise": rng.normal(size=n), "sneaky": leaked})
    findings = scan_leakage(feature_frame=df, close=close, forward_horizon=h)
    assert has_high_severity(findings)
    assert any(
        f.code == "equals_computed_forward_return" and f.details.get("feature") == "sneaky"
        for f in findings
    )


def test_leakage_train_test_overlap():
    findings = scan_leakage(train_idx=[1, 2, 3, 4], test_idx=[3, 4, 5])
    assert has_high_severity(findings)
    assert any(f.code == "train_test_overlap" for f in findings)


def test_leakage_label_in_features():
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] * 3)
    df = pd.DataFrame({"a": y.copy(), "b": np.arange(len(y), dtype=float)})
    findings = scan_leakage(feature_frame=df, label=y)
    assert any(f.code == "label_in_features" for f in findings)


def test_leakage_timestamp_not_sorted():
    ts = pd.to_datetime(
        ["2020-01-03", "2020-01-01", "2020-01-02"], utc=True
    )
    findings = scan_leakage(timestamps=ts)
    assert has_high_severity(findings)
    assert any(f.code == "timestamp_not_sorted" for f in findings)


def test_leakage_empty_feature_frame():
    findings = scan_leakage(feature_frame=pd.DataFrame())
    assert findings == [] or all(f.severity != "high" for f in findings)


# ---------------------------------------------------------------------------
# Book corr
# ---------------------------------------------------------------------------


def test_book_corr_identical_series():
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.01, 100)
    eq = pd.Series(100_000 * np.cumprod(1 + rets), index=idx)
    out = book_correlation(eq, eq.copy())
    assert out["corr"] == pytest.approx(1.0, abs=1e-9)
    assert out["beta"] == pytest.approx(1.0, abs=1e-6)
    assert out["kill_suggested"] is True
    assert out["reason"] == "style_clone_corr"


def test_book_corr_anti_correlated():
    idx = pd.date_range("2020-01-01", periods=80, freq="B", tz="UTC")
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 80)
    eq_a = pd.Series(100_000 * np.cumprod(1 + r), index=idx)
    eq_b = pd.Series(100_000 * np.cumprod(1 - r), index=idx)
    out = book_correlation(eq_a, eq_b, corr_kill=0.95)
    assert out["corr"] < 0
    assert out["kill_suggested"] is False
    assert out["reason"] == "ok"


# ---------------------------------------------------------------------------
# Research memory
# ---------------------------------------------------------------------------


def test_research_memory_append_count(tmp_path: Path):
    mem = ResearchMemory(tmp_path / "trials.jsonl")
    assert mem.count_trials() == 0
    mem.log_trial(name="a", metrics={"sharpe": 0.5}, verdict="HOLD", params={"k": 1})
    mem.log_trial(name="b", metrics={"sharpe": 0.1}, verdict="KILL", params={"k": 2})
    assert mem.count_trials() == 2
    trials = mem.list_trials()
    assert len(trials) == 2
    assert trials[-1]["n_trials_so_far"] == 2
    assert mem.effective_n_trials(configured=1) == 2
    assert mem.effective_n_trials(configured=1, include_current=True) == 3
    assert mem.effective_n_trials(configured=10, include_current=False) == 10


def test_research_memory_path_escape_rejected(tmp_path: Path):
    evil = Path("C:/Windows/System32/evil_trials.jsonl")
    with pytest.raises(ValueError, match="allowlisted|escapes"):
        ResearchMemory(evil)
    with pytest.raises(ValueError):
        resolve_memory_path(evil)
    # Whole temp tree is NOT open — bare gettempdir()/foo rejected
    bare = Path(tempfile.gettempdir()) / "not_falsify_or_pytest_trials.jsonl"
    with pytest.raises(ValueError):
        resolve_memory_path(bare)


def test_research_memory_tmp_allowed(tmp_path: Path):
    # pytest tmp_path contains "pytest" → allowed under narrow temp policy
    assert is_allowed_temp_memory_path(tmp_path / "ok.jsonl")
    p = resolve_memory_path(tmp_path / "ok.jsonl")
    assert p.name == "ok.jsonl"
    mem = ResearchMemory.temp()
    assert mem.path.is_file() or mem.path.parent.is_dir()
    assert "falsify" in str(mem.path).lower()


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


def test_scorecard_kill_when_leakage_fails():
    report = assemble_report(
        name="leaky",
        gates={"leakage": False, "dsr": True, "book_corr": True, "purged_cv": True},
        leakage_findings=[
            {
                "code": "equals_future_return",
                "severity": "high",
                "message": "future col",
                "details": {},
            }
        ],
        n_trials_used=3,
    )
    assert report.verdict == "KILL"
    assert any("leakage" in r for r in report.kill_reasons)
    md = report.to_markdown()
    assert "KILL" in md
    assert report.to_dict()["verdict"] == "KILL"


def test_scorecard_rejects_advance():
    with pytest.raises(ValueError, match="ADVANCE|KILL|HOLD"):
        FalsifyReport(name="x", verdict="ADVANCE")


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------


def test_feature_store_negative_lag_raises():
    store = FeatureStore()
    with pytest.raises(ValueError, match="asof_lag"):
        store.register("bad", lambda df: df["close"], asof_lag=-1)


def test_register_ohlcv_basics_default_no_double_lag():
    store = FeatureStore()
    register_ohlcv_basics(store, price_lag=0)
    assert store.specs["log_ret_1"].asof_lag == 0


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_cost_stress_grid_edge_dies():
    # Positive but tiny gross returns, high turnover → stress kills
    rets = np.full(100, 0.0002)
    out = cost_stress_grid(rets, turnover=0.5, base_cost_bps=50.0, mults=(1.0, 3.0))
    assert "grid" in out
    assert out["edge_dies_under_stress"] is True


# ---------------------------------------------------------------------------
# Suite integration
# ---------------------------------------------------------------------------


def _eq(mu: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, 0.01, n)
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(100_000 * np.cumprod(1.0 + rets), index=idx)


def test_run_falsify_suite_leakage_kills(tmp_path: Path):
    n = 120
    idx = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(2)
    rets = rng.normal(0.0008, 0.01, n)
    eq = pd.Series(100_000 * np.cumprod(1 + rets), index=idx)
    base = pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)
    fwd = pd.Series(rng.normal(0, 0.01, n), index=idx)
    feat = pd.DataFrame({"future_alpha": fwd.to_numpy(), "ok": rng.normal(size=n)}, index=idx)

    mem = ResearchMemory(tmp_path / "t.jsonl")
    cfg = FalsifyConfig(
        n_trials=1,
        dsr_min=0.0,
        book_corr_kill=0.999,
        memory_dir=tmp_path,
        cost_bps=1.0,
        cost_stress_mults=(1.0,),
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="leaky_cand",
            equity=eq,
            baseline_equity=base,
            feature_frame=feat,
            future_returns=fwd.to_numpy(),
            timestamps=list(idx),
            turnover=0.01,
        ),
        config=cfg,
        memory=mem,
        log_memory=True,
        n_trials_override=1,
    )
    assert rep.verdict == "KILL"
    assert rep.gates.get("leakage") is False
    assert mem.count_trials() == 1


def test_suite_hold_clean_candidate(tmp_path: Path):
    n = 400
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(9)
    rets = rng.normal(0.0015, 0.008, n)
    eq = pd.Series(100_000 * np.cumprod(1 + rets), index=idx)
    base_rets = rng.normal(0.0002, 0.01, n)
    base = pd.Series(100_000 * np.cumprod(1 + base_rets), index=idx)
    feat = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)

    cfg = FalsifyConfig(
        n_trials=1,
        dsr_min=0.05,
        book_corr_kill=0.95,
        memory_dir=tmp_path,
        cost_bps=5.0,
        cost_stress_mults=(1.0, 1.5),
        purge_bars=2,
        embargo_bars=2,
    )
    mem = ResearchMemory(tmp_path / "clean.jsonl")
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="clean",
            equity=eq,
            baseline_equity=base,
            feature_frame=feat,
            timestamps=list(idx),
            turnover=0.02,
            observed_sharpe=2.0,
            n_obs_sharpe=n,
        ),
        config=cfg,
        memory=mem,
        n_trials_override=1,
    )
    assert rep.verdict == "HOLD"
    assert rep.kill_reasons == []
    assert all(rep.gates.values())
    assert rep.cv is not None
    assert rep.cv.get("mode") == "structural_selfcheck"


def test_suite_dsr_kill(tmp_path: Path):
    eq = _eq(0.0001, 80, seed=4)
    base = _eq(0.0002, 80, seed=5)
    cfg = FalsifyConfig(
        dsr_min=0.99,  # impossible for weak SR
        n_trials=50,
        memory_dir=tmp_path,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
        book_corr_kill=0.99,
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="weak_dsr",
            equity=eq,
            baseline_equity=base,
            observed_sharpe=0.1,
            n_obs_sharpe=80,
            turnover=0.01,
        ),
        config=cfg,
        memory=ResearchMemory(tmp_path / "d.jsonl"),
        n_trials_override=50,
        log_memory=False,
    )
    assert rep.verdict == "KILL"
    assert rep.gates.get("dsr") is False


def test_suite_book_corr_style_clone_kill(tmp_path: Path):
    eq = _eq(0.001, 200, seed=11)
    cfg = FalsifyConfig(
        dsr_min=0.0,
        book_corr_kill=0.95,
        memory_dir=tmp_path,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="clone",
            equity=eq,
            baseline_equity=eq.copy(),
            observed_sharpe=2.0,
            n_obs_sharpe=200,
            turnover=0.01,
        ),
        config=cfg,
        memory=ResearchMemory(tmp_path / "c.jsonl"),
        n_trials_override=1,
        log_memory=False,
    )
    assert rep.verdict == "KILL"
    assert rep.gates.get("book_corr") is False
    assert any("book_corr" in r for r in rep.kill_reasons)


def test_suite_candidate_cv_overlap_kills(tmp_path: Path):
    eq = _eq(0.001, 100, seed=12)
    cfg = FalsifyConfig(
        dsr_min=0.0,
        purge_bars=0,
        embargo_bars=0,
        memory_dir=tmp_path,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
        book_corr_kill=0.99,
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="overlap_cv",
            equity=eq,
            train_idx=[0, 1, 2, 3, 4],
            test_idx=[3, 4, 5],  # overlap
            observed_sharpe=1.5,
            n_obs_sharpe=100,
            turnover=0.01,
        ),
        config=cfg,
        memory=ResearchMemory(tmp_path / "o.jsonl"),
        n_trials_override=1,
        log_memory=False,
    )
    assert rep.verdict == "KILL"
    assert rep.gates.get("purged_cv") is False
    assert rep.cv is not None
    assert rep.cv.get("mode") == "candidate_folds"


def test_suite_short_equity_kills(tmp_path: Path):
    eq = pd.Series([100.0, 101.0, 102.0])
    cfg = FalsifyConfig(
        min_obs_equity=20,
        dsr_min=0.0,
        memory_dir=tmp_path,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
    )
    rep = run_falsify_suite(
        FalsifyCandidate(name="short", equity=eq, observed_sharpe=1.0, n_obs_sharpe=3),
        config=cfg,
        memory=ResearchMemory(tmp_path / "s.jsonl"),
        n_trials_override=1,
        log_memory=False,
    )
    assert rep.verdict == "KILL"
    assert rep.gates.get("equity_length") is False


def test_suite_costs_kill(tmp_path: Path):
    # Strong reported sharpe but high turnover vs costs on tiny rets
    n = 100
    rets = np.full(n, 0.00015)
    eq = pd.Series(100_000 * np.cumprod(1 + rets))
    cfg = FalsifyConfig(
        dsr_min=0.0,
        cost_bps=40.0,
        cost_stress_mults=(1.0, 3.0),
        memory_dir=tmp_path,
        book_corr_kill=0.99,
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="costy",
            equity=eq,
            returns=rets,
            turnover=0.8,
            observed_sharpe=1.0,
            n_obs_sharpe=n,
        ),
        config=cfg,
        memory=ResearchMemory(tmp_path / "cost.jsonl"),
        n_trials_override=1,
        log_memory=False,
    )
    assert rep.verdict == "KILL"
    assert rep.gates.get("costs") is False


# ---------------------------------------------------------------------------
# Round-2: full sample universe, multi-horizon leakage, gaps E–H
# ---------------------------------------------------------------------------


def test_real_purged_kfold_folds_pass_validation_and_suite(tmp_path: Path):
    """Blocker A: correct PurgedKFold folds must not false-KILL on purge band."""
    n = 100
    purge, embargo = 3, 5
    pk = PurgedKFold(n_splits=5, purge_bars=purge, embargo_bars=embargo)
    folds = list(pk.split(n))
    universe = np.arange(n)
    errs = validate_folds_no_leakage(
        folds, purge_bars=purge, embargo_bars=embargo, all_idx=universe
    )
    assert errs == []
    # Issue I: all_idx=None must use dense [min,max] span, not train∪test
    errs_none = validate_folds_no_leakage(
        folds, purge_bars=purge, embargo_bars=embargo, all_idx=None
    )
    assert errs_none == []
    for f in folds:
        assert embargo_gap_ok(f.train, f.test, embargo, all_idx=universe)


def test_validate_folds_all_idx_none_dense_span_no_false_fail():
    """Direct caller with purge>0 and all_idx=None: correct folds still pass."""
    n = 60
    purge, embargo = 2, 3
    folds = list(PurgedKFold(n_splits=4, purge_bars=purge, embargo_bars=embargo).split(n))
    assert validate_folds_no_leakage(folds, purge_bars=purge, embargo_bars=embargo) == []

    eq = _eq(0.0012, n, seed=21)
    cfg = FalsifyConfig(
        n_splits=5,
        purge_bars=purge,
        embargo_bars=embargo,
        dsr_min=0.0,
        book_corr_kill=0.99,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
        memory_dir=tmp_path,
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="pkfold_ok",
            equity=eq,
            cv_folds=folds,
            sample_index=universe,
            observed_sharpe=1.5,
            n_obs_sharpe=n,
            turnover=0.01,
        ),
        config=cfg,
        memory=ResearchMemory(tmp_path / "pk.jsonl"),
        n_trials_override=1,
        log_memory=False,
    )
    assert rep.gates.get("purged_cv") is True
    assert rep.cv is not None
    assert rep.cv.get("mode") == "candidate_folds"
    assert rep.cv.get("passed") is True
    # Should not die solely on CV
    assert "purged_cv_candidate_folds" not in rep.kill_reasons


def test_leakage_one_day_forward_horizon():
    """Horizon 1 exact forward under neutral name is high severity."""
    rng = np.random.default_rng(42)
    n = 80
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    leaked = np.full(n, np.nan)
    leaked[: n - 1] = close[1:] / close[: n - 1] - 1.0
    df = pd.DataFrame({"noise": rng.normal(size=n), "signal": leaked})
    findings = scan_leakage(feature_frame=df, close=close, forward_horizons=(1, 5, 10))
    assert has_high_severity(findings)
    assert any(
        f.code == "equals_computed_forward_return" and f.details.get("horizon") == 1
        for f in findings
    )


def test_book_corr_insufficient_overlap_no_kill(tmp_path: Path):
    # Non-overlapping date indices → skip, not style-clone KILL
    a = pd.Series(
        100_000 * np.cumprod(1 + np.full(30, 0.001)),
        index=pd.date_range("2018-01-01", periods=30, freq="B", tz="UTC"),
    )
    b = pd.Series(
        100_000 * np.cumprod(1 + np.full(30, 0.001)),
        index=pd.date_range("2022-01-01", periods=30, freq="B", tz="UTC"),
    )
    out = book_correlation(a, b)
    assert out.get("reason") == "insufficient_overlap"
    assert out.get("kill_suggested") is False

    cfg = FalsifyConfig(
        dsr_min=0.0,
        book_corr_kill=0.95,
        memory_dir=tmp_path,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
        min_obs_equity=10,
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="misaligned",
            equity=a,
            baseline_equity=b,
            observed_sharpe=1.0,
            n_obs_sharpe=30,
            turnover=0.01,
        ),
        config=cfg,
        memory=ResearchMemory(tmp_path / "mis.jsonl"),
        n_trials_override=1,
        log_memory=False,
    )
    assert rep.gates.get("book_corr") is True
    assert not any("book_corr" in r for r in rep.kill_reasons)


def test_leakage_feature_after_label_time():
    n = 20
    feat_t = pd.date_range("2020-01-10", periods=n, freq="D", tz="UTC")
    # labels earlier than features → leakage
    lab_t = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    findings = scan_leakage(feature_times=feat_t, label_event_times=lab_t)
    assert has_high_severity(findings)
    assert any(f.code == "feature_after_label_time" for f in findings)


def test_suite_n_trials_from_memory(tmp_path: Path):
    mem = ResearchMemory(tmp_path / "seed.jsonl")
    for i in range(4):
        mem.log_trial(name=f"prior_{i}", metrics={"sharpe": 0.1}, verdict="KILL", params={"i": i})
    assert mem.count_trials() == 4
    eq = _eq(0.001, 120, seed=33)
    cfg = FalsifyConfig(
        n_trials=1,
        dsr_min=0.0,
        memory_dir=tmp_path,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
        book_corr_kill=0.99,
    )
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="mem_n",
            equity=eq,
            observed_sharpe=1.2,
            n_obs_sharpe=120,
            turnover=0.01,
        ),
        config=cfg,
        memory=mem,
        log_memory=True,
        # no n_trials_override — must use memory include_current
    )
    # 4 prior + current = 5
    assert rep.n_trials_used == 5
    assert rep.dsr is not None
    assert int(rep.dsr["n_trials"]) == 5
    assert mem.count_trials() == 5


def test_capacity_breach_gate(tmp_path: Path):
    eq = _eq(0.001, 80, seed=44)
    cfg = FalsifyConfig(
        dsr_min=0.0,
        capacity_adv_pct=0.01,
        memory_dir=tmp_path,
        cost_bps=0.0,
        cost_stress_mults=(1.0,),
        book_corr_kill=0.99,
    )
    # Order notional >> 1% ADV
    rep = run_falsify_suite(
        FalsifyCandidate(
            name="cap",
            equity=eq,
            observed_sharpe=1.0,
            n_obs_sharpe=80,
            turnover=0.01,
            order_notional=np.array([1_000_000.0, 500_000.0]),
            adv_dollars=np.array([1_000_000.0, 1_000_000.0]),  # 1% cap = 10k
        ),
        config=cfg,
        memory=ResearchMemory(tmp_path / "cap.jsonl"),
        n_trials_override=1,
        log_memory=False,
    )
    assert rep.verdict == "KILL"
    assert rep.gates.get("capacity") is False
    # direct helper
    chk = capacity_check([1e6], [1e6], adv_pct=0.01)
    assert chk["capacity_ok"] is False

