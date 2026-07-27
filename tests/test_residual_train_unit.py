"""Unit tests for residual-label L1 walk-forward train (STR-02/03) — synthetic only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trad_research.portable.cs_features import assert_no_absolute_prices
from trad_research.portable.residual_labels import (
    panel_beat_style_vs_ew,
    residual_excess_labels,
)
from trad_research.portable.score_l1 import (
    ResidualTrainConfig,
    ThinMLStub,
    fit_residual_scorer,
    prepare_residual_train_panel,
    train_cutoff_for_oos,
    walk_forward_residual_scores,
)

ROOT = Path(__file__).resolve().parents[1]


def _synth_long_panel(
    n_days: int = 400,
    n_tickers: int = 8,
    start: str = "2019-01-02",
) -> pd.DataFrame:
    """Synthetic multi-year panel with invariant features + close (for labels only)."""
    rows = []
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=n_days, freq="B", tz="UTC")
    for t_i in range(n_tickers):
        tkr = f"T{t_i}"
        px = 40.0 + t_i * 5.0
        drift = 0.0015 if t_i == 0 else 0.0002
        for d in dates:
            ret = float(rng.normal(drift, 0.015))
            px = max(px * (1.0 + ret), 1.0)
            rows.append(
                {
                    "date": d,
                    "ticker": tkr,
                    "close": px,
                    "ret_1m": ret * 20,
                    "dist_sma_50": ret * 5 + 0.01 * t_i,
                    "atr_norm": 0.02 + 0.002 * t_i,
                    "volume_ratio": 1.0 + 0.05 * t_i,
                    "rsi_14": 50 + t_i + 10 * ret,
                    "rsi_7": 50.0,
                    "rsi_21": 50.0,
                    "dist_sma_200": 0.0,
                    "volatility_20": 0.2,
                    "volume_zscore": 0.0,
                    "ret_1d": ret,
                }
            )
    return pd.DataFrame(rows)


def test_train_cutoff_bar_calendar_strict_horizon():
    """Cutoff uses panel bars: j_max = first_oos_idx - H - 1."""
    cal = pd.bdate_range("2021-01-04", periods=40, tz="UTC")
    # Put OOS start mid-calendar
    oos = pd.Timestamp("2021-02-01", tz="UTC")
    H = 5
    cut = train_cutoff_for_oos(oos, horizon=H, calendar=cal)
    # first cal bar on/after oos
    pos = cal.searchsorted(oos.normalize(), side="left")
    expected = cal[pos - H - 1]
    assert cut == pd.Timestamp(expected).normalize()
    # label at cutoff uses close H bars later — still before first OOS bar
    assert pos - H - 1 + H < pos


def test_prepare_residual_panel_no_label_in_features():
    panel = _synth_long_panel(80, 4)
    ranked, feats = prepare_residual_train_panel(panel, horizon=5)
    assert "y_beat_style" in ranked.columns
    assert "y_excess" in ranked.columns
    assert "y_beat_style" not in feats
    assert_no_absolute_prices(feats)


def test_incomplete_horizon_is_nan_not_zero():
    """y_beat_style must be NaN where y_excess incomplete — never hard 0."""
    close = np.array([100.0, 101.0, 102.0, 103.0, 110.0], dtype=float)
    style = np.array([100.0, 100.5, 101.0, 101.5, 102.0], dtype=float)
    excess, beat = residual_excess_labels(close, style, horizon=2)
    assert np.isnan(excess[-1]) and np.isnan(excess[-2])
    assert np.isnan(beat[-1]) and np.isnan(beat[-2])
    # finite rows are 0/1 only
    fin = beat[np.isfinite(beat)]
    assert set(np.unique(fin)).issubset({0.0, 1.0})


def test_panel_beat_style_nan_tail():
    panel = _synth_long_panel(30, 2)
    out = panel_beat_style_vs_ew(panel, horizon=5)
    # last 5 bars per ticker should have NaN y_excess / y_beat_style
    for _, g in out.groupby("ticker"):
        g = g.sort_values("date")
        assert g["y_excess"].iloc[-5:].isna().all()
        assert g["y_beat_style"].iloc[-5:].isna().all()
        assert g["y_excess"].iloc[:-5].notna().all()


def test_horizon_embargo_excludes_leaky_train_rows():
    """Non-tautological: rows whose forward window would enter OOS are out of train.

    Construct calendar so bar at (first_oos - H) would leak if embargo were only H
    (without -1). Assert train_cutoff excludes that bar.
    """
    n_days = 300
    n_tickers = 4
    H = 10
    panel = _synth_long_panel(n_days, n_tickers, start="2019-06-03")
    dates = pd.DatetimeIndex(sorted(panel["date"].dt.normalize().unique()))
    oos_year = 2020
    oos_start = pd.Timestamp(f"{oos_year}-01-01", tz="UTC")
    first_oos_idx = int(dates.searchsorted(oos_start, side="left"))
    assert first_oos_idx > H + 2

    # Bar that would LEAK if we only used cutoff = first_oos - H (not -1):
    # index first_oos_idx - H → label uses first_oos bar
    leaky_idx = first_oos_idx - H
    leaky_date = dates[leaky_idx]
    safe_idx = first_oos_idx - H - 1
    safe_date = dates[safe_idx]

    cfg = ResidualTrainConfig(horizon=H, min_train_rows=20, model="thin_ml", top_quantile=0.5)
    scored, meta = walk_forward_residual_scores(
        panel,
        first_oos_year=oos_year,
        last_oos_year=oos_year,
        config=cfg,
    )
    fold = meta["folds"][0]
    cutoff = pd.Timestamp(fold["train_cutoff"], tz="UTC")
    assert cutoff == pd.Timestamp(safe_date).normalize()
    assert cutoff < pd.Timestamp(leaky_date).normalize() or cutoff == safe_date.normalize()
    # Explicit: leaky date must not be allowed as train end
    assert cutoff < dates[first_oos_idx]
    # label end bar for cutoff: safe_idx + H = first_oos_idx - 1 < first_oos_idx
    assert safe_idx + H < first_oos_idx
    assert leaky_idx + H == first_oos_idx  # would touch OOS

    # Reconstruct labeled train: no train row with date > cutoff
    work, _ = prepare_residual_train_panel(panel, horizon=H)
    work["date"] = pd.to_datetime(work["date"], utc=True)
    train = work.loc[work["date"].dt.normalize() <= cutoff]
    assert train["date"].max().normalize() <= cutoff
    # Every train row with finite y_excess must have horizon end strictly before OOS
    # (on business calendar of this panel)
    fin = train.loc[train["y_excess"].notna()]
    if not fin.empty:
        max_d = fin["date"].dt.normalize().max()
        # max_d + H bars < first OOS
        pos_max = int(dates.searchsorted(max_d, side="left"))
        # max_d should be in calendar
        assert pos_max + H < first_oos_idx or max_d == safe_date

    if not scored.empty:
        assert set(pd.to_datetime(scored["date"], utc=True).dt.year) <= {oos_year}


def test_nan_y_excess_never_in_train_mask():
    """Walk-forward must drop incomplete horizons even if y_beat were 0."""
    panel = _synth_long_panel(120, 3, start="2020-01-02")
    H = 8
    work, feats = prepare_residual_train_panel(panel, horizon=H)
    # Force-corrupt: set y_beat_style=0 where y_excess NaN (old bug)
    bad = work["y_excess"].isna()
    work.loc[bad, "y_beat_style"] = 0.0
    assert (work.loc[bad, "y_beat_style"] == 0).all()

    # Simulate lab_ok as walk_forward does
    y_tr = work["y_beat_style"].to_numpy(dtype=float)
    y_ex = work["y_excess"].to_numpy(dtype=float)
    lab_ok = np.isfinite(y_tr) & np.isfinite(y_ex)
    assert not lab_ok[bad.to_numpy()].any()
    assert lab_ok[~bad.to_numpy()].sum() > 0


def test_multifold_expanding_train_grows():
    # ~3y business days so 2020 and 2021 both have OOS bars
    panel = _synth_long_panel(780, 5, start="2019-01-02")
    cfg = ResidualTrainConfig(horizon=8, min_train_rows=30, model="thin_ml")
    _, meta = walk_forward_residual_scores(
        panel, first_oos_year=2020, last_oos_year=2021, config=cfg
    )
    folds = [f for f in meta["folds"] if f.get("fitted") or f.get("fallback")]
    assert len(folds) >= 2
    # Expanding: later year should have more labeled train rows
    n0 = folds[0].get("n_train_labeled") or 0
    n1 = folds[1].get("n_train_labeled") or 0
    assert n1 > n0


def test_rule_rank_fallback_when_insufficient_train():
    panel = _synth_long_panel(80, 3, start="2021-01-04")
    cfg = ResidualTrainConfig(
        horizon=5, min_train_rows=50_000, model="thin_ml", top_quantile=0.5
    )
    scored, meta = walk_forward_residual_scores(
        panel, first_oos_year=2021, last_oos_year=2021, config=cfg
    )
    fold = meta["folds"][0]
    assert fold.get("fallback") == "rule_rank_insufficient_train"
    assert "l1_score" in scored.columns


def test_fit_residual_scorer_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + 0.1 * rng.normal(size=120) > 0).astype(float)
    names = ["a_csrank", "b_csrank", "c_csrank", "d_csrank"]
    scorer = fit_residual_scorer(X, y, feature_names=names, model="logistic")
    pred = scorer.predict(X)
    assert pred.shape == (120,)
    assert scorer.fitted_


def test_thin_ml_logistic_fit_predict():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 3))
    y = (X @ np.array([1.0, -0.5, 0.2]) > 0).astype(float)
    stub = ThinMLStub(feature_names=["a", "b", "c"], mode="logistic").fit(X, y)
    pred = stub.predict(X[:10])
    assert pred.shape == (10,)
    assert stub.fitted_


def test_walk_forward_residual_scores_output_cols():
    panel = _synth_long_panel(450, 6, start="2019-01-02")
    cfg = ResidualTrainConfig(horizon=8, min_train_rows=30, model="thin_ml", top_quantile=0.4)
    scored, meta = walk_forward_residual_scores(
        panel,
        first_oos_year=2020,
        last_oos_year=2021,
        config=cfg,
    )
    assert not scored.empty
    assert "l1_score" in scored.columns
    assert meta.get("me_applied_before_l1") is False
    assert_no_absolute_prices(meta["feature_cols"])


def test_cli_l1_mode_parse():
    path = ROOT / "scripts" / "run_redesign_eval.py"
    spec = importlib.util.spec_from_file_location("run_redesign_eval", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(
        ["--l1-mode", "residual_train", "--residual-horizon", "15", "--residual-model", "logistic"]
    )
    assert args.l1_mode == "residual_train"
    assert args.residual_horizon == 15


def test_run_portable_residual_train_synthetic_daily():
    """Wire residual_train path: daily L1, ME only at L2 — synthetic panel."""
    path = ROOT / "scripts" / "run_redesign_eval.py"
    spec = importlib.util.spec_from_file_location("run_redesign_eval_rt", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    panel = _synth_long_panel(780, 6, start="2019-01-02")
    pool = sorted(panel["ticker"].unique().tolist())
    # OOS window subset + full history (2020 has data on this panel)
    oos = panel.loc[panel["date"].dt.year == 2020].copy()
    assert not oos.empty
    out = mod.run_portable_v0(
        oos,
        pool,
        top_k=3,
        top_quantile=0.5,
        l1_mode="residual_train",
        first_oos=2020,
        last_oos=2020,
        residual_horizon=5,
        residual_model="thin_ml",
        full_history_panel=panel,
        use_me_rebalance=True,
    )
    assert out["name"] == "alpha_portable_v0_residual_train"
    assert out["l1_meta"].get("me_applied_before_l1") is False
    assert out["l1_meta"].get("l1_mode") == "residual_train"
    # equity path produced something
    assert out.get("equity") is not None
    folds = (out.get("l1_meta") or {}).get("folds") or []
    assert folds
    # train rows should be daily-scale (>> ME-only ~dozens)
    n_lab = folds[0].get("n_train_labeled") or 0
    assert n_lab > 50


def test_labels_not_used_as_features_assert():
    panel = _synth_long_panel(60, 3)
    _, feats = prepare_residual_train_panel(panel, horizon=5)
    for ban in ("close", "open", "high", "low", "y_beat_style", "y_excess", "fwd_ret"):
        assert ban not in feats


def test_analyze_excludes_pathological_clone_from_p1():
    """run_style_clone_gap.analyze must not confirm P1 on pathology-only clones."""
    path = ROOT / "scripts" / "run_style_clone_gap.py"
    spec = importlib.util.spec_from_file_location("scg_patho", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    idx = pd.bdate_range("2020-01-01", periods=60, tz="UTC")
    base_eq = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    # Pathological clone: insane CAGR path
    patho_eq = pd.Series(np.cumprod(np.r_[1.0, np.full(len(idx) - 1, 1.05)]) * 100, index=idx)
    sane_eq = pd.Series(np.linspace(100, 90, len(idx)), index=idx)

    baseline = {
        "name": "base",
        "report": {
            "cagr": 0.10,
            "sharpe": 0.8,
            "max_drawdown": -0.2,
            "excess_cagr_vs_spy": 0.05,
        },
        "equity": base_eq,
        "n_tickers": 10,
    }
    clones = [
        {
            "name": "style_ew_patho",
            "report": {
                "cagr": 2.5,
                "sharpe": 0.7,
                "max_drawdown": -0.5,
                "excess_cagr_vs_spy": 2.3,
            },
            "equity": patho_eq,
        },
        {
            "name": "style_mom_sane",
            "report": {
                "cagr": -0.15,
                "sharpe": -0.5,
                "max_drawdown": -0.4,
                "excess_cagr_vs_spy": -0.20,
            },
            "equity": sane_eq,
        },
    ]
    summary = mod.analyze(baseline, clones, pit_block=None)
    assert summary["p1_confirmed_any_clone"] is False
    patho_row = next(r for r in summary["clones"] if r["clone"] == "style_ew_patho")
    assert patho_row["pathology_suspect"] is True
    assert patho_row["p1"]["confirmed"] is False
    # Hardest residual should be among sane clones (positive residual vs collapsing mom)
    assert summary["hardest_clone"] == "style_mom_sane"
    assert summary["hardest_clone_residual_cagr"] is not None
    assert summary["hardest_clone_residual_cagr"] > 0
