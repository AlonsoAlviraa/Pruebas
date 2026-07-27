"""Unit tests for Kaggle Stage1 scorer + purged windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kaggle_redesign.src.grids import sample_configs
from kaggle_redesign.src.math.purged_score import assert_score_window_causal
from kaggle_redesign.src.stage1_scorer import run_stage1_sample, score_rules_config


def test_purged_window_rejects_leak():
    with pytest.raises(ValueError):
        assert_score_window_causal(2017, 2016)


def test_purged_window_ok():
    assert_score_window_causal(2015, 2016)


def test_stage1_sample_smoke():
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2014-01-01", periods=n, freq="B", tz="UTC")
    # feature weakly predicts future
    feat = pd.Series(rng.normal(0, 1, n), index=idx)
    fwd = feat.shift(1) * 0.1 + rng.normal(0, 0.02, n)
    df = run_stage1_sample(200, feature=feat, forward_ret=fwd, seed=1, top_k=20)
    assert len(df) == 20
    assert "stage1_score" in df.columns
    assert df["stage1_score"].notna().any()


def test_score_rules_config_id():
    cfg = sample_configs(3, seed=2)[0]
    idx = pd.date_range("2016-01-01", periods=300, freq="B", tz="UTC")
    feat = pd.Series(np.linspace(-1, 1, 300), index=idx)
    fwd = feat.shift(-1).fillna(0)
    res = score_rules_config(cfg, feature=feat, forward_ret=fwd)
    assert res.config_id == cfg.config_id
