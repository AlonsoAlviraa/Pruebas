"""Unit tests for Kaggle Stage1 grid sampling."""
from __future__ import annotations

from kaggle_redesign.src.grids import (
    AXES,
    full_grid_size,
    sample_configs,
    theoretical_millions,
)


def test_full_grid_is_millions():
    n = full_grid_size()
    assert n > 1_000_000
    info = theoretical_millions()
    assert info["full_grid_size"] == n


def test_sample_configs_sobol_unique_enough():
    pts = sample_configs(5000, seed=0, method="sobol")
    assert len(pts) == 5000
    ids = {p.config_id for p in pts}
    # high diversity
    assert len(ids) > 1000
    for p in pts[:20]:
        assert p.universe in AXES["universe"]
        assert p.signal in AXES["signal"]


def test_no_limit_54_in_axes():
    blob = " ".join(str(v) for vals in AXES.values() for v in vals)
    assert "54" not in blob or "L54" not in blob
    assert "longhist_L50" in AXES["universe"]
    assert "longhist_L80" in AXES["universe"]
