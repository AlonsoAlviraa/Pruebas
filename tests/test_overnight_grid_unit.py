"""Unit tests for smart overnight grid generation."""
from __future__ import annotations

from trad_research.overnight_grid import (
    build_phase1_risk_cells,
    build_phase2_overlay_cells,
    cells_to_mega_configs,
    estimate_grid_sizes,
)


def test_no_continuous_hard_circuit():
    cells = build_phase1_risk_cells(mode="full")
    for c in cells:
        if c.peak_mode == "continuous" and c.max_portfolio_dd < 0.9:
            assert c.dd_breach_size_scale is not None, c.fingerprint()


def test_full_is_large_but_bounded():
    sizes = estimate_grid_sizes()
    assert sizes["smoke"] < sizes["medium"] < sizes["full"]
    # "thousands" class for full, but not insane
    assert sizes["full"] >= 400
    assert sizes["full"] <= 5000


def test_anchors_present():
    cells = build_phase1_risk_cells(mode="medium")
    labels = {c.label for c in cells}
    assert "baseline" in labels
    assert "dd35_vt80_yr" in labels


def test_mega_config_shape():
    cells = build_phase1_risk_cells(mode="smoke")
    cfgs = cells_to_mega_configs(cells, strategy_overrides={"volatility_target_pct": 0.03})
    assert len(cfgs) == len(cells)
    for c in cfgs:
        assert c["base"] == "turbo_highvol_minalloc"
        assert "max_portfolio_dd" in c["extra_bt"]
        assert c["peak_mode"] in ("yearly", "continuous")


def test_phase2_expands():
    cells = build_phase1_risk_cells(mode="smoke")
    pairs = build_phase2_overlay_cells(cells, max_survivors=2)
    assert len(pairs) == 2 * 3
