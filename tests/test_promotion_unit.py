"""Unit tests for promotion funnel (PROMO-01)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trad_research.promotion import (
    CandidateInput,
    PromotionThresholds,
    apply_top_k,
    evaluate_candidate,
)


def _eq_path(mu: float, n: int = 400, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, 0.01, n)
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(100_000 * np.cumprod(1.0 + rets), index=idx)


def test_kill_on_pathology_cagr():
    # Explosive path
    idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="UTC")
    eq = pd.Series(100_000 * (1.05 ** np.arange(50)), index=idx)
    card = evaluate_candidate(
        CandidateInput(name="boom", equity=eq, product="STYLE-US", smoke=True),
        thr=PromotionThresholds(pathology_cagr_abs=1.0),
        n_sims=20,
    )
    assert card.label == "KILL"
    assert "pathology_cagr" in card.kill_reasons


def test_stage1_fail_low_sortino_never_advances():
    # High noise zero drift → weak Sortino
    eq = _eq_path(0.00005, n=300, seed=3)
    style = _eq_path(0.0004, n=300, seed=4)
    card = evaluate_candidate(
        CandidateInput(
            name="weak",
            equity=eq,
            style_equity=style,
            product="STYLE-US",
            smoke=True,
        ),
        thr=PromotionThresholds(sortino_min=5.0, sharpe_min=5.0),  # impossible
        n_sims=30,
    )
    assert card.label == "KILL"
    assert any("sortino" in r or "sharpe" in r for r in card.kill_reasons)


def test_alpha_requires_residual():
    eq = _eq_path(0.001, n=400, seed=1)
    card = evaluate_candidate(
        CandidateInput(name="alpha_no_style", equity=eq, product="ALPHA-PORTABLE", smoke=True),
        n_sims=20,
    )
    assert card.label == "KILL"
    assert "residual_required_for_alpha" in card.kill_reasons


def test_top_k_demotes():
    # Build fake ADVANCE cards
    from trad_research.promotion import PromotionCard, StageResult

    cards = []
    for i, ex in enumerate([0.1, 0.05, 0.02, 0.01]):
        c = PromotionCard(
            name=f"c{i}",
            product="STYLE-US",
            label="ADVANCE_STYLE",
            stages=[StageResult("s", True)],
            metrics={"sortino": 1.0},
            residual={"excess_cagr": ex},
        )
        cards.append(c)
    out = apply_top_k(cards, k=2)
    advances = [c for c in out if c.label.startswith("ADVANCE")]
    assert len(advances) == 2
    demoted = [c for c in out if "top_k_demoted" in c.kill_reasons]
    assert len(demoted) == 2
