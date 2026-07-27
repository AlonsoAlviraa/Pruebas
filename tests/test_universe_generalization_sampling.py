"""Unit tests for universe Monte Carlo sampling (no market data required)."""
from __future__ import annotations

from pathlib import Path

import pytest

from trad_research.universe_sampling import (
    aggregate_numeric,
    draw_seed,
    geo_verdict,
    materialize_draw,
    prefix_tickers,
    read_tickers,
    sample_without_replacement,
    us_verdict,
    write_tickers,
)


def test_sample_reproducible():
    pool = [f"T{i:03d}" for i in range(100)]
    a = sample_without_replacement(pool, 50, seed=20260726)
    b = sample_without_replacement(pool, 50, seed=20260726)
    c = sample_without_replacement(pool, 50, seed=20260727)
    assert a == b
    assert a != c
    assert len(a) == 50
    assert len(set(a)) == 50
    assert set(a).issubset(set(pool))


def test_sample_size_validation():
    pool = ["A", "B", "C"]
    with pytest.raises(ValueError):
        sample_without_replacement(pool, 5, seed=1)


def test_prefix_order():
    pool = ["Z", "A", "M"]
    assert prefix_tickers(pool, 2) == ["Z", "A"]


def test_draw_seed_namespace():
    us0 = draw_seed(20260726, "US", 0)
    es0 = draw_seed(20260726, "ES", 0)
    assert us0 != es0
    assert draw_seed(20260726, "ES", 1) == es0 + 1


def test_materialize_draw(tmp_path: Path):
    pool = [f"T{i}" for i in range(20)]
    p = tmp_path / "d.txt"
    tickers = materialize_draw(pool, series="R10", m=10, seed=42, out_path=p)
    assert p.is_file()
    assert read_tickers(p) == tickers
    assert len(tickers) == 10


def test_us_verdict_levels():
    assert (
        us_verdict(
            pass_rate=0.5,
            median_cagr=0.12,
            median_mdd=-0.40,
            prefix_pass=True,
        )
        == "GENERALIZES"
    )
    assert (
        us_verdict(
            pass_rate=0.05,
            median_cagr=0.15,
            median_mdd=-0.30,
            prefix_pass=True,
        )
        == "PREFIX-ONLY"
    )
    assert (
        us_verdict(
            pass_rate=0.25,
            median_cagr=0.11,
            median_mdd=-0.40,
            prefix_pass=False,
        )
        == "FRAGILE"
    )
    assert (
        us_verdict(
            pass_rate=0.5,
            median_cagr=0.05,
            median_mdd=-0.40,
            prefix_pass=False,
        )
        == "FAIL"
    )


def test_geo_verdict():
    assert geo_verdict({"ES": True, "FR": True, "DE": False}, uk_ok=True) == "TRANSFERS"
    assert geo_verdict({"ES": True, "FR": False, "DE": False}) == "MIXED"
    assert geo_verdict({"ES": False, "FR": False, "DE": False}) == "FAIL_GEO"


def test_aggregate_numeric():
    a = aggregate_numeric([0.1, 0.2, 0.3])
    assert a["n"] == 3
    assert abs(a["mean"] - 0.2) < 1e-9
    assert a["median"] == 0.2
    empty = aggregate_numeric([])
    assert empty["n"] == 0


def test_write_read_tickers(tmp_path: Path):
    p = tmp_path / "u.txt"
    write_tickers(p, ["AAA", "BBB"])
    assert read_tickers(p) == ["AAA", "BBB"]
