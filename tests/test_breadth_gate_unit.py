"""Unit tests for causal universe breadth gate (synthetic only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trad_research.breadth_gate import (
    BreadthGateConfig,
    and_regime_maps,
    breadth_to_risk_on_map,
    build_breadth_risk_on_map,
    closes_from_panels,
    compute_breadth_series,
)


def _make_uptrend(n: int = 120, start: float = 100.0, drift: float = 0.002) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    px = start * np.cumprod(1.0 + np.full(n, drift))
    return pd.Series(px, index=dates)


def _make_downtrend(n: int = 120, start: float = 100.0, drift: float = -0.003) -> pd.Series:
    return _make_uptrend(n=n, start=start, drift=drift)


def test_breadth_high_when_most_above_sma():
    closes = {f"T{i}": _make_uptrend() for i in range(12)}
    b = compute_breadth_series(closes, sma_period=20, min_names=8)
    valid = b.dropna()
    assert len(valid) > 20
    # Uptrend → mostly above SMA
    assert float(valid.iloc[-20:].mean()) > 0.7


def test_breadth_low_when_most_below_sma():
    closes = {f"T{i}": _make_downtrend() for i in range(12)}
    b = compute_breadth_series(closes, sma_period=20, min_names=8)
    valid = b.dropna()
    assert len(valid) > 20
    assert float(valid.iloc[-20:].mean()) < 0.35


def test_risk_on_map_fail_closed_nan():
    dates = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    b = pd.Series([0.5, np.nan, 0.3, 0.8, 0.1], index=dates)
    m = breadth_to_risk_on_map(b, min_breadth=0.40)
    vals = list(m.values())
    assert vals[0] is True
    assert vals[1] is False  # NaN → False
    assert vals[2] is False
    assert vals[3] is True
    assert vals[4] is False


def test_build_breadth_disabled():
    m, s, meta = build_breadth_risk_on_map(
        {"A": _make_uptrend()},
        BreadthGateConfig(enabled=False),
    )
    assert m == {}
    assert meta["enabled"] is False


def test_and_regime_maps():
    d0 = pd.Timestamp("2020-01-02", tz="UTC")
    d1 = pd.Timestamp("2020-01-03", tz="UTC")
    d2 = pd.Timestamp("2020-01-06", tz="UTC")
    a = {d0: True, d1: True, d2: False}
    b = {d0: True, d1: False, d2: True}
    out = and_regime_maps(a, b)
    assert out[d0] is True
    assert out[d1] is False
    assert out[d2] is False


def test_and_regime_empty_passthrough():
    d0 = pd.Timestamp("2020-01-02", tz="UTC")
    a = {d0: True}
    assert and_regime_maps(a, {}) == a
    assert and_regime_maps({}, a) == a
    assert and_regime_maps({}, {}) == {}


def test_closes_from_panels():
    dates = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    df = pd.DataFrame({"date": dates, "close": [1, 2, 3, 4, 5]})
    out = closes_from_panels({"AAA": df})
    assert "AAA" in out
    assert len(out["AAA"]) == 5


def test_min_names_fail_closed():
    # Only 3 names → breadth NaN with min_names=8
    closes = {f"T{i}": _make_uptrend() for i in range(3)}
    b = compute_breadth_series(closes, sma_period=20, min_names=8)
    assert b.dropna().empty
