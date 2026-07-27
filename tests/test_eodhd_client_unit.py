"""Unit tests for EODHD client helpers (no network except optional skip)."""
from __future__ import annotations

import os

import pytest

from paper_live.data.eodhd_client import resolve_symbol


def test_resolve_symbol_us_and_vix():
    assert resolve_symbol("AAPL") == "AAPL.US"
    assert resolve_symbol("spy") == "SPY.US"
    assert resolve_symbol("VIX") == "VIX.INDX"
    assert resolve_symbol("^VIX") == "VIX.INDX"
    assert resolve_symbol("VIX3M") == "VIX3M.INDX"
    assert resolve_symbol("MSFT.US") == "MSFT.US"


@pytest.mark.skipif(
    not (os.environ.get("EODHD_API_TOKEN") or os.environ.get("EODHD_API_KEY")),
    reason="no EODHD token",
)
def test_fetch_eod_smoke_optional():
    from paper_live.data.eodhd_client import fetch_eod

    df = fetch_eod("AAPL", start="2024-01-02", end="2024-01-10")
    assert not df.empty
    assert "close" in df.columns
