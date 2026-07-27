"""Unit tests for causal early-window universe builder (synthetic CSVs only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from trad_research.early_universe import (
    build_early_window_universe,
    build_early_window_universe_meta,
    mean_dollar_volume,
    write_universe_file,
)


def _write_hist(
    root: Path,
    ticker: str,
    *,
    start: str,
    end: str,
    close: float = 10.0,
    volume: float = 100_000.0,
) -> None:
    idx = pd.date_range(start, end, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "date": idx,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        }
    )
    df.to_csv(root / f"{ticker}_history.csv", index=False)


def test_build_early_as_of_no_post_oos_survivorship(tmp_path: Path):
    """Ticker delisted mid-OOS can still be in L0 as-of first OOS (no last_need)."""
    data = tmp_path / "data"
    data.mkdir()
    # Listed as-of 2010-01-01, deep history, good ADV — dies in 2012
    _write_hist(data, "KEEP", start="2004-01-01", end="2012-06-01", volume=1e6)
    # Only listed after OOS start — must exclude
    _write_hist(data, "LATE", start="2011-01-01", end="2015-01-01", volume=1e6)
    # Delisted before as_of — exclude
    _write_hist(data, "DEAD", start="2004-01-01", end="2009-06-01", volume=1e6)
    # Survives past 2014 but thin ADV — exclude if min high
    _write_hist(data, "THIN", start="2004-01-01", end="2015-01-01", volume=10.0)

    tickers = build_early_window_universe(
        data,
        as_of="2010-01-01",
        history_start_need="2005-06-01",
        min_adv_usd=50_000.0,
        max_names=10,
    )
    assert "KEEP" in tickers
    assert "LATE" not in tickers
    assert "DEAD" not in tickers
    assert "THIN" not in tickers


def test_adv_window_ends_before_as_of(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    _write_hist(data, "AAA", start="2004-01-01", end="2015-01-01", volume=5e5)
    meta = build_early_window_universe_meta(
        data, as_of="2010-01-01", history_start_need="2005-01-01", min_adv_usd=1.0
    )
    assert meta["adv_window_end"] < meta["as_of"]
    assert meta["no_post_oos_survivorship"] is True
    # ADV in 2010 calendar must not be required — force end before
    adv = mean_dollar_volume(
        data, "AAA", start=meta["adv_window_start"], end=meta["adv_window_end"]
    )
    assert adv > 0


def test_write_universe_file(tmp_path: Path):
    path = write_universe_file(["AAA", "BBB"], tmp_path / "u.txt")
    assert path.read_text(encoding="utf-8").strip().splitlines() == ["AAA", "BBB"]


def test_ensure_rebuilds_when_as_of_changes(tmp_path: Path):
    from trad_research.early_universe import ensure_early_universe_file

    data = tmp_path / "data"
    data.mkdir()
    _write_hist(data, "KEEP", start="2004-01-01", end="2015-01-01", volume=1e6)
    path = tmp_path / "u.txt"
    ensure_early_universe_file(
        path, data_root=data, as_of="2010-01-01", max_names=5, min_adv_usd=1.0
    )
    meta1 = (path.with_suffix(path.suffix + ".meta.json")).read_text(encoding="utf-8")
    # Same path, different as_of → rebuild (fingerprint change)
    ensure_early_universe_file(
        path, data_root=data, as_of="2012-01-01", max_names=5, min_adv_usd=1.0
    )
    meta2 = (path.with_suffix(path.suffix + ".meta.json")).read_text(encoding="utf-8")
    assert '"as_of": "2012-01-01"' in meta2
    assert meta1 != meta2
