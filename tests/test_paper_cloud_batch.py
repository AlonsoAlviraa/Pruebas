"""Cloud multi-strategy batch (free / synthetic)."""
from __future__ import annotations

from pathlib import Path

from paper_live.cloud.batch import load_zoo, run_cloud_batch
from paper_live.cloud.free_data import build_cloud_feed


def test_zoo_has_ten_strategies():
    zoo = load_zoo()
    assert len(zoo["strategies"]) == 10
    ids = [s["id"] for s in zoo["strategies"]]
    assert len(set(ids)) == 10


def test_synthetic_feed_builds():
    feed, sources = build_cloud_feed(
        ["AAA", "BBB", "QQQ", "SPY"],
        force_synthetic=True,
        lookback_calendar_days=400,
    )
    assert len(feed.days) > 100
    assert all(v == "synthetic" for v in sources.values())


def test_cloud_batch_synthetic(tmp_path: Path):
    out = tmp_path / "paper_cloud"
    result = run_cloud_batch(
        out_root=out,
        force_synthetic=True,
        lookback_days=400,
        keep_ledgers=False,
    )
    assert result.n_strategies == 10
    assert result.mode == "paper"
    # at least some strategies ran without error
    ok = [s for s in result.strategies if not s.error]
    assert len(ok) >= 8
    assert (out / "latest" / "SUMMARY.md").is_file()
    assert (out / "latest" / "dashboard.html").is_file()
    assert (out / "latest" / "summary.json").is_file()
    # history day folder
    hist = list((out / "history").glob("*/SUMMARY.md"))
    assert hist
    ranking = result.ranking()
    assert ranking
    assert ranking[0].final_equity > 0
