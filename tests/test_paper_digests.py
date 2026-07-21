"""LIV-08: daily/weekly digests + HTML dashboard."""
from __future__ import annotations

from pathlib import Path

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.freeze import load_freeze
from paper_live.ledger import PaperLedger
from paper_live.reports.daily_digest import build_daily_digest, write_daily_digest
from paper_live.reports.html_report import write_html_dashboard
from paper_live.reports.pipeline import generate_reports_for_run
from paper_live.reports.weekly_scorecard import build_weekly_scorecard, write_weekly_scorecard
from paper_live.runner import PaperRunner


def _run_short_paper(tmp_path: Path):
    freeze = load_freeze()
    feed = DailyReplayFeed.from_synthetic(
        ["AAA", "BBB", "QQQ"], n_days=350, start="2019-01-02", seed=9
    )
    led = PaperLedger.create_run(tmp_path / "ledger", freeze, run_id="digest_run_1")
    runner = PaperRunner(freeze, feed=feed, ledger=led)
    days = feed.days
    out = runner.run_replay(days[250], days[275])
    return led, out, days


def test_daily_digest_from_ledger(tmp_path: Path):
    led, runner_out, days = _run_short_paper(tmp_path)
    assert runner_out.result and runner_out.result.days_run > 5
    # pick a day that has NAV
    nav = led.list_nav()
    assert nav
    day = nav[len(nav) // 2]["date"]
    dig = build_daily_digest(led, day)
    assert dig.run_id == "digest_run_1"
    assert dig.day == day
    assert dig.mode == "paper"
    assert dig.capital_label == "VIRTUAL"
    assert dig.equity is not None
    md = dig.to_markdown()
    assert "Paper daily digest" in md
    assert "VIRTUAL" in md
    paths = write_daily_digest(dig, tmp_path / "out")
    assert paths["md"].is_file()
    assert paths["json"].is_file()
    led.close()


def test_weekly_scorecard_and_flags(tmp_path: Path):
    led, _, _ = _run_short_paper(tmp_path)
    nav = led.list_nav()
    assert nav
    card = build_weekly_scorecard(led, week_of=nav[-1]["date"])
    assert card.run_id == led.run_id
    assert card.week_start <= card.week_end
    assert card.n_fills >= 0
    assert card.commission >= 0
    md = card.to_markdown()
    assert "weekly scorecard" in md.lower()
    paths = write_weekly_scorecard(card, tmp_path / "out")
    assert paths["json"].is_file()
    led.close()


def test_html_dashboard(tmp_path: Path):
    led, _, _ = _run_short_paper(tmp_path)
    nav = led.list_nav()
    digests = [build_daily_digest(led, r["date"]) for r in nav[:10]]
    weekly = build_weekly_scorecard(led, week_of=nav[-1]["date"])
    path = write_html_dashboard(
        tmp_path / "dashboard.html",
        run_id=led.run_id,
        strategy_id=led.strategy_id,
        daily=digests,
        weekly=weekly,
    )
    text = path.read_text(encoding="utf-8")
    assert "PAPER" in text
    assert "VIRTUAL" in text
    assert led.run_id in text
    assert "<table>" in text
    led.close()


def test_generate_reports_pipeline(tmp_path: Path):
    led, _, _ = _run_short_paper(tmp_path)
    bundle = generate_reports_for_run(led, tmp_path / "reports")
    assert bundle.daily
    assert bundle.weekly is not None
    assert "html" in bundle.paths
    assert Path(bundle.paths["html"]).is_file()
    assert Path(bundle.paths["index"]).is_file()
    d = bundle.to_dict()
    assert d["mode"] == "paper"
    led.close()


def test_list_fills_day_filter(tmp_path: Path):
    led, _, _ = _run_short_paper(tmp_path)
    all_fills = led.list_fills()
    if all_fills:
        day = str(all_fills[0]["ts"])[:10]
        day_fills = led.list_fills(day=day)
        assert all(str(f["ts"])[:10] == day for f in day_fills)
    led.close()
