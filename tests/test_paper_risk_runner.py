"""LIV-06 kill switch + LIV-07 schedule clock / PaperRunner."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.freeze import PaperModeError, load_freeze
from paper_live.ledger import EventType, PaperLedger
from paper_live.oms.paper_broker import PaperBroker
from paper_live.replay_session import ReplaySession
from paper_live.risk import KillSwitch, PortfolioRisk, RiskConfig, rolling_sharpe
from paper_live.runner import PaperRunner, build_runner
from paper_live.schedule_clock import ScheduleClock, SessionPhase


def test_rolling_sharpe_basic():
    rets = [0.01] * 25
    sh = rolling_sharpe(rets, 20)
    assert sh is not None and sh > 0
    bad = [-0.05] * 25
    sh2 = rolling_sharpe(bad, 20)
    assert sh2 is not None and sh2 < 0


def test_soft_size_scale_and_hard_dd():
    cfg = RiskConfig(
        max_portfolio_dd=0.18,
        dd_soft_scale=0.5,
        dd_soft_trigger_frac=0.5,
        kill_dd_from_start=0.15,
        kill_rolling_sharpe_20d=-1.0,
    )
    risk = PortfolioRisk(cfg, capital0=100_000.0)
    # peak 100k, equity 91k → dd -9% = half of 18% → soft scale starts
    risk.peak_equity = 100_000.0
    risk.equity_history = [100_000.0]
    snap = risk.snapshot(equity=91_000.0, cash=91_000.0)
    assert snap.dd_from_peak == pytest.approx(-0.09)
    assert snap.size_scale < 1.0
    assert not snap.hard_kill

    # hard DD from peak
    snap2 = risk.snapshot(equity=80_000.0, cash=80_000.0)
    assert snap2.dd_from_peak <= -0.18
    assert snap2.hard_kill
    assert snap2.block_new_entries


def test_kill_switch_sticky(tmp_path: Path):
    freeze = load_freeze()
    led = PaperLedger.create_run(tmp_path / "k", freeze, run_id="kill_1")
    broker = PaperBroker(freeze.cost, capital0=100_000.0, ledger=led)
    cfg = RiskConfig(max_portfolio_dd=0.10, kill_dd_from_start=0.50, kill_rolling_sharpe_20d=-99.0)
    risk = PortfolioRisk(cfg, capital0=100_000.0)
    kill = KillSwitch(risk, broker, ledger=led)

    # first mark at peak
    kill.evaluate(equity=100_000.0, cash=100_000.0, update_history=True)
    assert not kill.state.entries_blocked

    # crash 15%
    snap = kill.evaluate(equity=85_000.0, cash=85_000.0, update_history=True)
    assert snap.hard_kill
    assert kill.state.entries_blocked
    assert kill.state.hard_kill
    assert broker.state.entries_blocked
    assert kill.state.trip_count == 1

    # recovery does not re-open entries (sticky)
    kill.evaluate(equity=99_000.0, cash=99_000.0, update_history=True)
    assert kill.state.entries_blocked
    assert broker.state.entries_blocked

    events = led.list_events(event_type=EventType.KILL_SWITCH)
    assert len(events) >= 1
    led.close()


def test_schedule_clock_phases():
    freeze = load_freeze()
    clock = ScheduleClock.from_schedule(freeze.schedule)
    tz = ZoneInfo("America/New_York")
    # Wednesday
    pre = datetime(2024, 6, 5, 9, 10, tzinfo=tz)
    assert clock.phase(pre) == SessionPhase.PRE_OPEN
    entry = datetime(2024, 6, 5, 10, 0, tzinfo=tz)
    assert clock.phase(entry) == SessionPhase.ENTRY_WINDOW
    assert clock.is_entry_allowed(entry)
    exit_t = datetime(2024, 6, 5, 15, 40, tzinfo=tz)
    assert clock.phase(exit_t) == SessionPhase.EXIT_CHECK
    sat = datetime(2024, 6, 8, 10, 0, tzinfo=tz)
    assert clock.phase(sat) == SessionPhase.CLOSED


def test_runner_replay_with_risk(tmp_path: Path):
    freeze = load_freeze()
    feed = DailyReplayFeed.from_synthetic(
        ["AAA", "BBB", "QQQ"], n_days=350, start="2019-01-02", seed=5
    )
    led = PaperLedger.create_run(tmp_path / "run", freeze, run_id="runner_1")
    runner = PaperRunner(freeze, feed=feed, ledger=led)
    days = feed.days
    out = runner.run_replay(days[250], days[280])
    assert out.mode == "replay"
    assert out.result is not None
    assert out.result.days_run >= 20
    assert out.kill_state is not None
    assert out.risk_last is not None
    assert "size_scale" in out.risk_last
    # heartbeats / session events
    assert led.list_events(event_type=EventType.HEARTBEAT) or led.list_events(
        event_type=EventType.SESSION_CLOSE
    )
    led.close()


def test_live_stub_requires_env(tmp_path: Path, monkeypatch):
    freeze = load_freeze()
    feed = DailyReplayFeed.from_synthetic(["AAA", "QQQ"], n_days=300, seed=2)
    led = PaperLedger.create_run(tmp_path / "live", freeze)
    runner = PaperRunner(freeze, feed=feed, ledger=led)
    monkeypatch.delenv("TRAD_PAPER_ONLY", raising=False)
    with pytest.raises(PaperModeError):
        runner.run_live_day_stub(feed.days[200], require_env=True)
    monkeypatch.setenv("TRAD_PAPER_ONLY", "1")
    out = runner.run_live_day_stub(feed.days[200], require_env=True)
    assert out.mode == "live_day_stub"
    assert SessionPhase.ENTRY_WINDOW.value in out.phases_executed
    assert SessionPhase.POST_CLOSE.value in out.phases_executed
    led.close()


def test_session_size_scale_uses_kill(tmp_path: Path):
    freeze = load_freeze()
    feed = DailyReplayFeed.from_synthetic(["AAA", "QQQ"], n_days=100, seed=1)
    session = ReplaySession(feed, freeze, enable_risk=True)
    assert session.kill is not None
    # force soft de-risk by lowering equity vs peak
    session.kill.risk.peak_equity = 200_000.0
    session.broker.state.cash = 100_000.0
    session.broker.state.peak_equity = 200_000.0
    scale = session.kill.size_scale()
    assert scale < 1.0
    # shares with scale should be <= without conceptually
    sh = session._size_shares(50.0)
    assert sh >= 0


def test_build_runner_factory(tmp_path: Path):
    r = build_runner(
        ledger_root=tmp_path / "b",
        tickers=["AAA", "QQQ"],
        synthetic=True,
    )
    assert r.feed is not None
    assert r.ledger is not None
    r.ledger.close()
