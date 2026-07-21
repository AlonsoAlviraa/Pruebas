"""LIV-01/LIV-02: config freeze + append-only paper ledger (virtual capital)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paper_live.freeze import (
    PaperModeError,
    assert_paper_only,
    compute_config_hash,
    load_freeze,
)
from paper_live.ledger import EventType, PaperLedger, new_run_id
from paper_live.ledger.events import EVENT_TYPES


def test_load_freeze_stable_hash():
    f1 = load_freeze()
    f2 = load_freeze()
    assert f1.config_hash == f2.config_hash
    assert len(f1.config_hash) == 64
    assert f1.strategy.mode == "paper"
    assert f1.strategy.strategy_id == "turbo_highvol_minalloc"
    assert f1.strategy.knobs["min_alloc_pct"] == 0.015
    assert f1.cost.commission["per_share"] > 0
    # hash ignores source_dir / is pure bundle
    assert f1.config_hash == compute_config_hash(f1.to_bundle_dict())


def test_commission_and_slip_helpers():
    cost = load_freeze().cost
    c = cost.estimate_commission(100, 50.0)
    assert c >= 1.0  # min_per_order
    buy_px = cost.slip_price("buy", 100.0)
    sell_px = cost.slip_price("sell", 100.0)
    assert buy_px > 100.0
    assert sell_px < 100.0
    stop_px = cost.slip_price("sell", 100.0, is_stop=True)
    assert stop_px < sell_px
    fees = cost.estimate_sell_fees(100, 50.0)
    assert fees > 0


def test_assert_paper_only_blocks_live_mode(monkeypatch):
    monkeypatch.setenv("TRAD_TRADING_MODE", "live")
    with pytest.raises(PaperModeError):
        assert_paper_only(require_env=False)
    monkeypatch.setenv("TRAD_TRADING_MODE", "paper")
    assert_paper_only(require_env=False)
    monkeypatch.delenv("TRAD_PAPER_ONLY", raising=False)
    with pytest.raises(PaperModeError):
        assert_paper_only(require_env=True)
    monkeypatch.setenv("TRAD_PAPER_ONLY", "1")
    assert_paper_only(require_env=True)


def test_ledger_run_order_fill_nav(tmp_path: Path):
    freeze = load_freeze()
    root = tmp_path / "ledger"
    led = PaperLedger.create_run(root, freeze, run_id="paper_test_run_001")
    assert led.run_id == "paper_test_run_001"
    assert led.get_run()["mode"] == "paper"
    assert led.get_run()["config_hash"] == freeze.config_hash

    # freeze snapshot written
    assert (root / "freeze_paper_test_run_001.json").is_file()

    events = led.list_events(event_type=EventType.RUN_INIT)
    assert len(events) == 1
    assert events[0]["payload"]["mode"] == "paper"

    did = led.record_decision(
        ticker="aaa",
        action="enter",
        p_buy=0.8,
        score=1.2,
        filters={"regime": True},
    )
    assert did.startswith("dec_")

    oid = led.record_order(ticker="AAA", side="buy", qty=10, order_type="market")
    comm = freeze.cost.estimate_commission(10, 25.0)
    fees = 0.0
    fid = led.record_fill(
        order_id=oid,
        ticker="AAA",
        side="buy",
        qty=10,
        price=25.0,
        commission=comm,
        fees=fees,
        slippage_bps=5.0,
    )
    assert fid.startswith("fill_")
    assert led.sum_commissions() == pytest.approx(comm)

    led.upsert_position(ticker="AAA", qty=10, avg_px=25.0, stop=23.0, hard_stop=23.25)
    pos = led.get_positions()
    assert len(pos) == 1
    assert pos[0]["ticker"] == "AAA"

    led.record_nav_daily(
        "2026-07-21",
        equity=100_000.0,
        cash=99_750.0 - comm,
        gross_exposure=250.0,
        n_positions=1,
        peak_equity=100_000.0,
    )
    led.record_costs_daily(
        "2026-07-21",
        commission=comm,
        fees=0.0,
        slippage_est=0.12,
        turnover=250.0,
    )
    snap = led.write_snapshot("eod")
    assert snap.is_file()

    # JSONL audit exists and is non-empty
    audit = root / "audit" / "2026-07-21.jsonl"
    # RUN_INIT may be on "today" UTC; FILL/NAV use event ts = now as well unless fixed
    # At least one audit file under audit/
    audit_files = list((root / "audit").glob("*.jsonl"))
    assert audit_files, "expected JSONL audit files"
    lines = []
    for p in audit_files:
        lines.extend(p.read_text(encoding="utf-8").strip().splitlines())
    assert any('"event_type": "fill"' in ln or '"event_type":"fill"' in ln for ln in lines)

    # reopen recovery
    led.close()
    led2 = PaperLedger.open_run(root, "paper_test_run_001")
    assert len(led2.get_positions()) == 1
    assert led2.sum_commissions() == pytest.approx(comm)
    fills = led2.list_events(event_type=EventType.FILL)
    assert len(fills) >= 1
    led2.close_position("AAA")
    assert led2.get_positions() == []
    led2.close()


def test_jsonl_is_append_only(tmp_path: Path):
    freeze = load_freeze()
    root = tmp_path / "ledger2"
    led = PaperLedger.create_run(root, freeze)
    led.append_event(EventType.HEARTBEAT, {"n": 1})
    led.append_event(EventType.HEARTBEAT, {"n": 2})
    audit_files = list((root / "audit").glob("*.jsonl"))
    assert len(audit_files) == 1
    text1 = audit_files[0].read_text(encoding="utf-8")
    n1 = text1.count("\n")
    led.append_event(EventType.HEARTBEAT, {"n": 3})
    text2 = audit_files[0].read_text(encoding="utf-8")
    assert text2.startswith(text1) or text1 in text2
    assert text2.count("\n") == n1 + 1
    led.close()


def test_unknown_event_type_rejected(tmp_path: Path):
    freeze = load_freeze()
    led = PaperLedger.create_run(tmp_path / "l3", freeze)
    with pytest.raises(ValueError, match="Unknown event_type"):
        led.append_event("not_a_real_event", {})
    led.close()


def test_event_types_cover_design_minimum():
    required = {
        "session_open",
        "session_close",
        "signal_computed",
        "entry_candidate",
        "entry_rejected",
        "order_submitted",
        "fill",
        "position_opened",
        "position_closed",
        "daily_nav",
        "kill_switch",
        "heartbeat",
        "run_init",
    }
    assert required <= EVENT_TYPES


def test_new_run_id_unique():
    a, b = new_run_id(), new_run_id()
    assert a != b
    assert a.startswith("paper_")
