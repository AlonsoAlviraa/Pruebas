"""LIV-05: paper OMS fill model + commissions (virtual capital only)."""
from __future__ import annotations

from pathlib import Path

import pytest

from paper_live.freeze import load_freeze
from paper_live.ledger import EventType, PaperLedger
from paper_live.oms import (
    FillModel,
    FillQuote,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperBroker,
    PaperOrder,
)
from paper_live.oms.order_types import RejectReason
from paper_live.ledger.events import new_order_id


@pytest.fixture
def freeze():
    return load_freeze()


def test_fill_model_buy_worse_than_mid(freeze):
    fm = FillModel(freeze.cost)
    order = PaperOrder(
        order_id="ord_x",
        ticker="AAA",
        side=OrderSide.BUY,
        qty=100,
    )
    sim = fm.simulate_fill(order, FillQuote(mid=50.0, adv_shares=1_000_000))
    assert sim.ok
    assert sim.price > 50.0
    assert sim.commission >= 1.0
    assert sim.fees == 0.0  # buys: no SEC/TAF
    assert sim.net_cash_delta < 0
    assert sim.slippage_bps > 0
    assert sim.slippage_cost > 0


def test_fill_model_sell_fees_and_slip(freeze):
    fm = FillModel(freeze.cost)
    order = PaperOrder(
        order_id="ord_y",
        ticker="AAA",
        side=OrderSide.SELL,
        qty=100,
    )
    sim = fm.simulate_fill(order, FillQuote(mid=50.0))
    assert sim.ok
    assert sim.price < 50.0
    assert sim.fees > 0
    assert sim.commission >= 1.0
    assert sim.net_cash_delta > 0
    # net = gross - commission - fees
    assert sim.net_cash_delta == pytest.approx(
        sim.gross_notional - sim.commission - sim.fees
    )


def test_stop_has_extra_slippage(freeze):
    fm = FillModel(freeze.cost)
    base = PaperOrder(order_id="a", ticker="T", side=OrderSide.SELL, qty=50)
    stop = PaperOrder(
        order_id="b", ticker="T", side=OrderSide.SELL, qty=50, is_stop=True
    )
    q = FillQuote(mid=100.0)
    s0 = fm.simulate_fill(base, q)
    s1 = fm.simulate_fill(stop, q)
    assert s1.price < s0.price  # more adverse
    assert s1.slippage_bps > s0.slippage_bps


def test_halt_and_min_price_reject(freeze):
    fm = FillModel(freeze.cost)
    order = PaperOrder(order_id="h", ticker="Z", side=OrderSide.BUY, qty=10)
    halt = fm.simulate_fill(order, FillQuote(mid=10.0, halted=True))
    assert not halt.ok
    assert halt.reject == RejectReason.HALT
    cheap = fm.simulate_fill(order, FillQuote(mid=0.5))
    assert not cheap.ok
    assert cheap.reject == RejectReason.MIN_PRICE


def test_adv_cap_truncates(freeze):
    fm = FillModel(freeze.cost)
    # max participation 2% of ADV=1000 → max 20 shares
    order = PaperOrder(order_id="adv", ticker="Z", side=OrderSide.BUY, qty=500)
    sim = fm.simulate_fill(order, FillQuote(mid=10.0, adv_shares=1000))
    assert sim.ok
    assert sim.qty == 20.0


def test_twap_clips(freeze):
    fm = FillModel(freeze.cost, n_clips=3)
    assert fm.clip_qtys(100) == [34.0, 33.0, 33.0]
    order = PaperOrder(order_id="tw", ticker="T", side=OrderSide.BUY, qty=90)
    fills = fm.simulate_twap(
        order,
        [FillQuote(mid=10.0), FillQuote(mid=10.1), FillQuote(mid=10.2)],
    )
    assert len(fills) == 3
    assert sum(f.qty for f in fills) == 90


def test_broker_buy_sell_roundtrip_with_ledger(freeze, tmp_path: Path):
    led = PaperLedger.create_run(tmp_path / "oms", freeze, run_id="oms_rt_1")
    broker = PaperBroker(freeze.cost, capital0=100_000.0, ledger=led, n_clips=1)

    order, fills = broker.submit_and_execute(
        "AAA",
        "buy",
        100,
        FillQuote(mid=50.0, adv_shares=5_000_000),
    )
    assert order.status in (OrderStatus.FILLED, OrderStatus.ACK, OrderStatus.PARTIAL)
    assert len(fills) == 1 and fills[0].ok
    assert "AAA" in broker.state.positions
    assert broker.state.positions["AAA"] == 100
    cash_after_buy = broker.state.cash
    assert cash_after_buy < 100_000.0
    assert broker.state.total_commission > 0

    # mark and equity
    broker.update_marks({"AAA": 55.0})
    eq = broker.state.equity()
    assert eq > cash_after_buy

    order2, fills2 = broker.submit_and_execute(
        "AAA",
        "sell",
        100,
        FillQuote(mid=55.0),
        is_stop=False,
    )
    assert fills2 and fills2[0].ok
    assert broker.state.n_positions() == 0
    assert broker.state.total_fees > 0  # sell fees
    assert broker.state.cash != 100_000.0  # costs dragged

    # ledger has fills
    fill_events = led.list_events(event_type=EventType.FILL)
    assert len(fill_events) >= 2
    assert led.sum_commissions() == pytest.approx(broker.state.total_commission)
    nav = broker.record_nav("2026-07-21")
    assert nav["capital_label"] == "VIRTUAL"
    assert nav["mode"] if "mode" in nav else True
    led.close()


def test_broker_reject_short(freeze):
    broker = PaperBroker(freeze.cost, capital0=50_000.0, long_only=True)
    order, fills = broker.submit_and_execute("BBB", "sell", 10, FillQuote(mid=20.0))
    assert order.status == OrderStatus.REJECTED
    assert fills == []
    assert "short" in (order.reason or "").lower() or "SHORT" in (order.reason or "")


def test_broker_kill_switch_blocks_entries(freeze):
    broker = PaperBroker(freeze.cost, capital0=50_000.0)
    broker.set_entries_blocked(True, reason="test")
    order, _ = broker.submit_and_execute("CCC", "buy", 10, FillQuote(mid=10.0))
    assert order.status == OrderStatus.REJECTED
    assert "kill" in (order.reason or "").lower()


def test_broker_insufficient_cash_downsizes_or_rejects(freeze):
    broker = PaperBroker(freeze.cost, capital0=100.0)  # tiny book
    order, fills = broker.submit_and_execute("DDD", "buy", 1000, FillQuote(mid=50.0))
    # either fully rejected or downsized to what cash allows
    if fills and fills[0].ok:
        assert fills[0].qty * fills[0].price <= 100.0 + 1.0
        assert broker.state.cash >= -1e-6
    else:
        assert order.status == OrderStatus.REJECTED or (
            fills and fills[0].reject == RejectReason.INSUFFICIENT_CASH
        )


def test_cancel_open_order(freeze):
    broker = PaperBroker(freeze.cost, capital0=100_000.0)
    order = broker.submit("EEE", "buy", 50)
    assert order.status == OrderStatus.ACK
    assert order.order_id in broker.open_orders
    assert broker.cancel(order.order_id)
    assert order.order_id not in broker.open_orders
    assert order.status == OrderStatus.CANCELLED


def test_limit_not_marketable(freeze):
    fm = FillModel(freeze.cost)
    order = PaperOrder(
        order_id=new_order_id(),
        ticker="LIM",
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_px=40.0,  # mid 50 with slip still > 40
    )
    sim = fm.simulate_fill(order, FillQuote(mid=50.0))
    assert not sim.ok
