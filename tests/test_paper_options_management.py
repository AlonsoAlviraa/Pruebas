"""Unit tests: premium-seller management, haircut, assignment (no network)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.management import (
    apply_bid_haircut,
    can_roll,
    check_assignment,
    credit_captured_frac,
    management_from_meta,
    should_stop_loss,
    should_take_profit,
    structure_mark_to_close,
)
from paper_live.options.replay_options import book_delta_report, run_options_strategy
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.vol_surface import synthetic_vix_path


def test_bid_haircut_direction():
    mid = 2.0
    sell = apply_bid_haircut(mid, side="sell", haircut=0.05)
    buy = apply_bid_haircut(mid, side="buy", haircut=0.05)
    assert sell < mid < buy
    assert abs(sell - 1.9) < 1e-9
    assert abs(buy - 2.1) < 1e-9


def test_take_profit_50pct():
    # credit 100, mark 50 → captured 50%
    assert should_take_profit(100.0, 50.0, frac=0.50)
    assert not should_take_profit(100.0, 60.0, frac=0.50)
    assert abs(credit_captured_frac(100.0, 50.0) - 0.5) < 1e-9


def test_stop_loss_2x():
    # credit 100, mark 300 → loss 200 = 2× credit
    assert should_stop_loss(100.0, 300.0, mult=2.0)
    assert not should_stop_loss(100.0, 250.0, mult=2.0)


def test_max_rolls():
    assert can_roll(0, 1)
    assert not can_roll(1, 1)
    assert can_roll(1, 2)


def test_structure_mark_to_close_signed_no_zero_floor():
    """Long-wing-dominated book can produce negative debit (credit to close)."""
    m = structure_mark_to_close(
        short_put_mid=0.5,
        long_put_mid=2.0,
        contracts=1,
    )
    assert m < 0  # net credit to flatten
    # capture > 100% when mark negative
    assert credit_captured_frac(100.0, m) > 1.0
    assert should_take_profit(100.0, m, frac=0.50)


def test_assignment_put_expiry_itm():
    events = check_assignment(
        spot=90.0,
        short_put_k=100.0,
        short_call_k=None,
        contracts=2,
        stock_qty=0.0,
        at_expiry=True,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.leg == "short_put"
    assert ev.shares_delta == 200.0
    assert ev.cash_delta == -100.0 * 100.0 * 2
    assert ev.label == "assignment_proxy"


def test_assignment_call_covered_expiry():
    events = check_assignment(
        spot=110.0,
        short_put_k=None,
        short_call_k=100.0,
        contracts=1,
        stock_qty=100.0,
        at_expiry=True,
    )
    assert len(events) == 1
    assert events[0].shares_delta == -100.0
    assert events[0].cash_delta == 100.0 * 100.0


def test_management_meta_defaults():
    cfg = management_from_meta({}, kind="cash_secured_put")
    assert cfg.take_profit_credit_frac == 0.50
    assert cfg.stop_loss_credit_mult == 2.0
    assert cfg.max_rolls == 1
    assert cfg.bid_haircut == 0.05


def _feed_with_crash(
    n: int = 120,
    start: str = "2024-01-02",
    crash_start_i: int = 50,
    crash_days: int = 10,
    shock: float = -0.35,
) -> DailyReplayFeed:
    dates = pd.bdate_range(start=start, periods=n, tz="UTC")
    px = 400.0
    closes = []
    for i in range(n):
        if crash_start_i <= i < crash_start_i + crash_days:
            px *= (1.0 + shock / crash_days)
        else:
            px *= 1.0005
        closes.append(px)
    c = np.asarray(closes)
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": np.full(n, 2e6),
        }
    )
    vix = synthetic_vix_path(n, level=22.0, seed=3, start=start)
    # elevate VIX during crash
    vix = vix.copy()
    vix.loc[crash_start_i : crash_start_i + crash_days, "close"] = 40.0
    return DailyReplayFeed({"SPY": spy, "VIX": vix, "VIX3M": vix.copy()})


def test_replay_reports_vix_surface_label():
    feed = _feed_with_crash(n=100)
    days = feed.days
    spec = OptionStrategySpec(
        id="t_csp",
        label="test csp",
        kind="cash_secured_put",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        contracts=1,
        max_margin_fraction=0.8,
        meta={"max_rolls": 1, "bid_haircut": 0.05, "take_profit_credit_frac": 0.50},
    )
    r = run_options_strategy(
        feed, spec, start=days[20], end=days[-1], capital0=100_000.0
    )
    assert "vix_surface" in r.data_label or r.iv_source == "vix_surface"
    assert r.days_run > 0
    assert r.management.get("max_rolls") == 1


def test_replay_tp_on_quiet_path():
    """Low-vol grind up → short premium decays → TP must fire at least once."""
    n = 100
    dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    px = 400.0
    closes = []
    for _ in range(n):
        px *= 1.0012  # steady grind higher → OTM puts decay
        closes.append(px)
    c = np.asarray(closes)
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.001,
            "low": c * 0.999,
            "close": c,
            "volume": np.full(n, 1e6),
        }
    )
    vix = synthetic_vix_path(n, level=12.0, seed=0, start="2024-01-02")
    # flat low VIX
    vix = vix.copy()
    vix["close"] = 12.0
    vix["open"] = 12.0
    vix["high"] = 12.5
    vix["low"] = 11.5
    feed = DailyReplayFeed({"SPY": spy, "VIX": vix})
    days = feed.days
    spec = OptionStrategySpec(
        id="t_pcs",
        label="pcs tp",
        kind="put_credit_spread",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        wing_otm_pct=0.12,
        contracts=1,
        max_margin_fraction=0.5,
        roll_when_dte_below=3,
        meta={
            "take_profit_credit_frac": 0.25,  # reachable on quiet path
            "stop_loss_credit_mult": 99.0,
            "max_rolls": 0,
            "bid_haircut": 0.0,
            "enable_assignment_proxy": False,
        },
    )
    r = run_options_strategy(
        feed, spec, start=days[10], end=days[-1], capital0=100_000.0
    )
    assert r.n_tp >= 1, f"expected TP on quiet path; notes={r.notes[:8]}"
    assert r.n_sl == 0
    assert r.final_equity > 50_000


def test_replay_sl_on_crash_path():
    """Near-ATM short put into crash must hit stop-loss (≥1)."""
    feed = _feed_with_crash(n=100, crash_start_i=35, crash_days=10, shock=-0.45)
    days = feed.days
    spec = OptionStrategySpec(
        id="t_csp_sl",
        label="csp sl",
        kind="cash_secured_put",
        underlying="SPY",
        dte_days=60,
        otm_pct=0.01,  # nearly ATM → crash inflates mark hard
        contracts=1,
        max_margin_fraction=0.90,
        roll_when_dte_below=0,
        meta={
            "take_profit_credit_frac": 0.99,  # disable TP
            "stop_loss_credit_mult": 1.0,  # stop at 1× credit loss
            "max_rolls": 0,
            "bid_haircut": 0.0,
            "enable_assignment_proxy": False,
        },
    )
    r = run_options_strategy(
        feed, spec, start=days[15], end=days[-1], capital0=100_000.0
    )
    assert r.days_run > 10
    assert r.n_sl >= 1, f"expected SL; n_sl={r.n_sl} max_dd={r.max_dd} notes={r.notes[:10]}"


def test_haircut_reduces_entry_credit_vs_zero():
    n = 70
    dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    c = 400 * np.cumprod(1 + np.full(n, 0.0003))
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e6,
        }
    )
    vix = synthetic_vix_path(n, level=20.0, seed=2, start="2024-01-02")
    feed = DailyReplayFeed({"SPY": spy, "VIX": vix})
    days = feed.days
    base = dict(
        kind="cash_secured_put",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        contracts=1,
        max_margin_fraction=0.8,
        roll_when_dte_below=3,
    )
    r0 = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="h0",
            label="h0",
            meta={
                "bid_haircut": 0.0,
                "max_rolls": 0,
                "take_profit_credit_frac": 0.99,
                "stop_loss_credit_mult": 99,
                "enable_assignment_proxy": False,
            },
            **base,
        ),
        start=days[15],
        end=days[45],
        capital0=100_000.0,
    )
    r1 = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="h1",
            label="h1",
            meta={
                "bid_haircut": 0.20,
                "max_rolls": 0,
                "take_profit_credit_frac": 0.99,
                "stop_loss_credit_mult": 99,
                "enable_assignment_proxy": False,
            },
            **base,
        ),
        start=days[15],
        end=days[45],
        capital0=100_000.0,
    )
    # With haircut, seller keeps less premium → final equity should not exceed zero-haircut
    assert r1.final_equity <= r0.final_equity + 1.0


def test_assignment_on_expiry_itm_path():
    """Force short put deep ITM into expiry → assignment_proxy note."""
    n = 50
    dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    # drop hard after open
    closes = []
    px = 100.0
    for i in range(n):
        if i < 10:
            px = 100.0
        else:
            px = 70.0
        closes.append(px)
    c = np.asarray(closes, dtype=float)
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e6,
        }
    )
    vix = synthetic_vix_path(n, level=30.0, seed=5, start="2024-01-02")
    feed = DailyReplayFeed({"SPY": spy, "VIX": vix})
    days = feed.days
    spec = OptionStrategySpec(
        id="assign",
        label="assign",
        kind="cash_secured_put",
        underlying="SPY",
        dte_days=15,
        otm_pct=0.01,
        contracts=1,
        max_margin_fraction=0.9,
        roll_when_dte_below=0,  # never roll early
        meta={
            "max_rolls": 0,
            "take_profit_credit_frac": 0.99,
            "stop_loss_credit_mult": 99,
            "enable_assignment_proxy": True,
            "bid_haircut": 0.0,
            "deep_itm_assign_pct": 0.05,
        },
    )
    r = run_options_strategy(
        feed, spec, start=days[5], end=days[-1], capital0=100_000.0
    )
    assert r.n_assign >= 1 or any("ASSIGNMENT_PROXY" in n for n in r.notes)


def test_max_one_dte_roll_blocks_second():
    """With max_rolls=1, n_dte_rolls per structure lifetime is capped; counter is DTE-only."""
    n = 160
    dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    c = 400 * np.cumprod(1 + np.full(n, 0.0004))
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": 1e6,
        }
    )
    vix = synthetic_vix_path(n, level=18.0, seed=7, start="2024-01-02")
    feed = DailyReplayFeed({"SPY": spy, "VIX": vix})
    days = feed.days
    # Short DTE + early roll threshold → many roll opportunities if uncapped
    spec = OptionStrategySpec(
        id="roll_cap",
        label="roll cap",
        kind="cash_secured_put",
        underlying="SPY",
        dte_days=20,
        otm_pct=0.08,
        contracts=1,
        max_margin_fraction=0.85,
        roll_when_dte_below=10,  # roll half-way through life
        meta={
            "max_rolls": 1,
            "take_profit_credit_frac": 0.99,  # no TP
            "stop_loss_credit_mult": 99.0,
            "bid_haircut": 0.0,
            "enable_assignment_proxy": False,
        },
    )
    r = run_options_strategy(
        feed, spec, start=days[10], end=days[-1], capital0=100_000.0
    )
    # Opens include initial + rolls + re-entries after structure ends
    assert r.n_opens == r.n_rolls  # legacy alias
    assert r.n_opens >= 1
    # DTE rolls are a subset of opens and must be less than opens if any initial entry
    assert r.n_dte_rolls < r.n_opens or r.n_opens == 0
    # With max_rolls=1, each structure allows at most 1 DTE roll.
    # Over a long window many structures can each roll once → n_dte_rolls can be > 1,
    # but ratio of dte rolls to opens should stay ≤ 0.5 when every structure rolls once
    # (1 open + 1 roll) → 0.5. Allow small slack for partial last structure.
    if r.n_opens >= 2:
        assert r.n_dte_rolls / r.n_opens <= 0.55 + 1e-9
    # Sanity: without cap we'd get ~1 roll per ~10 days of life; with cap still works
    assert r.n_dte_rolls >= 1, "expected at least one DTE roll in long window"


def test_pcs_assignment_settles_long_put_not_forfeit():
    """PCS short put ITM: long put value must be credited (assign-on vs assign-off equity gap small)."""
    n = 45
    dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    closes = []
    px = 100.0
    for i in range(n):
        # flat then crash through short and into long wing
        if i < 8:
            px = 100.0
        else:
            px = 70.0
        closes.append(px)
    c = np.asarray(closes, dtype=float)
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e6,
        }
    )
    vix = synthetic_vix_path(n, level=25.0, seed=9, start="2024-01-02")
    feed = DailyReplayFeed({"SPY": spy, "VIX": vix})
    days = feed.days
    common = dict(
        kind="put_credit_spread",
        underlying="SPY",
        dte_days=12,
        otm_pct=0.02,
        wing_otm_pct=0.15,
        contracts=1,
        max_margin_fraction=0.50,
        roll_when_dte_below=0,
    )
    meta_base = {
        "max_rolls": 0,
        "take_profit_credit_frac": 0.99,
        "stop_loss_credit_mult": 99.0,
        "bid_haircut": 0.0,
        "deep_itm_assign_pct": 0.05,
    }
    r_on = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="pcs_on",
            label="pcs on",
            meta={**meta_base, "enable_assignment_proxy": True},
            **common,
        ),
        start=days[3],
        end=days[-1],
        capital0=100_000.0,
    )
    r_off = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="pcs_off",
            label="pcs off",
            meta={**meta_base, "enable_assignment_proxy": False},
            **common,
        ),
        start=days[3],
        end=days[-1],
        capital0=100_000.0,
    )
    assert r_on.n_assign >= 1
    assert any("settle long_put" in n for n in r_on.notes)
    # Equity with assignment must not be massively worse due to forfeited long wing
    # (allow small differences from exercise stock path vs mid settle)
    gap = r_on.final_equity - r_off.final_equity
    assert gap > -500.0, (
        f"assignment forfeited long wing value: on={r_on.final_equity:.2f} "
        f"off={r_off.final_equity:.2f} gap={gap:.2f}"
    )


def test_ccs_assignment_settles_long_call():
    """CCS: short call ITM cash-assign must settle long call (no free wipe)."""
    n = 45
    dates = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    closes = []
    for i in range(n):
        closes.append(100.0 if i < 8 else 140.0)
    c = np.asarray(closes, dtype=float)
    spy = pd.DataFrame(
        {
            "date": dates,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e6,
        }
    )
    vix = synthetic_vix_path(n, level=22.0, seed=11, start="2024-01-02")
    feed = DailyReplayFeed({"SPY": spy, "VIX": vix})
    days = feed.days
    common = dict(
        kind="call_credit_spread",
        underlying="SPY",
        dte_days=12,
        otm_pct=0.02,
        wing_otm_pct=0.15,
        contracts=1,
        max_margin_fraction=0.50,
        roll_when_dte_below=0,
    )
    meta_base = {
        "max_rolls": 0,
        "take_profit_credit_frac": 0.99,
        "stop_loss_credit_mult": 99.0,
        "bid_haircut": 0.0,
        "deep_itm_assign_pct": 0.05,
    }
    r_on = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="ccs_on",
            label="ccs on",
            meta={**meta_base, "enable_assignment_proxy": True},
            **common,
        ),
        start=days[3],
        end=days[-1],
        capital0=100_000.0,
    )
    r_off = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="ccs_off",
            label="ccs off",
            meta={**meta_base, "enable_assignment_proxy": False},
            **common,
        ),
        start=days[3],
        end=days[-1],
        capital0=100_000.0,
    )
    assert r_on.n_assign >= 1
    assert any("settle long_call" in n for n in r_on.notes)
    gap = r_on.final_equity - r_off.final_equity
    assert gap > -500.0, (
        f"CCS long call forfeited: on={r_on.final_equity:.2f} off={r_off.final_equity:.2f}"
    )


def test_book_delta_report():
    feed = _feed_with_crash(n=80)
    days = feed.days
    specs = [
        OptionStrategySpec(
            id="cc",
            label="cc",
            kind="covered_call",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            max_margin_fraction=0.95,
            meta={"max_rolls": 1, "enable_assignment_proxy": False},
        ),
        OptionStrategySpec(
            id="cash",
            label="cash",
            kind="cash",
            underlying="SPY",
            hard_kill_enabled=False,
        ),
    ]
    from paper_live.options.replay_options import run_options_batch

    results = run_options_batch(
        feed, specs, start=days[15], end=days[-1], capital0=100_000.0
    )
    book = book_delta_report(results)
    assert book["n_strategies"] >= 1
    assert "sum_delta_end" in book
    assert book["label"] == "approx_bs_delta_book"
