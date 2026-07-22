"""OPT-PR3: risk gates, margin sizing, CVaR metrics (synthetic, no network)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.metrics import cvar, max_drawdown, metrics_from_curve
from paper_live.options.replay_options import run_options_strategy
from paper_live.options.risk import (
    OptionsRiskConfig,
    check_hard_kill,
    margin_at_risk_per_contract,
    size_contracts,
)
from paper_live.options.strategies import OptionStrategySpec


def _synthetic_feed(
    n: int = 120,
    start: date = date(2024, 1, 2),
    spot0: float = 400.0,
    daily_ret: float = 0.0005,
    shock_day: int | None = None,
    shock: float = -0.12,
    gap_day: int | None = None,
) -> DailyReplayFeed:
    rng = np.random.default_rng(42)
    rows = []
    px = spot0
    d = start
    for i in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        if gap_day is not None and i == gap_day:
            # skip writing this session (missing bar) but advance calendar
            d += timedelta(days=1)
            continue
        r = daily_ret + float(rng.normal(0, 0.005))
        if shock_day is not None and i == shock_day:
            r = shock
        px = px * (1.0 + r)
        rows.append(
            {
                "date": pd.Timestamp(d, tz="UTC"),
                "open": px,
                "high": px * 1.005,
                "low": px * 0.995,
                "close": px,
                "volume": 1e6,
            }
        )
        d += timedelta(days=1)
    df = pd.DataFrame(rows)
    return DailyReplayFeed({"SPY": df, "QQQ": df.copy()})


def test_cvar_worst_tail():
    rets = [0.01, 0.02, -0.05, -0.08, 0.0, 0.01, -0.03, 0.02, -0.01, 0.0]
    es = cvar(rets, alpha=0.05)
    assert es is not None
    assert es <= -0.05
    es2 = cvar(rets, alpha=0.20)
    assert es2 is not None
    assert es2 < 0


def test_cvar_edge_cases():
    assert cvar([]) is None
    assert cvar([0.01], alpha=0) is None
    assert cvar([0.01], alpha=1.0) is None
    assert cvar([0.0, 0.0, 0.0], alpha=0.05) == pytest.approx(0.0)
    m_empty = metrics_from_curve([], capital0=100_000.0)
    assert m_empty["cvar_5pct"] is None
    assert m_empty["n_days"] == 0
    m_one = metrics_from_curve(
        [{"date": "2024-01-02", "equity": 100_000.0}], capital0=100_000.0
    )
    assert m_one["cvar_5pct"] is None
    assert m_one["worst_day"] is None
    flat = [
        {"date": f"2024-01-{i:02d}", "equity": 100_000.0}
        for i in range(2, 12)
    ]
    m_flat = metrics_from_curve(flat, capital0=100_000.0)
    assert m_flat["cvar_5pct"] == pytest.approx(0.0)


def test_max_drawdown_and_metrics():
    curve = [
        {"date": "2024-01-01", "equity": 100.0},
        {"date": "2024-01-02", "equity": 110.0},
        {"date": "2024-01-03", "equity": 90.0},
        {"date": "2024-01-04", "equity": 95.0},
    ]
    assert max_drawdown([100, 110, 90, 95]) == pytest.approx(90 / 110 - 1.0)
    m = metrics_from_curve(curve, capital0=100.0)
    assert m["max_dd"] < 0
    assert m["cvar_5pct"] is not None
    assert m["total_return"] == pytest.approx(-0.05)


def test_margin_pcs_defined_risk():
    mar = margin_at_risk_per_contract(
        "put_credit_spread",
        spot=400.0,
        short_strike=380.0,
        long_strike=360.0,
    )
    assert mar == pytest.approx(20.0 * 100.0)
    n = size_contracts(
        "put_credit_spread",
        capital0=100_000.0,
        spot=400.0,
        risk=OptionsRiskConfig(max_margin_fraction=0.40),
        short_strike=380.0,
        long_strike=360.0,
        requested=50,
    )
    assert n == 20


def test_margin_strict_no_fallback():
    """size_contracts returns 0 when budget cannot fund 1 contract — no silent 1-lot."""
    n = size_contracts(
        "cash_secured_put",
        capital0=10_000.0,
        spot=400.0,
        risk=OptionsRiskConfig(max_margin_fraction=0.10),
        short_strike=380.0,
        requested=1,
    )
    # budget 1k < 38k collateral
    assert n == 0


def test_margin_csp_full_collateral():
    mar = margin_at_risk_per_contract(
        "cash_secured_put", spot=400.0, short_strike=380.0
    )
    assert mar == pytest.approx(380.0 * 100.0)
    n = size_contracts(
        "cash_secured_put",
        capital0=100_000.0,
        spot=400.0,
        risk=OptionsRiskConfig(max_margin_fraction=0.50),
        short_strike=380.0,
        requested=10,
    )
    assert n == 1


def test_hard_kill_dd_and_day():
    risk = OptionsRiskConfig(max_portfolio_dd=0.10, max_single_day_drop=0.05)
    kill, reason = check_hard_kill(equity=85_000, peak=100_000, prev_equity=100_000, risk=risk)
    assert kill and "max_portfolio_dd" in reason
    kill2, reason2 = check_hard_kill(
        equity=94_000, peak=100_000, prev_equity=100_000, risk=risk
    )
    assert kill2 and "max_single_day_drop" in reason2
    ok, _ = check_hard_kill(equity=99_000, peak=100_000, prev_equity=100_000, risk=risk)
    assert not ok
    # gap: prev_equity None → day-drop not checked
    ok2, _ = check_hard_kill(equity=90_000, peak=100_000, prev_equity=None, risk=risk)
    assert not ok2  # -10% dd vs 10% threshold is equal? 90/100-1 = -0.10 <= -0.10 → kill DD
    # use mild equity still above DD
    ok3, _ = check_hard_kill(equity=95_000, peak=100_000, prev_equity=None, risk=risk)
    assert not ok3


def test_put_credit_spread_replay_defined_risk():
    feed = _synthetic_feed(n=90)
    days = feed.days
    start, end = days[20], days[-1]
    spec = OptionStrategySpec(
        id="t_pcs",
        label="test pcs",
        kind="put_credit_spread",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        wing_otm_pct=0.15,
        contracts=5,
        max_portfolio_dd=0.50,
        max_single_day_drop=0.50,
        max_margin_fraction=0.40,
    )
    r = run_options_strategy(feed, spec, start=start, end=end, capital0=100_000.0)
    assert r.data_label.startswith("proxy_bs")
    assert r.iv_source in ("proxy_hv", "vix_surface", "vix_surface_partial")
    assert r.defined_risk is True
    assert r.days_run > 10
    assert r.cvar_5pct is not None or r.days_run < 5


def test_collar_replay_no_negative_cash():
    feed = _synthetic_feed(n=80, spot0=400.0)
    days = feed.days
    start, end = days[15], days[-1]
    spec = OptionStrategySpec(
        id="t_collar",
        label="test collar",
        kind="collar",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        wing_otm_pct=0.08,
        max_portfolio_dd=0.50,
        max_single_day_drop=0.50,
        max_margin_fraction=0.95,
        premium_mult=2.0,  # rich IV → expensive puts
        contracts=1,
    )
    r = run_options_strategy(feed, spec, start=start, end=end, capital0=100_000.0)
    assert r.defined_risk is True
    for row in r.equity_curve:
        assert row["equity"] > 0


def test_collar_refuse_unwinds_stock_no_naked_long():
    """Failed collar open must not leave stock without the long put (Issue 19)."""
    from paper_live.options import replay_options as ro

    # Patch put marks to be huge so put debit is never affordable after stock buy
    real_bs = ro.black_scholes_price

    def expensive_put(spot, k, t, vol, r=0.02, *, option_type="call", q=0.0):
        px = real_bs(spot, k, t, vol, r, option_type=option_type, q=q)
        if option_type == "put":
            return max(px, 50.0)  # $50 * 100 = $5k/contract — drain cash after stock
        return px

    feed = _synthetic_feed(n=40, spot0=100.0)
    days = feed.days
    start, end = days[5], days[25]
    spec = OptionStrategySpec(
        id="t_collar_refuse",
        label="collar refuse",
        kind="collar",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        wing_otm_pct=0.08,
        contracts=1,
        max_portfolio_dd=0.90,
        max_single_day_drop=0.90,
        max_margin_fraction=0.95,
        premium_mult=1.15,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ro, "black_scholes_price", expensive_put)
        r = run_options_strategy(feed, spec, start=start, end=end, capital0=12_000.0)

    # Never opened a protected collar → contracts_used may be 0; equity ≈ capital
    assert r.contracts_used == 0 or any("collar skip" in n for n in r.notes)
    # Final equity should be near cash capital (stock not stuck naked)
    assert r.final_equity > 10_000.0
    # No open contracts on curve if structure never succeeded
    if r.n_rolls == 0:
        assert all(int(row.get("contracts") or 0) == 0 for row in r.equity_curve)


def test_hard_kill_liquidates_and_blocks_new_entries():
    """Integration: forced breach → hard_kill, flat book, no later rolls/contracts."""
    # Nearly fully invested CC: −20% shock vs 8% DD gate must hard-kill and liquidate.
    feed = _synthetic_feed(n=60, daily_ret=0.0002, shock_day=30, shock=-0.20, spot0=100.0)
    days = feed.days
    start, end = days[10], days[-1]
    spec = OptionStrategySpec(
        id="t_cc_kill",
        label="cc kill",
        kind="covered_call",
        underlying="SPY",
        dte_days=45,
        otm_pct=0.10,
        contracts=9,  # ~90k stock notion at $100 → ~18% equity hit on −20% shock
        max_portfolio_dd=0.08,
        max_single_day_drop=0.50,  # only DD gate fires (not day-drop)
        max_margin_fraction=0.95,
        hard_kill_enabled=True,
    )
    r = run_options_strategy(feed, spec, start=start, end=end, capital0=100_000.0)
    assert r.hard_kill is True, (r.max_dd, r.notes, r.contracts_used)
    assert "max_portfolio_dd" in r.hard_kill_reason
    assert any("HARD_KILL" in n for n in r.notes)
    kill_seen = False
    for row in r.equity_curve:
        if row.get("hard_kill"):
            kill_seen = True
            assert int(row.get("contracts") or 0) == 0
        elif kill_seen:
            assert int(row.get("contracts") or 0) == 0
    assert kill_seen
    tail = r.equity_curve[-5:]
    assert all(int(x.get("contracts") or 0) == 0 for x in tail)
    assert r.n_rolls >= 1


class _GapFeed:
    """Feed wrapper that returns None bar on one session (multi-session gap)."""

    def __init__(self, base, gap_on: date, und: str = "SPY"):
        self._base = base
        self._gap_on = gap_on
        self._und = und

    def session_days(self, start, end):
        return self._base.session_days(start, end)

    def bar(self, ticker, day):
        if ticker.upper() == self._und and day == self._gap_on:
            return None
        return self._base.bar(ticker, day)

    def history(self, ticker, *, through, include_through=True):
        return self._base.history(ticker, through=through, include_through=include_through)


def test_gap_day_does_not_false_kill_on_multiday_move():
    """Post-gap −10% move must not trip max_single_day_drop when prev_eq cleared."""
    # Nearly fully invested CC; day-drop gate tight; portfolio DD gate loose.
    # Without gap handling, equity≈−10% day would hard-kill via max_single_day_drop.
    rows = []
    px = 100.0
    d = date(2024, 3, 1)
    dates: list[date] = []
    for i in range(45):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        if i == 22:
            px *= 0.90  # −10% after gap day (gap_on = dates[21])
        else:
            px *= 1.0002
        dates.append(d)
        rows.append(
            {
                "date": pd.Timestamp(d, tz="UTC"),
                "open": px,
                "high": px * 1.002,
                "low": px * 0.998,
                "close": px,
                "volume": 1e6,
            }
        )
        d += timedelta(days=1)
    base = DailyReplayFeed({"SPY": pd.DataFrame(rows)})
    gap_on = dates[21]
    gfeed = _GapFeed(base, gap_on)
    spec = OptionStrategySpec(
        id="t_gap_cc",
        label="gap cc",
        kind="covered_call",
        underlying="SPY",
        dte_days=60,
        otm_pct=0.15,
        contracts=9,
        max_portfolio_dd=0.50,  # −10% is OK for DD gate
        max_single_day_drop=0.05,  # would fire if prev_eq spanned the gap
        max_margin_fraction=0.95,
        hard_kill_enabled=True,
    )
    r = run_options_strategy(
        gfeed, spec, start=dates[5], end=dates[-1], capital0=100_000.0
    )
    assert r.hard_kill is False, (r.hard_kill_reason, r.max_dd, r.notes)
    # Curve marks session_gap on first bar after missing day
    gap_flags = [row.get("session_gap") for row in r.equity_curve]
    assert any(gap_flags), "expected session_gap annotation after missing bar"


def test_session_gap_excluded_from_cvar_worst_day():
    """Gap-spanning equity jump must not dominate CVaR/worst_day (Issue 20)."""
    from paper_live.options.metrics import metrics_from_curve, session_returns_from_curve

    curve = [
        {"date": "2024-01-02", "equity": 100.0},
        {"date": "2024-01-03", "equity": 101.0},
        # missing sessions then resume — large jump
        {"date": "2024-01-10", "equity": 80.0, "session_gap": True},
        {"date": "2024-01-11", "equity": 80.5},
    ]
    rets = session_returns_from_curve(curve)
    # Only 100→101 and 80→80.5; not 101→80
    assert all(r > -0.05 for r in rets)
    m = metrics_from_curve(curve, capital0=100.0)
    assert m["worst_day"] is not None
    assert m["worst_day"] > -0.15  # not the −20.8% gap jump
