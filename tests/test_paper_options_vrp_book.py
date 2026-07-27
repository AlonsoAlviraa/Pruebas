"""Tests: VRP/IV-rank gates, time exit, beta-weighted book, sleeve."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.book import (
    beta_weighted_delta,
    book_delta_report_beta,
    build_sleeve_portfolio,
    rolling_beta_to_spy,
)
from paper_live.options.management import (
    management_action,
    management_from_meta,
    should_time_exit,
)
from paper_live.options.vol_surface import (
    series_percentile_rank,
    synthetic_vix_path,
    vrp_proxy,
    vix_term_contango,
    atm_iv_proxy_for_day,
)
from paper_live.options.ta_gates import evaluate_ta_gates


def test_vrp_proxy_and_percentile():
    assert abs(vrp_proxy(0.20, 0.15) - 0.05) < 1e-9
    assert vrp_proxy(float("nan"), 0.1) is None
    s = pd.Series(np.linspace(10, 30, 100))
    r = series_percentile_rank(s, lookback=100)
    assert r is not None and r > 0.9


def test_vix_term_contango():
    assert vix_term_contango(18.0, 20.0) is True
    assert vix_term_contango(22.0, 20.0) is False
    assert vix_term_contango(None, 20.0) is None


def test_atm_iv_proxy_labels():
    iv, src = atm_iv_proxy_for_day(vix=20.0, vix3m=21.0)
    assert src == "vix_surface"
    assert 0.15 < iv < 0.30
    iv2, src2 = atm_iv_proxy_for_day(vix=None, hv=0.16, premium_mult=1.15)
    assert src2 == "proxy_hv"
    assert iv2 > 0


def test_time_exit_rules():
    assert should_time_exit(
        dte=5, initial_credit=100.0, mark_to_close=20.0, time_exit_dte=7, residual_frac=0.25
    )
    # residual still high
    assert not should_time_exit(
        dte=5, initial_credit=100.0, mark_to_close=80.0, time_exit_dte=7, residual_frac=0.25
    )
    # DTE too high
    assert not should_time_exit(
        dte=20, initial_credit=100.0, mark_to_close=10.0, time_exit_dte=7, residual_frac=0.25
    )
    # TP disabled/high so time_exit can fire (default TP@50% would win first)
    cfg = management_from_meta(
        {
            "take_profit_credit_frac": 0.99,
            "stop_loss_credit_mult": 99.0,
            "time_exit_dte": 7,
            "time_exit_residual_credit_frac": 0.25,
        },
        kind="put_credit_spread",
    )
    act = management_action(
        kind="put_credit_spread",
        initial_credit=100.0,
        mark_to_close=20.0,  # 80% captured → residual 20% ≤ 25%
        cfg=cfg,
        dte=5,
    )
    assert act == "time_exit"


def test_iv_rank_gate_on_synthetic_vix():
    n = 280
    spy = DailyReplayFeed.from_synthetic(["SPY"], start="2023-01-02", n_days=n, seed=1)
    vix_df = synthetic_vix_path(n, level=22.0, seed=2, start="2023-01-02")
    # merge VIX into feed panels
    raw = dict(spy._raw)  # noqa: SLF001
    raw["VIX"] = vix_df
    feed = DailyReplayFeed(raw, min_history=50)
    day = feed.days[-1]
    r = evaluate_ta_gates(
        feed,
        "SPY",
        day,
        meta={
            "require_iv_rank_above": True,
            "min_iv_rank": 0.01,  # almost always pass if series exists
        },
    )
    assert r.reason in ("ta_gates_pass", "iv_rank_too_low", "no_features")
    assert "iv_rank" in (r.features or {}) or r.reason == "iv_rank_too_low"

    r2 = evaluate_ta_gates(
        feed,
        "SPY",
        day,
        meta={"require_vrp_proxy_above": True, "min_vrp_proxy": 99.0},
    )
    assert r2.allow is False
    assert r2.reason == "vrp_proxy_too_low"


def test_beta_weighted_and_sleeve():
    feed = DailyReplayFeed.from_synthetic(
        ["SPY", "QQQ"], start="2024-01-02", n_days=120, seed=9
    )
    day = feed.days[-1]
    b = rolling_beta_to_spy(feed, "QQQ", day, window=40)
    assert b is None or np.isfinite(b)
    assert abs(beta_weighted_delta(100.0, 1.2) - 120.0) < 1e-9

    class R:
        def __init__(self, sid, kind, und, curve, fe, de):
            self.strategy_id = sid
            self.kind = kind
            self.underlying = und
            self.equity_curve = curve
            self.final_equity = fe
            self.approx_delta_end = de
            self.total_return = fe / 100_000.0 - 1.0

    # two rising curves
    dates = [day - timedelta(days=i) for i in range(30, 0, -1)]
    c1 = [{"date": d.isoformat(), "equity": 100_000 * (1 + 0.001 * i)} for i, d in enumerate(dates)]
    c2 = [{"date": d.isoformat(), "equity": 100_000 * (1 + 0.0005 * i)} for i, d in enumerate(dates)]
    results = [
        R("a", "covered_call", "SPY", c1, c1[-1]["equity"], 50.0),
        R("b", "put_credit_spread", "SPY", c2, c2[-1]["equity"], 10.0),
        R("c", "cash", "SPY", [{"date": d.isoformat(), "equity": 100_000} for d in dates], 100_000, 0.0),
    ]
    sleeve = build_sleeve_portfolio(results, capital0=100_000.0)
    assert sleeve.total_return != 0.0 or len(sleeve.equity_curve) > 0
    assert "covered_call" in sleeve.weights or sleeve.notes

    bw = book_delta_report_beta(results, feed, day)
    assert bw["label"] == "beta_weighted_delta"
    assert bw["n_strategies"] >= 2
