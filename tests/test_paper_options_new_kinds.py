"""Smoke tests for iron condor / CCS / protective put (proxy_bs)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.replay_options import run_options_strategy
from paper_live.options.strategies import OptionStrategySpec


def _tiny_feed(ticker: str = "SPY", n: int = 80) -> DailyReplayFeed:
    return DailyReplayFeed.from_synthetic(
        [ticker],
        start="2024-01-02",
        n_days=n,
        seed=1,
    )


def test_iron_condor_runs():
    feed = _tiny_feed()
    days = feed.days
    r = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="t_ic",
            label="ic",
            kind="iron_condor",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            wing_otm_pct=0.10,
            max_margin_fraction=0.5,
        ),
        start=days[20],
        end=days[-1],
        capital0=100_000.0,
    )
    assert r.days_run > 0
    assert r.defined_risk is True


def test_call_credit_spread_runs():
    feed = _tiny_feed()
    days = feed.days
    r = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="t_ccs",
            label="ccs",
            kind="call_credit_spread",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            wing_otm_pct=0.10,
            max_margin_fraction=0.5,
        ),
        start=days[20],
        end=days[-1],
        capital0=100_000.0,
    )
    assert r.days_run > 0
