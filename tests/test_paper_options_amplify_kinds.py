"""Tests for debit/PMCC amplify option kinds (synthetic, no network)."""
from __future__ import annotations

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.replay_options import run_options_strategy
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.vol_surface import synthetic_vix_path


def _feed(n=150):
    feed = DailyReplayFeed.from_synthetic(["SPY"], start="2023-01-03", n_days=n, seed=3)
    vix = synthetic_vix_path(n, level=17, seed=4, start="2023-01-03")
    raw = dict(feed._raw)
    raw["VIX"] = vix
    return DailyReplayFeed(raw, min_history=40)


def test_long_call_and_debit_spreads_run():
    feed = _feed()
    start, end = feed.days[60], feed.days[-1]
    for kind in ("long_call", "long_put", "call_debit_spread", "put_debit_spread", "pmcc"):
        sp = OptionStrategySpec(
            id=f"t_{kind}",
            label=kind,
            kind=kind,
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            wing_otm_pct=0.12,
            contracts=3,
            meta={"max_premium_budget_frac": 0.12},
        )
        r = run_options_strategy(feed, sp, start=start, end=end, capital0=100_000.0)
        assert r.days_run > 0
        assert "proxy" in r.data_label or "vix" in r.data_label
        assert r.n_opens >= 0
