"""Unit + smoke tests for options TA/volume gates (synthetic feed, no network)."""
from __future__ import annotations

import pandas as pd

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.options.replay_options import run_options_strategy
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.ta_gates import evaluate_ta_gates, should_skip_new_from_meta


def _tiny_feed(tickers=None, n: int = 100, seed: int = 7) -> DailyReplayFeed:
    return DailyReplayFeed.from_synthetic(
        tickers or ["SPY", "QQQ"],
        start="2024-01-02",
        n_days=n,
        seed=seed,
    )


def _short_history_feed(n: int = 80) -> DailyReplayFeed:
    """Panel long enough for features but often short of solid SMA200."""
    dates = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    close = 100.0 * (1.002 ** pd.Series(range(n))).values
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [100_000.0] * n,
        }
    )
    return DailyReplayFeed({"SPY": df}, min_history=50)


def test_evaluate_no_ta_gates_allows():
    feed = _tiny_feed()
    day = feed.days[-1]
    r = evaluate_ta_gates(feed, "SPY", day, meta={})
    assert r.allow is True
    assert r.reason == "no_ta_gates"


def test_require_uptrend_gate_runs():
    feed = _tiny_feed(n=220, seed=3)
    day = feed.days[-1]
    r = evaluate_ta_gates(feed, "SPY", day, meta={"require_uptrend": True})
    assert r.reason in (
        "ta_gates_pass",
        "uptrend_below_sma50",
        "uptrend_below_sma200",
        "uptrend_no_close",
        "uptrend_missing_sma",
        "no_features",
    )
    assert isinstance(r.allow, bool)


def test_require_uptrend_fails_closed_when_sma200_missing():
    """Issue 1: short history → SMA200 NaN must not fail-open on require_uptrend."""
    feed = _short_history_feed(n=80)
    day = feed.days[-1]
    feat = feed.featured("SPY", through=day)
    assert not feat.empty
    row = feat.iloc[-1]
    sma200 = row.get("sma_200")
    # If SMA200 is still finite (feature min_periods lower), force check via meta path
    r = evaluate_ta_gates(feed, "SPY", day, meta={"require_uptrend": True})
    if pd.isna(sma200) or not pd.notna(sma200) or float(sma200) != float(sma200):
        assert r.allow is False
        assert r.reason == "uptrend_missing_sma"
    else:
        # Long enough SMA200: still must require both SMAs finite and close above
        assert r.reason in (
            "ta_gates_pass",
            "uptrend_below_sma50",
            "uptrend_below_sma200",
            "uptrend_missing_sma",
        )
        # Explicit NaN override via synthetic row is covered by force-panel below
        dates = pd.bdate_range("2024-01-02", periods=60, tz="UTC")
        close = [100.0 + i * 0.1 for i in range(60)]
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": [c * 1.01 for c in close],
                "low": [c * 0.99 for c in close],
                "close": close,
                "volume": [50_000.0] * 60,
            }
        )
        short = DailyReplayFeed({"SPY": df}, min_history=40)
        d2 = short.days[-1]
        r2 = evaluate_ta_gates(short, "SPY", d2, meta={"require_uptrend": True})
        # 60 bars: SMA200 typically NaN
        assert r2.allow is False
        assert r2.reason in ("uptrend_missing_sma", "no_features", "uptrend_no_close")


def test_volume_confirm_and_dryup_opposite():
    """Construct synthetic panels with forced volume extremes."""
    dates = pd.bdate_range("2023-01-02", periods=80, tz="UTC")
    close = 100.0 * (1.01 ** pd.Series(range(80))).values
    vol = [100_000.0] * 79 + [500_000.0]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": vol,
        }
    )
    feed = DailyReplayFeed({"SPY": df}, min_history=50)
    day = feed.days[-1]
    conf = evaluate_ta_gates(
        feed,
        "SPY",
        day,
        meta={"require_volume_confirm": True, "min_volume_ratio": 1.5},
    )
    dry = evaluate_ta_gates(
        feed,
        "SPY",
        day,
        meta={"require_volume_dryup": True, "max_volume_ratio": 0.8},
    )
    assert conf.reason != "no_features"
    assert dry.reason != "no_features"
    assert conf.allow is True
    assert dry.allow is False


def test_should_skip_helper_blocks_impossible_rsi():
    feed = _tiny_feed()
    day = feed.days[-1]
    skip = should_skip_new_from_meta(
        feed, "SPY", day, {"require_rsi_overbought": True, "min_rsi": 99.9}
    )
    assert skip is True
    r = evaluate_ta_gates(
        feed, "SPY", day, meta={"require_rsi_overbought": True, "min_rsi": 99.9}
    )
    assert r.allow is False
    assert r.reason == "rsi_not_overbought"


def test_options_replay_with_ta_meta_smoke():
    feed = _tiny_feed(n=90, seed=2)
    days = feed.days
    r = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="t_csp_range",
            label="csp range",
            kind="cash_secured_put",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            max_margin_fraction=0.8,
            meta={
                "require_range_regime": True,
                "max_atr_pctile": 0.95,
                "max_dist_sma50": 0.50,
            },
        ),
        start=days[25],
        end=days[-1],
        capital0=100_000.0,
    )
    assert r.days_run > 0
    assert r.data_label.startswith("proxy_bs")
    assert r.final_equity > 0


def test_options_replay_strict_gate_blocks_opens():
    """Impossible RSI overbought gate: never open; equity stays cash; proxy_bs."""
    capital0 = 100_000.0
    feed = _tiny_feed(n=80, seed=11)
    days = feed.days
    r = run_options_strategy(
        feed,
        OptionStrategySpec(
            id="t_pp_strict",
            label="pp strict",
            kind="protective_put",
            underlying="SPY",
            dte_days=21,
            otm_pct=0.05,
            max_margin_fraction=0.95,
            meta={"require_rsi_overbought": True, "min_rsi": 99.0},
        ),
        start=days[25],
        end=days[-1],
        capital0=capital0,
    )
    assert r.days_run > 0
    assert r.n_rolls == 0
    assert r.final_equity == capital0
    assert r.data_label.startswith("proxy_bs")
    assert any("TA gate skip" in n for n in r.notes)


def test_range_fail_reason_not_inverted():
    """Issue 5: extended price uses not_in_range (not not_range_extended)."""
    feed = _tiny_feed(n=120, seed=1)
    day = feed.days[-1]
    r = evaluate_ta_gates(
        feed,
        "SPY",
        day,
        meta={
            "require_range_regime": True,
            "max_atr_pctile": 1.0,  # do not fail on ATR
            "max_dist_sma50": 0.0,  # any nonzero dist fails
        },
    )
    if not r.allow and r.reason not in ("atr_not_low", "no_features"):
        assert r.reason == "not_in_range"
