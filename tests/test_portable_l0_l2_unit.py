"""Unit tests for ALPHA-PORTABLE L0/L1/L2 pure logic — synthetic panels only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trad_research.portable.membership_l0 import (
    L0Config,
    estimate_adv_usd,
    filter_liquidity,
    membership_mask_panel,
    rebalance_dates,
    select_members,
)
from trad_research.portable.portfolio_l2 import (
    PortfolioL2Config,
    build_weight_panel,
    build_weights_for_date,
    equity_from_returns,
    hold_weights_across_calendar,
    portfolio_returns_from_weights,
    top_k_equal_weight,
    top_k_score_weight,
)
from trad_research.portable.residual_labels import (
    beat_style_meta_frame,
    residual_excess_series,
)
from trad_research.portable.score_l1 import (
    ThinMLStub,
    rule_rank_scores,
    score_panel_l1,
    scores_to_signals,
)


def _synth_panel(n_days: int = 30, n_tickers: int = 5) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B", tz="UTC")
    for t_i in range(n_tickers):
        tkr = f"T{t_i}"
        px = 50 + t_i * 10
        for d in dates:
            ret = float(rng.normal(0.001, 0.02))
            px = px * (1 + ret)
            rows.append(
                {
                    "date": d,
                    "ticker": tkr,
                    "close": px,
                    "volume": float(rng.uniform(1e5, 5e5)),
                    "ret_1m": ret * 5,
                    "dist_sma_50": ret,
                    "atr_norm": 0.03 + 0.01 * t_i,
                    "volume_ratio": 1.0 + 0.1 * t_i,
                    "rsi_14": 50 + t_i,
                    "volatility_20": 0.2,
                    "rsi_7": 50.0,
                    "rsi_21": 50.0,
                    "dist_sma_200": 0.0,
                    "volume_zscore": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_select_members_liquidity_and_cap():
    pool = ["AAA", "BBB", "CCC", "DDD"]
    snap = select_members(
        pool,
        pd.Timestamp("2020-06-01", tz="UTC"),
        config=L0Config(require_pit_listed=False, min_adv_usd=1e6, max_names=2),
        membership=None,
        adv_map={"AAA": 5e6, "BBB": 2e6, "CCC": 100.0, "DDD": 3e6},
        rank_key={"AAA": 5e6, "BBB": 2e6, "CCC": 100.0, "DDD": 3e6},
    )
    assert snap.n_pool == 4
    assert snap.n_after_liquidity == 3  # CCC dropped
    assert len(snap.members) == 2
    assert "CCC" not in snap.members
    assert snap.members[0] == "AAA"


def test_filter_liquidity_min_price():
    out = filter_liquidity(
        ["A", "B"],
        adv_map={"A": 1e7, "B": 1e7},
        min_adv_usd=0,
        min_price=10.0,
        price_map={"A": 5.0, "B": 20.0},
    )
    assert out == ["B"]


def test_estimate_adv_usd_causal_shape():
    panel = _synth_panel(40, 2)
    adv = estimate_adv_usd(panel, window=10)
    assert "adv_usd" in adv.columns
    assert len(adv) == len(panel)


def test_rebalance_dates_monthly():
    idx = pd.date_range("2020-01-01", periods=60, freq="B", tz="UTC")
    rb = rebalance_dates(idx, freq="M")
    assert len(rb) >= 2
    assert rb.is_monotonic_increasing


def test_membership_mask_panel():
    panel = _synth_panel(5, 3)
    d0 = panel["date"].iloc[0]
    members = {d0: ["T0", "T1"]}
    # only first date key — others False
    mask = membership_mask_panel(panel, members)
    assert mask.dtype == bool
    assert mask.sum() >= 1


def test_rule_rank_and_signals():
    panel = _synth_panel(10, 6)
    scored = score_panel_l1(panel, top_quantile=0.5)
    assert "l1_score" in scored.columns
    assert "l1_signal" in scored.columns
    # roughly half signal true per day
    frac = scored.groupby("date")["l1_signal"].mean().mean()
    assert 0.2 <= frac <= 0.9


def test_top_k_equal_weight_sums_to_one():
    w = top_k_equal_weight(
        ["A", "B", "C", "D"],
        [0.9, 0.1, 0.5, 0.8],
        k=3,
        max_weight=0.5,
    )
    assert len(w) == 3
    assert abs(sum(w.values()) - 1.0) < 1e-8
    assert "B" not in w  # lowest score among top logic — B is lowest overall


def test_top_k_score_weight():
    w = top_k_score_weight(["A", "B", "C"], [3.0, 1.0, 2.0], k=2, max_weight=0.8)
    assert set(w) == {"A", "C"}
    assert abs(sum(w.values()) - 1.0) < 1e-8


def test_build_weight_panel_and_returns():
    panel = _synth_panel(15, 4)
    scored = score_panel_l1(panel, top_quantile=0.5)
    scored["ret_1d"] = 0.01
    # max_weight 0.5 so 2-name EW can fully invest
    cfg = PortfolioL2Config(top_k=2, equal_weight=True, score_col="l1_score", max_weight=0.5)
    wp = build_weight_panel(scored, config=cfg)
    assert not wp.empty
    sums = wp["weight"].groupby(wp["date"]).sum()
    assert (sums <= 1.0 + 1e-9).all()
    assert (sums > 0).all()
    rets = portfolio_returns_from_weights(
        wp, scored[["date", "ticker", "ret_1d"]], ret_col="ret_1d"
    )
    assert len(rets) > 0
    eq = equity_from_returns(rets, start_equity=1000.0)
    assert eq.iloc[-1] > 0


def test_thin_ml_stub_fit_predict():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = (X @ np.array([0.5, -0.2, 0.1]) + 0.01).astype(float)
    stub = ThinMLStub(feature_names=["a", "b", "c"]).fit(X, y)
    pred = stub.predict(X)
    assert pred.shape == (100,)
    assert stub.fitted_


def test_residual_excess_series_and_meta():
    idx = pd.date_range("2020-01-01", periods=20, freq="B", tz="UTC")
    s = pd.Series(np.linspace(100, 120, 20), index=idx)
    b = pd.Series(np.linspace(100, 110, 20), index=idx)
    resid = residual_excess_series(s, b)
    assert len(resid) > 5
    ex, beat = beat_style_meta_frame(
        np.array([0.1, 0.0, -0.05]),
        np.array([0.0, 0.0, 0.0]),
    )
    assert beat.tolist() == [1.0, 0.0, 0.0]
    assert abs(ex[0] - 0.1) < 1e-9


def test_build_weights_for_date_respects_signal():
    day = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "l1_score": [1.0, 0.5, 0.9],
            "l1_signal": [True, False, True],
        }
    )
    w = build_weights_for_date(
        day,
        config=PortfolioL2Config(top_k=5, signal_col="l1_signal", max_weight=0.5),
    )
    assert "B" not in w
    assert abs(sum(w.values()) - 1.0) < 1e-8


def test_require_pit_without_membership_raises():
    with pytest.raises(ValueError, match="membership is None"):
        select_members(
            ["AAA", "BBB"],
            pd.Timestamp("2020-01-01", tz="UTC"),
            config=L0Config(require_pit_listed=True),
            membership=None,
        )


class _MockMem:
    def __init__(self, listed):
        self.listed = {str(x).upper() for x in listed}

    def is_listed(self, ticker: str, as_of) -> bool:
        return str(ticker).upper() in self.listed

    def members_as_of(self, as_of, *, tickers=None):
        pool = tickers or list(self.listed)
        return [t for t in pool if self.is_listed(t, as_of)]


def test_pit_membership_mock_filters():
    mem = _MockMem(["AAA", "CCC"])
    snap = select_members(
        ["AAA", "BBB", "CCC"],
        pd.Timestamp("2020-06-15", tz="UTC"),
        config=L0Config(require_pit_listed=True, max_names=10),
        membership=mem,
    )
    assert set(snap.members) == {"AAA", "CCC"}
    assert snap.n_after_pit == 2


def test_membership_mask_exact_tickers():
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02", tz="UTC")] * 3,
            "ticker": ["T0", "T1", "T2"],
        }
    )
    d0 = panel["date"].iloc[0]
    mask = membership_mask_panel(panel, {d0: ["T0", "T2"]})
    assert mask.tolist() == [True, False, True]


def test_max_weight_binds_and_leaves_cash():
    # 2 names, max_weight 0.3 → sum ≤ 0.6 (cash residual)
    w = top_k_equal_weight(["A", "B"], [1.0, 0.9], k=2, max_weight=0.3)
    assert abs(w["A"] - 0.3) < 1e-9
    assert abs(w["B"] - 0.3) < 1e-9
    assert abs(sum(w.values()) - 0.6) < 1e-9


def test_max_weight_score_weight_infeasible():
    w = top_k_score_weight(
        ["A", "B", "C"],
        [10.0, 1.0, 1.0],
        k=1,
        max_weight=0.4,
    )
    assert list(w.keys()) == ["A"]
    assert abs(w["A"] - 0.4) < 1e-9


def test_tickers_scores_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        top_k_equal_weight(["A", "B"], [1.0], k=2)


def test_empty_top_k():
    assert top_k_equal_weight([], [], k=5) == {}


def test_portfolio_returns_preserve_cash_no_renorm():
    """max_weight cash + NaN sleeve must not renorm remaining to 100%."""
    d = pd.Timestamp("2020-01-02", tz="UTC")
    wp = pd.DataFrame(
        {
            "date": [d, d],
            "ticker": ["A", "B"],
            "weight": [0.25, 0.25],  # sum=0.5 cash residual
        }
    )
    rets = pd.DataFrame(
        {
            "date": [d, d],
            "ticker": ["A", "B"],
            "ret_1d": [0.10, np.nan],
        }
    )
    pr = portfolio_returns_from_weights(wp, rets)
    # A contributes 0.25*0.10; B missing → 0; cash 0.5 → total 0.025 (NOT 0.10)
    assert len(pr) == 1
    assert abs(float(pr.iloc[0]) - 0.025) < 1e-9


def test_hold_weights_across_calendar_me():
    """ME snapshot held mid-month until next rebalance."""
    me = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-31", tz="UTC"),
                pd.Timestamp("2020-01-31", tz="UTC"),
                pd.Timestamp("2020-02-28", tz="UTC"),
            ],
            "ticker": ["A", "B", "A"],
            "weight": [0.3, 0.3, 0.5],
        }
    )
    cal = pd.date_range("2020-01-31", "2020-02-10", freq="B", tz="UTC")
    held = hold_weights_across_calendar(me, cal)
    mid = pd.Timestamp("2020-02-05", tz="UTC").normalize()
    day = held.loc[held["date"].dt.normalize() == mid]
    assert set(day["ticker"]) == {"A", "B"}
    assert abs(float(day.loc[day["ticker"] == "A", "weight"].iloc[0]) - 0.3) < 1e-9
    assert abs(day["weight"].sum() - 0.6) < 1e-9


def test_hold_weights_full_snapshot_turnover_no_name_leak():
    """A,B book → C,D book: old names must be 0; sum(w) never exceeds 1 after turnover."""
    me = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-31", tz="UTC"),
                pd.Timestamp("2020-01-31", tz="UTC"),
                pd.Timestamp("2020-02-28", tz="UTC"),
                pd.Timestamp("2020-02-28", tz="UTC"),
            ],
            "ticker": ["A", "B", "C", "D"],
            "weight": [0.4, 0.4, 0.35, 0.35],
        }
    )
    cal = pd.date_range("2020-01-31", "2020-03-05", freq="B", tz="UTC")
    held = hold_weights_across_calendar(me, cal)

    # Before Feb rebalance: only A,B
    pre = held.loc[held["date"] == pd.Timestamp("2020-02-14", tz="UTC").normalize()]
    assert set(pre["ticker"]) == {"A", "B"}
    assert abs(pre["weight"].sum() - 0.8) < 1e-9

    # On/after Feb rebalance: only C,D — A,B must not leak
    post = held.loc[held["date"] == pd.Timestamp("2020-03-02", tz="UTC").normalize()]
    assert set(post["ticker"]) == {"C", "D"}
    assert "A" not in set(post["ticker"]) and "B" not in set(post["ticker"])
    assert abs(post["weight"].sum() - 0.7) < 1e-9
    # Every day: sum(w) <= 1 + eps
    daily = held.groupby(held["date"].dt.normalize())["weight"].sum()
    assert (daily <= 1.0 + 1e-9).all()
    assert daily.max() <= 0.8 + 1e-9


def test_score_weight_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        top_k_score_weight(["A", "B"], [1.0], k=2)


def test_heterogeneous_returns_portfolio():
    panel = _synth_panel(8, 3)
    scored = score_panel_l1(panel, top_quantile=0.5)
    # heterogeneous next-day rets by ticker
    scored["ret_1d"] = scored["ticker"].map({"T0": 0.02, "T1": -0.01, "T2": 0.00})
    wp = build_weight_panel(
        scored, config=PortfolioL2Config(top_k=2, equal_weight=True, score_col="l1_score")
    )
    pr = portfolio_returns_from_weights(wp, scored[["date", "ticker", "ret_1d"]])
    assert len(pr) > 0
    # Not all zeros
    assert float(pr.abs().sum()) > 0
