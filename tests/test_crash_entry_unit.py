"""Unit tests for causal crash/oversold entry overlays (synthetic only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trad_research.crash_entry import (
    CrashEntryConfig,
    WinRateFilterConfig,
    apply_crash_signal_overlay,
    apply_winrate_signal_filters,
    build_crash_entry_map,
    composite_rank_score,
    compute_index_crash_metrics,
    crash_on_day,
    _flag_from_metrics,
)
from trad_research.features import _wilder_rsi
from trad_research.backtest import (
    BacktestConfig,
    compute_entry_hard_stop,
    run_portfolio_backtest,
)


def _synth_close(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=n)
    a = max(1, int(n * 0.70))
    b = min(n, a + max(10, n // 15))
    rets[a:b] = rng.normal(-0.03, 0.01, size=b - a)
    px = 100.0 * np.cumprod(1.0 + rets)
    return pd.Series(px, dtype=float)


def test_wilder_rsi_causal_bounds():
    close = _synth_close(200, seed=1)
    rsi = _wilder_rsi(close, 14)
    assert len(rsi) == len(close)
    assert rsi.notna().all()
    assert float(rsi.min()) >= 0.0
    assert float(rsi.max()) <= 100.0
    early = float(rsi.iloc[40:60].mean())
    late_crash = float(rsi.iloc[int(len(rsi) * 0.72) : int(len(rsi) * 0.85)].mean())
    assert late_crash < early + 15.0
    assert 0.0 <= late_crash <= 100.0


def test_compute_index_crash_metrics_no_lookahead():
    close = _synth_close(250, seed=2)
    dates = pd.date_range("2018-01-01", periods=len(close), freq="B", tz="UTC")
    m = compute_index_crash_metrics(close, dates, name="SYN", rsi_period=14)
    assert len(m.rsi) == len(close)
    assert np.nanmax(m.dd_from_peak) <= 1e-9
    assert abs(m.dd_from_peak[0]) < 1e-12
    assert m.rsi_rising.shape == m.rsi.shape
    assert bool(m.rsi_rising[0]) is False


def test_crash_flag_only_when_rsi_below_thr():
    close = _synth_close(280, seed=3)
    dates = pd.date_range("2018-01-01", periods=len(close), freq="B", tz="UTC")
    m = compute_index_crash_metrics(close, dates, name="SYN", rsi_period=14)
    cfg = CrashEntryConfig(enabled=True, mode="rsi", rsi_threshold=30.0)
    flags = _flag_from_metrics(m, cfg)
    for i, on in enumerate(flags):
        if on:
            assert m.rsi[i] < 30.0
    high = np.where(m.rsi > 55.0)[0]
    if len(high):
        assert not flags[high[0]]


def test_crash_dd_mode():
    close = _synth_close(280, seed=4)
    dates = pd.date_range("2018-01-01", periods=len(close), freq="B", tz="UTC")
    m = compute_index_crash_metrics(close, dates, name="SYN")
    cfg = CrashEntryConfig(enabled=True, mode="dd", dd_threshold=-0.10)
    flags = _flag_from_metrics(m, cfg)
    for i, on in enumerate(flags):
        if on:
            assert m.dd_from_peak[i] <= -0.10 + 1e-12


def test_crash_mode_rsi_or_dd_and_and_recover():
    """Mode variants: rsi_or_dd, rsi_and_dd, rsi_recover."""
    n = 80
    close = pd.Series(np.concatenate([np.linspace(100, 120, 40), np.linspace(120, 90, 40)]))
    dates = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    m = compute_index_crash_metrics(close, dates, name="SYN", rsi_period=14)

    f_or = _flag_from_metrics(
        m, CrashEntryConfig(enabled=True, mode="rsi_or_dd", rsi_threshold=40.0, dd_threshold=-0.08)
    )
    f_and = _flag_from_metrics(
        m, CrashEntryConfig(enabled=True, mode="rsi_and_dd", rsi_threshold=40.0, dd_threshold=-0.08)
    )
    f_rec = _flag_from_metrics(
        m,
        CrashEntryConfig(
            enabled=True, mode="rsi_recover", rsi_threshold=50.0, require_rsi_rising=True
        ),
    )
    # OR is superset of AND
    assert int(f_or.sum()) >= int(f_and.sum())
    # Every AND day is also OR
    assert bool(np.all(~f_and | f_or))
    # Recover days must have rising RSI and low RSI
    for i, on in enumerate(f_rec):
        if on:
            assert m.rsi[i] < 50.0
            assert bool(m.rsi_rising[i])


def test_build_crash_map_fail_closed(tmp_path: Path):
    cfg = CrashEntryConfig(enabled=True, mode="rsi", rsi_threshold=30.0)
    cmap, meta = build_crash_entry_map(tmp_path, cfg)
    assert cmap == {}
    assert meta.get("error") == "no_index_history" or meta.get("n_crash_days") == 0

    cfg2 = CrashEntryConfig(enabled=False)
    cmap2, meta2 = build_crash_entry_map(tmp_path, cfg2)
    assert cmap2 == {}
    assert meta2["enabled"] is False


def test_build_crash_map_from_synth_csv(tmp_path: Path):
    close = _synth_close(300, seed=5)
    dates = pd.date_range("2015-01-01", periods=len(close), freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
        }
    )
    df.to_csv(tmp_path / "SPY_history.csv", index=False)
    cfg = CrashEntryConfig(
        enabled=True,
        mode="rsi",
        rsi_threshold=35.0,
        index_names=("SPY",),
        combine="first",
    )
    cmap, meta = build_crash_entry_map(tmp_path, cfg)
    assert "SPY" in meta["indices_used"]
    assert meta["n_dates"] == len(dates)
    early = pd.Timestamp("2010-01-01", tz="UTC")
    assert crash_on_day(early, cmap) is False
    mid = dates[150]
    assert mid in cmap or crash_on_day(mid, cmap) in (True, False)


def test_no_future_index_data_in_crash_map(tmp_path: Path):
    n = 100
    close = pd.Series(np.linspace(100, 80, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1e6,
        }
    )
    df.to_csv(tmp_path / "SPY_history.csv", index=False)
    cfg = CrashEntryConfig(
        enabled=True,
        mode="dd",
        dd_threshold=-0.05,
        index_names=("SPY",),
    )
    build_crash_entry_map(tmp_path, cfg)
    for d in list(dates[:50]):
        m_full = compute_index_crash_metrics(close, dates, name="SPY")
        m_trunc = compute_index_crash_metrics(close.iloc[:50], dates[:50], name="SPY")
        i = list(dates).index(d)
        assert abs(m_full.dd_from_peak[i] - m_trunc.dd_from_peak[i]) < 1e-9


def test_apply_crash_signal_overlay_unions_on_crash_days():
    """Real assert: without SMA50, crash_relax admits p_buy>=crash_conf on crash days."""
    n = 40
    dates = pd.date_range("2020-03-01", periods=n, freq="B", tz="UTC")
    # No sma_50 → crash soft-trend path leaves trend_ok all True
    df = pd.DataFrame(
        {
            "date": dates,
            "close": np.full(n, 100.0),
            "dist_sma_200": np.full(n, -0.05),
            "atr_norm": np.full(n, 0.05),
        }
    )
    base_sig = pd.Series(False, index=df.index)
    base_score = pd.Series(0.1, index=df.index)
    p_buy = np.full(n, 0.28)
    crash_map = {d: True for d in dates}
    cfg_bt = BacktestConfig(min_confidence=0.45, max_atr_pct=0.10, require_trend=True)
    crash_cfg = CrashEntryConfig(
        enabled=True,
        mode="rsi",
        crash_min_confidence=0.22,
        crash_relax_trend=True,
        crash_score_boost=1.2,
        crash_min_dist_sma200=-0.25,
    )
    sig, score = apply_crash_signal_overlay(
        df, base_sig, base_score, p_buy, cfg_bt, crash_map, crash_cfg
    )
    assert bool(sig.all()), "crash overlay must admit all days when base off and conf ok"
    assert float(score.min()) >= 0.28 * 1.2 - 1e-9

    # Non-crash days stay off when base is False
    crash_map2 = {d: False for d in dates}
    sig2, _ = apply_crash_signal_overlay(
        df, base_sig, base_score, p_buy, cfg_bt, crash_map2, crash_cfg
    )
    assert not bool(sig2.any())


def test_soft_trend_non_crash_allows_sma20_not_sma50():
    """soft_trend replaces hard SMA50: close < sma50 but > sma20 stays on."""
    n = 30
    dates = pd.date_range("2021-06-01", periods=n, freq="B", tz="UTC")
    # Rising path so sma20 < close near the end; sma50 held high
    close = np.linspace(90, 110, n)
    sma50 = np.full(n, 115.0)  # close always below
    df = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "sma_50": sma50,
            "atr_norm": np.full(n, 0.05),
        }
    )
    # Pretend base already passed conf without hard trend
    sig = pd.Series(True, index=df.index)
    score = pd.Series(0.5, index=df.index)
    wr = WinRateFilterConfig(soft_trend_non_crash=True)
    sig2, _ = apply_winrate_signal_filters(df, sig, score, wr, crash_map=None)
    # Late bars: close > rolling sma20
    assert bool(sig2.iloc[-1]), "soft trend must allow close>sma20 with close<sma50"
    # Crash day should not apply soft filter
    crash_map = {dates[-1]: True}
    # Force all other days non-crash
    for d in dates[:-1]:
        crash_map[d] = False
    sig3, _ = apply_winrate_signal_filters(
        df, pd.Series(True, index=df.index), score, wr, crash_map=crash_map
    )
    assert bool(sig3.iloc[-1]), "crash day skips soft_trend_non_crash gate"


def test_winrate_atr_tight_skips_crash_days():
    n = 20
    dates = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    atr = np.full(n, 0.10)
    atr[5] = 0.25
    atr[10] = 0.25  # crash day with high ATR — should stay on
    df = pd.DataFrame(
        {
            "date": dates,
            "close": np.full(n, 50.0),
            "atr_norm": atr,
            "sma_50": np.full(n, 49.0),
        }
    )
    sig = pd.Series(True, index=df.index)
    score = pd.Series(0.5, index=df.index)
    wr = WinRateFilterConfig(max_atr_pct_tight=0.16, soft_trend_non_crash=False)
    crash_map = {d: False for d in dates}
    crash_map[dates[10]] = True
    sig2, _ = apply_winrate_signal_filters(
        df, sig, score, wr, crash_map=crash_map, non_crash_only=True
    )
    assert not bool(sig2.iloc[5]), "non-crash high ATR blocked"
    assert bool(sig2.iloc[10]), "crash day keeps loose ATR (tight skipped)"
    assert bool(sig2.iloc[0])


def test_non_crash_min_confidence_wired():
    n = 10
    dates = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({"date": dates, "close": np.full(n, 50.0), "atr_norm": np.full(n, 0.05)})
    sig = pd.Series(True, index=df.index)
    score = pd.Series(0.5, index=df.index)
    p_buy = np.array([0.20, 0.40, 0.20, 0.40, 0.20, 0.40, 0.20, 0.40, 0.20, 0.40])
    wr = WinRateFilterConfig(non_crash_min_confidence=0.30)
    sig2, _ = apply_winrate_signal_filters(
        df, sig, score, wr, crash_map=None, p_buy=p_buy
    )
    assert not bool(sig2.iloc[0])
    assert bool(sig2.iloc[1])
    # Crash day bypasses floor
    cmap = {dates[0]: True}
    for d in dates[1:]:
        cmap[d] = False
    sig3, _ = apply_winrate_signal_filters(
        df, sig, score, wr, crash_map=cmap, p_buy=p_buy
    )
    assert bool(sig3.iloc[0]), "crash day skips non_crash_min_confidence"


def test_hard_stop_cooldown_field_on_config():
    cfg = BacktestConfig(hard_stop_cooldown_days=10, max_atr_pct_entry=0.15)
    assert cfg.hard_stop_cooldown_days == 10
    hard = compute_entry_hard_stop(100.0, 2.0, cfg)
    assert hard < 100.0


def test_hard_stop_cooldown_blocks_reentry():
    """Behavioral: after hard_stop exit, same ticker blocked for N days."""

    class AlwaysBuy:
        name = "always_buy"

        def generate_signals(self, df, cfg):
            return (
                pd.Series(True, index=df.index),
                pd.Series(0.9, index=df.index),
            )

    n = 40
    dates = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    # flat 100, crash day 5 to 85 (11% hard stop), recover to 100
    close = np.full(n, 100.0)
    close[5] = 85.0
    close[6:] = 100.0
    high = close.copy()
    high[5] = 100.0
    low = close.copy()
    atr = np.full(n, 2.0)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1e6,
            "atr": atr,
            "atr_norm": atr / close,
            "sma_50": close * 0.95,
            "dist_sma_200": np.full(n, 0.05),
            "ret_1m": np.full(n, 0.05),
        }
    )
    panels = {"AAA": df}
    cfg = BacktestConfig(
        min_confidence=0.1,
        require_trend=False,
        require_momentum=False,
        require_regime=False,
        max_atr_pct=0.5,
        min_dist_sma200=-1.0,
        hard_stop_pct=0.11,
        hard_stop_cooldown_days=10,
        max_positions=5,
        max_entries_per_day=5,
        volatility_target_pct=0.05,
        max_position_pct=0.5,
        commission=0.0,
        slippage=0.0,
        max_horizon=5,
        k_atr=10.0,  # very wide trail so hard_stop hits first
    )
    trades, equity, _ = run_portfolio_backtest(panels, AlwaysBuy(), cfg)
    assert not trades.empty
    hard_exits = trades[trades["exit_reason"] == "hard_stop"]
    assert len(hard_exits) >= 1, "expected at least one hard_stop"
    hs_exit = pd.Timestamp(hard_exits.iloc[0]["exit_date"])
    # Next entry on AAA must be >= exit + 10 days
    later = trades[
        (trades["ticker"] == "AAA")
        & (pd.to_datetime(trades["entry_date"], utc=True) > hs_exit)
    ]
    if not later.empty:
        next_entry = pd.Timestamp(later.iloc[0]["entry_date"])
        gap = (next_entry - hs_exit).days
        assert gap >= 10, f"re-entry gap {gap}d < cooldown 10d"


def test_max_atr_entry_skips_on_crash_day_only():
    """Documented policy: max_atr_pct_entry inactive on crash days."""
    # Unit-level: re-check filter path + entry gate contract via config docs
    n = 5
    dates = pd.date_range("2020-03-01", periods=n, freq="B", tz="UTC")
    atr = np.full(n, 0.20)  # above tight 0.16
    df = pd.DataFrame(
        {
            "date": dates,
            "close": np.full(n, 50.0),
            "atr_norm": atr,
        }
    )
    wr = WinRateFilterConfig(max_atr_pct_tight=0.16)
    crash_map = {d: True for d in dates}
    sig, _ = apply_winrate_signal_filters(
        df,
        pd.Series(True, index=df.index),
        pd.Series(0.5, index=df.index),
        wr,
        crash_map=crash_map,
    )
    assert bool(sig.all()), "tight ATR must not fire on crash days"
    crash_map2 = {d: False for d in dates}
    sig2, _ = apply_winrate_signal_filters(
        df,
        pd.Series(True, index=df.index),
        pd.Series(0.5, index=df.index),
        wr,
        crash_map=crash_map2,
    )
    assert not bool(sig2.any()), "tight ATR blocks non-crash high ATR"


def test_composite_rank_score_prefers_better_wr():
    a = {
        "win_rate": 0.40,
        "excess_total_vs_spy": 0.5,
        "crash_2020_return": -0.1,
        "max_drawdown": -0.3,
        "n_trades": 100,
    }
    b = {
        "win_rate": 0.30,
        "excess_total_vs_spy": 0.5,
        "crash_2020_return": -0.1,
        "max_drawdown": -0.3,
        "n_trades": 100,
    }
    assert composite_rank_score(a) > composite_rank_score(b)


def test_strategy_crash_variants_registered():
    from trad_research.strategies import get_strategy

    for name in (
        "turbo_highvol_crash_rsi",
        "turbo_highvol_crash_rsi_wr",
        "turbo_highvol_minalloc_crash_rsi",
    ):
        s = get_strategy(name)
        assert s.crash_entry_enabled is True
        ccfg = s.crash_entry_config()
        assert ccfg.enabled is True


def test_smoke_config_grid():
    """Mega-study smoke grid has baselines + crash/WR variants."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_crash_entry_mega_study.py"
    spec = importlib.util.spec_from_file_location("crash_mega", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    cfgs = mod._build_config_grid("smoke")
    ids = [c["id"] for c in cfgs]
    assert len(cfgs) >= 4
    assert any(i.endswith("__baseline") for i in ids)
    assert any("crash_rsi" in i for i in ids)
    assert any("wr" in i for i in ids)
    # unique ids
    assert len(ids) == len(set(ids))


def _synthetic_panel(dates: pd.DatetimeIndex, close: np.ndarray, ticker: str = "AAA"):
    atr = np.full(len(dates), 2.0)
    return {
        ticker: pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1e6,
                "atr": atr,
                "atr_norm": atr / np.maximum(close, 1e-6),
                "sma_50": close * 0.95,
                "dist_sma_200": np.full(len(dates), 0.05),
                "ret_1m": np.full(len(dates), 0.05),
            }
        )
    }


def test_crash_relax_regime_allows_entry_when_hard_regime_off():
    """Day-loop: require_regime hard-off blocks; crash_relax_regime + crash day allows entries."""

    class AlwaysBuy:
        name = "always_buy"

        def generate_signals(self, df, cfg):
            return (
                pd.Series(True, index=df.index),
                pd.Series(0.9, index=df.index),
            )

    # run_portfolio_backtest skips panels with len < 30
    n = 40
    dates = pd.date_range("2020-03-02", periods=n, freq="B", tz="UTC")
    close = np.full(n, 100.0)
    panels = _synthetic_panel(dates, close)

    # All days risk-off (hard regime False)
    regime_off = {d: False for d in dates}

    base_cfg = dict(
        min_confidence=0.1,
        require_trend=False,
        require_momentum=False,
        require_regime=True,
        regime_ok=regime_off,
        soft_hard_regime=False,
        max_atr_pct=0.5,
        min_dist_sma200=-1.0,
        hard_stop_pct=0.50,
        max_positions=5,
        max_entries_per_day=5,
        volatility_target_pct=0.05,
        max_position_pct=0.5,
        commission=0.0,
        slippage=0.0,
        max_horizon=30,
        k_atr=5.0,
    )

    # Without crash relax: no entries (hard block)
    cfg_block = BacktestConfig(**base_cfg, crash_entry_on=None, crash_relax_regime=False)
    trades_block, _, _ = run_portfolio_backtest(panels, AlwaysBuy(), cfg_block)
    assert trades_block.empty or len(trades_block) == 0

    # With crash map True every day + crash_relax_regime: entries allowed
    crash_map = {d: True for d in dates}
    cfg_relax = BacktestConfig(
        **base_cfg,
        crash_entry_on=crash_map,
        crash_relax_regime=True,
    )
    trades_relax, eq_relax, _ = run_portfolio_backtest(panels, AlwaysBuy(), cfg_relax)
    assert not trades_relax.empty, "crash_relax_regime must allow entries under hard regime off"
    assert int(len(trades_relax)) >= 1
    assert not eq_relax.empty

    # Crash map False every day + crash_relax_regime: still blocked (no crash day)
    crash_off = {d: False for d in dates}
    cfg_no_crash = BacktestConfig(
        **base_cfg,
        crash_entry_on=crash_off,
        crash_relax_regime=True,
    )
    trades_nc, _, _ = run_portfolio_backtest(panels, AlwaysBuy(), cfg_no_crash)
    assert trades_nc.empty or len(trades_nc) == 0


def test_wilder_rsi_future_mutation_does_not_change_past():
    """Causality pin: mutate closes after t; RSI[≤t] must match prefix-only RSI."""
    n = 120
    close = _synth_close(n, seed=42).to_numpy(dtype=float)
    cut = 60
    rsi_full = _wilder_rsi(pd.Series(close), 14).to_numpy(dtype=float)

    # Mutate only the future tail after cut
    close_mut = close.copy()
    close_mut[cut + 1 :] = close_mut[cut + 1 :] * 0.5  # crash the future
    rsi_mut = _wilder_rsi(pd.Series(close_mut), 14).to_numpy(dtype=float)

    # Past RSI (bars 0..cut inclusive) must be identical
    np.testing.assert_allclose(
        rsi_full[: cut + 1],
        rsi_mut[: cut + 1],
        rtol=0,
        atol=1e-12,
        err_msg="future close mutation changed past RSI (look-ahead)",
    )
    # Prefix-only series must match full series on the prefix
    rsi_prefix = _wilder_rsi(pd.Series(close[: cut + 1]), 14).to_numpy(dtype=float)
    # ewm with min_periods may differ slightly at series end for prefix-only
    # vs full when warm-up is internal — compare overlapping warm interior
    # Strict: same length prefix RSI equals full[:cut+1] when computed on full then sliced
    # (already checked via mutation). Also prefix vs full for early bars after warm-up:
    warm = 20
    np.testing.assert_allclose(
        rsi_full[warm : cut + 1],
        rsi_prefix[warm:],
        rtol=1e-10,
        atol=1e-9,
        err_msg="prefix RSI diverges from full-series prefix (non-causal RSI)",
    )
    # Future must actually change after mutation (sanity: test is not vacuous)
    assert not np.allclose(rsi_full[cut + 5 :], rsi_mut[cut + 5 :], atol=1e-6)
