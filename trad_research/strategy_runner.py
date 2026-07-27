"""Walk-forward runner for pluggable strategies (bake-off)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from trad_research.backtest import BacktestConfig, run_portfolio_backtest
from trad_research.features import list_tickers
from trad_research.metrics import acceptance_gates, equity_metrics
from trad_research.strategies import Strategy
from trad_research.regime import build_all_regime_maps
from trad_research.universe import build_scored_universe, select_high_vol
from trad_research.walk_forward import (
    _build_training_frame,
    _load_panels,
    load_benchmark_equity,
)

logger = logging.getLogger(__name__)


def _dynamic_highvol_tickers(
    data_root: Path,
    source_file: Path,
    as_of: pd.Timestamp,
    n: int,
    scan_limit: int,
) -> List[str]:
    """Causal high-vol re-rank: only bars/info ≤ as_of."""
    rows = build_scored_universe(
        data_root,
        source_file,
        as_of=as_of,
        limit_scan=scan_limit,
    )
    tickers = select_high_vol(rows, n=n)
    logger.info(
        "Dynamic highvol as_of=%s scored=%d selected=%d",
        as_of.date(),
        len(rows),
        len(tickers),
    )
    return tickers


def run_strategy_walk_forward(
    strategy: Strategy,
    *,
    data_root: Union[str, Path] = "data",
    ticker_file: Union[str, Path] = "good_tickers_wf80.txt",
    universe_limit: int = 80,
    first_oos_year: int = 2018,
    last_oos_year: int = 2025,
    base_bt: Optional[BacktestConfig] = None,
    k_tp: float = 2.5,
    k_sl: float = 1.5,
    label_horizon: int = 20,
    min_train_rows: int = 5000,
    preferred_index: Optional[Sequence[str]] = None,
    # Survivorship-free / point-in-time
    membership_path: Optional[Union[str, Path]] = None,
    use_pit_membership: bool = False,
    pit_equal_weight_benchmark: bool = False,
    pit_dv_weight_benchmark: bool = False,
    roll_on_delist: bool = False,
) -> Dict[str, Any]:
    data_root = Path(data_root)
    ticker_file = Path(ticker_file)
    if not ticker_file.is_file():
        ticker_file = Path("good_tickers_filtrados.txt")

    membership = None
    if use_pit_membership:
        from trad_research.pit_universe import MembershipIndex, DEFAULT_MEMBERSHIP_PATH

        mp = Path(membership_path) if membership_path else Path(DEFAULT_MEMBERSHIP_PATH)
        if not mp.is_file():
            mp = data_root / "pit" / "membership_index.json"
        if not mp.is_file():
            raise FileNotFoundError(
                f"PIT membership required but missing: {mp}. "
                "Run scripts/download_pit_universe.py first."
            )
        membership = MembershipIndex.load(mp)
        logger.info("PIT membership loaded n=%d from %s", len(membership), mp)

    dynamic_hv = bool(getattr(strategy, "dynamic_highvol", False))
    source_file = Path(
        getattr(strategy, "universe_source_file", None) or "good_tickers_filtrados.txt"
    )
    if not source_file.is_file():
        source_file = ticker_file
    scan_limit = int(getattr(strategy, "universe_scan_limit", 500))
    univ_n = int(getattr(strategy, "universe_n", universe_limit))

    # Static panels unless dynamic highvol rebuilds per year
    static_tickers = list_tickers(ticker_file, data_root, limit=universe_limit)
    with_fund = bool(getattr(strategy, "needs_fundamentals", False))
    panels = (
        {}
        if dynamic_hv
        else _load_panels(static_tickers, data_root, with_fundamentals=with_fund)
    )

    all_regimes = build_all_regime_maps(data_root, preferred_index=preferred_index)
    regime_key = getattr(strategy, "regime_filter", "legacy_sma50") or "legacy_sma50"
    if regime_key not in all_regimes:
        logger.warning("Unknown regime_filter=%s; using legacy_sma50", regime_key)
        regime_key = "legacy_sma50" if "legacy_sma50" in all_regimes else next(iter(all_regimes))
    regime_map, soft_regime, regime_desc = all_regimes[regime_key]
    logger.info("Regime filter: %s — %s", regime_key, regime_desc)
    if dynamic_hv:
        logger.info(
            "Dynamic highvol ON: source=%s n=%d scan=%d (as_of=year-start each OOS year)",
            source_file,
            univ_n,
            scan_limit,
        )

    bt0 = base_bt or BacktestConfig()
    overrides = strategy.backtest_overrides()
    bt_fields = {**bt0.__dict__, **overrides}
    capital = float(bt_fields.get("initial_capital", 100_000.0))

    year_results: List[Dict[str, Any]] = []
    all_trades: List[pd.DataFrame] = []
    equity_segments: List[pd.Series] = []
    positive_years = 0
    universe_by_year: Dict[int, List[str]] = {}

    for year in range(first_oos_year, last_oos_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        test_end = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        train_end = test_start

        if dynamic_hv:
            # Only information available at/before last day of prior year
            as_of = train_end - pd.Timedelta(days=1)
            year_tickers = _dynamic_highvol_tickers(
                data_root, source_file, as_of, n=univ_n, scan_limit=scan_limit
            )
            if len(year_tickers) < max(10, univ_n // 4):
                logger.warning("%s %s: dynamic highvol too small (%d)", strategy.name, year, len(year_tickers))
                continue
            universe_by_year[year] = year_tickers
            year_panels = _load_panels(year_tickers, data_root, with_fundamentals=with_fund)
        else:
            year_panels = panels
            universe_by_year[year] = static_tickers

        # Point-in-time: only names listed during the OOS year (and train uses same panels clipped)
        if membership is not None:
            from trad_research.pit_universe import filter_panels_pit

            # Train frame built from panels listed as of train_end (not future IPOs)
            train_members = set(membership.members_as_of(train_end - pd.Timedelta(days=1)))
            train_panels = {
                t: df
                for t, df in year_panels.items()
                if t.upper() in train_members or membership.is_listed(t, train_end - pd.Timedelta(days=1))
            }
            if len(train_panels) < 10:
                train_panels = year_panels  # fallback thin history
            year_panels = filter_panels_pit(year_panels, membership, test_start, test_end)
            universe_by_year[year] = list(year_panels.keys())
            logger.info(
                "%s %s: PIT members eval=%d train_pool=%d",
                strategy.name,
                year,
                len(year_panels),
                len(train_panels),
            )
        else:
            train_panels = year_panels

        if strategy.needs_training:
            train_df = _build_training_frame(
                train_panels,
                train_end=train_end,
                embargo_days=5,
                k_tp=k_tp,
                k_sl=k_sl,
                label_horizon=label_horizon,
            )
            if len(train_df) < min_train_rows:
                logger.warning("%s %s: skip train rows=%d", strategy.name, year, len(train_df))
                continue
            y = train_df["y_side"]
            hold = train_df[y == 1]
            nonhold = train_df[y != 1]
            if len(hold) > 2 * len(nonhold) and len(nonhold) > 1000:
                hold = hold.sample(n=min(len(hold), 2 * len(nonhold)), random_state=year)
                train_df = pd.concat([nonhold, hold], ignore_index=True)
            logger.info("%s %s: training...", strategy.name, year)
            strategy.train(train_df, year)

        bt = BacktestConfig(**{k: v for k, v in bt_fields.items() if k in BacktestConfig.__dataclass_fields__})
        bt.initial_capital = capital
        if bt.require_regime and regime_map:
            bt.regime_ok = regime_map
            bt.soft_regime_ok = soft_regime if soft_regime else None
        else:
            bt.regime_ok = None
            bt.soft_regime_ok = None
        if strategy.name == "defensive_trend" and soft_regime:
            bt.regime_ok = soft_regime
            bt.soft_regime_ok = soft_regime

        # Crash / oversold entry + win-rate overlays (causal index maps)
        if hasattr(strategy, "crash_entry_config"):
            try:
                from trad_research.crash_entry import build_crash_entry_map

                ccfg = strategy.crash_entry_config()
                if ccfg is not None and getattr(ccfg, "enabled", False):
                    cmap, cmeta = build_crash_entry_map(data_root, ccfg)
                    bt.crash_entry_on = cmap
                    bt.crash_entry_cfg = ccfg
                    bt.crash_relax_regime = bool(getattr(ccfg, "relax_regime", True))
                    logger.info(
                        "%s: crash_entry mode=%s indices=%s crash_days=%s",
                        strategy.name,
                        ccfg.mode,
                        cmeta.get("indices_used"),
                        cmeta.get("n_crash_days"),
                    )
            except Exception as e:
                logger.warning("%s: crash_entry map failed: %s", strategy.name, e)
        if hasattr(strategy, "winrate_filter_config"):
            try:
                wcfg = strategy.winrate_filter_config()
                if wcfg is not None:
                    bt.winrate_filter_cfg = wcfg
                    if int(getattr(wcfg, "hard_stop_cooldown_days", 0) or 0) > 0:
                        bt.hard_stop_cooldown_days = int(wcfg.hard_stop_cooldown_days)
                    if getattr(wcfg, "max_atr_pct_tight", None) is not None:
                        bt.max_atr_pct_entry = float(wcfg.max_atr_pct_tight)
            except Exception as e:
                logger.warning("%s: winrate filter failed: %s", strategy.name, e)

        # Sector ETF gate + rotation (strategy flags → BacktestConfig)
        if getattr(strategy, "require_sector_trend", False) or getattr(
            strategy, "enable_rotation", False
        ):
            from trad_research.sector_filter import (
                build_all_sector_etf_maps,
                load_ticker_sector_map,
            )

            if getattr(strategy, "require_sector_trend", False):
                map_path = Path(
                    getattr(strategy, "sector_map_path", None) or "data/ticker_sector_map.csv"
                )
                if not map_path.is_file():
                    map_path = data_root / "ticker_sector_map.csv"
                bt.require_sector_trend = True
                bt.ticker_sector = load_ticker_sector_map(map_path)
                bt.sector_etf_maps = build_all_sector_etf_maps(
                    data_root,
                    ma=int(getattr(strategy, "sector_ma", 50) or 50),
                    require_sma200=bool(getattr(strategy, "sector_require_sma200", False)),
                )
                bt.sector_allow_unmapped = bool(
                    getattr(strategy, "sector_allow_unmapped", True)
                )
                if not bt.sector_etf_maps:
                    logger.warning(
                        "%s: sector trend requested but no sector ETF histories — gate inactive",
                        strategy.name,
                    )
            if getattr(strategy, "enable_rotation", False):
                bt.enable_rotation = True
                bt.rotation_min_score_edge = float(
                    getattr(strategy, "rotation_min_score_edge", 0.05) or 0.05
                )
                bt.rotation_min_bars = int(getattr(strategy, "rotation_min_bars", 3) or 3)
                bt.rotation_max_per_day = int(
                    getattr(strategy, "rotation_max_per_day", 2) or 2
                )

        # Delisting residual → cash (+ optional ISIN successor roll)
        if membership is not None:
            from trad_research.pit_universe import attach_delist_dates_to_config

            bt.delist_dates = attach_delist_dates_to_config(
                membership, list(year_panels.keys())
            )
            if roll_on_delist:
                succ = {}
                for t in year_panels.keys():
                    s = membership.successor_after_delist(t)
                    if s:
                        succ[t.upper()] = s
                bt.delist_successors = succ
                bt.roll_on_delist = True

        trades, equity, _ = run_portfolio_backtest(
            year_panels, strategy, bt, start=test_start, end=test_end
        )
        if equity.empty:
            continue
        yrep = equity_metrics(equity, start_equity=capital, trades=trades)
        year_results.append(
            {
                "year": year,
                "n_trades": yrep.n_trades,
                "year_return": yrep.final_equity / capital - 1.0,
                "sharpe": yrep.sharpe,
                "max_drawdown": yrep.max_drawdown,
                "win_rate": yrep.win_rate,
                "end_equity": yrep.final_equity,
                "n_universe": len(year_panels),
            }
        )
        if year_results[-1]["year_return"] > 0:
            positive_years += 1
        if not trades.empty:
            t = trades.copy()
            t["oos_year"] = year
            all_trades.append(t)
        equity_segments.append(equity)
        capital = float(equity.iloc[-1])
        logger.info(
            "%s %s: ret=%.1f%% trades=%d sharpe=%.2f univ=%d",
            strategy.name,
            year,
            year_results[-1]["year_return"] * 100,
            yrep.n_trades,
            yrep.sharpe,
            len(year_panels),
        )

    if not equity_segments:
        raise RuntimeError(f"No OOS equity for strategy {strategy.name}")

    full_equity = pd.concat(equity_segments)
    full_equity = full_equity[~full_equity.index.duplicated(keep="last")].sort_index()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    start_eq = float(bt_fields.get("initial_capital", 100_000.0))
    bench = load_benchmark_equity(
        data_root,
        full_equity.index.min(),
        full_equity.index.max(),
        preferred=preferred_index,
    )
    pit_bench = None
    pit_bench_kind = None
    if membership is not None and panels and (
        pit_equal_weight_benchmark or pit_dv_weight_benchmark
    ):
        from trad_research.pit_universe import (
            build_dollar_volume_weight_benchmark,
            build_equal_weight_benchmark,
        )

        # Prefer DVW when both requested (cap-weight proxy); else EW
        if pit_dv_weight_benchmark:
            pit_bench = build_dollar_volume_weight_benchmark(
                panels,
                membership,
                full_equity.index.min(),
                full_equity.index.max(),
            )
            pit_bench_kind = "pit_dv_weight"
        if (pit_bench is None or pit_bench.empty) and pit_equal_weight_benchmark:
            pit_bench = build_equal_weight_benchmark(
                panels,
                membership,
                full_equity.index.min(),
                full_equity.index.max(),
            )
            pit_bench_kind = "pit_equal_weight"
        if pit_bench is not None and not pit_bench.empty:
            pit_bench = pit_bench / float(pit_bench.iloc[0]) * start_eq
            bench = pit_bench
            logger.info("Using %s benchmark for metrics", pit_bench_kind)
    report = equity_metrics(
        full_equity,
        start_equity=start_eq,
        trades=trades_df,
        benchmark=bench,
        positive_year_frac=positive_years / max(len(year_results), 1),
    )
    gates = acceptance_gates(report)
    gates["years_ok"] = len(year_results) >= 6
    required = {k: v for k, v in gates.items() if not str(k).startswith("stretch_")}
    n_delist_exits = 0
    n_ma_rolls = 0
    if not trades_df.empty and "exit_reason" in trades_df.columns:
        n_delist_exits = int(
            trades_df["exit_reason"].isin(["delisting", "delisting_no_bar"]).sum()
        )
        n_ma_rolls = int((trades_df["exit_reason"] == "ma_roll_open").sum()) if "exit_reason" in trades_df.columns else 0
    return {
        "strategy": strategy.name,
        "description": strategy.description,
        "report": report,
        "gates": gates,
        "passed": all(bool(v) for v in required.values()),
        "year_results": year_results,
        "n_tickers": len(static_tickers) if not dynamic_hv else univ_n,
        "trades": trades_df,
        "equity": full_equity,
        "dynamic_highvol": dynamic_hv,
        "universe_by_year": {str(k): v for k, v in universe_by_year.items()},
        "use_pit_membership": bool(membership is not None),
        "pit_equal_weight_benchmark": bool(
            pit_bench_kind == "pit_equal_weight" and pit_bench is not None
        ),
        "pit_dv_weight_benchmark": bool(
            pit_bench_kind == "pit_dv_weight" and pit_bench is not None
        ),
        "n_delist_exits": n_delist_exits,
        "n_ma_rolls": n_ma_rolls,
        "benchmark": bench,
    }
