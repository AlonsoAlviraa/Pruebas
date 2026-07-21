"""VAL-03: Frozen US → foreign market transfer validation.

FROZEN_US_TRANSFER: fit models only on US panels; backtest on foreign panels.
LOCAL_WF (legacy bake-off) is pipeline stress only — never sets product_mode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from trad_research.backtest import BacktestConfig, run_portfolio_backtest
from trad_research.features import list_tickers
from trad_research.metrics import PerformanceReport, equity_metrics
from trad_research.policies import DeploymentPolicy
from trad_research.regime import build_all_regime_maps
from trad_research.strategies import Strategy
from trad_research.walk_forward import (
    _build_training_frame,
    _load_panels,
    load_benchmark_equity,
)

logger = logging.getLogger(__name__)

PRODUCT_US_ONLY = "US_ONLY"
PRODUCT_TRANSFER_CANDIDATE = "TRANSFER_CANDIDATE"
PRODUCT_MULTI_REGION = "MULTI_REGION_PORTABLE"


@dataclass
class TransferConfig:
    train_data_root: Path = Path("data")
    train_ticker_file: Path = Path("good_tickers_wf80.txt")
    eval_data_root: Path = Path("data_es")
    eval_ticker_file: Path = Path("spain_wf_universe.txt")
    preferred_index: tuple[str, ...] = ("IBEX",)
    first_oos_year: int = 2018
    last_oos_year: int = 2025
    universe_limit_train: int = 80
    universe_limit_eval: int = 80
    k_tp: float = 2.5
    k_sl: float = 1.5
    label_horizon: int = 20
    min_train_rows: int = 5000
    model_cache_dir: Optional[Path] = None
    foreign_suffix_denylist: tuple[str, ...] = (".MC",)
    market_id: str = "ES"


def transfer_acceptance_gates(
    report: PerformanceReport,
    *,
    min_years: float = 6.0,
) -> Dict[str, bool]:
    """Primary transfer gates (honest vs US research bars).

    BH missing → no_collapse_cagr fails (no silent pass).
    """
    bh_cagr = report.benchmark_cagr
    bh_ok = bh_cagr is not None
    if bh_ok:
        no_collapse = report.cagr >= (float(bh_cagr) - 0.05)
    else:
        no_collapse = False
        logger.error("transfer gates: benchmark BH missing → no_collapse_cagr FAIL")

    pos_frac = report.positive_year_frac
    consistency = pos_frac is not None and pos_frac >= 0.50

    return {
        "years_ok": report.years >= min_years * 0.9,
        "no_collapse_cagr": no_collapse,
        "sharpe_nonneg": report.sharpe >= 0.0,
        "mdd_bound": report.max_drawdown >= -0.50,
        "consistency": consistency,
        "bh_present": bh_ok,
        # stretch
        "stretch_sharpe_0_30": report.sharpe >= 0.30,
        "stretch_mdd_40": report.max_drawdown >= -0.40,
        "stretch_consistency_60": pos_frac is not None and pos_frac >= 0.60,
    }


def transfer_primary_passed(gates: Dict[str, bool]) -> bool:
    required = (
        "years_ok",
        "no_collapse_cagr",
        "sharpe_nonneg",
        "mdd_bound",
        "consistency",
        "bh_present",
    )
    return all(bool(gates.get(k)) for k in required)


def compute_product_mode(
    *,
    us_home_pass: bool,
    foreign_results: Sequence[Dict[str, Any]],
    user_promotion: bool = False,
) -> str:
    """Outcome label — never stored on DeploymentPolicy.

    TRANSFER_CANDIDATE: US home PASS + exactly 1 foreign FROZEN primary pass
    MULTI_REGION_PORTABLE: US home PASS + (≥2 foreign pass OR user promotion)
    else US_ONLY
    """
    n_pass = sum(1 for r in foreign_results if r.get("transfer_passed"))
    if us_home_pass and user_promotion:
        return PRODUCT_MULTI_REGION
    if us_home_pass and n_pass >= 2:
        return PRODUCT_MULTI_REGION
    if us_home_pass and n_pass == 1:
        return PRODUCT_TRANSFER_CANDIDATE
    return PRODUCT_US_ONLY


def _assert_isolation(
    train_root: Path,
    eval_root: Path,
    train_tickers: Sequence[str],
    eval_tickers: Sequence[str],
    suffix_denylist: Sequence[str],
) -> None:
    tr = train_root.resolve()
    er = eval_root.resolve()
    if tr == er:
        raise ValueError(
            f"FROZEN_US_TRANSFER requires train_data_root != eval_data_root (got {tr})"
        )
    overlap = set(train_tickers) & set(eval_tickers)
    if overlap:
        raise ValueError(f"Train/eval ticker overlap forbidden: {sorted(overlap)[:10]}")
    for t in train_tickers:
        for suf in suffix_denylist:
            if str(t).upper().endswith(suf.upper()):
                raise ValueError(
                    f"Train ticker {t} matches foreign denylist suffix {suf} — isolation fail"
                )


def run_frozen_us_transfer(
    strategy: Strategy,
    *,
    train_data_root: Union[str, Path] = "data",
    train_ticker_file: Union[str, Path] = "good_tickers_wf80.txt",
    eval_data_root: Union[str, Path] = "data_es",
    eval_ticker_file: Union[str, Path] = "spain_wf_universe.txt",
    preferred_index: Sequence[str] = ("IBEX",),
    first_oos_year: int = 2018,
    last_oos_year: int = 2025,
    universe_limit_train: int = 80,
    universe_limit_eval: int = 80,
    base_bt: Optional[BacktestConfig] = None,
    policy: Optional[DeploymentPolicy] = None,
    k_tp: float = 2.5,
    k_sl: float = 1.5,
    label_horizon: int = 20,
    min_train_rows: int = 5000,
    model_cache_dir: Optional[Path] = None,
    foreign_suffix_denylist: Sequence[str] = (".MC",),
    market_id: str = "ES",
    us_home_pass: Optional[bool] = None,
) -> Dict[str, Any]:
    """Train on US panels only; evaluate on foreign panels (zero foreign bars in fit)."""
    train_data_root = Path(train_data_root)
    eval_data_root = Path(eval_data_root)
    train_ticker_file = Path(train_ticker_file)
    eval_ticker_file = Path(eval_ticker_file)
    pref = tuple(preferred_index)

    if not train_ticker_file.is_file():
        train_ticker_file = Path("good_tickers_filtrados.txt")
    if not eval_ticker_file.is_file():
        raise FileNotFoundError(f"eval ticker file missing: {eval_ticker_file}")

    train_tickers = list_tickers(train_ticker_file, train_data_root, limit=universe_limit_train)
    eval_tickers = list_tickers(eval_ticker_file, eval_data_root, limit=universe_limit_eval)
    _assert_isolation(
        train_data_root,
        eval_data_root,
        train_tickers,
        eval_tickers,
        foreign_suffix_denylist,
    )

    with_fund = bool(getattr(strategy, "needs_fundamentals", False))
    train_panels = _load_panels(train_tickers, train_data_root, with_fundamentals=with_fund)
    eval_panels = _load_panels(eval_tickers, eval_data_root, with_fundamentals=with_fund)
    if not train_panels:
        raise RuntimeError(f"No train panels under {train_data_root}")
    if not eval_panels:
        raise RuntimeError(f"No eval panels under {eval_data_root}")

    # Regime + index from FOREIGN root
    all_regimes = build_all_regime_maps(eval_data_root, preferred_index=pref)
    regime_key = getattr(strategy, "regime_filter", "legacy_sma50") or "legacy_sma50"
    if policy is not None and policy.regime_filter:
        regime_key = policy.regime_filter
    if regime_key not in all_regimes:
        logger.warning("Unknown regime_filter=%s; using legacy_sma50", regime_key)
        regime_key = "legacy_sma50" if "legacy_sma50" in all_regimes else next(iter(all_regimes))
    regime_map, soft_regime, regime_desc = all_regimes[regime_key]
    logger.info(
        "FROZEN transfer %s→%s regime=%s — %s policy=%s",
        train_data_root.name,
        eval_data_root.name,
        regime_key,
        regime_desc,
        policy.policy_id if policy else None,
    )

    bt0 = base_bt or BacktestConfig()
    strat_overrides = strategy.backtest_overrides()
    if policy is not None:
        overrides = policy.to_backtest_overrides(strat_overrides)
    else:
        overrides = dict(strat_overrides)
    bt_fields = {**bt0.__dict__, **overrides}
    capital = float(bt_fields.get("initial_capital", 100_000.0))
    start_eq = capital

    year_results: List[Dict[str, Any]] = []
    all_trades: List[pd.DataFrame] = []
    equity_segments: List[pd.Series] = []
    positive_years = 0
    n_train_rows_total = 0

    for year in range(first_oos_year, last_oos_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        test_end = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        train_end = test_start

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
                logger.warning(
                    "%s %s: skip train rows=%d (US only)", strategy.name, year, len(train_df)
                )
                continue
            n_train_rows_total += len(train_df)
            # Integrity: no foreign suffix tickers in train rows if column present
            if "ticker" in train_df.columns:
                bad = train_df["ticker"].astype(str).str.upper()
                for suf in foreign_suffix_denylist:
                    if bad.str.endswith(suf.upper()).any():
                        raise RuntimeError(f"Foreign suffix {suf} leaked into US train frame")
            y = train_df["y_side"]
            hold = train_df[y == 1]
            nonhold = train_df[y != 1]
            if len(hold) > 2 * len(nonhold) and len(nonhold) > 1000:
                hold = hold.sample(n=min(len(hold), 2 * len(nonhold)), random_state=year)
                train_df = pd.concat([nonhold, hold], ignore_index=True)
            logger.info("%s %s: training on US only rows=%d...", strategy.name, year, len(train_df))
            strategy.train(train_df, year)

        bt = BacktestConfig(**{k: v for k, v in bt_fields.items() if k in BacktestConfig.__dataclass_fields__})
        bt.initial_capital = capital
        if bt.require_regime and regime_map:
            bt.regime_ok = regime_map
            bt.soft_regime_ok = soft_regime if soft_regime else None
        else:
            bt.regime_ok = None
            bt.soft_regime_ok = None

        trades, equity, _ = run_portfolio_backtest(
            eval_panels, strategy, bt, start=test_start, end=test_end
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
            "%s %s FROZEN: ret=%.1f%% trades=%d sharpe=%.2f",
            strategy.name,
            year,
            year_results[-1]["year_return"] * 100,
            yrep.n_trades,
            yrep.sharpe,
        )

    if not equity_segments:
        raise RuntimeError(f"No OOS equity for frozen transfer {strategy.name}")

    full_equity = pd.concat(equity_segments)
    full_equity = full_equity[~full_equity.index.duplicated(keep="last")].sort_index()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    bench = load_benchmark_equity(
        eval_data_root,
        full_equity.index.min(),
        full_equity.index.max(),
        preferred=pref,
    )
    report = equity_metrics(
        full_equity,
        start_equity=start_eq,
        trades=trades_df,
        benchmark=bench,
        positive_year_frac=positive_years / max(len(year_results), 1),
    )
    gates = transfer_acceptance_gates(report)
    passed = transfer_primary_passed(gates)

    # Single-foreign product mode (multi-market aggregation done by CLI/report)
    us_pass = True if us_home_pass is None else bool(us_home_pass)
    product_mode = compute_product_mode(
        us_home_pass=us_pass,
        foreign_results=[{"market": market_id, "transfer_passed": passed}],
    )

    return {
        "mode": "FROZEN_US_TRANSFER",
        "strategy": strategy.name,
        "description": strategy.description,
        "market_id": market_id,
        "policy_id": policy.policy_id if policy else None,
        "regime_key": regime_key,
        "regime_desc": regime_desc,
        "preferred_index": list(pref),
        "benchmark_name": pref[0] if pref else None,
        "benchmark_cagr": report.benchmark_cagr,
        "report": report,
        "gates": gates,
        "transfer_passed": passed,
        "product_mode": product_mode,
        "year_results": year_results,
        "n_train_tickers": len(train_panels),
        "n_eval_tickers": len(eval_panels),
        "n_train_rows_us": n_train_rows_total,
        "train_data_root": str(train_data_root),
        "eval_data_root": str(eval_data_root),
        "feature_set": getattr(strategy, "feature_names", None),
        "train_markets": ["US"],
        "trades": trades_df,
        "equity": full_equity,
    }
