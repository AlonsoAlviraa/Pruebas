"""Expanding-window walk-forward training and multi-year OOS evaluation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trad_research.backtest import BacktestConfig, run_portfolio_backtest
from trad_research.features import (
    M2_FEATURE_NAMES,
    M2_REL_FEATURE_NAMES,
    feature_matrix,
    load_featured,
    list_tickers,
)
from trad_research.config import DEFAULT_LABEL_CONFIG, LabelConfig
from trad_research.labels import attach_labels
from trad_research.metrics import PerformanceReport, acceptance_gates, equity_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    data_root: Path = Path("data")
    ticker_file: Path = Path("good_tickers_filtrados.txt")
    universe_limit: int = 80
    first_oos_year: int = 2018
    last_oos_year: int = 2025
    embargo_days: int = 5
    k_tp: float = 2.0
    k_sl: float = 1.5
    label_horizon: int = 20
    min_train_rows: int = 5000
    use_relative_features: bool = False  # Lean M2 17-feature set by default
    use_meta_label: bool = True
    meta_threshold: float = 0.50
    label_config: LabelConfig = None  # type: ignore
    backtest: BacktestConfig = None  # type: ignore

    def __post_init__(self) -> None:
        if self.backtest is None:
            self.backtest = BacktestConfig()
        if self.label_config is None:
            self.label_config = LabelConfig(
                k_tp=self.k_tp,
                k_sl=self.k_sl,
                max_horizon=self.label_horizon,
            )

    @property
    def feature_names(self) -> tuple:
        return M2_REL_FEATURE_NAMES if self.use_relative_features else M2_FEATURE_NAMES


def _load_panels(
    tickers: Sequence[str],
    data_root: Path,
    *,
    with_fundamentals: bool = False,
) -> Dict[str, pd.DataFrame]:
    panels: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = load_featured(t, data_root)
        if df.empty:
            continue
        if with_fundamentals:
            from trad_research.universe import attach_fundamental_flags

            df = attach_fundamental_flags(df, t, data_root)
        panels[t] = df
    return panels


def _build_training_frame(
    panels: Dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    embargo_days: int,
    k_tp: float,
    k_sl: float,
    label_horizon: int,
) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    cutoff = train_end - pd.Timedelta(days=int(embargo_days * 1.5))
    for t, df in panels.items():
        d = df[df["date"] < cutoff].copy()
        if len(d) < 250:
            continue
        d = attach_labels(
            d,
            k_tp=k_tp,
            k_sl=k_sl,
            max_horizon=label_horizon,
            config=LabelConfig(k_tp=k_tp, k_sl=k_sl, max_horizon=label_horizon),
        )
        # Drop last label_horizon rows (incomplete barriers into embargo)
        if len(d) > label_horizon + 50:
            d = d.iloc[: -label_horizon]
        chunks.append(d)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _xgb_device_kwargs() -> Dict[str, Any]:
    """Optional GPU for XGBoost 2.x (Kaggle T4 / local CUDA).

    Set env ``XGB_DEVICE=cuda`` (or ``cuda:0``) to enable. Default CPU hist.
    """
    import os

    dev = (os.environ.get("XGB_DEVICE") or "").strip()
    if not dev or dev.lower() in ("cpu", "none", "0", "false"):
        return {"tree_method": "hist", "n_jobs": 4}
    # XGBoost 2.0+: device=cuda|cuda:0, tree_method=hist
    return {
        "tree_method": "hist",
        "device": dev if dev.startswith("cuda") else "cuda",
        "n_jobs": 1,
    }


def train_meta_model(
    train_df: pd.DataFrame,
    primary: XGBClassifier,
    feature_names: Sequence[str],
    primary_threshold: float = 0.40,
    random_state: int = 42,
) -> Optional[XGBClassifier]:
    """Meta-label: given primary would buy, will trade be a winner (y_meta)?"""
    X = feature_matrix(train_df, feature_names)
    if hasattr(primary, "predict_proba"):
        proba = primary.predict_proba(X)
        classes = list(getattr(primary, "classes_", [0, 1]))
        buy_i = classes.index(1) if 1 in classes else classes.index(max(classes))
        p_buy = proba[:, buy_i]
    else:
        p_buy = (primary.predict(X) == 1).astype(float)
    mask = p_buy >= primary_threshold
    if mask.sum() < 500:
        return None
    Xm = X.loc[mask]
    ym = train_df.loc[mask, "y_meta"].astype(int)
    if ym.nunique() < 2:
        return None
    pos = max(int(ym.sum()), 1)
    neg = max(int((ym == 0).sum()), 1)
    kw = _xgb_device_kwargs()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=30,
        reg_lambda=3.0,
        objective="binary:logistic",
        scale_pos_weight=min(neg / pos, 5.0),
        random_state=random_state,
        **kw,
    )
    model.fit(Xm, ym)
    return model


def train_side_model(
    train_df: pd.DataFrame,
    feature_names: Sequence[str] = M2_FEATURE_NAMES,
    random_state: int = 42,
    binary_buy: bool = True,
) -> XGBClassifier:
    """
    Train primary signal model.
    binary_buy=True maps triple-barrier to {0: not BUY, 1: BUY} for cleaner P(buy).
    """
    X = feature_matrix(train_df, feature_names)
    y_raw = train_df["y_side"].astype(int)
    kw = _xgb_device_kwargs()
    if binary_buy:
        y = (y_raw == 2).astype(int)
        pos = max(int(y.sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        spw = neg / pos
        model = XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            min_child_weight=60,
            reg_lambda=4.0,
            objective="binary:logistic",
            scale_pos_weight=min(spw, 6.0),
            random_state=random_state,
            eval_metric="logloss",
            **kw,
        )
        model.fit(X, y)
        return model

    counts = y_raw.value_counts().to_dict()
    n = len(y_raw)
    weight_map = {c: n / (len(counts) * counts[c]) for c in counts}
    sw = y_raw.map(weight_map).astype(float)
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        random_state=random_state,
        eval_metric="mlogloss",
        **kw,
    )
    model.fit(X, y_raw, sample_weight=sw)
    return model


def load_benchmark_equity(
    data_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    preferred: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Load index close series for BH metrics.

    preferred: basenames without _history.csv, e.g. ("IBEX",) or ("IBEX", "QQQ").
    Default order if preferred is None: ("QQQ", "SPY", "IVV") — US research unchanged.
    Aligns with regime._load_index_close preferred semantics.
    """
    names = tuple(preferred) if preferred else ("QQQ", "SPY", "IVV")
    for name in names:
        p = Path(data_root) / f"{name}_history.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
        if df.empty:
            continue
        s = df.set_index("date")["close"].astype(float)
        return s
    return pd.Series(dtype=float)


def build_regime_map(
    data_root: Path,
    sma_len: int = 50,
) -> Tuple[Dict[pd.Timestamp, bool], Dict[pd.Timestamp, bool]]:
    """
    Returns (hard_risk_on, soft_risk_on).
    hard: QQQ > SMA50 or SMA20 (v3-style)
    soft: QQQ > SMA100 (full size)
    """
    for name in ("QQQ", "SPY"):
        p = data_root / f"{name}_history.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date")
        close = pd.to_numeric(df["close"], errors="coerce")
        sma = close.rolling(sma_len, min_periods=max(20, sma_len // 2)).mean()
        sma_fast = close.rolling(20, min_periods=10).mean()
        sma_slow = close.rolling(100, min_periods=50).mean()
        hard = (close > sma) | (close > sma_fast)
        soft = close > sma_slow
        hard_map = {d: bool(b) for d, b in zip(df["date"], hard.fillna(True))}
        soft_map = {d: bool(b) for d, b in zip(df["date"], soft.fillna(True))}
        return hard_map, soft_map
    return {}, {}


def run_walk_forward(cfg: WalkForwardConfig) -> Dict[str, Any]:
    tickers = list_tickers(cfg.ticker_file, cfg.data_root, limit=cfg.universe_limit)
    logger.info("Loading %d tickers from %s", len(tickers), cfg.data_root)
    panels = _load_panels(tickers, cfg.data_root)
    logger.info("Loaded featured panels: %d", len(panels))
    regime_map, soft_regime = build_regime_map(cfg.data_root, sma_len=50)
    logger.info("Regime map days: %d soft=%d", len(regime_map), len(soft_regime))

    year_results: List[Dict[str, Any]] = []
    all_trades: List[pd.DataFrame] = []
    equity_segments: List[pd.Series] = []
    capital = cfg.backtest.initial_capital
    positive_years = 0
    oos_years = list(range(cfg.first_oos_year, cfg.last_oos_year + 1))

    for year in oos_years:
        train_end = pd.Timestamp(f"{year}-01-01", tz="UTC")
        test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        test_end = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")

        train_df = _build_training_frame(
            panels,
            train_end=train_end,
            embargo_days=cfg.embargo_days,
            k_tp=cfg.k_tp,
            k_sl=cfg.k_sl,
            label_horizon=cfg.label_horizon,
        )
        if len(train_df) < cfg.min_train_rows:
            logger.warning("Year %s: insufficient train rows (%d)", year, len(train_df))
            continue

        # Optional: only train BUY/SELL samples with clear barriers for cleaner edge
        # Keep HOLDs but downsample HOLDs to balance
        y = train_df["y_side"]
        hold = train_df[y == 1]
        nonhold = train_df[y != 1]
        if len(hold) > 2 * len(nonhold) and len(nonhold) > 1000:
            hold = hold.sample(n=min(len(hold), 2 * len(nonhold)), random_state=year)
            train_df = pd.concat([nonhold, hold], ignore_index=True)

        logger.info("Year %s: training on %d rows...", year, len(train_df))
        model = train_side_model(
            train_df,
            feature_names=cfg.feature_names,
            random_state=year,
        )
        meta = None
        if cfg.use_meta_label:
            meta = train_meta_model(
                train_df,
                model,
                cfg.feature_names,
                primary_threshold=cfg.backtest.min_confidence,
                random_state=year,
            )
            if meta is not None:
                logger.info("Year %s: meta-label model trained", year)

        bt_cfg = BacktestConfig(**{**cfg.backtest.__dict__})
        bt_cfg.initial_capital = capital
        bt_cfg.regime_ok = regime_map if bt_cfg.require_regime else None
        bt_cfg.soft_regime_ok = soft_regime if bt_cfg.require_regime else None
        bt_cfg.feature_names = list(cfg.feature_names)
        bt_cfg.meta_model = meta
        bt_cfg.meta_threshold = cfg.meta_threshold
        if bt_cfg.qqq_sleeve_pct > 0 and "QQQ" in panels:
            bt_cfg.qqq_panel = panels["QQQ"]
        elif bt_cfg.qqq_sleeve_pct > 0:
            qqq_df = load_featured("QQQ", cfg.data_root)
            bt_cfg.qqq_panel = qqq_df if not qqq_df.empty else None
        trades, equity, _ = run_portfolio_backtest(
            panels, model, bt_cfg, start=test_start, end=test_end
        )
        if equity.empty:
            logger.warning("Year %s: empty equity", year)
            continue

        year_pnl = float(equity.iloc[-1] - capital)
        if year_pnl > 0:
            positive_years += 1
        y_report = equity_metrics(equity, start_equity=capital, trades=trades)
        year_results.append(
            {
                "year": year,
                "train_rows": len(train_df),
                "n_trades": y_report.n_trades,
                "start_equity": capital,
                "end_equity": y_report.final_equity,
                "year_return": y_report.final_equity / capital - 1.0,
                "sharpe": y_report.sharpe,
                "max_drawdown": y_report.max_drawdown,
                "win_rate": y_report.win_rate,
            }
        )
        if not trades.empty:
            trades = trades.copy()
            trades["oos_year"] = year
            all_trades.append(trades)
        # Continuity: chain equity; rebase segment for concat
        seg = equity.copy()
        equity_segments.append(seg)
        capital = float(equity.iloc[-1])
        logger.info(
            "Year %s done: ret=%.2f%% trades=%d sharpe=%.2f",
            year,
            year_results[-1]["year_return"] * 100,
            y_report.n_trades,
            y_report.sharpe,
        )

    if not equity_segments:
        raise RuntimeError("No OOS years produced equity — check data/universe.")

    # Stitch equity: use last segment chain already continuous if capital carried
    full_equity = pd.concat(equity_segments)
    full_equity = full_equity[~full_equity.index.duplicated(keep="last")].sort_index()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    start_eq = cfg.backtest.initial_capital
    bench = load_benchmark_equity(
        cfg.data_root,
        full_equity.index.min(),
        full_equity.index.max(),
    )
    pos_frac = positive_years / max(len(year_results), 1)
    report = equity_metrics(
        full_equity,
        start_equity=start_eq,
        trades=trades_df,
        benchmark=bench,
        positive_year_frac=pos_frac,
    )
    gates = acceptance_gates(report, min_years=max(1.0, len(year_results) * 0.9))
    # years_ok based on calendar coverage
    gates["years_ok"] = len(year_results) >= 8 or (
        report.years >= 7.0 and len(year_results) >= 6
    )
    # stretch_* keys are informational only — not required to pass
    required = {k: v for k, v in gates.items() if not k.startswith("stretch_")}
    passed = all(required.values())

    return {
        "report": report,
        "gates": gates,
        "year_results": year_results,
        "trades": trades_df,
        "equity": full_equity,
        "passed": passed,
        "n_tickers": len(panels),
        "oos_years": [r["year"] for r in year_results],
    }
