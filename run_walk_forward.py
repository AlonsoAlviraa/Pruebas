#!/usr/bin/env python3
"""Walk-Forward Analysis script for the Sniper strategy."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from train_signal_model import build_model

logger = logging.getLogger("run_walk_forward")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis")
    parser.add_argument("--tickers", help="Comma separated tickers")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-start", help="Force training start date")
    parser.add_argument("--first-test-start", help="First OOS date (defaults to 2022-01-01)")
    parser.add_argument("--test-period-days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--take-profit", type=float, help="Optional fixed TP")
    parser.add_argument("--stop-loss", type=float, help="Optional fixed SL")
    parser.add_argument("--atr-multiplier-tp", type=float, default=2.0)
    parser.add_argument("--atr-multiplier-sl", type=float, default=2.0)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--model-type", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _parse_date(value: Optional[str], default: Optional[str] = None) -> Optional[pd.Timestamp]:
    if value:
        ts = pd.Timestamp(value)
    elif default:
        ts = pd.Timestamp(default)
    else:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _ticker_list(args: argparse.Namespace, data_root: Path) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(part.strip().upper() for part in args.tickers.split(",") if part.strip())
    if not tickers:
        tickers = [path.name.replace("_history.csv", "") for path in data_root.glob("*_history.csv")]
    if not tickers:
        raise ValueError("No tickers available for walk-forward")
    return tickers


def _load_master_frame(
    tickers: List[str],
    pipeline: DataPipeline,
    horizon: int,
    tp: Optional[float],
    sl: Optional[float],
    atr_tp: float,
    atr_sl: float,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        try:
            df = pipeline.load_feature_view(ticker, indicators=True)
            labeled = pipeline.create_triple_barrier_labels(
                df,
                horizon=horizon,
                take_profit=tp,
                stop_loss=sl,
                atr_multiplier_tp=atr_tp,
                atr_multiplier_sl=atr_sl,
            )
            labeled = labeled[labeled["label"].notna()].copy()
            labeled["ticker"] = ticker
            frames.append(labeled)
        except Exception as exc:
            logger.warning("Unable to prepare %s: %s", ticker, exc)
    if not frames:
        raise ValueError("No labelled data for walk-forward analysis")
    master = pd.concat(frames, axis=0, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"], utc=True)
    master = master.sort_values("date").reset_index(drop=True)
    return master


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["date", "label", "ticker", "index", "tp_pct", "sl_pct", "time_exit_return", "summary"]
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return features.select_dtypes(include=[np.number]).fillna(0.0)


def _label_to_return(row: pd.Series) -> float:
    label = row.get("label")
    if label == 1:
        return float(row.get("tp_pct", 0.0))
    if label == -1:
        return float(row.get("sl_pct", 0.0))
    return float(row.get("time_exit_return", 0.0))


def _append_equity(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades.empty:
        return trades
    trades = trades.sort_values("date").reset_index(drop=True)
    compounded = (1 + trades["return"]).cumprod()
    trades["equity"] = initial_capital * compounded
    trades["cum_return"] = compounded - 1
    return trades


def run_walk_forward(
    master: pd.DataFrame,
    first_test_start: pd.Timestamp,
    test_period_days: int,
    model_params: Dict[str, object],
    min_confidence: float,
    initial_capital: float,
) -> Dict[str, pd.DataFrame]:
    max_date = master["date"].max()
    current_cutoff = first_test_start
    all_predictions: List[pd.DataFrame] = []

    while current_cutoff < max_date:
        test_end = current_cutoff + pd.Timedelta(days=test_period_days)
        logger.info("Walk iteration | Train < %s | Test [%s -> %s]", current_cutoff.date(), current_cutoff.date(), test_end.date())

        train_mask = master["date"] < current_cutoff
        test_mask = (master["date"] >= current_cutoff) & (master["date"] < test_end)
        train_df = master.loc[train_mask]
        test_df = master.loc[test_mask]

        if len(train_df) < 100 or test_df.empty:
            logger.warning("Skipping iteration due to insufficient data")
            current_cutoff = test_end
            continue

        X_train = _feature_matrix(train_df)
        y_train = train_df["label"].astype(int)
        X_test = _feature_matrix(test_df)

        model = build_model(**model_params)
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)
            classes = list(model.classes_)
            buy_idx = classes.index(1) if 1 in classes else None
            buy_probs = probas[:, buy_idx] if buy_idx is not None else np.zeros(len(X_test))
        else:
            buy_probs = np.zeros(len(X_test))
        preds = model.predict(X_test)

        fold_results = test_df[["date", "ticker", "label", "tp_pct", "sl_pct", "time_exit_return", "close"]].copy()
        fold_results["prob_buy"] = buy_probs
        fold_results["prediction"] = preds
        all_predictions.append(fold_results)

        current_cutoff = test_end

    if not all_predictions:
        raise ValueError("Walk-forward produced no predictions")

    results = pd.concat(all_predictions, axis=0, ignore_index=True)

    trades = results[results["prob_buy"] > min_confidence].copy()
    trades["return"] = trades.apply(_label_to_return, axis=1)
    trades = trades.dropna(subset=["return"])  # Remove NaNs from tail
    equity = _append_equity(trades, initial_capital)

    return {"predictions": results, "trades": equity}


def summarize_equity(trades: pd.DataFrame) -> None:
    if trades.empty:
        logger.warning("No trades triggered during WFA")
        return
    total_return = trades["cum_return"].iloc[-1]
    win_rate = (trades["return"] > 0).mean()
    logger.info("Walk-forward trades: %d", len(trades))
    logger.info("Win rate: %.2f%%", win_rate * 100)
    logger.info("Total compounded return: %.2f%%", total_return * 100)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    data_root = args.data_root
    tickers = _ticker_list(args, data_root)
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))

    master = _load_master_frame(
        tickers,
        pipeline,
        args.horizon,
        args.take_profit,
        args.stop_loss,
        args.atr_multiplier_tp,
        args.atr_multiplier_sl,
    )

    first_test_start = _parse_date(args.first_test_start or args.train_start, default="2022-01-01")
    if first_test_start is None:
        raise ValueError("Unable to determine the first test date")

    model_params: Dict[str, object] = dict(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        n_jobs=args.n_jobs,
        class_weighted=args.class_weighted,
        random_state=args.random_state,
    )

    results = run_walk_forward(
        master,
        first_test_start,
        args.test_period_days,
        model_params,
        args.min_confidence,
        args.initial_capital,
    )

    predictions = results["predictions"]
    trades = results["trades"]

    predictions.to_csv("walk_forward_predictions.csv", index=False)
    trades.to_csv("walk_forward_equity.csv", index=False)

    summarize_equity(trades)