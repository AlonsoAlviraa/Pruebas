#!/usr/bin/env python3
"""Train the event-driven ML signal model with Purged K-Fold validation."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from drl_platform.validation import PurgedKFoldConfig, PurgedKFoldValidator
from drl_platform.model_factory import (
    BUY_CLASS,
    CLASS_MAPPING,
    CLASS_NAMES,
    HOLD_CLASS,
    ModelParams,
    STOP_CLASS,
    build_model,
)

logger = logging.getLogger("train_signal_model")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the financial event-driven signal model")
    parser.add_argument("--tickers", help="Comma separated tickers")
    parser.add_argument("--ticker-file", type=Path, help="Optional file with one ticker per line")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/signal_model.joblib"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--take-profit", type=float, help="Fixed TP override")
    parser.add_argument("--stop-loss", type=float, help="Fixed SL override")
    parser.add_argument("--atr-multiplier-tp", type=float, default=2.0)
    parser.add_argument("--atr-multiplier-sl", type=float, default=2.0)
    parser.add_argument("--train-until", help="Use data <= this date (YYYY-MM-DD)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--purge-window", type=int, default=5)

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
    parser.add_argument("--min-confidence", type=float, default=0.6)
    return parser.parse_args()


def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _load_tickers(args: argparse.Namespace, data_root: Path) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(part.strip().upper() for part in args.tickers.split(",") if part.strip())
    if args.ticker_file and args.ticker_file.exists():
        tickers.extend(line.strip().upper() for line in args.ticker_file.read_text().splitlines() if line.strip())
    if not tickers:
        tickers = [path.name.replace("_history.csv", "") for path in data_root.glob("*_history.csv")]
    if not tickers:
        raise ValueError("No tickers provided or discovered in data directory")
    return sorted(set(tickers))


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["date", "label", "ticker", "index", "tp_pct", "sl_pct", "time_exit_return", "summary"]
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return features.select_dtypes(include=[np.number]).fillna(0.0)


def _prepare_master(
    tickers: Sequence[str],
    pipeline: DataPipeline,
    horizon: int,
    take_profit: Optional[float],
    stop_loss: Optional[float],
    atr_tp: float,
    atr_sl: float,
    train_until: Optional[pd.Timestamp],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        try:
            frame = pipeline.load_feature_view(ticker, indicators=True)
        except Exception as exc:
            logger.warning("Unable to load %s: %s", ticker, exc)
            continue
        labeled = pipeline.create_triple_barrier_labels(
            frame,
            horizon=horizon,
            take_profit=take_profit,
            stop_loss=stop_loss,
            atr_multiplier_tp=atr_tp,
            atr_multiplier_sl=atr_sl,
        )
        labeled["ticker"] = ticker
        frames.append(labeled)
    if not frames:
        raise ValueError("No labelled samples were produced")
    master = pd.concat(frames, axis=0, ignore_index=True)
    master = master.dropna(subset=["label"]).copy()
    master["date"] = pd.to_datetime(master["date"], utc=True)
    if train_until is not None:
        master = master[master["date"] <= train_until]
        if master.empty:
            raise ValueError("No samples remain after applying --train-until filter")
    master = master.sort_values("date").reset_index(drop=True)
    return master


def _purged_kfold_scores(master: pd.DataFrame, params: ModelParams, n_splits: int, purge_window: int) -> List[Dict[str, float]]:
    validator = PurgedKFoldValidator(PurgedKFoldConfig(n_splits=n_splits, purge_window=purge_window))
    X = _feature_matrix(master)
    y = master["label"].map(CLASS_MAPPING).astype(int)
    results: List[Dict[str, float]] = []
    for fold, (train_idx, test_idx) in enumerate(validator.split(master)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        if y_train.nunique() < 2:
            logger.warning("Fold %d skipped due to single class in training data", fold)
            continue
        model = build_model(**asdict(params))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1_buy = f1_score(y_test, preds, labels=[BUY_CLASS], average="micro", zero_division=0)
        report = classification_report(y_test, preds, labels=[STOP_CLASS, HOLD_CLASS, BUY_CLASS], output_dict=True, zero_division=0)
        results.append(
            {
                "fold": fold,
                "f1_buy": float(f1_buy),
                "support_buy": int((y_test == BUY_CLASS).sum()),
                "accuracy": float(report.get("accuracy", 0.0)),
            }
        )
    return results


def _train_final_model(master: pd.DataFrame, params: ModelParams):
    X = _feature_matrix(master)
    y = master["label"].map(CLASS_MAPPING).astype(int)
    model = build_model(**asdict(params))
    model.fit(X, y)
    return model, list(X.columns)


def _serialise(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    return obj


def _save_metadata(
    output_path: Path,
    tickers: Sequence[str],
    master: pd.DataFrame,
    params: ModelParams,
    cv_results: List[Dict[str, float]],
    feature_names: Sequence[str],
    pipeline: DataPipeline,
    strategy_cfg: Dict[str, float],
) -> Path:
    if cv_results:
        cv_mean = float(np.mean([row["f1_buy"] for row in cv_results]))
        cv_std = float(np.std([row["f1_buy"] for row in cv_results]))
    else:
        cv_mean = cv_std = 0.0
    metadata = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_path": str(output_path.resolve()),
        "tickers": list(tickers),
        "train_rows": int(len(master)),
        "train_range": {
            "start": master["date"].min().isoformat() if not master.empty else None,
            "end": master["date"].max().isoformat() if not master.empty else None,
        },
        "class_mapping": CLASS_NAMES,
        "feature_columns": list(feature_names),
        "model_params": asdict(params),
        "pipeline_config": asdict(pipeline.config),
        "strategy": strategy_cfg,
        "cv_metrics": {
            "folds": cv_results,
            "f1_buy_mean": cv_mean,
            "f1_buy_std": cv_std,
        },
    }
    cleaned = _serialise(metadata)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    tickers = _load_tickers(args, args.data_root)
    logger.info("Preparing training data for %d tickers", len(tickers))

    pipeline = DataPipeline(PipelineConfig(data_root=args.data_root))
    train_until = _parse_date(args.train_until)
    master = _prepare_master(
        tickers,
        pipeline,
        args.horizon,
        args.take_profit,
        args.stop_loss,
        args.atr_multiplier_tp,
        args.atr_multiplier_sl,
        train_until,
    )

    params = ModelParams(
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

    logger.info("Running Purged K-Fold validation (splits=%d, purge=%d)", args.n_splits, args.purge_window)
    cv_results = _purged_kfold_scores(master, params, args.n_splits, args.purge_window)
    if cv_results:
        logger.info("Mean F1(BUY): %.4f ± %.4f", np.mean([r["f1_buy"] for r in cv_results]), np.std([r["f1_buy"] for r in cv_results]))
    else:
        logger.warning("Cross-validation skipped due to insufficient folds")

    model, feature_names = _train_final_model(master, params)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Model saved to %s", output_path)

    strategy_cfg = {
        "horizon": args.horizon,
        "take_profit": args.take_profit,
        "stop_loss": args.stop_loss,
        "atr_multiplier_tp": args.atr_multiplier_tp,
        "atr_multiplier_sl": args.atr_multiplier_sl,
    }
    metadata_path = _save_metadata(output_path, tickers, master, params, cv_results, feature_names, pipeline, strategy_cfg)
    logger.info("Metadata saved to %s", metadata_path)


if __name__ == "__main__":
    main()