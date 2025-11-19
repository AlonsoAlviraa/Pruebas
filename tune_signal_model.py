#!/usr/bin/env python3
"""Optuna based hyper-parameter tuning for the Sniper signal model."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
except ImportError as exc:  # pragma: no cover - CLI entry point
    raise SystemExit("Optuna is required for tuning: pip install optuna") from exc

try:  # Optional dependency
    from xgboost import XGBClassifier  # type: ignore[import-untyped]

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from train_signal_model import build_model

logger = logging.getLogger("tune_signal_model")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning via Optuna")
    parser.add_argument("--tickers", help="Comma separated ticker universe")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--model-type", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--objective", choices=["f1_buy", "sharpe"], default="f1_buy")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--take-profit", type=float, help="Optional fixed TP")
    parser.add_argument("--stop-loss", type=float, help="Optional fixed SL")
    parser.add_argument("--atr-multiplier-tp", type=float, default=2.0)
    parser.add_argument("--atr-multiplier-sl", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _ticker_list(args: argparse.Namespace, data_root: Path) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(part.strip().upper() for part in args.tickers.split(",") if part.strip())
    if not tickers:
        tickers = [path.name.replace("_history.csv", "") for path in data_root.glob("*_history.csv")]
    if not tickers:
        raise ValueError("No tickers available")
    return tickers


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


def _assemble_dataset(
    raw_data: Dict[str, pd.DataFrame],
    pipeline: DataPipeline,
    horizon: int,
    tp: Optional[float],
    sl: Optional[float],
    atr_tp: float,
    atr_sl: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    frames: List[pd.DataFrame] = []
    for ticker, frame in raw_data.items():
        labeled = pipeline.create_triple_barrier_labels(
            frame,
            horizon=horizon,
            take_profit=tp,
            stop_loss=sl,
            atr_multiplier_tp=atr_tp,
            atr_multiplier_sl=atr_sl,
        )
        labeled = labeled[labeled["label"].notna()].copy()
        labeled["ticker"] = ticker
        frames.append(labeled)
    if not frames:
        raise ValueError("No labelled samples available for tuning")
    master = pd.concat(frames, axis=0, ignore_index=True)
    X = _feature_matrix(master)
    y = master["label"].astype(int)
    realized = master.apply(_label_to_return, axis=1)
    return X, y, realized


def _build_model_from_trial(trial: "optuna.trial.Trial", model_type: str):
    if model_type == "rf":
        params = dict(
            model_type="rf",
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 5, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=-1,
            class_weighted=True,
            random_state=trial.suggest_int("random_state", 1, 5000),
        )
        return build_model(**params)
    params = dict(
        model_type="xgb",
        n_estimators=trial.suggest_int("n_estimators", 200, 600),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_samples_split=2,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        n_jobs=-1,
        class_weighted=False,
        random_state=trial.suggest_int("random_state", 1, 5000),
    )
    if not HAS_XGB:
        raise RuntimeError("xgboost is required for model_type='xgb'")
    return build_model(**params)


def _score_trial(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    objective: str,
) -> float:
    splitter = TimeSeriesSplit(n_splits=3)
    scores: List[float] = []
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if objective == "f1_buy":
            scores.append(f1_score(y_test, preds, labels=[1], average="micro", zero_division=0))
        else:
            mask = preds == 1
            trade_returns = returns.iloc[test_idx][mask]
            if trade_returns.empty:
                scores.append(-1.0)
            else:
                mean = trade_returns.mean()
                std = trade_returns.std(ddof=1)
                sharpe = np.sqrt(252) * mean / (std + 1e-8)
                scores.append(float(sharpe))
    return float(np.mean(scores))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    data_root = args.data_root
    tickers = _ticker_list(args, data_root)
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    raw_data = {ticker: pipeline.load_feature_view(ticker, indicators=True) for ticker in tickers}

    def objective(trial: "optuna.trial.Trial") -> float:
        horizon = trial.suggest_int("horizon", max(3, args.horizon - 2), args.horizon + 2)
        tp = trial.suggest_float("take_profit", 0.02, 0.10) if args.take_profit is None else args.take_profit
        sl = trial.suggest_float("stop_loss", -0.08, -0.01) if args.stop_loss is None else args.stop_loss
        atr_tp = trial.suggest_float("atr_multiplier_tp", 1.0, 3.0)
        atr_sl = trial.suggest_float("atr_multiplier_sl", 1.0, 3.0)

        X, y, realized = _assemble_dataset(raw_data, pipeline, horizon, tp, sl, atr_tp, atr_sl)
        model = _build_model_from_trial(trial, args.model_type)
        score = _score_trial(model, X, y, realized, args.objective)
        return score

    logger.info("Launching Optuna study with %d trials", args.n_trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Best score: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entry point
        logger.exception("Tuning failed: %s", exc)
        sys.exit(1)