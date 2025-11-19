#!/usr/bin/env python3
"""Train the event-driven ML signal model with Purged K-Fold validation."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    parser.add_argument("--shap-report-dir", type=Path, help="Directorio para guardar gráficos y tablas SHAP")
    parser.add_argument("--shap-threshold", type=float, default=0.01, help="Impacto normalizado mínimo para conservar una feature")
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
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


def _generate_shap_report(
    model: Any,
    master: pd.DataFrame,
    feature_names: Sequence[str],
    report_dir: Path,
    threshold: float,
) -> None:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo importar SHAP. Instala la dependencia con 'pip install shap'"
        ) from exc

    if not feature_names:
        raise ValueError("El modelo no tiene columnas de features registradas")

    X = _feature_matrix(master)
    X = X.reindex(columns=feature_names).fillna(0.0)
    if X.empty:
        raise ValueError("No hay datos para calcular valores SHAP")

    sample_size = min(len(X), 5000)
    sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # --- LOGICA ROBUSTA DE EXTRACCION SHAP ---
    shap_buy = None
    buy_idx = 2 # Default a la clase 2 si no se encuentra

    # Intentar detectar indice de BUY_CLASS
    if hasattr(model, "classes_"):
        matches = np.where(model.classes_ == BUY_CLASS)[0]
        if len(matches):
            buy_idx = int(matches[0])

    if isinstance(shap_values, list):
        # Caso 1: Lista de arrays [stop, hold, buy]
        # Proteccion contra indices fuera de rango
        safe_idx = buy_idx if buy_idx < len(shap_values) else -1
        shap_buy = shap_values[safe_idx]
    
    elif isinstance(shap_values, np.ndarray):
        # Caso 2: Array numpy
        if len(shap_values.shape) == 3: # (Samples, Features, Classes)
            safe_idx = buy_idx if buy_idx < shap_values.shape[2] else -1
            shap_buy = shap_values[:, :, safe_idx]
        else:
            # Caso Binario (Samples, Features) - Asumimos que es la clase positiva
            shap_buy = shap_values
    else:
         raise TypeError(f"Formato de shap_values no reconocido: {type(shap_values)}")
    # -----------------------------------------

    mean_abs = np.abs(shap_buy).mean(axis=0)
    importance = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": mean_abs,
    })
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    max_val = importance["mean_abs_shap"].max()
    if not max_val or np.isnan(max_val):
        max_val = 1.0
    importance["normalized_importance"] = importance["mean_abs_shap"] / max_val

    filtered = importance[importance["normalized_importance"] >= threshold]
    low_impact = importance[importance["normalized_importance"] < threshold]

    report_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(report_dir / "shap_feature_importance.csv", index=False)
    filtered.to_csv(report_dir / "shap_feature_importance_filtered.csv", index=False)
    low_impact[["feature"]].to_csv(report_dir / "low_impact_features.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(filtered["feature"][::-1], filtered["mean_abs_shap"][::-1])
    plt.xlabel("|SHAP|")
    plt.title("Top Features - Clase BUY")
    plt.tight_layout()
    plt.savefig(report_dir / "shap_bar_buy.png", dpi=150)
    plt.close()

    shap.summary_plot(shap_buy, sample, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(report_dir / "shap_summary_buy.png", dpi=150)
    plt.close()


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

    if args.shap_report_dir:
        try:
            _generate_shap_report(model, master, feature_names, args.shap_report_dir, args.shap_threshold)
            logger.info("SHAP report saved to %s", args.shap_report_dir)
        except Exception as exc:  # pragma: no cover - plotting heavy
            logger.warning("No se pudo generar el reporte SHAP: %s", exc)


if __name__ == "__main__":
    main()