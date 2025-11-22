#!/usr/bin/env python3
"""
TRAIN MODEL v2.0 - Modelo ML Mejorado con Optimización de Hiperparámetros
- Datos completos 2015-2024
- Optimización Bayesiana (Optuna)
- Enfoque en features importantes (SHAP)
- Triple Barrier Labels mejorado
- Purged K-Fold temporal
"""
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
from sklearn.metrics import classification_report, f1_score, roc_auc_score

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

logger = logging.getLogger("train_signal_model_v2")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IMPROVED ML signal model v2.0")
    parser.add_argument("--tickers", help="Comma separated tickers")
    parser.add_argument("--ticker-file", type=Path, help="Optional file with one ticker per line")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/signal_model_v2.joblib"))
    
    # Triple Barrier Labels
    parser.add_argument("--horizon", type=int, default=10, help="Horizonte de trading (días)")
    parser.add_argument("--take-profit", type=float, help="Fixed TP override")
    parser.add_argument("--stop-loss", type=float, help="Fixed SL override")
    parser.add_argument("--atr-multiplier-tp", type=float, default=3.0, help="Multiplicador ATR para TP")
    parser.add_argument("--atr-multiplier-sl", type=float, default=2.0, help="Multiplicador ATR para SL")
    
    # Data Filtering
    parser.add_argument("--train-from", default="2015-01-01", help="Inicio del período de entrenamiento")
    parser.add_argument("--train-until", default="2022-12-31", help="Fin del período de entrenamiento")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--purge-window", type=int, default=10, help="Ventana de purga entre folds")

    # Hyperparameter Optimization
    parser.add_argument("--optimize-hyperparams", action="store_true", help="Use Optuna para optimizar hiperparámetros")
    parser.add_argument("--n-trials", type=int, default=50, help="Número de trials para Optuna")
    
    # Memory Management
    parser.add_argument("--max-tickers", type=int, help="Máximo número de tickers a usar (para limitar memoria)")
    parser.add_argument("--max-samples-per-ticker", type=int, default=5000, help="Máximo de muestras por ticker")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Ratio de sampling del dataset final (0.1-1.0)")
    parser.add_argument("--min-ticker-samples", type=int, default=252, help="Mínimo de muestras requeridas por ticker")
    
    # Model Parameters (si no se optimizan)
    parser.add_argument("--model-type", choices=["rf", "xgb", "lgbm"], default="xgb")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    
    # Analysis
    parser.add_argument("--shap-report-dir", type=Path, help="Directorio para guardar análisis SHAP")
    parser.add_argument("--shap-threshold", type=float, default=0.01)
    
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
    features = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    categorical_cols = features.select_dtypes(include=["category"]).columns
    for col in categorical_cols:
        codes = features[col].cat.codes.replace(-1, np.nan)
        features[col] = codes
    
    return features.select_dtypes(include=[np.number]).fillna(0.0)


def _prepare_master(
    tickers: Sequence[str],
    pipeline: DataPipeline,
    horizon: int,
    take_profit: Optional[float],
    stop_loss: Optional[float],
    atr_tp: float,
    atr_sl: float,
    train_from: Optional[pd.Timestamp],
    train_until: Optional[pd.Timestamp],
    max_samples_per_ticker: int = 5000,
    min_ticker_samples: int = 252,
    sample_ratio: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Prepara el dataset maestro con todas las features y labels, optimizado para memoria."""
    frames: List[pd.DataFrame] = []
    
    logger.info(f"Cargando datos para hasta {len(tickers)} tickers...")
    logger.info(f"Límite por ticker: {max_samples_per_ticker:,} muestras")
    logger.info(f"Mínimo por ticker: {min_ticker_samples:,} muestras")
    
    tickers_loaded = 0
    tickers_skipped = 0
    
    for ticker in tickers:
        try:
            frame = pipeline.load_feature_view(ticker, indicators=True)
            
            if frame.empty:
                tickers_skipped += 1
                continue
            
            # Filtrar por fechas ANTES de crear labels (ahorro de memoria)
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            if train_from is not None:
                frame = frame[frame["date"] >= train_from]
            if train_until is not None:
                frame = frame[frame["date"] <= train_until]
            
            # Si después del filtro tiene pocas muestras, skip
            if len(frame) < min_ticker_samples:
                logger.debug(f"Ticker {ticker} skipped: solo {len(frame)} muestras (min={min_ticker_samples})")
                tickers_skipped += 1
                continue
            
            # Limitar muestras por ticker (tomar las más recientes)
            if len(frame) > max_samples_per_ticker:
                frame = frame.sort_values("date").tail(max_samples_per_ticker).reset_index(drop=True)
                logger.debug(f"Ticker {ticker}: limitado a {max_samples_per_ticker} muestras más recientes")
            
            # Crear labels con Triple Barrier
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
            tickers_loaded += 1
            
            # Log progreso cada 50 tickers
            if tickers_loaded % 50 == 0:
                logger.info(f"Progreso: {tickers_loaded} tickers cargados, {tickers_skipped} omitidos")
            
        except Exception as exc:
            logger.warning("Unable to load %s: %s", ticker, exc)
            tickers_skipped += 1
            continue
    
    if not frames:
        raise ValueError("No labelled samples were produced")
    
    logger.info(f"Concatenando {len(frames)} dataframes...")
    master = pd.concat(frames, axis=0, ignore_index=True)
    
    # Liberar memoria de frames individuales
    del frames
    import gc
    gc.collect()
    
    logger.info(f"Dataset antes de limpieza: {len(master):,} muestras")
    
    # Dropna en subset (más eficiente que todo)
    master = master.dropna(subset=["label"])
    
    logger.info(f"Dataset después de dropna: {len(master):,} muestras")
    
    # Si todavía es muy grande, aplicar sampling
    if sample_ratio < 1.0:
        original_size = len(master)
        master = master.sample(frac=sample_ratio, random_state=random_state)
        master = master.sort_values("date").reset_index(drop=True)
        logger.info(f"Sampling aplicado: {original_size:,} -> {len(master):,} muestras ({sample_ratio:.1%})")
    else:
        master = master.sort_values("date").reset_index(drop=True)
    
    logger.info(f"Dataset maestro final: {len(master):,} muestras de {tickers_loaded} tickers")
    logger.info(f"Rango de fechas: {master['date'].min()} a {master['date'].max()}")
    
    # Distribución de clases
    class_dist = master["label"].value_counts()
    logger.info(f"Distribución de clases:\n{class_dist}")
    
    # Uso de memoria aproximado
    memory_mb = master.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"Uso de memoria del dataset: {memory_mb:.1f} MB")
    
    return master


def optimize_hyperparameters_optuna(
    master: pd.DataFrame,
    model_type: str,
    n_trials: int,
    n_splits: int,
    purge_window: int,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Optimiza hiperparámetros usando Optuna (Bayesian Optimization).
    Objetivo: Maximizar F1-Score de la clase BUY.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError("Instala Optuna: pip install optuna")
    
    logger.info(f"Iniciando optimización Bayesiana con Optuna ({n_trials} trials)...")
    
    X = _feature_matrix(master)
    y = master["label"].map(CLASS_MAPPING).astype(int)
    
    validator = PurgedKFoldValidator(PurgedKFoldConfig(n_splits=n_splits, purge_window=purge_window))
    
    def objective(trial):
        # Definir espacio de búsqueda según el modelo
        if model_type == "xgb":
            params = ModelParams(
                model_type="xgb",
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 4, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_samples_split=4,  # No se usa en XGB
                n_jobs=-1,
                class_weighted=True,
                random_state=random_state
            )
        elif model_type == "rf":
            params = ModelParams(
                model_type="rf",
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 5, 50),
                learning_rate=0.05,  # No se usa en RF
                subsample=0.8,  # No se usa en RF
                colsample_bytree=trial.suggest_float("max_features", 0.5, 1.0),
                n_jobs=-1,
                class_weighted=True,
                random_state=random_state
            )
        else:  # lgbm
            params = ModelParams(
                model_type="lgbm",
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 4, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("feature_fraction", 0.6, 1.0),
                min_samples_split=4,
                n_jobs=-1,
                class_weighted=True,
                random_state=random_state
            )
        
        # Cross-validation
        f1_scores = []
        for fold, (train_idx, test_idx) in enumerate(validator.split(master)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            if y_train.nunique() < 2:
                continue
            
            model = build_model(**asdict(params))
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # F1-Score de la clase BUY (la más importante)
            f1_buy = f1_score(y_test, preds, labels=[BUY_CLASS], average="micro", zero_division=0)
            f1_scores.append(f1_buy)
        
        if not f1_scores:
            return 0.0
        
        return np.mean(f1_scores)
    
    # Crear estudio y optimizar
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"trend_following_{model_type}"
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Mejor F1-Score (BUY): {study.best_value:.4f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
    
    return study.best_params


def _purged_kfold_scores(
    master: pd.DataFrame,
    params: ModelParams,
    n_splits: int,
    purge_window: int
) -> List[Dict[str, float]]:
    """Validación Cross-Validated con Purged K-Fold."""
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
        
        # Métricas
        f1_buy = f1_score(y_test, preds, labels=[BUY_CLASS], average="micro", zero_division=0)
        report = classification_report(
            y_test, preds,
            labels=[STOP_CLASS, HOLD_CLASS, BUY_CLASS],
            output_dict=True,
            zero_division=0
        )
        
        results.append({
            "fold": fold,
            "f1_buy": float(f1_buy),
            "support_buy": int((y_test == BUY_CLASS).sum()),
            "accuracy": float(report.get("accuracy", 0.0)),
        })
    
    return results


def _train_final_model(master: pd.DataFrame, params: ModelParams):
    """Entrena el modelo final con todo el dataset."""
    X = _feature_matrix(master)
    y = master["label"].map(CLASS_MAPPING).astype(int)
    
    logger.info(f"Entrenando modelo final con {len(X):,} muestras...")
    
    model = build_model(**asdict(params))
    model.fit(X, y)
    
    return model, list(X.columns)


def _generate_shap_report(
    model: Any,
    master: pd.DataFrame,
    feature_names: Sequence[str],
    report_dir: Path,
    threshold: float,
) -> None:
    """Genera reporte SHAP para interpretar el modelo."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Instala SHAP: pip install shap") from exc

    if not feature_names:
        raise ValueError("El modelo no tiene columnas de features registradas")

    X = _feature_matrix(master)
    X = X.reindex(columns=feature_names).fillna(0.0)
    
    if X.empty:
        raise ValueError("No hay datos para calcular valores SHAP")

    # Sample para acelerar (máximo 5000 muestras)
    sample_size = min(len(X), 5000)
    sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Extraer SHAP values para clase BUY
    shap_buy = None
    buy_idx = 2  # Default

    if hasattr(model, "classes_"):
        matches = np.where(model.classes_ == BUY_CLASS)[0]
        if len(matches):
            buy_idx = int(matches[0])

    if isinstance(shap_values, list):
        safe_idx = buy_idx if buy_idx < len(shap_values) else -1
        shap_buy = shap_values[safe_idx]
    elif isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            safe_idx = buy_idx if buy_idx < shap_values.shape[2] else -1
            shap_buy = shap_values[:, :, safe_idx]
        else:
            shap_buy = shap_values
    else:
        raise TypeError(f"Formato de shap_values no reconocido: {type(shap_values)}")

    # Feature Importance
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
    importance.to_csv(report_dir / "shap_feature_importance_v2.csv", index=False)
    filtered.to_csv(report_dir / "shap_feature_importance_filtered_v2.csv", index=False)
    low_impact[["feature"]].to_csv(report_dir / "low_impact_features_v2.csv", index=False)

    # Gráficos
    plt.figure(figsize=(12, 8))
    plt.barh(filtered["feature"].head(20)[::-1], filtered["mean_abs_shap"].head(20)[::-1])
    plt.xlabel("|SHAP|")
    plt.title("Top 20 Features - Clase BUY")
    plt.tight_layout()
    plt.savefig(report_dir / "shap_bar_buy_v2.png", dpi=150)
    plt.close()

    shap.summary_plot(shap_buy, sample, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(report_dir / "shap_summary_buy_v2.png", dpi=150)
    plt.close()
    
    logger.info(f"Reporte SHAP guardado en {report_dir}")


def _serialise(obj):
    """Serializa objetos para JSON."""
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
    strategy_cfg: Dict[str, Any],
    best_hyperparams: Optional[Dict[str, Any]] = None,
) -> Path:
    """Guarda metadata del modelo."""
    if cv_results:
        cv_mean = float(np.mean([row["f1_buy"] for row in cv_results]))
        cv_std = float(np.std([row["f1_buy"] for row in cv_results]))
    else:
        cv_mean = cv_std = 0.0
    
    metadata = {
        "version": "2.0",
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
        "best_hyperparams_optuna": best_hyperparams,
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
    
    logger.info(f"Metadata guardada en {metadata_path}")
    return metadata_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    args = _parse_args()

    logger.info("="*70)
    logger.info("  ENTRENAMIENTO MODELO ML v2.0 - TREND FOLLOWING")
    logger.info("="*70)
    logger.info(f"  Período: {args.train_from} a {args.train_until}")
    logger.info(f"  Modelo: {args.model_type.upper()}")
    logger.info(f"  Optimización: {'SÍ' if args.optimize_hyperparams else 'NO'}")
    logger.info("="*70)

    tickers = _load_tickers(args, args.data_root)
    logger.info(f"Preparando datos para {len(tickers)} tickers...")

    pipeline = DataPipeline(PipelineConfig(data_root=args.data_root))
    train_from = _parse_date(args.train_from)
    train_until = _parse_date(args.train_until)
    
    # Limitar tickers si se solicita (para ahorrar memoria)
    if args.max_tickers and len(tickers) > args.max_tickers:
        logger.info(f"Limitando a {args.max_tickers} tickers de {len(tickers)} disponibles")
        tickers = tickers[:args.max_tickers]
    
    master = _prepare_master(
        tickers, pipeline,
        args.horizon,
        args.take_profit,
        args.stop_loss,
        args.atr_multiplier_tp,
        args.atr_multiplier_sl,
        train_from,
        train_until,
        max_samples_per_ticker=args.max_samples_per_ticker,
        min_ticker_samples=args.min_ticker_samples,
        sample_ratio=args.sample_ratio,
        random_state=args.random_state,
    )

    # Optimización de hiperparámetros si se solicita
    best_hyperparams = None
    if args.optimize_hyperparams:
        best_hyperparams = optimize_hyperparameters_optuna(
            master,
            args.model_type,
            args.n_trials,
            args.n_splits,
            args.purge_window,
            args.random_state
        )
        
        # Construir ModelParams con los mejores hiperparámetros
        params = ModelParams(
            model_type=args.model_type,
            n_estimators=best_hyperparams.get("n_estimators", args.n_estimators),
            max_depth=best_hyperparams.get("max_depth", args.max_depth),
            min_samples_split=best_hyperparams.get("min_samples_split", args.min_samples_split),
            learning_rate=best_hyperparams.get("learning_rate", args.learning_rate),
            subsample=best_hyperparams.get("subsample", args.subsample),
            colsample_bytree=best_hyperparams.get("colsample_bytree", args.colsample_bytree),
            n_jobs=args.n_jobs,
            class_weighted=True,  # Siempre True para balancear clases
            random_state=args.random_state,
        )
    else:
        # Usar parámetros por defecto/argumentos
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

    logger.info(f"Ejecutando Purged K-Fold validation (splits={args.n_splits}, purge={args.purge_window})...")
    cv_results = _purged_kfold_scores(master, params, args.n_splits, args.purge_window)
    
    if cv_results:
        mean_f1 = np.mean([r["f1_buy"] for r in cv_results])
        std_f1 = np.std([r["f1_buy"] for r in cv_results])
        logger.info(f" F1(BUY) Cross-Validation: {mean_f1:.4f} ± {std_f1:.4f}")
    else:
        logger.warning("Cross-validation skipped due to insufficient folds")

    # Entrenar modelo final
    model, feature_names = _train_final_model(master, params)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Modelo guardado en {output_path}")

    strategy_cfg = {
        "horizon": args.horizon,
        "take_profit": args.take_profit,
        "stop_loss": args.stop_loss,
        "atr_multiplier_tp": args.atr_multiplier_tp,
        "atr_multiplier_sl": args.atr_multiplier_sl,
    }
    
    metadata_path = _save_metadata(
        output_path, tickers, master, params, cv_results, feature_names,
        pipeline, strategy_cfg, best_hyperparams
    )

    # Generar reporte SHAP si se solicita
    if args.shap_report_dir:
        try:
            _generate_shap_report(
                model, master, feature_names,
                args.shap_report_dir,
                args.shap_threshold
            )
        except Exception as exc:
            logger.warning(f"No se pudo generar el reporte SHAP: {exc}")

    logger.info("="*70)
    logger.info("  ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
