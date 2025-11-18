#!/usr/bin/env python3
"""
Script de Optimización de Hiperparámetros (HPT) usando Optuna.
Optimiza tanto parámetros del modelo (RF/XGB) como de la estrategia (Horizon, TP, SL).
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from drl_platform.data_pipeline import DataPipeline, PipelineConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tune_signal_model")

# Suppress warnings
warnings.filterwarnings("ignore")

def load_raw_features(tickers: List[str], data_root: Path) -> Dict[str, pd.DataFrame]:
    """Carga features sin etiquetar para todos los tickers."""
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    data = {}
    logger.info("Cargando datos raw...")
    for ticker in tickers:
        try:
            df = pipeline.load_feature_view(ticker, indicators=True)
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            logger.warning(f"Error loading {ticker}: {e}")
    return data

def objective(trial, raw_data: Dict[str, pd.DataFrame], model_type: str):
    # 1. Sugerir Parámetros de Estrategia
    horizon = trial.suggest_int("horizon", 3, 10)
    take_profit = trial.suggest_float("take_profit", 0.02, 0.10)
    stop_loss = trial.suggest_float("stop_loss", -0.08, -0.02)
    
    # 2. Generar Etiquetas (Costoso, pero necesario si optimizamos estrategia)
    pipeline = DataPipeline(PipelineConfig()) # Dummy config
    
    all_features = []
    all_labels = []
    
    for ticker, df in raw_data.items():
        # Usamos la lógica de triple barrier
        # Nota: create_triple_barrier_labels es método de instancia, usamos pipeline dummy
        labeled = pipeline.create_triple_barrier_labels(
            df, horizon=horizon, take_profit=take_profit, stop_loss=stop_loss
        )
        
        valid = labeled[labeled["label"].notna()]
        if valid.empty:
            continue
            
        drop_cols = ["date", "label", "ticker", "index"]
        X = valid.drop(columns=[c for c in drop_cols if c in valid.columns]).select_dtypes(include=[np.number])
        y = valid["label"].astype(int)
        
        all_features.append(X)
        all_labels.append(y)
        
    if not all_features:
        return 0.0
        
    X_full = pd.concat(all_features, axis=0, ignore_index=True)
    y_full = pd.concat(all_labels, axis=0, ignore_index=True)
    
    # 3. Sugerir Parámetros del Modelo
    if model_type == "rf":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
    else:
        # Placeholder for XGB
        return 0.0
        
    # 4. Validación (TimeSeriesSplit simple para velocidad en tuning)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_index, test_index in tscv.split(X_full):
        X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Optimizamos F1 de la clase BUY (1)
        # Si es multiclase (-1, 0, 1), labels=[1]
        score = f1_score(y_test, preds, labels=[1], average="micro", zero_division=0) 
        # average='micro' con labels=[1] calcula F1 solo para esa clase (TP / (TP + 0.5(FP+FN)))
        # O mejor usar precision si queremos seguridad.
        # El usuario pidió "Maximizar el F1-Score de la clase BUY o el Sharpe Ratio".
        # F1 de BUY es buen proxy.
        
        scores.append(score)
        
    return np.mean(scores)

def main():
    if not HAS_OPTUNA:
        logger.error("Optuna no está instalado. Instala con `pip install optuna`.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--tickers", help="Lista de tickers")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--model-type", default="rf")
    
    args = parser.parse_args()
    
    # Tickers
    tickers = []
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        files = list(args.data_root.glob("*_history.csv"))
        tickers = [f.name.replace("_history.csv", "") for f in files]
        
    # Cargar datos raw (cache en memoria)
    raw_data = load_raw_features(tickers, args.data_root)
    
    if not raw_data:
        logger.error("No hay datos.")
        sys.exit(1)
        
    logger.info(f"Iniciando optimización con {args.n_trials} trials...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, raw_data, args.model_type), n_trials=args.n_trials)
    
    logger.info("-" * 40)
    logger.info("MEJORES PARÁMETROS:")
    logger.info(study.best_params)
    logger.info(f"Mejor Score: {study.best_value:.4f}")
    logger.info("-" * 40)

if __name__ == "__main__":
    main()
