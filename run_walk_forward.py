#!/usr/bin/env python3
"""
Script de Walk-Forward Analysis (WFA) para la estrategia Event-Driven.
Simula el re-entrenamiento periódico para validar la robustez de la estrategia.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from train_signal_model import train_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_walk_forward")

def load_all_data(
    tickers: List[str],
    data_root: Path,
    horizon: int,
    take_profit: float,
    stop_loss: float,
    use_dynamic_barriers: bool,
    barrier_std: float
) -> pd.DataFrame:
    """Carga y etiqueta todos los datos disponibles."""
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    all_dfs = []
    
    logger.info(f"Cargando datos históricos para {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            df = pipeline.load_feature_view(ticker, indicators=True)
            if df.empty:
                continue
                
            labeled_df = pipeline.create_triple_barrier_labels(
                df, 
                horizon=horizon, 
                take_profit=take_profit, 
                stop_loss=stop_loss,
                use_dynamic_barriers=use_dynamic_barriers,
                barrier_std=barrier_std
            )
            
            # Limpiar NaNs de etiquetas (final de serie)
            valid_mask = labeled_df["label"].notna()
            clean_df = labeled_df[valid_mask].copy()
            
            if clean_df.empty:
                continue
                
            clean_df["ticker"] = ticker
            all_dfs.append(clean_df)
            
        except Exception as e:
            logger.warning(f"Error cargando {ticker}: {e}")
            
    if not all_dfs:
        raise ValueError("No se cargaron datos.")
        
    master_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df = master_df.sort_values("date")
    
    return master_df

def run_walk_forward(
    master_df: pd.DataFrame,
    start_date: str,
    train_period_days: int, # Periodo inicial de entrenamiento
    test_period_days: int,  # Ventana de test (re-entrenamiento)
    model_params: Dict[str, Any]
):
    """
    Ejecuta el bucle de Walk-Forward.
    Expanding Window: Train [Start, T], Test [T, T+step]
    """
    dates = master_df["date"].unique()
    dates.sort()
    
    min_date = dates[0]
    max_date = dates[-1]
    
    # Definir fecha de corte inicial
    if start_date:
        current_cutoff = pd.Timestamp(start_date)
    else:
        current_cutoff = min_date + pd.Timedelta(days=train_period_days)
        
    logger.info(f"Rango de datos: {min_date} a {max_date}")
    logger.info(f"Inicio de Walk-Forward (Primer Test): {current_cutoff}")
    
    all_predictions = []
    
    iteration = 1
    
    while current_cutoff < max_date:
        test_end = current_cutoff + pd.Timedelta(days=test_period_days)
        if test_end > max_date:
            test_end = max_date
            
        logger.info(f"Iteración {iteration}: Train [Start -> {current_cutoff.date()}] | Test [{current_cutoff.date()} -> {test_end.date()}]")
        
        # Slice Data
        train_mask = master_df["date"] < current_cutoff
        test_mask = (master_df["date"] >= current_cutoff) & (master_df["date"] < test_end)
        
        train_df = master_df[train_mask]
        test_df = master_df[test_mask]
        
        if train_df.empty or test_df.empty:
            logger.warning("Datos insuficientes para esta iteración.")
            current_cutoff = test_end
            continue
            
        # Prepare Features
        drop_cols = ["date", "label", "ticker", "index"]
        X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).select_dtypes(include=[np.number])
        y_train = train_df["label"].astype(int)
        
        X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).select_dtypes(include=[np.number])
        y_test = test_df["label"].astype(int)
        
        # Train
        model = train_model(X_train, y_train, **model_params)
        model.fit(X_train, y_train)
        
        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            # Asumimos que la clase 1 es BUY. Necesitamos saber el índice.
            classes = model.classes_
            if 1 in classes:
                buy_idx = np.where(classes == 1)[0][0]
                buy_probs = probs[:, buy_idx]
            else:
                buy_probs = np.zeros(len(X_test))
            
            preds = model.predict(X_test)
        else:
            preds = model.predict(X_test)
            buy_probs = np.zeros(len(X_test)) # No probs
            
        # Store results
        results = test_df[["date", "ticker", "label", "close"]].copy()
        results["pred"] = preds
        results["prob_buy"] = buy_probs
        
        all_predictions.append(results)
        
        # Metrics for this fold
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        logger.info(f"  -> Precision (Weighted): {prec:.4f}")
        
        # Advance
        current_cutoff = test_end
        iteration += 1
        
    if not all_predictions:
        logger.error("No se generaron predicciones.")
        return pd.DataFrame()
        
    full_results = pd.concat(all_predictions, axis=0, ignore_index=True)
    return full_results

def analyze_results(results: pd.DataFrame, min_confidence: float = 0.6):
    """Calcula curva de equidad sobre los resultados concatenados."""
    logger.info("-" * 40)
    logger.info("ANÁLISIS DE RESULTADOS WALK-FORWARD")
    
    # Filtrar operaciones
    # Compramos si prob_buy > min_confidence (o si pred == 1 si no hay probs)
    # Asumimos retorno simple: si label == 1 (Buy) ganamos TakeProfit? 
    # No, label es el resultado perfecto. Si label=1, ganamos. Si label=-1, perdemos.
    # Pero necesitamos saber CUÁNTO ganamos.
    # El label 1 significa "Tocó TP antes que SL". Ganancia = TP.
    # El label -1 significa "Tocó SL antes que TP". Pérdida = SL.
    # El label 0 significa "Time Exit". Retorno = (Close_end - Close_start) / Close_start.
    # Para simplificar, usaremos una aproximación basada en labels, 
    # pero idealmente deberíamos simular día a día.
    # Dado que create_triple_barrier_labels ya codifica el resultado, podemos usar proxies:
    # Label 1 -> +TP
    # Label -1 -> +SL (que es negativo)
    # Label 0 -> 0 (Neutral/Flat) o un promedio pequeño.
    
    # Vamos a usar una simulación simplificada.
    # Necesitamos los parámetros TP/SL usados para etiquetar.
    # Como no los tenemos aquí pasados explícitamente, asumiremos los defaults o los pasados por args.
    # O mejor, calculamos retornos reales si tuviéramos precios de salida, pero aquí solo tenemos 'close' de entrada.
    
    # Para este reporte, usaremos Precision/Recall real.
    
    y_true = results["label"]
    y_pred = (results["prob_buy"] > min_confidence).astype(int)
    # Ajustar y_pred: 1 es Buy, 0 es Hold/Sell.
    # Pero el modelo predice -1, 0, 1.
    # Si prob_buy > conf, predecimos 1. Si no, 0 (o lo que sea).
    # Comparemos solo cuando decidimos comprar.
    
    trades = results[results["prob_buy"] > min_confidence]
    
    if trades.empty:
        logger.warning("No se ejecutaron trades con la confianza mínima.")
        return
        
    n_trades = len(trades)
    winners = (trades["label"] == 1).sum()
    losers = (trades["label"] == -1).sum()
    timed_out = (trades["label"] == 0).sum()
    
    win_rate = winners / n_trades
    
    logger.info(f"Total Trades Simulados: {n_trades}")
    logger.info(f"Winners (TP): {winners}")
    logger.info(f"Losers (SL): {losers}")
    logger.info(f"Timeouts: {timed_out}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    
    # Guardar CSV
    results.to_csv("walk_forward_results.csv", index=False)
    logger.info("Resultados guardados en walk_forward_results.csv")

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis")
    parser.add_argument("--tickers", help="Lista de tickers")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--start-date", help="Fecha inicio test (YYYY-MM-DD)")
    parser.add_argument("--initial-train-days", type=int, default=730, help="Días iniciales de entrenamiento (si no hay start-date)")
    parser.add_argument("--test-period-days", type=int, default=90, help="Tamaño ventana de test (días)")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--take-profit", type=float, default=0.05)
    parser.add_argument("--stop-loss", type=float, default=-0.03)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--model-type", default="rf")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--dynamic-barriers", action="store_true")
    parser.add_argument("--barrier-std", type=float, default=2.0)
    
    args = parser.parse_args()
    
    # Tickers
    tickers = []
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        files = list(args.data_root.glob("*_history.csv"))
        tickers = [f.name.replace("_history.csv", "") for f in files]
        
    # Cargar datos
    master_df = load_all_data(
        tickers, 
        args.data_root, 
        args.horizon, 
        args.take_profit, 
        args.stop_loss,
        args.dynamic_barriers,
        args.barrier_std
    )
    
    # Config modelo
    model_params = {
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "max_depth": 10 # Default
    }
    
    # Ejecutar WFA
    results = run_walk_forward(
        master_df,
        args.start_date,
        args.initial_train_days,
        args.test_period_days,
        model_params
    )
    
    if not results.empty:
        analyze_results(results, args.min_confidence)

if __name__ == "__main__":
    main()
