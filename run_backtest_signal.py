#!/usr/bin/env python3
"""
Backtester rápido vectorizado para la estrategia Event-Driven (Sniper).
Simula la ejecución de señales generadas por el modelo ML.
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drl_platform.data_pipeline import DataPipeline, PipelineConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_backtest_signal")

def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    try:
        return joblib.load(model_path)
    except:
        with open(model_path, "rb") as f:
            return pickle.load(f)

def calculate_trade_result(
    prices: np.ndarray,
    t: int,
    horizon: int,
    take_profit: float,
    stop_loss: float
) -> float:
    """
    Calcula el retorno de una operación iniciada en t+1 (Open) hasta el cierre.
    Retorna el % de ganancia/pérdida.
    """
    # Asumimos entrada en Open de t+1. 
    # Si no tenemos Open, usamos Close de t (aproximación) o Close de t+1.
    # Para ser conservadores y simples con datos diarios (que suelen tener OHLC):
    # Si tenemos array de precios 'close', usamos close[t] como proxy de entrada o close[t+1].
    # Lo ideal es usar Open[t+1].
    
    # Aquí recibimos solo 'prices' (Close).
    # Entrada: Close[t] (Simulando compra al cierre de la señal o apertura siguiente muy cercana)
    entry_price = prices[t]
    
    upper = entry_price * (1 + take_profit)
    lower = entry_price * (1 + stop_loss)
    
    # Ventana de precios futuros
    end_idx = min(t + 1 + horizon, len(prices))
    window = prices[t+1 : end_idx]
    
    if len(window) == 0:
        return 0.0
        
    # Chequear barreras
    # Nota: Esto es una simplificación. En la realidad, High/Low determinan si tocamos barrera.
    # Aquí solo tenemos Close.
    
    hit_upper = window >= upper
    hit_lower = window <= lower
    
    idx_upper = np.argmax(hit_upper) if hit_upper.any() else len(window) + 1
    idx_lower = np.argmax(hit_lower) if hit_lower.any() else len(window) + 1
    
    if idx_upper < idx_lower:
        # Take Profit
        return take_profit
    elif idx_lower < idx_upper:
        # Stop Loss
        return stop_loss
    else:
        # Time Exit
        exit_price = window[-1]
        return (exit_price - entry_price) / entry_price

def run_backtest(
    tickers: List[str],
    data_root: Path,
    model: Any,
    min_confidence: float,
    horizon: int,
    take_profit: float,
    stop_loss: float
) -> pd.DataFrame:
    
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    trades = []
    
    logger.info(f"Iniciando backtest para {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            # Cargar datos
            df = pipeline.load_feature_view(ticker, indicators=True)
            if df.empty:
                continue
                
            # Preparar features
            # Mismas columnas que en entrenamiento
            drop_cols = ["date", "label", "ticker", "index"]
            features = df.drop(columns=[c for c in drop_cols if c in df.columns])
            features = features.select_dtypes(include=[np.number])
            
            # Predecir
            # Necesitamos alinear features con el modelo.
            # Si el modelo espera nombres específicos, esto podría fallar si hay discrepancia.
            # Asumimos que el pipeline es consistente.
            
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features)
                # Asumimos clases [-1, 0, 1] mapeadas a índices 0, 1, 2
                # Ojo: verificar classes_ del modelo
                classes = model.classes_
                buy_idx = np.where(classes == 1)[0]
                
                if len(buy_idx) > 0:
                    buy_probs = probas[:, buy_idx[0]]
                    signals = buy_probs > min_confidence
                else:
                    signals = np.zeros(len(df), dtype=bool)
            else:
                preds = model.predict(features)
                signals = preds == 1
                
            # Simular Trades
            prices = df["close"].values
            dates = df["date"].values
            
            # Indices donde hay señal de compra
            buy_indices = np.where(signals)[0]
            
            # Filtrar señales consecutivas (opcional, para no comprar cada día si ya estamos dentro)
            # Implementación simple: si compramos, saltamos 'horizon' días.
            
            t = 0
            while t < len(prices) - 1:
                if t in buy_indices:
                    # Ejecutar trade
                    ret = calculate_trade_result(prices, t, horizon, take_profit, stop_loss)
                    
                    trades.append({
                        "ticker": ticker,
                        "entry_date": dates[t],
                        "return": ret
                    })
                    
                    # Saltar horizonte (asumiendo capital ocupado)
                    t += horizon
                else:
                    t += 1
                    
        except Exception as e:
            logger.warning(f"Error en backtest de {ticker}: {e}")
            continue
            
    return pd.DataFrame(trades)

def main():
    parser = argparse.ArgumentParser(description="Backtester Event-Driven")
    parser.add_argument("--tickers", help="Lista de tickers")
    parser.add_argument("--ticker-file", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--take-profit", type=float, default=0.05)
    parser.add_argument("--stop-loss", type=float, default=-0.03)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    
    args = parser.parse_args()
    
    # Obtener tickers
    tickers = []
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    if args.ticker_file and args.ticker_file.exists():
        content = args.ticker_file.read_text(encoding="utf-8").splitlines()
        tickers.extend([t.strip().upper() for t in content if t.strip()])
    if not tickers:
        files = list(args.data_root.glob("*_history.csv"))
        tickers = [f.name.replace("_history.csv", "") for f in files]
        
    # Cargar modelo
    model = load_model(args.model_path)
    
    # Ejecutar
    results = run_backtest(
        tickers, 
        args.data_root, 
        model, 
        args.min_confidence,
        args.horizon,
        args.take_profit,
        args.stop_loss
    )
    
    if results.empty:
        logger.warning("No se generaron operaciones.")
        return
        
    # Análisis de Resultados
    results["entry_date"] = pd.to_datetime(results["entry_date"])
    results = results.sort_values("entry_date")
    
    # Curva de Equidad Simple (Interés Compuesto)
    # Asumimos que invertimos todo el capital en cada trade (o una fracción fija)
    # Para simplificar: Suma de retornos (Interés Simple sobre base 1)
    
    results["cum_return"] = results["return"].cumsum()
    results["equity"] = args.initial_capital * (1 + results["cum_return"])
    
    total_trades = len(results)
    win_rate = (results["return"] > 0).mean()
    avg_return = results["return"].mean()
    total_return = results["return"].sum()
    
    logger.info("-" * 40)
    logger.info(f"RESULTADOS DEL BACKTEST ({total_trades} operaciones)")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Retorno Promedio por Trade: {avg_return:.2%}")
    logger.info(f"Retorno Total Acumulado: {total_return:.2%}")
    logger.info("-" * 40)
    
    # Guardar CSV
    out_csv = "backtest_signal_results.csv"
    results.to_csv(out_csv, index=False)
    logger.info(f"Detalle de operaciones guardado en {out_csv}")

if __name__ == "__main__":
    main()