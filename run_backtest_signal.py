#!/usr/bin/env python3
"""
Backtester rápido vectorizado para la estrategia Event-Driven (Sniper).
Simula la ejecución de señales generadas por el modelo ML.
"""
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from drl_platform.model_factory import BUY_CLASS


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
    stop_loss: float,
) -> float:
    """
    Calcula el retorno de una operación iniciada en t+1 (Open) hasta el cierre.
    """
    entry_price = prices[t]
    
    upper = entry_price * (1 + take_profit)
    lower = entry_price * (1 + stop_loss)
    
    end_idx = min(t + 1 + horizon, len(prices))
    window = prices[t + 1 : end_idx]
    
    if len(window) == 0:
        return 0.0
        
    hit_upper = window >= upper
    hit_lower = window <= lower
    
    idx_upper = np.argmax(hit_upper) if hit_upper.any() else len(window) + 1
    idx_lower = np.argmax(hit_lower) if hit_lower.any() else len(window) + 1
    
    if idx_upper < idx_lower:
        return take_profit
    elif idx_lower < idx_upper:
        return stop_loss
    else:
        exit_price = window[-1]
        return (exit_price - entry_price) / entry_price


def _align_features(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Reordena las columnas para coincidir con lo que el modelo espera."""
    numeric = frame.select_dtypes(include=[np.number])
    if hasattr(model, "feature_names_in_"):
        # Rellenar con 0 si falta alguna columna y ordenar estrictamente
        ordered = numeric.reindex(columns=list(model.feature_names_in_), fill_value=0.0)
        return ordered
    return numeric


def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _filter_date_window(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if start is None and end is None:
        return df
    dates = pd.to_datetime(df["date"], utc=True)
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return df.loc[mask].copy()


def run_backtest(
    tickers: List[str],
    data_root: Path,
    model: Any,
    min_confidence: float,
    horizon: int,
    take_profit: float,
    stop_loss: float,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    trades = []
    
    logger.info(f"Iniciando backtest para {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            df = pipeline.load_feature_view(ticker, indicators=True)
            if df.empty:
                continue

            df = _filter_date_window(df, start_date, end_date)
            if df.empty:
                continue

            drop_cols = ["date", "label", "ticker", "index", "tp_pct", "sl_pct", "time_exit_return"]
            raw_features = df.drop(columns=[c for c in drop_cols if c in df.columns])
            features = _align_features(raw_features, model)

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features)
                classes = model.classes_
                buy_idx = np.where(classes == BUY_CLASS)[0]
                
                if len(buy_idx) > 0:
                    buy_probs = probas[:, buy_idx[0]]
                    signals = buy_probs > min_confidence
                else:
                    signals = np.zeros(len(df), dtype=bool)
            else:
                preds = model.predict(features)
                signals = preds == BUY_CLASS

                
            prices = df["close"].values
            dates = pd.to_datetime(df["date"]).values
            
            buy_indices = np.where(signals)[0]
            
            t = 0
            while t < len(prices) - 1:
                if t in buy_indices:
                    ret = calculate_trade_result(prices, t, horizon, take_profit, stop_loss)
                    
                    trades.append({
                        "ticker": ticker,
                        "entry_date": dates[t],
                        "return": ret
                    })
                    
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
    parser.add_argument("--start-date", help="YYYY-MM-DD start date for the evaluation")
    parser.add_argument("--end-date", help="YYYY-MM-DD end date for the evaluation")
    
    args = parser.parse_args()
    
    tickers = []
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    if args.ticker_file and args.ticker_file.exists():
        content = args.ticker_file.read_text(encoding="utf-8").splitlines()
        tickers.extend([t.strip().upper() for t in content if t.strip()])
    if not tickers:
        files = list(args.data_root.glob("*_history.csv"))
        tickers = [f.name.replace("_history.csv", "") for f in files]
        
    model = load_model(args.model_path)
    
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    results = run_backtest(
        tickers,
        args.data_root,
        model,
        args.min_confidence,
        args.horizon,
        args.take_profit,
        args.stop_loss,
        start_date,
        end_date,
    )
    
    if results.empty:
        logger.warning("No se generaron operaciones.")
        return
        
    results["entry_date"] = pd.to_datetime(results["entry_date"])
    results = results.sort_values("entry_date")
    
    results["cum_return"] = results["return"].cumsum()
    results["equity"] = args.initial_capital * (1 + results["cum_return"])
    
    total_trades = len(results)
    win_rate = (results["return"] > 0).mean()
    
    logger.info("-" * 40)
    logger.info(f"RESULTADOS DEL BACKTEST ({total_trades} operaciones)")
    logger.info(f"Win Rate: {win_rate:.2%}")

if __name__ == "__main__":
    main()