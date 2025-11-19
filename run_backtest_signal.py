#!/usr/bin/env python3
"""
Backtester Vectorizado con Gestión Dinámica (Chandelier Exit) y Filtro de Régimen.
Estrategia: "XGBoost Entry + Chandelier Exit"
"""
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Dict

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore
import yfinance as yf

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


def get_market_regime(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Descarga QQQ y calcula la SMA 50 para determinar el régimen de mercado.
    Retorna un DataFrame con índice de fecha y columna 'is_bullish'.
    """
    logger.info("Descargando datos de QQQ para filtro de régimen...")
    # Descargar un poco antes para tener datos para la media móvil
    start_buffer = start_date - pd.Timedelta(days=100)
    qqq = yf.download("QQQ", start=start_buffer, end=end_date, progress=False)
    
    if qqq.empty:
        logger.warning("No se pudieron descargar datos de QQQ. El filtro de régimen estará desactivado.")
        return pd.DataFrame()

    # Aplanar MultiIndex si existe (yfinance v0.2+)
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)
    
    qqq = qqq.rename(columns={"Close": "close"})
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None) # Asegurar naive para compatibilidad fácil o convertir todo a UTC después
    
    # Calcular SMA 50
    qqq["sma_50"] = ta.sma(qqq["close"], length=50)
    qqq["is_bullish"] = qqq["close"] > qqq["sma_50"]
    
    # Filtrar solo el rango relevante y asegurar UTC
    qqq = qqq.loc[start_date.tz_localize(None) : end_date.tz_localize(None)].copy()
    qqq.index = qqq.index.tz_localize("UTC")
    
    return qqq[["is_bullish"]]


def calculate_chandelier_exit(
    prices: np.ndarray,
    highs: np.ndarray,
    atrs: np.ndarray,
    start_idx: int,
    k: float = 3.0,
    max_horizon: int = 20
) -> Dict[str, Any]:
    """
    Simula una operación usando Chandelier Exit (Trailing Stop).
    Stop = Highest High desde entrada - k * ATR.
    """
    entry_price = prices[start_idx]
    initial_atr = atrs[start_idx]
    
    # Stop Loss inicial
    stop_loss = entry_price - (k * initial_atr)
    highest_high = highs[start_idx]
    
    for i in range(1, max_horizon + 1):
        curr_idx = start_idx + i
        if curr_idx >= len(prices):
            # Fin de datos, cerrar al cierre
            return {
                "exit_price": prices[-1],
                "exit_idx": len(prices) - 1,
                "reason": "end_of_data",
                "bars_held": i
            }
            
        curr_price = prices[curr_idx]
        curr_high = highs[curr_idx]
        curr_atr = atrs[curr_idx]
        
        # Verificar si tocó el stop
        if curr_price < stop_loss: # Asumimos cierre por debajo, o low (si tuviéramos low intradía sería mejor)
            return {
                "exit_price": stop_loss, # Ejecución al precio del stop (slippage aparte)
                "exit_idx": curr_idx,
                "reason": "stop_loss",
                "bars_held": i
            }
            
        # Actualizar Trailing Stop
        if curr_high > highest_high:
            highest_high = curr_high
        
        # El stop solo sube
        new_stop = highest_high - (k * curr_atr)
        stop_loss = max(stop_loss, new_stop)
        
    # Salida por tiempo (horizonte máximo alcanzado sin saltar stop)
    return {
        "exit_price": prices[start_idx + max_horizon],
        "exit_idx": start_idx + max_horizon,
        "reason": "horizon_limit",
        "bars_held": max_horizon
    }


def _align_features(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
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


def run_backtest(
    tickers: List[str],
    data_root: Path,
    model: Any,
    min_confidence: float,
    k_atr: float,
    max_horizon: int,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    commission: float = 0.001, # 0.1% por operación
) -> pd.DataFrame:
    
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    trades = []
    
    # Obtener Régimen de Mercado
    regime_df = pd.DataFrame()
    if start_date and end_date:
        try:
            regime_df = get_market_regime(start_date, end_date)
        except Exception as e:
            logger.warning(f"Error obteniendo régimen de mercado: {e}")

    logger.info(f"Iniciando backtest para {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            df = pipeline.load_feature_view(ticker, indicators=True)
            if df.empty:
                continue
            
            # Filtrar por fecha
            dates = pd.to_datetime(df["date"], utc=True)
            mask = pd.Series(True, index=df.index)
            if start_date: mask &= dates >= start_date
            if end_date: mask &= dates <= end_date
            df = df.loc[mask].copy()
            
            if df.empty: continue

            # Preparar Features
            drop_cols = ["date", "label", "ticker", "index", "tp_pct", "sl_pct", "time_exit_return", "summary"]
            raw_features = df.drop(columns=[c for c in drop_cols if c in df.columns])
            features = _align_features(raw_features, model)

            # Generar Señales IA
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(features)
                classes = model.classes_
                buy_idx = np.where(classes == 1)[0] # Asumiendo 1 es BUY (o 2 según tu mapping)
                # Ajuste rápido: si tu mapping es 0:STOP, 1:HOLD, 2:BUY, busca el índice de 2
                if 2 in classes:
                    buy_idx = np.where(classes == 2)[0]
                elif 1 in classes and len(classes) == 2: # Binario
                     buy_idx = np.where(classes == 1)[0]
                
                if len(buy_idx) > 0:
                    buy_probs = probas[:, buy_idx[0]]
                    signals = buy_probs > min_confidence
                else:
                    signals = np.zeros(len(df), dtype=bool)
            else:
                preds = model.predict(features)
                signals = preds == 2 # Asumiendo 2 es BUY

            # Arrays numpy para velocidad
            prices = df["close"].values
            highs = df["high"].values
            atrs = df["atr"].values
            dates_arr = df["date"].values
            
            # Loop de Simulación
            t = 0
            while t < len(prices) - 1:
                # 1. Verificar Señal IA
                if signals[t]:
                    current_date = dates_arr[t]
                    
                    # 2. Filtro de Régimen (QQQ > SMA50)
                    is_bullish_regime = True
                    if not regime_df.empty:
                        # Buscar el valor más cercano anterior o igual
                        try:
                            # Asumiendo regime_df indexado por fecha UTC
                            loc_idx = regime_df.index.get_indexer([current_date], method='pad')[0]
                            if loc_idx != -1:
                                is_bullish_regime = regime_df.iloc[loc_idx]["is_bullish"]
                        except:
                            pass # Si falla, asumimos bullish para no bloquear por error técnico
                    
                    if is_bullish_regime:
                        # 3. Ejecutar Operación (Chandelier Exit)
                        result = calculate_chandelier_exit(
                            prices, highs, atrs, t, k=k_atr, max_horizon=max_horizon
                        )
                        
                        entry_price = prices[t]
                        exit_price = result["exit_price"]
                        
                        # Calcular Retorno Neto (con comisiones entrada y salida)
                        gross_return = (exit_price - entry_price) / entry_price
                        net_return = (1 + gross_return) * (1 - commission) * (1 - commission) - 1
                        
                        trades.append({
                            "ticker": ticker,
                            "entry_date": dates_arr[t],
                            "exit_date": dates_arr[result["exit_idx"]],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "return": net_return,
                            "bars_held": result["bars_held"],
                            "exit_reason": result["reason"]
                        })
                        
                        # Saltar hasta el cierre de la operación
                        t = result["exit_idx"]
                    else:
                        t += 1
                else:
                    t += 1
                    
        except Exception as e:
            logger.warning(f"Error en backtest de {ticker}: {e}")
            continue
            
    return pd.DataFrame(trades)

def main():
    parser = argparse.ArgumentParser(description="Backtester Sniper + Chandelier Exit")
    parser.add_argument("--tickers", help="Lista de tickers")
    parser.add_argument("--ticker-file", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--k-atr", type=float, default=3.0, help="Multiplicador ATR para Chandelier Exit")
    parser.add_argument("--max-horizon", type=int, default=20, help="Horizonte máximo de retención")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--start-date", help="YYYY-MM-DD start date")
    parser.add_argument("--end-date", help="YYYY-MM-DD end date")
    
    args = parser.parse_args()
    
    # Cargar Tickers
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

    # Ejecutar Backtest
    results = run_backtest(
        tickers,
        args.data_root,
        model,
        args.min_confidence,
        args.k_atr,
        args.max_horizon,
        start_date,
        end_date,
    )
    
    if results.empty:
        logger.warning("No se generaron operaciones.")
        return
        
    # Análisis de Resultados
    results["entry_date"] = pd.to_datetime(results["entry_date"])
    results = results.sort_values("entry_date")
    
    results["cum_return"] = (1 + results["return"]).cumprod()
    results["equity"] = args.initial_capital * results["cum_return"]
    
    total_trades = len(results)
    win_rate = (results["return"] > 0).mean()
    avg_return = results["return"].mean()
    
    # Sharpe Ratio (Simplificado diario)
    if results["return"].std() > 0:
        sharpe = np.sqrt(252) * (avg_return / results["return"].std())
    else:
        sharpe = 0.0
        
    # Max Drawdown
    rolling_max = results["equity"].cummax()
    drawdown = (results["equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    logger.info("-" * 40)
    logger.info(f"RESULTADOS STRATEGY: XGBoost + Chandelier Exit (k={args.k_atr})")
    logger.info(f"Filtro Régimen: QQQ > SMA50")
    logger.info("-" * 40)
    logger.info(f"Total Operaciones: {total_trades}")
    logger.info(f"Win Rate:          {win_rate:.2%}")
    logger.info(f"Sharpe Ratio:      {sharpe:.2f}")
    logger.info(f"Max Drawdown:      {max_dd:.2%}")
    logger.info(f"Retorno Final:     {(results['equity'].iloc[-1] - args.initial_capital) / args.initial_capital:.2%}")
    logger.info("-" * 40)

    # Guardar CSV
    output_file = "backtest_chandelier_results.csv"
    results.to_csv(output_file, index=False)
    logger.info(f"Resultados detallados guardados en {output_file}")

if __name__ == "__main__":
    main()
