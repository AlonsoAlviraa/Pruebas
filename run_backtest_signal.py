#!/usr/bin/env python3
"""
Backtest Strategy: XGBoost Signal + Chandelier Exit + Inverse Volatility Sizing.
Includes 'Alpha' dampening factor to control aggression on volatile assets.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ajusta estos imports según la estructura de tu proyecto
try:
    from src.data_pipeline import DataPipeline, PipelineConfig
    from src.model_utils import load_model
except ImportError:
    # Fallback para evitar errores de linter si no existen los módulos
    from sys import exit
    print("Error: No se encuentran los módulos 'src'. Asegúrate de estar en la raíz del proyecto.")

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_market_regime(start_date: pd.Timestamp, end_date: pd.Timestamp, data_root: Path = Path("data")) -> pd.DataFrame:
    """
    Determina si el mercado (QQQ) está alcista o bajista.
    Regla: Precio > SMA 50.
    """
    qqq_path = data_root / "QQQ_history.csv"
    if not qqq_path.exists():
        logger.warning("QQQ_history.csv no encontrado. Saltando filtro de régimen.")
        return pd.DataFrame()

    qqq = pd.read_csv(qqq_path)
    qqq.columns = [c.lower() for c in qqq.columns]
    
    if "date" in qqq.columns:
        qqq["date"] = pd.to_datetime(qqq["date"], utc=True)
        qqq.set_index("date", inplace=True)
        qqq.sort_index(inplace=True)

    if qqq.empty or "close" not in qqq.columns:
        logger.warning("No hay datos de QQQ. Filtro de régimen DESACTIVADO.")
        return pd.DataFrame()

    # Calcular SMA 50
    qqq["sma_50"] = ta.sma(qqq["close"], length=50)
    qqq["is_bullish"] = qqq["close"] > qqq["sma_50"]
    
    # Filtrar rango solicitado
    if start_date.tzinfo is None: start_date = start_date.tz_localize("UTC")
    if end_date.tzinfo is None: end_date = end_date.tz_localize("UTC")
    
    try:
        qqq = qqq.loc[start_date : end_date].copy()
    except KeyError:
        pass 
        
    return qqq[["is_bullish"]]


def calculate_chandelier_exit(
    prices: np.ndarray,
    highs: np.ndarray,
    atrs: np.ndarray,
    start_idx: int,
    k: float = 3.0,
    max_horizon: int = 20,
    hard_stop_pct: float = 0.07,
    volatility_stop_scale: float = 1.0,
    max_volatility_stop_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Simula una operación con doble mecanismo de salida:
    1. Chandelier Exit (Trailing Stop dinámico).
    2. Hard Stop (Stop de emergencia basado en volatilidad).
    """
    entry_price = prices[start_idx]
    initial_atr = atrs[start_idx]
    
    # --- 1. Configuración de Stops ---
    initial_stop = entry_price - (k * initial_atr)
    stop_loss = initial_stop
    highest_high = highs[start_idx]

    # Hard Stop Dinámico
    atr_pct = 0.0
    if entry_price > 0:
        atr_pct = (initial_atr / entry_price) * volatility_stop_scale
    
    atr_pct = np.clip(atr_pct, 0.0, max_volatility_stop_pct)
    
    dynamic_hard_stop_pct = max(0.0, hard_stop_pct + atr_pct)
    hard_stop_price = entry_price * (1 - dynamic_hard_stop_pct)
    
    # --- 2. Simulación Barra a Barra ---
    for i in range(1, max_horizon + 1):
        curr_idx = start_idx + i
        if curr_idx >= len(prices):
            return {
                "exit_price": prices[-1],
                "exit_idx": len(prices) - 1,
                "reason": "end_of_data",
                "bars_held": i,
                "initial_stop": initial_stop,
                "hard_stop_price": hard_stop_price,
                "dynamic_hard_stop_pct": dynamic_hard_stop_pct,
            }
            
        curr_price = prices[curr_idx]
        curr_high = highs[curr_idx]
        curr_atr = atrs[curr_idx]
        
        # Stop activo: el precio más alto entre el trailing y el hard stop
        active_stop = max(stop_loss, hard_stop_price)
        
        if curr_price < active_stop:
            reason = "hard_stop" if np.isclose(active_stop, hard_stop_price) else "stop_loss"
            return {
                "exit_price": active_stop,
                "exit_idx": curr_idx,
                "reason": reason,
                "bars_held": i,
                "initial_stop": initial_stop,
                "hard_stop_price": hard_stop_price,
                "dynamic_hard_stop_pct": dynamic_hard_stop_pct,
            }
            
        if curr_high > highest_high:
            highest_high = curr_high
        
        new_stop = highest_high - (k * curr_atr)
        stop_loss = max(stop_loss, new_stop)
        
    return {
        "exit_price": prices[start_idx + max_horizon],
        "exit_idx": start_idx + max_horizon,
        "reason": "horizon_limit",
        "bars_held": max_horizon,
        "initial_stop": initial_stop,
        "hard_stop_price": hard_stop_price,
        "dynamic_hard_stop_pct": dynamic_hard_stop_pct,
    }


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
    initial_capital: float,
    volatility_target_pct: float,
    volatility_exponent: float,
    max_position_pct: float,
    hard_stop_pct: float,
    volatility_stop_scale: float,
    max_volatility_stop_pct: float,
    commission: float = 0.001,
) -> pd.DataFrame:
    
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    trades = []
    
    # Inicialización de Gestión de Capital
    volatility_target_pct = max(0.0, volatility_target_pct)
    volatility_exponent = max(0.0, volatility_exponent)
    max_position_pct = max(0.0, min(1.0, max_position_pct))
    equity = initial_capital
    
    regime_df = pd.DataFrame()
    if start_date and end_date:
        try:
            regime_df = get_market_regime(start_date, end_date, data_root)
        except Exception as e:
            logger.warning(f"Error obteniendo régimen de mercado: {e}")

    logger.info(f"Iniciando backtest para {len(tickers)} tickers...")
    logger.info(f"Estrategia: Inverse Volatility (Target={volatility_target_pct:.2%}, Alpha={volatility_exponent})")
    
    for ticker in tickers:
        try:
            df = pipeline.load_feature_view(ticker, indicators=True)
            if df.empty: continue
            
            dates = pd.to_datetime(df["date"], utc=True)
            mask = pd.Series(True, index=df.index)
            if start_date: mask &= dates >= start_date
            if end_date: mask &= dates <= end_date
            df = df.loc[mask].copy()
            
            if df.empty: continue

            feature_cols = [c for c in df.columns if c not in ["date", "ticker", "target", "open", "high", "low", "close", "volume", "atr"]]
            if hasattr(model, "feature_names_in_"):
                 X = df[feature_cols].reindex(columns=model.feature_names_in_, fill_value=0)
            else:
                 X = df[feature_cols]

            preds_proba = model.predict_proba(X)[:, 1]
            signals = preds_proba >= min_confidence
            
            prices = df["close"].values
            highs = df["high"].values
            atrs = df["atr"].values
            dates_arr = df["date"].values
            
            t = 0
            while t < len(prices) - 1:
                if signals[t]:
                    current_date = dates_arr[t]
                    
                    is_bullish_regime = True
                    if not regime_df.empty:
                        try:
                            loc_idx = regime_df.index.get_indexer([current_date], method='pad')[0]
                            if loc_idx != -1:
                                is_bullish_regime = regime_df.iloc[loc_idx]["is_bullish"]
                        except:
                            pass 
                    
                    if is_bullish_regime:
                        result = calculate_chandelier_exit(
                            prices, highs, atrs, t,
                            k=k_atr, max_horizon=max_horizon,
                            hard_stop_pct=hard_stop_pct,
                            volatility_stop_scale=volatility_stop_scale,
                            max_volatility_stop_pct=max_volatility_stop_pct,
                        )

                        entry_price = prices[t]
                        exit_price = result["exit_price"]
                        
                        entry_stop = result.get("initial_stop", entry_price)
                        hard_stop_price = result.get("hard_stop_price", entry_stop)
                        dynamic_hard_stop_pct = result.get("dynamic_hard_stop_pct", hard_stop_pct)

                        # --- NUEVA LÓGICA: INVERSE VOLATILITY SIZING ---
                        atr_value = atrs[t]
                        volatility_current = 0.0
                        if entry_price > 0:
                            volatility_current = atr_value / entry_price
                        volatility_current = max(volatility_current, 1e-8)

                        # Fórmula Maestra con Factor Alpha
                        allocation_dollars = equity * volatility_target_pct
                        # Aplicamos el exponente (Alpha) aquí:
                        allocation_dollars /= max(volatility_current ** volatility_exponent, 1e-8)
                        
                        # Aplicar Límites (Safety Caps)
                        max_capital = equity * max_position_pct
                        allocation_dollars = max(0.0, min(allocation_dollars, max_capital, equity))

                        position_size = int(allocation_dollars / entry_price)
                        
                        if position_size == 0:
                            t += 1
                            continue

                        # Cálculo PnL
                        entry_commission = position_size * entry_price * commission
                        exit_commission = position_size * exit_price * commission
                        gross_profit = position_size * (exit_price - entry_price)
                        net_profit = gross_profit - (entry_commission + exit_commission)
                        
                        invested_capital = position_size * entry_price
                        trade_return = net_profit / invested_capital if invested_capital > 0 else 0.0

                        equity += net_profit

                        trades.append({
                            "ticker": ticker,
                            "entry_date": dates_arr[t],
                            "exit_date": dates_arr[result["exit_idx"]],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "initial_stop": entry_stop,
                            "hard_stop_price": hard_stop_price,
                            "hard_stop_pct": dynamic_hard_stop_pct,
                            "shares": position_size,
                            "capital_used": invested_capital,
                            "net_profit": net_profit,
                            "return": trade_return,
                            "bars_held": result["bars_held"],
                            "exit_reason": result["reason"],
                            "equity_after": equity,
                        })
                        
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
    parser = argparse.ArgumentParser(description="Backtester Sniper + Inverse Volatility Sizing")
    parser.add_argument("--tickers", help="Lista de tickers separados por coma")
    parser.add_argument("--ticker-file", type=Path, help="Archivo TXT con tickers")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--k-atr", type=float, default=3.0, help="Multiplicador ATR para Chandelier Exit")
    parser.add_argument("--max-horizon", type=int, default=20, help="Horizonte máximo de retención")
    
    # Argumentos de Gestión de Capital (NUEVOS)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--volatility-target-pct", type=float, default=0.005, help="Volatilidad objetivo por posición (por defecto 0.5%)")
    parser.add_argument("--volatility-exponent", type=float, default=1.0, help="Exponente Alpha: 1.0=Conservador, 0.5=Agresivo/Growth, 0.0=Igualdad")
    parser.add_argument("--max-position-pct", type=float, default=0.25, help="Límite máximo de capital por operación respecto al equity (Hard Cap)")
    
    # Argumentos de Stop Loss
    parser.add_argument("--hard-stop-pct", type=float, default=0.07, help="Stop porcentual de emergencia base")
    parser.add_argument("--volatility-stop-scale", type=float, default=1.0, help="Factor multiplicador para ATR/Precio al ampliar el hard stop")
    parser.add_argument("--max-volatility-stop-pct", type=float, default=0.05, help="Límite extra que la volatilidad puede añadir al stop")
    parser.add_argument("--commission", type=float, default=0.001, help="Comisión por lado (defecto: 0.1%)")
    
    parser.add_argument("--start-date", help="YYYY-MM-DD start date")
    parser.add_argument("--end-date", help="YYYY-MM-DD end date")
    
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
        args.k_atr,
        args.max_horizon,
        start_date,
        end_date,
        args.initial_capital,
        args.volatility_target_pct,
        args.volatility_exponent,
        args.max_position_pct,
        args.hard_stop_pct,
        args.volatility_stop_scale,
        args.max_volatility_stop_pct,
        args.commission,
    )
    
    if results.empty:
        logger.warning("No se generaron operaciones.")
        return
        
    results["entry_date"] = pd.to_datetime(results["entry_date"])
    results = results.sort_values("entry_date")
    
    if "equity_after" not in results.columns:
         results["equity_after"] = args.initial_capital * (1 + results["return"]).cumprod()

    results["equity"] = results["equity_after"]
    results["cum_return"] = results["equity"] / args.initial_capital
    
    total_trades = len(results)
    win_rate = (results["return"] > 0).mean()
    avg_return = results["return"].mean()
    
    sharpe = 0.0
    if results["return"].std() > 0:
        sharpe = np.sqrt(252) * (avg_return / results["return"].std())
        
    rolling_max = results["equity"].cummax()
    drawdown = (results["equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    output_file = "backtest_results.csv"
    results.to_csv(output_file, index=False)

    logger.info("-" * 50)
    logger.info(f"RESULTADOS: XGBoost + Inverse Volatility Sizing (Alpha={args.volatility_exponent})")
    logger.info("-" * 50)
    logger.info(f"Total Operaciones: {total_trades}")
    logger.info(f"Win Rate:          {win_rate:.2%}")
    logger.info(f"Sharpe Ratio:      {sharpe:.2f}")
    logger.info(f"Max Drawdown:      {max_dd:.2%}")
    logger.info(f"Capital Final:     ${results['equity'].iloc[-1]:,.2f}")
    logger.info(f"Retorno Total:     {(results['equity'].iloc[-1] - args.initial_capital) / args.initial_capital:.2%}")
    logger.info(f"Detalles en:       {output_file}")
    logger.info("-" * 50)

if __name__ == "__main__":
    main()