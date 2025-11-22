#!/usr/bin/env python3
"""
Backtest Strategy: XGBoost Signal + Chandelier Exit + Inverse Volatility Sizing.
Includes Point-in-Time Fundamental Validation, Hybrid Time/EMA Exit, and Chronological Cashflow Replay.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

# ---------------------------------------------------------------------------
# Configuración de Rutas y Dependencias
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DataPipeline = None
PipelineConfig = None

try:
    from src.data_pipeline import DataPipeline, PipelineConfig  # type: ignore
except ImportError:
    try:
        from drl_platform.data_pipeline import DataPipeline, PipelineConfig  # type: ignore
    except ImportError:
        try:
            from data_pipeline import DataPipeline, PipelineConfig  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "No se pudieron importar DataPipeline/PipelineConfig. Asegúrate de que estén accesibles."
            ) from exc

try:
    from src.model_utils import load_model  # type: ignore
except ImportError:
    def load_model(model_path: Any) -> Any:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {path}")
        import joblib
        return joblib.load(path)

# ---------------------------------------------------------------------------
# Lógica del Backtest
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_market_regime(start_date: pd.Timestamp, end_date: pd.Timestamp, data_root: Path = Path("data")) -> pd.DataFrame:
    """Determina si el mercado (QQQ) está alcista o bajista (Cierre > SMA 50)."""
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

    qqq["sma_50"] = ta.sma(qqq["close"], length=50)
    qqq["is_bullish"] = qqq["close"] > qqq["sma_50"]
    
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
    emas: np.ndarray,
    start_idx: int,
    k: float = 3.0,
    max_horizon: int = 1250,
    hard_stop_pct: float = 0.07,
    volatility_stop_scale: float = 1.0,
    max_volatility_stop_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Simula operación con Chandelier Exit (Trailing Stop), Hard Stop Dinámico, 
    y Salida Híbrida (Tiempo/EMA).
    """
    entry_price = prices[start_idx]
    initial_atr = atrs[start_idx]
    
    initial_stop = entry_price - (k * initial_atr)
    stop_loss = initial_stop
    highest_high = highs[start_idx]

    # Hard Stop Dinámico: Ajustado por volatilidad (ATR/Precio)
    atr_pct = 0.0
    if entry_price > 0:
        atr_pct = (initial_atr / entry_price) * volatility_stop_scale
    
    atr_pct = np.clip(atr_pct, 0.0, max_volatility_stop_pct)
    dynamic_hard_stop_pct = max(0.0, hard_stop_pct + atr_pct)
    hard_stop_price = entry_price * (1 - dynamic_hard_stop_pct)
    
    bars_held = 0
    while True:
        bars_held += 1
        curr_idx = start_idx + bars_held
        
        if curr_idx >= len(prices):
            # Salida por final de datos
            return {
                "exit_price": prices[-1], "exit_idx": len(prices) - 1, "reason": "end_of_data",
                "bars_held": bars_held, "initial_stop": initial_stop,
                "hard_stop_price": hard_stop_price, "dynamic_hard_stop_pct": dynamic_hard_stop_pct,
            }
            
        curr_price = prices[curr_idx]
        curr_high = highs[curr_idx]
        curr_atr = atrs[curr_idx]
        
        active_stop = max(stop_loss, hard_stop_price)
        
        if curr_price < active_stop:
            # Salida por Stop Loss (Trailing o Hard Stop)
            reason = "hard_stop" if np.isclose(active_stop, hard_stop_price) else "stop_loss"
            return {
                "exit_price": active_stop, "exit_idx": curr_idx, "reason": reason,
                "bars_held": bars_held, "initial_stop": initial_stop,
                "hard_stop_price": hard_stop_price, "dynamic_hard_stop_pct": dynamic_hard_stop_pct,
            }
            
        # Actualizar Highest High para el Trailing Stop (Chandelier)
        if curr_high > highest_high:
            highest_high = curr_high
        
        new_stop = highest_high - (k * curr_atr)
        stop_loss = max(stop_loss, new_stop)

       


def _load_fundamentals(ticker: str, data_root: Path, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Carga y cachea datos fundamentales para un ticker."""
    if ticker in cache: return cache[ticker]

    fundamentals_path = data_root / f"{ticker}_fundamentals.csv"
    if not fundamentals_path.exists():
        cache[ticker] = pd.DataFrame()
        return cache[ticker]

    df = pd.read_csv(fundamentals_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    for col in ["as_of", "available_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    # Ordenar por la fecha en que los datos estaban disponibles
    if "available_at" in df.columns:
        df = df.sort_values("available_at")
    
    cache[ticker] = df
    return df


def validate_fundamentals_at_date(
    ticker: str,
    current_simulation_date: Any,
    min_growth_pct: float,
    fundamentals_cache: Dict[str, pd.DataFrame],
    data_root: Path,
) -> bool:
    """Valida crecimiento fundamental Point-in-Time para evitar Look-Ahead Bias."""
    df = _load_fundamentals(ticker, data_root, fundamentals_cache)
    if df.empty: return False

    current_ts = pd.to_datetime(current_simulation_date, utc=True)
    
    # Solo considerar fundamentales *ya* disponibles en la fecha actual de la simulación
    if "available_at" not in df.columns: return False
    df_available = df[df["available_at"] <= current_ts]
    
    if len(df_available) < 2: return False # Necesitamos al menos dos puntos para calcular crecimiento

    latest = df_available.iloc[-1]
    previous = df_available.iloc[-2]

    # Usar 'revenue' o 'eps' como métrica principal de crecimiento
    metric_col = "revenue" if "revenue" in df_available.columns else "eps" if "eps" in df_available.columns else None
    if metric_col is None: return False

    prev_value = previous[metric_col]
    latest_value = latest[metric_col]
    
    if pd.isna(prev_value) or pd.isna(latest_value) or prev_value == 0:
        return False

    growth_pct = (latest_value - prev_value) / abs(prev_value)
    return growth_pct >= min_growth_pct


def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    """Convierte un string de fecha a pd.Timestamp con UTC."""
    if not value: return None
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
    ema_length: int,
    min_growth_pct: float,
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
    
    if DataPipeline is None: # type: ignore
        raise ImportError("DataPipeline no inicializado.")

    pipeline = DataPipeline(PipelineConfig(data_root=data_root)) # type: ignore
    trades = []
    fundamentals_cache: Dict[str, pd.DataFrame] = {}
    
    volatility_target_pct = max(0.0, volatility_target_pct)
    volatility_exponent = max(0.0, volatility_exponent)
    max_position_pct = max(0.0, min(1.0, max_position_pct))
    
    # ATENCIÓN: Esta 'equity' se usa para el sizing en la fase de GENERACIÓN de trades, 
    # y se asume el capital inicial. El control de liquidez REAL se hace en el 
    # CASH FLOW REPLAY en la función 'main'.
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
            dates_arr = df["date"].values
            
            t = 0
            while t < len(prices) - 1:
                if signals[t]:
                    current_date = dates_arr[t]
                    
                    # 1. Validación Fundamental Point-in-Time
                    fundamentals_ok = validate_fundamentals_at_date(
                        ticker, current_date, min_growth_pct, fundamentals_cache, data_root
                    )
                    if not fundamentals_ok:
                        t += 1
                        continue

                    # 2. Filtro Régimen de Mercado
                    is_bullish_regime = True
                    if not regime_df.empty:
                        try:
                            # 'pad' para obtener el régimen anterior si no hay un match exacto
                            loc_idx = regime_df.index.get_indexer([current_date], method='pad')[0]
                            if loc_idx != -1:
                                is_bullish_regime = regime_df.iloc[loc_idx]["is_bullish"]
                        except:
                            pass 
                        
                    if is_bullish_regime:
                        # Determinar Stop Loss y Horizonte de Salida
                        result = calculate_chandelier_exit(
                            prices, highs, atrs, emas, t,
                            k=k_atr, max_horizon=max_horizon,
                            hard_stop_pct=hard_stop_pct,
                            volatility_stop_scale=volatility_stop_scale,
                            max_volatility_stop_pct=max_volatility_stop_pct,
                        )

                        entry_price = prices[t]
                        exit_price = result["exit_price"]
                        
                        if entry_price <= 0: 
                            t += 1
                            continue
                        
                        entry_stop = result.get("initial_stop", entry_price)
                        hard_stop_price = result.get("hard_stop_price", entry_stop)
                        dynamic_hard_stop_pct = result.get("dynamic_hard_stop_pct", hard_stop_pct)

                        # --- LÓGICA: INVERSE VOLATILITY SIZING ---
                        atr_value = atrs[t]
                        volatility_current = 0.0
                        if entry_price > 0:
                            volatility_current = atr_value / entry_price
                        volatility_current = max(volatility_current, 1e-8)

                        # 1. Base Sizing: Target Volatility
                        allocation_dollars = equity * volatility_target_pct
                        allocation_dollars /= max(volatility_current ** volatility_exponent, 1e-8)
                        
                        # 2. Hard Caps (basados en capital inicial, optimista)
                        max_capital_dollars = max(0.0, min(equity * max_position_pct, equity))
                        capped_allocation = max(0.0, min(allocation_dollars, max_capital_dollars))

                        # 3. Cálculo de Posición
                        shares_by_allocation = capped_allocation / entry_price if entry_price > 0 else 0.0
                        shares_by_cash_cap = max_capital_dollars / entry_price if entry_price > 0 else 0.0
                        
                        # El tamaño de posición es el mínimo de los límites
                        share_candidates = [shares_by_allocation, shares_by_cash_cap]
                        share_candidates = [s for s in share_candidates if np.isfinite(s) and s > 0]
                        position_size = int(min(share_candidates)) if share_candidates else 0
                        
                        if position_size <= 0:
                            t += 1
                            continue

                        # --- REGISTRO DEL TRADE PARA REPLAY ---
                        invested_capital = position_size * entry_price

                        entry_comm = position_size * entry_price * commission
                        exit_comm = position_size * exit_price * commission
                        gross_profit = position_size * (exit_price - entry_price)
                        net_profit = gross_profit - (entry_comm + exit_comm)
                        
                        trade_ret = net_profit / invested_capital if invested_capital > 0 else 0.0

                        exit_idx = min(result["exit_idx"], len(dates_arr) - 1)

                        trades.append({
                            "ticker": ticker,
                            "entry_date": dates_arr[t],
                            "exit_date": dates_arr[exit_idx],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "initial_stop": entry_stop,
                            "hard_stop_price": hard_stop_price,
                            "hard_stop_pct": dynamic_hard_stop_pct,
                            "shares": position_size,
                            "capital_used": invested_capital,
                            "net_profit": net_profit,
                            "return": trade_ret,
                            "bars_held": result["bars_held"],
                            "exit_reason": result["reason"],
                            "equity_after": np.nan,  # Se calcula en el replay loop
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
    parser.add_argument("--k-atr", type=float, default=3.0, help="Multiplicador ATR")
    parser.add_argument("--max-horizon", type=int, default=20, help="Horizonte máximo")
    
    # Argumentos Nuevos (EMA y Fundamentales)
    parser.add_argument("--ema-length", type=int, default=20, help="Longitud de la EMA para extender horizonte")
    parser.add_argument("--min-growth-pct", type=float, default=0.05, help="Crecimiento mínimo QoQ/YoY requerido")
    
    # Gestión de Capital
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--volatility-target-pct", type=float, default=0.005)
    parser.add_argument("--volatility-exponent", type=float, default=1.0)
    parser.add_argument("--max-position-pct", type=float, default=0.25, help="Límite máximo de capital por operación respecto al equity (Hard Cap)")
    
    # Stops
    parser.add_argument("--hard-stop-pct", type=float, default=0.07)
    parser.add_argument("--volatility-stop-scale", type=float, default=1.0)
    parser.add_argument("--max-volatility-stop-pct", type=float, default=0.05)
    parser.add_argument("--commission", type=float, default=0.001)
    
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    tickers = []
    if args.tickers: tickers.extend([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    if args.ticker_file and args.ticker_file.exists():
        content = args.ticker_file.read_text(encoding="utf-8").splitlines()
        tickers.extend([t.strip().upper() for t in content if t.strip()])
    if not tickers:
        files = list(args.data_root.glob("*_history.csv"))
        tickers = [f.name.replace("_history.csv", "") for f in files]
        
    model = load_model(args.model_path)
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    # Paso 1: Generación de trades potenciales (con sizing optimista)
    results = run_backtest(
        tickers, args.data_root, model, args.min_confidence, args.k_atr, args.max_horizon,
        args.ema_length, args.min_growth_pct, start_date, end_date, args.initial_capital,
        args.volatility_target_pct, args.volatility_exponent, args.max_position_pct, args.hard_stop_pct,
        args.volatility_stop_scale, args.max_volatility_stop_pct, args.commission,
    )
    
    if results.empty:
        logger.warning("No se generaron operaciones.")
        return
        
    results["entry_date"] = pd.to_datetime(results["entry_date"])
    results["exit_date"] = pd.to_datetime(results["exit_date"])

    # ------------------------------------------------------------------
    # CASH FLOW REPLAY (Soluciona el Bug Millonario)
    # Re-ejecutar cronológicamente para control de liquidez.
    # ------------------------------------------------------------------
    results = results.sort_values("entry_date").reset_index(drop=True)
    
    # Inicialización para el replay (MODIFICADO)
    cash = args.initial_capital
    open_exposure = 0.0 # Capital actualmente comprometido en posiciones abiertas (NUEVO)
    equity_history: List[Dict[str, Any]] = []
    accepted_trades: List[Dict[str, Any]] = []
    
    # Heap de eventos de salida: (fecha_salida, idx_trade, cash_delta, capital_usado) (MODIFICADO)
    exit_events: List[Tuple[Any, int, float, float]] = []
    
    # Registrar punto inicial (MODIFICADO)
    if not results.empty:
        total_equity = cash + open_exposure
        equity_history.append({"date": results["entry_date"].min(), "equity": total_equity})

    for idx, trade in results.iterrows():
        # 1. Procesar salidas pendientes (liberar cash)
        # Se liberan fondos de operaciones ya cerradas cuya fecha de salida es anterior o igual a la entrada actual
        while exit_events and exit_events[0][0] <= trade["entry_date"]:
            exit_date, trade_idx, cash_delta, capital_used = exit_events.pop(0)
            
            # Actualizar Exposición Abierta (NUEVO)
            open_exposure = max(0.0, open_exposure - capital_used)
            
            # Actualizar Cash y Equity Total (MODIFICADO)
            cash += cash_delta
            total_equity = cash + open_exposure
            equity_history.append({"date": exit_date, "equity": total_equity})
            if 0 <= trade_idx < len(accepted_trades):
                accepted_trades[trade_idx]["equity_after"] = total_equity

        # 2. Intentar la nueva entrada
        capital_required = trade.get("capital_used", 0.0)
        
        # EL CHECK CLAVE: Si el cash actual es insuficiente, descartamos la orden
        if capital_required <= 0 or cash < capital_required:
            continue

        # Compra: Reducir caja por el capital usado (MODIFICADO)
        cash -= capital_required
        open_exposure += capital_required
        total_equity = cash + open_exposure
        
        equity_history.append({"date": trade["entry_date"], "equity": total_equity})

        # Aceptar y registrar el trade
        trade_record = trade.to_dict()
        trade_record["equity_after"] = np.nan
        accepted_idx = len(accepted_trades)
        accepted_trades.append(trade_record)

        # 3. Planear la salida: Se programa el evento de liberación de capital (MODIFICADO)
        cash_delta_on_exit = capital_required + trade.get("net_profit", 0.0)
        # Se añade el capital_required para poder descontarlo de open_exposure en la salida
        exit_events.append((trade["exit_date"], accepted_idx, cash_delta_on_exit, capital_required))
        exit_events.sort(key=lambda x: x[0]) # Mantener orden cronológico

    # 4. Procesar eventos finales de salida (MODIFICADO)
    for exit_date, trade_idx, cash_delta, capital_used in exit_events:
        open_exposure = max(0.0, open_exposure - capital_used)
        cash += cash_delta
        total_equity = cash + open_exposure
        
        equity_history.append({"date": exit_date, "equity": total_equity})
        if 0 <= trade_idx < len(accepted_trades):
            accepted_trades[trade_idx]["equity_after"] = total_equity

    results = pd.DataFrame(accepted_trades)
    if results.empty:
        logger.warning("No se generaron operaciones tras aplicar control de liquidez cronológico.")
        return

    equity_df = pd.DataFrame(equity_history)
    equity_df = equity_df.sort_values("date")
    # Eliminar duplicados de fecha manteniendo el último valor (el más actualizado)
    equity_df = equity_df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    
    results = results.reset_index(drop=True)
    results["cumulative_profit"] = results.get("net_profit", 0).cumsum()
    
    # Rellenar los valores de equity faltantes (entre trades)
    equity_series = equity_df["equity"] if not equity_df.empty else pd.Series([args.initial_capital])
    # Rellenar NANs con el valor anterior/siguiente conocido de capital final
    results["equity_after"] = results["equity_after"].fillna(method="ffill").fillna(method="bfill")
    final_equity = equity_series.iloc[-1]
    
    # La columna 'equity' debe contener el capital final después de la operación (el valor rellenado)
    results["equity"] = results["equity_after"].fillna(final_equity)
    
    # ------------------------------------------------------------------
    # CÁLCULOS FINALES
    # ------------------------------------------------------------------
    
    total_trades = len(results)
    final_return_pct = (final_equity - args.initial_capital) / args.initial_capital
    win_rate = (results["net_profit"] > 0).mean()
    avg_return = results["return"].mean()
    
    # Sharpe Ratio (Asumiendo 252 días de trading)
    sharpe = 0.0
    if results["return"].std() > 0:
        sharpe = np.sqrt(252) * (avg_return / results["return"].std())
        
    # Max Drawdown (Calculado sobre la serie de equity generada cronológicamente)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min()

    output_file = "backtest_results.csv"
    results.to_csv(output_file, index=False)

    logger.info("-" * 50)
    logger.info(f"RESULTADOS: XGBoost + Inverse Vol + Fundamentales Point-in-Time")
    logger.info("-" * 50)
    logger.info(f"Total Operaciones: {total_trades}")
    logger.info(f"Win Rate:          {win_rate:.2%}")
    logger.info(f"Sharpe Ratio:      {sharpe:.2f}")
    logger.info(f"Max Drawdown:      {max_dd:.2%}")
    logger.info(f"Capital Final:     ${final_equity:,.2f}")
    logger.info(f"Retorno Total:     {final_return_pct:.2%}")
    logger.info(f"Detalles en:       {output_file}")
    logger.info("-" * 50)

if __name__ == "__main__":
    main()