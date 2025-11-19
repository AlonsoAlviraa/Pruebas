#!/usr/bin/env python3
"""
Backtester rápido vectorizado para la estrategia Event-Driven (Sniper).
Simula la ejecución de señales generadas por el modelo ML.
"""
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    dates: np.ndarray,
    t: int,
    horizon: int,
    take_profit: float,
    stop_loss: float,
) -> Dict[str, Any]:
    """Calcula métricas detalladas de una operación iniciada en t."""

    entry_price = prices[t]
    entry_date = dates[t]

    upper = entry_price * (1 + take_profit)
    lower = entry_price * (1 + stop_loss)

    end_idx = min(t + 1 + horizon, len(prices))
    window_prices = prices[t + 1 : end_idx]
    window_dates = dates[t + 1 : end_idx]

    if len(window_prices) == 0:
        return {
            "entry_price": entry_price,
            "entry_date": entry_date,
            "exit_price": entry_price,
            "exit_date": entry_date,
            "gross_return": 0.0,
            "bars_held": 0,
            "exit_reason": "no_data",
        }

    hit_upper = np.where(window_prices >= upper)[0]
    hit_lower = np.where(window_prices <= lower)[0]

    idx_upper = hit_upper[0] if hit_upper.size else len(window_prices) + 1
    idx_lower = hit_lower[0] if hit_lower.size else len(window_prices) + 1

    if idx_upper < idx_lower:
        exit_idx = idx_upper
        exit_reason = "take_profit"
    elif idx_lower < idx_upper:
        exit_idx = idx_lower
        exit_reason = "stop_loss"
    else:
        exit_idx = len(window_prices) - 1
        exit_reason = "horizon"

    exit_price = window_prices[exit_idx]
    exit_date = window_dates[exit_idx]
    gross_return = (exit_price - entry_price) / entry_price

    return {
        "entry_price": entry_price,
        "entry_date": entry_date,
        "exit_price": exit_price,
        "exit_date": exit_date,
        "gross_return": gross_return,
        "bars_held": int(exit_idx + 1),
        "exit_reason": exit_reason,
    }


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


def _inverse_volatility_weights(
    pipeline: DataPipeline, tickers: List[str], lookback: int = 60
) -> Dict[str, float]:
    vols: Dict[str, float] = {}
    for ticker in tickers:
        try:
            frame = pipeline.load_feature_view(ticker, indicators=False)
        except Exception as exc:
            logger.warning("No se pudo calcular la volatilidad de %s: %s", ticker, exc)
            continue
        closes = frame.get("close")
        if closes is None:
            continue
        returns = pd.Series(closes).pct_change().dropna()
        if returns.empty:
            continue
        vol = float(returns.tail(lookback).std())
        if vol and not np.isnan(vol):
            vols[ticker] = max(vol, 1e-6)
    if not vols:
        weight = 1.0 / max(len(tickers), 1)
        return {ticker: weight for ticker in tickers}
    inv_vol = {ticker: 1.0 / vol for ticker, vol in vols.items()}
    total = sum(inv_vol.values())
    return {ticker: inv_vol.get(ticker, 0.0) / total for ticker in tickers}


def run_backtest(
    tickers: List[str],
    data_root: Path,
    model: Any,
    min_confidence: float,
    horizon: int,
    take_profit: float,
    stop_loss: float,
    commission_pct: float,
    slippage_pct: float,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:

    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    trades = []
    weights = _inverse_volatility_weights(pipeline, tickers)
    
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
                    trade = calculate_trade_result(
                        prices,
                        dates,
                        t,
                        horizon,
                        take_profit,
                        stop_loss,
                    )
                    gross = trade["gross_return"]
                    net = gross - commission_pct - slippage_pct

                    trades.append(
                        {
                            "ticker": ticker,
                            "entry_date": trade["entry_date"],
                            "exit_date": trade["exit_date"],
                            "entry_price": trade["entry_price"],
                            "exit_price": trade["exit_price"],
                            "gross_return": gross,
                            "net_return": net,
                            "bars_held": trade["bars_held"],
                            "exit_reason": trade["exit_reason"],
                            "weight": weights.get(ticker, 0.0),
                            "commission_pct": commission_pct,
                            "slippage_pct": slippage_pct,
                        }
                    )

                    t += horizon
                else:
                    t += 1
                    
        except Exception as e:
            logger.warning(f"Error en backtest de {ticker}: {e}")
            continue
            
    return pd.DataFrame(trades)


def _plot_equity_curve(results: pd.DataFrame, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(results["entry_date"], results["equity"], label="Equity")
    plt.fill_between(
        results["entry_date"],
        results["equity"].cummax(),
        results["equity"],
        color="red",
        alpha=0.15,
        label="Drawdown",
    )
    plt.title("Curva de Equidad del Backtest")
    plt.xlabel("Fecha")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


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
    parser.add_argument("--commission-pct", type=float, default=0.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument("--start-date", help="YYYY-MM-DD start date for the evaluation")
    parser.add_argument("--end-date", help="YYYY-MM-DD end date for the evaluation")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("backtest_signal_trades.csv"),
        help="Ruta para guardar el detalle de operaciones",
    )
    parser.add_argument(
        "--equity-plot",
        type=Path,
        default=Path("backtest_signal_equity.png"),
        help="Ruta del gráfico de la curva de equidad",
    )
    
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
        args.commission_pct,
        args.slippage_pct,
        start_date,
        end_date,
    )
    
    if results.empty:
        logger.warning("No se generaron operaciones.")
        return
        
    results["entry_date"] = pd.to_datetime(results["entry_date"])
    results = results.sort_values("entry_date").reset_index(drop=True)

    equity = args.initial_capital
    peak = equity
    equities = []
    drawdowns = []
    portfolio_returns = []
    for _, row in results.iterrows():
        weight = row["weight"] if row["weight"] > 0 else 1.0 / len(tickers)
        equity *= 1 + (weight * row["net_return"])
        peak = max(peak, equity)
        equities.append(equity)
        drawdown = (equity / peak) - 1
        drawdowns.append(drawdown)
        portfolio_returns.append((equity / args.initial_capital) - 1)

    results["equity"] = equities
    results["portfolio_return"] = portfolio_returns
    results["drawdown"] = drawdowns

    total_trades = len(results)
    win_rate = (results["net_return"] > 0).mean()
    expectancy = results["net_return"].mean()
    max_drawdown = results["drawdown"].min()

    logger.info("-" * 40)
    logger.info(f"RESULTADOS DEL BACKTEST ({total_trades} operaciones)")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Expectancy por trade: {expectancy:.4f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    logger.info("Detalle de operaciones guardado en %s", args.results_csv)

    _plot_equity_curve(results, args.equity_plot)
    logger.info("Curva de equidad guardada en %s", args.equity_plot)

if __name__ == "__main__":
    main()