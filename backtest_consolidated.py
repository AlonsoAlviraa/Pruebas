#!/usr/bin/env python3
"""
BACKTEST VECTORIZADO CON VECTORBT - CAPITAL CONSOLIDADO
Portfolio unificado de $10K distribuido entre todos los tickers
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any

# VectorBT imports
try:
    import vectorbt as vbt
except ImportError:
    print("ERROR: VectorBT no instalado. Ejecuta: pip install vectorbt")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from drl_platform.data_pipeline import DataPipeline, PipelineConfig


def filter_existing_tickers(tickers: List[str], data_root: Path) -> List[str]:
    """Filtra solo tickers que tienen archivo _history.csv"""
    existing = []
    for ticker in tickers:
        history_file = data_root / f"{ticker}_history.csv"
        if history_file.exists():
            existing.append(ticker)
    return existing


def load_all_data_vectorized(
    tickers: List[str],
    data_root: Path,
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """Carga datos de múltiples tickers en paralelo"""
    existing_tickers = filter_existing_tickers(tickers, data_root)
    
    print(f"Cargando {len(existing_tickers)} tickers...", end=" ", flush=True)
    
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    all_data = {}
    
    for ticker in existing_tickers:
        try:
            df = pipeline.load_feature_view(ticker, indicators=True)
            if df.empty:
                continue
            
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if len(df) < 50:
                continue
            
            df = df.set_index('date')
            all_data[ticker] = df
            
        except Exception:
            continue
    
    print(f"OK ({len(all_data)} cargados)")
    return all_data


def generate_signals_vectorized(
    all_data: Dict[str, pd.DataFrame],
    model: Any,
    min_confidence: float = 0.50
) -> pd.DataFrame:
    """Genera señales de trading para TODOS los tickers"""
    signals_dict = {}
    
    print("Generando señales...", end=" ", flush=True)
    
    for ticker, df in all_data.items():
        try:
            df['ma_10'] = df['close'].rolling(10).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['ma_50'] = df['close'].rolling(50).mean()
            df['ret_1m'] = df['close'].pct_change(21)
            
            feature_cols = [c for c in df.columns if c not in [
                "ticker", "target", "open", "high", "low", "close", "volume",
                "atr", "ma_10", "ma_20", "ma_50", "ret_1m"
            ]]
            
            if hasattr(model, 'feature_names_in_'):
                X = df[feature_cols].reindex(columns=model.feature_names_in_, fill_value=0)
            else:
                X = df[feature_cols]
            
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            preds_proba = model.predict_proba(X)[:, 1]
            
            trend_filter = df['close'] > df['ma_50']
            momentum_filter = df['ret_1m'] >= 0.03
            
            signals = (
                (preds_proba >= min_confidence) &
                trend_filter.values &
                momentum_filter.values
            )
            
            signals_dict[ticker] = pd.Series(signals, index=df.index)
            
        except Exception:
            continue
    
    print(f"OK ({len(signals_dict)} tickers)")
    return pd.DataFrame(signals_dict).fillna(False).astype(bool)


def run_consolidated_backtest(
    tickers: List[str],
    data_root: Path,
    model_path: Path,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    min_confidence: float = 0.50,
    init_cash: float = 10000.0,
    fees: float = 0.001,
    max_positions: int = 10  # Máximo de posiciones simultáneas
):
    """
    Backtest CONSOLIDADO con capital único de $10K
    Distribuye capital entre señales, no por ticker
    """
    print(f"\n[Backtest] Capital: ${init_cash:,.0f} | Max posiciones: {max_positions}")
    
    # 1. Cargar modelo y datos
    print(f"Cargando modelo...", end=" ", flush=True)
    model = joblib.load(model_path)
    print("OK")
    
    all_data = load_all_data_vectorized(tickers, data_root, start_date, end_date)
    
    if not all_data:
        print("ERROR: No se cargaron datos")
        return None
    
    # 2. Crear DataFrames de precios
    close_dict = {ticker: df['close'] for ticker, df in all_data.items()}
    close_df = pd.DataFrame(close_dict)
    
    # 3. Generar señales
    entries_df = generate_signals_vectorized(all_data, model, min_confidence)
    
    # 4. BACKTEST CON CAPITAL CONSOLIDADO
    print("Ejecutando backtest...", end=" ", flush=True)
    
    # Simular manualmente para tener control total del capital
    cash = init_cash
    positions = {}  # {ticker: {shares, entry_price, entry_date}}
    trades = []
    equity_curve = [init_cash]
    dates = close_df.index
    
    for date in dates:
        # 1. Actualizar equity con posiciones actuales
        position_value = sum(
            pos['shares'] * close_df.loc[date, ticker]
            for ticker, pos in positions.items()
            if ticker in close_df.columns and date in close_df.index
        )
        current_equity = cash + position_value
        equity_curve.append(current_equity)
        
        # 2. Buscar señales de entrada
        if len(positions) < max_positions:
            # Obtener señales para este día
            signals_today = entries_df.loc[date] if date in entries_df.index else pd.Series()
            
            # Filtrar tickers ya en posición
            available_signals = signals_today[
                (signals_today == True) & 
                (~signals_today.index.isin(positions.keys()))
            ]
            
            # Entrar en nuevas posiciones
            for ticker in available_signals.index[:max_positions - len(positions)]:
                if ticker not in close_df.columns or date not in close_df.index:
                    continue
                
                price = close_df.loc[date, ticker]
                if pd.isna(price) or price <= 0:
                    continue
                
                # Calcular tamaño de posición
                position_size_dollars = current_equity / max_positions
                shares = int(position_size_dollars / price)
                
                if shares > 0 and shares * price <= cash:
                    cost = shares * price * (1 + fees)
                    
                    if cost <= cash:
                        cash -= cost
                        positions[ticker] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': date,
                            'cost': cost
                        }
        
        # 3. Verificar salidas (cuando señal desaparece o fin de período)
        tickers_to_exit = []
        for ticker in list(positions.keys()):
            # Salir si ya no hay señal o es el último día
            has_signal = (
                date in entries_df.index and 
                ticker in entries_df.columns and 
                entries_df.loc[date, ticker]
            )
            
            is_last_day = (date == dates[-1])
            
            if not has_signal or is_last_day:
                tickers_to_exit.append(ticker)
        
        for ticker in tickers_to_exit:
            pos = positions[ticker]
            exit_price = close_df.loc[date, ticker]
            
            if pd.isna(exit_price):
                continue
            
            proceeds = pos['shares'] * exit_price * (1 - fees)
            cash += proceeds
            
            profit = proceeds - pos['cost']
            ret = profit / pos['cost'] if pos['cost'] > 0 else 0
            
            trades.append({
                'ticker': ticker,
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'cost': pos['cost'],
                'proceeds': proceeds,
                'profit': profit,
                'return': ret
            })
            
            del positions[ticker]
    
    # Cerrar posiciones restantes al final
    if len(dates) > 0:
        final_date = dates[-1]
        for ticker, pos in list(positions.items()):
            if ticker in close_df.columns and final_date in close_df.index:
                exit_price = close_df.loc[final_date, ticker]
                proceeds = pos['shares'] * exit_price * (1 - fees)
                cash += proceeds
                
                profit = proceeds - pos['cost']
                
                trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': final_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'cost': pos['cost'],
                    'proceeds': proceeds,
                    'profit': profit,
                    'return': profit / pos['cost'] if pos['cost'] > 0 else 0
                })
    
    # Calcular métricas
    trades_df = pd.DataFrame(trades)
    final_equity = cash
    
    results = {
        'trades_df': trades_df,
        'equity_curve': np.array(equity_curve),
        'final_equity': final_equity,
        'init_cash': init_cash,
        'total_trades': len(trades_df),
        'total_return': (final_equity - init_cash) / init_cash if init_cash > 0 else 0,
        'net_profit': final_equity - init_cash
    }
    
    if len(trades_df) > 0:
        results['win_rate'] = (trades_df['profit'] > 0).mean()
        results['avg_return'] = trades_df['return'].mean()
        results['sharpe'] = (
            np.sqrt(252) * trades_df['return'].mean() / trades_df['return'].std()
            if trades_df['return'].std() > 0 else 0
        )
        
        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        results['max_dd'] = drawdown.min()
    else:
        results['win_rate'] = 0
        results['avg_return'] = 0
        results['sharpe'] = 0
        results['max_dd'] = 0
    
    print("OK")
    
    return results


def test_3_configs_consolidated():
    """Prueba rápida con 3 configuraciones - CAPITAL CONSOLIDADO"""
    print("\n" + "="*70)
    print("  BACKTEST CONSOLIDADO - 3 CONFIGURACIONES ($10K)")
    print("="*70)
    
    configs = {
        "CONSERVADOR": 0.60,
        "BALANCEADO": 0.50,
        "AGRESIVO": 0.40
    }
    
    ticker_file = Path("good.txt")
    if ticker_file.exists():
        tickers = [line.strip().upper() for line in ticker_file.read_text().splitlines() if line.strip()]
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
    
    all_results = []
    
    for name, conf in configs.items():
        print(f"\n[{name:<12}] Confidence={conf:.2f}")
        
        results = run_consolidated_backtest(
            tickers=tickers,
            data_root=Path("data"),
            model_path=Path("models/trend_model_2015_2024_OPTUNA_FIXED.joblib"),
            start_date="2023-01-01",
            end_date="2024-12-31",
            min_confidence=conf,
            init_cash=10000.0,
            fees=0.001,
            max_positions=10
        )
        
        if results is None:
            continue
        
        print(f"  → Trades: {results['total_trades']:<4} | Profit: ${results['net_profit']:>8,.2f} | "
              f"Return: {results['total_return']:>6.2%} | Sharpe: {results['sharpe']:>5.2f} | "
              f"MaxDD: {results['max_dd']:>6.2%} | WinRate: {results['win_rate']:>5.2%}")
        
        all_results.append({
            'Config': name,
            'Confidence': conf,
            'Net_Profit': results['net_profit'],
            'Total_Return_%': results['total_return'] * 100,
            'Trades': results['total_trades'],
            'Win_Rate_%': results['win_rate'] * 100,
            'Sharpe': results['sharpe'],
            'Max_DD_%': results['max_dd'] * 100
        })
    
    # Resumen
    print("\n" + "="*70)
    print("  RESUMEN - CAPITAL CONSOLIDADO $10K")
    print("="*70)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Formatear para mostrar
        print(f"\n{'Config':<15} {'Conf':>6} {'Trades':>7} {'Profit':>12} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'WinRate':>8}")
        print("-" * 70)
        
        for _, row in df.iterrows():
            print(f"{row['Config']:<15} {row['Confidence']:>6.2f} {row['Trades']:>7.0f} "
                  f"${row['Net_Profit']:>10,.2f} {row['Total_Return_%']:>7.2f}% "
                  f"{row['Sharpe']:>7.2f} {row['Max_DD_%']:>7.2f}% {row['Win_Rate_%']:>7.2f}%")
        
        print("\n" + "="*70)
        print("  DIFERENCIAS vs MÉTODO ANTERIOR:")
        print("="*70)
        print("  ANTES (VectorBT por ticker):")
        print("    - Capital: $10K × 1,061 tickers = $10.6M total")
        print("    - Trades: ~1,049 (todos los tickers)")
        print("    - Profit: ~$16M (inflado)")
        print("  ")
        print("  AHORA (Consolidado):")
        print("    - Capital: $10K total")
        print("    - Trades: Variable según señales + capital disponible")
        print("    - Profit: Realista ($500-$5K)")
        
        print("\n" + "="*70)
        print("  [OK] BACKTEST CONSOLIDADO COMPLETADO")
        print("="*70)
    else:
        print("\n[ERROR] No se generaron resultados")


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    test_3_configs_consolidated()
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  TIEMPO TOTAL: {elapsed:.2f} segundos ({elapsed/60:.1f} minutos)")
    print(f"{'='*70}")
