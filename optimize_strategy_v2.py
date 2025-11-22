import subprocess
import pandas as pd
import numpy as np
import itertools
import sys
import time

# ==============================================================================
# CONFIGURACIÓN DEL GRID DE OPTIMIZACIÓN v2.0 - 120 COMBINACIONES
# ==============================================================================
# Modelo entrenado con Optuna (mejor F1-Score)
# Exploración exhaustiva del espacio de hiperparámetros

param_grid = {
    # 1. UMBRAL DE ENTRADA (Confianza del modelo)
    # Rango ampliado para explorar desde señales agresivas a conservadoras
    "min_confidence": [0.40, 0.45, 0.50, 0.55, 0.60],
    
    # 2. GESTIÓN DE CAPITAL (Volatility Targeting)
    # Más opciones para encontrar el sweet spot
    "volatility_target_pct": [0.10, 0.15, 0.20, 0.25],
    
    # 3. STOP LOSS TRAILING (K × ATR)
    # Desde stops muy ajustados hasta generosos
    "k_atr": [2.0, 2.5, 3.0],
    
    # 4. LÍMITE DE POSICIÓN INDIVIDUAL
    # Balance entre diversificación y concentración
    "max_position_pct": [0.10, 0.15, 0.20, 0.25]
}

# Parámetros fijos (NO cambian)
FIXED_PARAMS = {
    "ticker_file": "good.txt",
    "model_path": "models/trend_model_2015_2024_OPTUNA.joblib",  # Nuevo modelo Optuna
    "start_date": "2023-01-01",  # Out-of-sample testing (2023-2024)
    "end_date": "2024-12-31",
    "hard_stop_pct": "0.08",       # Hard stop al 8%
    "volatility_exponent": "1.0",  # Escalamiento lineal
    "commission": "0.001"          # 0.1% comisión
}

# Total de combinaciones: 5 × 4 × 3 × 4 = 120

# ==============================================================================
# MOTOR DE EJECUCIÓN
# ==============================================================================

def run_backtest(params):
    """Construye el comando y ejecuta el backtest MEJORADO v2, capturando métricas."""
    cmd = [
        sys.executable, "run_backtest_signal_v2.py",
        "--ticker-file", FIXED_PARAMS["ticker_file"],
        "--model-path", FIXED_PARAMS["model_path"],
        "--start-date", FIXED_PARAMS["start_date"],
        "--end-date", FIXED_PARAMS["end_date"],
        "--hard-stop-pct", FIXED_PARAMS["hard_stop_pct"],
        "--volatility-exponent", FIXED_PARAMS["volatility_exponent"],
        "--commission", FIXED_PARAMS["commission"],
        # Parámetros variables del Grid
        "--min-confidence", str(params["min_confidence"]),
        "--volatility-target-pct", str(params["volatility_target_pct"]),
        "--k-atr", str(params["k_atr"]),
        "--max-position-pct", str(params["max_position_pct"])
    ]

    try:
        # Ejecutar proceso silenciando el output
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Leer resultados generados
        df = pd.read_csv("backtest_results.csv")
        
        if df.empty:
            return None

        # CÁLCULO DE MÉTRICAS MEJORADAS
        net_profit = df["net_profit"].sum()
        total_trades = len(df)
        win_rate = (df["net_profit"] > 0).mean()
        
        # Sharpe aproximado (trade-based)
        avg_ret = df["return"].mean()
        std_ret = df["return"].std()
        sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
        
        # Drawdown sobre equity curve
        if "equity" in df.columns:
            equity = df.sort_values("exit_date")["equity"]
            roll_max = equity.cummax()
            dd = (equity - roll_max) / roll_max
            max_dd = dd.min()
        else:
            max_dd = 0.0
        
        # Métrica de calidad: Sharpe / |MaxDD| (mayor es mejor)
        quality_score = sharpe / abs(max_dd) if max_dd < 0 else sharpe

        return {
            **params,
            "Trades": total_trades,
            "Net Profit ($)": round(net_profit, 2),
            "Win Rate (%)": round(win_rate * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Max DD (%)": round(max_dd * 100, 2),
            "Quality Score": round(quality_score, 2)
        }

    except Exception as e:
        return None

def main():
    # Generar todas las combinaciones
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print("="*70)
    print("  OPTIMIZACION MEJORADA - TREND FOLLOWING v2.0")
    print("="*70)
    print(f"  Total de combinaciones: {len(combinations)}")
    print("  Estrategia: Inverse Volatility + Filtros Inteligentes")
    print("  Basado en SHAP: dmp_14, ATR, volatility_20")
    print("="*70)
    print()
    
    results = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        print(f"[{i+1:>3}/{len(combinations)}] Conf={params['min_confidence']:.2f} | Vol={params['volatility_target_pct']:.2f} | K={params['k_atr']:.1f} | Pos={params['max_position_pct']:.2f} ... ", end="", flush=True)
        
        res = run_backtest(params)
        
        if res:
            print(f"OK -> Profit: ${res['Net Profit ($)']:>10,.2f} | Trades: {res['Trades']:>3} | Sharpe: {res['Sharpe']:>5.2f}")
            results.append(res)
        else:
            print("Sin Operaciones")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"FINALIZADO en {elapsed:.2f} segundos ({elapsed/60:.1f} minutos)")
    print(f"{'='*70}\n")

    if not results:
        print("Ninguna configuracion genero operaciones. Revisa los filtros base.")
        return

    # CREAR DATAFRAME Y RANKING
    df_res = pd.DataFrame(results)
    
    # Filtrar configuraciones con al menos 10 trades
    df_valid = df_res[df_res["Trades"] >= 10].copy()
    
    if df_valid.empty:
        print("No hay configuraciones con al menos 10 trades. Mostrando todas...")
        df_valid = df_res
    
    # Rankings múltiples
    print("\n" + "="*80)
    print(" TOP 5 CONFIGURACIONES (Por CALIDAD: Sharpe / |MaxDD|) ")
    print("="*80)
    df_best_quality = df_valid.sort_values(by="Quality Score", ascending=False).head(5)
    print(df_best_quality.to_string(index=False))
    
    print("\n" + "="*80)
    print(" TOP 5 CONFIGURACIONES (Por Ganancia Neta) ")
    print("="*80)
    df_best_profit = df_valid.sort_values(by="Net Profit ($)", ascending=False).head(5)
    print(df_best_profit.to_string(index=False))
    
    print("\n" + "="*80)
    print(" TOP 5 CONFIGURACIONES (Por Sharpe Ratio) ")
    print("="*80)
    df_best_sharpe = df_valid.sort_values(by="Sharpe", ascending=False).head(5)
    print(df_best_sharpe.to_string(index=False))
    
    # Exportar todo a CSV
    df_res.to_csv("optimization_results_v2.csv", index=False)
    print(f"\nResultados completos guardados en 'optimization_results_v2.csv'")
    
    # RECOMENDACIÓN FINAL (Por Quality Score)
    best = df_best_quality.iloc[0]
    print("\n" + "="*78)
    print(" MEJOR CONFIGURACION (Quality Score)")
    print("="*78)
    print(f"  Confidence:        {best['min_confidence']}")
    print(f"  Volatility Target: {best['volatility_target_pct']}")
    print(f"  K-ATR (Stop):      {best['k_atr']}")
    print(f"  Max Position:      {best['max_position_pct']}")
    print("="*78)
    print(f"  Resultado:  ${best['Net Profit ($)']:,.2f} | {best['Trades']} Trades | Sharpe {best['Sharpe']:.2f} | DD {best['Max DD (%)']:.1f}%")
    print(f"  Quality:    {best['Quality Score']:.2f}")
    print("="*78)

if __name__ == "__main__":
    main()
