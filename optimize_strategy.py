import subprocess
import pandas as pd
import numpy as np
import itertools
import sys
import time

# ==============================================================================
# CONFIGURACIÓN DEL GRID DE OPTIMIZACIÓN
# ==============================================================================
# Aquí definimos los rangos a probar. 
# Objetivo: Desbloquear volumen de operaciones y maximizar retorno.

param_grid = {
    # 1. UMBRAL DE ENTRADA: ¿Qué tan exigente es el modelo?
    # Bajamos hasta 0.50 para forzar más operaciones.
    "min_confidence": [0.50, 0.53, 0.55, 0.60],

    # 2. GESTIÓN DE CAPITAL: ¿Cuánto apostamos?
    # Probamos ser muy agresivos (30-40%) para mover la aguja.
    "volatility_target_pct": [0.20, 0.30, 0.40],

    # 3. STOP LOSS (TRAILING): ¿Cuánto espacio damos?
    # 3.0 es estándar, 5.0 es para dejar correr mucho la tendencia.
    "k_atr": [3.0, 4.0, 5.0],

    # 4. DIVERSIFICACIÓN: ¿Cuánto máximo en una sola acción?
    # 0.10 (10 acciones) vs 0.25 (4 acciones concentradas)
    "max_position_pct": [0.10, 0.20, 0.25]
}

# Parámetros fijos (No cambian en el loop)
FIXED_PARAMS = {
    "ticker_file": "good.txt",
    "model_path": "models/stress_test_2022.joblib",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "hard_stop_pct": "0.15",       # Stop de desastre fijo al 15%
    "volatility_exponent": "1.0",  # Lineal
    "commission": "0.001"
}

# ==============================================================================
# MOTOR DE EJECUCIÓN
# ==============================================================================

def run_backtest(params):
    """Construye el comando y ejecuta el backtest, capturando métricas."""
    cmd = [
        sys.executable, "run_backtest_signal.py",
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
        # Ejecutar proceso silenciando el output normal para no ensuciar la consola
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Leer resultados generados
        df = pd.read_csv("backtest_results.csv")
        
        if df.empty:
            return None

        # CÁLCULO DE MÉTRICAS
        net_profit = df["net_profit"].sum()
        total_trades = len(df)
        win_rate = (df["net_profit"] > 0).mean()
        
        # Sharpe aproximado (trade-based)
        avg_ret = df["return"].mean()
        std_ret = df["return"].std()
        sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
        
        # Drawdown aproximado (sobre equity curve si existe, sino simplificado)
        if "equity" in df.columns:
            equity = df.sort_values("exit_date")["equity"]
            roll_max = equity.cummax()
            dd = (equity - roll_max) / roll_max
            max_dd = dd.min()
        else:
            max_dd = 0.0

        return {
            **params,
            "Trades": total_trades,
            "Net Profit ($)": round(net_profit, 2),
            "Win Rate (%)": round(win_rate * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Max DD (%)": round(max_dd * 100, 2)
        }

    except Exception as e:
        return None

def main():
    # Generar todas las combinaciones
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- INICIANDO GRID SEARCH ---")
    print(f"Total de combinaciones a probar: {len(combinations)}")
    print(f"Esto puede tomar unos minutos...\n")
    
    results = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Probando: Conf={params['min_confidence']} | Vol={params['volatility_target_pct']} | K={params['k_atr']} | Pos={params['max_position_pct']} ... ", end="", flush=True)
        
        res = run_backtest(params)
        
        if res:
            print(f"OK -> Profit: ${res['Net Profit ($)']} (Trades: {res['Trades']})")
            results.append(res)
        else:
            print("Sin Operaciones")

    elapsed = time.time() - start_time
    print(f"\n--- FINALIZADO en {elapsed:.2f} segundos ---")

    if not results:
        print("Ninguna configuración generó operaciones. Revisa los filtros base (pipeline).")
        return

    # CREAR DATAFRAME Y RANKING
    df_res = pd.DataFrame(results)
    
    # Ordenar por Ganancia Neta (Net Profit)
    df_best_profit = df_res.sort_values(by="Net Profit ($)", ascending=False).head(10)
    
    print("\nTOP 10 CONFIGURACIONES (Por Ganancia Neta):")
    print(df_best_profit.to_markdown(index=False))
    
    # Exportar todo a CSV
    df_res.to_csv("optimization_results.csv", index=False)
    print("\nResultados completos guardados en 'optimization_results.csv'")
    
    # RECOMENDACIÓN FINAL
    best = df_best_profit.iloc[0]
    print("\n" + "="*60)
    print(" MEJOR CONFIGURACIÓN ENCONTRADA ")
    print("="*60)
    print(f"CONFIDENCE:       {best['min_confidence']}")
    print(f"VOLATILITY TGT:   {best['volatility_target_pct']}")
    print(f"K-ATR (STOP):     {best['k_atr']}")
    print(f"MAX POSITION:     {best['max_position_pct']}")
    print("-" * 30)
    print(f"RESULTADO:        ${best['Net Profit ($)']} | {best['Trades']} Trades | Sharpe {best['Sharpe']}")
    print("="*60)

if __name__ == "__main__":
    main()