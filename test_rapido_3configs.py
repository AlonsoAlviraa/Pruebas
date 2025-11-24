import subprocess
import pandas as pd
import sys

#===============================================================================
# PRUEBA RAPIDA - Solo 3 configuraciones para verificar
# ==============================================================================

print("="*70)
print("  PRUEBA RAPIDA - 3 CONFIGURACIONES")
print("="*70)
print()

FIXED_PARAMS = {
    "ticker_file": "good.txt",
    "model_path": "models/trend_model_2015_2024_OPTUNA_FIXED.joblib",
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "hard_stop_pct": "0.08",
    "volatility_exponent": "1.0",
    "commission": "0.001"
}

# 3 configuraciones representativas
configs = [
    {
        "name": "CONSERVADOR",
        "min_confidence": 0.60,
        "volatility_target_pct": 0.15,
        "k_atr": 3.0,
        "max_position_pct": 0.15
    },
    {
        "name": "BALANCEADO",
        "min_confidence": 0.50,
        "volatility_target_pct": 0.20,
        "k_atr": 2.5,
        "max_position_pct": 0.20
    },
    {
        "name": "AGRESIVO",
        "min_confidence": 0.40,
        "volatility_target_pct": 0.25,
        "k_atr": 2.0,
        "max_position_pct": 0.25
    }
]

results = []

for i, config in enumerate(configs):
    print(f"\n[{i+1}/3] Probando: {config['name']}")
    print(f"  Confidence={config['min_confidence']}, Vol={config['volatility_target_pct']}, "
          f"K={config['k_atr']}, Pos={config['max_position_pct']}")
    
    cmd = [
        sys.executable, "run_backtest_signal_v2.py",
        "--ticker-file", FIXED_PARAMS["ticker_file"],
        "--model-path", FIXED_PARAMS["model_path"],
        "--start-date", FIXED_PARAMS["start_date"],
        "--end-date", FIXED_PARAMS["end_date"],
        "--hard-stop-pct", FIXED_PARAMS["hard_stop_pct"],
        "--volatility-exponent", FIXED_PARAMS["volatility_exponent"],
        "--commission", FIXED_PARAMS["commission"],
        "--min-confidence", str(config["min_confidence"]),
        "--volatility-target-pct", str(config["volatility_target_pct"]),
        "--k-atr", str(config["k_atr"]),
        "--max-position-pct", str(config["max_position_pct"])
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        df = pd.read_csv("backtest_results.csv")
        
        if df.empty:
            print(f"  [NO] Sin operaciones")
            continue
        
        net_profit = df["net_profit"].sum()
        total_trades = len(df)
        win_rate = (df["net_profit"] > 0).mean() * 100
        
        avg_ret = df["return"].mean()
        std_ret = df["return"].std()
        sharpe = (avg_ret / std_ret * (252**0.5)) if std_ret > 0 else 0
        
        print(f"  [OK] Trades: {total_trades} | Profit: ${net_profit:,.2f} | "
              f"Win%: {win_rate:.1f}% | Sharpe: {sharpe:.2f}")
        
        results.append({
            "Config": config["name"],
            "Confidence": config["min_confidence"],
            "Vol_Target": config["volatility_target_pct"],
            "K_ATR": config["k_atr"],
            "Max_Pos": config["max_position_pct"],
            "Trades": total_trades,
            "Net_Profit": net_profit,
            "Win_Rate_%": win_rate,
            "Sharpe": sharpe
        })
        
    except Exception as e:
        print(f"  [ERROR]: {e}")

print("\n" + "="*70)
print("  RESUMEN")
print("="*70)

if results:
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    print("\n" + "="*70)
    print("  [OK] MODELO FUNCIONA CORRECTAMENTE")
    print("="*70)
    print("\nAhora puedes:")
    print("  1. Dejar corriendo optimize_strategy_v2.py esta noche (120 configs)")
    print("  2. O usar una de estas configuraciones probadas")
else:
    print("\n[ERROR] No se generaron resultados. Verifica el modelo y datos.")
