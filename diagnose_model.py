#!/usr/bin/env python3
"""
DIAGNÓSTICO DEL MODELO - ¿Por qué solo 12 trades?
Vamos a verificar:
1. Qué está prediciendo el modelo
2. Cómo se ven las probabilidades
3. Si los filtros están funcionando
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from drl_platform.data_pipeline import DataPipeline, PipelineConfig

def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def main():
    print("="*70)
    print("  DIAGNÓSTICO DEL MODELO")
    print("="*70)
    
    # Cargar modelo
    model_path = "models/trend_model_2015_2024_OPTUNA_FIXED.joblib"
    print(f"\nCargando modelo: {model_path}")
    model = load_model(model_path)
    
    # Cargar datos de prueba
    pipeline = DataPipeline(PipelineConfig(data_root=Path("data")))
    
    # Probar con algunos tickers conocidos
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
    
    all_predictions = []
    
    for ticker in test_tickers:
        try:
            print(f"\nAnalizando {ticker}...")
            df = pipeline.load_feature_view(ticker, indicators=True)
            
            if df.empty:
                print(f"  [ERROR] Sin datos")
                continue
            
            # Filtrar 2023-2024
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df[(df["date"] >= "2023-01-01") & (df["date"] <= "2024-12-31")]
            
            if len(df) < 50:
                print(f"  [ERROR] Datos insuficientes: {len(df)} filas")
                continue
            
            # Calcular features necesarias
            df["ma_50"] = ta.sma(df["close"], length=50)
            df["ret_1m"] = df["close"].pct_change(periods=21)
            df = df.dropna(subset=["ma_50", "ret_1m"])
            
            if df.empty:
                print(f"  [ERROR] Sin datos despues de features")
                continue
            
            # Preparar features para modelo
            feature_cols = [c for c in df.columns if c not in [
                "date", "ticker", "target", "open", "high", "low", "close", "volume", 
                "atr", "ma_50", "ret_1m"
            ]]
            
            if hasattr(model, "feature_names_in_"):
                X = df[feature_cols].reindex(columns=model.feature_names_in_, fill_value=0)
            else:
                X = df[feature_cols]
            
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            
            # PREDICCIONES
            preds_proba = model.predict_proba(X)[:, 1]  # Probabilidad de BUY
            preds_class = model.predict(X)
            
            # Aplicar filtros
            trend_filter = df["close"] > df["ma_50"]
            momentum_filter = df["ret_1m"] >= 0.03
            
            # Contar
            total_rows = len(df)
            pass_trend = trend_filter.sum()
            pass_momentum = momentum_filter.sum()
            pass_both = (trend_filter & momentum_filter).sum()
            
            # Probabilidades
            high_conf_40 = (preds_proba >= 0.40).sum()
            high_conf_50 = (preds_proba >= 0.50).sum()
            high_conf_60 = (preds_proba >= 0.60).sum()
            
            # Señales finales
            signals_40 = (preds_proba >= 0.40) & trend_filter & momentum_filter
            signals_50 = (preds_proba >= 0.50) & trend_filter & momentum_filter
            signals_60 = (preds_proba >= 0.60) & trend_filter & momentum_filter
            
            print(f"  [OK] Datos: {total_rows} filas (2023-2024)")
            print(f"  [FILTROS]:")
            print(f"     - Trend (close>MA50): {pass_trend}/{total_rows} ({pass_trend/total_rows*100:.1f}%)")
            print(f"     - Momentum (ret_1m>=3%): {pass_momentum}/{total_rows} ({pass_momentum/total_rows*100:.1f}%)")
            print(f"     - Ambos: {pass_both}/{total_rows} ({pass_both/total_rows*100:.1f}%)")
            print(f"  [PREDICCIONES]:")
            print(f"     - Prob >= 0.40: {high_conf_40}/{total_rows} ({high_conf_40/total_rows*100:.1f}%)")
            print(f"     - Prob >= 0.50: {high_conf_50}/{total_rows} ({high_conf_50/total_rows*100:.1f}%)")
            print(f"     - Prob >= 0.60: {high_conf_60}/{total_rows} ({high_conf_60/total_rows*100:.1f}%)")
            print(f"  [SENALES FINALES] (Modelo + Filtros):")
            print(f"     - Conf=0.40: {signals_40.sum()} senales")
            print(f"     - Conf=0.50: {signals_50.sum()} senales")
            print(f"     - Conf=0.60: {signals_60.sum()} senales")
            print(f"  [DISTRIBUCION] Probabilidades:")
            print(f"     - Min: {preds_proba.min():.4f}")
            print(f"     - Media: {preds_proba.mean():.4f}")
            print(f"     - Max: {preds_proba.max():.4f}")
            print(f"     - Percentil 25: {np.percentile(preds_proba, 25):.4f}")
            print(f"     - Mediana: {np.percentile(preds_proba, 50):.4f}")
            print(f"     - Percentil 75: {np.percentile(preds_proba, 75):.4f}")
            
            all_predictions.extend(preds_proba.tolist())
            
        except Exception as e:
            print(f"  [ERROR]: {e}")
            continue
    
    # Resumen global
    if all_predictions:
        all_preds = np.array(all_predictions)
        print("\n" + "="*70)
        print("  RESUMEN GLOBAL DE PREDICCIONES")
        print("="*70)
        print(f"Total predicciones: {len(all_preds):,}")
        print(f"Prob >= 0.40: {(all_preds >= 0.40).sum():,} ({(all_preds >= 0.40).mean()*100:.1f}%)")
        print(f"Prob >= 0.50: {(all_preds >= 0.50).sum():,} ({(all_preds >= 0.50).mean()*100:.1f}%)")
        print(f"Prob >= 0.60: {(all_preds >= 0.60).sum():,} ({(all_preds >= 0.60).mean()*100:.1f}%)")
        print(f"Media: {all_preds.mean():.4f}")
        print(f"Mediana: {np.median(all_preds):.4f}")
        print(f"Std: {all_preds.std():.4f}")
        
if __name__ == "__main__":
    main()
