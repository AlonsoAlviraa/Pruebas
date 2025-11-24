#!/usr/bin/env python3
"""
DIAGNOSTICO PROFUNDO - Comparar Features de Entrenamiento vs Predicción
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from drl_platform.data_pipeline import DataPipeline, PipelineConfig

def main():
    print("="*70)
    print("  DIAGNOSTICO PROFUNDO: FEATURES DEL MODELO")
    print("="*70)
    
    # Cargar modelo
    model_path = "models/trend_model_2015_2024_OPTUNA_FIXED.joblib"
    print(f"\nCargando modelo: {model_path}")
    
    model_data = joblib.load(model_path)
    
    # Verificar estructura del modelo
    if isinstance(model_data, dict):
        print("\n[INFO] El modelo fue guardado como diccionario")
        print(f"Keys disponibles: {list(model_data.keys())}")
        
        if 'model' in model_data:
            model = model_data['model']
        else:
            print("[ERROR] No se encontro la clave 'model'")
            return
            
        if 'metadata' in model_data:
            metadata = model_data['metadata']
            print(f"\n[METADATA] Informacion del modelo:")
            print(f"  - Modelo entrenado: {metadata.get('timestamp', 'N/A')}")
            print(f"  - Tickers: {len(metadata.get('tickers', []))}")
            print(f"  - Samples: {metadata.get('n_samples', 'N/A')}")
            print(f"  - Features originales: {metadata.get('n_features', 'N/A')}")
            
            if 'feature_names' in metadata:
                feature_names = metadata['feature_names']
                print(f"\n[FEATURES] {len(feature_names)} features esperadas:")
                for i, f in enumerate(feature_names[:20]):
                    print(f"  {i+1}. {f}")
                if len(feature_names) > 20:
                    print(f"  ... y {len(feature_names) - 20} mas")
    else:
        model = model_data
        feature_names = None
    
    # Verificar features del modelo sklearn
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        print(f"\n[SKLEARN] {len(model_features)} features in model.feature_names_in_:")
        for i, f in enumerate(model_features[:20]):
            print(f"  {i+1}. {f}")
        if len(model_features) > 20:
            print(f"  ... y {len(model_features) - 20} mas")
    else:
        print("\n[WARNING] El modelo no tiene feature_names_in_")
        model_features = None
    
    # Cargar datos de ejemplo
    print("\n" + "="*70)
    print("  COMPARAR CON DATOS REALES")
    print("="*70)
    
    pipeline = DataPipeline(PipelineConfig(data_root=Path("data")))
    ticker = "AAPL"
    
    print(f"\nCargando datos de {ticker}...")
    df = pipeline.load_feature_view(ticker, indicators=True)
    
    if df.empty:
        print("[ERROR] Sin datos")
        return
    
    # Filtrar 2023
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[(df["date"] >= "2023-01-01") & (df["date"] <= "2023-12-31")]
    
    # Preparar features
    feature_cols = [c for c in df.columns if c not in [
        "date", "ticker", "target", "open", "high", "low", "close", "volume"
    ]]
    
    print(f"\n[DATOS] {len(feature_cols)} features disponibles en los datos:")
    for i, f in enumerate(feature_cols[:20]):
        print(f"  {i+1}. {f}")
    if len(feature_cols) > 20:
        print(f"  ... y {len(feature_cols) - 20} mas")
    
    # Comparar
    if model_features:
        print("\n" + "="*70)
        print("  COMPARACION")
        print("="*70)
        
        missing_in_data = set(model_features) - set(feature_cols)
        extra_in_data = set(feature_cols) - set(model_features)
        
        print(f"\nFeatures en MODELO pero NO en DATOS: {len(missing_in_data)}")
        if missing_in_data:
            for f in list(missing_in_data)[:10]:
                print(f"  - {f}")
            if len(missing_in_data) > 10:
                print(f"  ... y {len(missing_in_data) - 10} mas")
        
        print(f"\nFeatures en DATOS pero NO en MODELO: {len(extra_in_data)}")
        if extra_in_data:
            for f in list(extra_in_data)[:10]:
                print(f"  - {f}")
            if len(extra_in_data) > 10:
                print(f"  ... y {len(extra_in_data) - 10} mas")
        
        # PRUEBA DE PREDICCION
        print("\n" + "="*70)
        print("  PRUEBA DE PREDICCION")
        print("="*70)
        
        X = df[feature_cols].reindex(columns=model_features, fill_value=0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"\nShape de X: {X.shape}")
        print(f"Valores NaN en X: {X.isna().sum().sum()}")
        print(f"Valores Inf en X: {np.isinf(X.values).sum()}")
        
        # Verificar estadisticas
        print(f"\n[ESTADISTICAS] Top 10 features con mayor varianza:")
        variances = X.var().sort_values(ascending=False)
        for i, (feat, var) in enumerate(variances.head(10).items()):
            mean = X[feat].mean()
            std = X[feat].std()
            print(f"  {i+1}. {feat}: mean={mean:.4f}, std={std:.4f}, var={var:.4f}")
        
        # Predecir
        print(f"\nRealizando predicciones...")
        try:
            proba = model.predict_proba(X)[:, 1]
            print(f"Predicciones exitosas!")
            print(f"  - Min: {proba.min():.4f}")
            print(f"  - Media: {proba.mean():.4f}")
            print(f"  - Max: {proba.max():.4f}")
            print(f"  - Std: {proba.std():.4f}")
            
            # Ver cuantas features tienen valores != 0
            non_zero_features = (X != 0).any(axis=0).sum()
            print(f"\n[INFO] Features con valores no-cero: {non_zero_features}/{len(model_features)}")
            
            # Mostrar cuales features estan en 0
            zero_features = X.columns[(X == 0).all(axis=0)]
            if len(zero_features) > 0:
                print(f"\n[WARNING] {len(zero_features)} features SIEMPRE en 0:")
                for f in list(zero_features)[:15]:
                    print(f"  - {f}")
                if len(zero_features) > 15:
                    print(f"  ... y {len(zero_features) - 15} mas")
            
        except Exception as e:
            print(f"[ERROR] Prediccion fallo: {e}")
    
    print("\n" + "="*70)
    print("  DIAGNOSTICO COMPLETO")
    print("="*70)

if __name__ == "__main__":
    main()
