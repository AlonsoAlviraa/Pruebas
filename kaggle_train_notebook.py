"""
ENTRENAMIENTO EN KAGGLE - TRIPLE BARRIER MODEL
Ejecuta esto en un Kaggle Notebook con "Save & Run All"
"""

# ==========================================
# SETUP
# ==========================================

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import optuna

print("✓ Imports completados")

# ==========================================
# CARGAR DATOS
# ==========================================

# En Kaggle, datasets están en /kaggle/input/
DATA_PATH = Path('/kaggle/input/trading-system-data')

# Cargar dataset de Triple Barrier
df = pd.read_csv(DATA_PATH / 'triple_barrier_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n✓ Dataset cargado: {len(df):,} samples")
print(f"  Tickers: {df['ticker'].nunique()}")
print(f"  Fechas: {df['date'].min()} a {df['date'].max()}")

# ==========================================
# PREPARAR FEATURES
# ==========================================

# Columnas a excluir
exclude_cols = [
    'date', 'ticker', 'label', 'holding_days', 'return_pct', 
    'entry_price', 'atr'
]

# Features
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]
y = df['label']

# Convertir label a binario: BUY (1) vs REST (0, -1)
y_binary = (y == 1).astype(int)

print(f"\n✓ Features preparadas: {len(feature_cols)} features")
print(f"  Features: {feature_cols[:5]}...")

# ==========================================
# SPLIT TEMPORAL (NO RANDOM!)
# ==========================================

# Ordenar por fecha
df = df.sort_values('date')
split_date = pd.Timestamp('2024-01-01')

train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y_binary[train_mask]
y_test = y_binary[test_mask]

print(f"\n✓ Split temporal:")
print(f"  Train: {len(X_train):,} samples (hasta {split_date})")
print(f"  Test:  {len(X_test):,} samples (desde {split_date})")
print(f"  Train BUY%: {y_train.mean():.2%}")
print(f"  Test BUY%:  {y_test.mean():.2%}")

# ==========================================
# OPTIMIZAR HIPERPARÁMETROS CON OPTUNA
# ==========================================

def objective(trial):
    """Función objetivo para Optuna"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.2)
    }
    
    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Más rápido
    )
    
    # 5-fold cross-validation temporal
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    f1_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

print(f"\n{'='*70}")
print(f"  OPTIMIZACIÓN OPTUNA")
print(f"{'='*70}")

# Ejecutar optimización
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n✓ Optimización completada")
print(f"  Mejor F1-Score: {study.best_value:.4f}")
print(f"  Mejores parámetros:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# ==========================================
# ENTRENAR MODELO FINAL
# ==========================================

print(f"\n{'='*70}")
print(f"  ENTRENANDO MODELO FINAL")
print(f"{'='*70}")

final_model = xgb.XGBClassifier(
    **study.best_params,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

final_model.fit(X_train, y_train)

print(f"\n✓ Modelo entrenado")

# ==========================================
# EVALUACIÓN
# ==========================================

# Predicciones
y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

print(f"\n{'='*70}")
print(f"  RESULTADOS - TRAIN SET")
print(f"{'='*70}")
print(classification_report(y_train, y_pred_train, target_names=['NO-BUY', 'BUY']))

print(f"\n{'='*70}")
print(f"  RESULTADOS - TEST SET")
print(f"{'='*70}")
print(classification_report(y_test, y_pred_test, target_names=['NO-BUY', 'BUY']))

# F1-Score específico
f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"\n✓ F1-Score BUY (Train): {f1_train:.4f}")
print(f"✓ F1-Score BUY (Test):  {f1_test:.4f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'='*70}")
print(f"  TOP 15 FEATURES MÁS IMPORTANTES")
print(f"{'='*70}")
print(feature_importance.head(15).to_string(index=False))

# ==========================================
# GUARDAR MODELO
# ==========================================

output_path = Path('/kaggle/working')

# Guardar modelo
model_file = output_path / 'trend_model_triple_barrier.joblib'
joblib.dump(final_model, model_file)
print(f"\n✓ Modelo guardado: {model_file}")

# Guardar metadata
metadata = {
    'best_params': study.best_params,
    'f1_score_train': float(f1_train),
    'f1_score_test': float(f1_test),
    'features': feature_cols,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'date_split': str(split_date)
}

import json
metadata_file = output_path / 'model_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata guardada: {metadata_file}")

# Guardar feature importance
feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)

print(f"\n{'='*70}")
print(f"  COMPLETADO")
print(f"{'='*70}")
print(f"\nDescarga los archivos de: /kaggle/working/")
print(f"  - trend_model_triple_barrier.joblib")
print(f"  - model_metadata.json")
print(f"  - feature_importance.csv")
