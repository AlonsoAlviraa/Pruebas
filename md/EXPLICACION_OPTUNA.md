# EXPLICACIÓN OPTUNA - Optimización de Hiperparámetros

## 🎯 ¿Qué es Optuna?

Optuna es una librería de **optimización bayesiana** que encuentra automáticamente los mejores hiperparámetros para tu modelo de Machine Learning.

En lugar de probar manualmente:
```
n_estimators = 100, 200, 300, 400, 500
max_depth = 4, 6, 8, 10, 12
learning_rate = 0.01, 0.05, 0.1, 0.2, 0.3
```

Optuna **aprende** qué combinaciones funcionan mejor y explora inteligentemente el espacio de búsqueda.

## 📊 Cómo funciona en `train_signal_model_v2.py`

### 1. Definición del Espacio de Búsqueda (líneas 276-316)

```python
def objective(trial):
    if model_type == "xgb":
        params = ModelParams(
            model_type="xgb",
            # Optuna sugiere valores en rangos definidos
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            ...
        )
```

**Rangos definidos:**
- `n_estimators`: 100 a 500 árboles
- `max_depth`: profundidad de 4 a 12
- `learning_rate`: 0.01 a 0.3 (escala logarítmica)
- `subsample`: 60% a 100% de muestras
- `colsample_bytree`: 60% a 100% de features

### 2. Función Objetivo (métrica a maximizar)

```python
# Cross-validation con Purged K-Fold
f1_scores = []
for fold, (train_idx, test_idx) in enumerate(validator.split(master)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    model = build_model(**asdict(params))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # F1-Score de la clase BUY (la más importante)
    f1_buy = f1_score(y_test, preds, labels=[BUY_CLASS], average="micro")
    f1_scores.append(f1_buy)

return np.mean(f1_scores)  # ← Optuna maximiza esto
```

**Objetivo:** Maximizar el F1-Score promedio de la clase **BUY** en cross-validation.

### 3. Ejecución del Estudio (líneas 340-348)

```python
sampler = TPESampler(seed=random_state)  # Tree-structured Parzen Estimator
study = optuna.create_study(
    direction="maximize",  # Queremos MAXIMIZAR F1-Score
    sampler=sampler,
    study_name=f"trend_following_{model_type}"
)

study.optimize(objective, n_trials=50, show_progress_bar=True)
```

**TPESampler:** Algoritmo bayesiano inteligente que:
1. Prueba configuraciones iniciales aleatorias
2. Aprende cuáles áreas del espacio funcionan mejor
3. Concentra búsquedas en regiones prometedoras
4. Evita pérdida de tiempo en configuraciones malas

### 4. Resultados (líneas 350-353)

```python
logger.info(f"Mejor F1-Score (BUY): {study.best_value:.4f}")
logger.info(f"Mejores parámetros: {study.best_params}")

return study.best_params
```

**Ejemplo de output:**
```
Mejor F1-Score (BUY): 0.6842
Mejores parámetros: {
    'n_estimators': 387,
    'max_depth': 9,
    'learning_rate': 0.0742,
    'subsample': 0.87,
    'colsample_bytree': 0.73
}
```

## 🔄 Proceso Completo del Entrenamiento

```
┌──────────────────────────────────────────────────────────────┐
│ 1. CARGAR DATOS                                              │
│    └→ _prepare_master(): Label con Triple Barrier           │
│       - 2015-2024, múltiples tickers                         │
│       - Target: BUY/HOLD/STOP                                │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. OPTIMIZACIÓN BAYESIANA (Optuna)                           │
│    └→ optimize_hyperparameters_optuna()                      │
│       Trial 1:  n_estimators=245, max_depth=7  → F1=0.52    │
│       Trial 2:  n_estimators=412, max_depth=5  → F1=0.48    │
│       Trial 3:  n_estimators=189, max_depth=11 → F1=0.61    │
│       ...                                                     │
│       Trial 50: n_estimators=387, max_depth=9  → F1=0.68 ★  │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. VALIDACIÓN CRUZADA (Purged K-Fold)                        │
│    └→ _purged_kfold_scores()                                 │
│       Fold 1: F1(BUY)=0.65                                   │
│       Fold 2: F1(BUY)=0.71                                   │
│       Fold 3: F1(BUY)=0.62                                   │
│       Fold 4: F1(BUY)=0.69                                   │
│       Fold 5: F1(BUY)=0.67                                   │
│       ────────────────────                                   │
│       Media:  F1(BUY)=0.668 ± 0.032                          │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. ENTRENAMIENTO FINAL                                       │
│    └→ _train_final_model()                                   │
│       - Usar TODOS los datos (2015-2024)                     │
│       - Usar mejores hiperparámetros de Optuna               │
│       - Entrenar modelo XGBoost final                        │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. ANÁLISIS SHAP                                             │
│    └→ _generate_shap_report()                                │
│       - ¿Qué features son más importantes?                   │
│       - dmp_14, atr_norm, volatility_20, etc.                │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. GUARDAR MODELO                                            │
│    └→ models/trend_model_2015_2024_OPTUNA_FIXED.joblib       │
│       + metadata.json (hiperparámetros, features, métricas)  │
└──────────────────────────────────────────────────────────────┘
```

## 🎨 Ventajas de Optuna vs Grid Search

### Grid Search (método tradicional)
```python
# Probar TODAS las combinaciones
for n_est in [100, 200, 300, 400, 500]:      # 5 opciones
    for depth in [4, 6, 8, 10, 12]:          # 5 opciones
        for lr in [0.01, 0.05, 0.1, 0.2]:    # 4 opciones
            train_and_evaluate(...)
            
# Total: 5 × 5 × 4 = 100 combinaciones
# Tiempo: ~3-4 horas
```

### Optuna (Bayesian Optimization)
```python
# Explorar inteligentemente
study.optimize(objective, n_trials=50)

# Total: 50 trials (MEJOR que 100 random)
# Tiempo: ~1-2 horas
# Resultado: MEJOR que grid search exhaustivo
```

**¿Por qué es mejor?**
1. **Más eficiente:** Encuentra buenos parámetros con menos trials
2. **Más flexible:** Puede explorar valores continuos (ej. lr=0.0742)
3. **Adaptativo:** Aprende de trials anteriores

## 📈 Ejemplo Real - Optuna en Acción

```
Trial  1: n_est=245, depth=7,  lr=0.15   → F1=0.52
Trial  2: n_est=412, depth=5,  lr=0.03   → F1=0.48
Trial  3: n_est=189, depth=11, lr=0.21   → F1=0.61  ← mejor hasta ahora
Trial  4: n_est=156, depth=10, lr=0.19   → F1=0.58  ← cerca de trial 3
Trial  5: n_est=203, depth=12, lr=0.18   → F1=0.62  ← nueva mejor!
...
Trial 48: n_est=379, depth=9,  lr=0.08   → F1=0.67
Trial 49: n_est=401, depth=9,  lr=0.07   → F1=0.68  ← MEJOR GLOBAL
Trial 50: n_est=392, depth=8,  lr=0.07   → F1=0.66

Mejor configuración encontrada:
  n_estimators: 401
  max_depth: 9
  learning_rate: 0.07
  F1-Score: 0.68
```

Nota cómo Optuna **converge** hacia buenos valores (trials 48-50 están explorando la región óptima).

## ⚙️ Parámetros del Script

```bash
--optimize-hyperparams       # Activa Optuna
--n-trials 50                # Número de configuraciones a probar
--model-type xgb             # XGBoost, Random Forest, o LightGBM
--n-splits 5                 # Folds para cross-validation
--purge-window 10            # Días entre train/test para evitar leakage
```

## 🎯 Métricas Optimizadas

### F1-Score de la clase BUY

```python
f1_buy = f1_score(y_test, preds, labels=[BUY_CLASS], average="micro")
```

**¿Por qué F1-Score?**
- Balancea Precision y Recall
- Importante para clases desbalanceadas
- Mide qué tan bien identifica señales de COMPRA

**Fórmula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Precision = TP / (TP + FP)  ← De las que predijo BUY, cuántas acertó
Recall    = TP / (TP + FN)  ← De las BUY reales, cuántas encontró
```

## 🚀 Ejecutar Entrenamiento con Optuna

```bash
# Opción fácil (usa el BAT)
retrain_model_optuna.bat

# Opción manual
python train_signal_model_v2.py \
  --ticker-file good.txt \
  --output models/trend_model_2015_2024_OPTUNA_FIXED.joblib \
  --train-from 2015-01-01 \
  --train-until 2024-12-31 \
  --model-type xgb \
  --optimize-hyperparams \
  --n-trials 50 \
  --shap-report-dir reports/shap_v2_fixed
```

## 📊 Output Esperado

```
======================================================================
  ENTRENAMIENTO MODELO ML v2.0 - TREND FOLLOWING
======================================================================
  Período: 2015-01-01 a 2024-12-31
  Modelo: XGB
  Optimización: SÍ
======================================================================

Cargando datos para 2000+ tickers...
Dataset maestro final: 1,245,892 muestras de 1847 tickers
Rango de fechas: 2015-01-01 a 2024-12-31

Iniciando optimización Bayesiana con Optuna (50 trials)...
[I 2024-11-23 11:30:15,742] Trial 1 finished with value: 0.5234
[I 2024-11-23 11:32:48,156] Trial 2 finished with value: 0.4892
[I 2024-11-23 11:35:21,893] Trial 3 finished with value: 0.6145
...
[I 2024-11-23 13:15:32,412] Trial 50 finished with value: 0.6621

Mejor F1-Score (BUY): 0.6842
Mejores parámetros: {'n_estimators': 401, 'max_depth': 9, ...}

Ejecutando Purged K-Fold validation (splits=5, purge=10)...
F1(BUY) Cross-Validation: 0.6680 ± 0.0324

Entrenando modelo final con 1,245,892 muestras...
Modelo guardado en models/trend_model_2015_2024_OPTUNA_FIXED.joblib

Reporte SHAP guardado en reports/shap_v2_fixed

======================================================================
  ENTRENAMIENTO COMPLETADO EXITOSAMENTE
======================================================================
```

---

**TL;DR:** Optuna encuentra automáticamente los mejores hiperparámetros probando inteligentemente 50 configuraciones y aprendiendo cuáles funcionan mejor.
