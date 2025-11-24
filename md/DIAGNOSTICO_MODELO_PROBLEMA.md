# DIAGNÓSTICO COMPLETO - ¿Por qué el modelo funciona mal?

## 🔴 PROBLEMA IDENTIFICADO

El modelo está prediciendo muy mal porque **hay un MISMATCH entre las features de entrenamiento y predicción**.

### Detalles Técnicos:

1. **Modelo entrenado con:** 32 features (incluyendo `open`, `high`, `low`, `close`, `volume`)
2. **Predicción usando:** 27 features (EXCLUYENDO `open`, `high`, `low`, `close`, `volume`)
3. **Resultado:** Las 5 features faltantes se rellenan con CEROS, confundiendo al modelo

### Evidencia:

```
[DIAGNOSTICO] diagnose_features.py mostró:

Features en MODELO pero NO en DATOS: 5
  - close
  - volume
  - open
  - low
  - high

[WARNING] 5 features SIEMPRE en 0:
  - open
  - high
  - low
  - close
  - volume

Predicciones:
  - Min: 0.2063
  - Media: 0.2176  ← MUY BAJO!
  - Max: 0.2297
  - Std: 0.0046   ← CASI NO HAY VARIACIÓN
```

## 🎯 CAUSA RAÍZ

El problema está en `train_signal_model_v2.py` líneas 114-133:

```python
def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Esta función ELIMINA OHLCV del entrenamiento
    non_stationary = [
        "open", "high", "low", "close", "volume", 
        "atr", "ma10", "ma20", "sma_50", "sma_200", "volume_sma"
    ]
    
    drop_cols = meta_cols + non_stationary
    features = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
```

**PERO** el modelo actual (`trend_model_2015_2024_OPTUNA.joblib`) fue entrenado ANTES de que se agregara esta exclusión, por lo que tiene OHLCV en las features.

Cuando haces predicciones en `run_backtest_signal.py`, también EXCLUYES OHLCV, pero el modelo espera recibirlas.

## ✅ SOLUCIÓN

### Opción 1: RÁPIDA (No recomendada)
Incluir OHLCV en las predicciones. **PROBLEMA:** Esto causa look-ahead bias y overfitting.

### Opción 2: CORRECTA (Recomendada) ⭐
Re-entrenar el modelo SIN OHLCV, como debería estar.

## 📋 PASOS PARA SOLUCIONAR

### 1. Verificar que `train_signal_model_v2.py` excluye OHLCV correctamente

✅ Ya está correcto en líneas 119-122:

```python
non_stationary = [
    "open", "high", "low", "close", "volume", 
    "atr", "ma10", "ma20", "sma_50", "sma_200", "volume_sma"
]
```

### 2. Re-entrenar el modelo

Ejecuta el script que creé:

```bash
retrain_model_optuna.bat
```

Este script:
- ✅ Entrena el modelo SIN OHLCV (solo features estacionarias/derivadas)
- ✅ Usa Optuna para optimización bayesiana (50 trials)
- ✅ Datos completos 2015-2024
- ✅ Purged K-Fold validation
- ✅ Genera reporte SHAP

**Tiempo estimado:** 1-2 horas

### 3. Actualizar el path del modelo en optimización

Después del entrenamiento, edita `optimize_strategy_v2.py` línea 35:

```python
# ANTES:
"model_path": "models/trend_model_2015_2024_OPTUNA.joblib",

# DESPUÉS:
"model_path": "models/trend_model_2015_2024_OPTUNA_FIXED.joblib",
```

### 4. Ejecutar optimización de nuevo

```bash
python optimize_strategy_v2.py
```

## 🔍 ¿POR QUÉ EXCLUIR OHLCV?

Las columnas OHLCV son **no estacionarias** (crecen con el tiempo) y causan:

1. **Look-Ahead Bias:** El modelo memoriza niveles de precios específicos
2. **Overfitting:** Funciona en entrenamiento pero falla en datos nuevos
3. **No generaliza:** Un precio de $150 en 2020 ≠ $150 en 2024

### Features que SÍ debemos usar:

✅ **Estacionarias/Derivadas:**
- RSI, ADX, ATR normalizado
- Distancia a medias móviles (`dist_sma_50`, `price_rel_ma10`)
- Pendientes (`slope_sma_50`, `slope_rsi_14`)
- Volatilidad relativa
- Crosses y señales

## 📊 EXPECTATIVAS DESPUÉS DE RE-ENTRENAR

Con el modelo corregido, deberías ver:

- **Probabilidades más distribuidas:** 0.1 a 0.9 (no solo 0.18-0.22)
- **Mayor varianza:** Std > 0.10 (actualmente 0.0046)
- **Señales generadas:** Con conf=0.40, deberías tener 100-500+ señales en 2023-2024
- **Mejores métricas:** Sharpe > 1.0, Win Rate > 50%

## 🚀 COMANDOS RÁPIDOS

```bash
# 1. Diagnosticar problema (ya hecho)
python diagnose_features.py

# 2. Re-entrenar modelo
retrain_model_optuna.bat

# 3. Verificar nuevo modelo
python diagnose_features.py  # Debería mostrar 27 features, todas sin 0s

# 4. Optimizar estrategia con modelo nuevo
python optimize_strategy_v2.py
```

## 📝 NOTAS ADICIONALES

### ¿Por qué el modelo anterior "funcionó" en entrenamiento?

Porque TAMBIÉN tenía OHLCV durante entrenamiento. El problema solo apareció al hacer predicciones porque las excluías.

### ¿Esto afecta solo las probabilidades?

Sí. El modelo toma decisiones basándose en esas 5 features faltantes. Sin ellas, está "ciego" y predice lo mismo para todo.

### ¿Necesito re-descargar datos?

No. Los datos están bien. El problema es solo el modelo.

## ✅ VERIFICACIÓN POST-CORRECCIÓN

Después de re-entrenar, ejecuta:

```bash
python diagnose_features.py
```

Deberías ver:

```
Features en MODELO pero NO en DATOS: 0
Features en DATOS pero NO en MODELO: 0

[INFO] Features con valores no-cero: 27/27

Predicciones:
  - Min: 0.05
  - Media: 0.45
  - Max: 0.95
  - Std: 0.18
```

---

**Conclusión:** El modelo actual está roto por un mismatch de features. Re-entrénalo con `retrain_model_optuna.bat`.
