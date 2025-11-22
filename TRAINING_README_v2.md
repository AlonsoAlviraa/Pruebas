# 🚀 ENTRENAMIENTO MODELO ML v2.0 - Trend Following System

## 📋 Descripción

Este documento describe el proceso de entrenamiento mejorado del modelo ML para el sistema de trend following.

---

## 🎯 Mejoras Implementadas

### 1. **Más Datos**: 2015-2024 (9 años)
- **Antes**: Solo 2022 (~1 año)
- **Ahora**: 2015-2022 para entrenamiento, 2023-2024 para testing
- **Beneficio**: Modelo más robusto que ha visto múltiples ciclos de mercado

### 2. **Optimización de Hiperparámetros** (Optuna)
- **Método**: Bayesian Optimization (TPE Sampler)
- **Objetivo**: Maximizar F1-Score de clase BUY
- **Espacio de búsqueda**:
  - `n_estimators`: 100-500
  - `max_depth`: 4-12
  - `learning_rate`: 0.01-0.3 (log scale)
  - `subsample`: 0.6-1.0
  - `colsample_bytree`: 0.6-1.0

### 3. **Triple Barrier Labels Mejorado**
- **Horizonte**: 10 días (antes 5)
- **Take Profit**: 3.0 × ATR (antes 2.0)
- **Stop Loss**: 2.0 × ATR
- **Beneficio**: Labels más realistas alineados con la estrategia

### 4. **Validación Temporal Robusta**
- **Purged K-Fold**: Ventana de purga de 10 días
- **5 Folds**: Split temporales sin contaminación
- **Métrica**: F1-Score para clase BUY (la más importante)

### 5. **Class Balancing**
- **Problema**: Clases desbalanceadas (HOLD >> BUY/STOP)
- **Solución**: `class_weighted=True` automático
- **Resultado**: Mejor detección de señales BUY

---

## 📁 Archivos Creados

### Scripts:
1. `train_signal_model_v2.py` - Script principal de entrenamiento
2. `train_model_v2.bat` - Batch script para Windows (3 opciones)

### Modelos que se generarán:
1. `models/trend_model_2015_2024_quick.joblib` - Sin optimización (~5-10 min)
2. `models/trend_model_2015_2024_optimized.joblib` - Optuna 50 trials (~30-60 min)
3. `models/trend_model_2015_2024_exhaustive.joblib` - Optuna 100 trials (~1-2 horas)

### Reportes SHAP:
- `reports/shap_v2_quick/` - Análisis del modelo rápido
- `reports/shap_v2_optimized/` - Análisis del modelo optimizado
- `reports/shap_v2_exhaustive/` - Análisis del modelo exhaustivo

---

## 🚀 Uso

### Opción 1: Script Batch (Recomendado para Windows)
```batch
train_model_v2.bat
```
Luego selecciona:
- **[1]** Rápido - Sin optimización
- **[2]** Optimizado - 50 trials Optuna
- **[3]** Exhaustivo - 100 trials Optuna

### Opción 2: Línea de Comandos

#### Entrenamiento Rápido:
```bash
python train_signal_model_v2.py \
    --ticker-file good.txt \
    --output models/trend_model_2015_2024_quick.joblib \
    --train-from 2015-01-01 \
    --train-until 2022-12-31 \
    --horizon 10 \
    --model-type xgb \
    --n-estimators 300 \
    --class-weighted \
    --shap-report-dir reports/shap_v2_quick
```

#### Entrenamiento con Optimización (Recomendado):
```bash
python train_signal_model_v2.py \
    --ticker-file good.txt \
    --output models/trend_model_2015_2024_optimized.joblib \
    --train-from 2015-01-01 \
    --train-until 2022-12-31 \
    --horizon 10 \
    --model-type xgb \
    --optimize-hyperparams \
    --n-trials 50 \
    --class-weighted \
    --shap-report-dir reports/shap_v2_optimized
```

---

## 📊 Parámetros Clave

### Datos:
- `--train-from`: Fecha inicio entrenamiento (default: 2015-01-01)
- `--train-until`: Fecha fin entrenamiento (default: 2022-12-31)
- `--ticker-file`: Archivo con lista de tickers (default: good.txt)

### Triple Barrier Labels:
- `--horizon`: Días para evaluar señal (default: 10)
- `--atr-multiplier-tp`: Multiplicador ATR para Take Profit (default: 3.0)
- `--atr-multiplier-sl`: Multiplicador ATR para Stop Loss (default: 2.0)

### Modelo:
- `--model-type`: Tipo de modelo [xgb, rf, lgbm] (default: xgb)
- `--optimize-hyperparams`: Activar optimización Bayesiana
- `--n-trials`: Número de trials Optuna (default: 50)
- `--class-weighted`: Balancear clases automáticamente

### Validación:
- `--n-splits`: Número de folds (default: 5)
- `--purge-window`: Ventana de purga en días (default: 10)

---

## 📈 Resultados Esperados

### Métricas de Cross-Validation:
Con los datos 2015-2022, deberías ver:
- **F1-Score (BUY)**: 0.35-0.55 (balance precisión/recall)
- **Accuracy Total**: 0.50-0.65
- **Support BUY**: ~500-2000 muestras por fold

### Comparación con Modelo Anterior:

| Métrica | Modelo Anterior | Modelo v2.0 (Esperado) |
|---------|----------------|------------------------|
| Datos | 2022 (1 año) | 2015-2022 (8 años) |
| Samples | ~5,000 | ~30,000-50,000 |
| F1 (BUY) | ~0.30 | ~0.40-0.50 |
| Features importantes | dmp_14, atr, volatility | Mismo enfoque optimizado |

---

## 🔍 Interpretación SHAP

Después del entrenamiento, revisa:

1. **`shap_feature_importance_v2.csv`**:
   - Features ordenadas por impacto absoluto
   - Normalized importance (0-1)

2. **`shap_bar_buy_v2.png`**:
   - Top 20 features más importantes
   - Visual rápido de qué mueve las predicciones

3. **`shap_summary_buy_v2.png`**:
   - Bee swarm plot
   - Muestra dirección del impacto (positivo/negativo)

### Features Esperadas (Top 10):
Basándote en el análisis anterior, deberías ver:
1. `dmp_14` - Directional Movement Plus
2. `atr` - Average True Range
3. `volatility_20` - Volatilidad 20 períodos  
4. `dist_sma_200` - Distancia a SMA 200
5. `log_return_3m` / `log_return_6m` - Retornos log
6. `ma20`, `sma_50`, `sma_200` - Moving averages
7. `adx_14`, `dmn_14` - ADX system
8. `rsi_7`, `rsi_21` - RSI indicators

---

## ⚙️ Troubleshooting

### Error: "No labelled samples were produced"
**Causa**: No hay suficientes datos históricos para algunos tickers  
**Solución**: Verifica que los tickers en `good.txt` tengan datos desde 2015

### Error: "Fold skipped due to single class"
**Causa**: En algún fold no hay suficientes muestras BUY  
**Solución**: Normal si tienes pocos tickers. Aumenta el número de tickers o reduce `n_splits`

### Optuna muy lento
**Causa**: Cada trial requiere entrenar el modelo 5 veces (K-Fold)  
**Solución**:
- Reduce `--n-trials` (ej: 30 en vez de 50)
- Reduce `--n-splits` (ej: 3 en vez de 5)
- Usa menos tickers para pruebas iniciales

### Memoria insuficiente
**Causa**: Dataset muy grande para XGBoost  
**Solución**:
- Usa `--model-type rf` (RandomForest usa menos RAM)
- Reduce `--n-estimators`
- Filtra tickers con menos liquidez

---

## 📝 Workflow Recomendado

### 1. Entrenamiento Inicial (Rápido)
```bash
python train_signal_model_v2.py --ticker-file good.txt --model-type xgb --shap-report-dir reports/shap_quick
```
- **Tiempo**: ~5-10 minutos
- **Objetivo**: Validar que todo funciona
- **Revisar**: SHAP report para confirmar features importantes

### 2. Optimización de Hiperparámetros
```bash
python train_signal_model_v2.py --ticker-file good.txt --optimize-hyperparams --n-trials 50 --shap-report-dir reports/shap_optimized
```
- **Tiempo**: ~30-60 minutos
- **Objetivo**: Encontrar mejores hiperparámetros
- **Revisar**: F1-Score mejorado vs modelo rápido

### 3. Backtest con Modelo Optimizado
```bash
python run_backtest_signal_v2.py \
    --ticker-file good.txt \
    --model-path models/trend_model_2015_2024_optimized.joblib \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --min-confidence 0.50
```
- **Datos**: Out-of-sample (2023-2024)
- **Objetivo**: Validar performance real
- **Esperar**: Más trades que con modelo anterior, mejor Sharpe

### 4. Optimización de Estrategia
```bash
python optimize_strategy_v2.py
```
- **Usa**: Modelo optimizado recién entrenado
- **Busca**: Mejores parámetros de trading (confidence, k_atr, etc)

---

## 🎓 Próximos Pasos Avanzados

Una vez tengas el modelo base funcionando bien:

### 1. **Multi-Horizon Models**
Entrenar modelos para diferentes horizontes:
- Corto plazo: 5 días
- Medio plazo: 10 días
- Largo plazo: 20 días

### 2. **Ensemble de Modelos**
Combinar predicciones:
```python
# XGBoost + RandomForest + LightGBM
pred_final = (pred_xgb + pred_rf + pred_lgbm) / 3.0
```

### 3. **Feature Engineering Avanzado**
- Sector rotation indicators
- Market breadth signals
- Options flow data (si disponible)
- Sentiment analysis

### 4. **Walk-Forward Optimization**
- Reentrenamiento cada 6 meses
- Validación temporal continua
- Adaptive hyperparameters

---

## ⚠️ Notas Importantes

1. **Overfitting Risk**:
   - Con 8 años de datos, el riesgo es menor
   - Siempre valida en out-of-sample (2023-2024)
   - Usa Purged K-Fold para evitar data leakage

2. **Computing Resources**:
   - Optimización Optuna es CPU-intensive
   - Usa `--n-jobs -1` para paralelizar
   - Considera Google Colab si tu PC es lento

3. **Model Drift**:
   - Los mercados cambian con el tiempo
   - Reentrenar cada 6-12 meses
   - Monitorear performance en producción

---

## 📞 Soporte

Si encuentras problemas:
1. Revisa los logs (nivel INFO muestra todo el progreso)
2. Verifica que tienes todas las dependencias: `pip install optuna shap xgboost lightgbm`
3. Confirma que tus datos tienen el rango 2015-2024

---

*Versión: 2.0*  
*Fecha: 2025-11-22*  
*Framework: XGBoost + Optuna + SHAP*
