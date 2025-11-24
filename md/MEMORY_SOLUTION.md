# 🛠️ SOLUCIÓN AL PROBLEMA DE MEMORIA - Training v2.0

## ❌ Problema Original

```
numpy._core._exceptions._ArrayMemoryError: 
Unable to allocate 18.8 MiB for an array with shape (1, 2458828)
```

**Causa**: Intentar cargar 2.4 millones de filas en RAM (~1-2 GB) excedía la memoria disponible.

---

## ✅ Solución Implementada

###  1. **Control de Tickers**
```bash
--max-tickers 100
```
- Limita cuántos tickers cargar
- Recomendado: 50-100 para sistemas con 8GB RAM
- Puedes aumentar si tienes más RAM

### 2. **Límite por Ticker**
```bash
--max-samples-per-ticker 2000
```
- Toma las 2000 muestras más recientes de cada ticker
- Suficiente para capturar ~8 años de datos diarios
- Recomendado: 1000-3000

### 3. **Filtro de Calidad**
```bash
--min-ticker-samples 500
```
- Descarta tickers con menos de 500 muestras (~2 años)
- Evita ruido de tickers con pocos datos
- Recomendado: 252-500 (1-2 años)

### 4. **Sampling Final**
```bash
--sample-ratio 0.5
```
- Usa 50% del dataset final
- Acelera entrenamiento manteniendo representatividad
- Valores: 0.3-1.0 (30%-100%)

---

## 📊 Comparación de Configuraciones

| Configuración | RAM Requerida | Tiempo Entrena | Muestras Totales | Recomendado Para |
|--------------|---------------|----------------|------------------|------------------|
| **FULL** (sin límites) | ~2-4 GB | ~30-60 min | ~500K-1M | Servidores, 16GB+ RAM |
| **MEDIUM** (100 tickers, 3000/ticker) | ~1-1.5 GB | ~15-30 min | ~150K-300K | PC 8-16GB RAM |
| **LIGHT** (50 tickers, 2000/ticker, 50% sample) | ~500 MB | ~5-10 min | ~50K-100K | PC 4-8GB RAM | ← **ACTUAL**
| **MINIMAL** (30 tickers, 1000/ticker, 30% sample) | ~200 MB | ~2-5 min | ~10K-30K | Pruebas rápidas |

---

## 🎯 Configuraciones Recomendadas

### Para 4GB RAM (Laptop Básico):
```bash
python train_signal_model_v2.py \
    --ticker-file good.txt \
    --max-tickers 30 \
    --max-samples-per-ticker 1000 \
    --min-ticker-samples 300 \
    --sample-ratio 0.3 \
    --n-estimators 100 \
    --max-depth 5
```

### Para 8GB RAM (PC Standard):
```bash
python train_signal_model_v2.py \
    --ticker-file good.txt \
    --max-tickers 100 \
    --max-samples-per-ticker 2000 \
    --min-ticker-samples 500 \
    --sample-ratio 0.5 \
    --n-estimators 200 \
    --max-depth 6
```
← **CONFIGURACIÓN ACTUAL**

### Para 16GB+ RAM (Workstation):
```bash
python train_signal_model_v2.py \
    --ticker-file good.txt \
    --max-tickers 300 \
    --max-samples-per-ticker 5000 \
    --min-ticker-samples 252 \
    --sample-ratio 1.0 \
    --n-estimators 300 \
    --max-depth 8
```

### Para Servidor (32GB+):
```bash
python train_signal_model_v2.py \
    --ticker-file good.txt \
    --max-samples-per-ticker 10000 \
    --sample-ratio 1.0 \
    --optimize-hyperparams \
    --n-trials 100
```

---

## 🔍 Monitoreo de Memoria Durante Entrenamiento

El script ahora muestra:
```
Dataset maestro final: 98,543 muestras de 87 tickers
Uso de memoria del dataset: 547.3 MB
```

**Qué observar**:
- `muestras`: Debe ser 50K-200K idealmente
- `memoria`: Debe ser < 1GB para sistemas de 8GB RAM
- `tickers`: Más tickers = mayor diversidad, pero más RAM

---

## ⚡ Optimizaciones Adicionales

### 1. **Usar RandomFor est en vez de XGBoost**
```bash
--model-type rf
```
- RF usa ~30-40% menos RAM que XGB
- Puede ser más lento pero más estable

### 2. **Reducir Features (si es necesario)**
Edita `data_pipeline.py` para reducir indicadores:
```python
# Comentar indicators muy pesados
# ta.bbands(), ta.ichi() etc
```

### 3. **Incrementar `purge_window`**
```bash
--purge-window 20
```
- Reduce overlap entre folds
- Menos datos duplicados en memoria

### 4. **Usar Menos Folds**
```bash
--n-splits 3
```
- 3 folds en vez de 5
- 40% menos uso de RAM durante CV

---

## 📈 Impacto en Performance del Modelo

**¿Entrenar con menos datos empeora el modelo?**

**No necesariamente**:
- 100K muestras bien seleccionadas > 1M muestras ruidosas
- La diversidad de tickers importa más que el volumen
- Los filtros de calidad (`min_ticker_samples`) ayudan

**Métricas esperadas**:

| Dataset Size | F1-Score Esperado | Notas |
|-------------|-------------------|-------|
| 500K+ samples | 0.45-0.55 | Óptimo, pero requiere >16GB RAM |
| 100K-300K | 0.40-0.50 | **Muy bueno**, balance ideal |
| 50K-100K | 0.35-0.45 | Aceptable, puede tener más varianza |
| <50K | 0.25-0.40 | Riesgo de overfitting, solo para pruebas |

---

## 🚨 Troubleshooting

### Error persiste: "Unable to allocate memory"
**Solución**:
1. Reduce `--max-tickers` a 50
2. Reduce `--sample-ratio` a 0.3
3. Usa `--model-type rf`
4. Cierra otros programas (navegador, etc)

### Warning: "Ticker X skipped: solo Y muestras"
**Normal**: Algunos tickers no tienen suficientes datos históricos.
**Si es excesivo** (>50% skipped):
- Reduce `--min-ticker-samples` a 252
- Verifica calidad de los datos descargados

### Performance degradado
**Si F1 < 0.30**:
1. Aumenta `--max-tickers` si te lo permite la RAM
2. Aumenta `--sample-ratio`
3. Revisa que tienes diversidad de sectores en `good.txt`

---

## 📝 Ejemplo Completo (Usado Ahora)

```bash
python train_signal_model_v2.py \
    --ticker-file good.txt \
    --output models/trend_model_2015_2024_quick.joblib \
    --train-from 2015-01-01 \
    --train-until 2022-12-31 \
    --horizon 10 \
    --model-type xgb \
    --n-estimators 200 \
    --max-depth 6 \
    --learning-rate 0.05 \
    --class-weighted \
    --max-tickers 100 \
    --max-samples-per-ticker 2000 \
    --min-ticker-samples 500 \
    --sample-ratio 0.5 \
    --shap-report-dir reports/shap_v2_quick
```

**Resultado esperado**:
- Muestras: ~80K-120K
- RAM: ~500-800 MB
- Tiempo: ~8-15 minutos
- F1-Score: ~0.38-0.48

---

## 🎓 Recomendaciones Finales

1. **Empieza con LIGHT** (configuración actual)
2. **Valida que funciona** (revisa F1-Score)
3. **Si todo va bien**, aumenta gradualmente:
   - Primero `max_tickers`
   - Luego `sample_ratio`
   - Finalmente `max_samples_per_ticker`
4. **Monitorea RAM** con Task Manager mientras entrena
5. **Si falla**, reduce parámetros y vuelve a intentar

---

*Versión: 2.0*  
*Optimizado para: 8GB RAM*  
*Configuración: LIGHT (100 tickers, 50% sample)*
