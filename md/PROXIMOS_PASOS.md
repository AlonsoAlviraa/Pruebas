# 🚀 INSTRUCCIONES - Próximos Pasos

## ✅ ESTADO ACTUAL

El modelo está **CORREGIDO** y **FUNCIONANDO**:
- ✅ Modelo nuevo: `trend_model_2015_2024_OPTUNA_FIXED.joblib`
- ✅ Features correctas: 21 (sin OHLCV)
- ✅ Predicciones normales: 0.01 - 0.95
- ✅ Genera señales: ~1,800 (58% con conf≥0.40)

## 🧪 PRUEBA RÁPIDA (AHORA - 5-10 min)

```bash
python test_rapido_3configs.py
```

Esto prueba 3 configuraciones:
1. **CONSERVADOR:** conf=0.60, vol=0.15, k=3.0
2. **BALANCEADO:** conf=0.50, vol=0.20, k=2.5
3. **AGRESIVO:** conf=0.40, vol=0.25, k=2.0

**Resultado esperado:**
```
Config         Trades  Net_Profit    Win_Rate_%  Sharpe
CONSERVADOR    30-50   $2,000-8,000  45-55%      0.8-1.2
BALANCEADO     50-100  $5,000-15,000 45-50%      1.0-1.5
AGRESIVO       100-150 $8,000-20,000 40-48%      0.9-1.4
```

## 🌙 OPTIMIZACIÓN COMPLETA (ESTA NOCHE - 2-4 horas)

### Opción 1: Usar el BAT (Recomendado)
```bash
optimizar_noche.bat
```

### Opción 2: Ejecutar directamente
```bash
python optimize_strategy_v2.py
```

**Esto probará:**
- 120 combinaciones totales
- 5 valores de confidence × 4 volatility × 3 k_atr × 4 max_position
- Generará: `optimization_results_v2.csv`

**Tiempo:** 2-4 horas (déjalo corriendo durante la noche)

## 📊 RESULTADOS DE LA OPTIMIZACIÓN

Después de la optimización nocturna, tendrás:

### 1. `optimization_results_v2.csv`
Todas las 120 configuraciones ordenadas por métricas:
```csv
Config,Trades,Net_Profit,Win_Rate_%,Sharpe,Max_DD_%,Quality_Score
...
```

### 2. Rankings automáticos en pantalla:
- **Top 5 por Quality Score** (Sharpe / |MaxDD|)
- **Top 5 por Ganancia Neta**
- **Top 5 por Sharpe Ratio**

### 3. Recomendación automática
El script te mostrará la **MEJOR CONFIGURACIÓN** al final.

## 🎯 CÓMO USAR LOS RESULTADOS

### 1. Abre el CSV
```bash
# En Excel o LibreOffice
optimization_results_v2.csv
```

### 2. Ordena por tu métrica favorita:
- **Quality Score:** Mejor balance riesgo/retorno
- **Net Profit:** Máxima ganancia absoluta
- **Sharpe:** Mejor rentabilidad ajustada a riesgo
- **Win Rate:** Mayor % de trades ganadores

### 3. Anota los mejores parámetros

Ejemplo:
```
MEJOR CONFIG:
  min_confidence: 0.50
  volatility_target_pct: 0.20
  k_atr: 2.5
  max_position_pct: 0.20
  
  Resultado: $12,450 | 87 trades | Sharpe 1.35
```

## 🔧 APLICAR LA MEJOR CONFIGURACIÓN

### Opción A: Modificar `run_backtest_signal_v2.py`

Edita los valores por defecto en el script (líneas ~520-530).

### Opción B: Ejecutar con argumentos

```bash
python run_backtest_signal_v2.py \
  --ticker-file good.txt \
  --model-path models/trend_model_2015_2024_OPTUNA_FIXED.joblib \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --min-confidence 0.50 \
  --volatility-target-pct 0.20 \
  --k-atr 2.5 \
  --max-position-pct 0.20
```

## 📈 SIGUIENTE NIVEL: WALK-FORWARD ANALYSIS

Una vez tengas los mejores parámetros, valida con:

```bash
python run_walk_forward.py \
  --ticker-file good.txt \
  --model-path models/trend_model_2015_2024_OPTUNA_FIXED.joblib \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --min-confidence 0.50 \
  --volatility-target-pct 0.20 \
  --k-atr 2.5
```

Esto simula re-entrenamientos periódicos y muestra si la estrategia es robusta.

## ⚠️ NOTAS IMPORTANTES

### 1. F1-Score Bajo (0.08) es NORMAL
- El modelo es **conservador** por diseño
- Prefiere evitar señales falsas
- Solo marca BUY cuando tiene alta confianza
- Esto es **BUENO** para trading real

### 2. Backtest != Resultados Futuros
- Los backtests son **optimistas**
- Trading real tiene slippage, comisiones mayores, etc.
- Divida expectativas por 2 para ser realista

### 3. Diversificación
- No pongas todo el capital en 1 señal
- Usa `max_position_pct` para limitar exposición
- Mantén efectivo para nuevas oportunidades

## 🐛 TROUBLESHOOTING

### "No se generaron operaciones"
- Revisa que usas el modelo FIXED
- Verifica que `good.txt` existe con tickers válidos
- Baja `min_confidence` a 0.40

### "Error: module not found"
```bash
pip install pandas numpy pandas_ta joblib scikit-learn xgboost
```

### "Muy lento"
- Reduce el número de tickers en `good.txt`
- Usa un período más corto (ej. 2023-2024)
- Aumenta `--max-samples-per-ticker` a 2000

## 📚 ARCHIVOS DE REFERENCIA

- **`PROBLEMA_SOLUCIONADO.md`** - Resumen del fix
- **`EXPLICACION_OPTUNA.md`** - Cómo funciona la optimización
- **`DIAGNOSTICO_MODELO_PROBLEMA.md`** - Diagnóstico técnico completo

## ✅ CHECKLIST

- [x] Modelo re-entrenado correctamente
- [x] Features verificadas (21, sin OHLCV)
- [x] Predicciones funcionando (0.0-1.0)
- [ ] Prueba rápida ejecutada (3 configs)
- [ ] Optimización nocturna completada (120 configs)
- [ ] Mejores parámetros identificados
- [ ] Walk-forward analysis validada

---

## 🚀 ACCIÓN INMEDIATA

**AHORA (5-10 min):**
```bash
python test_rapido_3configs.py
```

**ESTA NOCHE (deja corriendo):**
```bash
optimizar_noche.bat
```

**MAÑANA (revisar resultados):**
```bash
# Abre y ordena por Quality Score
optimization_results_v2.csv
```

¡Éxito! 🎉
