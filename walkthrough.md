# Guía de Uso: Arquitectura Event-Driven (Sniper)

Esta guía explica cómo utilizar los nuevos scripts para entrenar, validar y ejecutar la estrategia de trading basada en eventos (Triple Barrier Method).

## 1. Preparación del Entorno

Asegúrate de tener las dependencias instaladas y datos en la carpeta `data/`.
Los datos deben ser archivos `TICKER_history.csv` (ej. `AAPL_history.csv`) con columnas `date`, `open`, `high`, `low`, `close`, `volume`.

## 2. Entrenamiento del Modelo (`train_signal_model.py`)

Este script carga los datos, genera etiquetas usando el Método de Triple Barrera, entrena un modelo (RandomForest o XGBoost) y lo guarda.

**Comando Básico:**
```bash
python train_signal_model.py --tickers "AAPL,MSFT,GOOGL,AMZN,TSLA" --horizon 5 --take-profit 0.05 --stop-loss -0.03
```

**Opciones Importantes:**
- `--tickers`: Lista de tickers separados por comas.
- `--data-root`: Carpeta de datos (default: `data`).
- `--model-type`: `rf` (RandomForest) o `xgb` (XGBoost).
- `--output`: Ruta para guardar el modelo (default: `models/signal_model.pkl`).
- `--horizon`: Días máximos de retención.
- `--take-profit` / `--stop-loss`: Umbrales para etiquetar BUY/SELL.

**Ejemplo Avanzado (XGBoost):**
```bash
python train_signal_model.py --tickers "AAPL,MSFT" --model-type xgb --n-estimators 200 --output models/my_xgb_model.pkl
```

## 3. Backtesting Rápido (`run_backtest_signal.py`)

Valida la estrategia simulando operaciones sobre datos históricos usando el modelo entrenado.

**Comando:**
```bash
python run_backtest_signal.py --tickers "AAPL,MSFT,GOOGL" --model-path models/signal_model.pkl --min-confidence 0.6
```

**Salida:**
- Reporte en consola con Win Rate y Retornos.
- Archivo `backtest_signal_results.csv` con el detalle de cada operación simulada.

## 4. Ejecución en Vivo (Simulada) (`async_event_trader.py`)

Ejecuta el bucle de trading asíncrono que escanea tickers y genera señales en tiempo real (o simulado con datos recientes).

**Comando:**
```bash
python async_event_trader.py --model-path models/signal_model.pkl --tickers "AAPL,MSFT" --min-confidence 0.7
```

**Notas:**
- Este script carga los datos más recientes disponibles en `data/`.
- Imprime las decisiones de trading (BUY/HOLD/SELL) en la consola.
- Usa `--output-csv decisions.csv` para guardar un registro de las decisiones.

## Flujo de Trabajo Recomendado

1.  **Entrenar**: Genera un modelo robusto con `train_signal_model.py`.
2.  **Validar**: Usa `run_backtest_signal.py` para ver cómo se habría comportado. Ajusta `min_confidence` según los resultados.
3.  **Operar**: Lanza `async_event_trader.py` para monitorear el mercado.
