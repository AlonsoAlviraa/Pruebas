# 📥 Sistema de Descarga de Datos Mejorado

## Características Principales

### ✨ Mejoras sobre el sistema anterior:

1. **Descarga Paralela**: Hasta 10 threads simultáneos (configurable) → **10x más rápido**
2. **Filtros de Calidad Integrados**:
   - ❌ Excluye SPACs automáticamente (warrants, units, rights)
   - 💰 Precio mínimo configurable (default: $0.50)
   - 📊 Volumen mínimo promedio (default: 100,000 acciones/día)
   - 📅 Mínimo 1 año de datos históricos
3. **Rango Extendido**: Descarga desde 2018 por defecto (vs 2020) → **más datos para entrenar**
4. **Limpieza Automática**: Elimina archivos de tickers rechazados
5. **Lista de Salida**: Genera `good_tickers_filtrados.txt` con solo los tickers aprobados
6. **Manejo de Errores Robusto**: Rate limiting automático + reintentos

## 🚀 Uso Rápido

### Descarga Básica (NASDAQ completo)
```bash
python download_data.py
```
Esto:
- Descarga TODOS los tickers de `nasdaqlisted.txt`
- Aplica filtros de calidad automáticamente
- Guarda solo los buenos en `data/`
- Genera `good_tickers_filtrados.txt`

### Descarga desde Lista Personalizada
```bash
python download_data.py --input tickers.txt
```

### Descarga con Parámetros Personalizados
```bash
python download_data.py \
    --start-date 2015-01-01 \
    --min-price 1.00 \
    --min-volume 250000 \
    --max-workers 20 \
    --force
```

### Descarga Solo Empresas de Alta Calidad
```bash
python download_data.py \
    --min-price 5.00 \
    --min-volume 500000 \
    --start-date 2015-01-01
```

## 📋 Argumentos Disponibles

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--input` | `nasdaqlisted.txt` | Archivo con lista de tickers |
| `--start-date` | `2018-01-01` | Fecha inicial de descarga |
| `--end-date` | Hoy | Fecha final de descarga |
| `--output-dir` | `data` | Directorio de salida |
| `--lag-days` | `45` | Días de lag para fundamentales |
| `--min-price` | `0.50` | Precio mínimo (USD) |
| `--min-volume` | `100000` | Volumen mínimo diario promedio |
| `--max-workers` | `10` | Threads paralelos |
| `--force` | `False` | Re-descargar todo |
| `--output-list` | `good_tickers_filtrados.txt` | Lista de aprobados |

## 📊 Filtros de Calidad

### Automáticamente Excluye:
- ❌ SPACs (tickers terminados en W, U, R, -WT, -UN, -RT)
- ❌ Tickers con precio < $0.50
- ❌ Tickers con volumen < 100,000 acciones/día
- ❌ Tickers con menos de 1 año de datos

### Automáticamente Incluye:
- ✅ QQQ (para régimen de mercado)

## 🔄 Flujo de Trabajo Completo

### 1. Primera Vez - Descarga Completa
```bash
# Descargar NASDAQ completo con filtros de calidad
python download_data.py --max-workers 20
```

### 2. Entrenar Modelo
```bash
# Usar la lista filtrada automáticamente
python train_signal_model.py \
    --ticker-file good_tickers_filtrados.txt \
    --data-root data
```

### 3. Actualizaciones Diarias
```bash
# Solo actualizar datos existentes (rápido)
python download_data.py \
    --input good_tickers_filtrados.txt \
    --start-date 2024-01-01
```

## 📈 Resultados Esperados

### Antes (script antiguo):
- ⏱️  **3341 tickers**: ~2-3 horas (secuencial)
- 🗑️  Incluye basura: SPACs, penny stocks, datos insuficientes
- 🔄  Descarga manual duplicada

### Después (script mejorado):
- ⏱️  **3341 tickers**: ~15-20 minutos (paralelo)
- ✨  Solo empresas de calidad
- 🎯  ~500-800 tickers finales (alta calidad)
- 🚀  **10x más rápido**

## 🛠️ Troubleshooting

### Error: ModuleNotFoundError: a
```bash
# Asegúrate de que existe ANTIGUOPROGRAMA/a.py o copia/mueve los archivos necesarios
```

### Error: Rate Limit
El script maneja automáticamente los rate limits, pero si persiste:
```bash
# Reduce workers
python download_data.py --max-workers 5
```

### Error: nasdaqlisted.txt no encontrado
```bash
# Descarga desde NASDAQ
curl -o nasdaqlisted.txt https://www.nasdaq.com/trading/nasdaq-listed.aspx
# O especifica tu lista
python download_data.py --input mi_lista.txt
```

## 💡 Consejos de Uso

### Para Entrenamiento Rápido (Prototipos)
```bash
python download_data.py \
    --min-price 10.00 \
    --min-volume 1000000 \
    --start-date 2020-01-01
```
Resultado: ~100-200 tickers de muy alta calidad

### Para Entrenamiento Profundo (Producción)
```bash
python download_data.py \
    --min-price 0.50 \
    --min-volume 100000 \
    --start-date 2015-01-01 \
    --max-workers 20
```
Resultado: ~500-800 tickers, máxima historia

### Para Re-entrenar Modelo
```bash
# Usa siempre la lista filtrada existente
python train_signal_model.py \
    --ticker-file good_tickers_filtrados.txt \
    --data-root data
```

## 📝 Notas Importantes

1. **Primera ejecución**: Puede tardar 15-30 minutos dependiendo de cuántos tickers descargues
2. **Actualizaciones**: Son mucho más rápidas (solo nuevos datos)
3. **Archivos generados**:
   - `data/{TICKER}_history.csv` → Precios OHLCV
   - `data/{TICKER}_fundamentals.csv` → EPS y Revenue trimestrales
   - `good_tickers_filtrados.txt` → Lista de tickers aprobados
4. **Limpieza automática**: Los tickers rechazados se eliminan del disco
