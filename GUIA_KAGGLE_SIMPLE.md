# 🚀 GUÍA SIMPLIFICADA - ENTRENAR EN KAGGLE

**Plan Nuevo (SIMPLE):**
1. Subir solo archivos CSV raw a Kaggle
2. TODO lo demás en Kaggle (generar dataset + entrenar)
3. Descargar solo modelo final

---

## 📋 PASO 1: SETUP KAGGLE (5 min)

### 1.1 Crear cuenta
- https://www.kaggle.com → Register

### 1.2 Configurar Token
1. Settings → API → Create New API Token
2. Copiar token: `KGAT_...`
3. En PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable('KAGGLE_API_TOKEN', 'TU_TOKEN_AQUI', 'User')
$env:KAGGLE_API_TOKEN = "TU_TOKEN_AQUI"
```

4. Verificar:
```bash
pip install kaggle
kaggle competitions list
```

---

## 📋 PASO 2: SUBIR DATOS RAW A KAGGLE (10 min)

Solo necesitas subir tus archivos CSV (data/*.csv).

### Opción A: Subir Manual (MÁS FÁCIL) ⭐

1. Ve a https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Title: `trading-raw-data`
4. Arrastra carpeta `data/` completa (todos los CSV)
5. Click "Create"

**Listo!** Tu dataset está en:
```
https://www.kaggle.com/datasets/TU_USERNAME/trading-raw-data
```

### Opción B: Subir con CLI

```bash
# Desde tu carpeta data/
cd data
kaggle datasets create -p . --dir-mode zip
```

---

## 📋 PASO 3: CREAR NOTEBOOK EN KAGGLE (5 min)

1. Ve a https://www.kaggle.com/code
2. Click "New Notebook"
3. Title: "Triple Barrier Training"

### Configurar:
- Settings → **GPU P100** (o T4)
- Persistence: **ON**

### Agregar Dataset:
1. Click "+ Add Data" (derecha)
2. Buscar: `trading-raw-data` (el que subiste)
3. Click "Add"

### Copiar Código:
1. Abre: `kaggle_notebook_completo.py`
2. Copia TODO
3. Pega en Kaggle notebook

### IMPORTANTE - Ajustar Path:
Encuentra esta línea:
```python
DATA_PATH = Path('/kaggle/input/your-dataset-name')  # CAMBIAR ESTO
```

Cambia por:
```python
DATA_PATH = Path('/kaggle/input/trading-raw-data')  # Tu dataset
```

---

## 📋 PASO 4: EJECUTAR (OFFLINE) ⭐

```
1. Click "Save Version"
2. Settings:
   - Accelerator: GPU P100
   - Persistence: ON
3. Click "Save & Run All"
4. CIERRA NAVEGADOR ✓
```

**Duración:** 2-4 horas
**Email:** Recibirás cuando termine

---

## 📋 PASO 5: DESCARGAR MODELO (2 min)

Cuando recibas el email:

1. Ve al notebook
2. Output → Data
3. Descargar:
   - `trend_model_triple_barrier.joblib`
   - `model_metadata.json`

4. Copiar a tu proyecto:
```
models/trend_model_triple_barrier.joblib
```

---

## 📋 PASO 6: PROBAR MODELO (5 min)

```bash
# Modificar backtest para usar nuevo modelo
python backtest_trend_following.py \
    --model models/trend_model_triple_barrier.joblib
```

Comparar vs modelo anterior.

---

## 🎯 VENTAJAS DE ESTE MÉTODO:

✅ **No genera dataset en tu PC** (lento)
✅ **No sube dataset grande** (200MB+)
✅ **Todo en Kaggle** (4 cores, 16GB RAM)
✅ **Offline** (cierra navegador)
✅ **Solo descarga modelo** (~10MB)

---

## 📊 COMPARACIÓN

| Método | Tu Tiempo | Espera | Upload |
|--------|-----------|--------|--------|
| **Local** | 30 min | 8-16h | 0 MB |
| **Método Viejo** | 20 min | 2-4h | 200 MB |
| **Método Nuevo** ⭐ | 15 min | 2-4h | 50 MB |

---

## ⚠️ TROUBLESHOOTING

**Problema:** "Dataset not found"
**Solución:** Verificar path en línea `DATA_PATH = ...`

**Problema:** "No module pandas_ta"
**Solución:** En notebook agregar: `!pip install pandas_ta`

**Problema:** "Out of memory"
**Solución:** Procesar menos tickers (modificar código)

---

## 🏁 RESUMEN RÁPIDO

```bash
# 1. Configurar token (PowerShell)
[System.Environment]::SetEnvironmentVariable('KAGGLE_API_TOKEN', 'TU_TOKEN', 'User')

# 2. Subir data/ a Kaggle (manual o CLI)
# → https://www.kaggle.com/datasets/new

# 3. Crear notebook, copiar código

# 4. Ajustar path en notebook

# 5. "Save & Run All" → Cerrar navegador

# 6. Esperar email (2-4h)

# 7. Descargar modelo

# 8. Probar backtest
```

---

## 🎯 RESULTADO ESPERADO

```
MODELO ACTUAL:
F1-Score: 0.0812

MODELO TRIPLE BARRIER:
F1-Score: 0.12-0.15
Win Rate: +3-5%
```

---

**¿Listo?** Sube tu carpeta `data/` a Kaggle y empezamos. 🚀
