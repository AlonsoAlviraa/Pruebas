# 🚀 GUÍA COMPLETA - ENTRENAR EN KAGGLE

**Problema:** Tu PC es lento para entrenar modelos
**Solución:** Usar Kaggle Kernels (GRATIS, 4 cores, 16GB RAM)

---

## 📋 PASO 1: SETUP KAGGLE (5 min)

### 1.1 Crear Cuenta
1. Ve a https://www.kaggle.com
2. Click "Register"
3. Usa tu Gmail o email
4. Verifica email

### 1.2 Verificar Teléfono (IMPORTANTE)
1. Settings → Account
2. Phone Verification → Add phone number
3. **Sin esto no tienes GPU ni recursos completos**

### 1.3 Obtener API Key
1. Settings → API
2. Click "Create New Token"
3. Se descarga `kaggle.json`
4. Moverlo a: `C:\Users\alons\.kaggle\kaggle.json`

---

## 📋 PASO 2: GENERAR DATASET TRIPLE BARRIER (10 min)

**En tu PC local:**

```bash
# Generar dataset completo (~200K samples)
python triple_barrier_labeling.py

# Verificar
python triple_barrier_labeling.py --analyze
```

**Output esperado:**
- `data/triple_barrier_dataset.csv` (~200K+ samples)

---

## 📋 PASO 3: SUBIR DATOS A KAGGLE (5 min)

```bash
# Instalar Kaggle API
pip install kaggle

# Subir datos
upload_to_kaggle.bat
```

**Esto creará:**
- Dataset en: https://www.kaggle.com/datasets/alonsoa/trading-system-data

---

## 📋 PASO 4: CREAR NOTEBOOK EN KAGGLE (5 min)

### 4.1 Crear Notebook
1. Ve a https://www.kaggle.com/code
2. Click "New Notebook"
3. Title: "Triple Barrier Training"

### 4.2 Configurar recursos
1. Settings (derecha) → Accelerator
2. Seleccionar: **GPU P100** (o T4)
3. **Persistence:** ON (importante)

### 4.3 Agregar tu dataset
1. Click "+ Add Data" (derecha)
2. Buscar: "trading-system-data"
3. Click "Add"

### 4.4 Copiar código
1. Abre: `kaggle_train_notebook.py`
2. Copia TODO el contenido
3. Pega en el notebook de Kaggle

---

## 📋 PASO 5: EJECUTAR ENTRENAMIENTO (OFFLINE) ⭐

### Opción A: Ejecutar y esperar (en tu PC)
```
1. Click "Save Version" (arriba derecha)
2. Session Settings:
   - Accelerator: GPU P100
   - Persistence: ON
3. Click "Save & Run All"
4. PUEDES CERRAR EL NAVEGADOR ✓
```

**Duración:** ~2-4 horas
**Email:** Recibirás notificación cuando termine

### Opción B: Ejecutar y monitorear
```
1. Click "Run All" (arriba)
2. Ve progreso en tiempo real
3. Mantén navegador abierto
```

---

## 📋 PASO 6: DESCARGAR MODELO ENTRENADO (2 min)

Cuando termine (recibes email):

1. Ve al notebook
2. Output (derecha) → Data
3. Descargar:
   - `trend_model_triple_barrier.joblib` ← Modelo
   - `model_metadata.json` ← Métricas
   - `feature_importance.csv` ← Features

4. Copiar a tu proyecto:
```bash
# Poner en:
models/trend_model_triple_barrier.joblib
```

---

## 📋 PASO 7: PROBAR MODELO LOCAL (5 min)

```bash
# Modificar backtest para usar nuevo modelo
python backtest_trend_following.py \
    --model models/trend_model_triple_barrier.joblib

# Comparar vs modelo anterior
```

---

## 🎯 COMPARACIÓN ESPERADA

### MODELO ACTUAL:
```
F1-Score BUY: 0.0812
Dataset: 16.7% BUY (desbalanceado)
```

### MODELO CON TRIPLE BARRIER:
```
F1-Score BUY: 0.12-0.15 (esperado)
Dataset: 37% BUY (balanceado)
Win Rate Backtest: +3-5%
```

---

## 💡 TIPS KAGGLE

### ✅ Ventajas:
- **OFFLINE:** "Save & Run All" → Cierra PC
- **Email:** Te avisa cuando termina
- **Estable:** 12 horas continuas
- **Gratis:** 30-40h/semana

### ⚠️ Limitaciones:
- **12h máximo** por sesión
- Si tarda más → dividir en 2 partes
- **30-40h semanales** → planifica bien

### 🔧 Troubleshooting:

**Problema:** "Session timed out"
**Solución:** Re-ejecutar, Kaggle guarda progreso

**Problema:** "Dataset not found"
**Solución:** Verificar que agregaste dataset en Step 4.3

**Problema:** "Out of memory"
**Solución:** Reducir n_trials de Optuna (30 → 20)

---

## 🚀 FLUJO COMPLETO RESUMIDO

```
┌─────────────────────────────────┐
│ LOCAL: Generar dataset          │ 10 min
│ python triple_barrier_labeling  │
├─────────────────────────────────┤
│ LOCAL: Subir a Kaggle           │ 5 min
│ upload_to_kaggle.bat            │
├─────────────────────────────────┤
│ KAGGLE: Crear notebook          │ 5 min
│ Copiar código, configurar       │
├─────────────────────────────────┤
│ KAGGLE: "Save & Run All"        │ 2-4h
│ CIERRA PC, recibe email         │
├─────────────────────────────────┤
│ LOCAL: Descargar modelo         │ 2 min
│ Copiar .joblib a models/        │
├─────────────────────────────────┤
│ LOCAL: Probar backtest          │ 5 min
│ Comparar resultados             │
└─────────────────────────────────┘

TOTAL TIEMPO ACTIVO: ~30 min
TOTAL TIEMPO ESPERA: 2-4h (automático)
```

---

## 📊 PRÓXIMOS PASOS

Una vez tengas el modelo entrenado:

1. ✅ **Comparar métricas** (F1-Score, Precision, Recall)
2. ✅ **Backtest** con nuevo modelo
3. ✅ **Si mejora → Usar en producción**
4. ✅ **Si no mejora → Ajustar parámetros Triple Barrier**

---

## 🎯 ALTERNATIVA: GOOGLE COLAB

Si Kaggle no funciona, usa Colab:

```python
# En Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

# Subir dataset a Google Drive
# Ejecutar entrenamiento
# Guardar modelo en Drive
```

**Desventaja:** No puedes cerrar navegador

**Ventaja:** Simpler setup

---

## ❓ PREGUNTAS FRECUENTES

**P: ¿Necesito GPU para XGBoost?**
R: No, pero acelera ~2x. CPU también funciona.

**P: ¿Puedo entrenar otros modelos?**
R: Sí, modifica `kaggle_train_notebook.py` con tu código.

**P: ¿Los datos son privados?**
R: Solo tú ves tu dataset privado. Puede ser público si quieres.

**P: ¿Cuánto cuesta?**
R: **GRATIS** completamente. Kaggle es 100% gratuito.

---

## 🏁 CHECKLIST COMPLETO

- [ ] Cuenta Kaggle creada
- [ ] Teléfono verificado
- [ ] API key descargada (kaggle.json)
- [ ] Dataset Triple Barrier generado
- [ ] Datos subidos a Kaggle
- [ ] Notebook creado en Kaggle
- [ ] Código copiado
- [ ] GPU P100 seleccionada
- [ ] "Save & Run All" ejecutado
- [ ] Email recibido (termino)
- [ ] Modelo descargado
- [ ] Backtest ejecutado
- [ ] Resultados comparados

---

**¿Listo para empezar?** 

Ejecuta en orden:
```bash
1. python triple_barrier_labeling.py
2. upload_to_kaggle.bat
3. Crea notebook en Kaggle
4. "Save & Run All"
```

🚀 ¡Éxito!
