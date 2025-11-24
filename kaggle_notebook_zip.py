"""
NOTEBOOK COMPLETO KAGGLE - TODO EN UNO
1. Descomprime datos (si están en ZIP)
2. Genera dataset Triple Barrier  
3. Entrena modelo con Optuna
4. Guarda modelo entrenado

INSTRUCCIONES:
1. Sube trading_data.zip a Kaggle
2. Copia este código
3. "Save & Run All"
4. Descarga modelo
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm.notebook import tqdm
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score
import zipfile

print("✓ Imports completados")

# ==========================================
# DESCOMPRIMIR DATOS
# ==========================================

DATA_PATH = Path('/kaggle/input/trading-raw-data-compressed')

# Buscar ZIP
zip_files = list(DATA_PATH.glob('*.zip'))

if zip_files:
    print(f"\n[Descomprimiendo {zip_files[0].name}...]")
    temp_path = Path('/kaggle/working/data')
    temp_path.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(temp_path)
    
    DATA_PATH = temp_path
    csv_count = len(list(DATA_PATH.glob('*.csv')))
    print(f"✓ {csv_count} archivos CSV descomprimidos")
else:
    print(f"✓ Usando datos en: {DATA_PATH}")

# ... resto del código igual que antes ...
# (Copiar el resto del kaggle_train_notebook.py aquí)
