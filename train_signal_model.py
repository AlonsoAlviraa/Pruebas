#!/usr/bin/env python3
"""
Script de entrenamiento para el modelo de señales (Event-Driven).
Usa Triple Barrier Method, Class Balancing y Purged K-Fold CV.
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.base import clone

# Intentar importar XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from drl_platform.validation import PurgedKFoldValidator, PurgedKFoldConfig

# Configurar logging
logging.basicConfig(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth
    )
    
    # Validación
    run_validation(X, y, model)
    
    # Entrenamiento Final
    logger.info("Entrenando modelo final con todos los datos...")
    model.fit(X, y)
    
    # Guardar
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    logger.info(f"Modelo guardado en {args.output}")
    
    # Reporte final en datos de entrenamiento (sanity check)
    y_pred = model.predict(X)
    logger.info("Reporte de Clasificación (Training Set - Sanity Check):")
    logger.info("\n" + classification_report(y, y_pred))

if __name__ == "__main__":
    main()