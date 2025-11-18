#!/usr/bin/env python3
"""Data pipeline for loading and transforming financial data."""
from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
            # Si no hay True, devuelve 0. Hay que chequear si el 0 es realmente un cruce.
            
            hit_upper = window >= upper
            hit_lower = window <= lower
            
            if not hit_upper.any() and not hit_lower.any():
                out_labels[t] = 0 # Hold (Time barrier)
                continue
                
            # Encontrar índices relativos del primer toque
            idx_upper = np.argmax(hit_upper) if hit_upper.any() else horizon + 1
            idx_lower = np.argmax(hit_lower) if hit_lower.any() else horizon + 1
            
            if idx_upper < idx_lower:
                out_labels[t] = 1 # Buy
            elif idx_lower < idx_upper:
                out_labels[t] = -1 # Sell / Stop
            else:
                # Empate (mismo día), priorizamos stop loss por conservadurismo o hold?
                # Asumimos Stop Loss ocurre primero en el peor caso intradía
                out_labels[t] = -1
                
        # Asignar al dataframe
        result = df.copy()
        result["label"] = out_labels
        
        # Los últimos 'horizon' días no tienen etiqueta válida (no sabemos el futuro)
        # Podemos ponerlos a 0 o NaN. Para entrenamiento supervisado, mejor eliminar esas filas después.
        # Aquí las dejamos como 0 (HOLD) por defecto o NaN si se prefiere.
        # Vamos a marcar con NaN para que el entrenador sepa descartarlas.
        result.loc[result.index[-horizon:], "label"] = np.nan
        
        return result