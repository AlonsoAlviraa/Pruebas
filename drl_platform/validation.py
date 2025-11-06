#!/usr/bin/env python3
"""Validation harness implementing Purged K-Fold cross validation."""
from __future_ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class PurgedKFoldConfig:
    """Configuración para el validador Purged K-Fold."""
    n_splits: int = 5
    purge_window: int = 5  # número de días/filas a purgar
    date_column: str = "date" # Columna de fecha usada para ordenar


class PurgedKFoldValidator:
    """Implements the Purged K-Fold CV described by López de Prado."""

    def __init__(self, config: PurgedKFoldConfig):
        self.config = config

    def split(self, data: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera tuplas de índices (train, test) para cada fold.
        """
        if self.config.date_column not in data.columns:
            raise KeyError(f"DataFrame must contain '{self.config.date_column}' column")
            
        df = data.sort_values(self.config.date_column).reset_index(drop=True)
        n_samples = len(df)
        
        # Calcular los tamaños de los folds
        fold_sizes = np.full(self.config.n_splits, n_samples // self.config.n_splits, dtype=int)
        fold_sizes[: n_samples % self.config.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = np.arange(start, stop)
            
            # Obtener índices de entrenamiento purgados
            train_indices = self._purge_indices(n_samples, start, stop)
            
            yield train_indices, test_indices
            current = stop

    def _purge_indices(self, n_samples: int, test_start: int, test_stop: int) -> np.ndarray:
        """
        Purga los índices de entrenamiento alrededor del set de prueba.
        """
        purge = self.config.purge_window
        keep = np.ones(n_samples, dtype=bool)
        
        # Purgar al final del test set (para eliminar datos futuros)
        keep[test_start : test_stop + purge] = False
        
        # Purgar al inicio del test set (para eliminar datos pasados cercanos)
        lower_bound = max(0, test_start - purge)
        keep[lower_bound:test_start] = False
        
        return np.where(keep)[0]

    def evaluate(
        self,
        data: pd.DataFrame,
        model_builder: Callable[[pd.DataFrame], Callable[[pd.DataFrame], Dict]],
    ) -> List[Dict]:
        """
        Evalúa un 'model_builder' a través de los folds purgados.

        Parameters
        ----------
        data:
            Dataset completo a splitear.
        model_builder:
            Una función que:
            1. Acepta (train_df)
            2. Entrena un modelo
            3. Devuelve una función `evaluate` que acepta (test_df) y devuelve un dict de métricas.
        """

        results = []
        for fold, (train_idx, test_idx) in enumerate(self.split(data)):
            train_df = data.iloc[train_idx]
            test_df = data.iloc[test_idx]
            
            # 1. Construir/Entrenar el modelo
            evaluate_model = model_builder(train_df)
            
            # 2. Evaluar el modelo
            evaluation = evaluate_model(test_df)
            evaluation["fold"] = fold
            results.append(evaluation)
            
        return results