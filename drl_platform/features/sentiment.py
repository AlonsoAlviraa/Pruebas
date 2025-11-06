#!/usr/bin/env python3
"""Utilities for integrating NLP-driven sentiment features into the state."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentConfig:
    """Configuración para el extractor de sentimiento."""
    model_name: str = "ProsusAI/finbert"
    cache_dir: Optional[Path] = None


class SentimentFeatureExtractor:
    """Extracts sentiment scores from textual datasets using an NLP model."""

    def __init__(self, config: SentimentConfig):
        self.config = config
        self._model = None  # Lazy-loading del modelo

    def _load_model(self):  # pragma: no cover - heavyweight dependency
        """Carga el modelo de transformers bajo demanda."""
        if self._model is not None:
            return self._model
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "transformers y torch son necesarios para la extracción de sentimiento. "
                "Instálalos con `pip install transformers torch`"
            ) from exc
            
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        
        # device=0 para GPU si está disponible, -1 para CPU
        device = 0 if torch.cuda.is_available() else -1
        self._model = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer, device=device
        )
        logger.info(f"Sentiment model '{self.config.model_name}' loaded on device {'GPU' if device == 0 else 'CPU'}.")
        return self._model

    def transform(self, documents: Iterable[str]) -> List[float]:  # pragma: no cover - heavy
        """Ejecuta el pipeline de transformers sobre una lista de textos."""
        pipeline = self._load_model()
        # Truncation=True es importante para manejar textos largos
        outputs = pipeline(list(documents), truncation=True)
        scores = [self._map_output(output) for output in outputs]
        return scores

    @staticmethod
    def _map_output(output: Dict[str, Any]) -> float:
        """Convierte la salida de FinBERT (positivo, negativo, neutral) a un score de -1.0 a 1.0."""
        label = output.get("label", "neutral").lower()
        score = float(output.get("score", 0.0))
        if "positive" in label:
            return score
        if "negative" in label:
            return -score
        return 0.0  # Neutral

    def merge_with_prices(
        self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame, date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Une un DataFrame de precios con un DataFrame de sentimiento usando merge_asof.
        
        Asegura que solo se use el sentimiento 'conocido' en cada día.
        """
        sentiment_df_copy = sentiment_df.copy()
        
        if date_col not in sentiment_df_copy.columns:
            raise KeyError(f"Sentiment DataFrame must contain '{date_col}' column")
        if date_col not in price_df.columns:
            raise KeyError(f"Price DataFrame must contain '{date_col}' column")

        sentiment_df_copy[date_col] = pd.to_datetime(sentiment_df_copy[date_col], utc=True)
        sentiment_df_copy = sentiment_df_copy.sort_values(date_col)
        
        price_df_copy = price_df.sort_values(date_col)

        column = "sentiment_score"
        if column not in sentiment_df_copy.columns:
            if "score" in sentiment_df_copy.columns:
                sentiment_df_copy[column] = sentiment_df_copy["score"]
            else:
                raise KeyError("Sentiment DataFrame must contain 'sentiment_score' or 'score' column")

        # Unir el precio de cada día con el último sentimiento disponible
        merged = pd.merge_asof(
            price_df_copy,
            sentiment_df_copy[[date_col, column]].sort_values(date_col),
            on=date_col,
            direction="backward",  # Hacia atrás (último valor conocido)
        )
        
        # Rellenar los NaNs al inicio (antes del primer score) con 0 (neutral)
        merged[column] = merged[column].fillna(0.0)
        return merged