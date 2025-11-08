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
class IndicatorConfig:
    """Configuración para indicadores técnicos."""
    rsi_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_length: int = 20
    bollinger_std: float = 2.0


@dataclass
class PipelineConfig:
    """Configuración del pipeline de datos."""
    data_root: Path = Path("data")
    date_column: str = "date"
    summary_filename: str = "_summary.json"
    history_filename: str = "_history.csv"
    fundamentals_lookback_days: int = 90
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)


class DataPipeline:
    """Carga, combina y transforma datos de precios, fundamentales y resúmenes."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _load_history(self, ticker: str) -> pd.DataFrame:
        """Carga el historial de precios desde un archivo CSV cacheado."""
        file_path = self.config.data_root / f"{ticker}{self.config.history_filename}"
        if not file_path.exists():
            raise FileNotFoundError(f"Price file not found for {ticker}: {file_path}")

        df = pd.read_csv(file_path)
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
        return df.set_index(self.config.date_column).sort_index()

    def _load_summary(self, ticker: str) -> Dict[str, Any]:
        """Carga el resumen de la compañía desde un archivo JSON cacheado."""
        file_path = self.config.data_root / f"{ticker}{self.config.summary_filename}"
        if not file_path.exists():
            return {}
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load or parse summary file %s: %s", file_path, e)
            return {}

    def _flatten_json(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Aplana una estructura JSON anidada para el DataFrame."""
        flat = {}
        for key, value in data.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_json(value, new_key))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat[new_key] = value
        return flat

    def load_feature_view(
        self,
        ticker: str,
        indicators: bool = True,
        include_summary: bool = False,
    ) -> pd.DataFrame:
        """
        Construye la vista de características combinada para un ticker.
        """
        # 1. Cargar historial de precios
        try:
            merged = self._load_history(ticker)
        except FileNotFoundError as e:
            logger.error("Data loading failed for %s: %s", ticker, e)
            raise

        # 2. (Opcional) Añadir indicadores técnicos
        if indicators:
            cfg = self.config.indicators
            merged.ta.rsi(length=cfg.rsi_length, append=True)
            merged.ta.macd(fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal, append=True)
            bbands = merged.ta.bbands(length=cfg.bollinger_length, std=cfg.bollinger_std, append=False)
            if bbands is not None and not bbands.empty:
                merged["bb_upper"] = bbands[f"BBU_{cfg.bollinger_length}_{cfg.bollinger_std}"]
                merged["bb_lower"] = bbands[f"BBL_{cfg.bollinger_length}_{cfg.bollinger_std}"]
                merged["bb_width"] = (merged["bb_upper"] - merged["bb_lower"]) / merged["close"]

        # 3. (Opcional) Cargar y aplanar datos fundamentales/resumen
        if include_summary:
            summary = self._load_summary(ticker)
            flat_summary = self._flatten_json(summary, "summary")
            # >>> Cambiado: en vez de asignar columna a columna, construimos un bloque y concatenamos
            if flat_summary:
                idx = merged.index
                new_cols: Dict[str, pd.Series] = {}
                for key, value in flat_summary.items():
                    if key in merged.columns:
                        continue
                    # Si 'value' es vector del mismo largo que merged, lo alineamos; si es escalar, replicamos
                    if isinstance(value, pd.Series):
                        s = value.reindex(idx)
                    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == len(idx):
                        s = pd.Series(value, index=idx, name=key)
                    else:
                        s = pd.Series([value] * len(idx), index=idx, name=key)
                    s.name = key
                    new_cols[key] = s
                if new_cols:
                    merged = pd.concat([merged, pd.DataFrame(new_cols)], axis=1)
                    merged = merged.copy()  # defragmenta

        # Asegurar que el índice sea la columna de fecha para el entorno
        # (El downloader guarda la fecha como índice, pd.read_csv la carga como columna 'date')
        merged = merged.set_index(self.config.date_column).sort_index()

        # Eliminar NaNs solo en las columnas críticas (precios + indicadores) sin
        # descartar filas antiguas cuyo único "hueco" proviene de fundamentales
        # caducados. Antes usábamos ``dropna()`` sin argumentos, lo que obligaba
        # a que *todas* las columnas estuviesen completas. Como los fundamentales
        # se invalidan pasado ``fundamentals_lookback_days`` y se rellenan con
        # NaN, terminábamos quedándonos únicamente con ~90 filas recientes. Eso
        # provocaba que el entorno tuviese episodios diminutos y el entrenamiento
        # se volviese extremadamente lento al necesitar reiniciar el episodio
        # decenas de veces para completar un batch de RLlib.

        price_cols = [
            col
            for col in ("open", "high", "low", "close", "volume")
            if col in merged.columns
        ]
        indicator_cols = []
        if indicators:
            indicator_cols = [
                col
                for col in (
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "bb_upper",
                    "bb_lower",
                    "bb_width",
_                )
                if col in merged.columns
            ]

        dropna_subset = price_cols + indicator_cols
        if dropna_subset:
            merged = merged.dropna(subset=dropna_subset)
        merged = merged.reset_index(drop=False)

        # Renombrar la columna 'index' (que era la fecha) de nuevo a 'date'
        if "index" in merged.columns and self.config.date_column not in merged.columns:
            merged = merged.rename(columns={"index": self.config.date_column})

        logger.info("Feature view for %s contains %d rows", ticker, len(merged))
        return merged

    def load_feature_views(
        self,
        tickers: Iterable[str],
        indicators: bool = True,
        include_summary: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Batch load multiple tickers into feature views."""

        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.load_feature_view(
                    ticker, indicators=indicators, include_summary=include_summary
                )
            except FileNotFoundError as e:
                logger.error("Failed to load data for ticker %s: %s", ticker, e)
            except Exception as e:
                logger.warning("Unhandled error loading %s: %s", ticker, e)
        return results

    def _normalise_ticker(self, raw: str) -> str | None:
        """
        Normaliza los strings de tickers, eliminando espacios (incl. Unicode) y BOMs.
        """
        if raw is None:
            return None
        try:
            normalised = unicodedata.normalize("NFKC", raw)
            cleaned = normalised.replace("\ufeff", "").strip()
        except TypeError:
            return None

        cleaned = "".join(ch for ch in cleaned if not ch.isspace())
        if not cleaned:
            return None
        return cleaned.upper()