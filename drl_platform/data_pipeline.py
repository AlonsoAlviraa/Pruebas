#!/usr/bin/env python3
"""Data pipeline module for loading and aligning heterogeneous financial datasets.

This module provides the :class:`DataPipeline` which is capable of reading
price, fundamentals and summary metadata files from a local cache. The
pipeline aligns data of different frequencies by forward filling the latest
available fundamentals information so that each trading day only has access to
what was known at that time.

The resulting dataset can optionally include a suite of technical indicators
that are computed on demand. The pipeline outputs feature views that are ready
for consumption by the reinforcement learning environments defined in
``drl_platform.env``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration object describing which indicators to compute."""

    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0


@dataclass
class PipelineConfig:
    """Configuration for :class:`DataPipeline`."""

    data_root: Path
    price_pattern: str = "{ticker}.csv"
    fundamentals_pattern: str = "{ticker}_fundamentals.csv"
    summary_pattern: str = "{ticker}_summary.json"
    date_column: str = "date"  # Columna de fecha en el archivo de precios
    available_at_column: str = "available_at" # Columna de 'disponibilidad' en fundamentales
    fundamentals_lookback_days: int = 90


class DataPipeline:
    """Loads raw market data and builds feature views for RL agents."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_root = Path(config.data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist or is not a directory")
        logger.debug("DataPipeline initialized with root %s", self.data_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_feature_view(
        self,
        ticker: str,
        indicators: bool = True,
        include_summary: bool = False,
    ) -> pd.DataFrame:
        """Return an aligned feature view for ``ticker``.

        Parameters
        ----------
        ticker:
            The instrument identifier whose data should be loaded.
        indicators:
            Whether to compute technical indicators as part of the feature view.
        include_summary:
            When ``True`` the JSON summary metadata is joined as static columns.
        """

        price_df = self._load_price_data(ticker)
        fundamental_df = self._load_fundamentals(ticker)
        merged = self._merge_price_and_fundamentals(price_df, fundamental_df)

        if "close" not in merged.columns:
            raise KeyError("Price data must contain 'close' column for indicator computation")

        if indicators:
            merged = self._append_indicators(merged)

        if include_summary:
            summary = self._load_summary(ticker)
            # Aplanar JSON anidado y añadir como columnas
            flat_summary = self._flatten_json(summary, "summary")
            for key, value in flat_summary.items():
                merged[key] = value

        # Asegurar que el índice sea la columna de fecha para el entorno
        merged = merged.set_index(self.config.date_column).sort_index()
        
        # Eliminar filas con NaNs (generalmente al inicio, por los indicadores)
        merged = merged.dropna().reset_index(drop=False)
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
                logger.error("Error processing ticker %s: %s", ticker, e)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_price_data(self, ticker: str) -> pd.DataFrame:
        path = self.data_root / self.config.price_pattern.format(ticker=ticker)
        if not path.exists():
            raise FileNotFoundError(f"Price file not found for {ticker}: {path}")
        df = pd.read_csv(path)
        
        # Normalizar nombres de columnas a minúsculas
        df.columns = [col.lower() for col in df.columns]
        
        if self.config.date_column not in df.columns:
            # Fallback para índices de fecha comunes si 'date' no existe
            for potential_col in ['datetime', 'timestamp', 'time']:
                if potential_col in df.columns:
                    self.config.date_column = potential_col
                    break
            else:
                 raise KeyError(f"Date column '{self.config.date_column}' not found in {path}")

        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column], utc=True)
        df = df.sort_values(self.config.date_column).reset_index(drop=True)
        logger.debug("Loaded price data for %s with %d rows", ticker, len(df))
        return df

    def _load_fundamentals(self, ticker: str) -> pd.DataFrame:
        path = self.data_root / self.config.fundamentals_pattern.format(ticker=ticker)
        if not path.exists():
            logger.warning("Fundamentals file not found for %s: %s", ticker, path)
            return pd.DataFrame(columns=[self.config.available_at_column])
        
        df = pd.read_csv(path)
        available_col = self.config.available_at_column
        
        if available_col not in df.columns:
            logger.warning("'%s' not found in fundamentals, using '%s' as fallback.",
                           available_col, self.config.date_column)
            available_col = self.config.date_column # Fallback a 'date' o 'as_of'

        if available_col not in df.columns:
             raise KeyError(f"Niether '{self.config.available_at_column}' nor '{self.config.date_column}' "
                            f"found in fundamentals file {path}")

        df[available_col] = pd.to_datetime(df[available_col], utc=True)
        df = df.sort_values(available_col).reset_index(drop=True)
        logger.debug("Loaded fundamentals for %s with %d rows", ticker, len(df))
        return df

    def _load_summary(self, ticker: str) -> Dict[str, Any]:
        path = self.data_root / self.config.summary_pattern.format(ticker=ticker)
        if not path.exists():
            logger.info("Summary file not found for %s: %s", ticker, path)
            return {}
        with open(path) as fp:
            data = json.load(fp)
        logger.debug("Loaded summary metadata for %s", ticker)
        return data

    def _flatten_json(self, y: Any, prefix: str = '') -> Dict[str, Any]:
        """Aplanar un JSON/dict anidado para columnas de DataFrame."""
        out = {}
        def flatten(x: Any, name: str = ''):
            if isinstance(x, dict):
                for a in x:
                    flatten(x[a], f"{name}{a}_")
            elif isinstance(x, list):
                i = 0
                for a in x:
                    flatten(a, f"{name}{i}_")
                    i += 1
            # Solo guardar tipos primitivos (str, int, float, bool)
            elif isinstance(x, (str, int, float, bool)):
                out[name[:-1]] = x
        flatten(y, prefix + "_")
        return out

    def _merge_price_and_fundamentals(
        self, price_df: pd.DataFrame, fundamental_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        date_col = self.config.date_column
        if fundamental_df.empty:
            return price_df.copy()

        available_col = (
            self.config.available_at_column
            if self.config.available_at_column in fundamental_df.columns
            else date_col
        )

        fundamentals = fundamental_df.copy()
        fundamentals = fundamentals.rename(columns={available_col: "available_at"})
        fundamentals = fundamentals.drop_duplicates("available_at")

        # Re-muestrear fundamentales a diario y 'forward fill'
        fundamentals = fundamentals.set_index("available_at").resample("D").ffill()
        fundamentals = fundamentals.reset_index()
        
        # Asegurarse de que ambas columnas de fecha estén ordenadas antes de merge_asof
        price_df = price_df.sort_values(date_col)
        fundamentals = fundamentals.sort_values("available_at")

        merged = pd.merge_asof(
            price_df,
            fundamentals,
            left_on=date_col,
            right_on="available_at",
            direction="backward", # Busca el último fundamental disponible
        )

        # Aplicar el lookback: eliminar fundamentales si son "demasiado viejos"
        lookback = self.config.fundamentals_lookback_days
        if lookback > 0:
            cutoff = merged[date_col] - pd.Timedelta(days=lookback)
            mask = merged["available_at"] >= cutoff
            
            # Obtener columnas de fundamentales (todas excepto las de precio)
            price_cols = set(price_df.columns)
            fund_cols = [col for col in merged.columns if col not in price_cols and col != "available_at"]
            
            # Poner NaN en las columnas fundamentales donde la máscara es Falsa
            merged.loc[~mask, fund_cols] = np.nan

        merged = merged.drop(columns=["available_at"], errors="ignore")
        return merged

    def _append_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.indicators
        result = df.copy()
        
        if "close" not in result.columns:
            raise KeyError("'close' column is required to append indicators")
            
        close = result["close"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).fillna(0)
        loss = -delta.where(delta < 0, 0.0).fillna(0)
        avg_gain = gain.rolling(window=cfg.rsi_window, min_periods=cfg.rsi_window).mean()
        avg_loss = loss.rolling(window=cfg.rsi_window, min_periods=cfg.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        result["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = macd - signal

        # Bollinger Bands
        rolling_mean = close.rolling(cfg.bollinger_window).mean()
        rolling_std = close.rolling(cfga.bollinger_window).std()
        result["bb_upper"] = rolling_mean + cfg.bollinger_std * rolling_std
        result["bb_lower"] = rolling_mean - cfg.bollinger_std * rolling_std
        result["bb_width"] = result["bb_upper"] - result["bb_lower"]

        return result