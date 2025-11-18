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
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0


@dataclass
class PipelineConfig:
    """Configuración del pipeline de datos."""
    data_root: Path = Path("data")
    date_column: str = "date"
    summary_filename: str = "_summary.json"
    history_filename: str = "_history.csv"
    fundamentals_filename: Optional[str] = None
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

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read CSV %s: %s", file_path, e)
            return pd.DataFrame()

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

    def _load_fundamentals(self, ticker: str, price_df: pd.DataFrame) -> pd.DataFrame:
        """Carga y alinea los datos fundamentales con el historial de precios."""
        file_path = self.config.data_root / f"{ticker}{self.config.fundamentals_filename}"
        date_col = self.config.date_column
        if not file_path.exists():
            return price_df

        try:
            fund_df = pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read fundamentals CSV %s: %s", file_path, e)
            return price_df

        if fund_df.empty or "reportedDate" not in fund_df.columns:
            return price_df

        # Preparar datos fundamentales
        fund_df = fund_df.rename(columns={"reportedDate": "available_at"})
        fund_df["available_at"] = pd.to_datetime(fund_df["available_at"])
        fund_df = fund_df.drop_duplicates(subset=["available_at"]).set_index("available_at").sort_index()

        # Alinear con el índice de precios
        price_idx = price_df.index
        merged = pd.merge_asof(
            price_df,
            fund_df,
            left_index=True,
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(days=self.config.fundamentals_lookback_days + 5),
        )
        merged = merged.set_index(price_idx)

        # Aplicar el lookback: eliminar fundamentales si son "demasiado viejos"
        lookback = self.config.fundamentals_lookback_days
        if lookback > 0 and "available_at" in merged.columns:
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
        cfg = self.config.indicators  # <-- Esta línea era la que fallaba si no había default_factory
        base = df.copy()

        if "close" not in base.columns:
            raise KeyError("'close' column is required to append indicators")

        close = base["close"]

        # RSI (versión sin periodo de "calentamiento" artificial). Utilizamos
        # el suavizado exponencial clásico de Welles Wilder para que exista un
        # valor desde el primer día sin necesidad de recortar filas.
        delta = close.diff().fillna(0.0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        alpha = 1 / cfg.rsi_window
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)

        # MACD
        ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
        macd_hist = macd - signal

        # Bollinger Bands
        rolling_mean = close.rolling(cfg.bollinger_window, min_periods=1).mean()
        rolling_std = close.rolling(cfg.bollinger_window, min_periods=1).std(ddof=0)
        rolling_std = rolling_std.fillna(0.0)
        bb_upper = rolling_mean + cfg.bollinger_std * rolling_std
        bb_lower = rolling_mean - cfg.bollinger_std * rolling_std
        bb_width = bb_upper - bb_lower

        # >>> Concatenamos las columnas de indicadores en bloque para evitar fragmentación
        indicators_df = pd.DataFrame(
            {
                "rsi": rsi,
                "macd": macd,
                "macd_signal": signal,
                "macd_hist": macd_hist,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_width": bb_width,
            },
            index=base.index,
        )
        result = pd.concat([base, indicators_df], axis=1)
        return result.copy()

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
            if merged.empty:
                logger.warning("No price data found for %s", ticker)
                return pd.DataFrame()
            merged = merged.reset_index(drop=False)
        except FileNotFoundError as e:
            logger.error("Data loading failed for %s: %s", ticker, e)
            raise

        # 2. (Opcional) Cargar y alinear fundamentales
        if self.config.fundamentals_filename:
            merged = self._load_fundamentals(ticker, merged)

        # 3. (Opcional) Añadir indicadores técnicos
        if indicators:
            merged = self._append_indicators(merged)

        # 4. (Opcional) Cargar y aplanar datos fundamentales/resumen
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
        # Solo eliminamos filas con precios nulos. Los indicadores se rellenan
        # explícitamente para evitar perder histórico en los primeros días del
        # ticker (que es precisamente lo que originaba la reducción a ~97
        # filas).
        dropna_subset = price_cols
        if dropna_subset:
            merged = merged.dropna(subset=dropna_subset)

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
                )
                if col in merged.columns
            ]
            if indicator_cols:
                merged.loc[:, indicator_cols] = merged[indicator_cols].ffill().bfill()
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
    def create_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        take_profit: float = 0.05,
        stop_loss: float = -0.03,
    ) -> pd.DataFrame:
        """
        Genera etiquetas usando el Método de Triple Barrera.
        
        Etiquetas:
        1 (BUY): El precio toca la barrera superior (Take Profit) antes que la inferior o el límite de tiempo.
        -1 (SELL/STOP): El precio toca la barrera inferior (Stop Loss) antes que la superior.
        0 (HOLD): El precio no toca ninguna barrera antes del límite de tiempo (horizon).
        
        Args:
            df: DataFrame con datos de precios (debe tener columna 'close').
            horizon: Número máximo de días para mantener la posición.
            take_profit: Retorno objetivo positivo (ej. 0.05 para 5%).
            stop_loss: Límite de pérdida negativo (ej. -0.03 para -3%).
            
        Returns:
            DataFrame con las mismas filas que df, añadiendo la columna 'label'.
            Las filas donde no se puede calcular el futuro (últimos 'horizon' días) tendrán NaN.
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        close = df["close"]
        labels = pd.Series(0, index=df.index, name="label")
        
        # Iteramos (vectorizado sería ideal pero complejo para barreras de ruta dependiente)
        # Para eficiencia en Python puro, usamos numpy arrays
        prices = close.values
        n = len(prices)
        
        # Arrays para resultados
        out_labels = np.zeros(n, dtype=int)
        
        # Pre-calcular barreras relativas
        # upper_barrier = prices * (1 + take_profit)
        # lower_barrier = prices * (1 + stop_loss) # stop_loss es negativo, ej -0.03 -> * 0.97
        
        # Bucle optimizado con numba sería mejor, pero usamos bucle simple por ahora
        # Ojo: stop_loss debe ser negativo en el input, ej -0.03
        
        for t in range(n - horizon):
            price_t = prices[t]
            upper = price_t * (1 + take_profit)
            lower = price_t * (1 + stop_loss)
            
            # Ventana futura
            window = prices[t+1 : t+1+horizon]
            
            # Buscar cruces
            # np.argmax devuelve el primer índice del máximo (True), que es lo que queremos
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