#!/usr/bin/env python3
"""Data pipeline responsible for loading prices, engineering features and labelling."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration controlling how the pipeline behaves."""

    data_root: Path = Path("data")
    indicator_windows: Sequence[int] = (5, 10, 20)
    rsi_windows: Sequence[int] = (7, 14, 21)
    sma_windows: Sequence[int] = (50, 200)
    volatility_window: int = 20
    atr_window: int = 14
    volume_window: int = 20
    min_history: int = 250


class DataPipeline:
    """Utility class to load raw history and generate ML-ready datasets."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Loading and feature engineering
    # ------------------------------------------------------------------
    def load_feature_view(
        self,
        ticker: str,
        indicators: bool = True,
        include_summary: bool = False,
    ) -> pd.DataFrame:
        """Load a ticker history and optionally attach engineered indicators."""

        path = self.config.data_root / f"{ticker}_history.csv"
        if not path.exists():
            raise FileNotFoundError(f"History file not found for {ticker}: {path}")

        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            raise KeyError(f"Ticker {ticker} is missing the 'date' column")

        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.sort_values("date").reset_index(drop=True)
        frame["ticker"] = ticker

        numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in frame.columns]
        for column in numeric_cols:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if indicators:
            frame = self._engineer_indicators(frame)

        if include_summary and "summary" in frame.columns:
            # Some datasets include a JSON fundamentals summary. Keep as-is.
            pass

        frame = frame.dropna(subset=["close"])
        return frame

    def _engineer_indicators(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        cfg = self.config

        if len(result) < cfg.min_history:
            logger.debug("Ticker %s has only %d rows; indicators may be unstable", result["ticker"].iloc[0], len(result))

        close = result["close"].astype(float)
        high = result.get("high", close)
        low = result.get("low", close)
        volume = result.get("volume", pd.Series(index=result.index, dtype=float))

        # ATR and normalized ATR (volatility proxy)
        atr = ta.atr(high=high, low=low, close=close, length=cfg.atr_window)
        result["atr"] = atr
        # Feature normalization: ATR relative to price
        result["atr_norm"] = atr / close.clip(lower=1e-6)

        # RSI with multiple windows
        for window in cfg.rsi_windows:
            result[f"rsi_{window}"] = ta.rsi(close, length=window)

        # Moving averages and distance to them
        for window in cfg.sma_windows:
            sma = close.rolling(window=window, min_periods=window // 2).mean()
            result[f"sma_{window}"] = sma
            result[f"dist_sma_{window}"] = (close - sma) / sma.replace(0, np.nan)

        # Historical volatility (annualised std of returns)
        returns = close.pct_change()
        vol = returns.rolling(cfg.volatility_window, min_periods=cfg.volatility_window // 2).std()
        result[f"volatility_{cfg.volatility_window}"] = vol * np.sqrt(252)

        # Volume features (Z-Score adds regime detection capability)
        volume_sma = volume.rolling(cfg.volume_window, min_periods=max(2, cfg.volume_window // 2)).mean()
        volume_std = volume.rolling(cfg.volume_window, min_periods=max(2, cfg.volume_window // 2)).std()
        result["volume_sma"] = volume_sma
        result["volume_ratio"] = volume / volume_sma.replace(0, np.nan)
        result["volume_zscore"] = (volume - volume_sma) / volume_std.replace(0, np.nan)

        return result

    # ------------------------------------------------------------------
    # Labelling
    # ------------------------------------------------------------------
    def create_triple_barrier_labels(
        self,
        frame: pd.DataFrame,
        horizon: int = 5,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        atr_multiplier_tp: float = 2.0,
        atr_multiplier_sl: float = 2.0,
    ) -> pd.DataFrame:
        """Generate triple-barrier labels with volatility-adjusted targets."""

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        df = frame.copy().reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        close = df["close"].astype(float).values
        high = df.get("high", df["close"]).astype(float).values
        low = df.get("low", df["close"]).astype(float).values

        # --- CORRECCIÓN DE WARNINGS AQUÍ ---
        if "atr" in df.columns and df["atr"].notna().any():
            atr = df["atr"].astype(float).ffill().bfill().values
        else:
            cfg = self.config
            atr_series = ta.atr(high=df.get("high", df["close"]), low=df.get("low", df["close"]), close=df["close"], length=cfg.atr_window)
            atr = atr_series.ffill().bfill().fillna(0.0).values
            df["atr"] = atr_series
        # -----------------------------------

        atr_norm = atr / np.clip(close, 1e-6, None)
        dynamic_tp = atr_multiplier_tp * atr_norm
        dynamic_sl = atr_multiplier_sl * atr_norm

        if take_profit is not None:
            dynamic_tp = np.full_like(dynamic_tp, take_profit)
        if stop_loss is not None:
            dynamic_sl = np.full_like(dynamic_sl, abs(stop_loss))

        tp_pct = dynamic_tp
        sl_pct = -dynamic_sl

        labels = np.full(len(df), np.nan)
        time_exit_return = np.full(len(df), np.nan)

        for idx in range(len(df) - horizon):
            entry = close[idx]
            if not np.isfinite(entry) or entry <= 0:
                continue

            upper = entry * (1 + tp_pct[idx])
            lower = entry * (1 + sl_pct[idx])

            window_slice = slice(idx + 1, min(len(df), idx + 1 + horizon))
            window_high = high[window_slice]
            window_low = low[window_slice]
            window_close = close[window_slice]

            hit_upper = np.where(window_high >= upper)[0]
            hit_lower = np.where(window_low <= lower)[0]

            horizon_exit_price = window_close[-1]
            time_exit_return[idx] = (horizon_exit_price - entry) / entry

            first_upper = hit_upper[0] if len(hit_upper) > 0 else horizon + 1
            first_lower = hit_lower[0] if len(hit_lower) > 0 else horizon + 1

            if first_upper < first_lower:
                labels[idx] = 1
            elif first_lower < first_upper:
                labels[idx] = -1
            elif first_lower == first_upper == horizon + 1:
                labels[idx] = 0
            else:
                labels[idx] = -1

        if horizon > 0:
            labels[-horizon:] = np.nan
            tp_pct[-horizon:] = np.nan
            sl_pct[-horizon:] = np.nan
            time_exit_return[-horizon:] = np.nan

        out = df.copy()
        out["label"] = labels
        out["tp_pct"] = tp_pct
        out["sl_pct"] = sl_pct
        out["time_exit_return"] = time_exit_return

        return out

__all__ = ["PipelineConfig", "DataPipeline"]