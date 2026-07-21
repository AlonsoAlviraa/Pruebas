"""Causal OHLCV features aligned with Lean M2 (17 features).

SSOT for research feature engineering (FEA-01). Lean runtime still computes
indicators via QC; names and order must match lean_strategy.modules.config.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# Ground truth from lean_strategy.modules.config.StrategyConfig
M2_FEATURE_NAMES: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "atr",
    "atr_norm",
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "sma_50",
    "dist_sma_50",
    "sma_200",
    "dist_sma_200",
    "volatility_20",
    "volume_sma",
    "volume_ratio",
    "volume_zscore",
)

M1_FEATURE_NAMES: tuple[str, ...] = (
    "close",
    "atr",
    "ma_50",
    "ma_200",
    "rsi_14",
    "volume_sma",
)

# Scale-free features for cross-sectional ML (preferred for walk-forward research)
M2_REL_FEATURE_NAMES: tuple[str, ...] = (
    "atr_norm",
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "dist_sma_50",
    "dist_sma_200",
    "volatility_20",
    "volume_ratio",
    "volume_zscore",
    "ret_1m",
)


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def engineer_m2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add M2 features to an OHLCV frame. Expects columns: open,high,low,close,volume,date.
    All rolling stats are causal (use only past + current bar).
    """
    out = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            raise KeyError(f"Missing column {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"].fillna(0.0)

    atr = _atr(high, low, close, 14)
    out["atr"] = atr
    out["atr_norm"] = atr / close.clip(lower=1e-6)

    out["rsi_7"] = _wilder_rsi(close, 7)
    out["rsi_14"] = _wilder_rsi(close, 14)
    out["rsi_21"] = _wilder_rsi(close, 21)

    out["sma_50"] = close.rolling(50, min_periods=25).mean()
    out["sma_200"] = close.rolling(200, min_periods=100).mean()
    out["dist_sma_50"] = (close - out["sma_50"]) / out["sma_50"].replace(0, np.nan)
    out["dist_sma_200"] = (close - out["sma_200"]) / out["sma_200"].replace(0, np.nan)

    rets = close.pct_change()
    out["volatility_20"] = rets.rolling(20, min_periods=10).std() * np.sqrt(252)

    vol_sma = volume.rolling(20, min_periods=10).mean()
    vol_std = volume.rolling(20, min_periods=10).std().replace(0, np.nan)
    out["volume_sma"] = vol_sma
    out["volume_ratio"] = volume / vol_sma.replace(0, np.nan)
    out["volume_zscore"] = (volume - vol_sma) / vol_std

    # Aux for filters / M1 naming
    out["ma_50"] = out["sma_50"]
    out["ma_200"] = out["sma_200"]
    out["ret_1m"] = close.pct_change(21)

    return out


def load_history(ticker: str, data_root: Path) -> pd.DataFrame:
    path = data_root / f"{ticker}_history.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.columns = [c.lower().strip() for c in frame.columns]
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    frame["ticker"] = ticker
    return frame


def load_featured(
    ticker: str,
    data_root: Path,
    min_history: int = 220,
) -> pd.DataFrame:
    raw = load_history(ticker, data_root)
    if raw.empty or len(raw) < min_history:
        return pd.DataFrame()
    feat = engineer_m2_features(raw)
    feat = feat.dropna(subset=list(M2_FEATURE_NAMES) + ["ret_1m"]).reset_index(drop=True)
    return feat


def feature_matrix(df: pd.DataFrame, names: Sequence[str] = M2_FEATURE_NAMES) -> pd.DataFrame:
    X = df.reindex(columns=list(names)).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def list_tickers(ticker_file: Path, data_root: Path, limit: Optional[int] = None) -> List[str]:
    if ticker_file.is_file():
        tickers = [
            ln.strip().upper()
            for ln in ticker_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        tickers = sorted({p.name.replace("_history.csv", "") for p in data_root.glob("*_history.csv")})
    # Prefer names with history present
    out: List[str] = []
    for t in tickers:
        if (data_root / f"{t}_history.csv").exists():
            out.append(t)
        if limit is not None and len(out) >= limit:
            break
    return out
