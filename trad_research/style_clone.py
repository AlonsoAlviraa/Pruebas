"""Style-clone strategies: same L0 portfolio shell, dumb L1 (no ML).

Used to falsify whether turbo/ML edge is residual alpha or universe+trend style.
See docs/design/2026-07-23_structural_redesign_alpha.md (P1, STR-04).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.backtest import BacktestConfig
from trad_research.strategies import Strategy


def _minalloc_portfolio_overrides(
    *,
    min_alloc_pct: float = 0.015,
    max_atr_pct: float = 0.22,
    vol_target: float = 0.04,
    max_position_pct: float = 0.22,
    max_positions: int = 16,
    max_horizon: int = 38,
    k_atr: float = 3.5,
    hard_stop_pct: float = 0.11,
    risk_off_scale: float = 0.90,
) -> Dict[str, Any]:
    """Portfolio / risk shell aligned with highvol minalloc research path."""
    return {
        "min_confidence": 0.0,
        "require_trend": False,
        "require_momentum": False,
        "momentum_min": 0.0,
        "min_dist_sma200": -1.0,
        "max_atr_pct": max_atr_pct,
        "k_atr": k_atr,
        "max_horizon": max_horizon,
        "volatility_target_pct": vol_target,
        "max_position_pct": max_position_pct,
        "max_positions": max_positions,
        "risk_off_scale": risk_off_scale,
        "require_regime": True,
        "max_entries_per_day": 10,
        "meta_threshold": 0.0,
        "hard_stop_pct": hard_stop_pct,
        "min_alloc_pct": min_alloc_pct,
        "soft_hard_regime": False,
    }


@dataclass
class StyleCloneBase(Strategy):
    """Shared L0 settings for high-vol style clones (not ML)."""

    needs_training: bool = False
    universe_source_file: str = "universe_highvol80.txt"
    universe_n: int = 80
    universe_scan_limit: int = 500
    regime_filter: str = "strict_dual_golden"
    min_alloc_pct: float = 0.015
    min_atr_norm: float = 0.02
    # Prefer juicier names like highvol score boost (style, not ML)
    boost_score_by_atr: bool = True

    def backtest_overrides(self) -> Dict[str, Any]:
        return _minalloc_portfolio_overrides(min_alloc_pct=float(self.min_alloc_pct))

    def _post_score(
        self, df: pd.DataFrame, sig: np.ndarray, score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        sig = np.asarray(sig, dtype=bool)
        score = np.asarray(score, dtype=float)
        if "atr_norm" in df.columns:
            atr = df["atr_norm"].to_numpy(dtype=float)
            sig = sig & np.isfinite(atr) & (atr >= self.min_atr_norm)
            if self.boost_score_by_atr:
                score = score * (1.0 + np.clip(np.nan_to_num(atr, nan=0.0), 0, 0.15) / 0.05)
        if self.boost_score_by_atr and "volatility_20" in df.columns:
            v = df["volatility_20"].to_numpy(dtype=float)
            score = score * (1.0 + np.clip(np.nan_to_num(v, nan=0.0), 0, 1.0))
        score = np.where(sig, score, 0.0)
        return sig, score


@dataclass
class StyleEWClone(StyleCloneBase):
    """Always-in when liquid enough: equal-weight-ish long within L0 + portfolio caps."""

    name: str = "style_ew_hv"
    description: str = "Style clone: EW long highvol shell (no trend filter)"

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        n = len(df)
        sig = np.ones(n, dtype=bool)
        # Mild liquidity / activity score
        score = np.ones(n, dtype=float)
        if "volume_ratio" in df.columns:
            score = np.nan_to_num(df["volume_ratio"].to_numpy(dtype=float), nan=1.0)
        sig, score = self._post_score(df, sig, score)
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


@dataclass
class StyleTrendSMA50Clone(StyleCloneBase):
    """Long when close > SMA50; score = dist_sma_50."""

    name: str = "style_trend_sma50_hv"
    description: str = "Style clone: SMA50 trend on highvol shell"

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        n = len(df)
        if "sma_50" not in df.columns or "close" not in df.columns:
            z = np.zeros(n, dtype=bool)
            return pd.Series(z, index=df.index), pd.Series(0.0, index=df.index)
        close = df["close"].to_numpy(dtype=float)
        sma = df["sma_50"].to_numpy(dtype=float)
        sig = np.isfinite(sma) & (close > sma)
        if "dist_sma_50" in df.columns:
            score = np.nan_to_num(df["dist_sma_50"].to_numpy(dtype=float), nan=0.0)
        else:
            score = (close / np.clip(sma, 1e-9, None)) - 1.0
        score = np.clip(score, -1.0, 5.0) + 0.05  # keep positive when in trend
        sig, score = self._post_score(df, sig, score)
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


@dataclass
class StyleMomClone(StyleCloneBase):
    """Long when ret_1m > 0; score = ret_1m (12-1 proxy via 21d)."""

    name: str = "style_mom_1m_hv"
    description: str = "Style clone: 1m momentum on highvol shell"

    momentum_min: float = 0.0

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        n = len(df)
        if "ret_1m" not in df.columns:
            z = np.zeros(n, dtype=bool)
            return pd.Series(z, index=df.index), pd.Series(0.0, index=df.index)
        ret = df["ret_1m"].to_numpy(dtype=float)
        sig = np.isfinite(ret) & (ret >= float(self.momentum_min))
        score = np.nan_to_num(ret, nan=0.0)
        # Optional trend confirm without ML
        if "sma_50" in df.columns and "close" in df.columns:
            sig = sig & (df["close"].to_numpy(dtype=float) > df["sma_50"].to_numpy(dtype=float))
        sig, score = self._post_score(df, sig, score)
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


@dataclass
class StyleTrendMomClone(StyleCloneBase):
    """SMA50 + positive 1m mom — strongest style control for P1."""

    name: str = "style_trend_mom_hv"
    description: str = "Style clone: SMA50 + ret_1m>0 on highvol shell"

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        n = len(df)
        need = ("close", "sma_50", "ret_1m")
        if any(c not in df.columns for c in need):
            z = np.zeros(n, dtype=bool)
            return pd.Series(z, index=df.index), pd.Series(0.0, index=df.index)
        close = df["close"].to_numpy(dtype=float)
        sma = df["sma_50"].to_numpy(dtype=float)
        ret = df["ret_1m"].to_numpy(dtype=float)
        sig = np.isfinite(sma) & np.isfinite(ret) & (close > sma) & (ret > 0.0)
        dist = (
            df["dist_sma_50"].to_numpy(dtype=float)
            if "dist_sma_50" in df.columns
            else (close / np.clip(sma, 1e-9, None) - 1.0)
        )
        score = np.nan_to_num(dist, nan=0.0) + np.nan_to_num(ret, nan=0.0)
        sig, score = self._post_score(df, sig, score)
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


STYLE_CLONE_NAMES: Tuple[str, ...] = (
    "style_ew_hv",
    "style_trend_sma50_hv",
    "style_mom_1m_hv",
    "style_trend_mom_hv",
)


def all_style_clones() -> List[Strategy]:
    return [
        StyleEWClone(),
        StyleTrendSMA50Clone(),
        StyleMomClone(),
        StyleTrendMomClone(),
    ]


def get_style_clone(name: str) -> Strategy:
    key = (name or "").strip()
    for s in all_style_clones():
        if s.name == key:
            return s
    raise KeyError(f"Unknown style clone: {name}. Available: {list(STYLE_CLONE_NAMES)}")


def style_clone_registry() -> Dict[str, Strategy]:
    return {s.name: s for s in all_style_clones()}
