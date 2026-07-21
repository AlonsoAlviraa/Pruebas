"""Pluggable entry strategies for research backtests (BKT / strategy bake-off)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.backtest import BacktestConfig
from trad_research.features import M2_FEATURE_NAMES, M2_REL_FEATURE_NAMES, feature_matrix
from trad_research.walk_forward import train_meta_model, train_side_model


class Strategy(ABC):
    """Strategy that can optionally train yearly and always emit entry signals."""

    name: str = "base"
    description: str = ""
    needs_training: bool = False
    # Key into trad_research.regime.build_all_regime_maps (industry index filters)
    regime_filter: str = "legacy_sma50"
    # FEA-03: feature columns used by ML train/signal (default Lean-aligned M2 17)
    feature_names: Sequence[str] = M2_FEATURE_NAMES
    # When True, panels load point-in-time fund_rev_yoy / fund_eps_yoy / fund_quality
    needs_fundamentals: bool = False
    # Dynamic high-vol universe re-ranked each OOS year (as_of = train_end - 1d)
    dynamic_highvol: bool = False
    universe_source_file: str = "good_tickers_filtrados.txt"
    # Sector ETF gate + rotation (wired by strategy_runner when True)
    require_sector_trend: bool = False
    enable_rotation: bool = False
    rotation_min_score_edge: float = 0.05
    rotation_min_bars: int = 3
    rotation_max_per_day: int = 2
    sector_map_path: str = "data/ticker_sector_map.csv"
    sector_ma: int = 50
    sector_require_sma200: bool = False
    sector_allow_unmapped: bool = True
    universe_n: int = 80
    universe_scan_limit: int = 500

    def train(self, train_df: pd.DataFrame, year: int) -> None:
        """Optional yearly fit. Default no-op."""
        return None

    def backtest_overrides(self) -> Dict[str, Any]:
        """Optional BacktestConfig field overrides."""
        return {}

    @abstractmethod
    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        """Return (boolean_mask, score) aligned to df index. Higher score = preferred."""


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)


@dataclass
class ChampionMLStrategy(Strategy):
    """Binary XGB + optional meta (champion path)."""

    name: str = "champion_ml"
    description: str = "WF XGB buy model + meta-label (CHAMPION_EODHD_v2 style)"
    needs_training: bool = True
    use_meta: bool = True
    min_confidence: float = 0.38
    feature_names: Sequence[str] = M2_FEATURE_NAMES
    _model: Any = None
    _meta: Any = None

    def backtest_overrides(self) -> Dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "require_trend": True,
            "require_momentum": True,
            "momentum_min": 0.015,
            "min_dist_sma200": -0.05,
            "max_atr_pct": 0.09,
            "k_atr": 3.0,
            "max_horizon": 20,
            "volatility_target_pct": 0.018,
            "max_position_pct": 0.15,
            "max_positions": 12,
            "risk_off_scale": 0.55,
            "meta_threshold": 0.48,
        }

    def train(self, train_df: pd.DataFrame, year: int) -> None:
        names = list(self.feature_names)
        self._model = train_side_model(
            train_df, feature_names=names, random_state=year, binary_buy=True
        )
        self._meta = None
        if self.use_meta:
            self._meta = train_meta_model(
                train_df,
                self._model,
                names,
                primary_threshold=self.min_confidence,
                random_state=year,
            )

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        if self._model is None:
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        X = feature_matrix(df, self.feature_names)
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        buy_i = classes.index(1) if 1 in classes else 0
        p_buy = proba[:, buy_i]
        sig = p_buy >= cfg.min_confidence
        if cfg.require_trend and "sma_50" in df.columns:
            sig = sig & (df["close"].to_numpy() > df["sma_50"].to_numpy())
        if cfg.require_momentum and "ret_1m" in df.columns:
            sig = sig & (df["ret_1m"].to_numpy() >= cfg.momentum_min)
        if "atr_norm" in df.columns and cfg.max_atr_pct is not None:
            sig = sig & (df["atr_norm"].to_numpy() <= cfg.max_atr_pct)
        if "dist_sma_200" in df.columns and cfg.min_dist_sma200 is not None:
            sig = sig & (df["dist_sma_200"].to_numpy() >= cfg.min_dist_sma200)
        score = p_buy.copy()
        if self._meta is not None and hasattr(self._meta, "predict_proba"):
            p_meta = self._meta.predict_proba(X)[:, 1]
            sig = sig & (p_meta >= cfg.meta_threshold)
            score = p_buy * p_meta
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


@dataclass
class AggressiveMLStrategy(Strategy):
    """
    High-octane ML: more trades, bigger size, looser filters.
    Prioritizes CAGR over smooth equity; accepts ugly years (e.g. 2022).
    """

    name: str = "aggressive_ml"
    description: str = "Aggressive XGB: low conf, no meta, high size, long holds"
    needs_training: bool = True
    use_meta: bool = False
    min_confidence: float = 0.30
    momentum_min: float = 0.0
    require_trend: bool = True
    require_momentum: bool = False
    min_dist_sma200: float = -0.12
    max_atr_pct: float = 0.14
    vol_target: float = 0.032
    max_position_pct: float = 0.25
    max_positions: int = 16
    k_atr: float = 3.5
    max_horizon: int = 35
    risk_off_scale: float = 0.90
    require_regime: bool = True
    max_entries_per_day: int = 10
    feature_names: Sequence[str] = M2_FEATURE_NAMES
    _model: Any = None
    _meta: Any = None

    def backtest_overrides(self) -> Dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "require_trend": self.require_trend,
            "require_momentum": self.require_momentum,
            "momentum_min": self.momentum_min,
            "min_dist_sma200": self.min_dist_sma200,
            "max_atr_pct": self.max_atr_pct,
            "k_atr": self.k_atr,
            "max_horizon": self.max_horizon,
            "volatility_target_pct": self.vol_target,
            "max_position_pct": self.max_position_pct,
            "max_positions": self.max_positions,
            "risk_off_scale": self.risk_off_scale,
            "require_regime": self.require_regime,
            "max_entries_per_day": self.max_entries_per_day,
            "meta_threshold": 0.35,
            "hard_stop_pct": 0.10,  # wider hard stop — less shaken out
        }

    def train(self, train_df: pd.DataFrame, year: int) -> None:
        names = list(self.feature_names)
        self._model = train_side_model(
            train_df, feature_names=names, random_state=year, binary_buy=True
        )
        self._meta = None
        if self.use_meta:
            self._meta = train_meta_model(
                train_df,
                self._model,
                names,
                primary_threshold=self.min_confidence,
                random_state=year,
            )

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        # Reuse champion path logic with this instance's model
        if self._model is None:
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        X = feature_matrix(df, self.feature_names)
        proba = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        buy_i = classes.index(1) if 1 in classes else 0
        p_buy = proba[:, buy_i]
        sig = p_buy >= cfg.min_confidence
        if cfg.require_trend and "sma_50" in df.columns:
            sig = sig & (df["close"].to_numpy() > df["sma_50"].to_numpy())
        if cfg.require_momentum and "ret_1m" in df.columns:
            sig = sig & (df["ret_1m"].to_numpy() >= cfg.momentum_min)
        if "atr_norm" in df.columns and cfg.max_atr_pct is not None:
            sig = sig & (df["atr_norm"].to_numpy() <= cfg.max_atr_pct)
        if "dist_sma_200" in df.columns and cfg.min_dist_sma200 is not None:
            sig = sig & (df["dist_sma_200"].to_numpy() >= cfg.min_dist_sma200)
        score = p_buy.copy()
        if self._meta is not None and hasattr(self._meta, "predict_proba"):
            p_meta = self._meta.predict_proba(X)[:, 1]
            sig = sig & (p_meta >= cfg.meta_threshold)
            score = p_buy * p_meta
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


@dataclass
class AggressiveMaxStrategy(AggressiveMLStrategy):
    """Maximum aggression: no soft de-risk, no meta, highest size."""

    name: str = "aggressive_max"
    description: str = "Max size ML: conf 0.25, vol_target 4%, no meta, soft size 1.0"
    min_confidence: float = 0.25
    vol_target: float = 0.040
    max_position_pct: float = 0.30
    max_positions: int = 18
    risk_off_scale: float = 1.0
    max_atr_pct: float = 0.18
    min_dist_sma200: float = -0.20
    max_horizon: int = 40
    k_atr: float = 4.0
    hard_stop: float = 0.12

    def backtest_overrides(self) -> Dict[str, Any]:
        o = super().backtest_overrides()
        o["hard_stop_pct"] = 0.12
        o["max_entries_per_day"] = 12
        return o


@dataclass
class AggressiveLetRunStrategy(AggressiveMLStrategy):
    """Aggressive entries + very wide trail to ride bull legs hard."""

    name: str = "aggressive_let_run"
    description: str = "Aggressive entry + wide trail (k=4.5) horizon 45"
    min_confidence: float = 0.32
    use_meta: bool = False
    vol_target: float = 0.028
    max_position_pct: float = 0.22
    k_atr: float = 4.5
    max_horizon: int = 45
    risk_off_scale: float = 0.85
    require_momentum: bool = True
    momentum_min: float = 0.01


@dataclass
class AggressiveTurboStrategy(AggressiveMLStrategy):
    """
    User-preferred balance: beat champion CAGR, allow ~40% DD, no meta.
    """

    name: str = "aggressive_turbo"
    description: str = "Turbo ML: conf 0.28, vol 3.5%, pos 28%, hold 38d — max return bias"
    min_confidence: float = 0.28
    use_meta: bool = False
    vol_target: float = 0.035
    max_position_pct: float = 0.28
    max_positions: int = 16
    k_atr: float = 3.8
    max_horizon: int = 38
    risk_off_scale: float = 0.95
    max_atr_pct: float = 0.16
    min_dist_sma200: float = -0.15
    require_momentum: bool = False
    require_trend: bool = True
    regime_filter: str = "legacy_sma50"

    def backtest_overrides(self) -> Dict[str, Any]:
        o = super().backtest_overrides()
        o["hard_stop_pct"] = 0.11
        o["max_entries_per_day"] = 10
        return o


def turbo_with_regime(regime_name: str, short_label: str) -> AggressiveTurboStrategy:
    """Factory: aggressive_turbo + industry index gate."""
    return AggressiveTurboStrategy(
        name=f"turbo_{short_label}",
        description=f"aggressive_turbo + index regime [{regime_name}]",
        regime_filter=regime_name,
    )


def turbo_rel_with_regime(regime_name: str, short_label: str) -> AggressiveTurboStrategy:
    """FEA-03: aggressive_turbo with scale-free M2_REL features (US train still)."""
    return AggressiveTurboStrategy(
        name=f"turbo_rel_{short_label}",
        description=f"turbo M2_REL + regime [{regime_name}] (fea-03.v1)",
        regime_filter=regime_name,
        feature_names=M2_REL_FEATURE_NAMES,
    )


@dataclass
class HighVolTurboStrategy(AggressiveTurboStrategy):
    """Turbo tuned for high-volatility names (wider ATR allowance, score × vol)."""

    name: str = "turbo_highvol"
    description: str = "Turbo for high-vol names: higher ATR cap, score×atr_norm, strict regime"
    max_atr_pct: float = 0.22
    vol_target: float = 0.04
    max_position_pct: float = 0.22
    min_confidence: float = 0.30
    regime_filter: str = "strict_dual_golden"
    min_atr_norm: float = 0.02  # skip dead quiet names
    # Bottleneck-fix knobs (defaults = legacy champion behaviour)
    boost_score_by_atr: bool = True
    soft_trend: bool = False  # True: trend via SMA50 OR causal SMA20 (not hard SMA50 only)
    min_alloc_pct: float = 0.0
    soft_hard_regime: bool = False
    regime_hard_size_scale: Optional[float] = None
    hard_stop_pct: float = 0.11
    hard_stop_atr_mult: Optional[float] = None

    def backtest_overrides(self) -> Dict[str, Any]:
        o = super().backtest_overrides()
        o["hard_stop_pct"] = self.hard_stop_pct
        o["max_entries_per_day"] = 10
        o["min_alloc_pct"] = float(self.min_alloc_pct)
        o["soft_hard_regime"] = bool(self.soft_hard_regime)
        if self.regime_hard_size_scale is not None:
            o["regime_hard_size_scale"] = float(self.regime_hard_size_scale)
        if self.hard_stop_atr_mult is not None:
            o["hard_stop_atr_mult"] = float(self.hard_stop_atr_mult)
        return o

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        # Soft trend: disable hard SMA50 gate, re-apply SMA50 | SMA20 after base ML filters
        if self.soft_trend:
            cfg_use = replace(cfg, require_trend=False)
            sig, score = AggressiveMLStrategy.generate_signals(self, df, cfg_use)
            close = df["close"].to_numpy(dtype=float)
            trend_ok = np.zeros(len(df), dtype=bool)
            has_ma = False
            if "sma_50" in df.columns:
                trend_ok = trend_ok | (close > df["sma_50"].to_numpy(dtype=float))
                has_ma = True
            # Ad-hoc causal SMA20 (not in M2 feature_names / Lean config).
            # Research-only softtrend filter: rolling(20, min_periods=10) on close.
            # NaN early bars fail the close>sma20 leg (safe). If softtrend is promoted
            # to production, add sma_20 to the feature pipeline for train/serve parity.
            sma20 = df["close"].rolling(20, min_periods=10).mean().to_numpy(dtype=float)
            trend_ok = trend_ok | (close > sma20)
            has_ma = True
            if has_ma:
                sig = sig & trend_ok
        else:
            sig, score = AggressiveMLStrategy.generate_signals(self, df, cfg)

        score_arr = np.asarray(score, dtype=float)
        sig_arr = np.asarray(sig, dtype=bool)
        if "atr_norm" in df.columns:
            atr = df["atr_norm"].to_numpy(dtype=float)
            sig_arr = sig_arr & (atr >= self.min_atr_norm)
            if self.boost_score_by_atr:
                # Prefer juicier vol within allowed band (legacy highvol ranking)
                score_arr = score_arr * (1.0 + np.clip(atr, 0, 0.15) / 0.05)
        if self.boost_score_by_atr and "volatility_20" in df.columns:
            v = df["volatility_20"].to_numpy(dtype=float)
            score_arr = score_arr * (1.0 + np.clip(v, 0, 1.0))
        return pd.Series(sig_arr, index=df.index), pd.Series(score_arr, index=df.index)


# --- Bottleneck-fix ablations (champion turbo_highvol base; additive only) ---


@dataclass
class HighVolMinAllocStrategy(HighVolTurboStrategy):
    """EXP1: skip micro allocations that clog max_positions.

    **US research baseline (2026-07-18 close-out):** best single-knob return pick
    from bottleneck bake-off (see reports/RESEARCH_BASELINE.md).
    """

    name: str = "turbo_highvol_minalloc"
    description: str = "US research baseline: highvol + min_alloc_pct=1.5% (skip micro slots)"
    min_alloc_pct: float = 0.015


@dataclass
class HighVolMinAllocSoftRegStrategy(HighVolTurboStrategy):
    """EXP1+EXP3 only: min alloc floor + soft hard-regime (keep ATR score boost).

    Research composite recommended after bottleneck audit — does **not** include
    noboost (which destroyed US CAGR) nor softtrend/atrstop extras.
    """

    name: str = "turbo_highvol_minalloc_softreg"
    description: str = (
        "Highvol + min_alloc 1.5% + soft hard-regime scale 0.40 (keep ATR boost)"
    )
    min_alloc_pct: float = 0.015
    soft_hard_regime: bool = True
    regime_hard_size_scale: Optional[float] = 0.40
    boost_score_by_atr: bool = True


@dataclass
class HighVolMinAllocSectorRotStrategy(HighVolTurboStrategy):
    """Baseline minalloc + sector ETF SMA gate + rotation when full (slots or cash).

    Sector map: data/ticker_sector_map.csv (corr-to-SPDR proxy when fundamentals N/A).
    Rotation: free worst-score position when a better candidate appears and book is full.
    """

    name: str = "turbo_highvol_minalloc_sector_rot"
    description: str = (
        "minalloc + sector ETF>SMA50 gate + rotation (cash/slot full)"
    )
    min_alloc_pct: float = 0.015
    require_sector_trend: bool = True
    enable_rotation: bool = True
    rotation_min_score_edge: float = 0.05
    rotation_min_bars: int = 3
    rotation_max_per_day: int = 2
    sector_map_path: str = "data/ticker_sector_map.csv"
    sector_ma: int = 50
    sector_require_sma200: bool = False
    sector_allow_unmapped: bool = True

    def backtest_overrides(self) -> Dict[str, Any]:
        o = super().backtest_overrides()
        o["min_alloc_pct"] = float(self.min_alloc_pct)
        o["require_sector_trend"] = True
        o["enable_rotation"] = True
        o["rotation_min_score_edge"] = float(self.rotation_min_score_edge)
        o["rotation_min_bars"] = int(self.rotation_min_bars)
        o["rotation_max_per_day"] = int(self.rotation_max_per_day)
        o["sector_allow_unmapped"] = bool(self.sector_allow_unmapped)
        return o


@dataclass
class HighVolNoBoostStrategy(HighVolTurboStrategy):
    """EXP2: no score×atr_norm / vol boost (keep min_atr filter)."""

    name: str = "turbo_highvol_noboost"
    description: str = "Highvol without ATR/vol score boost"
    boost_score_by_atr: bool = False


@dataclass
class HighVolSoftRegimeStrategy(HighVolTurboStrategy):
    """EXP3: hard regime risk-off → size scale, not zero entries."""

    name: str = "turbo_highvol_softreg"
    description: str = "Highvol + soft hard-regime (size scale 0.40, not block)"
    soft_hard_regime: bool = True
    regime_hard_size_scale: Optional[float] = 0.40


@dataclass
class HighVolAtrStopStrategy(HighVolTurboStrategy):
    """EXP4: wider ATR-aware hard stop to cut shake-out churn.

    Formula: hard_stop = entry - max(entry * hard_stop_pct, hard_stop_atr_mult * atr)
    with hard_stop_pct=0.12 and hard_stop_atr_mult=2.5.
    """

    name: str = "turbo_highvol_atrstop"
    description: str = "Highvol + ATR-aware hard stop (max of 12% and 2.5×ATR)"
    hard_stop_pct: float = 0.12
    hard_stop_atr_mult: Optional[float] = 2.5


@dataclass
class HighVolSoftTrendStrategy(HighVolTurboStrategy):
    """EXP5: softer trend filter (SMA50 OR SMA20) + slightly looser SMA200 dist."""

    name: str = "turbo_highvol_softtrend"
    description: str = "Highvol + soft trend (SMA50|SMA20) + min_dist_sma200=-0.18"
    soft_trend: bool = True
    min_dist_sma200: float = -0.18


@dataclass
class HighVolFixpackStrategy(HighVolTurboStrategy):
    """Combined bottleneck fix pack (primary bake-off candidate).

    - min_alloc_pct 1.5% (no micro slots)
    - no ATR score boost
    - soft hard-regime size scale 0.40
    - ATR-aware hard stop (12% / 2.5×ATR)
    - soft trend light (SMA50|SMA20, min_dist -0.18)
    """

    name: str = "turbo_highvol_fixpack"
    description: str = (
        "Highvol fixpack: min_alloc + no boost + softreg + atrstop + softtrend"
    )
    min_alloc_pct: float = 0.015
    boost_score_by_atr: bool = False
    soft_hard_regime: bool = True
    regime_hard_size_scale: Optional[float] = 0.40
    hard_stop_pct: float = 0.12
    hard_stop_atr_mult: Optional[float] = 2.5
    soft_trend: bool = True
    min_dist_sma200: float = -0.18


def _risk_pack_adaptive(
    *,
    min_profit: float = 0.15,
    trail_only_profit: float = 0.40,
    max_dd: float = 0.18,
    ticker_cap: float = 0.12,
    ticker_pnl_frac: float = 0.20,
) -> Dict[str, Any]:
    """Shared risk pack: adaptive exit + DD kill 15–20% + ticker caps."""
    return {
        "adaptive_exit": True,
        "adaptive_mode": "auto",  # big winners → trail-only; else extend
        "adaptive_min_profit": min_profit,
        "adaptive_trail_only_profit": trail_only_profit,
        "adaptive_extend_bars": 20,
        "adaptive_max_extensions": 2,
        "adaptive_trail_k_mult": 1.25,
        "adaptive_require_regime": True,
        "adaptive_min_atr_norm": 0.015,
        "max_portfolio_dd": max_dd,  # kill new entries when DD ≥ this
        "dd_soft_scale": 0.50,
        "ticker_max_capital_pct": ticker_cap,
        "ticker_max_realized_pnl_frac": ticker_pnl_frac,
    }


@dataclass
class AdaptiveHighVolTurboStrategy(HighVolTurboStrategy):
    """Highvol + adaptive exit + DD kill + ticker caps (static universe)."""

    name: str = "turbo_highvol_adaptive"
    description: str = "Highvol+strict + adaptive auto + DD kill 18% + ticker caps"
    vol_target: float = 0.038
    max_position_pct: float = 0.12
    max_positions: int = 12
    max_horizon: int = 38

    def backtest_overrides(self) -> Dict[str, Any]:
        o = super().backtest_overrides()
        o.update(
            {
                "hard_stop_pct": 0.11,
                "max_entries_per_day": 8,
                **_risk_pack_adaptive(
                    min_profit=0.15,
                    trail_only_profit=0.35,
                    max_dd=0.18,
                    ticker_cap=0.12,
                    ticker_pnl_frac=0.20,
                ),
                "max_position_pct": self.max_position_pct,
                "volatility_target_pct": self.vol_target,
                "max_positions": self.max_positions,
                "max_horizon": self.max_horizon,
                "k_atr": self.k_atr,
            }
        )
        return o


@dataclass
class AdaptiveTurboStrictStrategy(AggressiveTurboStrategy):
    """turbo_strict + full risk pack (adaptive + DD kill + ticker caps)."""

    name: str = "turbo_strict_adaptive"
    description: str = "turbo_strict + adaptive auto + DD kill 18% + ticker caps"
    min_confidence: float = 0.28
    vol_target: float = 0.032
    max_position_pct: float = 0.12
    max_positions: int = 12
    k_atr: float = 3.8
    max_horizon: int = 38
    regime_filter: str = "strict_dual_golden"
    require_momentum: bool = False
    require_trend: bool = True
    max_atr_pct: float = 0.16
    min_dist_sma200: float = -0.15
    risk_off_scale: float = 0.95

    def backtest_overrides(self) -> Dict[str, Any]:
        o = super().backtest_overrides()
        o.update(
            {
                "hard_stop_pct": 0.11,
                "max_entries_per_day": 8,
                **_risk_pack_adaptive(
                    min_profit=0.12,
                    trail_only_profit=0.35,
                    max_dd=0.18,
                    ticker_cap=0.12,
                    ticker_pnl_frac=0.20,
                ),
                "max_position_pct": self.max_position_pct,
                "volatility_target_pct": self.vol_target,
                "max_positions": self.max_positions,
            }
        )
        return o


@dataclass
class RobustHighVolDynamicStrategy(HighVolTurboStrategy):
    """Full audit pack: adaptive + DD kill + ticker caps + yearly highvol re-rank.

    Universe each OOS year: top-N realized vol using only data ≤ Dec 31 of prior year.
    ES is NEVER used to tune these parameters (domain stress only).
    """

    name: str = "turbo_highvol_robust"
    description: str = (
        "Dynamic highvol (causal yearly) + adaptive auto + DD kill 18% + ticker caps"
    )
    dynamic_highvol: bool = True
    universe_source_file: str = "good_tickers_filtrados.txt"
    universe_n: int = 80
    universe_scan_limit: int = 500
    vol_target: float = 0.038
    max_position_pct: float = 0.12
    max_positions: int = 12
    max_horizon: int = 38

    def backtest_overrides(self) -> Dict[str, Any]:
        o = super().backtest_overrides()
        o.update(
            {
                "hard_stop_pct": 0.11,
                "max_entries_per_day": 8,
                **_risk_pack_adaptive(
                    min_profit=0.15,
                    trail_only_profit=0.35,
                    max_dd=0.18,
                    ticker_cap=0.12,
                    ticker_pnl_frac=0.18,
                ),
                "max_position_pct": self.max_position_pct,
                "volatility_target_pct": self.vol_target,
                "max_positions": self.max_positions,
                "max_horizon": self.max_horizon,
                "k_atr": self.k_atr,
            }
        )
        return o


@dataclass
class QualityTurboStrategy(AggressiveTurboStrategy):
    """Turbo + quality filter: fund PIT when present, else price quality (trend+mom).

    Local fundamentals often lack pre-2018 history; price proxy = above SMA200,
    positive ret_1m, and not-extreme vol (stable compounders / winners).
    """

    name: str = "turbo_quality"
    description: str = "Turbo + quality (fund PIT if any, else price quality) + strict"
    needs_fundamentals: bool = True
    min_fund_quality: float = 0.5
    require_positive_rev_yoy: bool = False
    regime_filter: str = "strict_dual_golden"
    min_confidence: float = 0.30
    vol_target: float = 0.032
    max_position_pct: float = 0.24
    use_price_quality_fallback: bool = True

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        sig, score = super().generate_signals(df, cfg)
        fund_mask = None
        if "fund_quality" in df.columns:
            fq = df["fund_quality"].fillna(0.0).to_numpy()
            # fund data only meaningful when non-zero history has been merged
            has_signal = np.isfinite(df["fund_rev_yoy"].to_numpy()) if "fund_rev_yoy" in df.columns else (fq != 0)
            fund_mask = (fq >= self.min_fund_quality) & has_signal
            score = np.where(has_signal, score * (1.0 + np.clip(fq, 0, 3.0) / 2.0), score)

        price_mask = None
        if self.use_price_quality_fallback:
            ok = np.ones(len(df), dtype=bool)
            if "dist_sma_200" in df.columns:
                ok = ok & (df["dist_sma_200"].to_numpy() > 0.0)
            if "ret_1m" in df.columns:
                ok = ok & (df["ret_1m"].to_numpy() >= 0.02)
            if "volatility_20" in df.columns:
                # avoid pure lottery tickets: moderate-high vol ok, chaos no
                ok = ok & (df["volatility_20"].to_numpy() <= 0.85)
            price_mask = ok
            score = score * np.where(ok, 1.25, 0.85)

        if fund_mask is not None and price_mask is not None:
            # Pass if fund quality OR price quality (when fund missing)
            has_fund = fund_mask | (
                np.isfinite(df["fund_rev_yoy"].to_numpy())
                if "fund_rev_yoy" in df.columns
                else np.zeros(len(df), dtype=bool)
            )
            # Prefer: (has fund and fund_mask) or (no fund and price_mask) or (both)
            no_fund = ~np.isfinite(df["fund_rev_yoy"].to_numpy()) if "fund_rev_yoy" in df.columns else np.ones(len(df), dtype=bool)
            combined = (fund_mask) | (no_fund & price_mask) | (price_mask & fund_mask)
            sig = sig & combined
        elif fund_mask is not None:
            sig = sig & fund_mask
        elif price_mask is not None:
            sig = sig & price_mask
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


@dataclass
class QualityHighVolTurboStrategy(QualityTurboStrategy):
    """Blend: quality filter + high-vol scoring."""

    name: str = "turbo_quality_highvol"
    description: str = "Quality PIT fund filter + high-vol score boost + strict"
    max_atr_pct: float = 0.20
    min_atr_norm: float = 0.015
    vol_target: float = 0.038

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        sig, score = QualityTurboStrategy.generate_signals(self, df, cfg)
        if "atr_norm" in df.columns:
            atr = df["atr_norm"].to_numpy()
            sig = sig & (atr >= self.min_atr_norm)
            score = score * (1.0 + np.clip(atr, 0, 0.15) / 0.05)
        if "volatility_20" in df.columns:
            score = score * (1.0 + np.clip(df["volatility_20"].to_numpy(), 0, 1.0))
        return pd.Series(sig, index=df.index), pd.Series(score, index=df.index)


@dataclass
class TrendMomentumStrategy(Strategy):
    """Classic trend + momentum (no ML)."""

    name: str = "trend_momentum"
    description: str = "Close>SMA50, dist_SMA200>0, ret_1m high; score=momentum"
    needs_training: bool = False
    momentum_min: float = 0.03

    def backtest_overrides(self) -> Dict[str, Any]:
        return {
            "min_confidence": 0.0,  # unused
            "require_trend": False,
            "require_momentum": False,
            "k_atr": 3.0,
            "max_horizon": 25,
            "volatility_target_pct": 0.016,
            "max_position_pct": 0.12,
            "max_positions": 10,
            "risk_off_scale": 0.45,
            "max_atr_pct": 0.08,
        }

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        need = ["close", "sma_50", "dist_sma_200", "ret_1m", "atr_norm"]
        if not _ensure_cols(df, need):
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        sig = (
            (df["close"] > df["sma_50"])
            & (df["dist_sma_200"] > 0.0)
            & (df["ret_1m"] >= self.momentum_min)
            & (df["atr_norm"] <= cfg.max_atr_pct)
        )
        score = df["ret_1m"].clip(lower=0) * (1.0 + df["dist_sma_50"].fillna(0).clip(lower=0))
        return sig.fillna(False), score.fillna(0.0)


@dataclass
class RsiPullbackStrategy(Strategy):
    """Buy pullbacks in uptrend (RSI oversold)."""

    name: str = "rsi_pullback"
    description: str = "Uptrend SMA200 + RSI14 oversold bounce"
    needs_training: bool = False
    rsi_max: float = 38.0

    def backtest_overrides(self) -> Dict[str, Any]:
        return {
            "k_atr": 2.5,
            "max_horizon": 15,
            "hard_stop_pct": 0.06,
            "volatility_target_pct": 0.014,
            "max_position_pct": 0.12,
            "max_positions": 12,
            "risk_off_scale": 0.5,
            "max_atr_pct": 0.07,
        }

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        need = ["close", "sma_200", "rsi_14", "ret_1m", "atr_norm"]
        if not _ensure_cols(df, need):
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        # mild recovery day: ret_1m not deeply negative
        sig = (
            (df["close"] > df["sma_200"])
            & (df["rsi_14"] <= self.rsi_max)
            & (df["rsi_14"] >= 20)
            & (df["ret_1m"] > -0.05)
            & (df["atr_norm"] <= cfg.max_atr_pct)
        )
        score = (self.rsi_max - df["rsi_14"]).clip(lower=0) / 20.0
        return sig.fillna(False), score.fillna(0.0)


@dataclass
class VolBreakoutStrategy(Strategy):
    """Breakout near 20d high with volume confirmation."""

    name: str = "vol_breakout"
    description: str = "Near 20d high + volume_ratio surge + trend"
    needs_training: bool = False

    def backtest_overrides(self) -> Dict[str, Any]:
        return {
            "k_atr": 2.8,
            "max_horizon": 12,
            "volatility_target_pct": 0.015,
            "max_position_pct": 0.10,
            "max_positions": 8,
            "risk_off_scale": 0.4,
            "max_atr_pct": 0.09,
        }

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        need = ["close", "high", "sma_50", "volume_ratio", "atr_norm"]
        if not _ensure_cols(df, need):
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        high_20 = df["high"].rolling(20, min_periods=10).max()
        near_high = df["close"] >= high_20 * 0.98
        sig = (
            near_high
            & (df["close"] > df["sma_50"])
            & (df["volume_ratio"] >= 1.25)
            & (df["atr_norm"] <= cfg.max_atr_pct)
        )
        score = (df["volume_ratio"].clip(upper=3) - 1.0) * (
            df["close"] / high_20.replace(0, np.nan)
        ).fillna(0)
        return sig.fillna(False), score.fillna(0.0)


@dataclass
class DefensiveTrendStrategy(Strategy):
    """Trend momentum with stricter regime and stops (target: survive 2022)."""

    name: str = "defensive_trend"
    description: str = "Trend mom + soft regime hard + tighter risk"
    needs_training: bool = False

    def backtest_overrides(self) -> Dict[str, Any]:
        return {
            "k_atr": 2.2,
            "max_horizon": 12,
            "hard_stop_pct": 0.05,
            "volatility_target_pct": 0.012,
            "max_position_pct": 0.10,
            "max_positions": 8,
            "risk_off_scale": 0.25,
            "max_atr_pct": 0.06,
            "min_dist_sma200": 0.0,
            "max_entries_per_day": 3,
            # force soft regime as hard: only full risk when soft True handled in backtest
        }

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        need = ["close", "sma_50", "dist_sma_200", "ret_1m", "atr_norm", "rsi_14"]
        if not _ensure_cols(df, need):
            z = pd.Series(False, index=df.index)
            return z, pd.Series(0.0, index=df.index)
        sig = (
            (df["close"] > df["sma_50"])
            & (df["dist_sma_200"] >= 0.0)
            & (df["ret_1m"] >= 0.025)
            & (df["atr_norm"] <= cfg.max_atr_pct)
            & (df["rsi_14"] < 70)
            & (df["rsi_14"] > 45)
        )
        score = df["ret_1m"].clip(lower=0) * (1 + df["dist_sma_200"].clip(lower=0))
        return sig.fillna(False), score.fillna(0.0)


@dataclass
class HybridTrendMLStrategy(Strategy):
    """ML primary + hard rule overlay (trend + not overextended RSI)."""

    name: str = "hybrid_trend_ml"
    description: str = "Champion ML signals AND close>SMA50 AND RSI14<65"
    needs_training: bool = True
    _inner: Optional[ChampionMLStrategy] = None

    def __post_init__(self) -> None:
        self._inner = ChampionMLStrategy(use_meta=True, min_confidence=0.40)

    def backtest_overrides(self) -> Dict[str, Any]:
        o = self._inner.backtest_overrides() if self._inner else {}
        o.update(
            {
                "max_horizon": 18,
                "k_atr": 2.8,
                "volatility_target_pct": 0.015,
                "max_position_pct": 0.12,
                "risk_off_scale": 0.40,
            }
        )
        return o

    def train(self, train_df: pd.DataFrame, year: int) -> None:
        assert self._inner is not None
        self._inner.train(train_df, year)

    def generate_signals(
        self, df: pd.DataFrame, cfg: BacktestConfig
    ) -> Tuple[pd.Series, pd.Series]:
        assert self._inner is not None
        sig, score = self._inner.generate_signals(df, cfg)
        if "sma_50" in df.columns:
            sig = sig & (df["close"].to_numpy() > df["sma_50"].to_numpy())
        if "rsi_14" in df.columns:
            sig = sig & (df["rsi_14"].to_numpy() < 65)
        if "dist_sma_200" in df.columns:
            sig = sig & (df["dist_sma_200"].to_numpy() > -0.02)
        return pd.Series(sig, index=df.index), score


def all_strategies() -> List[Strategy]:
    return [
        ChampionMLStrategy(),
        AggressiveMLStrategy(),
        AggressiveMaxStrategy(),
        AggressiveLetRunStrategy(),
        AggressiveTurboStrategy(),
        # Industry index overlays on turbo
        turbo_with_regime("sma200", "sma200"),
        turbo_with_regime("sma50", "sma50"),
        turbo_with_regime("dual_50_200", "dual_ma"),
        turbo_with_regime("golden_cross", "golden"),
        turbo_with_regime("strict_dual_golden", "strict"),
        turbo_with_regime("abs_mom_12m", "abs12m"),
        turbo_with_regime("abs_mom_6m", "abs6m"),
        turbo_with_regime("sma200_hysteresis", "sma200h"),
        turbo_with_regime("none", "no_regime"),
        # IBEX-specific regime family (design 2010–2017; use with preferred_index=IBEX)
        turbo_with_regime("ibex_mom12", "ibex_mom12"),
        turbo_with_regime("ibex_sma200_hyst3", "ibex_hyst"),
        turbo_with_regime("ibex_not_bear", "ibex_not_bear"),
        turbo_with_regime("ibex_or_trend", "ibex_or"),
        turbo_with_regime("ibex_soft_block", "ibex_soft"),
        turbo_with_regime("portable_not_deep_bear", "portable_ndb"),
        turbo_with_regime("portable_sma200", "portable_sma200"),
        turbo_with_regime("portable_abs_mom_12m", "portable_mom12"),
        turbo_rel_with_regime("legacy_sma50", "legacy"),
        turbo_rel_with_regime("portable_not_deep_bear", "portable_ndb"),
        ChampionMLStrategy(
            name="champion_ml_rel",
            description="Champion ML with M2_REL features (fea-03.v1)",
            feature_names=M2_REL_FEATURE_NAMES,
        ),
        HighVolTurboStrategy(),
        HighVolMinAllocStrategy(),
        HighVolMinAllocSoftRegStrategy(),
        HighVolMinAllocSectorRotStrategy(),
        HighVolNoBoostStrategy(),
        HighVolSoftRegimeStrategy(),
        HighVolAtrStopStrategy(),
        HighVolSoftTrendStrategy(),
        HighVolFixpackStrategy(),
        AdaptiveHighVolTurboStrategy(),
        AdaptiveTurboStrictStrategy(),
        RobustHighVolDynamicStrategy(),
        QualityTurboStrategy(),
        QualityHighVolTurboStrategy(),
        HybridTrendMLStrategy(),
        TrendMomentumStrategy(),
        RsiPullbackStrategy(),
        VolBreakoutStrategy(),
        DefensiveTrendStrategy(),
    ]


# US research baseline alias (promote minalloc — not live trading).
RESEARCH_BASELINE_US: str = "turbo_highvol_minalloc"
_STRATEGY_ALIASES: Dict[str, str] = {
    "us_research_baseline": RESEARCH_BASELINE_US,
    "research_baseline": RESEARCH_BASELINE_US,
    "research_baseline_us": RESEARCH_BASELINE_US,
}


def get_strategy(name: str) -> Strategy:
    key = (name or "").strip()
    key = _STRATEGY_ALIASES.get(key, key)
    for s in all_strategies():
        if s.name == key:
            return s
    raise KeyError(f"Unknown strategy: {name}. Available: {[s.name for s in all_strategies()]}")
