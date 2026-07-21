"""Versioned research configs (features, labels) — SSOT for trad_research."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from trad_research.features import M1_FEATURE_NAMES, M2_FEATURE_NAMES, M2_REL_FEATURE_NAMES


@dataclass(frozen=True)
class FeatureConfig:
    """Feature contract aligned with lean_strategy.modules.config StrategyConfig M2."""

    m2_names: Tuple[str, ...] = M2_FEATURE_NAMES
    m1_names: Tuple[str, ...] = M1_FEATURE_NAMES
    relative_names: Tuple[str, ...] = M2_REL_FEATURE_NAMES
    atr_period: int = 14
    rsi_periods: Tuple[int, ...] = (7, 14, 21)
    sma_periods: Tuple[int, ...] = (50, 200)
    volatility_window: int = 20
    volume_window: int = 20
    min_history: int = 220
    version: str = "fea-01.v1"

    @property
    def m2_count(self) -> int:
        return len(self.m2_names)

    def assert_lean_parity(self) -> None:
        """Fail if research names diverge from Lean ground-truth tuple."""
        # Import lazily so tests work without QuantConnect
        expected = (
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
        if tuple(self.m2_names) != expected:
            raise AssertionError(
                f"M2 feature names drift from Lean contract.\n"
                f"expected={expected}\nactual={self.m2_names}"
            )
        if self.m2_count != 17:
            raise AssertionError(f"M2 must have 17 features, got {self.m2_count}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelConfig:
    """Triple-barrier labeling parameters (targets only — never features)."""

    k_tp: float = 2.5
    k_sl: float = 1.5
    max_horizon: int = 20
    # Encoding for ML: 0=SELL(SL), 1=HOLD, 2=BUY(TP)
    sell_class: int = 0
    hold_class: int = 1
    buy_class: int = 2
    version: str = "lab-01.v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Defaults used by walk-forward champion path
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_LABEL_CONFIG = LabelConfig()
