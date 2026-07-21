"""Local equity ML research: features, labels, walk-forward, backtest, metrics."""

from trad_research.config import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_LABEL_CONFIG,
    FeatureConfig,
    LabelConfig,
)

__version__ = "0.2.0"
__all__ = [
    "FeatureConfig",
    "LabelConfig",
    "DEFAULT_FEATURE_CONFIG",
    "DEFAULT_LABEL_CONFIG",
]
