"""Public API for the DRL research platform.

This module intentionally avoids importing heavy optional dependencies at import
time. Objects are exposed lazily via ``__getattr__`` so that modules that
require extra packages (torch, transformers, etc.) are only imported when they
are actually used.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, Tuple

__all__ = [
    # Data
    "DataPipeline",
    "PipelineConfig",
    "IndicatorConfig",

    # Environment
    "TradingEnvironment",
    "EnvironmentConfig",
    "REWARD_REGISTRY",

    # Features
    "SentimentFeatureExtractor",
    "SentimentConfig",
    "Autoencoder",
    "TemporalConvNet",
    "EncoderConfig",

    # Training
    "TrainingOrchestrator",
    "TrainingConfig",
    "TrackingConfig",

    # Validation
    "PurgedKFoldValidator",
    "PurgedKFoldConfig",

    # Multi-agent
    "SignalAgent",
    "PortfolioAgent",
    "SignalAgentOutput",
]

# Mapping from exported names to ``(module, attribute)`` pairs.
# Modules are only imported when their attributes are accessed.
_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Data
    "DataPipeline": ("drl_platform.data_pipeline", "DataPipeline"),
    "PipelineConfig": ("drl_platform.data_pipeline", "PipelineConfig"),
    "IndicatorConfig": ("drl_platform.data_pipeline", "IndicatorConfig"),

    # Environment
    "TradingEnvironment": ("drl_platform.env.trading_env", "TradingEnvironment"),
    "EnvironmentConfig": ("drl_platform.env.trading_env", "EnvironmentConfig"),
    "REWARD_REGISTRY": ("drl_platform.env.rewards", "REWARD_REGISTRY"),

    # Features
    "SentimentFeatureExtractor": ("drl_platform.features.sentiment", "SentimentFeatureExtractor"),
    "SentimentConfig": ("drl_platform.features.sentiment", "SentimentConfig"),
    "Autoencoder": ("drl_platform.features.state_encoders", "Autoencoder"),
    "TemporalConvNet": ("drl_platform.features.state_encoders", "TemporalConvNet"),
    "EncoderConfig": ("drl_platform.features.state_encoders", "EncoderConfig"),

    # Training
    "TrainingOrchestrator": ("drl_platform.orchestrator", "TrainingOrchestrator"),
    "TrainingConfig": ("drl_platform.orchestrator", "TrainingConfig"),
    "TrackingConfig": ("drl_platform.orchestrator", "TrackingConfig"),

    # Validation
    "PurgedKFoldValidator": ("drl_platform.validation", "PurgedKFoldValidator"),
    "PurgedKFoldConfig": ("drl_platform.validation", "PurgedKFoldConfig"),

    # Multi-agent
    "SignalAgent": ("drl_platform.multi_agent", "SignalAgent"),
    "PortfolioAgent": ("drl_platform.multi_agent", "PortfolioAgent"),
    "SignalAgentOutput": ("drl_platform.multi_agent", "SignalAgentOutput"),
}


def __getattr__(name: str) -> Any:
    """Dynamically resolve attributes defined in :data:`__all__`."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'drl_platform' has no attribute '{name}'")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr  # Cache for subsequent lookups
    return attr


def __dir__() -> list[str]:
    """Return dynamic module attributes for autocomplete and introspection."""
    return sorted(set(__all__))


if TYPE_CHECKING:  # pragma: no cover - hints for static analyzers
    from drl_platform.data_pipeline import DataPipeline, PipelineConfig, IndicatorConfig
    from drl_platform.env.trading_env import TradingEnvironment, EnvironmentConfig
    from drl_platform.env.rewards import REWARD_REGISTRY
    from drl_platform.features.sentiment import SentimentFeatureExtractor, SentimentConfig
    from drl_platform.features.state_encoders import Autoencoder, TemporalConvNet, EncoderConfig
    from drl_platform.orchestrator import TrainingOrchestrator, TrainingConfig, TrackingConfig
    from drl_platform.validation import PurgedKFoldValidator, PurgedKFoldConfig
    from drl_platform.multi_agent import SignalAgent, PortfolioAgent, SignalAgentOutput
