"""DRL research platform modules."""
from .data_pipeline import DataPipeline, PipelineConfig, IndicatorConfig
from .env.trading_env import TradingEnvironment, EnvironmentConfig
from .env.rewards import REWARD_REGISTRY
from .features.sentiment import SentimentFeatureExtractor, SentimentConfig
from .features.state_encoders import Autoencoder, TemporalConvNet, EncoderConfig
from .orchestrator import TrainingOrchestrator, TrainingConfig, TrackingConfig
from .validation import PurgedKFoldValidator, PurgedKFoldConfig
from .multi_agent import SignalAgent, PortfolioAgent, SignalAgentOutput


__all__ = [
    # Módulo 1: Data
    "DataPipeline",
    "PipelineConfig",
    "IndicatorConfig",
    
    # Módulo 2: Environment
    "TradingEnvironment",
    "EnvironmentConfig",
    "REWARD_REGISTRY",
    
    # Módulos Avanzados: Features
    "SentimentFeatureExtractor",
    "SentimentConfig",
    "Autoencoder",
    "TemporalConvNet",
    "EncoderConfig",
    
    # Módulo 3: Training
    "TrainingOrchestrator",
    "TrainingConfig",
    "TrackingConfig",
    
    # Módulo 4: Validation
    "PurgedKFoldValidator",
    "PurgedKFoldConfig",
    
    # Módulo Avanzado: MARL
    "SignalAgent",
    "PortfolioAgent",
    "SignalAgentOutput"
]