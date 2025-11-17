"""Portfolio-aware neural network architectures for RLlib."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType


@dataclass
class PortfolioModelConfig:
    """Hyperparameters for :class:`PortfolioSpatialModel`."""

    per_ticker_dim: int = 64
    global_dim: int = 256
    dropout: float = 0.0


class PortfolioSpatialModel(TorchModelV2, nn.Module):
    """Processes (tickers, features) observations without flattening."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if len(obs_space.shape) != 2:
            raise ValueError(
                "PortfolioSpatialModel espera observaciones 2D (tickers, features)."
            )

        self.ticker_dim = int(obs_space.shape[0])
        self.feature_dim = int(obs_space.shape[1])
        cfg = self._coerce_config(model_config.get("custom_model_config", {}))

        self.embed = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, cfg.per_ticker_dim),
            nn.GELU(),
            nn.Linear(cfg.per_ticker_dim, cfg.per_ticker_dim),
            nn.GELU(),
        )

        aggregated = cfg.per_ticker_dim * self.ticker_dim
        head_layers = [nn.LayerNorm(aggregated), nn.Linear(aggregated, cfg.global_dim), nn.GELU()]
        if cfg.dropout > 0:
            head_layers.append(nn.Dropout(cfg.dropout))
        head_layers.append(nn.Linear(cfg.global_dim, num_outputs))
        self.policy_head = nn.Sequential(*head_layers)

        value_layers = [
            nn.LayerNorm(aggregated),
            nn.Linear(aggregated, cfg.global_dim),
            nn.GELU(),
            nn.Linear(cfg.global_dim, 1),
        ]
        self.value_head = nn.Sequential(*value_layers)
        self._last_features: TensorType | None = None

    @staticmethod
    def _coerce_config(raw: Dict[str, Any]) -> PortfolioModelConfig:
        config = PortfolioModelConfig()
        if not isinstance(raw, dict):
            return config
        if "per_ticker_dim" in raw:
            config.per_ticker_dim = max(16, int(raw["per_ticker_dim"]))
        if "global_dim" in raw:
            config.global_dim = max(32, int(raw["global_dim"]))
        if "dropout" in raw:
            config.dropout = float(raw["dropout"])
        return config

    def forward(self, input_dict: Dict[str, TensorType], state, seq_lens):
        obs = input_dict["obs"].float()
        obs = self._restore_observation_shape(obs)
        per_ticker = self.embed(obs)
        flat = per_ticker.reshape(per_ticker.shape[0], -1)
        self._last_features = flat
        logits = self.policy_head(flat)
        return logits, state

    def value_function(self) -> TensorType:
        if self._last_features is None:
            raise RuntimeError("value_function() llamado antes de forward()")
        value = self.value_head(self._last_features)
        return value.squeeze(-1)

    def _restore_observation_shape(self, obs: TensorType) -> TensorType:
        if obs.dim() == 3 and obs.shape[1] == self.ticker_dim and obs.shape[2] == self.feature_dim:
            return obs
        if obs.dim() == 2 and obs.shape[1] == self.ticker_dim * self.feature_dim:
            return obs.view(obs.shape[0], self.ticker_dim, self.feature_dim)
        if obs.dim() > 3:
            return obs.view(obs.shape[0], self.ticker_dim, self.feature_dim)
        return obs.view(obs.shape[0], self.ticker_dim, self.feature_dim)


_MODEL_REGISTERED = False


def ensure_portfolio_model_registered() -> None:
    global _MODEL_REGISTERED
    if _MODEL_REGISTERED:
        return
    ModelCatalog.register_custom_model("portfolio_spatial_model", PortfolioSpatialModel)
    _MODEL_REGISTERED = True