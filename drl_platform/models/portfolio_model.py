"""Portfolio-aware neural network architectures for RLlib."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
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
    num_tickers: Optional[int] = None
    feature_dim: Optional[int] = None


class PortfolioSpatialModel(TorchModelV2, nn.Module):
    """Processes (tickers, features) observations without flattening."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **model_kwargs,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        raw_cfg = dict(model_config.get("custom_model_config", {}))
        if model_kwargs:
            raw_cfg.update(model_kwargs)
        cfg = self._coerce_config(raw_cfg)

        self.ticker_dim, self.feature_dim = self._resolve_dimensions(obs_space, cfg)

        self.embed = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, cfg.per_ticker_dim),
            nn.GELU(),
            nn.Linear(cfg.per_ticker_dim, cfg.per_ticker_dim),
            nn.GELU(),
        )

        # --- INICIO DE LA CORRECCIÓN (Global Pooling) ---
        # La dimensión agregada ya no es (N_Tickers * Emb_Dim), 
        # sino solo (Emb_Dim) porque usaremos Global Average Pooling.
        aggregated = cfg.per_ticker_dim
        # --- FIN DE LA CORRECCIÓN ---
        
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
        if "num_tickers" in raw:
            try:
                config.num_tickers = max(1, int(raw["num_tickers"]))
            except (TypeError, ValueError):
                config.num_tickers = None
        if "feature_dim" in raw:
            try:
                config.feature_dim = max(1, int(raw["feature_dim"]))
            except (TypeError, ValueError):
                config.feature_dim = None
        return config

    @staticmethod
    def _resolve_dimensions(obs_space, cfg: PortfolioModelConfig) -> tuple[int, int]:
        shape = tuple(int(dim) for dim in getattr(obs_space, "shape", ()) if dim is not None)
        if len(shape) == 2:
            return shape[0], shape[1]

        total = 1
        if not shape:
            raise ValueError("PortfolioSpatialModel requiere un espacio de observación válido")
        for dim in shape:
            total *= max(1, dim)

        tickers = cfg.num_tickers or 0
        features = cfg.feature_dim or 0
        if tickers > 0 and features > 0:
            product = tickers * features
            if product != total:
                if total % tickers == 0:
                    features = total // tickers
                elif total % features == 0:
                    tickers = total // features
                else:
                    tickers, features = 1, total
        elif tickers > 0:
            if total % tickers == 0:
                features = total // tickers
            else:
                tickers, features = 1, total
        elif features > 0:
            if total % features == 0:
                tickers = total // features
            else:
                tickers, features = 1, total
        else:
            tickers, features = 1, total

        return int(tickers), int(features)

    def forward(self, input_dict: Dict[str, TensorType], state, seq_lens):
        obs = input_dict["obs"].float()
        obs = self._restore_observation_shape(obs)
        per_ticker = self.embed(obs)
        
        # --- INICIO DE LA CORRECCIÓN (Global Pooling) ---
        # En lugar de aplanar (N, Tickers, Feats) -> (N, Tickers * Feats),
        # agregamos los tickers calculando la media.
        # (N, Tickers, Feats) -> (N, Feats)
        flat = torch.mean(per_ticker, dim=1)
        # --- FIN DE LA CORRECCIÓN ---
        
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
        # Esta lógica maneja si RLlib aplana la observación 2D a 1D
        if obs.dim() == 2 and obs.shape[1] == self.ticker_dim * self.feature_dim:
            return obs.view(obs.shape[0], self.ticker_dim, self.feature_dim)
        if obs.dim() > 3:
             # Caso de borde: si hay una dimensión de tiempo, tomar el último paso
            if obs.shape[1] > 100: # Asumir que [0] es el batch
                 return obs[:,-1,:,:].view(obs.shape[0], self.ticker_dim, self.feature_dim)
            else:
                 return obs.view(obs.shape[0], self.ticker_dim, self.feature_dim)
        
        # Fallback
        return obs.view(obs.shape[0], self.ticker_dim, self.feature_dim)


_MODEL_REGISTERED = False


def ensure_portfolio_model_registered() -> None:
    global _MODEL_REGISTERED
    if _MODEL_REGISTERED:
        return
    ModelCatalog.register_custom_model("portfolio_spatial_model", PortfolioSpatialModel)
    _MODEL_REGISTERED = True