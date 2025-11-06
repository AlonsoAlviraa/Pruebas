#!/usr/bin/env python3
"""State representation learning utilities (autoencoders/TCN).

Estos módulos (requieren PyTorch) están diseñados para comprimir el
vector de estado de entrada (ej. 50 features) en una representación
latente más pequeña (ej. 32 features), que luego se pasa al
agente DRL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None  # type: ignore


@dataclass
class EncoderConfig:
    """Configuración para modelos de codificación de estado."""
    latent_dim: int = 32
    input_dim: Optional[int] = None
    sequence_length: int = 30  # Para modelos temporales como TCN


class Autoencoder(nn.Module):  # pragma: no cover - neural network component
    """Un Autoencoder simple (Linear) para compresión de estado."""
    def __init__(self, config: EncoderConfig):
        if torch is None or nn is None:
            raise ImportError("PyTorch es requerido para Autoencoder. Instala con `pip install torch`")
        super().__init__()
        
        if config.input_dim is None:
            raise ValueError("input_dim debe ser provisto para Autoencoder")
            
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Devuelve (reconstrucción, representación_latente)."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class TemporalConvNet(nn.Module):  # pragma: no cover - neural network component
    """Una Red Convolucional Temporal (TCN) simple para codificar secuencias."""
    def __init__(self, config: EncoderConfig):
        if torch is None or nn is None:
            raise ImportError("PyTorch es requerido para TemporalConvNet. Instala con `pip install torch`")
        super().__init__()
        
        if config.input_dim is None:
            raise ValueError("input_dim debe ser provisto para TCN (como num_features)")
            
        self.network = nn.Sequential(
            # Entrada: (Batch, NumFeatures, SeqLength)
            nn.Conv1d(config.input_dim, 64, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(64, config.latent_dim, kernel_size=3, padding=2, dilation=2),
            nn.AdaptiveAvgPool1d(1), # Global Average Pooling
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve (representación_latente)."""
        # x debe tener la forma (Batch, NumFeatures, SeqLength)
        if x.dim() != 3:
             raise ValueError(f"TCN espera entrada (Batch, Features, SeqLen), pero recibió {x.shape}")
             
        latent = self.network(x)
        return latent.squeeze(-1) # (Batch, LatentDim)