#!/usr/bin/env python3
"""Multi-agent scaffolding for hierarchical trading systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Any

import numpy as np


@dataclass
class SignalAgentOutput:
    """Datos de salida de un agente de señal."""
    ticker: str
    action: float  # -1 short, +1 long
    confidence: float


class SignalAgent:
    """Consumes ticker specific state and proposes trade signals."""

    def __init__(self, policy: Any):
        self.policy = policy

    def act(self, ticker: str, state: np.ndarray) -> SignalAgentOutput:
        """Propone una acción basada en el estado del ticker."""
        # La lógica de 'policy' dependerá de la implementación (ej. un modelo DRL)
        action, confidence = self.policy(state)
        return SignalAgentOutput(ticker=ticker, action=action, confidence=confidence)


class PortfolioAgent:
    """Aggregates outputs from signal agents and determines allocations."""

    def __init__(self, allocator: Any):
        self.allocator = allocator

    def allocate(self, signals: Iterable[SignalAgentOutput], macro_state: np.ndarray) -> Dict[str, float]:
        """Asigna capital basado en señales de agentes y estado macro."""
        # 'allocator' sería un optimizador o un segundo agente DRL
        return self.allocator(signals, macro_state)