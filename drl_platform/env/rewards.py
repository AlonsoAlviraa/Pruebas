#!/usr/bin/env python3
"""Reward calculation utilities for advanced trading environments."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RewardState:
    """Keeps track of portfolio metrics required for reward calculations."""

    equity_curve: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)


class RewardCalculator(abc.ABC):
    """Interface for all reward calculators."""

    def __init__(self):
        self.state = RewardState(equity_curve=[], returns=[])

    def reset(self, initial_equity: float) -> None:
        """Reinicia el estado del calculador con el capital inicial."""
        self.state = RewardState(equity_curve=[initial_equity], returns=[])

    def update(self, new_equity: float) -> float:
        """Actualiza el estado con el nuevo capital y calcula la recompensa."""
        curve = self.state.equity_curve
        ret_series = self.state.returns
        
        prev_equity = curve[-1] if curve else new_equity
        curve.append(new_equity)
        
        portfolio_return = (new_equity - prev_equity) / prev_equity if prev_equity != 0 else 0.0
        ret_series.append(portfolio_return)
        
        return self._compute_reward(curve, ret_series)

    @abc.abstractmethod
    def _compute_reward(self, equity_curve: List[float], returns: List[float]) -> float:
        """Lógica de cálculo de recompensa específica de la subclase."""
        raise NotImplementedError


class PnLReward(RewardCalculator):
    """Reward equal to the change in portfolio value."""

    def _compute_reward(self, equity_curve: List[float], returns: List[float]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        # Recompensa es el PnL del último paso
        return equity_curve[-1] - equity_curve[-2]


class SharpeReward(RewardCalculator):
    """Reward based on incremental Sharpe ratio."""

    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: float = 252):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self._prev_sharpe = 0.0

    def reset(self, initial_equity: float) -> None:
        super().reset(initial_equity)
        self._prev_sharpe = 0.0

    def _compute_reward(self, equity_curve: List[float], returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
            
        excess_returns = np.array(returns) - self.risk_free_rate / self.annualization_factor
        mean = excess_returns.mean()
        std = excess_returns.std(ddof=1)
        
        current_sharpe = (mean / std) * np.sqrt(self.annualization_factor) if std > 0 else 0.0
        
        reward = current_sharpe - self._prev_sharpe
        self._prev_sharpe = current_sharpe
        return reward


class SortinoReward(RewardCalculator):
    """Reward derived from incremental Sortino ratio."""

    def __init__(self, target_return: float = 0.0, annualization_factor: float = 252):
        super().__init__()
        self.target_return_daily = target_return / annualization_factor
        self.annualization_factor = annualization_factor
        self._prev_sortino = 0.0

    def reset(self, initial_equity: float) -> None:
        super().reset(initial_equity)
        self._prev_sortino = 0.0

    def _compute_reward(self, equity_curve: List[float], returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
            
        returns_np = np.array(returns)
        excess_returns = returns_np - self.target_return_daily
        
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std(ddof=1)
        
        mean = excess_returns.mean()
        
        current_sortino = (mean / downside_std) * np.sqrt(self.annualization_factor) if downside_std > 0 else 0.0
        
        reward = current_sortino - self._prev_sortino
        self._prev_sortino = current_sortino
        return reward


class CalmarReward(RewardCalculator):
    """Reward derived from incremental Calmar ratio."""

    def __init__(self, annualization_factor: float = 252):
        super().__init__()
        self.annualization_factor = annualization_factor
        self._prev_calmar = 0.0

    def reset(self, initial_equity: float) -> None:
        super().reset(initial_equity)
        self._prev_calmar = 0.0

    def _compute_reward(self, equity_curve: List[float], returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0

        equity_curve_np = np.array(equity_curve)
        drawdown = self._max_drawdown(equity_curve_np)
        
        total_return = (equity_curve_np[-1] / equity_curve_np[0]) - 1 if equity_curve_np[0] != 0 else 0.0
        years = max(len(returns) / self.annualization_factor, 1e-9)
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        current_calmar = annual_return / drawdown if drawdown > 0 else 0.0
        
        reward = current_calmar - self._prev_calmar
        self._prev_calmar = current_calmar
        return reward

    @staticmethod
    def _max_drawdown(curve: np.ndarray) -> float:
        """Calcula el Max Drawdown (como un valor positivo)."""
        running_max = np.maximum.accumulate(curve)
        # Asegurarse de que running_max no sea cero para evitar división por cero
        running_max[running_max == 0] = 1 
        drawdowns = (running_max - curve) / running_max
        max_dd = np.nanmax(drawdowns)
        return float(max_dd) if np.isfinite(max_dd) else 0.0


REWARD_REGISTRY = {
    "pnl": PnLReward,
    "sharpe": SharpeReward,
    "sortino": SortinoReward,
    "calmar": CalmarReward,
}