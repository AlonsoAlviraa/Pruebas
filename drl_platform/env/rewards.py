#!/usr/bin/env python3
"""Calculadores de recompensa para el entorno de trading."""
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

        raw_reward = self._compute_reward(curve, ret_series)
        safe_array = np.nan_to_num(np.asarray(raw_reward, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        if safe_array.size == 0:
            return 0.0
        safe_value = float(safe_array.reshape(-1)[0])
        if not np.isfinite(safe_value):
            safe_value = 0.0
        return safe_value

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

        returns_arr = np.asarray(returns)
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr)

        if std_ret == 0:
            # Si no hay volatilidad, sharpe es 0 (o infinito si mean_ret > 0)
            current_sharpe = 0.0
        else:
            daily_sharpe = (mean_ret - self.risk_free_rate) / std_ret
            current_sharpe = daily_sharpe * (self.annualization_factor ** 0.5)

        reward = current_sharpe - self._prev_sharpe
        self._prev_sharpe = current_sharpe
        return reward


class SortinoReward(RewardCalculator):
    """Reward based on incremental Sortino ratio."""

    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: float = 252):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self._prev_sortino = 0.0

    def reset(self, initial_equity: float) -> None:
        super().reset(initial_equity)
        self._prev_sortino = 0.0

    def _compute_reward(self, equity_curve: List[float], returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0

        returns_arr = np.asarray(returns)
        mean_ret = np.mean(returns_arr)

        # Calcular downside deviation
        downside_returns = returns_arr[returns_arr < self.risk_free_rate]
        if downside_returns.size == 0:
            downside_std = 0.0
        else:
            downside_std = np.std(downside_returns)

        if downside_std == 0:
            current_sortino = 0.0
        else:
            daily_sortino = (mean_ret - self.risk_free_rate) / downside_std
            current_sortino = daily_sortino * (self.annualization_factor ** 0.5)

        reward = current_sortino - self._prev_sortino
        self._prev_sortino = current_sortino
        return reward


class CalmarReward(RewardCalculator):
    """Reward based on incremental Calmar ratio."""

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

        # Calcular annualized return
        returns_arr = np.asarray(returns)
        mean_ret = np.mean(returns_arr)
        annualized_return = (1 + mean_ret) ** self.annualization_factor - 1

        # Calcular max drawdown
        equity_arr = np.asarray(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        max_drawdown = np.min(drawdown)

        if max_drawdown == 0:
            current_calmar = 0.0
        else:
            current_calmar = annualized_return / abs(max_drawdown)

        reward = current_calmar - self._prev_calmar
        self._prev_calmar = current_calmar
        return reward


# Registro de calculadores de recompensa
REWARD_REGISTRY = {
    "pnl": PnLReward,
    "sharpe": SharpeReward,
    "sortino": SortinoReward,
    "calmar": CalmarReward,
}