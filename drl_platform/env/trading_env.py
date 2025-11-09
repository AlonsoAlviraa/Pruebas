#!/usr/bin/env python3
"""Advanced trading environment supporting shorts, slippage and commissions."""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from .rewards import REWARD_REGISTRY, RewardCalculator

logger = logging.getLogger(__name__)

# Almacén estático (nivel de clase) para payloads de datos compartidos.
# Esto permite al orquestador publicar un DataFrame grande una vez, y a los
# workers (incluido el local) leerlo sin pasarlo por la config de RLlib,
# lo que evita que `reset_config` falle al comparar DataFrames.
_SHARED_PAYLOADS: ClassVar[Dict[str, Dict[str, Any]]] = {}


@dataclass
class EnvironmentConfig:
    """Configuration object for :class:`TradingEnvironment`."""

    # --- Opciones de Carga de Datos ---
    # Opción 1: Pasar el DataFrame directamente (para uso simple)
    data: Optional[pd.DataFrame] = None
    # Opción 2: Cargar desde un payload compartido (para reutilización de workers)
    payload_key: Optional[str] = None

    # --- Opciones de Simulación ---
    initial_cash: float = 1_000_000
    commission_rate: float = 0.001
    slippage: float = 0.0005
    use_continuous_action: bool = False
    reward: str = "pnl"
    max_episode_steps: Optional[int] = None


class TradingEnvironment(gym.Env):
    """Single asset trading simulator with configurable reward functions."""

    metadata = {"render_modes": ["human"]}

    _SHARED_PAYLOADS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    def __init__(self, config: EnvironmentConfig):
        super().__init__()

        self.config = config
        self.trades: list[Dict[str, Any]] = []
        self._shared_payload_key = config.payload_key
        self._payload_version = -1
        self._max_episode_steps: Optional[int] = None
        self._episode_steps: int = 0

        self._prepare_from_payload(force=True)

        if config.use_continuous_action:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(3)

        feature_dim = self.features.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim + 3,),
            dtype=np.float32,
        )

        self._reset_portfolio_state()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno al estado inicial."""
        super().reset(seed=seed)

        self._prepare_from_payload(force=False)
        self.current_step = 0
        self._episode_steps = 0
        self.cash = float(self.config.initial_cash)
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = self.cash
        self.reward_calculator.reset(self.portfolio_value)
        self.trades.clear()

        return self._get_observation(), self._get_info()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= len(self.prices):
            raise RuntimeError("step() called on a terminated episode")

        if self.config.use_continuous_action:
            target_ratio = float(np.clip(np.asarray(action)[0], -1.0, 1.0))
        else:
            target_ratio = float(int(action) - 1)

        price = float(self.prices[self.current_step])

        target_notional = target_ratio * self.portfolio_value
        target_position = target_notional / price if price > 0 else 0.0
        delta = target_position - self.position

        executed_price = price
        if delta != 0.0:
            executed_price = price * (1 + self.config.slippage * np.sign(delta))

        trade_cost_notional = executed_price * delta
        commission = abs(trade_cost_notional) * self.config.commission_rate

        self.cash -= trade_cost_notional + commission
        self.position = target_position
        self.position_value = self.position * price
        self.portfolio_value = self.cash + self.position_value

        if delta != 0.0:
            self._record_trade(delta, executed_price, commission)

        reward = float(self.reward_calculator.update(self.portfolio_value))
        info = self._get_info()
        info.update(
            {
                "price": price,
                "executed_price": executed_price,
                "commission": commission,
                "trade_delta": delta,
                "reward": reward,
            }
        )

        next_step = self.current_step + 1
        terminated = next_step >= len(self.prices)
        truncated = False
        self._episode_steps += 1

        max_steps = self._max_episode_steps
        if max_steps is not None and max_steps > 0:
            if self._episode_steps >= max_steps:
                truncated = not terminated
                if self._episode_steps > max_steps and not terminated:
                    # Garantizar consistencia si el límite se excede por rounding
                    terminated = False

        self.current_step = min(next_step, len(self.prices) - 1)

        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_reward_calculator(self, config: EnvironmentConfig) -> RewardCalculator:
        reward_key = (config.reward or "pnl").lower()
        try:
            calculator_cls = REWARD_REGISTRY[reward_key]
        except KeyError as exc:  # pragma: no cover - validation
            valid = ", ".join(sorted(REWARD_REGISTRY))
            raise KeyError(f"Recompensa '{config.reward}' no soportada. Opciones: {valid}") from exc

        calculator = calculator_cls()
        calculator.reset(float(config.initial_cash))
        return calculator

    def _get_observation(self) -> np.ndarray:
        feature_vec = self.features[self.current_step]
        portfolio_state = np.array(
            [self.cash, self.position, self.portfolio_value], dtype=np.float32
        )
        return np.concatenate([feature_vec, portfolio_state], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "cash": float(self.cash),
            "position": float(self.position),
            "position_value": float(self.position_value),
            "portfolio_value": float(self.portfolio_value),
        }

    def _record_trade(self, delta: float, executed_price: float, commission: float) -> None:
        self.trades.append(
            {
                "step": self.current_step,
                "delta": float(delta),
                "executed_price": float(executed_price),
                "commission": float(commission),
                "cash": float(self.cash),
                "position": float(self.position),
                "portfolio_value": float(self.portfolio_value),
            }
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> None:  # pragma: no cover - debug helper
        print(
            f"Step={self.current_step} price={self.prices[self.current_step]:.2f} "
            f"cash={self.cash:.2f} position={self.position:.4f} "
            f"portfolio={self.portfolio_value:.2f}"
        )

    def close(self) -> None:  # pragma: no cover - compatibility hook
        pass

    # ------------------------------------------------------------------
    # Shared payload helpers
    # ------------------------------------------------------------------
    @classmethod
    def set_shared_payload(
        cls, key: str, *, data: pd.DataFrame, overrides: Optional[Dict[str, Any]] = None
    ) -> None:
        if not key:
            raise ValueError("Se requiere una clave no vacía para registrar el payload del entorno")
        if data is None or data.empty:
            raise ValueError("El payload compartido requiere un DataFrame con información")

        prev = cls._SHARED_PAYLOADS.get(key, {})
        version = int(prev.get("version", 0)) + 1
        cls._SHARED_PAYLOADS[key] = {
            "version": version,
            "data": data,
            "overrides": dict(overrides or {}),
        }

    @classmethod
    def _get_shared_payload(cls, key: Optional[str]) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        return cls._SHARED_PAYLOADS.get(key)

    def _prepare_from_payload(self, *, force: bool) -> None:
        payload = self._get_shared_payload(self._shared_payload_key)
        payload_version = payload.get("version") if payload else None
        if not force and payload_version == self._payload_version:
            return

        if payload is not None:
            data = payload["data"]
            overrides = payload.get("overrides", {})
            for field, value in overrides.items():
                if hasattr(self.config, field) and field not in {"data", "payload_key"}:
                    setattr(self.config, field, value)
            self.config.data = data
            self._payload_version = int(payload_version)
        elif force:
            self._payload_version = 0
        else:
            return

        if self.config.data is None or self.config.data.empty:
            raise ValueError("Environment requires a non-empty DataFrame of features")
        if "close" not in self.config.data.columns:
            raise KeyError("La columna 'close' es requerida en los datos para precios.")

        # Cache common series
        data = self.config.data
        self.dates = data.get("date", pd.RangeIndex(start=0, stop=len(data)))
        self.prices = data["close"].astype(float).to_numpy(copy=True)

        feature_df = data.drop(columns=["close", "date"], errors="ignore")
        for column in feature_df.select_dtypes(include=["datetime", "datetimetz"]).columns:
            feature_df[column] = feature_df[column].astype("int64")
        for column in feature_df.select_dtypes(include=["object"]).columns:
            feature_df[column] = feature_df[column].astype("category").cat.codes

        self.features = feature_df.astype(np.float32).to_numpy(copy=True)
        if len(self.features) != len(self.prices):
            raise ValueError("Features and prices must contain the same number of rows")

        max_steps = getattr(self.config, "max_episode_steps", None)
        if max_steps is not None:
            try:
                max_steps = int(max_steps)
            except (TypeError, ValueError):
                logger.warning(
                    "max_episode_steps=%r inválido; ignorando límite de episodio",
                    self.config.max_episode_steps,
                )
                max_steps = None
            else:
                if max_steps <= 0:
                    max_steps = None
                elif max_steps > len(self.prices):
                    max_steps = len(self.prices)
        self._max_episode_steps = max_steps

        self.reward_calculator = self._build_reward_calculator(self.config)

    def _reset_portfolio_state(self) -> None:
        self.current_step = 0
        self.cash = float(self.config.initial_cash)
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = float(self.config.initial_cash)
        self.trades.clear()
        self._episode_steps = 0

    def reload_shared_payload(self) -> None:
        """Permite a los workers actualizar el payload manualmente."""

        self._prepare_from_payload(force=True)
        self._reset_portfolio_state()