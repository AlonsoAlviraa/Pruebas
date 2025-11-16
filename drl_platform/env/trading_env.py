"""Advanced multi-asset trading environment for portfolio management."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from .rewards import REWARD_REGISTRY, RewardCalculator

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration object for :class:`TradingEnvironment`."""

    # --- Opciones de Carga de Datos ---
    data: Optional[pd.DataFrame | Dict[str, pd.DataFrame]] = None
    payload_key: Optional[str] = None

    # --- Opciones de Simulación ---
    initial_cash: float = 1_000_000
    commission_rate: float = 0.001
    slippage: float = 0.0005
    use_continuous_action: bool = False
    allow_short: bool = False
    reward: str = "pnl"
    max_episode_steps: Optional[int] = None


class TradingEnvironment(gym.Env):
    """Multi-asset trading simulator with configurable reward functions."""

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

        self.tickers: list[str] = []
        self.dates: pd.Index = pd.Index([])
        self.prices = np.zeros((0, 0), dtype=np.float32)
        self.features = np.zeros((0, 0, 0), dtype=np.float32)
        self.feature_names: list[str] = []
        self.num_tickers: int = 0
        self.num_steps: int = 0

        self.positions = np.zeros(0, dtype=np.float32)
        self.position_values = np.zeros(0, dtype=np.float32)
        self.current_weights = np.zeros(0, dtype=np.float32)
        self.cash = float(self.config.initial_cash)
        self.portfolio_value = self.cash
        self.reward_calculator = self._build_reward_calculator(self.config)

        self._prepare_from_payload(force=True)
        self._build_spaces()
        self._reset_portfolio_state()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno al estado inicial."""

        super().reset(seed=seed)
        self._prepare_from_payload(force=False)
        self._reset_portfolio_state()
        return self._get_observation(), self._get_info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= self.num_steps:
            raise RuntimeError("step() called on a terminated episode")

        action_values = self._normalise_action(action)
        target_weights = self._weights_from_action(action_values)

        prices = self.prices[self.current_step]
        target_values = target_weights * self.portfolio_value
        with np.errstate(divide="ignore", invalid="ignore"):
            target_positions = np.divide(
                target_values,
                prices,
                out=np.zeros_like(prices),
                where=prices != 0,
            )

        delta_positions = target_positions - self.positions
        executed_prices = prices * (1 + self.config.slippage * np.sign(delta_positions))
        trade_values = executed_prices * delta_positions
        commissions = np.abs(trade_values) * self.config.commission_rate

        self.cash -= trade_values.sum() + commissions.sum()
        self.positions = target_positions
        self.position_values = self.positions * prices
        previous_value = self.portfolio_value
        self.portfolio_value = self.cash + self.position_values.sum()

        total_value = max(self.portfolio_value, 1e-12)
        self.current_weights = np.divide(
            self.position_values,
            total_value,
            out=np.zeros_like(self.position_values),
            where=total_value > 0,
        )

        for idx, delta in enumerate(delta_positions):
            if delta == 0:
                continue
            self._record_trade(
                idx,
                float(delta),
                float(executed_prices[idx]),
                float(commissions[idx]),
            )

        reward = float(self.reward_calculator.update(self.portfolio_value))
        info = self._get_info()
        info.update(
            {
                "prices": prices.astype(float).tolist(),
                "executed_prices": executed_prices.astype(float).tolist(),
                "commissions": commissions.astype(float).tolist(),
                "reward": reward,
                "prev_portfolio_value": float(previous_value),
            }
        )

        next_step = self.current_step + 1
        terminated = next_step >= self.num_steps
        truncated = False
        self._episode_steps += 1

        if self._max_episode_steps is not None and self._max_episode_steps > 0:
            if self._episode_steps >= self._max_episode_steps:
                truncated = not terminated

        self.current_step = min(next_step, self.num_steps - 1)
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

    def _build_spaces(self) -> None:
        if self.features.size == 0 or self.num_tickers == 0:
            raise ValueError("El entorno requiere datos para inicializar los espacios")

        obs_features = self.features.shape[-1] + 2  # pesos y ratio de efectivo
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_tickers * obs_features,),
            dtype=np.float32,
        )

        if self.config.use_continuous_action:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_tickers,), dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                np.full(self.num_tickers, 3, dtype=np.int64)
            )

    def _normalise_action(self, action: Any) -> np.ndarray:
        if self.config.use_continuous_action:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            arr = np.asarray(action, dtype=np.int32).reshape(-1) - 1.0

        if arr.size == 0:
            arr = np.zeros(self.num_tickers, dtype=np.float32)
        elif arr.size == 1 and self.num_tickers > 1:
            arr = np.repeat(arr, self.num_tickers)
        elif arr.size != self.num_tickers:
            arr = np.resize(arr, self.num_tickers)

        return np.clip(arr.astype(np.float32), -1.0, 1.0)

    def _weights_from_action(self, action_values: np.ndarray) -> np.ndarray:
        if self.config.allow_short:
            clipped = np.clip(action_values, -1.0, 1.0)
            total = float(np.sum(np.abs(clipped)))
            if total <= 1e-6:
                return np.full(self.num_tickers, 1.0 / self.num_tickers, dtype=np.float32)
            return clipped / total

        clipped = np.clip(action_values, 0.0, 1.0)
        total = float(np.sum(clipped))
        if total <= 1e-6:
            return np.full(self.num_tickers, 1.0 / self.num_tickers, dtype=np.float32)
        return clipped / total

    def _get_observation(self) -> np.ndarray:
        feature_matrix = self.features[self.current_step]
        weights = self.current_weights.reshape(-1, 1)
        cash_ratio = (
            self.cash / max(self.portfolio_value, 1e-12) if self.portfolio_value > 0 else 1.0
        )
        cash_column = np.full((self.num_tickers, 1), cash_ratio, dtype=np.float32)
        obs = np.concatenate([feature_matrix, weights, cash_column], axis=1)
        return obs.astype(np.float32, copy=False).flatten()

    def _get_info(self) -> Dict[str, Any]:
        date_value: Any
        if len(self.dates) == 0:
            date_value = None
        else:
            idx = min(self.current_step, len(self.dates) - 1)
            date_value = self.dates[idx]

        return {
            "step": int(self.current_step),
            "date": None if date_value is None else str(date_value),
            "cash": float(self.cash),
            "portfolio_value": float(self.portfolio_value),
            "weights": self.current_weights.astype(float).tolist(),
            "positions": self.positions.astype(float).tolist(),
            "position_values": self.position_values.astype(float).tolist(),
            "tickers": list(self.tickers),
        }

    def _record_trade(self, ticker_idx: int, delta: float, executed_price: float, commission: float) -> None:
        self.trades.append(
            {
                "step": int(self.current_step),
                "ticker": self.tickers[ticker_idx],
                "delta": float(delta),
                "executed_price": float(executed_price),
                "commission": float(commission),
                "cash": float(self.cash),
                "position": float(self.positions[ticker_idx]),
                "portfolio_value": float(self.portfolio_value),
            }
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> None:  # pragma: no cover - debug helper
        allocations = ", ".join(
            f"{ticker}:{weight:.2f}" for ticker, weight in zip(self.tickers, self.current_weights)
        )
        print(
            f"Step={self.current_step} portfolio={self.portfolio_value:.2f} "
            f"cash={self.cash:.2f} weights=[{allocations}]"
        )

    def close(self) -> None:  # pragma: no cover - compatibility hook
        pass

    # ------------------------------------------------------------------
    # Shared payload helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_dataset_input(data: Any) -> Dict[str, pd.DataFrame]:
        if isinstance(data, dict):
            cleaned: Dict[str, pd.DataFrame] = {}
            for key, df in data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                cleaned[str(key)] = df.copy()
            if not cleaned:
                raise ValueError("El dataset de cartera no contiene DataFrames válidos")
            return cleaned

        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("El DataFrame proporcionado para el entorno está vacío")
            return {"asset": data.copy()}

        raise TypeError(
            "El dataset del entorno debe ser un DataFrame o un diccionario de DataFrames"
        )

    @classmethod
    def set_shared_payload(
        cls, key: str, *, data: pd.DataFrame | Dict[str, pd.DataFrame], overrides: Optional[Dict[str, Any]] = None
    ) -> None:
        if not key:
            raise ValueError("Se requiere una clave no vacía para registrar el payload del entorno")

        dataset = cls._coerce_dataset_input(data)
        prev = cls._SHARED_PAYLOADS.get(key, {})
        version = int(prev.get("version", 0)) + 1
        cls._SHARED_PAYLOADS[key] = {
            "version": version,
            "data": dataset,
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

        source_data = payload.get("data") if payload else self.config.data
        if source_data is None:
            raise ValueError("Environment requires a non-empty dataset of features")

        dataset = self._coerce_dataset_input(source_data)
        tickers, dates, prices, features, feature_names = self._build_panel(dataset)
        self.tickers = tickers
        self.num_tickers = len(tickers)
        self.dates = dates
        self.num_steps = len(dates)
        self.prices = prices
        self.features = features
        self.feature_names = feature_names

        overrides = payload.get("overrides", {}) if payload else {}
        for field, value in overrides.items():
            if hasattr(self.config, field) and field not in {"data", "payload_key"}:
                setattr(self.config, field, value)

        self._payload_version = int(payload_version or 0)
        self.reward_calculator = self._build_reward_calculator(self.config)
        self._max_episode_steps = self._resolve_episode_limit(self.num_steps)

    def _build_panel(
        self, dataset: Dict[str, pd.DataFrame]
    ) -> tuple[list[str], pd.DatetimeIndex, np.ndarray, np.ndarray, list[str]]:
        prepared: Dict[str, pd.DataFrame] = {}
        for ticker, df in dataset.items():
            if df.empty:
                continue
            if "date" not in df.columns:
                raise KeyError(f"La columna 'date' es requerida para el ticker {ticker}")
            if "close" not in df.columns:
                raise KeyError(f"La columna 'close' es requerida para el ticker {ticker}")

            working = df.copy()
            working["date"] = pd.to_datetime(working["date"])
            working = working.dropna(subset=["close"]).sort_values("date")
            working = working.drop_duplicates(subset=["date"], keep="last")
            prepared[ticker] = working

        if not prepared:
            raise ValueError("El dataset de cartera no contiene filas utilizables")

        common_index: Optional[pd.DatetimeIndex] = None
        for frame in prepared.values():
            idx = pd.DatetimeIndex(frame["date"])
            common_index = idx if common_index is None else common_index.intersection(idx)

        if common_index is None or common_index.empty:
            raise ValueError("Los tickers no comparten fechas en común para entrenar la cartera")

        common_index = pd.DatetimeIndex(sorted(common_index))
        tickers = sorted(prepared)

        feature_columns = sorted(
            {
                col
                for frame in prepared.values()
                for col in frame.columns
                if col not in {"date", "close"}
            }
        )

        feature_names = ["return_1d"] + [col for col in feature_columns if col != "return_1d"]
        num_steps = len(common_index)
        num_tickers = len(tickers)
        num_features = len(feature_names)

        price_matrix = np.zeros((num_steps, num_tickers), dtype=np.float32)
        feature_tensor = np.zeros((num_steps, num_tickers, num_features), dtype=np.float32)

        for idx, ticker in enumerate(tickers):
            frame = prepared[ticker].set_index("date").reindex(common_index)
            close_series = pd.to_numeric(frame["close"], errors="coerce").ffill()
            close_series = close_series.bfill()
            if close_series.isna().any():
                raise ValueError(f"No se pudo alinear precios para el ticker {ticker}")

            price_matrix[:, idx] = close_series.to_numpy(dtype=np.float32)
            returns = close_series.pct_change().fillna(0.0).to_numpy(dtype=np.float32)

            col_arrays: list[np.ndarray] = []
            for name in feature_names:
                if name == "return_1d":
                    col_arrays.append(returns)
                    continue
                series = frame.get(name)
                if series is None:
                    arr = np.zeros(num_steps, dtype=np.float32)
                else:
                    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                col_arrays.append(arr)

            feature_tensor[:, idx, :] = np.stack(col_arrays, axis=1)

        return tickers, common_index, price_matrix, feature_tensor, feature_names

    def _resolve_episode_limit(self, available_steps: int) -> Optional[int]:
        limit = getattr(self.config, "max_episode_steps", None)
        if limit is None:
            return None
        try:
            numeric = int(limit)
        except (TypeError, ValueError):
            logger.warning(
                "max_episode_steps=%r inválido; ignorando límite de episodio",
                self.config.max_episode_steps,
            )
            return None
        if numeric <= 0:
            return None
        return min(numeric, available_steps)

    def _reset_portfolio_state(self) -> None:
        self.current_step = 0
        self._episode_steps = 0
        self.cash = float(self.config.initial_cash)
        self.portfolio_value = self.cash
        self.positions = np.zeros(self.num_tickers, dtype=np.float32)
        self.position_values = np.zeros(self.num_tickers, dtype=np.float32)
        self.current_weights = np.zeros(self.num_tickers, dtype=np.float32)
        self.reward_calculator.reset(self.portfolio_value)
        self.trades.clear()

    def reload_shared_payload(self) -> None:
        """Permite a los workers actualizar el payload manualmente."""

        self._prepare_from_payload(force=True)
        self._build_spaces()
        self._reset_portfolio_state()