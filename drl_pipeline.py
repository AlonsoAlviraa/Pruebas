#!/usr/bin/env python3
"""Pipeline modular para entrenar agentes de *Deep Reinforcement Learning*.

Este módulo implementa las fases descritas en el plan propuesto:

* Fase 1 (Baseline): estado OHLCV + cartera, recompensa P&L y validación 80/20.
* Fase 2, Iteración 1: se añaden indicadores técnicos al estado.
* Fase 2, Iteración 2: se sustituye la recompensa por un Sharpe diferencial.
* Fase 2, Iteración 3: se cambia a validación Walk-Forward.
* Fase 3: se incorpora un ``feature`` externo (ej. sentimiento de un LLM).

El objetivo es disponer de una *plumbing* reproducible sobre la que iterar,
dejando el entrenamiento real a elección del usuario (timesteps, algoritmo,
etc.).  Se asume la existencia de ``stable-baselines3`` y ``gymnasium``; en
caso contrario se reportan mensajes informativos.
"""

from __future__ import annotations

import argparse
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np
import pandas as pd

try:  # gymnasium >= 0.27
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - fallback opcional
    import gym  # type: ignore

try:
    from stable_baselines3 import A2C, PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
except ModuleNotFoundError as exc:  # pragma: no cover - mensaje amigable
    raise ModuleNotFoundError(
        "stable-baselines3 es obligatorio para ejecutar este pipeline. "
        "Instálalo con `pip install stable-baselines3`."
    ) from exc


# ---------------------------------------------------------------------------
# Configuración de alto nivel
# ---------------------------------------------------------------------------


class PipelinePhase(enum.Enum):
    """Fases sugeridas en el plan de implementación."""

    BASELINE = "baseline"
    FEATURES = "features"
    REWARD = "reward"
    VALIDATION = "validation"
    LLM = "llm"


@dataclass
class PipelineConfig:
    """Parámetros generales del pipeline."""

    symbol: str
    phase: PipelinePhase
    total_timesteps: int = 10_000
    algorithm: str = "ppo"
    data_csv: Optional[Path] = None
    sentiment_csv: Optional[Path] = None
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    train_years: int = 3
    test_years: int = 1


@dataclass
class FeatureConfig:
    include_indicators: bool = False
    sentiment: Optional[pd.Series] = None


class RewardCalculator:
    """Interfaz para calcular recompensas acumuladas."""

    def reset(self, initial_value: float) -> None:
        raise NotImplementedError

    def step(self, portfolio_value: float) -> float:
        raise NotImplementedError


class PnLReward(RewardCalculator):
    """Recompensa basada en el cambio absoluto del valor de la cartera."""

    def __init__(self) -> None:
        self._prev_value: Optional[float] = None

    def reset(self, initial_value: float) -> None:
        self._prev_value = initial_value

    def step(self, portfolio_value: float) -> float:
        if self._prev_value is None:
            self._prev_value = portfolio_value
            return 0.0
        reward = portfolio_value - self._prev_value
        self._prev_value = portfolio_value
        return float(reward)


class SharpeReward(RewardCalculator):
    """Recompensa diferencial del Sharpe Ratio en una ventana móvil."""

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self._prev_value: Optional[float] = None
        self._returns: list[float] = []
        self._prev_sharpe: float = 0.0

    def reset(self, initial_value: float) -> None:
        self._prev_value = initial_value
        self._returns = []
        self._prev_sharpe = 0.0

    def step(self, portfolio_value: float) -> float:
        if self._prev_value is None:
            self.reset(portfolio_value)
            return 0.0
        prev = self._prev_value
        self._prev_value = portfolio_value
        if prev <= 0:
            return 0.0
        daily_return = (portfolio_value - prev) / prev
        self._returns.append(float(daily_return))
        if len(self._returns) < self.window:
            return 0.0
        window_returns = self._returns[-self.window :]
        std = float(np.std(window_returns, ddof=1))
        if std == 0:
            sharpe = 0.0
        else:
            sharpe = float(np.sqrt(252) * np.mean(window_returns) / std)
        reward = sharpe - self._prev_sharpe
        self._prev_sharpe = sharpe
        return reward


# ---------------------------------------------------------------------------
# Ingeniería de *features*
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mean + num_std * std
    lower = mean - num_std * std
    return upper, lower


class FeatureEngineer:
    """Crea el estado del agente."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        features = data.copy()
        if "volume" not in features.columns:
            features["volume"] = 0.0
        features["return_1d"] = features["close"].pct_change().fillna(0.0)

        if self.config.include_indicators:
            features["rsi_14"] = _rsi(features["close"], period=14)
            ema_fast = _ema(features["close"], span=12)
            ema_slow = _ema(features["close"], span=26)
            features["macd"] = ema_fast - ema_slow
            features["macd_signal"] = _ema(features["macd"], span=9)
            bb_upper, bb_lower = _bollinger(features["close"], window=20)
            features["bb_upper"] = bb_upper
            features["bb_lower"] = bb_lower

        if self.config.sentiment is not None:
            sentiment = self.config.sentiment.reindex(features.index).fillna(0.0)
            features["sentiment_score"] = sentiment

        features = features.dropna()
        return features


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def load_market_data(
    symbol: str,
    csv_path: Optional[Path],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Obtiene datos OHLCV ya sea desde CSV o descargando con yfinance."""

    if csv_path is not None:
        raw = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    else:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:  # pragma: no cover - mensaje amigable
            raise ModuleNotFoundError(
                "Se necesita yfinance para descargar datos si no se provee CSV. "
                "Instala la librería o pasa --data-csv."
            ) from exc

        ticker = yf.Ticker(symbol)
        raw = ticker.history(start=start, end=end, interval="1d")

    if raw.empty:
        raise ValueError("No se obtuvieron datos de mercado")

    columns = {col.lower(): col for col in raw.columns}
    required = {"open", "high", "low", "close"}
    if not required.issubset(columns):
        raise ValueError("El dataset debe incluir columnas OHLC")

    df = pd.DataFrame(index=pd.to_datetime(raw.index, utc=True))
    for name in ["open", "high", "low", "close", "volume"]:
        col = columns.get(name)
        if col is None:
            continue
        df[name] = pd.to_numeric(raw[col], errors="coerce")

    df = df.sort_index().dropna(subset=["close"])

    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]

    return df


def load_sentiment_feature(path: Path) -> pd.Series:
    """Lee un CSV con columnas ``date`` y ``sentiment``."""

    df = pd.read_csv(path)
    if "date" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("El CSV de sentimiento debe contener columnas 'date' y 'sentiment'")
    series = (
        pd.Series(df["sentiment"].astype(float).values, index=pd.to_datetime(df["date"], utc=True))
        .sort_index()
    )
    return series


# ---------------------------------------------------------------------------
# Entorno de trading
# ---------------------------------------------------------------------------


class TradingEnv(gym.Env):
    """Entorno discreto Long/Flat con compra total a cierre."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        reward_calc: RewardCalculator,
        initial_cash: float = 1.0,
        trading_cost: float = 0.0005,
    ) -> None:
        super().__init__()
        self.market_data = market_data
        self.features = features
        if not market_data.index.equals(features.index):
            raise ValueError("market_data y features deben compartir el índice")
        self.reward_calc = reward_calc
        self.initial_cash = float(initial_cash)
        self.trading_cost = float(trading_cost)

        sample_obs = self._build_observation(features.iloc[0], self.initial_cash, 0.0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)  # 0=Hold, 1=Long, 2=Flat

        self._current_step: int = 0
        self._cash: float = self.initial_cash
        self._position: float = 0.0
        self._portfolio_value: float = self.initial_cash
        self._history: list[float] = []

    @property
    def portfolio_history(self) -> pd.Series:
        return pd.Series(self._history, index=self.features.index[: len(self._history)])

    def _build_observation(
        self, row: pd.Series, cash: float, position: float
    ) -> np.ndarray:
        obs = np.concatenate([row.to_numpy(dtype=np.float32), np.array([cash, position], dtype=np.float32)])
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._current_step = 0
        self._cash = self.initial_cash
        self._position = 0.0
        self._portfolio_value = self.initial_cash
        self._history = [self._portfolio_value]
        self.reward_calc.reset(self._portfolio_value)
        first_obs = self._build_observation(self.features.iloc[self._current_step], self._cash, self._position)
        return first_obs, {}

    def _update_portfolio_value(self, price: float) -> float:
        self._portfolio_value = self._cash + self._position * price
        self._history.append(self._portfolio_value)
        return self._portfolio_value

    def step(self, action: int):
        price = float(self.market_data["close"].iloc[self._current_step])

        if action == 1:  # Long total
            total_value = self._cash + self._position * price
            delta_position = total_value / price - self._position
            cost = abs(delta_position) * price * self.trading_cost
            total_value -= cost
            self._position = total_value / price
            self._cash = 0.0
        elif action == 2:  # Flat (cerrar posición)
            total_value = self._cash + self._position * price
            cost = abs(self._position) * price * self.trading_cost
            total_value -= cost
            self._cash = total_value
            self._position = 0.0
        else:  # Hold
            pass

        self._current_step += 1
        terminated = self._current_step >= len(self.market_data)
        truncated = False

        price = float(self.market_data["close"].iloc[min(self._current_step - 1, len(self.market_data) - 1)])
        portfolio_value = self._update_portfolio_value(price)
        reward = self.reward_calc.step(portfolio_value)

        if terminated:
            obs = self._build_observation(self.features.iloc[-1], self._cash, self._position)
        else:
            obs = self._build_observation(self.features.iloc[self._current_step], self._cash, self._position)

        info = {
            "portfolio_value": portfolio_value,
            "cash": self._cash,
            "position": self._position,
        }

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


@dataclass
class PortfolioMetrics:
    final_value: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    history: pd.Series


def compute_metrics(history: pd.Series) -> PortfolioMetrics:
    if history.empty:
        raise ValueError("La serie de histórico está vacía")
    initial = float(history.iloc[0])
    final = float(history.iloc[-1])
    total_return = final / initial - 1.0
    returns = history.pct_change().dropna()
    if not returns.empty:
        sharpe = float(np.sqrt(252) * returns.mean() / (returns.std(ddof=1) + 1e-9))
    else:
        sharpe = 0.0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    days = max(len(history) - 1, 1)
    annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else total_return
    return PortfolioMetrics(final, total_return, annual_return, max_drawdown, sharpe, history)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------


def _select_algorithm(name: str):
    name = name.lower()
    if name == "ppo":
        return PPO
    if name == "a2c":
        return A2C
    if name == "sac":
        return SAC
    raise ValueError(f"Algoritmo no soportado: {name}")


def train_model(
    algorithm: str,
    train_data: pd.DataFrame,
    train_features: pd.DataFrame,
    reward_factory: Callable[[], RewardCalculator],
    total_timesteps: int,
):
    Algo = _select_algorithm(algorithm)

    def _make_env() -> TradingEnv:
        return TradingEnv(train_data, train_features, reward_factory())

    env = DummyVecEnv([_make_env])
    model = Algo("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_model(
    model,
    test_data: pd.DataFrame,
    test_features: pd.DataFrame,
    reward_factory: Callable[[], RewardCalculator],
) -> PortfolioMetrics:
    env = TradingEnv(test_data, test_features, reward_factory())
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        if terminated:
            break
    history = env.portfolio_history
    return compute_metrics(history)


def buy_and_hold_metrics(test_data: pd.DataFrame) -> PortfolioMetrics:
    prices = test_data["close"].astype(float)
    initial = prices.iloc[0]
    final = prices.iloc[-1]
    history = prices / initial
    history.iloc[0] = 1.0
    history = history.astype(float)
    return compute_metrics(history)


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------


def chronological_split(df: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(df) * ratio)
    if cutoff <= 0 or cutoff >= len(df):
        raise ValueError("El split cronológico no es válido")
    return df.iloc[:cutoff], df.iloc[cutoff:]


def walk_forward_splits(
    df: pd.DataFrame,
    train_years: int,
    test_years: int,
) -> Generator[tuple[pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
    if train_years <= 0 or test_years <= 0:
        raise ValueError("Los tamaños de ventana deben ser positivos")
    index = df.index
    start = index.min()
    end = index.max()
    anchor = start
    while True:
        train_end = anchor + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        train_mask = (index >= anchor) & (index < train_end)
        test_mask = (index >= train_end) & (index < test_end)
        if not train_mask.any() or not test_mask.any():
            break
        yield index[train_mask], index[test_mask]
        anchor = anchor + pd.DateOffset(years=1)
        if anchor + pd.DateOffset(years=train_years + test_years) > end + pd.DateOffset(days=1):
            break


def run_simple_pipeline(
    config: PipelineConfig,
    data: pd.DataFrame,
    features: pd.DataFrame,
    reward_factory: Callable[[], RewardCalculator],
) -> dict:
    train_idx, test_idx = chronological_split(features, ratio=0.8)
    train_data = data.loc[train_idx.index]
    test_data = data.loc[test_idx.index]

    model = train_model(
        config.algorithm,
        train_data,
        train_idx,
        reward_factory,
        config.total_timesteps,
    )
    agent_metrics = evaluate_model(model, test_data, test_idx, reward_factory)
    benchmark = buy_and_hold_metrics(test_data)

    return {
        "agent": agent_metrics,
        "benchmark": benchmark,
        "folds": 1,
    }


def run_walk_forward_pipeline(
    config: PipelineConfig,
    data: pd.DataFrame,
    features: pd.DataFrame,
    reward_factory: Callable[[], RewardCalculator],
) -> dict:
    splits = list(walk_forward_splits(features, config.train_years, config.test_years))
    if not splits:
        raise ValueError("No se pudieron generar splits Walk-Forward. Verifica el rango temporal.")

    agent_metrics: list[PortfolioMetrics] = []
    benchmark_metrics: list[PortfolioMetrics] = []

    for train_idx, test_idx in splits:
        train_feat = features.loc[train_idx]
        test_feat = features.loc[test_idx]
        train_data = data.loc[train_idx]
        test_data = data.loc[test_idx]

        model = train_model(
            config.algorithm,
            train_data,
            train_feat,
            reward_factory,
            config.total_timesteps,
        )
        agent_metrics.append(evaluate_model(model, test_data, test_feat, reward_factory))
        benchmark_metrics.append(buy_and_hold_metrics(test_data))

    def _aggregate(metrics: list[PortfolioMetrics]) -> PortfolioMetrics:
        final_values = float(np.mean([m.final_value for m in metrics]))
        total_return = float(np.mean([m.total_return for m in metrics]))
        annual = float(np.mean([m.annual_return for m in metrics]))
        max_dd = float(np.mean([m.max_drawdown for m in metrics]))
        sharpe = float(np.mean([m.sharpe for m in metrics]))
        combined_history = (
            pd.concat([m.history for m in metrics], axis=0)
            .sort_index()
            .astype(float)
        )
        return PortfolioMetrics(final_values, total_return, annual, max_dd, sharpe, combined_history)

    return {
        "agent": _aggregate(agent_metrics),
        "benchmark": _aggregate(benchmark_metrics),
        "folds": len(splits),
    }


# ---------------------------------------------------------------------------
# Orquestación por fase
# ---------------------------------------------------------------------------


def build_feature_config(config: PipelineConfig) -> FeatureConfig:
    include_indicators = config.phase in {
        PipelinePhase.FEATURES,
        PipelinePhase.REWARD,
        PipelinePhase.VALIDATION,
        PipelinePhase.LLM,
    }

    sentiment = None
    if config.phase == PipelinePhase.LLM:
        if config.sentiment_csv is None:
            raise ValueError("La fase LLM requiere --sentiment-csv")
        sentiment = load_sentiment_feature(config.sentiment_csv)

    return FeatureConfig(include_indicators=include_indicators, sentiment=sentiment)


def build_reward_factory(config: PipelineConfig) -> Callable[[], RewardCalculator]:
    if config.phase in {PipelinePhase.REWARD, PipelinePhase.VALIDATION, PipelinePhase.LLM}:
        return lambda: SharpeReward(window=30)
    return lambda: PnLReward()


def run_pipeline(config: PipelineConfig) -> dict:
    data = load_market_data(config.symbol, config.data_csv, config.start, config.end)
    feature_config = build_feature_config(config)
    features = FeatureEngineer(feature_config).transform(data)
    data = data.loc[features.index]

    reward_factory = build_reward_factory(config)

    if config.phase in {PipelinePhase.VALIDATION, PipelinePhase.LLM}:
        return run_walk_forward_pipeline(config, data, features, reward_factory)
    return run_simple_pipeline(config, data, features, reward_factory)


def format_metrics(prefix: str, metrics: PortfolioMetrics) -> str:
    return (
        f"{prefix}: final={metrics.final_value:.4f}, total={metrics.total_return:.2%}, "
        f"annual={metrics.annual_return:.2%}, maxDD={metrics.max_drawdown:.2%}, "
        f"sharpe={metrics.sharpe:.2f}"
    )


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Pipeline iterativo de DRL")
    parser.add_argument("symbol", help="Ticker a procesar (ej. BTC-USD o AAPL)")
    parser.add_argument(
        "--phase",
        choices=[phase.value for phase in PipelinePhase],
        default=PipelinePhase.BASELINE.value,
        help="Fase del plan que se desea ejecutar",
    )
    parser.add_argument("--data-csv", type=Path, help="CSV con OHLCV pre-descargado")
    parser.add_argument("--sentiment-csv", type=Path, help="CSV con feature de sentimiento")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total de timesteps para el entrenamiento")
    parser.add_argument("--algorithm", default="ppo", help="Algoritmo DRL: ppo, a2c, sac")
    parser.add_argument("--start", type=pd.Timestamp, help="Fecha inicial (YYYY-MM-DD)")
    parser.add_argument("--end", type=pd.Timestamp, help="Fecha final (YYYY-MM-DD)")
    parser.add_argument("--train-years", type=int, default=3, help="Años de ventana de entrenamiento en Walk-Forward")
    parser.add_argument("--test-years", type=int, default=1, help="Años de ventana de prueba en Walk-Forward")

    args = parser.parse_args()
    phase = PipelinePhase(args.phase)
    return PipelineConfig(
        symbol=args.symbol,
        phase=phase,
        total_timesteps=args.timesteps,
        algorithm=args.algorithm,
        data_csv=args.data_csv,
        sentiment_csv=args.sentiment_csv,
        start=args.start,
        end=args.end,
        train_years=args.train_years,
        test_years=args.test_years,
    )


def main() -> None:
    config = parse_args()
    result = run_pipeline(config)
    agent = result["agent"]
    benchmark = result["benchmark"]
    folds = result["folds"]

    print(format_metrics("Agente", agent))
    print(format_metrics("Benchmark", benchmark))
    print(f"Folds evaluados: {folds}")


if __name__ == "__main__":
    main()