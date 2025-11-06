#!/usr/bin/env python3
"""Training orchestrator built on Ray RLlib with experiment tracking."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Callable

import numpy as np
import pandas as pd

# Dependencias opcionales de tracking
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

# Ray / RLlib
try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
except Exception as e:  # pragma: no cover
    raise RuntimeError("No se pudo importar Ray/RLlib. Asegúrate de tenerlos instalados.") from e

# ----------------------------- Import robusto del entorno -----------------------------
# Intento 1: import relativo (paquete)
try:
    from .trading_env import EnvironmentConfig, TradingEnvironment  # type: ignore
except Exception:
    # Intento 2: import absoluto (cuando se ejecuta fuera de paquete)
    try:
        from drl_platform.trading_env import EnvironmentConfig, TradingEnvironment  # type: ignore
    except Exception:
        # Intento 3: carga directa desde el archivo (fallback)
        import importlib.util
        import sys
        _pkg_dir = Path(__file__).resolve().parent
        _env_py = _pkg_dir / "trading_env.py"
        if not _env_py.exists():
            raise ModuleNotFoundError(
                "No se encontró 'trading_env.py'. Asegúrate de que exista en drl_platform/trading_env.py."
            )
        spec = importlib.util.spec_from_file_location("drl_platform_trading_env_fallback", _env_py)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["drl_platform_trading_env_fallback"] = mod
        spec.loader.exec_module(mod)  # type: ignore
        EnvironmentConfig = getattr(mod, "EnvironmentConfig")
        TradingEnvironment = getattr(mod, "TradingEnvironment")

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------------------
@dataclass
class TrackingConfig:
    """Configuración de plataformas de tracking/artefactos."""
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "drl_platform"

    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: tuple[str, ...] = ()

    output_dir: Path = Path("models")


@dataclass
class TrainingConfig:
    """Configuración del entrenamiento RLlib."""
    algorithm: str = "PPO"
    total_iterations: int = 10
    num_workers: int = 0  # empieza en local
    rllib_config: Dict[str, Any] = field(default_factory=dict)  # hiperparámetros de PPO
    stop_reward: Optional[float] = None  # corta si se alcanza esta recompensa media


# --------------------------------------------------------------------------------------
# Orquestador
# --------------------------------------------------------------------------------------
class TrainingOrchestrator:
    """Coordina ejecuciones de entrenamiento con RLlib y registra métricas."""

    def __init__(self, tracking: TrackingConfig):
        self.tracking = tracking
        self._init_tracking()

    # ------------------------ Tracking helpers ---------------------------------
    def _init_tracking(self) -> None:  # pragma: no cover (side effects)
        if self.tracking.use_mlflow and mlflow is not None:
            if self.tracking.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.tracking.mlflow_tracking_uri)
            mlflow.set_experiment(self.tracking.mlflow_experiment)
            logger.info("MLflow inicializado en %s", self.tracking.mlflow_tracking_uri or "<default>")

        if self.tracking.use_wandb and wandb is not None:
            logger.info(
                "Weights & Biases configurado (project=%s, entity=%s)",
                self.tracking.wandb_project,
                self.tracking.wandb_entity,
            )

        self.tracking.output_dir.mkdir(parents=True, exist_ok=True)

    def _start_tracking_run(self, ticker: str) -> None:  # pragma: no cover (side effects)
        run_name = f"{ticker}-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if self.tracking.use_mlflow and mlflow is not None:
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({"ticker": ticker})
        if self.tracking.use_wandb and wandb is not None:
            wandb.init(
                project=self.tracking.wandb_project,
                entity=self.tracking.wandb_entity,
                tags=list(self.tracking.wandb_tags),
                config={"ticker": ticker},
                name=run_name,
                reinit=True,
            )

    def _end_tracking_run(self) -> None:  # pragma: no cover (side effects)
        if self.tracking.use_mlflow and mlflow is not None:
            mlflow.end_run()
        if self.tracking.use_wandb and wandb is not None:
            wandb.finish()

    def _log_metrics(self, metrics: Dict[str, Any], step: int) -> None:  # pragma: no cover
        clean = {
            k: float(v) for k, v in metrics.items()
            if v is not None and isinstance(v, (int, float, np.floating)) and not np.isnan(v)
        }
        if not clean:
            return
        if self.tracking.use_mlflow and mlflow is not None:
            mlflow.log_metrics(clean, step=step)
        if self.tracking.use_wandb and wandb is not None:
            wandb.log(clean, step=step)

    # --------------------------- RL helpers ------------------------------------
    @staticmethod
    def _make_env_creator(df: pd.DataFrame, env_kwargs: Dict[str, Any]) -> Callable[[dict], TradingEnvironment]:
        """Crea un factory que RLlib usará para instanciar el entorno."""
        def _creator(config: Dict[str, Any]) -> TradingEnvironment:
            cfg = EnvironmentConfig(
                data=df,
                reward=env_kwargs.get("reward", "pnl"),
                use_continuous_action=env_kwargs.get("use_continuous_action", False),
            )
            return TradingEnvironment(cfg)
        return _creator

    # ------------------------------ Train --------------------------------------
    def train(
        self,
        ticker: str,
        view: pd.DataFrame,
        env_kwargs: Dict[str, Any],
        training: TrainingConfig,
    ) -> Path:
        """Entrena un agente RLlib sobre `view` y devuelve la ruta del checkpoint final."""
        if view is None or view.empty:
            raise ValueError("El DataFrame de características `view` está vacío.")

        # Ray: inicializa una vez
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        self._start_tracking_run(ticker)
        try:
            # 1) Selección de algoritmo
            algo = training.algorithm.upper()
            if algo != "PPO":
                raise NotImplementedError(f"Algoritmo {training.algorithm} aún no soportado (solo PPO).")

            # 2) Builder de configuración (⚠️ métodos mutan in-place; NO reasignar)
            env_creator = self._make_env_creator(view, env_kwargs)

            algo_config = PPOConfig()
            algo_config.environment(env=env_creator)   # NO reasignar
            algo_config.rollouts(num_rollout_workers=training.num_workers)  # NO reasignar

            rllib_cfg = training.rllib_config or {}
            rllib_cfg = {
                "train_batch_size": rllib_cfg.get("train_batch_size", 8192),
                "sgd_minibatch_size": rllib_cfg.get("sgd_minibatch_size", 1024),
                "num_sgd_iter": rllib_cfg.get("num_sgd_iter", 10),
                "gamma": rllib_cfg.get("gamma", 0.99),
                "lr": rllib_cfg.get("lr", 3e-4),
                **{k: v for k, v in rllib_cfg.items() if k not in {
                    "train_batch_size", "sgd_minibatch_size", "num_sgd_iter", "gamma", "lr"
                }},
            }
            algo_config.training(**rllib_cfg)  # NO reasignar
            algo_config.framework("torch")     # NO reasignar

            # 3) Construir algoritmo
            algorithm = algo_config.build()

            # 4) Loop de entrenamiento
            best_reward = -float("inf")
            checkpoint_dir = self.tracking.output_dir / ticker
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            last_ckpt_path: Optional[Path] = None

            for it in range(1, int(training.total_iterations) + 1):
                result = algorithm.train()

                metrics = {
                    "episode_reward_mean": result.get("episode_reward_mean"),
                    "episode_reward_max": result.get("episode_reward_max"),
                    "episode_reward_min": result.get("episode_reward_min"),
                    "episode_len_mean": result.get("episode_len_mean"),
                    "iteration": it,
                }
                self._log_metrics(metrics, step=it)

                mean_r = metrics.get("episode_reward_mean")
                if mean_r is not None and mean_r > best_reward:
                    best_reward = mean_r

                # Guardado periódico
                if it % 5 == 0 or it == training.total_iterations:
                    ckpt = algorithm.save(checkpoint_dir.as_posix())
                    if isinstance(ckpt, dict) and "checkpoint_path" in ckpt:
                        last_ckpt_path = Path(ckpt["checkpoint_path"])
                    elif isinstance(ckpt, (str, Path)):
                        last_ckpt_path = Path(ckpt)
                    logger.info("Iter %d: checkpoint guardado en %s", it, last_ckpt_path)

                # Parada temprana
                if training.stop_reward is not None and mean_r is not None:
                    if mean_r >= training.stop_reward:
                        logger.info("Parada temprana: reward_mean=%.4f >= objetivo=%.4f",
                                    mean_r, training.stop_reward)
                        break

            if last_ckpt_path is None:
                ckpt = algorithm.save(checkpoint_dir.as_posix())
                if isinstance(ckpt, dict) and "checkpoint_path" in ckpt:
                    last_ckpt_path = Path(ckpt["checkpoint_path"])
                elif isinstance(ckpt, (str, Path)):
                    last_ckpt_path = Path(ckpt)

            logger.info("Entrenamiento de %s finalizado. Mejor reward_mean=%.4f. Checkpoint=%s",
                        ticker, best_reward, last_ckpt_path)
            return last_ckpt_path or checkpoint_dir

        finally:
            self._end_tracking_run()
            ray.shutdown()  # limpiar entre tickers
