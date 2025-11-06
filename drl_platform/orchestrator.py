#!/usr/bin/env python3
"""Training orchestrator built on Ray RLlib with experiment tracking."""
from __future_ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

try:  # Optional heavy dependencies
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None

try:  # pragma: no cover - optional
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

try:  # pragma: no cover - optional heavy dependency
    import ray
    from ray import tune
    # Importar PPO (u otros) directamente
    from ray.rllib.algorithms.ppo import PPOConfig
except Exception:  # pragma: no cover - optional dependency
    ray = None
    tune = None
    PPOConfig = None

from drl_platform.env.trading_env import EnvironmentConfig, TradingEnvironment

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuración para el seguimiento de experimentos (MLflow, W&B)."""
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuración para el proceso de entrenamiento."""
    algorithm: str = "PPO"
    total_iterations: int = 10
    rllib_config: Dict = field(default_factory=dict)
    num_workers: int = 1
    output_dir: Path = Path("models")


class TrainingOrchestrator:
    """Coordinates distributed RLlib training runs."""

    def __init__(self, tracking: TrackingConfig):
        self.tracking = tracking
        self._init_tracking()

    def _init_tracking(self) -> None:  # pragma: no cover - side effects only
        """Inicializa las conexiones con MLflow o W&B."""
        if self.tracking.use_mlflow and mlflow is not None:
            if self.tracking.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.tracking.mlflow_tracking_uri)
            mlflow.set_experiment("drl_platform_training")
                
        if self.tracking.use_wandb and wandb is not None:
            wandb.require("core")

    def train(
        self,
        ticker: str,
        data: pd.DataFrame,
        env_kwargs: Optional[Dict] = None,
        training: Optional[TrainingConfig] = None,
    ) -> Path:
        """Ejecuta un ciclo de entrenamiento para un ticker."""
        
        if ray is None or tune is None or PPOConfig is None:
            raise ImportError(
                "Ray[rllib] es requerido. Instala con `pip install ray[rllib]`"
            )
            
        training = training or TrainingConfig()
        env_kwargs = env_kwargs or {}
        env_name = f"TradingEnv-{ticker}"
        
        # Crear la configuración base del entorno
        env_config_base = EnvironmentConfig(data=data, **env_kwargs)

        # Inicializar Ray
        ray.init(ignore_reinit_error=True, include_dashboard=False)

        def env_creator(runtime_config: Dict) -> TradingEnvironment:
            """Función 'builder' que Ray usará para crear instancias del entorno."""
            # 'runtime_config' puede usarse para variar parámetros (ej. comisiones)
            merged_config = replace(env_config_base, **runtime_config)
            return TradingEnvironment(merged_config)

        tune.register_env(env_name, env_creator)

        # Configurar el algoritmo (PPO por defecto)
        if training.algorithm.upper() != "PPO":
            raise NotImplementedError(f"Algoritmo {training.algorithm} aún no soportado.")
            
        algo_config = PPOConfig()
        algo_config = algo_config.environment(env=env_name)
        algo_config = algo_config.rollouts(num_rollout_workers=training.num_workers)
        algo_config = algo_config.training(**training.rllib_config)
        algo_config = algo_config.framework("torch") # Usar PyTorch por defecto

        algorithm = algo_config.build()

        run_name = f"{ticker}_{training.algorithm}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self._maybe_start_run(run_name)

        try:
            output_dir = Path(training.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for iteration in range(training.total_iterations):
                result = algorithm.train()
                logger.info("Iteración %s: reward_mean=%.4f", iteration, result["episode_reward_mean"])
                self._log_metrics(iteration, result)
                
            checkpoint_dir = algorithm.save(str(output_dir))
            checkpoint_path = Path(checkpoint_dir)
            
            logger.info("Entrenamiento completado. Checkpoint guardado en %s", checkpoint_path)
        
        finally:
            algorithm.stop()
            ray.shutdown()
            self._maybe_end_run()

        return checkpoint_path

    # ------------------------------------------------------------------
    def _maybe_start_run(self, run_name: str) -> None:  # pragma: no cover - side effects only
        """Inicia el 'run' en las plataformas de tracking."""
        if self.tracking.use_mlflow and mlflow is not None:
            mlflow.start_run(run_name=run_name)
        if self.tracking.use_wandb and wandb is not None:
            wandb.init(
                project=self.tracking.wandb_project,
                entity=self.tracking.wandb_entity,
                name=run_name
            )

    def _maybe_end_run(self) -> None:  # pragma: no cover - side effects only
        """Finaliza el 'run' en las plataformas de tracking."""
        if self.tracking.use_mlflow and mlflow is not None:
            mlflow.end_run()
        if self.tracking.use_wandb and wandb is not None:
            wandb.finish()

    def _log_metrics(self, iteration: int, result: Dict) -> None:  # pragma: no cover - logging only
        """Registra las métricas de RLlib en las plataformas."""
        metrics = {
            "episode_reward_mean": result.get("episode_reward_mean"),
            "episode_reward_max": result.get("episode_reward_max"),
            "episode_reward_min": result.get("episode_reward_min"),
            "episode_len_mean": result.get("episode_len_mean"),
        }
        
        # Filtrar NaNs
        clean_metrics = {k: v for k, v in metrics.items() if v is not None}
        
        if self.tracking.use_mlflow and mlflow is not None:
            mlflow.log_metrics(clean_metrics, step=iteration)
        if self.tracking.use_wandb and wandb is not None:
            wandb.log(clean_metrics, step=iteration)