#!/usr/bin/env python3
"""Training orchestrator built on Ray RLlib with experiment tracking."""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from uuid import uuid4

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
    from ray.tune.registry import register_env
except Exception as e:  # pragma: no cover
    raise RuntimeError("No se pudo importar Ray/RLlib. Asegúrate de tenerlos instalados.") from e

# ----------------------------- Import robusto del entorno -----------------------------
# Intento 1: import relativo (paquete moderno)
try:
    from .env.trading_env import EnvironmentConfig, TradingEnvironment  # type: ignore
except Exception:
    # Intento 2: import absoluto (cuando se instala como paquete)
    try:
        from drl_platform.env.trading_env import EnvironmentConfig, TradingEnvironment  # type: ignore
    except Exception:
        # Intento 3: compatibilidad con versiones antiguas (archivo en el paquete raíz)
        try:
            from .trading_env import EnvironmentConfig, TradingEnvironment  # type: ignore
        except Exception:
            try:
                from drl_platform.trading_env import EnvironmentConfig, TradingEnvironment  # type: ignore
            except Exception:
                # Intento 4: carga directa desde el archivo (fallback)
                import importlib.util
                import sys

                _pkg_dir = Path(__file__).resolve().parent
                candidate_paths = [
                    _pkg_dir / "env" / "trading_env.py",
                    _pkg_dir / "trading_env.py",
                ]

                for _env_py in candidate_paths:
                    if _env_py.exists():
                        break
                else:
                    raise ModuleNotFoundError(
                        "No se encontró 'trading_env.py'. Asegúrate de que exista en drl_platform/env/trading_env.py."
                    )

                spec = importlib.util.spec_from_file_location(
                    "drl_platform_trading_env_fallback", _env_py
                )
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
    total_iterations: int = 100
    num_workers: int = 4
    stop_reward: Optional[float] = None
    rllib_config: Optional[Dict[str, Any]] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
class TrainingOrchestrator:
    """Orquesta el entrenamiento de un solo agente usando RLlib."""
    def __init__(self, tracking_config: Optional[TrackingConfig] = None):
        self.tracking = tracking_config or TrackingConfig()
        if self.tracking.use_mlflow and mlflow is None:  # pragma: no cover
            logger.warning("use_mlflow=True pero 'mlflow' no está instalado. Desactivando.")
            self.tracking.use_mlflow = False
        if self.tracking.use_wandb and wandb is None:  # pragma: no cover
            logger.warning("use_wandb=True pero 'wandb' no está instalado. Desactivando.")
            self.tracking.use_wandb = False
        self.tracking.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- Tracking helpers --------------------------------
    def _start_tracking_run(self, ticker: str) -> None:  # pragma: no cover (side effects)
        run_name = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.tracking.use_mlflow and mlflow is not None:
            if self.tracking.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.tracking.mlflow_tracking_uri)
            mlflow.set_experiment(self.tracking.mlflow_experiment)
            mlflow.start_run(run_name=run_name, tags={"ticker": ticker})

        if self.tracking.use_wandb and wandb is not None:
            wandb.init(
                project=self.tracking.wandb_project,
                entity=self.tracking.wandb_entity,
                tags=list(self.tracking.wandb_tags) + [ticker],
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
    def _make_env_creator(
        df: pd.DataFrame, env_kwargs: Optional[Dict[str, Any]]
    ) -> str:
        """Registra un entorno específico y devuelve su identificador para RLlib."""

        env_id = f"drl_platform_env_{uuid4().hex}"
        base_kwargs = dict(env_kwargs or {})
        valid_fields = {f.name for f in fields(EnvironmentConfig)}

        def _creator(config: Optional[Dict[str, Any]] = None) -> TradingEnvironment:
            merged: Dict[str, Any] = dict(base_kwargs)
            if config:
                merged.update(config)

            filtered = {k: v for k, v in merged.items() if k in valid_fields}
            filtered.setdefault("data", df)

            unknown = sorted(k for k in merged.keys() if k not in valid_fields)
            if unknown:
                logger.warning(
                    "Omitiendo configuraciones de entorno no soportadas: %s",
                    ", ".join(unknown),
                )

            cfg = EnvironmentConfig(**filtered)
            return TradingEnvironment(cfg)

        register_env(env_id, _creator)
        return env_id

    # ------------------------------ Train --------------------------------------
    def train(
        self,
        ticker: str,
        view: pd.DataFrame,
        env_kwargs: Optional[Dict[str, Any]],
        training: TrainingConfig,
    ) -> Path:
        """Entrena un agente RLlib y conserva compatibilidad binaria."""
        return self._run_training(ticker, view, env_kwargs, training)

    def _run_training(
        self,
        ticker: str,
        view: pd.DataFrame,
        env_kwargs: Optional[Dict[str, Any]],
        training: TrainingConfig,
    ) -> Path:
        """Implementación principal del loop de entrenamiento.

        Devuelve la ruta del último checkpoint generado para el ``ticker``.
        """
        if view is None or view.empty:
            raise ValueError("El DataFrame de características `view` está vacío.")

        # Normalizar kwargs del entorno
        env_kwargs = dict(env_kwargs or {})

        # Ray: inicializa una vez
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        algorithm: Optional[Any] = None

        self._start_tracking_run(ticker)
        try:
            # 1) Selección de algoritmo
            algo = training.algorithm.upper()
            if algo != "PPO":
                raise NotImplementedError(f"Algoritmo {training.algorithm} aún no soportado (solo PPO).")

            # 2) Builder de configuración (⚠️ métodos mutan in-place; NO reasignar)
            env_creator = self._make_env_creator(view, env_kwargs)

            algo_config = PPOConfig()
            algo_config.environment(env=env_creator, env_config=dict(env_kwargs))  # NO reasignar

            def _configure_env_workers(config: PPOConfig, workers: int) -> None:
                """Compatibilidad entre APIs antiguas y nuevas de RLlib."""

                workers = int(workers)
                env_runners = getattr(config, "env_runners", None)

                if env_runners is None:
                    config.rollouts(num_rollout_workers=workers)  # NO reasignar
                    return

                signature = inspect.signature(env_runners)
                kwargs: Dict[str, Any] = {}

                if "num_env_runners" in signature.parameters:
                    kwargs["num_env_runners"] = workers
                elif "num_rollout_workers" in signature.parameters:
                    kwargs["num_rollout_workers"] = workers
                else:  # Fallback genérico
                    kwargs["num_env_runners"] = workers

                try:
                    env_runners(**kwargs)
                except Exception as exc:
                    logger.warning(
                        "Fallo configurando env_runners (%s). Reintentando con rollouts legacy.",
                        exc,
                    )
                    config.rollouts(num_rollout_workers=workers)  # NO reasignar

            _configure_env_workers(algo_config, training.num_workers)

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

            def _apply_training_config(config: PPOConfig, params: Dict[str, Any]) -> None:
                method = config.training
                signature = inspect.signature(method)
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
                )
                valid_param_names = {
                    name
                    for name, param in signature.parameters.items()
                    if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                }

                adapted = dict(params)
                removed: Dict[str, Any] = {}

                if "sgd_minibatch_size" in adapted and "sgd_minibatch_size" not in valid_param_names:
                    if "minibatch_size" in valid_param_names and "minibatch_size" not in adapted:
                        adapted["minibatch_size"] = adapted.pop("sgd_minibatch_size")
                    else:
                        removed["sgd_minibatch_size"] = adapted.pop("sgd_minibatch_size")

                if not accepts_kwargs:
                    filtered = {
                        key: value for key, value in adapted.items() if key in valid_param_names
                    }
                    removed.update({
                        key: value for key, value in adapted.items() if key not in filtered
                    })
                else:
                    filtered = adapted

                if removed:
                    logger.warning(
                        "Omitiendo hiperparámetros RLlib no soportados por esta versión: %s",
                        ", ".join(sorted(removed)),
                    )

                method(**filtered)  # NO reasignar

            _apply_training_config(algo_config, rllib_cfg)
            algo_config.framework("torch")  # NO reasignar

            # 3) Construir algoritmo
            try:
                algorithm = algo_config.build()
            except Exception:
                logger.exception("Error durante la construcción del algoritmo")
                raise

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
                ckpt = algorithm.save(checkpoint_dir.resolve().as_uri())
                if isinstance(ckpt, dict) and "checkpoint_path" in ckpt:
                    last_ckpt_path = Path(ckpt["checkpoint_path"])
                elif isinstance(ckpt, (str, Path)):
                    last_ckpt_path = Path(ckpt)

            logger.info("Entrenamiento de %s finalizado. Mejor reward_mean=%.4f. Checkpoint=%s",
                        ticker, best_reward, last_ckpt_path)
            return last_ckpt_path or checkpoint_dir

        finally:
            try:
                if algorithm is not None:
                    algorithm.stop()
            except Exception:
                logger.debug("Error al detener el algoritmo", exc_info=True)

            self._end_tracking_run()
            ray.shutdown()  # limpiar entre tickers