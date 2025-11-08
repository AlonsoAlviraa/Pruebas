#!/usr/bin/env python3
"""Training orchestrator built on Ray RLlib with experiment tracking."""
from __future__ import annotations

import inspect
import logging
import warnings
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
        self._mlflow_run_active = False
        self._wandb_run_active = False
        self._owns_ray = False
        self._suppress_ray_warnings()
        self._init_tracking()

    # ------------------------ Tracking helpers ---------------------------------
    def _suppress_ray_warnings(self) -> None:
        """Reduce el ruido de advertencias de Ray para depurar con claridad."""

        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"ray(\.|$)",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"ray(\.|$)",
        )

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
            self._mlflow_run_active = True
        if self.tracking.use_wandb and wandb is not None:
            wandb.init(
                project=self.tracking.wandb_project,
                entity=self.tracking.wandb_entity,
                tags=list(self.tracking.wandb_tags),
                config={"ticker": ticker},
                name=run_name,
                reinit=True,
            )
            self._wandb_run_active = True

    def _end_tracking_run(self) -> None:  # pragma: no cover (side effects)
        if self._mlflow_run_active and self.tracking.use_mlflow and mlflow is not None:
            mlflow.end_run()
        if self._wandb_run_active and self.tracking.use_wandb and wandb is not None:
            wandb.finish()
        self._mlflow_run_active = False
        self._wandb_run_active = False

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

    # --------------------------- Auto tuning helpers -------------------------
    @staticmethod
    def _round_to_multiple(value: int, base: int = 32) -> int:
        if value <= 0:
            return base
        return max(base, int(np.ceil(value / base) * base))

    def _suggest_batch_hyperparams(
        self,
        view_length: int,
        training: TrainingConfig,
        existing: Dict[str, Any],
    ) -> Dict[str, int]:
        """Devuelve hiperparámetros recomendados si el usuario no los fijó."""

        suggestions: Dict[str, int] = {}
        if "train_batch_size" not in existing:
            target = min(max(view_length // 2, 256), 1024)
            suggestions["train_batch_size"] = self._round_to_multiple(target)

        train_batch = existing.get("train_batch_size", suggestions.get("train_batch_size"))
        if "sgd_minibatch_size" not in existing and train_batch is not None:
            minibatch = max(32, min(256, int(train_batch) // 2))
            suggestions["sgd_minibatch_size"] = self._round_to_multiple(minibatch)

        if "num_sgd_iter" not in existing:
            suggestions["num_sgd_iter"] = 4 if training.num_workers == 0 else 6

        return suggestions

    def _suggest_rollout_fragment(
        self,
        view_length: int,
        training: TrainingConfig,
        existing: Dict[str, Any],
    ) -> Optional[int]:
        if "rollout_fragment_length" in existing:
            return None

        if view_length <= 0:
            return None

        denominator = max(1, training.num_workers or 1)
        train_batch = existing.get("train_batch_size")
        if isinstance(train_batch, (int, float)):
            base = int(train_batch)
        else:
            base = view_length

        approx = max(32, min(view_length, max(base // denominator, 64)))
        return self._round_to_multiple(approx, base=16)

    @staticmethod
    def _apply_rollout_fragment(config: PPOConfig, fragment: int) -> None:
        setters = []
        env_runners = getattr(config, "env_runners", None)
        if callable(env_runners):
            setters.append((env_runners, {"rollout_fragment_length": fragment}))
        rollouts = getattr(config, "rollouts", None)
        if callable(rollouts):
            setters.append((rollouts, {"rollout_fragment_length": fragment}))

        for setter, kwargs in setters:
            try:
                setter(**kwargs)
                return
            except TypeError:
                continue
            except Exception as exc:
                logger.debug("Fallo aplicando rollout_fragment_length=%s: %s", fragment, exc)
                continue

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

        # Ray: inicializa una vez por proceso para evitar reinicios costosos por ticker
        self._ensure_ray_initialized()

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

                if callable(env_runners):
                    param_orders = (
                        {"num_env_runners": workers},
                        {"num_rollout_workers": workers},
                        {"num_workers": workers},
                    )

                    for candidate in param_orders:
                        try:
                            env_runners(**candidate)
                            return
                        except TypeError:
                            continue
                        except Exception as exc:
                            logger.warning(
                                "Fallo configurando env_runners (%s). Probando compatibilidad legacy.",
                                exc,
                            )
                            break
                rollouts = getattr(config, "rollouts", None)
                if callable(rollouts):
                    try:
                        rollouts(num_rollout_workers=workers)  # NO reasignar
                        return
                    except Exception as exc:
                        logger.warning(
                            "Fallo configurando rollouts legacy (%s).", exc
                        )
                logger.warning(
                    "No se pudo configurar el número de workers (%s). Continuando con valores por defecto de RLlib.",
                    workers,
                )

            _configure_env_workers(algo_config, training.num_workers)

            view_length = int(len(view))
            rllib_cfg = dict(training.rllib_config or {})

            auto_hparams = self._suggest_batch_hyperparams(view_length, training, rllib_cfg)
            if auto_hparams:
                logger.info(
                    "Aplicando hiperparámetros auto-ajustados para %s: %s",
                    ticker,
                    ", ".join(f"{k}={v}" for k, v in auto_hparams.items()),
                )
                for key, value in auto_hparams.items():
                    rllib_cfg.setdefault(key, value)

            rollout_fragment = self._suggest_rollout_fragment(
                view_length, training, rllib_cfg
            )
            if rollout_fragment is not None:
                self._apply_rollout_fragment(algo_config, rollout_fragment)

            rllib_cfg = {
                "gamma": rllib_cfg.get("gamma", 0.99),
                "lr": rllib_cfg.get("lr", 3e-4),
                **{k: v for k, v in rllib_cfg.items() if k not in {"gamma", "lr"}},
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
            checkpoint_dir = checkpoint_dir.resolve()
            last_ckpt_path: Optional[Path] = None

            def _save_checkpoint() -> Optional[Path]:
                try_paths = [checkpoint_dir.as_posix()]
                try:
                    try_paths.append(checkpoint_dir.as_uri())
                except ValueError:
                    pass

                last_error: Optional[Exception] = None
                for target in try_paths:
                    try:
                        ckpt = algorithm.save(target)
                        if isinstance(ckpt, dict) and "checkpoint_path" in ckpt:
                            return Path(ckpt["checkpoint_path"])  # type: ignore[arg-type]
                        if isinstance(ckpt, (str, Path)):
                            return Path(ckpt)
                        return checkpoint_dir
                    except Exception as exc:  # pragma: no cover - fallback path
                        last_error = exc
                        continue
                if last_error:
                    logger.error(
                        "No se pudo guardar el checkpoint en %s: %s",
                        checkpoint_dir,
                        last_error,
                    )
                    raise last_error
                return None

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
                    last_ckpt_path = _save_checkpoint()
                    logger.info("Iter %d: checkpoint guardado en %s", it, last_ckpt_path)

                # Parada temprana
                if training.stop_reward is not None and mean_r is not None:
                    if mean_r >= training.stop_reward:
                        logger.info("Parada temprana: reward_mean=%.4f >= objetivo=%.4f",
                                    mean_r, training.stop_reward)
                        break

            if last_ckpt_path is None:
                last_ckpt_path = _save_checkpoint()

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

    # ------------------------------ Lifecycle ---------------------------------
    def _ensure_ray_initialized(self) -> None:
        if ray.is_initialized():
            return

        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        self._owns_ray = True

    def close(self) -> None:
        """Libera recursos del orquestador (Ray y tracking)."""

        # Asegurar que no quede ningún run abierto en MLflow/W&B
        self._end_tracking_run()

        if self._owns_ray and ray.is_initialized():
            ray.shutdown()
        self._owns_ray = False

    def __enter__(self) -> "TrainingOrchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()