#!/usr/bin/env python3
"""Training orchestrator built on Ray RLlib with experiment tracking."""
from __future__ import annotations

import inspect
import re
import logging
import time
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
    time_budget_seconds: Optional[float] = 5.0  # Límite de segundos por ticker (None desactiva)
    min_iterations: int = 1  # Iteraciones mínimas antes de evaluar parada por tiempo
    max_view_rows: Optional[int] = 512  # Número máximo de filas recientes utilizadas (None = todas)
    max_episode_steps: Optional[int] = None  # Límite de pasos por episodio (None = auto)


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
        self._algorithm: Optional[Any] = None
        self._last_algo_signature: Optional[Any] = None
        self._baseline_checkpoint: Optional[Any] = None
        self._suppress_ray_warnings()
        self._init_tracking()
        self._env_id = f"drl_platform_env_shared_{uuid4().hex}"
        self._env_registered = False
        self._env_payload_key = f"drl_platform_payload_{uuid4().hex}"

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
    def _ensure_env_registered(self) -> None:
        """Registra el entorno compartido una única vez por orquestador."""

        if self._env_registered:
            return

        valid_fields = {f.name for f in fields(EnvironmentConfig)}

        def _creator(
            config: Optional[Dict[str, Any]] = None,
            *,
            _valid_fields: set[str] = valid_fields,
        ) -> TradingEnvironment:
            merged: Dict[str, Any] = dict(config or {})

            filtered = {k: v for k, v in merged.items() if k in _valid_fields}

            data_present = filtered.get("data") is not None
            payload_present = bool(filtered.get("payload_key"))
            if not data_present and not payload_present:
                raise ValueError(
                    "Environment config debe incluir una clave 'data' o 'payload_key'."
                )

            unknown = sorted(k for k in merged if k not in _valid_fields)
            if unknown:
                logger.warning(
                    "Omitiendo configuraciones de entorno no soportadas: %s",
                    ", ".join(unknown),
                )

            cfg = EnvironmentConfig(**filtered)
            return TradingEnvironment(cfg)

        try:
            register_env(self._env_id, _creator)
        except ValueError as exc:  # pragma: no cover - registro duplicado
            if "already registered" not in str(exc).lower():
                raise
        self._env_registered = True

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
            time_budget = training.time_budget_seconds
            if time_budget is not None and time_budget > 0:
                if time_budget <= 3:
                    target = max(64, min(view_length, 96))
                elif time_budget <= 5:
                    target = max(96, min(view_length, 160))
                elif time_budget <= 10:
                    target = max(128, min(view_length, 224))
                else:
                    target = min(max(view_length // 2, 256), 1024)
            else:
                target = min(max(view_length // 2, 256), 1024)
            suggestions["train_batch_size"] = self._round_to_multiple(target)

        train_batch = existing.get("train_batch_size", suggestions.get("train_batch_size"))
        if "sgd_minibatch_size" not in existing and train_batch is not None:
            minibatch = max(32, min(192, int(train_batch) // 2))
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

    def _suggest_episode_limit(
        self,
        view_length: int,
        training: TrainingConfig,
        env_kwargs: Dict[str, Any],
    ) -> Optional[int]:
        if view_length <= 0:
            return None

        if "max_episode_steps" in env_kwargs:
            try:
                user_value = int(env_kwargs["max_episode_steps"])
            except (TypeError, ValueError):
                return None
            else:
                if user_value > 0:
                    return min(view_length, user_value)
                return None

        if training.max_episode_steps is not None:
            try:
                limit = int(training.max_episode_steps)
            except (TypeError, ValueError):
                limit = None
            else:
                if limit <= 0:
                    limit = None
            if limit is not None:
                return min(view_length, max(limit, 1))

        time_budget = training.time_budget_seconds
        limit: Optional[int] = None
        if time_budget is not None and time_budget > 0:
            if time_budget <= 3:
                limit = min(view_length, 128)
            elif time_budget <= 5:
                limit = min(view_length, 192)
            elif time_budget <= 10:
                limit = min(view_length, 256)

        if limit is None and training.max_view_rows is not None:
            try:
                candidate = int(training.max_view_rows)
            except (TypeError, ValueError):
                candidate = None
            else:
                if candidate > 0:
                    limit = min(view_length, candidate)

        if limit is None:
            return None

        return max(32, limit)

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

    def _canonicalize_payload_dict(
        self,
        payload: Dict[str, Any],
        *,
        baseline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normaliza el diccionario de configuración para que sea reutilizable."""

        normalized = dict(payload)
        reference = baseline or self._last_algo_signature or {}

        simple_value = normalized.get("simple_optimizer")
        if isinstance(simple_value, (int, float)) and simple_value == -1:
            if "simple_optimizer" in reference:
                normalized["simple_optimizer"] = reference["simple_optimizer"]
            else:
                normalized["simple_optimizer"] = False

        return normalized

    def _prepare_reset_payload(self, config: PPOConfig) -> Dict[str, Any]:
        """Convierte un ``PPOConfig`` a dict listo para ``reset_config``."""

        # Partimos de la última configuración efectiva para preservar los
        # valores ya resueltos por RLlib (p.ej. ``simple_optimizer=False``).
        base_payload = dict(self._last_algo_signature or {})

        try:
            raw_payload = config.to_dict()
        except Exception:  # pragma: no cover - compatibilidad versiones
            raw_payload = {}

        normalized = self._canonicalize_payload_dict(
            raw_payload,
            baseline=base_payload if base_payload else None,
        )

        if base_payload:
            merged = base_payload.copy()
            merged.update(normalized)
            return merged
        return normalized

    def _sync_last_payload_from_algorithm(self) -> None:
        """Extrae y guarda la configuración efectiva del algoritmo reutilizable."""

        algo = self._algorithm
        if algo is None:
            self._last_algo_signature = None
            return

        payload: Optional[Dict[str, Any]] = None

        config_attr = getattr(algo, "config", None)
        if isinstance(config_attr, dict):
            payload = self._canonicalize_payload_dict(config_attr, baseline={})
        else:
            getter = getattr(algo, "get_config", None)
            if callable(getter):
                try:
                    result = getter()
                except Exception:  # pragma: no cover - defensivo
                    logger.debug("No se pudo obtener la configuración del algoritmo", exc_info=True)
                else:
                    if isinstance(result, dict):
                        payload = self._canonicalize_payload_dict(result, baseline={})

        if payload is not None:
            self._last_algo_signature = self._config_signature(payload)

    def _publish_env_payload(
        self, view: pd.DataFrame, env_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Actualiza el payload compartido y devuelve la configuración del entorno."""

        valid_fields = {f.name for f in fields(EnvironmentConfig)}
        overrides: Dict[str, Any] = {}
        ignored: list[str] = []

        for key, value in env_kwargs.items():
            if key in {"data", "payload_key"}:
                continue
            if key in valid_fields:
                overrides[key] = value
            else:
                ignored.append(key)

        if ignored:
            logger.warning(
                "Omitiendo configuraciones de entorno desconocidas: %s",
                ", ".join(sorted(ignored)),
            )

        TradingEnvironment.set_shared_payload(
            self._env_payload_key,
            data=view,
            overrides=overrides,
        )

        return {"payload_key": self._env_payload_key}

    def _notify_shared_payload_update(self) -> None:
        """Intenta avisar a los entornos existentes para que recarguen el payload."""

        algorithm = self._algorithm
        if algorithm is None:
            return

        def _reload_env(env_obj: Any) -> None:
            if env_obj is None:
                return
            handler = getattr(env_obj, "reload_shared_payload", None)
            if callable(handler):
                try:
                    handler()
                except Exception:
                    logger.debug("Fallo recargando payload en entorno", exc_info=True)

        applied = False

        env_runner_group = getattr(algorithm, "env_runner_group", None)
        if env_runner_group is not None:
            fn = getattr(env_runner_group, "foreach_env_runner", None)
            if callable(fn):
                try:
                    def _apply_runner(runner: Any) -> None:
                        for attr in ("env", "_env", "env_ref", "env_handler"):
                            target = getattr(runner, attr, None)
                            _reload_env(target)

                    fn(_apply_runner)
                    applied = True
                except Exception:
                    logger.debug(
                        "No se pudo refrescar payload mediante env_runner_group",
                        exc_info=True,
                    )

        if not applied:
            workers = getattr(algorithm, "workers", None)
            if workers is not None:
                fn = getattr(workers, "foreach_env", None)
                if callable(fn):
                    try:
                        fn(lambda env: _reload_env(env))
                        applied = True
                    except Exception:
                        logger.debug(
                            "No se pudo refrescar payload mediante workers.foreach_env",
                            exc_info=True,
                        )

        if not applied:
            logger.debug(
                "No se encontró un mecanismo para notificar la actualización del payload compartido"
            )

    @staticmethod
    def _config_signature(raw_config: Dict[str, Any]) -> Any:
        """Genera una firma estable para detectar cambios reales en la configuración."""

        def _normalise(value: Any) -> Any:
            if isinstance(value, dict):
                items = []
                for key in sorted(value):
                    if key.startswith("_"):
                        continue
                    items.append((key, _normalise(value[key])))
                return tuple(items)
            if isinstance(value, (list, tuple)):
                return tuple(_normalise(item) for item in value)
            if isinstance(value, set):
                return tuple(sorted(_normalise(item) for item in value))
            if isinstance(value, Path):
                return ("Path", str(value))
            if isinstance(value, np.ndarray):
                return ("ndarray", tuple(np.asarray(value).tolist()))
            if isinstance(value, (np.generic,)):
                return value.item()
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return (type(value).__name__, repr(value))

        return _normalise(raw_config)

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

        max_view_rows = training.max_view_rows
        if max_view_rows is not None:
            try:
                max_view_rows = int(max_view_rows)
            except (TypeError, ValueError):
                logger.warning(
                    "max_view_rows=%r no es un entero válido; se ignorará la restricción",
                    max_view_rows,
                )
                max_view_rows = None
            else:
                if max_view_rows <= 0:
                    max_view_rows = None

        if max_view_rows is not None and len(view) > max_view_rows:
            logger.info(
                "Reduciendo feature view de %s de %d a %d filas para respetar max_view_rows",
                ticker,
                len(view),
                max_view_rows,
            )
            view = view.iloc[-max_view_rows:].copy()

        def _safe_metric(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            return numeric

        def _recover_reward_mean(result: Dict[str, Any]) -> tuple[Optional[float], Optional[str]]:
            direct = _safe_metric(result.get("episode_reward_mean"))
            if direct is not None:
                return direct, None

            hist = result.get("hist_stats")
            if isinstance(hist, dict):
                for key in ("episode_reward", "episode_reward_mean"):
                    data = hist.get(key)
                    if data is None:
                        continue
                    arr = np.asarray(data, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        return float(arr.mean()), key

            sampler = result.get("sampler_results")
            if isinstance(sampler, dict):
                recovered = _safe_metric(sampler.get("episode_reward_mean"))
                if recovered is not None:
                    return recovered, "sampler_results"

            return 0.0, "forced_zero"

        # Normalizar kwargs del entorno
        env_kwargs = dict(env_kwargs or {})
        episode_limit = self._suggest_episode_limit(len(view), training, env_kwargs)
        if episode_limit is not None and "max_episode_steps" not in env_kwargs:
            env_kwargs["max_episode_steps"] = episode_limit
            logger.info(
                "Limitando episodios de %s a %d pasos para respetar el presupuesto (time_budget=%s)",
                ticker,
                episode_limit,
                training.time_budget_seconds,
            )

        # Publicar payload compartido antes de configurar el algoritmo
        env_config = self._publish_env_payload(view, env_kwargs)

        # Ray: inicializa una vez por proceso para evitar reinicios costosos por ticker
        self._ensure_ray_initialized()

        self._start_tracking_run(ticker)
        try:
            # 1) Selección de algoritmo
            algo = training.algorithm.upper()
            if algo != "PPO":
                raise NotImplementedError(f"Algoritmo {training.algorithm} aún no soportado (solo PPO).")

            # 2) Builder de configuración (⚠️ métodos mutan in-place; NO reasignar)
            algo_config = PPOConfig()
            self._ensure_env_registered()
            algo_config.environment(env=self._env_id, env_config=env_config)  # NO reasignar

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

            simple_opt = rllib_cfg.get("simple_optimizer", -1)
            if simple_opt in (-1, None):
                rllib_cfg["simple_optimizer"] = training.num_workers == 0
            elif isinstance(simple_opt, (int, np.integer)):
                rllib_cfg["simple_optimizer"] = bool(simple_opt)

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
                remapped: Dict[str, str] = {}

                if "sgd_minibatch_size" in adapted and "sgd_minibatch_size" not in valid_param_names:
                    if "minibatch_size" in valid_param_names and "minibatch_size" not in adapted:
                        adapted["minibatch_size"] = adapted.pop("sgd_minibatch_size")
                        remapped["sgd_minibatch_size"] = "minibatch_size"
                    else:
                        removed["sgd_minibatch_size"] = adapted.pop("sgd_minibatch_size")

                if "num_sgd_iter" in adapted:
                    if "num_epochs" in valid_param_names and "num_epochs" not in adapted:
                        adapted["num_epochs"] = adapted.pop("num_sgd_iter")
                        remapped["num_sgd_iter"] = "num_epochs"
                    elif "num_sgd_iter" not in valid_param_names and not accepts_kwargs:
                        removed["num_sgd_iter"] = adapted.pop("num_sgd_iter")

                if not accepts_kwargs:
                    filtered = {
                        key: value for key, value in adapted.items() if key in valid_param_names
                    }
                    removed.update({
                        key: value for key, value in adapted.items() if key not in filtered
                    })
                else:
                    filtered = adapted

                while True:
                    try:
                        method(**filtered)  # NO reasignar
                        break
                    except TypeError as exc:
                        message = str(exc)
                        match = re.search(r"unexpected keyword argument '([^']+)'", message)
                        if not match:
                            raise
                        bad_key = match.group(1)
                        if bad_key not in filtered:
                            raise
                        removed[bad_key] = filtered.pop(bad_key)
                        continue

                if removed:
                    logger.warning(
                        "Omitiendo hiperparámetros RLlib no soportados por esta versión: %s",
                        ", ".join(sorted(removed)),
                    )
                if remapped:
                    logger.info(
                        "Adaptando hiperparámetros RLlib a nuevos nombres: %s",
                        ", ".join(f"{old}->{new}" for old, new in remapped.items()),
                    )

            _apply_training_config(algo_config, rllib_cfg)
            algo_config.framework("torch")  # NO reasignar

            # 3) Construir algoritmo
            try:
                algorithm = self._acquire_algorithm(ticker, algo_config)
            except Exception:
                logger.exception("Error durante la construcción del algoritmo")
                raise

            # 4) Loop de entrenamiento
            best_reward = float("-inf")
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

            total_iterations = max(int(training.total_iterations), 1)
            min_iterations = max(int(training.min_iterations or 1), 1)
            if min_iterations > total_iterations:
                min_iterations = total_iterations
            time_budget = training.time_budget_seconds
            if time_budget is not None:
                try:
                    time_budget = float(time_budget)
                except (TypeError, ValueError):
                    logger.warning(
                        "time_budget_seconds=%r no es válido; se ignorará la parada por tiempo",
                        training.time_budget_seconds,
                    )
                    time_budget = None
                else:
                    if time_budget <= 0:
                        time_budget = None

            start_time = time.perf_counter()
            iteration_durations: list[float] = []
            max_iters = total_iterations
            it = 0
            while it < max_iters:
                it += 1
                iter_start = time.perf_counter()
                result = algorithm.train()
                iter_duration = time.perf_counter() - iter_start
                iteration_durations.append(iter_duration)

                reward_mean, reward_source = _recover_reward_mean(result)
                metrics = {
                    "episode_reward_mean": reward_mean,
                    "episode_reward_max": _safe_metric(result.get("episode_reward_max")),
                    "episode_reward_min": _safe_metric(result.get("episode_reward_min")),
                    "episode_len_mean": _safe_metric(result.get("episode_len_mean")),
                    "iteration": it,
                    "iteration_seconds": iter_duration,
                }
                self._log_metrics(metrics, step=it)

                if reward_source is not None:
                    if reward_source == "forced_zero":
                        logger.warning(
                            "Iter %d: reward_mean ausente tras %s; usando 0.0 como fallback",
                            it,
                            "tiempo insuficiente" if time_budget is not None else "entrenamiento",
                        )
                    else:
                        logger.info(
                            "Iter %d: reward_mean recuperado desde %s", it, reward_source
                        )

                mean_r = metrics.get("episode_reward_mean")
                if mean_r is not None:
                    if best_reward == float("-inf") or mean_r > best_reward:
                        best_reward = mean_r

                if it == max_iters or it == total_iterations:
                    last_ckpt_path = _save_checkpoint()
                    logger.info("Iter %d: checkpoint guardado en %s", it, last_ckpt_path)

                if training.stop_reward is not None and mean_r is not None:
                    if mean_r >= training.stop_reward:
                        logger.info(
                            "Parada temprana: reward_mean=%.4f >= objetivo=%.4f",
                            mean_r,
                            training.stop_reward,
                        )
                        break

                elapsed = time.perf_counter() - start_time
                if time_budget is not None and elapsed >= time_budget and it >= min_iterations:
                    logger.info(
                        "Parada temprana para %s: tiempo transcurrido %.2fs supera el presupuesto de %.2fs",
                        ticker,
                        elapsed,
                        time_budget,
                    )
                    break

                if (
                    time_budget is not None
                    and it == 1
                    and max_iters > min_iterations
                    and iteration_durations[0] > 0
                ):
                    estimated = max(iteration_durations[0], 1e-6)
                    predicted_iters = int(time_budget / estimated)
                    if predicted_iters <= 0:
                        predicted_iters = min_iterations
                    max_iters = max(min_iterations, min(predicted_iters, total_iterations))
                    if max_iters < it:
                        max_iters = it

            if last_ckpt_path is None:
                last_ckpt_path = _save_checkpoint()

            if best_reward == float("-inf"):
                best_reward = 0.0

            logger.info(
                "Entrenamiento de %s finalizado. Mejor reward_mean=%.4f. Checkpoint=%s",
                ticker,
                best_reward,
                last_ckpt_path,
            )
            return last_ckpt_path or checkpoint_dir

        except Exception:
            # Si el algoritmo queda en un estado inconsistente descartamos la instancia reutilizable
            self._discard_algorithm()
            raise
        finally:
            self._end_tracking_run()

    # ------------------------------ Lifecycle ---------------------------------
    def _ensure_ray_initialized(self) -> None:
        if ray.is_initialized():
            return

        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        self._owns_ray = True

    def _acquire_algorithm(self, ticker: str, algo_config: PPOConfig) -> Any:
        """Obtiene una instancia de algoritmo lista para entrenar."""

        desired_signature = self._config_signature(algo_config.to_dict())

        if self._algorithm is not None and self._last_algo_signature == desired_signature:
            if self._baseline_checkpoint is not None:
                try:
                    self._algorithm.restore_from_object(self._baseline_checkpoint)
                except Exception as exc:
                    logger.warning(
                        "Fallo restaurando el estado base del algoritmo para %s: %s. Se reconstruirá.",
                        ticker,
                        exc,
                    )
                else:
                    logger.debug("Reutilizando algoritmo existente para %s", ticker)
                    self._notify_shared_payload_update()
                    return self._algorithm
            else:
                logger.debug("No hay checkpoint base disponible; se reconstruirá algoritmo para %s", ticker)

        if self._algorithm is None:
            logger.info("Creando algoritmo %s inicial para %s", algo_config.__class__.__name__, ticker)
        else:
            logger.info("Reconstruyendo algoritmo %s para %s", algo_config.__class__.__name__, ticker)
            self._discard_algorithm()

        self._algorithm = algo_config.build()
        self._last_algo_signature = desired_signature

        try:
            checkpoint_blob = self._algorithm.save_to_object()
        except Exception:
            logger.debug("No se pudo capturar el checkpoint base del algoritmo", exc_info=True)
            self._baseline_checkpoint = None
        else:
            self._baseline_checkpoint = checkpoint_blob

        self._notify_shared_payload_update()
        return self._algorithm

    def _discard_algorithm(self) -> None:
        if self._algorithm is None:
            return
        try:
            self._algorithm.stop()
        except Exception:
            logger.debug("Error al detener algoritmo reutilizable", exc_info=True)
        finally:
            self._algorithm = None
            self._baseline_checkpoint = None
            self._last_algo_signature = None

    def close(self) -> None:
        """Libera recursos del orquestador (Ray y tracking)."""

        # Asegurar que no quede ningún run abierto en MLflow/W&B
        self._end_tracking_run()

        # Detener algoritmo reutilizable
        self._discard_algorithm()

        if self._owns_ray and ray.is_initialized():
            ray.shutdown()
        self._owns_ray = False

    def __enter__(self) -> "TrainingOrchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()