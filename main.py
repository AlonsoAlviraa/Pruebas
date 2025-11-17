#!/usr/bin/env python3
"""Command line interface for the DRL research platform."""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from dataclasses import fields
from pathlib import Path
from typing import List, Set, Callable, Dict, Any

import numpy as np
import pandas as pd

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from drl_platform.env.trading_env import EnvironmentConfig, TradingEnvironment
from drl_platform.orchestrator import TrainingConfig, TrainingOrchestrator, TrackingConfig
from drl_platform.models import ensure_portfolio_model_registered
from drl_platform.validation import PurgedKFoldConfig, PurgedKFoldValidator

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("drl_platform.cli")


ENV_ID_PATTERN = re.compile(rb"drl_platform_env_shared_[0-9a-f]+")
PAYLOAD_KEY_PATTERN = re.compile(rb"drl_platform_payload_[0-9a-f]+")


def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de la línea de comandos.
    Esta función encapsula la lógica de argparse para evitar conflictos
    de namespace ('email.parser') en el script principal.
    """
    parser = argparse.ArgumentParser(description="DRL Research Platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Comando: run-training ---
    train_parser = subparsers.add_parser("run-training", help="Launch an RLlib training job")
    
    ticker_group = train_parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--tickers", help="Comma separated list of tickers (ej. 'AAL,MSFT')")
    ticker_group.add_argument(
        "--ticker_file", type=Path, help="Path to a file with one ticker per line"
    )
    
    train_parser.add_argument("--data-root", default="data", help="Directory with cached datasets")
    train_parser.add_argument(
        "--reward",
        default="pnl",
        choices=["pnl", "sharpe", "sortino", "calmar"],
        help=(
            "Tipo de recompensa a optimizar. 'pnl' es el valor por defecto recomendado para"
            " validar la arquitectura de cartera y evitar NaNs en métricas tempranas."
        ),
    )
    train_parser.add_argument("--iterations", type=int, default=10)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument(
        "--time-budget",
        type=float,
        default=5.0,
        help="Máximo de segundos dedicados al entrenamiento de cada ticker (<=0 desactiva)",
    )
    train_parser.add_argument(
        "--min-iterations",
        type=int,
        default=1,
        help="Iteraciones mínimas antes de considerar la parada por tiempo",
    )
    train_parser.add_argument(
        "--max-view-rows",
        type=int,
        default=512,
        help="Límite de filas recientes usadas para entrenar cada ticker (<=0 usa todas)",
    )
    train_parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Máximo de pasos simulados por episodio (<=0 deja que la plataforma lo ajuste)",
    )
    train_parser.add_argument(
        "--use-continuous",
        action="store_true",
        help="Usar espacio de acciones continuo",
    )
    train_parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    train_parser.add_argument("--wandb-project", help="Weights & Biases project name")

    # --- Comando: run-backtest ---
    backtest_parser = subparsers.add_parser("run-backtest", help="Evaluate a trained policy")
    ticker_group = backtest_parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--tickers", help="Comma separated list of tickers (ej. 'AAL,MSFT')"
    )
    ticker_group.add_argument(
        "--ticker_file", type=Path, help="Path to a file with one ticker per line"
    )
    backtest_parser.add_argument("--data-root", default="data", help="Directory with cached datasets")
    backtest_parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the RLlib checkpoint produced during training",
    )
    backtest_parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    backtest_parser.add_argument(
        "--purge-window",
        type=int,
        default=5,
        help="Rows removed before/after each fold to avoid leakage",
    )

    return parser.parse_args()


def _normalise_ticker(raw: str) -> str | None:
    """
    Normaliza los strings de tickers, eliminando espacios (incl. Unicode) y BOMs.
    """
    if raw is None:
        return None

    # 1. Normalizar caracteres Unicode (ej. \xa0 -> espacio)
    #    y eliminar BOMs (Byte Order Marks) comunes al inicio de archivos.
    try:
        normalised = unicodedata.normalize("NFKC", raw)
        cleaned = normalised.replace("\ufeff", "").strip()
    except TypeError:
        return None # En caso de que 'raw' no sea un string

    if not cleaned:
        return None

    # 2. Eliminar cualquier espacio interno restante y convertir a mayúsculas
    cleaned = "".join(ch for ch in cleaned if not ch.isspace())
    if not cleaned:
        return None

    return cleaned.upper()


def get_tickers_from_args(args: argparse.Namespace) -> list[str]:
    """Retorna una lista limpia y de-duplicada de tickers desde los argumentos CLI."""

    candidates: list[str] = []
    if getattr(args, "tickers", None):
        expanded = args.tickers.replace("\n", ",")
        candidates.extend(expanded.split(","))

    ticker_file: Path | None = getattr(args, "ticker_file", None)
    if ticker_file is not None:
        try:
            # Usar utf-8-sig para manejar BOM (Byte Order Mark) en archivos de texto
            content = ticker_file.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ValueError(f"Unable to read ticker file: {ticker_file}") from exc
        candidates.extend(content.splitlines())

    cleaned: list[str] = []
    seen: set[str] = set()
    
    for raw in candidates:
        ticker = _normalise_ticker(raw)
        # Si el ticker es válido y no lo hemos visto, añadirlo
        if ticker and ticker not in seen:
            cleaned.append(ticker)
            seen.add(ticker)

    if not cleaned:
        raise ValueError("No valid tickers provided after cleaning input")

    logger.info(f"Cargados {len(cleaned)} tickers únicos y válidos.")
    return cleaned


def _build_portfolio_id(tickers: list[str]) -> str:
    """Genera un identificador compacto para el conjunto de tickers."""

    if not tickers:
        return "portfolio_empty"

    ordered = sorted(tickers)
    head = ordered[:3]
    suffix = ""
    if len(ordered) > 3:
        suffix = f"_plus{len(ordered) - 3}"
    slug = "_".join(head)
    return f"portfolio_{slug}{suffix}"


def _compute_common_date_index(dataset: Dict[str, pd.DataFrame]) -> pd.Index:
    """Return sorted date index that is common to every ticker in the dataset."""

    common_index: pd.Index | None = None
    for ticker, frame in dataset.items():
        if "date" not in frame.columns:
            logger.warning("Ticker %s carece de columna 'date'. Se omite.", ticker)
            continue
        dates = pd.Index(pd.to_datetime(frame["date"]))
        if common_index is None:
            common_index = dates
        else:
            common_index = common_index.intersection(dates)
        if common_index is not None and common_index.empty:
            break

    if common_index is None:
        return pd.Index([])

    # Ordenar y eliminar duplicados para garantizar un índice estable
    return pd.Index(sorted(common_index.unique()))


def _align_views_to_index(
    dataset: Dict[str, pd.DataFrame], index: pd.Index
) -> Dict[str, pd.DataFrame]:
    """Filter each ticker DataFrame so it matches the provided date index."""

    aligned: Dict[str, pd.DataFrame] = {}
    for ticker, frame in dataset.items():
        if "date" not in frame.columns:
            logger.warning("Ticker %s no tiene columna 'date'. Se omite del backtest.", ticker)
            continue
        filtered = frame[frame["date"].isin(index)].copy()
        filtered = filtered.sort_values("date").reset_index(drop=True)
        if len(filtered) != len(index):
            logger.warning(
                "Ticker %s tiene %d filas tras alinear con %d fechas comunes. Se omite.",
                ticker,
                len(filtered),
                len(index),
            )
            continue
        aligned[ticker] = filtered
    return aligned


def _slice_portfolio_fold(
    dataset: Dict[str, pd.DataFrame], indices: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """Slice each ticker frame using the provided positional indices."""

    fold_data: Dict[str, pd.DataFrame] = {}
    for ticker, frame in dataset.items():
        if len(frame) == 0:
            continue
        subset = frame.iloc[indices]
        if subset.empty:
            continue
        fold_data[ticker] = subset.reset_index(drop=True)
    return fold_data


def _resolve_policy_action_fn(algorithm: Any) -> Callable[[Any], Any]:
    """Produce a deterministic policy function compatible with multiple RLlib APIs."""

    compute_single = getattr(algorithm, "compute_single_action", None)
    if callable(compute_single):

        def _from_algorithm(observation: Any) -> Any:
            action = compute_single(observation, explore=False)
            if isinstance(action, tuple):
                return action[0]
            return action

        return _from_algorithm

    for getter_name in ("get_policy", "get_default_policy"):
        getter = getattr(algorithm, getter_name, None)
        if not callable(getter):
            continue
        try:
            policy = getter()
        except Exception:
            continue
        if policy is None:
            continue
        compute_from_policy = getattr(policy, "compute_single_action", None)
        if callable(compute_from_policy):

            def _from_policy(observation: Any) -> Any:
                action = compute_from_policy(observation, explore=False)
                if isinstance(action, tuple):
                    return action[0]
                return action

            return _from_policy

    raise RuntimeError("Unable to resolve a deterministic policy from the loaded checkpoint")


def _extract_env_overrides(algorithm: Algorithm) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return EnvironmentConfig kwargs and model hints from the checkpoint."""

    env_cfg: Dict[str, Any] = {}
    algo_config = getattr(algorithm, "config", {}) or {}
    base_env_cfg = dict(getattr(algo_config, "env_config", {}) or {})
    valid_fields = {field.name for field in fields(EnvironmentConfig)}
    for key, value in base_env_cfg.items():
        if key in {"data", "payload_key"}:
            continue
        if key == "max_episode_steps":
            # Durante el entrenamiento se limitaban episodios. En backtest evaluamos todo el rango.
            continue
        if key in valid_fields:
            env_cfg[key] = value
    model_cfg: Dict[str, Any] = {}
    model_section = dict(getattr(algo_config, "model", {}) or {})
    custom_model = model_section.get("custom_model")
    if custom_model:
        model_cfg["custom_model"] = custom_model
    custom_model_config = model_section.get("custom_model_config")
    if isinstance(custom_model_config, dict):
        model_cfg["custom_model_config"] = dict(custom_model_config)
    return env_cfg, model_cfg


def _detect_checkpoint_tokens(checkpoint_dir: Path) -> tuple[str | None, str | None]:
    """Search the checkpoint files for env_id/payload keys generated at training time."""

    if not checkpoint_dir.is_dir():
        return None, None

    env_id: str | None = None
    payload_key: str | None = None
    candidates = [
        checkpoint_dir / "class_and_ctor_args.pkl",
        checkpoint_dir / "params.pkl",
        checkpoint_dir / "algorithm_state.pkl",
        checkpoint_dir / "env_runner" / "class_and_ctor_args.pkl",
    ]

    for candidate in candidates:
        if env_id and payload_key:
            break
        try:
            blob = candidate.read_bytes()
        except OSError:
            continue
        if env_id is None:
            match = ENV_ID_PATTERN.search(blob)
            if match:
                env_id = match.group(0).decode("ascii", errors="ignore")
        if payload_key is None:
            match = PAYLOAD_KEY_PATTERN.search(blob)
            if match:
                payload_key = match.group(0).decode("ascii", errors="ignore")

    return env_id, payload_key


def _ensure_checkpoint_env_registered(env_id: str | None) -> None:
    """Register the environment id used during training so RLlib can restore it."""

    if not env_id:
        return

    def _creator(env_context: Dict[str, Any]) -> TradingEnvironment:
        cfg = EnvironmentConfig(**env_context)
        return TradingEnvironment(cfg)

    try:
        register_env(env_id, _creator)
    except ValueError as exc:
        if "already registered" not in str(exc).lower():
            raise


def _seed_checkpoint_payload(payload_key: str | None, dataset: Dict[str, pd.DataFrame]) -> None:
    """Populate the shared payload expected by restored checkpoints."""

    if not payload_key:
        return
    try:
        TradingEnvironment.set_shared_payload(payload_key, data=dataset, overrides={})
    except Exception as exc:
        logger.warning(
            "No se pudo inicializar el payload compartido '%s': %s", payload_key, exc
        )


def _compute_max_drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak <= 0:
            continue
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _compute_backtest_metrics(values: List[float], returns: List[float]) -> Dict[str, float]:
    if not values:
        return {"pnl": 0.0, "return_pct": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    initial_value = values[0]
    final_value = values[-1]
    pnl = final_value - initial_value
    total_return = (final_value / initial_value - 1.0) if initial_value else 0.0
    returns_arr = np.asarray(returns, dtype=np.float64)
    sharpe = 0.0
    if returns_arr.size:
        std = returns_arr.std(ddof=1 if returns_arr.size > 1 else 0)
        if std > 0:
            sharpe = (returns_arr.mean() / std) * np.sqrt(252)
    max_dd = _compute_max_drawdown(values)
    return {"pnl": pnl, "return_pct": total_return, "sharpe": float(sharpe), "max_drawdown": max_dd}


def _run_policy_episode(env: TradingEnvironment, compute_action: Callable[[Any], Any]) -> Dict[str, Any]:
    obs, _ = env.reset()
    portfolio_values = [float(env.portfolio_value)]
    step_returns: list[float] = []
    done = False

    while not done:
        action = compute_action(obs)
        obs, _, terminated, truncated, info = env.step(action)
        new_value = float(env.portfolio_value)
        prev_value = float(info.get("prev_portfolio_value", portfolio_values[-1]))
        if prev_value:
            step_returns.append((new_value / prev_value) - 1.0)
        portfolio_values.append(new_value)
        done = terminated or truncated

    metrics = _compute_backtest_metrics(portfolio_values, step_returns)
    metrics.update(
        {
            "initial_value": portfolio_values[0],
            "final_value": portfolio_values[-1],
            "steps": len(portfolio_values) - 1,
        }
    )
    return metrics


def run_training(args: argparse.Namespace) -> None:
    """Ejecuta el pipeline de entrenamiento."""
    try:
        tickers = get_tickers_from_args(args)
    except ValueError as e:
        logger.error(f"Error al cargar tickers: {e}")
        return

    pipeline = DataPipeline(
        PipelineConfig(data_root=Path(args.data_root))
    )
    tracking = TrackingConfig(
        use_mlflow=args.mlflow_uri is not None,
        mlflow_tracking_uri=args.mlflow_uri,
        use_wandb=args.wandb_project is not None,
        wandb_project=args.wandb_project,
    )
    with TrainingOrchestrator(tracking=tracking) as orchestrator:
        logger.info("Cargando feature views para %d tickers...", len(tickers))
        feature_views = pipeline.load_feature_views(
            tickers, indicators=True, include_summary=True
        )

        usable_views = {tk: df for tk, df in feature_views.items() if len(df) >= 2}
        missing = sorted(set(tickers) - set(usable_views))
        for ticker in missing:
            logger.warning(
                "Feature view para %s está vacía o no se pudo cargar correctamente. Se omite del entrenamiento de cartera.",
                ticker,
            )

        if not usable_views:
            logger.error("Ningún ticker dispone de datos válidos para entrenar la cartera.")
            return

        if len(usable_views) < 2:
            logger.warning(
                "Solo %d ticker(s) con datos válidos. El agente central funcionará, pero no aprovechará la diversificación esperada.",
                len(usable_views),
            )

        portfolio_id = _build_portfolio_id(list(usable_views))
        logger.info(
            "Entrenando agente central %s con %d tickers: %s",
            portfolio_id,
            len(usable_views),
            ", ".join(sorted(usable_views)[:10]),
        )

        training_config = TrainingConfig(
            total_iterations=args.iterations,
            num_workers=args.num_workers,
            time_budget_seconds=args.time_budget,
            min_iterations=args.min_iterations,
            max_view_rows=args.max_view_rows,
            max_episode_steps=(args.max_episode_steps if args.max_episode_steps > 0 else None),
        )

        use_continuous = args.use_continuous or len(usable_views) > 1
        if len(usable_views) > 1 and not args.use_continuous:
            logger.info(
                "Forzando espacio de acción continuo para permitir vectores de pesos en cartera."
            )

        if args.reward.lower() != "pnl":
            logger.warning(
                "Recompensa '%s' seleccionada. Si observas reward_mean ausente/NaN, "
                "vuelve a ejecutar con --reward pnl para validar el pipeline.",
                args.reward,
            )

        env_kwargs = {"reward": args.reward, "use_continuous_action": use_continuous}

        try:
            orchestrator.train_portfolio(
                portfolio_id, usable_views, env_kwargs=env_kwargs, training=training_config
            )
            logger.info("--- Entrenamiento completado para %s ---", portfolio_id)
        except FileNotFoundError as e:
            logger.error(
                "Error al cargar datos para la cartera %s: %s. Deteniendo entrenamiento.",
                portfolio_id,
                e,
            )
        except Exception as e:
            logger.error(
                "Error inesperado durante el entrenamiento de la cartera %s: %s",
                portfolio_id,
                e,
            )
            logger.exception(e)


def run_backtest(args: argparse.Namespace) -> None:
    """Ejecuta un backtest Purged K-Fold usando un checkpoint entrenado con RLlib."""

    try:
        tickers = get_tickers_from_args(args)
    except ValueError as exc:
        logger.error("Error al cargar tickers para el backtest: %s", exc)
        return

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        logger.error("Checkpoint %s no encontrado.", checkpoint_path)
        return

    logger.info(
        "--- Iniciando backtest para %d tickers usando checkpoint %s ---",
        len(tickers),
        checkpoint_path,
    )

    pipeline = DataPipeline(PipelineConfig(data_root=Path(args.data_root)))
    try:
        feature_views = pipeline.load_feature_views(
            tickers, indicators=True, include_summary=True
        )
    except FileNotFoundError as exc:
        logger.error("No se pudieron cargar los datos para el backtest: %s", exc)
        return

    usable_views = {
        tk: df.sort_values("date").reset_index(drop=True)
        for tk, df in feature_views.items()
        if isinstance(df, pd.DataFrame) and len(df) >= 2
    }
    missing = sorted(set(tickers) - set(usable_views))
    for ticker in missing:
        logger.warning(
            "Feature view para %s está vacía o no se pudo cargar correctamente. Se omite del backtest.",
            ticker,
        )
    if not usable_views:
        logger.error("Ningún ticker dispone de datos válidos para evaluar.")
        return

    common_index = _compute_common_date_index(usable_views)
    if common_index.empty:
        logger.error(
            "Los tickers seleccionados no comparten fechas en común. No es posible ejecutar backtests."
        )
        return

    if len(common_index) < 2:
        logger.error(
            "Solo se encontraron %d fechas comunes entre los tickers; se requieren al menos 2 para simular.",
            len(common_index),
        )
        return

    aligned_views = _align_views_to_index(usable_views, common_index)
    if len(aligned_views) < 2:
        logger.warning(
            "Solo %d ticker(s) permanecen tras alinear por fechas. El backtest funcionará, pero no evaluará una cartera completa.",
            len(aligned_views),
        )
    if not aligned_views:
        logger.error("Tras alinear por fechas no quedó ningún ticker válido para evaluar.")
        return

    portfolio_id = _build_portfolio_id(list(aligned_views))
    logger.info(
        "Evaluando el agente %s con %d tickers y %d fechas comunes.",
        portfolio_id,
        len(aligned_views),
        len(common_index),
    )

    env_id, payload_key = _detect_checkpoint_tokens(checkpoint_path)
    if not env_id:
        logger.warning(
            "No se pudo detectar el identificador del entorno dentro del checkpoint."
            " Se intentará continuar con el registro por defecto."
        )
    _ensure_checkpoint_env_registered(env_id)
    if payload_key:
        logger.debug(
            "Inicializando payload compartido %s con %d tickers para restaurar el checkpoint.",
            payload_key,
            len(aligned_views),
        )
    else:
        logger.warning(
            "El checkpoint no incluye un payload compartido. Se usará el dataset actual solo en el backtest."
        )
    _seed_checkpoint_payload(payload_key, aligned_views)

    effective_splits = min(args.n_splits, len(common_index))
    if effective_splits < 1:
        logger.error("No es posible dividir %d fechas en folds.", len(common_index))
        return
    if effective_splits != args.n_splits:
        logger.warning(
            "Reduciendo n_splits de %d a %d para ajustarse al número de fechas disponibles.",
            args.n_splits,
            effective_splits,
        )

    validator = PurgedKFoldValidator(
        PurgedKFoldConfig(
            n_splits=effective_splits,
            purge_window=args.purge_window,
            date_column="date",
        )
    )
    dates_df = pd.DataFrame({"date": common_index})

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    ensure_portfolio_model_registered()
    algorithm: Algorithm | None = None
    try:
        algorithm = Algorithm.from_checkpoint(str(checkpoint_path))
        compute_action = _resolve_policy_action_fn(algorithm)
        env_overrides, model_cfg = _extract_env_overrides(algorithm)
        custom_model = model_cfg.get("custom_model")
        if custom_model and custom_model != "portfolio_spatial_model":
            logger.warning(
                "El checkpoint usa el modelo personalizado '%s'. Se esperaba 'portfolio_spatial_model'.",
                custom_model,
            )
        expected_tickers = (
            model_cfg.get("custom_model_config", {}).get("num_tickers")
            if isinstance(model_cfg.get("custom_model_config"), dict)
            else None
        )
        if expected_tickers and expected_tickers != len(aligned_views):
            logger.warning(
                "El checkpoint fue entrenado con %d tickers, pero el backtest recibió %d.",
                expected_tickers,
                len(aligned_views),
            )

        fold_results: list[Dict[str, Any]] = []
        for fold_idx, (_, test_idx) in enumerate(validator.split(dates_df)):
            if test_idx.size < 2:
                logger.warning("Fold %d sin suficientes fechas (%d). Se omite.", fold_idx, test_idx.size)
                continue

            fold_dataset = _slice_portfolio_fold(aligned_views, test_idx)
            if not fold_dataset:
                logger.warning(
                    "Fold %d no contiene datos alineados para los tickers seleccionados. Se omite.",
                    fold_idx,
                )
                continue

            env_config = EnvironmentConfig(data=fold_dataset, **env_overrides)
            env = TradingEnvironment(env_config)
            try:
                metrics = _run_policy_episode(env, compute_action)
            except Exception as exc:
                logger.error("Error evaluando fold %d: %s", fold_idx, exc)
                logger.exception(exc)
                continue
            finally:
                env.close()

            metrics["fold"] = fold_idx
            fold_results.append(metrics)

        if not fold_results:
            logger.error("El backtest no produjo resultados válidos.")
            return

        logger.info("--- Resultados del Backtest ---")
        for result in fold_results:
            logger.info(
                "Fold %d -> pasos=%d | PnL: %.2f (%.2f%%) | Sharpe: %.2f | Max DD: %.2f%%",
                result["fold"],
                result.get("steps", 0),
                result["pnl"],
                result["return_pct"] * 100,
                result["sharpe"],
                result["max_drawdown"] * 100,
            )

        avg_pnl = float(np.mean([r["pnl"] for r in fold_results]))
        avg_return = float(np.mean([r["return_pct"] for r in fold_results]))
        avg_sharpe = float(np.mean([r["sharpe"] for r in fold_results]))
        avg_dd = float(np.mean([r["max_drawdown"] for r in fold_results]))

        logger.info(
            "Resumen -> PnL medio: %.2f | Retorno medio: %.2f%% | Sharpe medio: %.2f | Max DD medio: %.2f%%",
            avg_pnl,
            avg_return * 100,
            avg_sharpe,
            avg_dd * 100,
        )
        logger.info("---------------------------------")
    finally:
        if algorithm is not None:
            try:
                algorithm.stop()
            except Exception:
                logger.debug("Error al detener el algoritmo tras el backtest", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()


def main() -> None:
    """Punto de entrada principal de la CLI."""
    
    # Llamada a la función encapsulada. 
    # Esto evita el conflicto de 'parser' con 'email.parser'.
    args = parse_args()
    
    if args.command == "run-training":
        run_training(args)
    elif args.command == "run-backtest":
        run_backtest(args)
    else:  # pragma: no cover - safety
        raise ValueError(f"Comando desconocido {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()