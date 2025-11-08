#!/usr/bin/env python3
"""Command line interface for the DRL research platform."""
from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path
from typing import List, Set, Callable, Dict

import pandas as pd

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from drl_platform.orchestrator import TrainingConfig, TrainingOrchestrator, TrackingConfig
from drl_platform.validation import PurgedKFoldConfig, PurgedKFoldValidator

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("drl_platform.cli")


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
        "--reward", default="sortino", choices=["pnl", "sharpe", "sortino", "calmar"]
    )
    train_parser.add_argument("--iterations", type=int, default=10)
    train_parser.add_argument("--num-workers", type=int, default=1)
    train_parser.add_argument("--use-continuous", action="store_true", help="Usar espacio de acciones continuo")
    train_parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    train_parser.add_argument("--wandb-project", help="Weights & Biases project name")

    # --- Comando: run-backtest ---
    backtest_parser = subparsers.add_parser("run-backtest", help="Evaluate a trained policy")
    backtest_parser.add_argument("--data-root", default="data")
    backtest_parser.add_argument("--ticker", required=True)
    backtest_parser.add_argument("--n-splits", type=int, default=5)
    backtest_parser.add_argument("--purge-window", type=int, default=5)

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
        for ticker in tickers:
            logger.info("--- Iniciando entrenamiento para %s ---", ticker)
            try:
                # 1. Cargar y preparar datos
                logger.info("Cargando feature view para %s...", ticker)
                # El DataPipeline carga precios, fundamentales y resúmenes
                view = pipeline.load_feature_view(ticker, indicators=True, include_summary=True)

                if view.empty:
                    logger.warning(f"Feature view para {ticker} está vacía (NaNs?). Saltando ticker.")
                    continue

                # 2. Configurar entrenamiento
                training_config = TrainingConfig(
                    total_iterations=args.iterations,
                    num_workers=args.num_workers,
                )

                # 3. Configurar entorno
                env_kwargs = {"reward": args.reward, "use_continuous_action": args.use_continuous}

                # 4. Entrenar
                orchestrator.train(ticker, view, env_kwargs=env_kwargs, training=training_config)
                logger.info("--- Entrenamiento completado para %s ---", ticker)

            except FileNotFoundError as e:
                logger.error("Error al cargar datos para %s: %s. Saltando ticker.", ticker, e)
            except Exception as e:
                logger.error("Error inesperado durante el entrenamiento de %s: %s", ticker, e)
                # Imprimir el traceback completo para depuración
                logger.exception(e)


def run_backtest(args: argparse.Namespace) -> None:
    """Ejecuta el pipeline de backtesting (con un modelo 'dummy' por ahora)."""
    ticker = _normalise_ticker(args.ticker)
    if not ticker:
        raise ValueError("Ticker parameter is empty after cleaning")

    logger.info("--- Iniciando backtest para %s ---", ticker)
    
    pipeline = DataPipeline(PipelineConfig(data_root=Path(args.data_root)))
    
    try:
        data = pipeline.load_feature_view(ticker, indicators=True, include_summary=True)
    except FileNotFoundError as e:
        logger.error("No se pudieron cargar los datos para el backtest: %s", e)
        return
        
    validator = PurgedKFoldValidator(
        PurgedKFoldConfig(
            n_splits=args.n_splits, 
            purge_window=args.purge_window,
            date_column="date" # Asegurar que coincide con la columna de DataPipeline
        )
    )

    # --- INICIO DE LÓGICA 'DUMMY' (Paso siguiente: Reemplazar esto) ---
    def build_dummy_model(train_df: pd.DataFrame) -> Callable[[pd.DataFrame], Dict]:
        """Un 'dummy model' que solo calcula el retorno medio del set de entreno."""
        logger.info("Entrenando 'dummy model' en %d filas...", len(train_df))
        returns = train_df["close"].pct_change().dropna()
        mean_return = returns.mean()

        def evaluate(test_df: pd.DataFrame) -> Dict:
            """Un 'dummy backtest' que solo calcula métricas simples."""
            logger.info("Evaluando 'dummy model' en %d filas...", len(test_df))
            test_returns = test_df["close"].pct_change().dropna()
            pnl = (1 + test_returns).prod() - 1
            sharpe = (test_returns.mean() / (test_returns.std() + 1e-9)) * (252 ** 0.5)
            return {"pnl": pnl, "sharpe": sharpe, "train_mean_return": mean_return}

        return evaluate
    # --- FIN DE LÓGICA 'DUMMY' ---

    results = validator.evaluate(data, build_dummy_model)
    
    logger.info("--- Resultados del Backtest (Dummy) ---")
    for fold_result in results:
        logger.info("Fold %s -> PnL: %.2f%%, Sharpe: %.2f", 
                    fold_result.pop("fold"), 
                    fold_result["pnl"] * 100, 
                    fold_result["sharpe"])
    logger.info("---------------------------------")


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