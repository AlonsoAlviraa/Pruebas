#!/usr/bin/env python3
"""Command line interface for the DRL research platform."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

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
    de namespace en el script principal.
    """
    parser = argparse.ArgumentParser(description="DRL Research Platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Comando: run-training ---
    train_parser = subparsers.add_parser("run-training", help="Launch an RLlib training job")
    
    # --- GRUPO DE TICKERS MEJORADO ---
    ticker_group = train_parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--tickers", help="Comma separated list of tickers (ej. 'AAL,MSFT')")
    ticker_group.add_argument("--ticker-file", type=Path, help="Archivo de texto con un ticker por línea (ej. 'good_tickers.txt')")
    # --------------------------------
    
    train_parser.add_argument("--data-root", default="data", help="Directory with cached datasets")
    train_parser.add_argument(
        "--reward", default="sortino", choices=["pnl", "sharpe", "sortino", "calmar"]
    )
    train_parser.add_argument("--iterations", type=int, default=10)
    train_parser.add_argument("--num-workers", type=int, default=1)
    train_parser.add_argument("--use-continuous", action="store_true", help="Usar espacio de acciones continuo")
    train_parser.add_argument("--mlflow-uri", help="MLflow tracking URI (ej. http://127.0.0.1:5000)")
    train_parser.add_argument("--wandb-project", help="Weights & Biases project name")

    # --- Comando: run-backtest ---
    backtest_parser = subparsers.add_parser("run-backtest", help="Evaluate a trained policy")
    backtest_parser.add_argument("--data-root", default="data")
    backtest_parser.add_argument("--ticker", required=True)
    backtest_parser.add_argument("--n-splits", type=int, default=5)
    backtest_parser.add_argument("--purge-window", type=int, default=5)
    
    # Esta es la línea que fallaba en tu versión.
    # Aquí, 'parser' es local y no hay conflicto.
    return parser.parse_args() 


def get_tickers_from_args(args: argparse.Namespace) -> List[str]:
    """Carga la lista de tickers desde --tickers o --ticker-file."""
    tickers_raw = []
    if args.ticker_file:
        logger.info(f"Cargando tickers desde el archivo: {args.ticker_file}")
        if not args.ticker_file.exists():
            raise FileNotFoundError(f"El archivo de tickers no se encontró: {args.ticker_file}")
        with open(args.ticker_file, 'r') as f:
            tickers_raw = f.readlines()
        
    elif args.tickers:
        tickers_raw = args.tickers.split(",")

    # --- CORRECCIÓN DE ROBUSTEZ ---
    # Limpiar (strip) CADA ticker y filtrar los que queden vacíos
    tickers_clean = [t.strip() for t in tickers_raw]
    tickers = [t for t in tickers_clean if t] # Filtra los strings vacíos
    # ----------------------------

    if not tickers:
        logger.warning("No se encontraron tickers válidos para procesar.")
    else:
        logger.info(f"Cargados {len(tickers)} tickers válidos para procesar.")
        
    return tickers


def run_training(args: argparse.Namespace) -> None:
    """Ejecuta el pipeline de entrenamiento."""
    try:
        tickers = get_tickers_from_args(args)
    except FileNotFoundError as e:
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
    orchestrator = TrainingOrchestrator(tracking)

    for ticker in tickers:
        # --- CORRECCIÓN DE ROBUSTEZ (DOBLE COMPROBACIÓN) ---
        if not ticker:
            logger.warning("Se encontró un ticker vacío, saltando.")
            continue
        # ---------------------------------------------------

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
    logger.info("--- Iniciando backtest para %s ---", args.ticker)
    
    pipeline = DataPipeline(PipelineConfig(data_root=Path(args.data_root)))
    
    try:
        data = pipeline.load_feature_view(args.ticker, indicators=True, include_summary=True)
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
            logger.error("Evaluando 'dummy model' en %d filas...", len(test_df))
            test_returns = test_df["close"].pct_change().dropna()
            pnl = (1 + test_returns).prod() - 1
            sharpe = (test_returns.mean() / (test_returns.std() + 1e-9)) * (252 ** 0.5)
            return {"pnl": pnl, "sharpe": sharpe, "train_mean_return": mean_return}

        return evaluate
    # --- FIN DE LÓGICA 'DUMMY' ---

    # TODO: Reemplazar 'build_dummy_model' con una función que cargue
    # y evalúe el agente DRL (nuestro próximo paso).
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
    
    # Esta es la línea que falla en tu versión.
    # Aquí, 'parser' NO está en este scope.
    args = parse_args() # <-- Esta es la llamada correcta
    
    if args.command == "run-training":
        run_training(args)
    elif args.command == "run-backtest":
        run_backtest(args)
    else:  # pragma: no cover - safety
        raise ValueError(f"Comando desconocido {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()