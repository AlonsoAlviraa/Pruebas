
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Gestor para ejecutar optimizaciones de parámetros del ``portfolio_backtester``
# con concurrencia usando multiprocessing (ProcessPoolExecutor).
# Probado en Windows (spawn) y Linux. Requiere Python 3.10+.

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import pandas as pd

# Importa la función de simulación (debe estar en tu ruta PYTHONPATH)
from portfolio_backtester import (
    DEFAULT_END_DATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_INPUT,
    DEFAULT_RISK_PER_TRADE,
    DEFAULT_START_DATE,
    run_portfolio_backtest,
)

# ---------------------------------------------------------------------------
# Parrilla de parámetros a evaluar (modifica a tu gusto)
# ---------------------------------------------------------------------------
PARAM_GRID: Dict[str, Iterable[float]] = {
    "vol_stop_multiple": [1.0, 1.25, 1.5],
    "ema_period": [20, 35, 50],
    "ema_tolerance": [0.01, 0.015, 0.02],
    "min_momentum_3m": [0.25, 0.30, 0.35],
    "min_adr": [0.025, 0.03, 0.035],
    "max_week_volatility": [0.05, 0.06, 0.07],
}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _build_combinations(grid: Dict[str, Iterable[float]]) -> List[Dict[str, float]]:
    """Devuelve una lista de dicts con todas las combinaciones del grid."""
    from itertools import product
    keys = list(grid.keys())
    values_product = product(*(grid[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in values_product]


def _set_worker_blas_env() -> None:
    """
    Evita sobre-suscripción de hilos BLAS dentro de cada proceso.
    Llama a esto al inicio del worker (antes de cálculos pesados).
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte NaN/Inf en None para poder serializar/CSV sin problemas."""
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out[k] = None
            else:
                out[k] = float(v)
        else:
            out[k] = v
    return out


def _worker_run(args_tuple: Tuple[str, str, str, float, float, Dict[str, float]]) -> Dict[str, Any]:
    """
    Worker: ejecuta una simulación de cartera con una combinación dada.
    Retorna un diccionario combinando parámetros + métricas + estado.
    """
    _set_worker_blas_env()

    input_path, start_date, end_date, capital_inicial, riesgo_por_trade, combo = args_tuple
    record: Dict[str, Any] = {**combo}
    try:
        _, metrics = run_portfolio_backtest(
            input_path=input_path,
            start_date=start_date,
            end_date=end_date,
            vol_stop_multiple=float(combo["vol_stop_multiple"]),
            ema_period=int(combo["ema_period"]),
            ema_tolerance=float(combo["ema_tolerance"]),
            min_momentum_3m=float(combo["min_momentum_3m"]),
            min_adr=float(combo["min_adr"]),
            max_week_volatility=float(combo["max_week_volatility"]),
            capital_inicial=float(capital_inicial),
            riesgo_por_trade=float(riesgo_por_trade),
            equity_output=None,  # no generamos CSV por combinación
        )
        if metrics:
            record.update(_sanitize_metrics(metrics))
            record["status"] = "ok"
        else:
            record["status"] = "sin_resultados"
    except Exception as exc:  # pylint: disable=broad-except
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimiza parámetros del portfolio_backtester en paralelo usando procesos."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Archivo con la lista de tickers.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Fecha inicial (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Fecha final (YYYY-MM-DD).")
    parser.add_argument(
        "--capital-inicial", type=float, default=DEFAULT_INITIAL_CAPITAL,
        help="Capital inicial para el simulador de cartera."
    )
    parser.add_argument(
        "--riesgo-por-trade", type=float, default=DEFAULT_RISK_PER_TRADE,
        help="Porcentaje de capital arriesgado por operación (decimal)."
    )
    parser.add_argument(
        "--output", default="optimization_results.csv",
        help="Archivo CSV donde almacenar el ranking final."
    )
    parser.add_argument(
        "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Número de procesos en paralelo (por defecto: núm. CPUs - 1)."
    )
    parser.add_argument(
        "--chunksize", type=int, default=1,
        help="Tamaño de lote por proceso para mapear trabajos (1 suele ser seguro)."
    )
    parser.add_argument(
        "--live-write", default=None,
        help="Si se indica un CSV, va anexando filas conforme llegan los resultados."
    )
    return parser.parse_args()


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """Añade una fila a un CSV, creando cabecera si el fichero no existe."""
    create_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if create_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()

    # Nivel de LOG
    logging.info("Inicio de optimización en paralelo | %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Parámetros globales: input=%s, start=%s, end=%s, capital=%.2f, riesgo=%.4f",
                 args.input, args.start_date, args.end_date, args.capital_inicial, args.riesgo_por_trade)

    # Construye combinaciones
    combinations = _build_combinations(PARAM_GRID)
    total = len(combinations)
    logging.info("Se evaluarán %d combinaciones.", total)

    # Campo ordenado para el CSV final
    base_cols = list(PARAM_GRID.keys())
    metric_cols = ["final_equity", "total_return", "cagr", "sharpe", "realized_pnl",
                   "profit_factor", "max_drawdown", "num_trades"]
    aux_cols = ["status", "error"]
    fieldnames = base_cols + metric_cols + aux_cols

    results: List[Dict[str, Any]] = []
    output_path = Path(args.output)
    live_path = Path(args.live_write) if args.live_write else None
    if live_path:
        live_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepara argumentos para cada trabajo
    job_args = [
        (args.input, args.start_date, args.end_date, args.capital_inicial, args.riesgo_por_trade, combo)
        for combo in combinations
    ]

    # Ejecuta en paralelo
    completed = 0
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(_worker_run, ja): ja for ja in job_args}

        for future in cf.as_completed(futures):
            rec = future.result()
            completed += 1
            results.append(rec)
            if live_path:
                # aseguramos todas las claves para el CSV
                safe_row = {k: rec.get(k) for k in fieldnames}
                _append_csv_row(live_path, fieldnames, safe_row)

            if completed % max(1, total // 20) == 0 or completed == total:
                logging.info("Progreso: %d/%d (%.1f%%)",
                             completed, total, 100.0 * completed / total)

    # Construye DataFrame y ordena por métricas, si existen
    df = pd.DataFrame(results)
    if df.empty:
        logging.warning("No se generaron resultados de optimización.")
        return

    # Orden de columnas
    for col in base_cols + metric_cols + aux_cols:
        if col not in df.columns:
            df[col] = None
    df = df[base_cols + metric_cols + aux_cols]

    sort_columns = [c for c in ["profit_factor", "cagr", "sharpe"] if c in df.columns]
    if sort_columns:
        df = df.sort_values(by=sort_columns, ascending=[False]*len(sort_columns))
    df.insert(0, "rank", range(1, len(df) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("Resultados guardados en %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Nota Windows: usar guardia __main__ es imprescindible con 'spawn'
    main()
