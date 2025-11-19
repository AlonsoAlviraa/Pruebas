#!/usr/bin/env python3
"""Apply price and quality filters to the cached tickers list (Volume logic removed)."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# --- CONFIGURACIÓN ---
MIN_PRICE_DEFAULT = 0.30
HISTORY_SUFFIX = "_history.csv"
# Sufijos para identificar y excluir SPACs (Warrants, Units, Rights)
SPAC_SUFFIXES = ("W", "U", "R")


def safe_float(value: object) -> Optional[float]:
    """Return a float if ``value`` is numeric, otherwise ``None``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        try:
            result = float(str(value).strip())
        except (TypeError, ValueError):
            return None
    if not math.isfinite(result):
        return None
    return result


@dataclass
class TickerMetrics:
    """Estructura simplificada: solo transportamos el ticker y su último cierre."""
    ticker: str
    last_close: Optional[float]


def iter_tickers(input_path: Optional[Path], data_root: Path) -> Iterable[str]:
    """Yield uppercase ticker symbols to evaluate."""
    # 1. Leer desde archivo de lista si se proporciona
    if input_path and input_path.exists():
        for line in input_path.read_text(encoding="utf-8").splitlines():
            ticker = line.strip().upper()
            if ticker:
                yield ticker
        return

    # 2. Si no, escanear directorio de datos
    for path in sorted(data_root.glob(f"*{HISTORY_SUFFIX}")):
        yield path.name.replace(HISTORY_SUFFIX, "").upper()


def read_history_metrics(history_path: Path) -> TickerMetrics:
    """Return the last close stored in a history CSV."""
    last_close: Optional[float] = None
    ticker_name = history_path.name.replace(HISTORY_SUFFIX, "")

    try:
        with history_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return TickerMetrics(ticker_name, None)

            # Buscar columna de cierre (insensible a mayúsculas)
            close_field = next((f for f in reader.fieldnames if f and f.lower() == "close"), None)

            if close_field:
                for row in reader:
                    close_val = safe_float(row.get(close_field))
                    if close_val is not None and close_val > 0:
                        last_close = close_val

    except FileNotFoundError:
        last_close = None

    return TickerMetrics(ticker_name, last_close)


def filter_ticker(
    ticker: str,
    data_root: Path,
    *,
    min_price: float,
    verbose: bool,
) -> Optional[str]:
    """Return ``ticker`` if it satisfies SPAC and Price constraints."""
    upper = ticker.upper()
    
    # 1. Filtro de SPACs
    if upper.endswith(SPAC_SUFFIXES):
        if verbose:
            print(f"{ticker}: descartado (ticker SPAC)")
        return None

    # 2. Lectura de métricas (Solo Precio)
    history_path = data_root / f"{upper}{HISTORY_SUFFIX}"
    metrics = read_history_metrics(history_path)
    last_close = metrics.last_close
    
    # 3. Filtro de Precio Mínimo
    if last_close is None:
        if verbose:
            print(f"{ticker}: descartado (archivo no encontrado o sin precio)")
        return None

    if last_close <= min_price:
        if verbose:
            print(f"{ticker}: descartado (precio {last_close:.4f} <= {min_price})")
        return None

    # Si pasa los filtros, retornamos el ticker limpio
    return upper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtra tickers aplicando precio mínimo y exclusiones de SPAC."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directorio que contiene los archivos *_history.csv",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("good_tickers.txt"),
        help="Archivo opcional con la lista de tickers a evaluar",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("good_tickers.txt"),
        help="Archivo donde se guardará la lista filtrada",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=MIN_PRICE_DEFAULT,
        help="Precio mínimo permitido (último cierre)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime el motivo por el que se descartan los tickers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root

    if not data_root.is_dir():
        raise SystemExit(f"El directorio de datos {data_root} no existe")

    # Fase 1: Identificar candidatos
    candidates = list(iter_tickers(args.input, data_root))
    if not candidates:
        raise SystemExit("No hay tickers candidatos para evaluar")

    # Fase 2: Filtrado
    keep: list[str] = []
    # Usamos set() para evitar duplicados si la lista de entrada está sucia
    for ticker in sorted(set(candidates)):
        result = filter_ticker(
            ticker,
            data_root,
            min_price=args.min_price,
            verbose=args.verbose,
        )
        if result:
            keep.append(result)

    # Fase 3: Guardado
    args.output.write_text("\n".join(sorted(keep)) + ("\n" if keep else ""), encoding="utf-8")

    print(
        f"Tickers evaluados: {len(set(candidates))}. Superan el filtro: {len(keep)}. "
        f"Resultado guardado en {args.output}"
    )


if __name__ == "__main__":
    main()