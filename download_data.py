#!/usr/bin/env python3
"""
Advanced Data Downloader: Descarga inteligente y paralela de datos históricos y fundamentales.
Integra filtros de calidad (anti-SPAC, precio mínimo, volumen) y descarga concurrente.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Detectar dependencias
# ---------------------------------------------------------------------------
try:
    from ANTIGUOPROGRAMA.a import (
        RateLimitError,
        YahooFinanceEUClient,
        load_tickers,
        safe_float,
    )
except ImportError:
    try:
        from a import RateLimitError, YahooFinanceEUClient, load_tickers, safe_float
    except ImportError:
        print("ERROR: No se pueden importar las utilidades de Yahoo Finance.")
        print("Asegúrate de que 'ANTIGUOPROGRAMA/a.py' o 'a.py' existan.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DEFAULT_START_DATE = "2018-01-01"  # Más años para entrenar mejor
DEFAULT_END_DATE = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
DEFAULT_LAG_DAYS = 45
DEFAULT_MIN_PRICE = 0.50  # Más conservador
DEFAULT_MIN_VOLUME = 100_000  # Volumen mínimo diario promedio
SPAC_SUFFIXES = ("W", "U", "R", "-WT", "-UN", "-RT")
SPAC_KEYWORDS = ["SPAC", "ACQUISITION", "WARRANT", "UNIT", "RIGHT"]

# Descarga concurrente
MAX_WORKERS = 10  # Número de threads paralelos

# Endpoints Yahoo Finance
YAHOO_CHART_ENDPOINT = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
FUNDAMENTALS_ENDPOINT = (
    "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estructuras de Datos
# ---------------------------------------------------------------------------
@dataclass
class TickerQuality:
    """Métricas de calidad de un ticker."""

    ticker: str
    is_spac: bool
    last_price: Optional[float]
    avg_volume: Optional[float]
    data_points: int
    passes_filters: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _ensure_timestamp(value: str) -> pd.Timestamp:
    """Convierte string a Timestamp UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts


def is_likely_spac(ticker: str) -> bool:
    """Detecta si un ticker es probablemente un SPAC."""
    upper = ticker.upper()
    
    # Filtro por sufijos
    if upper.endswith(SPAC_SUFFIXES):
        return True
    
    # Filtro por keywords (requeriría name lookup, simplificado aquí)
    # En producción, podrías hacer una búsqueda en Yahoo de la compañía
    return False


# ---------------------------------------------------------------------------
# Descarga de Datos Históricos
# ---------------------------------------------------------------------------
def download_history(
    client: YahooFinanceEUClient,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Descarga OHLCV diario entre start y end."""
    start_epoch = int(start.timestamp())
    end_epoch = int((end + pd.Timedelta(days=1)).timestamp())
    params = {
        "period1": start_epoch,
        "period2": end_epoch,
        "interval": "1d",
        "events": "history",
    }
    payload = client._request(
        YAHOO_CHART_ENDPOINT.format(ticker=ticker),
        params,
        expected_root="chart",
    )
    chart = payload.get("chart", {})
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Histórico no disponible")

    result = results[0]
    indicators = result.get("indicators", {}).get("quote", [])
    if not indicators:
        raise RuntimeError("Histórico sin columnas de precios")

    frame = pd.DataFrame(indicators[0])
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols.difference(frame.columns)
    if missing:
        raise RuntimeError("Histórico incompleto: faltan columnas de precios")

    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise RuntimeError("Histórico sin marcas temporales")

    frame["date"] = pd.to_datetime(timestamps, unit="s", utc=True)
    frame = frame.set_index("date").sort_index()
    
    # Incluir volume si está disponible
    cols = ["open", "high", "low", "close"]
    if "volume" in frame.columns:
        cols.append("volume")
    
    frame = frame[cols].dropna(subset=["close"])
    if frame.empty:
        raise RuntimeError("Histórico vacío tras limpiar NaN")
    return frame


# ---------------------------------------------------------------------------
# Descarga de Fundamentales
# ---------------------------------------------------------------------------
def _combine_timeseries(result: list[dict]) -> pd.DataFrame:
    """Convierte respuesta de fundamentales en DataFrame."""
    combined: dict[pd.Timestamp, dict[str, float]] = {}
    for entry in result:
        meta = entry.get("meta", {})
        types = meta.get("type") or []
        if not types:
            continue
        series_name = types[0]
        values = entry.get(series_name) or []
        timestamps = entry.get("timestamp") or []
        for ts_raw, payload in zip(timestamps, values):
            if not isinstance(payload, dict):
                continue
            as_of: Optional[pd.Timestamp]
            raw_date = payload.get("asOfDate")
            if raw_date:
                try:
                    as_of = pd.to_datetime(raw_date, utc=True, errors="coerce")
                except (TypeError, ValueError):
                    as_of = None
            else:
                as_of = pd.to_datetime(ts_raw, unit="s", utc=True, errors="coerce")
            if as_of is None or pd.isna(as_of):
                continue
            bucket = combined.setdefault(as_of, {"as_of": as_of})
            value = safe_float(payload.get("reportedValue"))
            if value is None:
                continue
            if "EPS" in series_name.upper():
                bucket["eps"] = float(value)
            elif "REVENUE" in series_name.upper():
                bucket["revenue"] = float(value)
    if not combined:
        return pd.DataFrame(columns=["as_of", "eps", "revenue"])
    rows = sorted(combined.values(), key=lambda item: item["as_of"])
    frame = pd.DataFrame.from_records(rows)
    frame = frame.drop_duplicates(subset=["as_of"]).sort_values("as_of")
    return frame


def fetch_quarterly_fundamentals(
    client: YahooFinanceEUClient,
    ticker: str,
) -> pd.DataFrame:
    """Descarga EPS y Revenue trimestrales."""
    url = FUNDAMENTALS_ENDPOINT.format(ticker=ticker)
    now = datetime.now(tz=timezone.utc)
    params = {
        "type": "quarterlyDilutedEPS,quarterlyTotalRevenue",
        "period1": "0",
        "period2": str(int(now.timestamp())),
        "lang": "en-US",
        "region": "US",
    }
    payload = client._request(url, params, expected_root="timeseries")
    timeseries = payload.get("timeseries", {})
    result = timeseries.get("result") or []
    return _combine_timeseries(result)


# ---------------------------------------------------------------------------
# Evaluación de Calidad
# ---------------------------------------------------------------------------
def evaluate_ticker_quality(
    ticker: str,
    history_df: pd.DataFrame,
    min_price: float,
    min_volume: float,
) -> TickerQuality:
    """Evalúa si un ticker cumple con los estándares de calidad."""
    # SPAC check
    is_spac = is_likely_spac(ticker)
    
    # Price check
    last_price = None
    if not history_df.empty and "close" in history_df.columns:
        last_price = float(history_df["close"].iloc[-1])
    
    # Volume check
    avg_volume = None
    if not history_df.empty and "volume" in history_df.columns:
        avg_volume = float(history_df["volume"].mean())
    
    data_points = len(history_df)
    
    # Evaluar filtros
    passes = True
    reasons = []
    
    if is_spac:
        passes = False
        reasons.append("SPAC detectado")
    
    if last_price is None:
        passes = False
        reasons.append("Sin precio")
    elif last_price < min_price:
        passes = False
        reasons.append(f"Precio ${last_price:.2f} < ${min_price}")
    
    if avg_volume is None:
        passes = False
        reasons.append("Sin volumen")
    elif avg_volume < min_volume:
        passes = False
        reasons.append(f"Volumen {avg_volume:,.0f} < {min_volume:,.0f}")
    
    if data_points < 252:  # Al menos 1 año de datos
        passes = False
        reasons.append(f"Solo {data_points} días de datos")
    
    return TickerQuality(
        ticker=ticker,
        is_spac=is_spac,
        last_price=last_price,
        avg_volume=avg_volume,
        data_points=data_points,
        passes_filters=passes,
        reason="; ".join(reasons) if reasons else "OK",
    )


# ---------------------------------------------------------------------------
# Worker para descarga paralela
# ---------------------------------------------------------------------------
def download_ticker_worker(
    ticker: str,
    client: YahooFinanceEUClient,
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lag_days: int,
    min_price: float,
    min_volume: float,
    force: bool,
) -> tuple[str, bool, str]:
    """
    Descarga datos para un ticker y evalúa su calidad.
    Retorna: (ticker, success, message)
    """
    history_path = output_dir / f"{ticker}_history.csv"
    fundamentals_path = output_dir / f"{ticker}_fundamentals.csv"
    
    try:
        # 1. Descargar histórico
        if history_path.exists() and not force:
            history_df = pd.read_csv(history_path, index_col=0, parse_dates=True)
        else:
            history_df = download_history(client, ticker, start, end)
            history_df.to_csv(history_path)
        
        # 2. Evaluar calidad
        quality = evaluate_ticker_quality(ticker, history_df, min_price, min_volume)
        
        if not quality.passes_filters:
            # Eliminar archivos de tickers que no pasan filtros
            if history_path.exists():
                history_path.unlink()
            return (ticker, False, f"Rechazado: {quality.reason}")
        
        # 3. Descargar fundamentales solo si pasa filtros
        if ticker.upper() != "QQQ":
            if fundamentals_path.exists() and not force:
                pass  # Ya existe
            else:
                try:
                    fundamentals = fetch_quarterly_fundamentals(client, ticker)
                    if not fundamentals.empty:
                        lag = pd.to_timedelta(max(lag_days, 0), unit="D")
                        fundamentals["available_at"] = fundamentals["as_of"] + lag
                        fundamentals = fundamentals.dropna(subset=["available_at"])
                        fundamentals.to_csv(fundamentals_path, index=False)
                except Exception as e:
                    logger.debug(f"{ticker}: No fundamentals available ({e})")
        
        return (ticker, True, f"OK: ${quality.last_price:.2f}, Vol {quality.avg_volume:,.0f}")
    
    except RateLimitError:
        raise  # Re-lanzar para manejo especial
    except Exception as e:
        return (ticker, False, f"Error: {str(e)[:50]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga inteligente y paralela de datos históricos y fundamentales de alta calidad."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Archivo con lista de tickers (uno por línea). Si no se especifica, usa lista de NASDAQ.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Fecha inicial (YYYY-MM-DD). Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"Fecha final (YYYY-MM-DD). Default: hoy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directorio de salida para los CSVs.",
    )
    parser.add_argument(
        "--lag-days",
        type=int,
        default=DEFAULT_LAG_DAYS,
        help="Días de lag para fundamentales (point-in-time).",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=DEFAULT_MIN_PRICE,
        help=f"Precio mínimo para filtrar. Default: ${DEFAULT_MIN_PRICE}",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=DEFAULT_MIN_VOLUME,
        help=f"Volumen mínimo promedio diario. Default: {DEFAULT_MIN_VOLUME:,.0f}",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Número de threads paralelos. Default: {MAX_WORKERS}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar re-descarga aunque existan archivos.",
    )
    parser.add_argument(
        "--output-list",
        type=Path,
        default=Path("good_tickers_filtrados.txt"),
        help="Archivo donde guardar la lista de tickers aprobados.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    start_ts = _ensure_timestamp(args.start_date)
    end_ts = _ensure_timestamp(args.end_date)
    if end_ts <= start_ts:
        raise ValueError("La fecha final debe ser posterior a la inicial")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar lista de tickers
    if args.input and args.input.exists():
        tickers = load_tickers(args.input)
    else:
        # Usar NASDAQ listing como fallback
        nasdaq_file = Path("nasdaqlisted.txt")
        if nasdaq_file.exists():
            logger.info("No se especificó --input, usando nasdaqlisted.txt")
            df = pd.read_csv(nasdaq_file, sep="|")
            if "Symbol" in df.columns:
                tickers = df["Symbol"].dropna().str.strip().str.upper().tolist()
            else:
                raise ValueError("nasdaqlisted.txt no tiene columna 'Symbol'")
        else:
            raise ValueError("Especifica --input o coloca nasdaqlisted.txt en el directorio")

    # Añadir QQQ siempre
    tickers_set = set(t.upper() for t in tickers)
    tickers_set.add("QQQ")
    tickers_list = sorted(tickers_set)

    logger.info(f"Descargando datos para {len(tickers_list)} tickers candidatos")
    logger.info(f"Rango: {start_ts.date()} a {end_ts.date()}")
    logger.info(f"Filtros: Precio mín ${args.min_price}, Volumen mín {args.min_volume:,.0f}")
    logger.info(f"Threads paralelos: {args.max_workers}")

    # Añadir margen para indicadores técnicos
    margin = pd.Timedelta(days=200)
    start_download = start_ts - margin

    client = YahooFinanceEUClient()
    successful = []
    failed = []

    # Descarga paralela con ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                download_ticker_worker,
                ticker,
                client,
                args.output_dir,
                start_download,
                end_ts,
                args.lag_days,
                args.min_price,
                args.min_volume,
                args.force,
            ): ticker
            for ticker in tickers_list
        }

        with tqdm(total=len(futures), desc="Descargando", unit="ticker") as pbar:
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    ticker_result, success, message = future.result()
                    if success:
                        successful.append(ticker_result)
                        pbar.set_postfix_str(f"✓ {ticker_result}")
                    else:
                        failed.append((ticker_result, message))
                        logger.debug(f"{ticker_result}: {message}")
                except RateLimitError:
                    logger.warning(f"{ticker}: Rate limit alcanzado. Refrescando tokens...")
                    client.refresh_tokens()
                    failed.append((ticker, "Rate limit"))
                except Exception as e:
                    failed.append((ticker, str(e)))
                    logger.error(f"{ticker}: Error inesperado: {e}")
                finally:
                    pbar.update(1)

    # Guardar lista de tickers aprobados
    if successful:
        args.output_list.write_text("\n".join(sorted(successful)) + "\n", encoding="utf-8")

    # Resumen
    logger.info("=" * 60)
    logger.info(f"RESUMEN DE DESCARGA")
    logger.info("=" * 60)
    logger.info(f"Total candidatos:      {len(tickers_list)}")
    logger.info(f"Descargados con éxito: {len(successful)}")
    logger.info(f"Rechazados/Fallidos:   {len(failed)}")
    logger.info(f"Lista de aprobados:    {args.output_list}")
    logger.info(f"Datos guardados en:    {args.output_dir}/")
    logger.info("=" * 60)

    if len(successful) < 10:
        logger.warning("Muy pocos tickers aprobados. Considera relajar los filtros.")


if __name__ == "__main__":
    main()
