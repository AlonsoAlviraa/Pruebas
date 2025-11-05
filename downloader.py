#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descarga histórica de precios y fundamentales trimestrales."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from a import (
    DEFAULT_INPUT,
    RateLimitError,
    YahooFinanceEUClient,
    load_tickers,
    safe_float,
)
# backtester.py ahora no tiene estas constantes, así que las definimos aquí
CACHE_DIR = Path(".cache")
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Importamos la función de descarga de historial desde a.py (asumiendo que está allí)
# Si no, la copiamos de la versión anterior.
# Por coherencia con tu diff, la importaré desde el backtester v1
try:
    from backtester_v1 import download_history, _ensure_timestamp # Asumiendo que guardaste el anterior
except ImportError:
    # Si no, definimos las funciones necesarias que faltan
    def _ensure_timestamp(value: str) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return ts

    def download_history(
        client: YahooFinanceEUClient,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Descarga velas diarias entre ``start`` y ``end`` (ambos inclusive)."""
        from a import YAHOO_CHART_ENDPOINT
        start_epoch = int(start.timestamp())
        end_epoch = int((end + pd.Timedelta(days=1)).timestamp())
        params = {
            "period1": start_epoch,
            "period2": end_epoch,
            "interval": "1d",
            "events": "history",
        }
        payload = client._request(  # noqa: SLF001
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
        frame = frame[["open", "high", "low", "close"]].dropna()
        if frame.empty:
            raise RuntimeError("Histórico vacío tras limpiar NaN")
        return frame

FUNDAMENTALS_ENDPOINT = (
    "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/"
    "v1/finance/timeseries/{ticker}"
)
DEFAULT_LAG_DAYS = 45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga precios históricos y fundamentales trimestrales para el backtester."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Archivo con la lista de tickers a descargar.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Fecha inicial (YYYY-MM-DD) que se desea cubrir.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Fecha final (YYYY-MM-DD) que se desea cubrir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CACHE_DIR),
        help="Directorio donde se guardarán los ficheros cacheados.",
    )
    parser.add_argument(
        "--lag-days",
        type=int,
        default=DEFAULT_LAG_DAYS,
        help=(
            "Días de retraso asumidos entre el fin del trimestre y la "
            "disponibilidad pública del informe (aprox.)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redescarga aunque existan ficheros cacheados.",
    )
    return parser.parse_args()


def _combine_timeseries(result: list[dict]) -> pd.DataFrame:
    """Convierte la respuesta del endpoint de fundamentales en un DataFrame."""
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
    """Descarga EPS y Revenue trimestrales usando el endpoint timeseries."""
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


def save_history(
    client: YahooFinanceEUClient,
    ticker: str,
    history_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force: bool,
) -> bool:
    if history_path.exists() and not force:
        return True
    try:
        history = download_history(client, ticker, start, end)
    except RateLimitError:
        raise
    except Exception as exc:  # noqa: BLE001
        logging.warning("%s: error descargando histórico (%s)", ticker, exc)
        return False
    history.to_csv(history_path)
    return True


def save_fundamentals(
    client: YahooFinanceEUClient,
    ticker: str,
    fundamentals_path: Path,
    lag_days: int,
    force: bool,
) -> bool:
    if fundamentals_path.exists() and not force:
        return True
    try:
        fundamentals = fetch_quarterly_fundamentals(client, ticker)
    except RateLimitError:
        raise
    except Exception as exc:  # noqa: BLE001
        logging.warning("%s: error descargando fundamentales (%s)", ticker, exc)
        return False
    if fundamentals.empty:
        logging.warning("%s: Yahoo no devolvió fundamentales trimestrales", ticker)
        return False
    lag = pd.to_timedelta(max(lag_days, 0), unit="D")
    fundamentals["available_at"] = fundamentals["as_of"] + lag
    fundamentals = fundamentals.dropna(subset=["available_at"])
    fundamentals.to_csv(fundamentals_path, index=False)
    return True


def main() -> None:
    args = parse_args()
    start_ts = _ensure_timestamp(args.start_date)
    end_ts = _ensure_timestamp(args.end_date)
    if end_ts <= start_ts:
        raise ValueError("La fecha final debe ser posterior a la inicial")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers(Path(args.input))
    client = YahooFinanceEUClient()

    margin = pd.Timedelta(days=200)
    start_download = start_ts - margin

    logging.info(
        "Descargando datos para %s tickers (lag fundamentales: %s días)",
        len(tickers),
        args.lag_days,
    )

    for ticker in tqdm(tickers, desc="Descargas", unit="ticker"):
        history_path = output_dir / f"{ticker}_history.csv"
        fundamentals_path = output_dir / f"{ticker}_fundamentals.csv"
        try:
            ok_history = save_history(
                client,
                ticker,
                history_path,
                start_download,
                end_ts,
                args.force,
            )
            ok_fundamentals = save_fundamentals(
                client,
                ticker,
                fundamentals_path,
                args.lag_days,
                args.force,
            )
        except RateLimitError as exc:
            logging.warning("%s: límite de peticiones alcanzado (%s)", ticker, exc)
            client.refresh_tokens()
            tqdm.write("Rate limit alcanzado. Reintentando...")
            continue
        if ok_history and ok_fundamentals:
            logging.debug("%s: descarga completada", ticker)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()