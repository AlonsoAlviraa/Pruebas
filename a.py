from __future__ import annotations

import argparse
import logging
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuración global y constantes por defecto
# -----------------------------------------------------------------------------

DEFAULT_INPUT = "nasdaqlisted.txt"
DEFAULT_OUTPUT = "scanner_resultados.csv"
DEFAULT_SERIAL_DELAY = 0.15  # segundos

# Umbrales por defecto solicitados (50%-100% para crecimiento/momentum).
DEFAULT_MIN_EPS_GROWTH = 0.50
DEFAULT_MIN_REVENUE_GROWTH = 0.50
DEFAULT_MIN_MOMENTUM_3M = 0.50
DEFAULT_MAX_WEEKLY_VOL = 0.01  # 1%
DEFAULT_MIN_ADR = 0.07  # 7% promedio de rango diario

# Configuración de logging y warnings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass(frozen=True)
class Thresholds:
    """Agrupa todos los umbrales utilizados en el análisis."""

    min_eps_growth: float
    min_revenue_growth: float
    min_momentum_3m: float
    max_week_volatility: float
    min_adr: float


def parse_args() -> argparse.Namespace:
    """Crea el parser de argumentos y devuelve los parámetros."""

    parser = argparse.ArgumentParser(
        description="Scanner de acciones USA con criterios de crecimiento y volatilidad",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Ruta al archivo .txt con los tickers a analizar. Puede ser un listado "
            "simple o el fichero oficial nasdaqlisted.txt (delimitado por '|')."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Nombre del archivo CSV donde guardar los resultados.",
    )
    parser.add_argument(
        "--min-eps-growth",
        type=float,
        default=DEFAULT_MIN_EPS_GROWTH,
        help="Crecimiento mínimo de EPS (formato decimal, 0.50 = 50%).",
    )
    parser.add_argument(
        "--min-revenue-growth",
        type=float,
        default=DEFAULT_MIN_REVENUE_GROWTH,
        help="Crecimiento mínimo de ingresos (formato decimal).",
    )
    parser.add_argument(
        "--min-momentum-3m",
        type=float,
        default=DEFAULT_MIN_MOMENTUM_3M,
        help="Revalorización mínima en 3 meses (formato decimal).",
    )
    parser.add_argument(
        "--max-week-volatility",
        type=float,
        default=DEFAULT_MAX_WEEKLY_VOL,
        help="Volatilidad máxima (rango/primer precio) de la última semana.",
    )
    parser.add_argument(
        "--min-adr",
        type=float,
        default=DEFAULT_MIN_ADR,
        help="ADR mínimo medio de los últimos 3 meses (formato decimal).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_SERIAL_DELAY,
        help="Pausa (en segundos) entre solicitudes para evitar límites de la API.",
    )
    return parser.parse_args()


def load_tickers(path: Path) -> List[str]:
    """Lee y normaliza tickers desde un archivo de texto."""

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de tickers: {path}."
        )

    # Intento 1: fichero estilo nasdaqlisted.txt (delimitado por '|').
    try:
        df = pd.read_csv(path, delimiter="|")
        if "Symbol" in df.columns:
            tickers = (
                df["Symbol"].dropna().astype(str).str.extract(r"([A-Z\.]+)")[0].dropna()
            )
            if "ETF" in df.columns:
                tickers = tickers[df["ETF"].fillna("N") == "N"]
            if "Test Issue" in df.columns:
                tickers = tickers[df["Test Issue"].fillna("N") == "N"]
            cleaned = sorted(set(tickers.str.upper()))
            if cleaned:
                logging.info(
                    "Se cargaron %s tickers desde un fichero estilo nasdaqlisted.txt",
                    len(cleaned),
                )
                return cleaned
    except Exception as exc:  # noqa: BLE001 - Queremos capturar cualquier parsing error.
        logging.debug("No se pudo interpretar como nasdaqlisted.txt: %s", exc)

    # Intento 2: listado libre (símbolos separados por espacios, comas, etc.).
    content = path.read_text(encoding="utf-8", errors="ignore")
    raw_symbols = re.split(r"[\s,;|]+", content)
    cleaned = []
    for raw in raw_symbols:
        token = raw.strip().upper()
        if token and re.fullmatch(r"[A-Z\.]+", token):
            cleaned.append(token)

    unique = sorted(set(cleaned))
    if not unique:
        raise ValueError(
            "El archivo de tickers no contiene símbolos válidos (solo letras o '.')."
        )

    logging.info("Se cargaron %s tickers desde un listado plano", len(unique))
    return unique


def format_percentage(value: float) -> str:
    """Formatea un decimal como porcentaje con dos decimales."""

    return f"{value:.2%}"


def analyze_ticker(ticker: str, thresholds: Thresholds) -> Optional[dict]:
    """Obtiene datos del *ticker* y verifica los criterios solicitados."""

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        eps_growth = info.get("earningsQuarterlyGrowth")
        revenue_growth = info.get("revenueGrowth")

        if eps_growth is None or revenue_growth is None:
            return None
        if eps_growth < thresholds.min_eps_growth:
            return None
        if revenue_growth < thresholds.min_revenue_growth:
            return None

        hist_3m = yf_ticker.history(period="3mo")
        hist_1w = yf_ticker.history(period="7d")

        if hist_3m.empty or hist_1w.empty:
            return None

        hist_3m = hist_3m.dropna(subset=["Close", "High", "Low"])
        hist_1w = hist_1w.dropna(subset=["Open", "High", "Low", "Close"])

        if len(hist_3m) < 40 or len(hist_1w) < 5:
            return None

        price_now = float(hist_3m["Close"].iloc[-1])
        price_3m_ago = float(hist_3m["Close"].iloc[0])
        if price_3m_ago <= 0:
            return None
        momentum = (price_now - price_3m_ago) / price_3m_ago
        if momentum < thresholds.min_momentum_3m:
            return None

        adr_series = (hist_3m["High"] - hist_3m["Low"]) / hist_3m["Close"]
        adr_mean = float(adr_series.mean())
        if adr_mean < thresholds.min_adr:
            return None

        last_week = hist_1w.tail(5)
        ref_price = float(last_week["Open"].iloc[0])
        if ref_price <= 0:
            return None
        weekly_range = float(last_week["High"].max() - last_week["Low"].min())
        weekly_volatility = weekly_range / ref_price
        if weekly_volatility > thresholds.max_week_volatility:
            return None

        return {
            "Ticker": ticker,
            "Sector": info.get("sector", "N/A"),
            "Industria": info.get("industry", "N/A"),
            "Precio Cierre": price_now,
            "Momentum 3M": momentum,
            "ADR Medio": adr_mean,
            "Volatilidad 1W": weekly_volatility,
            "Crec. EPS (QoQ)": eps_growth,
            "Crec. Ingresos": revenue_growth,
        }
    except Exception as exc:  # noqa: BLE001 - Queremos capturar fallos de red/datos.
        message = str(exc)
        if "404" in message or "401" in message or "No data found" in message:
            return None
        logging.warning("Error procesando %s: %s", ticker, exc)
        return None


def build_results_dataframe(results: Iterable[dict]) -> pd.DataFrame:
    """Convierte los resultados en un DataFrame ordenado."""

    df = pd.DataFrame(results)
    if df.empty:
        return df

    numeric_cols = [
        "Precio Cierre",
        "Momentum 3M",
        "ADR Medio",
        "Volatilidad 1W",
        "Crec. EPS (QoQ)",
        "Crec. Ingresos",
    ]
    df = df.sort_values(by="Momentum 3M", ascending=False).reset_index(drop=True)
    for col in numeric_cols:
        if col not in df:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def display_results(df: pd.DataFrame) -> None:
    """Imprime los resultados en consola con formato legible."""

    if df.empty:
        logging.info(
            "No se encontraron acciones que cumplan todos los criterios. "
            "Puedes flexibilizar los umbrales con los argumentos de línea de comandos."
        )
        return

    printable = df.copy()
    printable["Precio Cierre"] = printable["Precio Cierre"].map(lambda x: f"${x:.2f}")
    for col in [
        "Momentum 3M",
        "ADR Medio",
        "Volatilidad 1W",
        "Crec. EPS (QoQ)",
        "Crec. Ingresos",
    ]:
        printable[col] = printable[col].map(format_percentage)

    print("\n" + "=" * 100)
    print("ACCIONES QUE CUMPLEN LOS CRITERIOS (ORDENADAS POR MOMENTUM 3M)")
    print("=" * 100)
    print(printable.to_string(index=False))


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda los resultados en CSV conservando valores numéricos."""

    if df.empty:
        return

    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logging.info("Resultados guardados en %s", output_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("No se pudo guardar el archivo CSV: %s", exc)


def main() -> None:
    args = parse_args()
    thresholds = Thresholds(
        min_eps_growth=args.min_eps_growth,
        min_revenue_growth=args.min_revenue_growth,
        min_momentum_3m=args.min_momentum_3m,
        max_week_volatility=args.max_week_volatility,
        min_adr=args.min_adr,
    )

    try:
        tickers = load_tickers(Path(args.input))
    except Exception as exc:  # noqa: BLE001
        logging.error("No fue posible cargar los tickers: %s", exc)
        return

    logging.info("Analizando %s tickers en modo secuencial", len(tickers))

    results = []
    for ticker in tqdm(tickers, desc="Analizando tickers", unit="ticker"):
        match = analyze_ticker(ticker, thresholds)
        if match:
            logging.info("✔ %s cumple todos los criterios", ticker)
            results.append(match)
        time.sleep(max(args.delay, 0))

    df_results = build_results_dataframe(results)
    display_results(df_results)
    save_results(df_results, Path(args.output))


if __name__ == "__main__":
    main()