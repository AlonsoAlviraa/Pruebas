from __future__ import annotations

import argparse
import logging
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm
from yahooquery import Ticker

# -----------------------------------------------------------------------------
# Configuración global y constantes por defecto
# -----------------------------------------------------------------------------

DEFAULT_INPUT = "nasdaqlisted.txt"
DEFAULT_OUTPUT = "scanner_resultados.csv"
# Parámetros por defecto para los lotes y reintentos.
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_RETRIES = 4
DEFAULT_RATE_LIMIT_COOLDOWN = 12.0  # segundos

# Umbrales por defecto solicitados (50%-100% para crecimiento/momentum).
DEFAULT_MIN_EPS_GROWTH = 0.20
DEFAULT_MIN_REVENUE_GROWTH = 0.20
DEFAULT_MIN_MOMENTUM_3M = 0.25
DEFAULT_MAX_WEEKLY_VOL = 0.1
DEFAULT_MIN_ADR = 0.03

# Configuración de logging y warnings
logging.basicConfig(
    level=logging.DEBUG,  # <-- CAMBIADO DE INFO A DEBUG
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Número de tickers procesados simultáneamente en cada lote.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Reintentos cuando Yahoo Finance devuelve errores de límite de tasa.",
    )
    parser.add_argument(
        "--cooldown",
        "--delay",
        dest="cooldown",
        type=float,
        default=DEFAULT_RATE_LIMIT_COOLDOWN,
        help=(
            "Tiempo base (en segundos) que se esperará tras recibir un 429. "
            "Se multiplica por el número de reintento aplicado."
        ),
    )
    return parser.parse_args()


def load_tickers(path: Path) -> list[str]:
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


class RateLimitError(RuntimeError):
    """Señala que Yahoo Finance ha devuelto un límite de peticiones (429)."""


def is_rate_limit_message(message: str) -> bool:
    """Detecta textos habituales relacionados con errores 429."""

    lowered = message.lower()
    return "too many requests" in lowered or "rate limit" in lowered or "429" in lowered


def detect_rate_limit(payload: object) -> bool:
    """Inspecciona respuestas anidadas en busca de textos de límite de tasa."""

    if isinstance(payload, str):
        return is_rate_limit_message(payload)
    if isinstance(payload, dict):
        return any(detect_rate_limit(value) for value in payload.values())
    if isinstance(payload, list):
        return any(detect_rate_limit(value) for value in payload)
    return False


def safe_float(value: Optional[object]) -> Optional[float]:
    """Convierte valores numéricos o cadenas en ``float`` de forma segura."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_eps_growth(earnings_data: object) -> Optional[float]:
    """Obtiene el crecimiento de EPS trimestral a partir de ``earningsTrend``."""

    if not isinstance(earnings_data, dict):
        return None
    trends = earnings_data.get("trend")
    if not isinstance(trends, list):
        return None
    for entry in trends:
        if not isinstance(entry, dict):
            continue
        period = entry.get("period")
        if period in {"0q", "+0q"}:  # Periodo actual trimestral
            return safe_float(entry.get("growth"))
    return None


def get_history_slice(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extrae y limpia el histórico de un ``ticker`` concreto."""

    if history.empty:
        return pd.DataFrame()
    try:
        symbol_frame = history.xs(ticker, level=0)
    except (KeyError, ValueError):
        return pd.DataFrame()
    columns_needed = {"close", "high", "low", "open"}
    missing = columns_needed.difference(symbol_frame.columns)
    if missing:
        return pd.DataFrame()
    return symbol_frame.sort_index().dropna(subset=["close", "high", "low", "open"])


def evaluate_ticker(
    ticker: str,
    history: pd.DataFrame,
    earnings: object,
    financial: object,
    profile: object,
    thresholds: Thresholds,
) -> Optional[dict]:
    """Calcula las métricas necesarias y valida los umbrales."""

    if history.empty or len(history) < 40:
        return None

    eps_growth = extract_eps_growth(earnings)
    revenue_growth = safe_float(financial.get("revenueGrowth") if isinstance(financial, dict) else None)

    if eps_growth is None or revenue_growth is None:
        # --- BLOQUE AÑADIDO PARA DEPURACIÓN ---
        logging.debug(
            f"[{ticker}] DESCARTADO (SILENCIOSO): Faltan datos fundamentales clave. "
            f"(EPS_G: {eps_growth}, REV_G: {revenue_growth})"
        )
        # -------------------------------------
        return None
        
    if eps_growth < thresholds.min_eps_growth or revenue_growth < thresholds.min_revenue_growth:
        return None

    price_now = safe_float(history["close"].iloc[-1])
    price_3m_ago = safe_float(history["close"].iloc[0])
    if price_now is None or price_3m_ago is None or price_3m_ago <= 0:
        return None
    momentum = (price_now - price_3m_ago) / price_3m_ago
    if momentum < thresholds.min_momentum_3m:
        return None

    adr_series = (history["high"] - history["low"]) / history["close"]
    adr_mean = safe_float(adr_series.mean())
    if adr_mean is None or adr_mean < thresholds.min_adr:
        return None

    last_week = history.tail(5)
    if len(last_week) < 5:
        return None
    ref_price = safe_float(last_week["open"].iloc[0])
    if ref_price is None or ref_price <= 0:
        return None
    weekly_range = (last_week["high"].max() - last_week["low"].min())
    weekly_volatility = weekly_range / ref_price
    if weekly_volatility > thresholds.max_week_volatility:
        return None

    sector = "N/A"
    industry = "N/A"
    if isinstance(profile, dict):
        sector = profile.get("sector", sector)
        industry = profile.get("industry", industry)

    return {
        "Ticker": ticker,
        "Sector": sector,
        "Industria": industry,
        "Precio Cierre": price_now,
        "Momentum 3M": momentum,
        "ADR Medio": adr_mean,
        "Volatilidad 1W": weekly_volatility,
        "Crec. EPS (QoQ)": eps_growth,
        "Crec. Ingresos": revenue_growth,
    }


def analyze_batch(tickers: list[str], thresholds: Thresholds) -> list[dict]:
    """Descarga datos en lote y devuelve los tickers que superan los filtros."""

    try:
        client = Ticker(tickers, asynchronous=False)
        history = client.history(period="3mo", interval="1d")
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if is_rate_limit_message(message):
            raise RateLimitError(message) from exc
        raise

    earnings = client.earnings_trend
    financial = client.financial_data
    profile = client.asset_profile

    for payload in (earnings, financial, profile):
        if detect_rate_limit(payload):
            raise RateLimitError("Too Many Requests detectado en la respuesta de Yahoo Finance")

    matches: list[dict] = []
    for ticker in tickers:
        history_slice = get_history_slice(history, ticker)
        result = evaluate_ticker(
            ticker,
            history_slice,
            earnings.get(ticker) if isinstance(earnings, dict) else None,
            financial.get(ticker) if isinstance(financial, dict) else None,
            profile.get(ticker) if isinstance(profile, dict) else None,
            thresholds,
        )
        if result:
            matches.append(result)
    return matches


def chunked(sequence: list[str], size: int) -> Iterable[list[str]]:
    """Divide una secuencia en bloques de longitud ``size``."""

    if size <= 0:
        raise ValueError("El tamaño de lote debe ser mayor que cero")
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


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

    batch_size = max(1, args.batch_size)
    max_retries = max(1, args.max_retries)
    cooldown = max(args.cooldown, 0.0)

    logging.info(
        "Analizando %s tickers en lotes de %s (reintentos máximos: %s)",
        len(tickers),
        batch_size,
        max_retries,
    )

    results: list[dict] = []
    with tqdm(total=len(tickers), desc="Analizando tickers", unit="ticker") as progress:
        for chunk in chunked(tickers, batch_size):
            success = False
            batch_results: list[dict] = []
            for attempt in range(1, max_retries + 1):
                try:
                    batch_results = analyze_batch(chunk, thresholds)
                except RateLimitError as exc:
                    wait_time = cooldown * attempt
                    logging.warning(
                        "Límite de peticiones al procesar el lote iniciado en %s (intento %s/%s). "
                        "Esperando %.1f s. Detalle: %s",
                        chunk[0],
                        attempt,
                        max_retries,
                        wait_time,
                        exc,
                    )
                    if wait_time > 0:
                        time.sleep(wait_time)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logging.error(
                        "Error inesperado procesando el lote iniciado en %s: %s",
                        chunk[0],
                        exc,
                    )
                    break
                else:
                    for match in batch_results:
                        logging.info("✔ %s cumple todos los criterios", match["Ticker"])
                    results.extend(batch_results)
                    success = True
                    break
            if not success:
                logging.error(
                    "No se pudieron procesar los tickers del lote iniciado en %s tras %s intentos.",
                    chunk[0],
                    max_retries,
                )
            progress.update(len(chunk))

    df_results = build_results_dataframe(results)
    display_results(df_results)
    save_results(df_results, Path(args.output))


if __name__ == "__main__":
    main()