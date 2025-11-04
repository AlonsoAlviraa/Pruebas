from __future__ import annotations

import argparse
import logging
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from requests import Response
from requests.exceptions import RequestException
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuración global y constantes por defecto
# -----------------------------------------------------------------------------

DEFAULT_INPUT = "nasdaqlisted.txt"
DEFAULT_OUTPUT = "scanner_resultados.csv"
# Parámetros por defecto para los lotes y reintentos.
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_RETRIES = 4
DEFAULT_RATE_LIMIT_COOLDOWN = 12.0  # segundos

# Endpoints y cabeceras necesarios para sortear el consentimiento europeo de Yahoo.
YAHOO_COOKIE_ENDPOINT = "https://fc.yahoo.com"
YAHOO_CRUMB_ENDPOINT = "https://query1.finance.yahoo.com/v1/test/getcrumb"
YAHOO_QUOTE_SUMMARY_ENDPOINT = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
YAHOO_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
REQUEST_TIMEOUT = 15  # segundos
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# --- Umbrales Estrictos (VCP Estándar) por Defecto ---
DEFAULT_MIN_EPS_GROWTH = 0.25
DEFAULT_MIN_REVENUE_GROWTH = 0.25
DEFAULT_MIN_MOMENTUM_3M = 0.40
DEFAULT_MAX_WEEKLY_VOL = 0.05  # 5%
DEFAULT_MIN_ADR = 0.04  # 4%

# Configuración de logging y warnings
logging.basicConfig(
    level=logging.INFO,  # <-- CAMBIADO DE DEBUG A INFO (Terminal Limpia)
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


def safe_float(value: Optional[object]) -> Optional[float]:
    """Convierte valores numéricos o cadenas en ``float`` de forma segura."""

    if value is None:
        return None
    if isinstance(value, dict):
        if "raw" in value:
            return safe_float(value.get("raw"))
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class YahooFinanceEUClient:
    """Cliente ligero para Yahoo Finance que acepta el consentimiento de cookies."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._crumb: Optional[str] = None
        self._refresh_tokens()

    def refresh_tokens(self) -> None:
        """Permite refrescar manualmente las cookies y el crumb."""

        self._refresh_tokens()

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------
    def fetch_summary(self, ticker: str) -> dict:
        """Obtiene ``earningsTrend``, ``financialData`` y ``assetProfile``."""

        # --- MODIFICADO: Añadido 'calendarEvents' ---
        params = {"modules": "earningsTrend,financialData,assetProfile,calendarEvents"}
        data = self._request(
            YAHOO_QUOTE_SUMMARY_ENDPOINT.format(ticker=ticker),
            params,
            expected_root="quoteSummary",
        )
        results = data.get("quoteSummary", {}).get("result")
        if not results:
            raise ValueError("Yahoo Finance devolvió un resultado vacío")
        return results[0]

    def fetch_history(self, ticker: str) -> pd.DataFrame:
        """Descarga el histórico de 3 meses en velas diarias."""

        params = {"range": "3mo", "interval": "1d", "events": "history"}
        data = self._request(
            YAHOO_CHART_ENDPOINT.format(ticker=ticker),
            params,
            expected_root="chart",
        )
        chart = data.get("chart", {})
        results = chart.get("result") or []
        if not results:
            raise ValueError("No hay datos de histórico para el ticker")

        payload = results[0]
        timestamps = payload.get("timestamp") or []
        if not timestamps:
            raise ValueError("Histórico sin marcas temporales")

        indicators = payload.get("indicators", {}).get("quote", [])
        if not indicators:
            raise ValueError("Histórico sin columnas de precios")

        quotes = indicators[0]
        frame = pd.DataFrame(quotes)
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols.difference(frame.columns)
        if missing:
            raise ValueError("Histórico sin columnas completas de precios")
        frame["date"] = pd.to_datetime(timestamps, unit="s", utc=True)
        frame = frame.set_index("date").sort_index()
        frame = frame[["open", "high", "low", "close"]].dropna()
        index = frame.index
        if getattr(index, "tz", None) is not None:
            frame.index = index.tz_convert("UTC").tz_localize(None)
        return frame

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------
    def _refresh_tokens(self) -> None:
        """Obtiene cookies y ``crumb`` válidos para superar el consentimiento."""

        self.session.cookies.clear()
        try:
            self.session.get(YAHOO_COOKIE_ENDPOINT, timeout=REQUEST_TIMEOUT)
        except RequestException as exc:
            logging.debug("No se pudo obtener la cookie base de Yahoo: %s", exc)

        # --- AÑADIDO BUCLE DE REINTENTO PARA OBTENER EL CRUMB ---
        crumb_response: Optional[Response] = None
        for attempt in range(1, 6): # Reintenta 5 veces
            try:
                crumb_response = self.session.get(
                    YAHOO_CRUMB_ENDPOINT,
                    timeout=REQUEST_TIMEOUT,
                )
                crumb_response.raise_for_status() # Lanza error si es 4xx o 5xx
                break # Éxito, salimos del bucle
            
            except RequestException as exc:
                # Si es un error 429 (Too Many Requests) o 403 (Forbidden)
                status = getattr(getattr(exc, "response", None), "status_code", 0)
                if status in {429, 403, 401} and attempt < 5:
                    wait_time = 5 * attempt # Espera 5, 10, 15, 20 segundos
                    logging.warning(
                        "Error %s obteniendo el CRUMB (intento %s/5). "
                        "Esperando %s s...",
                        status, attempt, wait_time
                    )
                    time.sleep(wait_time)
                    continue
                
                # Si es otro error, lanzamos la excepción
                raise RuntimeError(
                    f"Fallo de red obteniendo el crumb de Yahoo: {exc}"
                ) from exc
            
        if crumb_response is None or not crumb_response.ok:
             raise RuntimeError(
                f"No se pudo obtener el crumb de Yahoo tras 5 intentos. "
                f"Respuesta final: {crumb_response.text if crumb_response else 'No response'}"
            )

        crumb = crumb_response.text.strip()
        if not crumb:
            raise RuntimeError("Yahoo Finance no devolvió un crumb válido")
        
        logging.debug("Crumb y cookie de sesión obtenidos con éxito.")
        self._crumb = crumb


    def _request(self, url: str, params: dict, *, expected_root: str) -> dict:
        """Realiza una petición con gestión automática de consentimiento."""

        params = {**params, "crumb": self._crumb}
        for attempt in range(1, 5):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True,
                )
            except RequestException as exc:
                raise RuntimeError(f"Fallo de red solicitando {url}: {exc}") from exc

            if response.status_code == 429:
                raise RateLimitError("Yahoo Finance devolvió HTTP 429 (rate limit)")
            if response.status_code in {401, 403}:
                self._refresh_tokens()
                continue
            if "consent.yahoo.com" in response.url:
                logging.debug("Respuesta redirigida a la página de consentimiento. Reintentando...")
                self._refresh_tokens()
                continue

            if response.status_code >= 400:
                message = self._extract_error_message(response)
                if is_rate_limit_message(message):
                    raise RateLimitError(message)
                if "Invalid Crumb" in message:
                    self._refresh_tokens()
                    continue
                raise RuntimeError(
                    f"Error {response.status_code} al consultar Yahoo Finance: {message}"
                )

            payload = self._parse_json(response)
            error = self._read_embedded_error(payload, expected_root)
            if error:
                description = error.get("description") or error.get("message") or ""
                if "Invalid Crumb" in description:
                    self._refresh_tokens()
                    continue
                if is_rate_limit_message(description):
                    raise RateLimitError(description)
                if description:
                    raise RuntimeError(description)
                raise RuntimeError(f"Yahoo Finance devolvió un error genérico: {error}")

            return payload

        raise RuntimeError("No fue posible obtener datos válidos tras varios intentos")

    @staticmethod
    def _extract_error_message(response: Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text
        if isinstance(data, dict):
            for key in ("message", "error", "description"):
                value = data.get(key)
                if isinstance(value, str):
                    return value
        return response.text

    @staticmethod
    def _parse_json(response: Response) -> dict:
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("No se pudo interpretar la respuesta JSON de Yahoo Finance") from exc

    @staticmethod
    def _read_embedded_error(payload: dict, expected_root: str) -> Optional[dict]:
        section = payload.get(expected_root)
        if not isinstance(section, dict):
            return None
        error = section.get("error")
        if isinstance(error, dict):
            return error
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


def evaluate_ticker(
    ticker: str,
    history: pd.DataFrame,
    earnings: object,
    financial: object,
    profile: object,
    calendar: object,  # <-- NUEVO
    thresholds: Thresholds,
) -> Optional[dict]:
    """Calcula las métricas necesarias y valida los umbrales."""

    if history.empty or len(history) < 40:
        return None

    # --- NUEVO FILTRO DE FECHA DE RESULTADOS ---
    days_since_earnings: Optional[int] = None
    try:
        # 1. Get today's date (UTC)
        today = datetime.now(timezone.utc)
        
        # 2. Extract timestamp
        if not isinstance(calendar, dict):
            raise ValueError("Datos de calendario no válidos")
        
        earnings_timestamps = calendar.get("earnings", {}).get("earningsDate", [])
        if not earnings_timestamps:
            raise ValueError("No se encontró 'earningsDate'")
        
        # 3. Get the most recent (first) timestamp
        ts_data = earnings_timestamps[0]
        if not isinstance(ts_data, dict) or "raw" not in ts_data:
            raise ValueError("Formato de timestamp de 'earningsDate' inesperado")
        
        ts = int(ts_data["raw"]) # Timestamp UNIX
        earnings_date = datetime.fromtimestamp(ts, timezone.utc)
        
        # 4. Calculate days since
        days_since_earnings = (today - earnings_date).days
        
        # 5. Apply filter
        if days_since_earnings < 0:
            logging.debug(f"[{ticker}] DESCARTADO: Resultados en el futuro ({days_since_earnings} días).")
            return None
        if days_since_earnings > 60:
            logging.debug(f"[{ticker}] DESCARTADO: Resultados demasiado antiguos ({days_since_earnings} días).")
            return None
        
        # Si estamos aquí, los resultados son frescos (0 <= días <= 60)
        logging.debug(f"[{ticker}] OK: Resultados presentados hace {days_since_earnings} días.")
    
    except Exception as e:
        logging.debug(f"[{ticker}] DESCARTADO: No se pudo procesar la fecha de resultados. Error: {e}")
        return None
    # --- FIN DEL NUEVO FILTRO ---


    eps_growth = extract_eps_growth(earnings)
    revenue_growth = safe_float(financial.get("revenueGrowth") if isinstance(financial, dict) else None)

    if eps_growth is None or revenue_growth is None:
        logging.debug(
            f"[{ticker}] DESCARTADO (SILENCIOSO): Faltan datos fundamentales clave. "
            f"(EPS_G: {eps_growth}, REV_G: {revenue_growth})"
        )
        return None
    if eps_growth < thresholds.min_eps_growth or revenue_growth < thresholds.min_revenue_growth:
        logging.debug(f"[{ticker}] DESCARTADO: Crecimiento insuficiente (EPS_G: {eps_growth}, REV_G: {revenue_growth})")
        return None

    price_now = safe_float(history["close"].iloc[-1])
    price_3m_ago = safe_float(history["close"].iloc[0])
    if price_now is None or price_3m_ago is None or price_3m_ago <= 0:
        return None
    momentum = (price_now - price_3m_ago) / price_3m_ago
    if momentum < thresholds.min_momentum_3m:
        logging.debug(f"[{ticker}] DESCARTADO: Momentum 3M insuficiente ({momentum})")
        return None

    adr_series = (history["high"] - history["low"]) / history["close"]
    adr_mean = safe_float(adr_series.mean())
    if adr_mean is None or adr_mean < thresholds.min_adr:
        logging.debug(f"[{ticker}] DESCARTADO: ADR insuficiente ({adr_mean})")
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
        logging.debug(f"[{ticker}] DESCARTADO: Volatilidad semanal excesiva ({weekly_volatility})")
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
        "Días Res.": days_since_earnings,  # <-- NUEVA COLUMNA
    }


def analyze_batch(
    client: YahooFinanceEUClient,
    tickers: list[str],
    thresholds: Thresholds,
) -> list[dict]:
    """Descarga datos de cada ticker aplicando los filtros establecidos."""

    matches: list[dict] = []
    for ticker in tickers:
        try:
            summary = client.fetch_summary(ticker)
            history = client.fetch_history(ticker)
        except RateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001
            logging.debug("[%s] Descargado sin éxito: %s", ticker, exc)
            continue

        earnings = summary.get("earningsTrend") if isinstance(summary, dict) else None
        financial = summary.get("financialData") if isinstance(summary, dict) else None
        profile = summary.get("assetProfile") if isinstance(summary, dict) else None
        calendar = summary.get("calendarEvents") if isinstance(summary, dict) else None # <-- NUEVO

        result = evaluate_ticker(
            ticker,
            history,
            earnings,
            financial,
            profile,
            calendar, # <-- NUEVO
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
        "Días Res.",  # <-- NUEVA COLUMNA
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
        # 'Días Res.' no se formatea como porcentaje, así que se queda fuera de este bucle
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

    client = YahooFinanceEUClient()
    results: list[dict] = []
    with tqdm(total=len(tickers), desc="Analizando tickers", unit="ticker") as progress:
        for chunk in chunked(tickers, batch_size):
            success = False
            batch_results: list[dict] = []
            for attempt in range(1, max_retries + 1):
                try:
                    batch_results = analyze_batch(client, chunk, thresholds)
                except RateLimitError as exc:
                    client.refresh_tokens()
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