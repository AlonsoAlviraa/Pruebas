#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import json
from dataclasses import dataclass
from datetime import timezone, datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

# Asumimos que 'a.py' está en la misma carpeta
from a import (
    DEFAULT_INPUT,
    # ¡YA NO IMPORTAMOS LOS FILTROS DE CRECIMIENTO!
    DEFAULT_MAX_WEEKLY_VOL,
    DEFAULT_MIN_ADR,
    DEFAULT_MIN_MOMENTUM_3M,
    MAX_EARNINGS_AGE_DAYS,
    Thresholds,
    YahooFinanceEUClient,
    YAHOO_CHART_ENDPOINT,
    # ¡YA NO IMPORTAMOS LOS EXTRACTORES DE FUNDAMENTALES!
    load_tickers,
    safe_float,
    extract_last_earnings_date # <-- SÍ importamos este
)


# ---------------------------------------------------------------------------
# Configuración y estructuras de datos
# ---------------------------------------------------------------------------


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_STOP_LOSS_PCT = 0.04  # 4%
DEFAULT_PROFIT_MULTIPLE = 2.0  # 2R
CACHE_DIR = Path(".cache")


@dataclass
class BacktestConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    stop_loss_pct: float
    take_profit_multiple: float


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_date: pd.Timestamp
    exit_price: float
    outcome: str

    @property
    def risk_amount(self) -> float:
        return self.entry_price - self.stop_price

    @property
    def r_multiple(self) -> float:
        risk = self.risk_amount
        if risk <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / risk


# ---------------------------------------------------------------------------
# Utilidades de descarga de datos
# ---------------------------------------------------------------------------


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
    # ... (Esta función no cambia) ...
    start_epoch = int(start.timestamp())
    end_epoch = int((end + pd.Timedelta(days=1)).timestamp())
    params = {
        "period1": start_epoch,
        "period2": end_epoch,
        "interval": "1d",
        "events": "history",
    }
    payload = client._request(  # noqa: SLF001 - uso interno documentado
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

# --- build_fundamental_snapshot() HA SIDO ELIMINADO ---
# (Ya no es necesario, porque no podemos usar los fundamentales en el backtest)

# ---------------------------------------------------------------------------
# Métricas del escáner
# ---------------------------------------------------------------------------

def compute_momentum(history: pd.DataFrame, window: int = 63) -> Optional[float]:
    # ... (Esta función no cambia) ...
    if len(history) < window + 1:
        return None
    recent = history["close"].iloc[-1]
    past = history["close"].iloc[-1 - window]
    if past <= 0:
        return None
    return (recent - past) / past


def compute_average_adr(history: pd.DataFrame, window: int = 63) -> Optional[float]:
    # ... (Esta función no cambia) ...
    if len(history) < window:
        return None
    adr = (history["high"] - history["low"]) / history["close"]
    adr = adr.replace([pd.NA, pd.NaT], pd.NA).dropna()
    if len(adr) < window:
        return None
    return float(adr.tail(window).mean())


def compute_weekly_volatility(history: pd.DataFrame, window: int = 5) -> Optional[float]:
    # ... (Esta función no cambia) ...
    if len(history) < window:
        return None
    segment = history.tail(window)
    reference = segment["open"].iloc[0]
    if reference <= 0:
        return None
    weekly_range = segment["high"].max() - segment["low"].min()
    return float(weekly_range / reference)


def passes_scanner(history: pd.DataFrame, thresholds: Thresholds) -> bool:
    """Función de filtro TÉCNICO. Ignora EPS/Revenue."""
    momentum = compute_momentum(history)
    adr = compute_average_adr(history)
    week_vol = compute_weekly_volatility(history)
    if momentum is None or adr is None or week_vol is None:
        return False
    # NOTA: NO filtramos por EPS/Revenue
    if momentum < thresholds.min_momentum_3m:
        return False
    if adr < thresholds.min_adr:
        return False
    if week_vol > thresholds.max_week_volatility:
        return False
    return True


# ---------------------------------------------------------------------------
# Motor de simulación
# ---------------------------------------------------------------------------


def simulate_ticker(
    ticker: str,
    history: pd.DataFrame,
    # --- MODIFICADO: 'fundamentals' ya no se necesita aquí ---
    # fundamentals: Optional[FundamentalSnapshot], 
    last_earnings_date: Optional[pd.Timestamp], # <-- SÍ pasamos la fecha de ER
    thresholds: Thresholds,
    config: BacktestConfig,
) -> list[Trade]:
    trades: list[Trade] = []
    
    # --- MODIFICADO: Ya no filtramos por 'fundamentals is None' ---
    # if fundamentals is None:
    #     return trades

    open_trade: Optional[Trade] = None
    usable = history.loc[: config.end_date]
    if usable.empty:
        return trades

    dates = list(usable.index)
    total_days = len(dates)

    for idx, current_date in enumerate(dates):
        row = usable.iloc[idx]

        if open_trade and current_date >= open_trade.entry_date:
            # ... (Lógica de salida de la operación - no cambia) ...
            low_price = float(row["low"])
            high_price = float(row["high"])
            exit_trade: Optional[Trade] = None
            if low_price <= open_trade.stop_price:
                exit_trade = Trade(
                    ticker=open_trade.ticker,
                    entry_date=open_trade.entry_date,
                    entry_price=open_trade.entry_price,
                    stop_price=open_trade.stop_price,
                    target_price=open_trade.target_price,
                    exit_date=current_date,
                    exit_price=open_trade.stop_price,
                    outcome="stop",
                )
            elif high_price >= open_trade.target_price:
                exit_trade = Trade(
                    ticker=open_trade.ticker,
                    entry_date=open_trade.entry_date,
                    entry_price=open_trade.entry_price,
                    stop_price=open_trade.stop_price,
                    target_price=open_trade.target_price,
                    exit_date=current_date,
                    exit_price=open_trade.target_price,
                    outcome="target",
                )

            if exit_trade:
                trades.append(exit_trade)
                open_trade = None
                continue

        if open_trade is not None:
            continue

        if idx >= total_days - 1:
            continue

        next_date = dates[idx + 1]
        if next_date > config.end_date or next_date < config.start_date:
            continue

        current_ts = current_date
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("UTC")
        else:
            current_ts = current_ts.tz_convert("UTC")

        # --- MODIFICADO: Eliminados los filtros de EPS/Revenue ---
        # if fundamentals.eps_growth < thresholds.min_eps_growth:
        #     continue
        # if fundamentals.revenue_growth < thresholds.min_revenue_growth:
        #     continue

        # --- MODIFICADO: Lógica de fecha de ER simplificada ---
        if last_earnings_date: # Solo si tenemos una fecha de ER
            days_since_earnings = (current_ts - last_earnings_date).days
            if days_since_earnings < 0 or days_since_earnings >= MAX_EARNINGS_AGE_DAYS:
                continue
        # Si no hay 'last_earnings_date', no podemos aplicar el filtro, así que continuamos

        history_until_now = history.loc[:current_date]
        if not passes_scanner(history_until_now, thresholds):
            continue

        next_row = usable.iloc[idx + 1]
        if pd.isna(next_row["open"]):
            continue
        entry_price = float(next_row["open"])
        stop_price = entry_price * (1 - config.stop_loss_pct)
        target_price = entry_price + config.take_profit_multiple * (entry_price - stop_price)
        open_trade = Trade(
            ticker=ticker,
            entry_date=next_date,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            exit_date=next_date,
            exit_price=entry_price,
            outcome="open",
        )

    if open_trade:
        # ... (Lógica de cierre de operación por "timeout" - no cambia) ...
        remaining = usable.loc[open_trade.entry_date :]
        if remaining.empty:
            remaining = usable.iloc[[-1]]
        last_row = remaining.iloc[-1]
        trades.append(
            Trade(
                ticker=open_trade.ticker,
                entry_date=open_trade.entry_date,
                entry_price=open_trade.entry_price,
                stop_price=open_trade.stop_price,
                target_price=open_trade.target_price,
                exit_date=last_row.name,
                exit_price=float(last_row["close"]),
                outcome="timeout",
            )
        )

    return trades


def run_backtest(
    histories: dict[str, pd.DataFrame],
    earnings_dates: dict[str, Optional[pd.Timestamp]], # <-- MODIFICADO
    thresholds: Thresholds,
    config: BacktestConfig,
) -> list[Trade]:
    all_trades: list[Trade] = []
    # Usamos tqdm en el bucle principal de simulación
    for ticker, history in tqdm(histories.items(), desc="Simulando operaciones"):
        trades = simulate_ticker(
            ticker,
            history,
            earnings_dates.get(ticker), # <-- MODIFICADO
            thresholds,
            config,
        )
        if trades:
            logging.info("%s: %s operaciones simuladas", ticker, len(trades))
        all_trades.extend(trades)
    return all_trades


# ---------------------------------------------------------------------------
# Métricas estadísticas
# ---------------------------------------------------------------------------


def compute_statistics(trades: Iterable[Trade]) -> dict[str, float]:
    # ... (Esta función no cambia) ...
    trades = list(trades)
    if not trades:
        return {}

    r_values = [trade.r_multiple for trade in trades]
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r < 0]

    cumulative = []
    running_total = 0.0
    for value in r_values:
        running_total += value
        cumulative.append(running_total)

    peak = float("-inf")
    max_drawdown = 0.0
    for value in cumulative:
        peak = max(peak, value)
        drawdown = peak - value
        max_drawdown = max(max_drawdown, drawdown)

    stats = {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "average_r": mean(r_values),
        "profit_factor": (
            sum(wins) / abs(sum(losses)) if wins and losses else float("inf") if wins else 0.0
        ),
        "max_drawdown": max_drawdown,
    }
    if wins and losses:
        stats["avg_risk_reward"] = abs(mean(wins) / mean(losses))
    elif wins:
        stats["avg_risk_reward"] = float("inf")
    else:
        stats["avg_risk_reward"] = 0.0
    return stats


def trades_to_dataframe(trades: Iterable[Trade]) -> pd.DataFrame:
    # ... (Esta función no cambia) ...
    records = [
        {
            "Ticker": trade.ticker,
            "Entrada": trade.entry_date.date().isoformat(),
            "Salida": trade.exit_date.date().isoformat(),
            "Precio Entrada": trade.entry_price,
            "Precio Salida": trade.exit_price,
            "Stop": trade.stop_price,
            "Objetivo": trade.target_price,
            "Resultado (R)": trade.r_multiple,
            "Motivo": trade.outcome,
        }
        for trade in trades
    ]
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtester simplificado para el escáner de momentum",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Archivo con la lista de tickers a simular.",
    )
    # ... (El resto de esta función no cambia, EXCEPTO que eliminamos los args de fundamentales) ...
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Fecha inicial del backtest (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Fecha final del backtest (YYYY-MM-DD).")
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=DEFAULT_STOP_LOSS_PCT,
        help="Porcentaje de stop-loss respecto al precio de entrada (0.04 = 4%).",
    )
    parser.add_argument(
        "--take-profit-multiple",
        type=float,
        default=DEFAULT_PROFIT_MULTIPLE,
        help="Múltiplo de riesgo utilizado para calcular el objetivo (2.0 = 2R).",
    )
    parser.add_argument(
        "--output",
        default="backtest_resultados.csv",
        help="Fichero CSV donde guardar el registro de operaciones.",
    )
    # --- MODIFICADO: Eliminados los args de EPS/Revenue ---
    # parser.add_argument(
    #     "--min-eps-growth",
    #     ...
    # )
    # parser.add_argument(
    #     "--min-revenue-growth",
    #     ...
    # )
    parser.add_argument(
        "--min-momentum-3m",
        type=float,
        default=DEFAULT_MIN_MOMENTUM_3M,
        help="Revalorización mínima en 3 meses (decimal).",
    )
    parser.add_argument(
        "--min-adr",
        type=float,
        default=DEFAULT_MIN_ADR,
        help="ADR medio mínimo de los últimos 3 meses (decimal).",
    )
    parser.add_argument(
        "--max-week-volatility",
        type=float,
        default=DEFAULT_MAX_WEEKLY_VOL,
        help="Volatilidad máxima de la última semana (decimal).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ts = _ensure_timestamp(args.start_date)
    end_ts = _ensure_timestamp(args.end_date)
    if end_ts <= start_ts:
        raise ValueError("La fecha final debe ser posterior a la fecha inicial")

    config = BacktestConfig(
        start_date=start_ts,
        end_date=end_ts,
        stop_loss_pct=max(args.stop_loss_pct, 0.001),
        take_profit_multiple=max(args.take_profit_multiple, 0.1),
    )

    # --- MODIFICADO: Thresholds ya no incluye EPS/Revenue ---
    thresholds = Thresholds(
        min_eps_growth=0.0, # Ignorado
        min_revenue_growth=0.0, # Ignorado
        min_momentum_3m=max(args.min_momentum_3m, 0.0),
        max_week_volatility=max(args.max_week_volatility, 0.0),
        min_adr=max(args.min_adr, 0.0),
    )

    tickers = load_tickers(Path(args.input))
    
    # --- LÓGICA DE CACHÉ ---
    logging.info("Iniciando Fase 1: Carga de datos (usando caché si existe)")
    CACHE_DIR.mkdir(exist_ok=True) # Crea la carpeta .cache/ si no existe
    
    client = YahooFinanceEUClient()
    histories: dict[str, pd.DataFrame] = {}
    
    # --- MODIFICADO: Solo guardamos la fecha de ER, no el snapshot ---
    earnings_dates: dict[str, Optional[pd.Timestamp]] = {}
    
    margin = pd.Timedelta(days=200) # Margen para cálculos de momentum/ADR
    start_download = start_ts - margin 
    
    for ticker in tqdm(tickers, desc="Cargando datos de tickers"):
        summary_path = CACHE_DIR / f"{ticker}_summary.json"
        history_path = CACHE_DIR / f"{ticker}_history.csv"
        
        summary = None
        try:
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
            else:
                summary = client.fetch_summary(ticker)
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(summary, f)
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s: no se pudo descargar/cachear el resumen fundamental (%s)", ticker, exc)
            continue
        
        # --- MODIFICADO: Lógica de pre-filtrado eliminada ---
        # Solo extraemos la fecha de ER para el filtro de simulación
        try:
            last_earnings = extract_last_earnings_date(
                summary.get("calendarEvents"),
                summary.get("earningsHistory"),
                reference=end_ts.to_pydatetime(), # Usamos end_ts como referencia estática
            )
            earnings_dates[ticker] = pd.Timestamp(last_earnings, tz=timezone.utc) if last_earnings else None
        except Exception:
            earnings_dates[ticker] = None

        # --- MODIFICADO: Ahora SIEMPRE intentamos descargar el historial ---
        # (ya que el pre-filtro fundamental fue eliminado)
        try:
            if history_path.exists():
                history = pd.read_csv(history_path, index_col="date", parse_dates=True)
                if history.index.tz is None:
                    history.index = history.index.tz_localize(timezone.utc)
            else:
                history = download_history(client, ticker, start_download, end_ts)
                history.to_csv(history_path)
            
            histories[ticker] = history # <-- Añadido
            
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s: no se pudo descargar/cachear el histórico (%s)", ticker, exc)
            continue
            
    # --- FIN DE LA LÓGICA DE CACHÉ ---

    if not histories:
        logging.error("No se descargó ningún histórico válido. Abortando.")
        return

    logging.info("Iniciando Fase 2: Simulación de operaciones")
    trades = run_backtest(histories, earnings_dates, thresholds, config)
    stats = compute_statistics(trades)

    if not trades:
        logging.info("No se generaron operaciones con los parámetros actuales.")
        return

    df = trades_to_dataframe(trades)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logging.info("Registro de operaciones guardado en %s", args.output)

    logging.info("Resumen estadístico:")
    logging.info("- Operaciones: %s", stats.get("trades", 0))
    logging.info("- Win Rate: %.2f%%", stats.get("win_rate", 0.0) * 100)
    logging.info("- Promedio R: %.2f", stats.get("average_r", 0.0))
    logging.info("- Profit Factor: %.2f", stats.get("profit_factor", 0.0))
    logging.info("- Máx. Drawdown (R): %.2f", stats.get("max_drawdown", 0.0))
    logging.info("- Ratio Riesgo/Beneficio Promedio: %.2f", stats.get("avg_risk_reward", 0.0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()