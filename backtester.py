#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import timezone, datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

import pandas as pd

# Asumimos que 'a.py' está en la misma carpeta
from a import (
    DEFAULT_INPUT,
    DEFAULT_MIN_EPS_GROWTH,
    DEFAULT_MIN_REVENUE_GROWTH,
    DEFAULT_MAX_WEEKLY_VOL,
    DEFAULT_MIN_ADR,
    DEFAULT_MIN_MOMENTUM_3M,
    MAX_EARNINGS_AGE_DAYS,
    Thresholds,
    YahooFinanceEUClient,
    YAHOO_CHART_ENDPOINT,
    extract_eps_growth,
    extract_last_earnings_date,
    load_tickers,
    safe_float,
)


# ---------------------------------------------------------------------------
# Configuración y estructuras de datos
# ---------------------------------------------------------------------------


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_STOP_LOSS_PCT = 0.04  # 4%
DEFAULT_PROFIT_MULTIPLE = 2.0  # 2R


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


@dataclass
class FundamentalSnapshot:
    eps_growth: float
    revenue_growth: float
    last_earnings_date: pd.Timestamp


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

    # Yahoo requiere epoch segundos (UTC) y el final es exclusivo.
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


def build_fundamental_snapshot(
    summary: dict,
    *,
    reference: pd.Timestamp,
) -> Optional[FundamentalSnapshot]:
    if not isinstance(summary, dict):
        return None

    earnings = summary.get("earningsTrend")
    financial = summary.get("financialData")
    calendar = summary.get("calendarEvents")
    earnings_history = summary.get("earningsHistory")

    eps_growth = extract_eps_growth(earnings)
    revenue_growth = safe_float(financial.get("revenueGrowth") if isinstance(financial, dict) else None)
    if eps_growth is None or revenue_growth is None:
        return None

    last_earnings = extract_last_earnings_date(
        calendar,
        earnings_history,
        reference=reference.to_pydatetime(),
    )
    if last_earnings is None:
        return None

    last_ts = pd.Timestamp(last_earnings)
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")

    return FundamentalSnapshot(
        eps_growth=float(eps_growth),
        revenue_growth=float(revenue_growth),
        last_earnings_date=last_ts,
    )


# ---------------------------------------------------------------------------
# Métricas del escáner
# ---------------------------------------------------------------------------


def compute_momentum(history: pd.DataFrame, window: int = 63) -> Optional[float]:
    if len(history) < window + 1:
        return None
    recent = history["close"].iloc[-1]
    past = history["close"].iloc[-1 - window]
    if past <= 0:
        return None
    return (recent - past) / past


def compute_average_adr(history: pd.DataFrame, window: int = 63) -> Optional[float]:
    if len(history) < window:
        return None
    adr = (history["high"] - history["low"]) / history["close"]
    adr = adr.replace([pd.NA, pd.NaT], pd.NA).dropna()
    if len(adr) < window:
        return None
    return float(adr.tail(window).mean())


def compute_weekly_volatility(history: pd.DataFrame, window: int = 5) -> Optional[float]:
    if len(history) < window:
        return None
    segment = history.tail(window)
    reference = segment["open"].iloc[0]
    if reference <= 0:
        return None
    weekly_range = segment["high"].max() - segment["low"].min()
    return float(weekly_range / reference)


def passes_scanner(history: pd.DataFrame, thresholds: Thresholds) -> bool:
    momentum = compute_momentum(history)
    adr = compute_average_adr(history)
    week_vol = compute_weekly_volatility(history)
    if momentum is None or adr is None or week_vol is None:
        return False
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
    fundamentals: Optional[FundamentalSnapshot],
    thresholds: Thresholds,
    config: BacktestConfig,
) -> list[Trade]:
    trades: list[Trade] = []
    if fundamentals is None:
        return trades

    open_trade: Optional[Trade] = None
    # Reducimos el histórico al rango de fechas solicitado más un margen para cálculos.
    usable = history.loc[: config.end_date]
    if usable.empty:
        return trades

    dates = list(usable.index)
    total_days = len(dates)

    for idx, current_date in enumerate(dates):
        row = usable.iloc[idx]

        if open_trade and current_date >= open_trade.entry_date:
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

        if fundamentals.eps_growth < thresholds.min_eps_growth:
            continue
        if fundamentals.revenue_growth < thresholds.min_revenue_growth:
            continue

        days_since_earnings = (current_ts - fundamentals.last_earnings_date).days
        if days_since_earnings < 0 or days_since_earnings >= MAX_EARNINGS_AGE_DAYS:
            continue

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
        # Si la operación sigue abierta al finalizar el rango, cerramos al último cierre disponible.
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
    fundamentals_map: dict[str, FundamentalSnapshot],
    thresholds: Thresholds,
    config: BacktestConfig,
) -> list[Trade]:
    all_trades: list[Trade] = []
    for ticker, history in histories.items():
        trades = simulate_ticker(
            ticker,
            history,
            fundamentals_map.get(ticker),
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
    parser.add_argument(
        "--min-eps-growth",
        type=float,
        default=DEFAULT_MIN_EPS_GROWTH,
        help="Crecimiento mínimo de EPS trimestral (decimal, 0.20 = 20%).",
    )
    parser.add_argument(
        "--min-revenue-growth",
        type=float,
        default=DEFAULT_MIN_REVENUE_GROWTH,
        help="Crecimiento mínimo de ingresos (decimal).",
    )
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

    thresholds = Thresholds(
        min_eps_growth=max(args.min_eps_growth, 0.0),
        min_revenue_growth=max(args.min_revenue_growth, 0.0),
        min_momentum_3m=max(args.min_momentum_3m, 0.0),
        max_week_volatility=max(args.max_week_volatility, 0.0),
        min_adr=max(args.min_adr, 0.0),
    )

    tickers = load_tickers(Path(args.input))
    logging.info("Descargando históricos diarios para %s tickers", len(tickers))

    client = YahooFinanceEUClient()
    histories: dict[str, pd.DataFrame] = {}
    fundamentals_map: dict[str, FundamentalSnapshot] = {}
    margin = pd.Timedelta(days=200)
    start_download = start_ts - margin
    for ticker in tickers:
        try:
            summary = client.fetch_summary(ticker)
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s: no se pudo descargar el resumen fundamental (%s)", ticker, exc)
            continue

        fundamentals = build_fundamental_snapshot(summary, reference=end_ts)
        if fundamentals is None:
            logging.debug("%s: sin datos fundamentales suficientes", ticker)
            continue
        if (
            fundamentals.eps_growth < thresholds.min_eps_growth
            or fundamentals.revenue_growth < thresholds.min_revenue_growth
        ):
            logging.debug("%s: no supera los filtros de crecimiento", ticker)
            continue

        try:
            history = download_history(client, ticker, start_download, end_ts)
        except Exception as exc:  # noqa: BLE001
            logging.warning("%s: no se pudo descargar el histórico (%s)", ticker, exc)
            continue
        histories[ticker] = history
        fundamentals_map[ticker] = fundamentals

    if not histories:
        logging.error("No se descargó ningún histórico válido. Abortando.")
        return

    trades = run_backtest(histories, fundamentals_map, thresholds, config)
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