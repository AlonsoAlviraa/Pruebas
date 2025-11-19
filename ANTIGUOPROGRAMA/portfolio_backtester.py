#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulador de cartera basado en las operaciones generadas por ``backtester.py``.

El script toma las mismas opciones que el backtester original para obtener las
operaciones, pero añade una capa de gestión de capital. Utiliza un capital
inicial y un porcentaje fijo de riesgo por operación para dimensionar las
posiciones y construye una curva de patrimonio diaria. Al finalizar, muestra
métricas como CAGR y Ratio de Sharpe.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from a import (
    DEFAULT_INPUT,
    DEFAULT_MAX_WEEKLY_VOL,
    DEFAULT_MIN_ADR,
    DEFAULT_MIN_EPS_GROWTH,
    DEFAULT_MIN_MOMENTUM_3M,
    DEFAULT_MIN_REVENUE_GROWTH,
    Thresholds,
    load_tickers,
)
from backtester import (
    CACHE_DIR,
    DEFAULT_EMA_PERIOD,
    DEFAULT_EMA_TOLERANCE,
    DEFAULT_WEEKLY_VOL_STOP_MULTIPLE,
    BacktestConfig,
    Trade,
    _ensure_timestamp,
    load_fundamental_snapshots,
    load_history_from_cache,
    run_backtest,
)

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_RISK_PER_TRADE = 0.01


@dataclass
class PortfolioPosition:
    """Representa una posición abierta en la simulación de cartera."""

    trade: Trade
    shares: int
    entry_cost: float
    risk_per_share: float
    risk_capital: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price


def _prepare_histories(
    tickers: Iterable[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Iterable]]:
    """Carga los datos cacheados de historial y fundamentales."""
    histories: Dict[str, pd.DataFrame] = {}
    fundamentals: Dict[str, Iterable] = {}

    for ticker in tickers:
        if ticker.upper() == "QQQ":
            # El índice se usa únicamente como referencia en el backtester.
            continue

        history_path = CACHE_DIR / f"{ticker}_history.csv"
        fundamentals_path = CACHE_DIR / f"{ticker}_fundamentals.csv"

        if not history_path.exists():
            logging.debug(
                "%s: historial no encontrado en caché. Ejecuta downloader.py primero.",
                ticker,
            )
            continue

        try:
            history = load_history_from_cache(history_path)
        except Exception as exc:  # noqa: BLE001
            logging.debug("%s: no se pudo cargar el histórico (%s)", ticker, exc)
            continue

        histories[ticker] = history
        fundamentals[ticker] = load_fundamental_snapshots(fundamentals_path)

    return histories, fundamentals


def _build_price_cache(
    histories: Dict[str, pd.DataFrame],
    relevant_dates: pd.DatetimeIndex,
) -> Dict[str, pd.Series]:
    """Crea un lookup rápido de precios de cierre alineados al índice de fechas."""
    cache: Dict[str, pd.Series] = {}
    for ticker, history in histories.items():
        closes = history["close"].sort_index()
        aligned = closes.reindex(relevant_dates, method="ffill")
        cache[ticker] = aligned
    return cache


def _gather_trading_dates(
    histories: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Obtiene un índice único de todas las sesiones de mercado en el rango."""
    dates: set[pd.Timestamp] = set()
    for history in histories.values():
        window = history.loc[(history.index >= start) & (history.index <= end)]
        dates.update(window.index)
    if not dates:
        return pd.DatetimeIndex([], tz=start.tz)
    ordered = sorted(dates)
    return pd.DatetimeIndex(ordered)


def simulate_portfolio(
    trades: List[Trade],
    histories: Dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    risk_per_trade: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Ejecuta una simulación de cartera con gestión de capital (Fixed Fractional).
    """
    if not trades:
        return pd.DataFrame(columns=["cash", "equity"]), {}

    trading_dates = _gather_trading_dates(histories, start, end)
    if trading_dates.empty:
        raise RuntimeError(
            "No se encontraron sesiones de mercado en el rango solicitado."
        )

    price_cache = _build_price_cache(histories, trading_dates)

    entries_by_date: defaultdict[pd.Timestamp, List[Trade]] = defaultdict(list)
    exits_by_date: defaultdict[pd.Timestamp, List[Trade]] = defaultdict(list)
    for trade in trades:
        entries_by_date[trade.entry_date].append(trade)
        exits_by_date[trade.exit_date].append(trade)

    positions: Dict[Tuple[str, pd.Timestamp], PortfolioPosition] = {}
    cash = float(initial_capital)
    equity_records: List[Tuple[pd.Timestamp, float, float]] = []

    realized_pnl: List[float] = []

    for current_date in trading_dates:
        # Valor de la cartera antes de nuevas entradas (último precio conocido).
        equity_for_risk = cash
        for position in positions.values():
            equity_for_risk += position.market_value

        # Procesar nuevas entradas al inicio de la sesión.
        for trade in entries_by_date.get(current_date, []):
            risk_per_share = trade.risk_amount
            if risk_per_share <= 0:
                logging.debug(
                    "[%s] Operación ignorada por riesgo negativo o nulo.",
                    trade.ticker,
                )
                continue

            # Preservar el capital actual (incluyendo posiciones abiertas).
            risk_budget = equity_for_risk * risk_per_trade
            if risk_budget <= 0:
                continue

            theoretical_shares = int(risk_budget // risk_per_share)
            if theoretical_shares <= 0:
                continue

            affordable_shares = int(cash // trade.entry_price)
            shares = min(theoretical_shares, affordable_shares)
            if shares <= 0:
                continue

            cost = shares * trade.entry_price
            cash -= cost

            position_key = (trade.ticker, trade.entry_date)
            positions[position_key] = PortfolioPosition(
                trade=trade,
                shares=shares,
                entry_cost=cost,
                risk_per_share=risk_per_share,
                risk_capital=shares * risk_per_share,
                last_price=trade.entry_price,
            )
            
            # Recalcular el equity para la siguiente entrada del día
            equity_for_risk = cash + sum(pos.market_value for pos in positions.values())

        # Procesar salidas al cierre de la sesión.
        for trade in exits_by_date.get(current_date, []):
            position_key = (trade.ticker, trade.entry_date)
            position = positions.pop(position_key, None)
            if position is None:
                continue

            sale_value = position.shares * trade.exit_price
            cash += sale_value
            realized_pnl.append(sale_value - position.entry_cost)

        # Valorar las posiciones restantes con el cierre del día.
        equity = cash
        for key, position in positions.items():
            close_series = price_cache.get(position.trade.ticker)
            if close_series is None:
                price = position.last_price
            else:
                price = close_series.get(current_date, position.last_price)
                if pd.isna(price):
                    price = position.last_price
            position.last_price = float(price)
            equity += position.market_value

        equity_records.append((current_date, cash, equity))

    equity_curve = pd.DataFrame(
        equity_records, columns=["date", "cash", "equity"]
    ).set_index("date")

    # Calcular Métricas de Cartera
    metrics: Dict[str, float] = {}
    if not equity_curve.empty:
        metrics["final_equity"] = float(equity_curve["equity"].iloc[-1])
        metrics["total_return"] = (
            metrics["final_equity"] / float(initial_capital) - 1.0
        )

        delta_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = delta_days / 365.25
        if years > 0:
            metrics["cagr"] = (metrics["final_equity"] / float(initial_capital)) ** (1 / years) - 1
        else:
            metrics["cagr"] = float("nan")

        daily_returns = equity_curve["equity"].pct_change().dropna()
        if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
            metrics["sharpe"] = (
                daily_returns.mean() / daily_returns.std(ddof=0)
            ) * np.sqrt(252) # Asumiendo 252 días de trading
        else:
            metrics["sharpe"] = float("nan")

        if realized_pnl:
            metrics["realized_pnl"] = float(sum(realized_pnl))

    return equity_curve, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulador de cartera basado en los resultados del backtester",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Archivo con la lista de tickers a procesar.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Fecha inicial del backtest (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Fecha final del backtest (YYYY-MM-DD).")
    
    # Argumentos del Backtester (para generar las operaciones)
    parser.add_argument(
        "--weekly-vol-multiple",
        "--adr-stop-multiple",
        dest="vol_stop_multiple",
        type=float,
        default=DEFAULT_WEEKLY_VOL_STOP_MULTIPLE,
        help="Multiplicador de la volatilidad semanal para el stop inicial.",
    )
    parser.add_argument(
        "--ema-period",
        type=int,
        default=DEFAULT_EMA_PERIOD,
        help="Periodo de la EMA utilizada como stop dinámico.",
    )
    parser.add_argument(
        "--ema-tolerance",
        type=float,
        default=DEFAULT_EMA_TOLERANCE,
        help="Tolerancia (en tanto por uno) bajo la EMA antes de ejecutar la salida.",
    )
    parser.add_argument("--min-eps-growth", type=float, default=DEFAULT_MIN_EPS_GROWTH)
    parser.add_argument("--min-revenue-growth", type=float, default=DEFAULT_MIN_REVENUE_GROWTH)
    parser.add_argument("--min-momentum-3m", type=float, default=DEFAULT_MIN_MOMENTUM_3M)
    parser.add_argument("--min-adr", type=float, default=DEFAULT_MIN_ADR)
    parser.add_argument("--max-week-volatility", type=float, default=DEFAULT_MAX_WEEKLY_VOL)
    
    # --- INICIO MODIFICACIÓN 1: Añadir argumentos faltantes de backtester.py ---
    # (Usamos los valores default definidos en backtester.py)
    parser.add_argument(
        "--partial-fraction",
        type=float,
        default=0.5,  # Valor de DEFAULT_PARTIAL_PROFIT_FRACTION
        help=(
            "Fracción de la posición a tomar como ganancia parcial tras el "
            "período inicial (decimal, p.ej. 0.5 = 50%)."
        ),
    )
    parser.add_argument(
        "--partial-delay",
        type=int,
        default=3,  # Valor de DEFAULT_PARTIAL_PROFIT_DELAY
        help="Número mínimo de sesiones (barra diaria) antes de ejecutar la toma parcial.",
    )
    parser.add_argument(
        "--max-risk-adr",
        type=float,
        default=0.6,  # Valor de DEFAULT_MAX_RISK_TO_ADR_RATIO
        help=(
            "Máximo cociente permitido entre el riesgo inicial (% sobre la entrada) "
            "y el ADR de 3 meses. Valores < 1 fuerzan riesgos inferiores al ADR."
        ),
    )
    # --- FIN MODIFICACIÓN 1 ---

    # Argumentos del Simulador de Cartera
    parser.add_argument(
        "--capital-inicial",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help="Capital inicial de la cartera.",
    )
    parser.add_argument(
        "--riesgo-por-trade",
        type=float,
        default=DEFAULT_RISK_PER_TRADE,
        help="Porcentaje de capital arriesgado por operación (decimal).",
    )
    parser.add_argument(
        "--equity-output",
        default="equity_curve.csv",
        help="Archivo CSV donde guardar la curva de patrimonio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_ts = _ensure_timestamp(args.start_date)
    end_ts = _ensure_timestamp(args.end_date)
    if end_ts <= start_ts:
        raise ValueError("La fecha final debe ser posterior a la fecha inicial")

    # Configuración para el Backtester (generador de señales)
    
    # --- INICIO MODIFICACIÓN 2: Pasar los nuevos argumentos a BacktestConfig ---
    config = BacktestConfig(
        start_date=start_ts,
        end_date=end_ts,
        vol_stop_multiple=max(args.vol_stop_multiple, 0.1),
        ema_trail_period=max(int(args.ema_period), 1),
        ema_tolerance=max(args.ema_tolerance, 0.0),
        # --- Líneas añadidas ---
        partial_profit_fraction=max(0.0, min(1.0, args.partial_fraction)),
        partial_profit_delay=max(int(args.partial_delay), 1),
        max_risk_to_adr=max(args.max_risk_adr, 0.01),
    )
    # --- FIN MODIFICACIÓN 2 ---

    # Filtros para el Backtester
    thresholds = Thresholds(
        min_eps_growth=args.min_eps_growth,
        min_revenue_growth=args.min_revenue_growth,
        min_momentum_3m=max(args.min_momentum_3m, 0.0),
        max_week_volatility=max(args.max_week_volatility, 0.0),
        min_adr=max(args.min_adr, 0.0),
    )

    tickers = load_tickers(Path(args.input))
    CACHE_DIR.mkdir(exist_ok=True)

    # --- Fase 1: Cargar datos (igual que el backtester) ---
    logging.info("Iniciando Fase 1: Carga de datos (usando caché si existe)")
    histories, fundamentals = _prepare_histories(tickers)
    
    if not histories:
        logging.error(
            "No se encontraron históricos cacheados. Ejecuta downloader.py antes de simular."
        )
        return

    index_path = CACHE_DIR / "QQQ_history.csv"
    if index_path.exists():
        index_history = load_history_from_cache(index_path)
        index_history = index_history.sort_index()
        index_history["ma50"]  = index_history["close"].rolling(window=50).mean()
        index_history["ma200"] = index_history["close"].rolling(window=200).mean()
    else:
        index_history = None
        logging.warning(
            "Datos del índice QQQ no encontrados. Los filtros de régimen podrían no aplicarse."
        )

    # --- Fase 2: Generar Señales (correr el backtest) ---
    logging.info("Iniciando Fase 2: Generación de señales (ejecutando backtest)")
    trades = run_backtest(histories, fundamentals, thresholds, config, index_history)
    if not trades:
        logging.info("No se generaron operaciones con los parámetros actuales.")
        return
    logging.info(f"Se generaron {len(trades)} operaciones teóricas.")

    # --- Fase 3: Simular Cartera ---
    logging.info("Iniciando Fase 3: Simulación de cartera con gestión de capital")
    equity_curve, metrics = simulate_portfolio(
        trades,
        histories,
        start=start_ts,
        end=end_ts,
        initial_capital=args.capital_inicial,
        risk_per_trade=max(args.riesgo_por_trade, 0.0),
    )

    equity_curve.to_csv(args.equity_output, encoding="utf-8-sig")
    logging.info("Curva de patrimonio guardada en %s", args.equity_output)

    if metrics:
        logging.info("--- Resumen de la Cartera ---")
        logging.info("Capital final: %.2f", metrics.get("final_equity", 0.0))
        logging.info("Rentabilidad total: %.2f%%", metrics.get("total_return", 0.0) * 100)
        logging.info("CAGR: %.2f%%", metrics.get("cagr", float("nan")) * 100)
        logging.info("Sharpe Ratio: %.2f", metrics.get("sharpe", float("nan")))
        if "realized_pnl" in metrics:
            logging.info("Beneficio/Pérdida realizada: %.2f", metrics["realized_pnl"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()