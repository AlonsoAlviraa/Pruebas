#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Motor de Backtesting para Estrategias de Momentum.

Este script ejecuta una simulación de backtesting 'point-in-time' sobre
un conjunto de tickers, basándose en datos históricos y fundamentales cacheados.

Versión: 5.0 (Breakout Patterns + Risk Management)
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional, Sequence

import pandas as pd
from tqdm import tqdm

# Asumimos que 'a.py' está en la misma carpeta
from a import (
    DEFAULT_INPUT,
    DEFAULT_MAX_WEEKLY_VOL,
    DEFAULT_MIN_ADR,
    DEFAULT_MIN_EPS_GROWTH,
    DEFAULT_MIN_MOMENTUM_3M,
    DEFAULT_MIN_REVENUE_GROWTH,
    MAX_EARNINGS_AGE_DAYS,
    Thresholds,
    YahooFinanceEUClient,
    YAHOO_CHART_ENDPOINT,
    load_tickers,
)


# ---------------------------------------------------------------------------
# Configuración y estructuras de datos
# ---------------------------------------------------------------------------


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_WEEKLY_VOL_STOP_MULTIPLE = 2.0  # Mantener por compatibilidad con CLI
DEFAULT_EMA_PERIOD = 10
DEFAULT_EMA_TOLERANCE = 0.0
DEFAULT_PARTIAL_PROFIT_FRACTION = 0.5
DEFAULT_PARTIAL_PROFIT_DELAY = 3
DEFAULT_MAX_RISK_TO_ADR_RATIO = 0.6
CACHE_DIR = Path(".cache")


@dataclass
class BacktestConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    vol_stop_multiple: float
    ema_trail_period: int
    ema_tolerance: float
    partial_profit_fraction: float
    partial_profit_delay: int
    max_risk_to_adr: float


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    # target_price ha sido eliminado
    exit_date: pd.Timestamp
    exit_price: float
    outcome: str
    partial_exit_date: Optional[pd.Timestamp] = None
    partial_exit_price: Optional[float] = None
    partial_fraction: float = 0.0
    partial_reason: Optional[str] = None

    @property
    def risk_amount(self) -> float:
        return self.entry_price - self.stop_price

    @property
    def r_multiple(self) -> float:
        risk = self.risk_amount
        if risk <= 0:
            return 0.0
        realized = 0.0
        remaining_fraction = 1.0
        if self.partial_exit_price is not None and self.partial_fraction > 0:
            fraction = max(0.0, min(1.0, self.partial_fraction))
            realized += fraction * ((self.partial_exit_price - self.entry_price) / risk)
            remaining_fraction -= fraction
        realized += remaining_fraction * ((self.exit_price - self.entry_price) / risk)
        return realized


@dataclass(frozen=True)
class SetupSignal:
    kind: str
    trigger_price: float
    base_start: pd.Timestamp
    base_end: pd.Timestamp
    metadata: dict[str, float]


@dataclass(frozen=True)
class FundamentalSnapshot:
    available_at: pd.Timestamp
    eps: float
    revenue: float
    eps_growth: float
    revenue_growth: float


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


def load_history_from_cache(filepath: Path) -> pd.DataFrame:
    """Carga un historial desde un CSV cacheado."""
    try:
        history = pd.read_csv(
            filepath,
            parse_dates=["date"],
            index_col="date",
            dtype={
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": "Int64",
            },
        )
        if not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("La columna 'date' no se pudo parsear como índice.")
        history.index = history.index.map(_ensure_timestamp)
        return history
    except Exception as e:
        logging.error("Error al cargar %s: %s", filepath, e)
        return pd.DataFrame()


def load_fundamental_snapshots(filepath: Path) -> Sequence[FundamentalSnapshot]:
    """Carga los snapshots fundamentales desde un CSV cacheado."""
    try:
        df = pd.read_csv(filepath, parse_dates=["available_at"])
        snapshots = [
            FundamentalSnapshot(
                available_at=_ensure_timestamp(row["available_at"]),
                eps=float(row["eps"]),
                revenue=float(row["revenue"]),
                eps_growth=float(row["eps_growth"]),
                revenue_growth=float(row["revenue_growth"]),
            )
            for _, row in df.iterrows()
            if pd.notna(row["eps_growth"]) and pd.notna(row["revenue_growth"])
        ]
        # Ordenar por fecha de disponibilidad para búsqueda eficiente
        return sorted(snapshots, key=lambda s: s.available_at)
    except Exception as e:
        logging.error("Error al cargar %s: %s", filepath, e)
        return []


def select_snapshot(
    snapshots: Sequence[FundamentalSnapshot],
    current_date: pd.Timestamp,
    *,
    dates: Optional[list[pd.Timestamp]] = None,
) -> Optional[FundamentalSnapshot]:
    """
    Selecciona el snapshot fundamental más reciente disponible
    en 'current_date' (búsqueda binaria).
    """
    if not snapshots:
        return None

    # Si no se proveen fechas pre-cacheadas, se extraen.
    if dates is None:
        dates = [s.available_at for s in snapshots]

    # bisect_right encuentra el punto de inserción
    # El snapshot relevante es el que está justo antes (idx - 1)
    idx = pd.Timestamp.searchsorted(dates, current_date, side="right")
    if idx == 0:
        return None  # No hay datos fundamentales tan antiguos

    return snapshots[idx - 1]


# ---------------------------------------------------------------------------
# Funciones de cálculo técnico (Point-in-Time)
# ---------------------------------------------------------------------------


def compute_momentum(history: pd.DataFrame, days: int = 63) -> Optional[float]:
    """Calcula la revalorización en los últimos N días (aprox 3 meses)."""
    if len(history) < days:
        return None
    past_price = history["close"].iloc[-days]
    current_price = history["close"].iloc[-1]
    if pd.isna(past_price) or past_price <= 0:
        return None
    return (current_price - past_price) / past_price


def compute_average_adr(history: pd.DataFrame, days: int = 63) -> Optional[float]:
    """Calcula el Average Daily Range (ADR) % de los últimos N días."""
    if len(history) < days:
        return None
    recent_history = history.iloc[-days:]
    if recent_history["low"].min() <= 0:
        return None
    
    # (High - Low) / Close
    adr_pct = (recent_history["high"] - recent_history["low"]) / recent_history["close"]
    
    mean_adr = adr_pct.mean()
    if pd.isna(mean_adr):
        return None
    return float(mean_adr)


def compute_weekly_volatility(history: pd.DataFrame, days: int = 5) -> Optional[float]:
    """
    Calcula la volatilidad de N días (rango / cierre).
    Usado para el stop inicial.
    """
    if len(history) < days:
        return None
    recent_history = history.iloc[-days:]
    
    if recent_history.empty:
        return None
    
    high = recent_history["high"].max()
    low = recent_history["low"].min()
    close = recent_history["close"].iloc[-1]
    
    if pd.isna(high) or pd.isna(low) or pd.isna(close) or close <= 0:
        return None
        
    return (high - low) / close


def compute_average_weekly_range_pct(history: pd.DataFrame, window_weeks: int = 10) -> Optional[float]:
    """
    Calcula el Average Weekly Range (AWR) % de las últimas N semanas.
    NOTA: Esta función mira el historial completo, no es point-in-time
    para el escáner, sino para el backtest (donde se usa todo el historial).
    """
    
    if len(history) < window_weeks * 5:
        return None
    
    # Resamplear a velas semanales (Lunes-Viernes)
    # Usamos 'W-FRI' para que la semana termine el viernes
    weekly_data = history['close'].resample('W-FRI').last()
    weekly_data = pd.DataFrame(weekly_data)
    weekly_data['high'] = history['high'].resample('W-FRI').max()
    weekly_data['low'] = history['low'].resample('W-FRI').min()
    
    if weekly_data.empty or len(weekly_data) < window_weeks:
        return None

    # Calcular AWR% (Average Weekly Range %)
    weekly_data['awr_pct'] = (weekly_data['high'] - weekly_data['low']) / weekly_data['close']
    
    # Calcular la media de las últimas 'window_weeks'
    awr_mean = weekly_data['awr_pct'].rolling(window=window_weeks).mean().iloc[-1]
    
    if pd.isna(awr_mean):
        return None
    return float(awr_mean)


def _percentage_change(current: float, previous: float) -> Optional[float]:
    if previous <= 0:
        return None
    return (current - previous) / previous


def detect_continuation_breakout(
    history: pd.DataFrame,
    *,
    min_base_days: int = 10,
    max_base_days: int = 45,
    min_prior_run: float = 0.3,
    max_base_depth: float = 0.25,
) -> Optional[SetupSignal]:
    if len(history) < max_base_days + 25:
        return None

    recent = history.iloc[-(max_base_days + 1) :]
    if len(recent) <= min_base_days:
        return None

    pivot_row = recent.iloc[-1]
    base = recent.iloc[:-1]
    base_len = len(base)
    if base_len < min_base_days:
        return None

    base_high = float(base["high"].max())
    base_low = float(base["low"].min())
    if base_high <= 0 or base_low <= 0:
        return None

    pivot_high = float(pivot_row["high"])
    if pivot_high <= base_high:
        return None

    base_depth = (base_high - base_low) / base_high
    if base_depth > max_base_depth:
        return None

    midpoint = base_len // 2 or 1
    left = base.iloc[:midpoint]
    right = base.iloc[midpoint:]
    left_range = float(left["high"].max() - left["low"].min()) if not left.empty else float("inf")
    right_range = float(right["high"].max() - right["low"].min()) if not right.empty else float("inf")
    if right_range >= left_range:
        return None

    lows = base["low"].rolling(window=3, min_periods=1).min()
    if lows.isna().any():
        return None
    if float(lows.iloc[-1]) <= float(lows.iloc[midpoint // 2]):
        return None

    prior_window = history.iloc[-(base_len + 60) : -base_len]
    if len(prior_window) < 20:
        return None
    prior_start = float(prior_window["close"].iloc[0])
    prior_end = float(prior_window["close"].iloc[-1])
    prior_run = _percentage_change(prior_end, prior_start)
    if prior_run is None or prior_run < min_prior_run:
        return None

    return SetupSignal(
        kind="continuation",
        trigger_price=pivot_high,
        base_start=base.index[0],
        base_end=base.index[-1],
        metadata={
            "base_depth": base_depth,
            "prior_run": prior_run,
            "base_length": float(base_len),
        },
    )


def detect_episode_pivot(
    history: pd.DataFrame,
    *,
    min_gap: float = 0.1,
    min_base_days: int = 30,
) -> Optional[SetupSignal]:
    if len(history) < min_base_days + 2:
        return None

    today = history.iloc[-1]
    yesterday = history.iloc[-2]

    prev_close = float(yesterday["close"])
    today_open = float(today["open"])
    if prev_close <= 0:
        return None
    gap = (today_open - prev_close) / prev_close
    if gap < min_gap:
        return None

    base = history.iloc[-(min_base_days + 1) : -1]
    if base.empty:
        return None
    base_high = float(base["high"].max())
    base_low = float(base["low"].min())
    if base_high <= 0 or base_low <= 0:
        return None

    today_low = float(today["low"])
    today_high = float(today["high"])
    if today_low <= base_high:
        return None
    if today_high <= base_high:
        return None

    return SetupSignal(
        kind="ep",
        trigger_price=today_high,
        base_start=base.index[0],
        base_end=base.index[-1],
        metadata={
            "gap_pct": gap,
            "base_low": base_low,
            "base_high": base_high,
        },
    )


def identify_setup(history: pd.DataFrame) -> Optional[SetupSignal]:
    signal = detect_episode_pivot(history)
    if signal is not None:
        return signal
    return detect_continuation_breakout(history)


def passes_scanner(history: pd.DataFrame, thresholds: Thresholds) -> bool:
    """Función de filtro técnico basada únicamente en el precio."""
    momentum = compute_momentum(history)
    adr = compute_average_adr(history)
    week_vol = compute_weekly_volatility(history) # Esta es la volatilidad de 1 semana
    
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
    snapshots: Sequence[FundamentalSnapshot],
    thresholds: Thresholds,
    config: BacktestConfig,
    index_history: Optional[pd.DataFrame],  # Recibe el historial del índice
) -> list[Trade]:
    trades: list[Trade] = []

    if not snapshots:
        return trades

    availability_dates = [snap.available_at for snap in snapshots]

    open_trade: Optional[Trade] = None
    open_state: Optional[dict[str, object]] = None

    working_history = history.copy().sort_index()

    aligned_index: Optional[pd.DataFrame] = None
    if index_history is not None:
        aligned_index = index_history.reindex(working_history.index, method="ffill")

    working_history["ema_fast"] = working_history["close"].ewm(
        span=10, adjust=False
    ).mean()
    working_history["ema_slow"] = working_history["close"].ewm(
        span=20, adjust=False
    ).mean()
    working_history["ema_trail"] = working_history["close"].ewm(
        span=config.ema_trail_period, adjust=False
    ).mean()
    working_history["ma50"] = working_history["close"].rolling(window=50).mean()
    working_history["ma200"] = working_history["close"].rolling(window=200).mean()
    
    if aligned_index is not None:
        index_close = aligned_index["close"].replace(0, pd.NA)
        working_history["rs_line"] = working_history["close"] / index_close
        working_history["rs_ma21"] = working_history["rs_line"].ewm(
            span=21, adjust=False
        ).mean()
        # El filtro de RS_63_high ha sido eliminado
    
    usable = working_history.loc[: config.end_date]
    if usable.empty:
        return trades

    dates = list(usable.index)
    total_days = len(dates)

    for idx, current_date in enumerate(dates):
        row = usable.iloc[idx]

        # --- FILTRO DE RÉGIMEN DE MERCADO (GLOBAL) ---
        market_now = None
        if index_history is not None:
            try:
                market_now = index_history.loc[current_date]
            except KeyError:
                if aligned_index is not None:
                    try:
                        market_now = aligned_index.loc[current_date]
                    except KeyError:
                        pass # No hay datos de mercado para este día

        market_bearish = index_history is not None
        if market_now is not None:
            ema_fast_idx = market_now.get("ema10")
            ema_slow_idx = market_now.get("ema20")
            if not pd.isna(ema_fast_idx) and not pd.isna(ema_slow_idx):
                try:
                    market_bearish = float(ema_fast_idx) < float(ema_slow_idx)
                except TypeError:
                    pass
        elif index_history is None:
            market_bearish = False
        # --- FIN FILTRO DE RÉGIMEN ---

        exit_trade: Optional[Trade] = None
        if open_trade and current_date >= open_trade.entry_date:
            if open_state is None:
                open_state = {
                    "dynamic_stop": open_trade.stop_price,
                    "bars": 0,
                    "partial_done": False,
                }

            bars = int(open_state.get("bars", 0)) + 1
            open_state["bars"] = bars

            dynamic_stop = float(open_state.get("dynamic_stop", open_trade.stop_price))
            low_price = float(row["low"])
            close_price = float(row["close"])
            ema_value = row.get("ema_trail", pd.NA)

            if low_price <= dynamic_stop:
                open_trade.exit_date = current_date
                open_trade.exit_price = dynamic_stop
                open_trade.outcome = (
                    "stop_breakeven" if open_state.get("partial_done") else "stop_initial"
                )
                exit_trade = open_trade
            else:
                partial_done = bool(open_state.get("partial_done"))
                if (
                    not partial_done
                    and bars >= max(1, config.partial_profit_delay)
                    and close_price > open_trade.entry_price
                ):
                    fraction = max(0.0, min(1.0, config.partial_profit_fraction))
                    if fraction > 0 and open_trade.partial_exit_date is None:
                        open_trade.partial_exit_date = current_date
                        open_trade.partial_exit_price = close_price
                        open_trade.partial_fraction = fraction
                        open_trade.partial_reason = "partial_take_profit"
                        open_state["partial_done"] = True
                        dynamic_stop = open_trade.entry_price
                        open_state["dynamic_stop"] = dynamic_stop

                if exit_trade is None and not pd.isna(ema_value):
                    ema_threshold = float(ema_value) * (1.0 - config.ema_tolerance)
                    if close_price < ema_threshold:
                        open_trade.exit_date = current_date
                        open_trade.exit_price = close_price
                        open_trade.outcome = f"ema{config.ema_trail_period}_close"
                        exit_trade = open_trade

            if exit_trade:
                logging.info(
                    "[%s] SALIDA: %s @ $%.2f (Motivo: %s, R=%.2f)",
                    exit_trade.ticker,
                    exit_trade.exit_date.date().isoformat(),
                    exit_trade.exit_price,
                    exit_trade.outcome,
                    exit_trade.r_multiple,
                )
                trades.append(exit_trade)
                open_trade = None
                open_state = None
                continue

        # No buscar entradas si el mercado está bajista
        if market_bearish:
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

        # --- Lógica de Filtro Point-in-Time ---
        snapshot = select_snapshot(snapshots, current_ts, dates=availability_dates)
        
        # --- MODIFICADO: Permitir tickers sin fundamentales ---
        if snapshot is not None:
            # 1. Filtro de "Reciente" (Evita ER inminente/antiguo)
            days_since_earnings = (current_ts - snapshot.available_at).days
            if days_since_earnings < 0 or days_since_earnings >= MAX_EARNINGS_AGE_DAYS:
                continue

            # 2. Filtro de Crecimiento Fundamental
            if snapshot.eps_growth < thresholds.min_eps_growth:
                continue
            if snapshot.revenue_growth < thresholds.min_revenue_growth:
                continue
        # Si 'snapshot' es None (sin fundamentales), el script continúa (como pediste)
        
        # --- Si todo pasa, evaluamos el setup y la ruptura ---
        history_until_now = working_history.loc[:current_date]
        if not passes_scanner(history_until_now, thresholds):
            continue

        row_now = working_history.loc[current_date]
        ma50_value = row_now.get("ma50")
        ma200_value = row_now.get("ma200")
        ema_fast = row_now.get("ema_fast")
        ema_slow = row_now.get("ema_slow")
        if (
            pd.isna(ma50_value)
            or pd.isna(ma200_value)
            or pd.isna(ema_fast)
            or pd.isna(ema_slow)
        ):
            continue
        if float(row_now["close"]) < float(ma50_value):
            continue
        if float(ma50_value) < float(ma200_value):
            continue
        if float(row_now["close"]) < float(ema_fast):
            continue
        if float(ema_fast) < float(ema_slow):
            continue

        if index_history is not None:
            rs_line = row_now.get("rs_line")
            rs_ma21 = row_now.get("rs_ma21")
            if pd.isna(rs_line) or pd.isna(rs_ma21):
                continue
            if float(rs_line) < float(rs_ma21):
                continue

        setup_signal = identify_setup(history_until_now)
        if setup_signal is None:
            continue

        next_row = usable.iloc[idx + 1]
        if pd.isna(next_row["open"]) or pd.isna(next_row["high"]) or pd.isna(next_row["low"]):
            continue

        trigger_price = float(setup_signal.trigger_price)
        next_high = float(next_row["high"])
        if next_high < trigger_price:
            continue

        entry_price = max(float(next_row["open"]), trigger_price)
        stop_candidate = float(next_row["low"])
        if stop_candidate <= 0 or stop_candidate >= entry_price:
            continue

        risk_pct = (entry_price - stop_candidate) / entry_price
        adr_value = compute_average_adr(history_until_now)
        if adr_value is None or adr_value <= 0:
            continue
        adr_ratio = risk_pct / adr_value
        if adr_ratio > config.max_risk_to_adr:
            continue
        
        open_trade = Trade(
            ticker=ticker,
            entry_date=next_date,
            entry_price=entry_price,
            stop_price=stop_candidate,
            exit_date=next_date,
            exit_price=entry_price,
            outcome=f"open_{setup_signal.kind}",
        )
        open_state = {
            "dynamic_stop": stop_candidate,
            "bars": 0,
            "partial_done": False,
        }

        logging.info(
            "[%s] ENTRADA: %s @ $%.2f (Stop Día: $%.2f, Riesgo%%=%.2f, Setup=%s)",
            ticker,
            next_date.date().isoformat(),
            entry_price,
            stop_candidate,
            risk_pct * 100,
            setup_signal.kind,
        )

    if open_trade:
        remaining = usable.loc[open_trade.entry_date :]
        if remaining.empty:
            remaining = usable.iloc[[-1]]
        last_row = remaining.iloc[-1]

        open_trade.exit_date = last_row.name
        open_trade.exit_price = float(last_row["close"])
        open_trade.outcome = "timeout"
        trades.append(open_trade)

        logging.info(
            "[%s] SALIDA: %s @ $%.2f (Motivo: timeout, R=%.2f)",
            open_trade.ticker,
            open_trade.exit_date.date().isoformat(),
            open_trade.exit_price,
            open_trade.r_multiple,
        )

    return trades


def run_backtest(
    histories: dict[str, pd.DataFrame],
    fundamentals: dict[str, Sequence[FundamentalSnapshot]],
    thresholds: Thresholds,
    config: BacktestConfig,
    index_history: Optional[pd.DataFrame], # <-- MODIFICADO
) -> list[Trade]:
    all_trades: list[Trade] = []
    for ticker, history in tqdm(histories.items(), desc="Simulando operaciones"):
        trades = simulate_ticker(
            ticker,
            history,
            fundamentals.get(ticker, ()),
            thresholds,
            config,
            index_history, # <-- MODIFICADO
        )
        all_trades.extend(trades)
    return all_trades


# ---------------------------------------------------------------------------
# Análisis y Reporte
# ---------------------------------------------------------------------------


def compute_statistics(trades: Iterable[Trade]) -> dict[str, float]:
    """Calcula las métricas de rendimiento clave a partir de una lista de trades."""
    r_values = [trade.r_multiple for trade in trades]
    if not r_values:
        return {
            "total_trades": 0.0,
            "win_rate": 0.0,
            "average_r": 0.0,
            "profit_factor": 0.0,
            "avg_risk_reward": 0.0,
            "max_drawdown": 0.0,
        }
    
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]
    
    # Cálculo del Max Drawdown (basado en R)
    cumulative_r = 0.0
    peak_r = 0.0
    max_drawdown = 0.0
    for r in r_values:
        cumulative_r += r
        if cumulative_r > peak_r:
            peak_r = cumulative_r
        drawdown = peak_r - cumulative_r
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    stats = {
        "total_trades": float(len(trades)),
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
            "Fecha Parcial": (
                trade.partial_exit_date.date().isoformat()
                if trade.partial_exit_date is not None
                else None
            ),
            "Precio Parcial": trade.partial_exit_price,
            "Fracción Parcial": trade.partial_fraction if trade.partial_fraction > 0 else None,
            "Motivo Parcial": trade.partial_reason,
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
        "--weekly-vol-multiple",
        "--adr-stop-multiple", # Alias por compatibilidad
        dest="vol_stop_multiple",
        type=float,
        default=DEFAULT_WEEKLY_VOL_STOP_MULTIPLE,
        help=(
            "Multiplicador de la volatilidad semanal reciente (rango en 5 sesiones) "
            "para fijar el stop inicial."
        ),
    )
    parser.add_argument(
        "--ema-period",
        type=int,
        default=DEFAULT_EMA_PERIOD,
        help="Periodo de la EMA utilizada como stop dinámico (por defecto 50).",
    )
    parser.add_argument(
        "--ema-tolerance",
        type=float,
        default=DEFAULT_EMA_TOLERANCE,
        help=(
            "Porcentaje de tolerancia bajo la EMA antes de ejecutar la salida "
            "dinámica (decimal, p.ej. 0.03 = 3%)."
        ),
    )
    parser.add_argument(
        "--partial-fraction",
        type=float,
        default=DEFAULT_PARTIAL_PROFIT_FRACTION,
        help=(
            "Fracción de la posición a tomar como ganancia parcial tras el "
            "período inicial (decimal, p.ej. 0.5 = 50%)."
        ),
    )
    parser.add_argument(
        "--partial-delay",
        type=int,
        default=DEFAULT_PARTIAL_PROFIT_DELAY,
        help="Número mínimo de sesiones (barra diaria) antes de ejecutar la toma parcial.",
    )
    parser.add_argument(
        "--max-risk-adr",
        type=float,
        default=DEFAULT_MAX_RISK_TO_ADR_RATIO,
        help=(
            "Máximo cociente permitido entre el riesgo inicial (% sobre la entrada) "
            "y el ADR de 3 meses. Valores < 1 fuerzan riesgos inferiores al ADR."
        ),
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
        help="Crecimiento interanual mínimo del EPS (decimal).",
    )
    parser.add_argument(
        "--min-revenue-growth",
        type=float,
        default=DEFAULT_MIN_REVENUE_GROWTH,
        help="Crecimiento interanual mínimo de los ingresos (decimal).",
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
        vol_stop_multiple=max(args.vol_stop_multiple, 0.1),
        ema_trail_period=max(int(args.ema_period), 1),
        ema_tolerance=max(args.ema_tolerance, 0.0),
        partial_profit_fraction=max(0.0, min(1.0, args.partial_fraction)),
        partial_profit_delay=max(int(args.partial_delay), 1),
        max_risk_to_adr=max(args.max_risk_adr, 0.01),
    )

    thresholds = Thresholds(
        min_eps_growth=args.min_eps_growth,
        min_revenue_growth=args.min_revenue_growth,
        min_momentum_3m=max(args.min_momentum_3m, 0.0),
        max_week_volatility=max(args.max_week_volatility, 0.0),
        min_adr=max(args.min_adr, 0.0),
    )

    tickers = load_tickers(Path(args.input))
    
    logging.info("Iniciando Fase 1: Carga de datos (usando caché si existe)")
    CACHE_DIR.mkdir(exist_ok=True)

    histories: dict[str, pd.DataFrame] = {}
    fundamentals: dict[str, Sequence[FundamentalSnapshot]] = {}

    # --- NUEVO: Cargar el índice (QQQ) ---
    index_history: Optional[pd.DataFrame]
    index_path = CACHE_DIR / "QQQ_history.csv"
    if index_path.exists():
        index_history = load_history_from_cache(index_path)
        index_history = index_history.sort_index()
        index_history["ma50"] = index_history["close"].rolling(window=50).mean()
        index_history["ma200"] = index_history["close"].rolling(window=200).mean()
        index_history["ema10"] = index_history["close"].ewm(span=10, adjust=False).mean()
        index_history["ema20"] = index_history["close"].ewm(span=20, adjust=False).mean()
    else:
        index_history = None
        logging.warning(
            "Datos del índice QQQ no encontrados. Se omitirán los filtros de régimen y RS."
        )
    # --- FIN CARGA DE ÍNDICE ---

    # margin = pd.Timedelta(days=200) # El margen se maneja en el downloader
    # start_download = start_ts - margin

    for ticker in tqdm(tickers, desc="Cargando datos de tickers"):
        if ticker.upper() == "QQQ": # Ignorar el propio índice
            continue
            
        history_path = CACHE_DIR / f"{ticker}_history.csv"
        fundamentals_path = CACHE_DIR / f"{ticker}_fundamentals.csv"

        try:
            if history_path.exists():
                history = load_history_from_cache(history_path)
            else:
                logging.debug( # Silenciado
                    "%s: no se encontró historial. Ejecuta downloader.py primero.",
                    ticker,
                )
                continue
            
            if fundamentals_path.exists():
                snapshots = load_fundamental_snapshots(fundamentals_path)
            else:
                logging.debug(
                    "%s: no se encontraron fundamentales. Se operará sin filtro fundamental.",
                    ticker,
                )
                snapshots = [] # Permitir operar sin fundamentales

            if not history.empty:
                histories[ticker] = history
                fundamentals[ticker] = snapshots
        except Exception as e:
            logging.warning("Error procesando %s: %s", ticker, e, exc_info=True)

    if not histories:
        logging.error(
            "No se cargaron datos históricos para ningún ticker. "
            "Asegúrate de ejecutar 'downloader.py' primero."
        )
        return

    logging.info("Iniciando Fase 2: Simulación de operaciones")
    trades = run_backtest(histories, fundamentals, thresholds, config, index_history)
    
    if not trades:
        logging.info("No se generaron operaciones con los parámetros actuales.")
        return

    logging.info("Iniciando Fase 3: Cálculo de estadísticas y guardado")
    
    stats = compute_statistics(trades)
    print("\n--- Estadísticas del Backtest (basadas en R) ---")
    print(f"Total de Operaciones: {stats['total_trades']:.0f}")
    print(f"Tasa de Acierto:     {stats['win_rate'] * 100:.2f}%")
    print(f"R-Múltiplo Medio:    {stats['average_r']:.2f} R")
    print(f"Profit Factor:       {stats['profit_factor']:.2f}")
    print(f"Avg. Risk/Reward:    {stats['avg_risk_reward']:.2f}")
    print(f"Max Drawdown (R):    {stats['max_drawdown']:.2f} R")
    print("--------------------------------------------------")

    results_df = trades_to_dataframe(trades)
    output_path = Path(args.output)
    results_df.to_csv(output_path, index=False)
    logging.info("Resultados guardados en: %s", output_path.resolve())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()