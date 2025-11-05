#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import bisect
import logging
from dataclasses import dataclass
from datetime import timezone, datetime
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
DEFAULT_WEEKLY_VOL_STOP_MULTIPLE = 2.0  # <-- REQUISITO 1 (Stop por Vol Semanal)
DEFAULT_EMA_PERIOD = 50
DEFAULT_EMA_TOLERANCE = 0.03            # <-- REQUISITO 2 (Buffer EMA 3%)
CACHE_DIR = Path(".cache")


@dataclass
class BacktestConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    vol_stop_multiple: float  # <-- MODIFICADO
    ema_trail_period: int
    ema_tolerance: float      # <-- NUEVO


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

    @property
    def risk_amount(self) -> float:
        return self.entry_price - self.stop_price

    @property
    def r_multiple(self) -> float:
        risk = self.risk_amount
        if risk <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / risk


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


def download_history(
    client: YahooFinanceEUClient,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Descarga velas diarias entre ``start`` y ``end`` (ambos inclusive)."""
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


def load_history_from_cache(path: Path) -> pd.DataFrame:
    """Lee un CSV de histórico previamente cacheado."""
    history = pd.read_csv(path, index_col="date", parse_dates=True)
    index = history.index
    if getattr(index, "tz", None) is None:
        history.index = index.tz_localize(timezone.utc)
    else:
        history.index = index.tz_convert(timezone.utc)
    history = history.sort_index()
    return history


def _prepare_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _normalize_fundamental_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de columna para admitir cabeceras variadas."""

    def _clean(name: object) -> str:
        raw = str(name)
        raw = raw.lstrip("\ufeff").strip()
        normalized = raw.lower().replace("-", "_").replace(" ", "_")
        return normalized

    canonical_map = {
        "available_at": "available_at",
        "availableat": "available_at",
        "fecha_disponible": "available_at",
        "as_of": "as_of",
        "asof": "as_of",
        "eps": "eps",
        "dilutedeps": "eps",
        "reported_eps": "eps",
        "revenue": "revenue",
        "total_revenue": "revenue",
        "ingresos": "revenue",
    }

    renamed: dict[str, str] = {}
    for column in df.columns:
        cleaned = _clean(column)
        if cleaned.startswith("unnamed"):
            continue
        renamed[column] = canonical_map.get(cleaned, cleaned)

    if renamed:
        df = df.rename(columns=renamed)

    df = df.loc[:, ~df.columns.duplicated()]

    return df


def load_fundamental_snapshots(path: Path) -> list[FundamentalSnapshot]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logging.debug("%s: no se pudo leer el CSV de fundamentales (%s)", path.name, exc)
        return []
    if df.empty:
        return []

    df = _normalize_fundamental_columns(df)

    if "available_at" not in df or "eps" not in df or "revenue" not in df:
        logging.debug("%s: formato de fundamentales inesperado", path.name)
        return []

    df = df.copy()
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    if "as_of" in df:
        df["as_of"] = pd.to_datetime(df["as_of"], utc=True, errors="coerce")
    df["eps"] = _prepare_series(df["eps"])
    df["revenue"] = _prepare_series(df["revenue"])
    df = df.dropna(subset=["available_at", "eps", "revenue"])
    if df.empty:
        return []

    df = df.sort_values("available_at").drop_duplicates(subset=["available_at"])

    selected: Optional[pd.DataFrame] = None
    for lag in (4, 1):  # Preferimos interanual; si no hay datos suficientes, usamos QoQ.
        working = df.copy()
        working["eps_prev"] = working["eps"].shift(lag)
        working["revenue_prev"] = working["revenue"].shift(lag)

        eps_valid = working["eps_prev"].abs() > 1e-9
        rev_valid = working["revenue_prev"].abs() > 1e-9
        working.loc[eps_valid, "eps_growth"] = (
            working.loc[eps_valid, "eps"] - working.loc[eps_valid, "eps_prev"]
        ) / working.loc[eps_valid, "eps_prev"]
        working.loc[rev_valid, "revenue_growth"] = (
            working.loc[rev_valid, "revenue"] - working.loc[rev_valid, "revenue_prev"]
        ) / working.loc[rev_valid, "revenue_prev"]

        working = working.dropna(subset=["eps_growth", "revenue_growth"])
        if not working.empty:
            selected = working
            break

    if selected is None:
        return []

    df = selected

    snapshots: list[FundamentalSnapshot] = []
    for row in df.itertuples():
        available_at = getattr(row, "available_at", None)
        if isinstance(available_at, pd.Timestamp):
            if available_at.tzinfo is None:
                available_at = available_at.tz_localize(timezone.utc)
            else:
                available_at = available_at.tz_convert(timezone.utc)
        else:
            continue
        snapshots.append(
            FundamentalSnapshot(
                available_at=available_at,
                eps=float(row.eps),
                revenue=float(row.revenue),
                eps_growth=float(row.eps_growth),
                revenue_growth=float(row.revenue_growth),
            )
        )
    return snapshots


def select_snapshot(
    snapshots: Sequence[FundamentalSnapshot],
    current_date: pd.Timestamp,
    *,
    dates: Optional[Sequence[pd.Timestamp]] = None,
) -> Optional[FundamentalSnapshot]:
    if not snapshots:
        return None
    if dates is None:
        dates = [snap.available_at for snap in snapshots]
    position = bisect.bisect_right(dates, current_date) - 1
    if position < 0:
        return None
    return snapshots[position]


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
    if segment.empty or pd.isna(segment["open"].iloc[0]):
        return None
    reference = segment["open"].iloc[0]
    if reference <= 0:
        return None
    weekly_range = segment["high"].max() - segment["low"].min()
    return float(weekly_range / reference)


# --- REQUISITO 1: Nueva función de Volatilidad Semanal Promedio ---
def compute_average_weekly_range_pct(history: pd.DataFrame, window_weeks: int = 10) -> Optional[float]:
    """Calcula el rango semanal promedio como % del cierre, usando 'window_weeks' semanas."""
    
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
    
    working_history = history.copy().sort_index()
    
    aligned_index: Optional[pd.DataFrame] = None
    if index_history is not None:
        aligned_index = index_history.reindex(working_history.index, method="ffill")

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

        market_bearish = index_history is not None # Asumir bajista si el filtro está activo
        if market_now is not None:
            market_close = market_now.get("close")
            market_ma50 = market_now.get("ma50")
            if not pd.isna(market_close) and not pd.isna(market_ma50):
                try:
                    market_bearish = float(market_close) < float(market_ma50)
                except TypeError:
                    pass # Dejar como True
        elif index_history is None:
            market_bearish = False # Ignorar filtro si no hay índice
        # --- FIN FILTRO DE RÉGIMEN ---

        if open_trade and current_date >= open_trade.entry_date:
            low_price = float(row["low"])
            close_price = float(row["close"])
            ema_value = row.get("ema_trail", pd.NA)
            exit_trade: Optional[Trade] = None

            # 1. Comprobar Stop Catastrófico (Volatilidad Semanal)
            if low_price <= open_trade.stop_price:
                exit_trade = Trade(
                    ticker=open_trade.ticker,
                    entry_date=open_trade.entry_date,
                    entry_price=open_trade.entry_price,
                    stop_price=open_trade.stop_price,
                    exit_date=current_date,
                    exit_price=open_trade.stop_price,
                    outcome="stop_vol", # Motivo actualizado
                )
            
            # 2. Comprobar Trailing Stop (EMA) con Banda de Tolerancia
            elif not pd.isna(ema_value):
                # --- REQUISITO 2: Lógica de Buffer/Tolerancia ---
                ema_con_buffer = float(ema_value) * (1.0 - config.ema_tolerance)
                
                if close_price < ema_con_buffer:
                    exit_trade = Trade(
                        ticker=open_trade.ticker,
                        entry_date=open_trade.entry_date,
                        entry_price=open_trade.entry_price,
                        stop_price=open_trade.stop_price,
                        exit_date=current_date,
                        exit_price=close_price,
                        outcome=f"ema{config.ema_trail_period}_break",
                    )

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
        
        # 3. Filtro Técnico
        history_until_now = working_history.loc[:current_date]
        if not passes_scanner(history_until_now, thresholds):
            continue

        # --- FILTROS v4.1 (Qullamaggie/Rai) ---
        row_now = working_history.loc[current_date]
        
        # 4. Filtro de Calidad de Setup (Posición de MAs)
        ma50_value = row_now.get("ma50")
        ma200_value = row_now.get("ma200")
        if pd.isna(ma50_value) or pd.isna(ma200_value):
            continue
        if float(row_now["close"]) < float(ma50_value): # Precio > MA50
            continue
        if float(ma50_value) < float(ma200_value): # MA50 > MA200 (tendencia alcista)
            continue

        # 5. Filtro de Fuerza Relativa (RS)
        if index_history is not None: # Solo aplicar si tenemos datos del índice
            rs_line = row_now.get("rs_line")
            rs_ma21 = row_now.get("rs_ma21")
            if pd.isna(rs_line) or pd.isna(rs_ma21):
                continue
            if float(rs_line) < float(rs_ma21): # RS Line > RS MA21
                continue
            # Filtro de RS_63_high ELIMINADO como se solicitó

        # --- Si todo pasa, abrimos la operación en la apertura del día siguiente ---
        next_row = usable.iloc[idx + 1]
        if pd.isna(next_row["open"]):
            continue
        entry_price = float(next_row["open"])

        # --- REQUISITO 1: NUEVA LÓGICA DE STOP BASADO EN VOL SEMANAL ---
        week_vol = compute_weekly_volatility(history_until_now)
        if week_vol is None:
            continue
        
        risk_dollars = entry_price * week_vol * config.vol_stop_multiple
        stop_price = max(entry_price - risk_dollars, 0.01) 

        open_trade = Trade(
            ticker=ticker,
            entry_date=next_date,
            entry_price=entry_price,
            stop_price=stop_price,
            exit_date=next_date,
            exit_price=entry_price,
            outcome="open",
        )
        
        logging.info(
            "[%s] ENTRADA: %s @ $%.2f (Stop VolSem: $%.2f, VolSem1S: %.4f)",
            ticker,
            next_date.date().isoformat(),
            entry_price,
            stop_price,
            week_vol,
        )

    if open_trade:
        remaining = usable.loc[open_trade.entry_date :]
        if remaining.empty:
            remaining = usable.iloc[[-1]]
        last_row = remaining.iloc[-1]
        
        final_trade = Trade(
            ticker=open_trade.ticker,
            entry_date=open_trade.entry_date,
            entry_price=open_trade.entry_price,
            stop_price=open_trade.stop_price,
            exit_date=last_row.name,
            exit_price=float(last_row["close"]),
            outcome="timeout",
        )
        trades.append(final_trade)
        
        logging.info(
            "[%s] SALIDA: %s @ $%.2f (Motivo: timeout, R=%.2f)",
            final_trade.ticker,
            final_trade.exit_date.date().isoformat(),
            final_trade.exit_price,
            final_trade.r_multiple,
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
        except Exception as exc:  # noqa: BLE001
            logging.debug("%s: no se pudo preparar el histórico (%s)", ticker, exc)
            continue

        snapshots = load_fundamental_snapshots(fundamentals_path)
        # --- MODIFICADO: PERMITE TICKERS SIN FUNDAMENTALES ---
        if not snapshots:
            logging.debug(
                "%s: fundamentales no disponibles. Se ejecutará sin filtros fundamentales.",
                ticker,
            )
            # No continuamos, permitimos que el ticker se añada
        
        histories[ticker] = history
        fundamentals[ticker] = snapshots # Estará vacío si no se encontró

    if not histories:
        logging.error(
            "No se encontraron suficientes datos cacheados. Asegúrate de ejecutar downloader.py con los tickers requeridos."
        )
        return

    logging.info("Iniciando Fase 2: Simulación de operaciones")
    trades = run_backtest(histories, fundamentals, thresholds, config, index_history)
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