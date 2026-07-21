"""Event-driven multi-ticker backtest with vol targeting and chandelier exits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trad_research.features import M2_FEATURE_NAMES, M2_REL_FEATURE_NAMES, feature_matrix
from trad_research.sector_filter import sector_allows_entry


@dataclass
class BacktestConfig:
    min_confidence: float = 0.45
    k_atr: float = 2.5
    max_horizon: int = 20
    hard_stop_pct: float = 0.07
    commission: float = 0.001
    slippage: float = 0.0005
    volatility_target_pct: float = 0.01
    max_position_pct: float = 0.20
    max_positions: int = 10
    initial_capital: float = 100_000.0
    require_trend: bool = True
    require_momentum: bool = True
    momentum_min: float = 0.02
    buy_class: int = 2
    # Optional bullish regime calendar: date -> True if risk-on
    regime_ok: Optional[Dict[pd.Timestamp, bool]] = None
    require_regime: bool = True
    # Skip entries when ATR/price above this (too wild)
    max_atr_pct: float = 0.08
    # Prefer quality: min dist above SMA200
    min_dist_sma200: float = -0.02
    # Portfolio peak-to-trough circuit: block new entries if DD worse than this
    # Default 0.99 effectively disables; enable (e.g. 0.18) for kill-switch
    max_portfolio_dd: float = 0.99
    # Scale new entries when soft regime is False
    risk_off_scale: float = 0.55
    soft_regime_ok: Optional[Dict[pd.Timestamp, bool]] = None
    # Max new entries considered per day (top-k by p_buy)
    max_entries_per_day: int = 8
    feature_names: Optional[Sequence[str]] = None
    meta_model: Any = None
    meta_threshold: float = 0.50
    # Fraction of equity always allocated to QQQ (or cash if QQQ missing)
    qqq_sleeve_pct: float = 0.0
    qqq_panel: Optional[pd.DataFrame] = None
    # --- Adaptive exit (audit): time_stop + profit + bull → extend or trail-only ---
    adaptive_exit: bool = False
    # "extend" | "trail_only" | "auto" (big winners → trail_only, else extend)
    adaptive_mode: str = "auto"
    adaptive_min_profit: float = 0.15  # unrealized ret to allow extension
    adaptive_trail_only_profit: float = 0.40  # uret ≥ this → trail-only (no time_stop)
    adaptive_extend_bars: int = 20
    adaptive_max_extensions: int = 2
    adaptive_trail_k_mult: float = 1.20  # wider trail after extend
    adaptive_require_regime: bool = True  # only when index regime risk-on
    adaptive_min_atr_norm: float = 0.02
    # Cap capital per ticker (fraction of equity); 1.0 = only max_position_pct applies
    ticker_max_capital_pct: float = 1.0
    # After realized PnL on a ticker exceeds this fraction of peak equity, skip re-entry
    ticker_max_realized_pnl_frac: float = 1.0  # 1.0 disables
    # Soft de-risk size scale when DD reaches this fraction of max_portfolio_dd
    dd_soft_scale: float = 0.55
    # --- Bottleneck-fix knobs (defaults preserve legacy champions) ---
    # Skip entry if computed alloc < min_alloc_pct * equity (0.0 = off).
    min_alloc_pct: float = 0.0
    # When True: hard regime risk-off scales size instead of blocking all entries.
    soft_hard_regime: bool = False
    # Size multiplier when hard regime is False under soft_hard_regime.
    # If None, falls back to risk_off_scale.
    regime_hard_size_scale: Optional[float] = None
    # ATR-aware hard stop: distance = max(price * hard_stop_pct, hard_stop_atr_mult * atr).
    # None disables ATR floor (legacy: pure percent hard stop).
    hard_stop_atr_mult: Optional[float] = None
    # --- Sector ETF gate: ticker -> sector name; etf_maps: ETF -> {date: above_MA} ---
    ticker_sector: Optional[Dict[str, str]] = None
    sector_etf_maps: Optional[Dict[str, Dict[pd.Timestamp, bool]]] = None
    sector_allow_unmapped: bool = True  # if no sector map for ticker, allow entry
    require_sector_trend: bool = False  # master switch
    # --- Rotation when full: sell worst open score to buy better candidate ---
    enable_rotation: bool = False
    # New candidate p_buy must beat worst held score by at least this margin
    rotation_min_score_edge: float = 0.05
    # Min bars held before a position can be rotated out
    rotation_min_bars: int = 3
    # Max rotations per day
    rotation_max_per_day: int = 2
    # Survivorship-free: ticker -> last trade date (UTC); force exit at/after that day
    delist_dates: Optional[Dict[str, pd.Timestamp]] = None
    # Optional M&A: ticker -> successor ticker (same ISIN); if set, try roll on delist
    delist_successors: Optional[Dict[str, str]] = None
    roll_on_delist: bool = False  # if True and successor listed, open successor with same $ 


@dataclass
class OpenPosition:
    ticker: str
    entry_date: pd.Timestamp
    entry_idx: int
    entry_price: float
    shares: int
    stop: float
    hard_stop: float
    highest_high: float
    bars_held: int
    capital_used: float
    horizon_limit: int = 20
    extensions: int = 0
    k_atr_scale: float = 1.0


def _regime_on_day(
    day: pd.Timestamp,
    regime_ok: Optional[Dict[pd.Timestamp, bool]],
) -> bool:
    if regime_ok is None:
        return True
    if day in regime_ok:
        return bool(regime_ok[day])
    prior = [d for d in regime_ok if d <= day]
    if prior:
        return bool(regime_ok[max(prior)])
    return True


def compute_entry_hard_stop(price: float, atr: float, cfg: BacktestConfig) -> float:
    """Entry hard-stop level (price units). Uses only entry-bar price/ATR (causal).

    EXP4 formula when hard_stop_atr_mult is set:
        stop_dist = max(price * hard_stop_pct, hard_stop_atr_mult * atr)
        stop_dist = min(stop_dist, price * (1 - 1e-6))  # keep hard in (0, price)
        hard = price - stop_dist
    Legacy (mult None): hard = price * (1 - hard_stop_pct).
    """
    if price <= 0:
        return 0.0
    if cfg.hard_stop_atr_mult is not None and cfg.hard_stop_atr_mult > 0 and atr > 0:
        stop_dist = max(price * cfg.hard_stop_pct, float(cfg.hard_stop_atr_mult) * atr)
        # Bound so hard stays in (0, price); avoids non-binding negative stops
        stop_dist = min(stop_dist, price * (1.0 - 1e-6))
        return float(price - stop_dist)
    return float(price * (1.0 - cfg.hard_stop_pct))


def _chandelier_step(
    pos: OpenPosition,
    high: float,
    low: float,
    close: float,
    atr: float,
    cfg: BacktestConfig,
    *,
    regime_on: bool = True,
    atr_norm: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Update trail/stop; optional adaptive time extension for winners.

    Returns (should_exit, reason, fill_price_hint).
    reason may be adaptive_extend (should_exit=False) for logging only — caller
    treats should_exit=False as hold; we use reason "" for normal hold.
    """
    pos.bars_held += 1
    if high > pos.highest_high:
        pos.highest_high = high
    k = cfg.k_atr * pos.k_atr_scale
    trail = pos.highest_high - k * atr
    pos.stop = max(pos.stop, trail)
    active = max(pos.stop, pos.hard_stop)
    if low <= active or close <= active:
        reason = "hard_stop" if active <= pos.hard_stop + 1e-9 else "trail_stop"
        return True, reason, active

    if pos.bars_held >= pos.horizon_limit:
        # trail-only positions never time-stop
        if pos.horizon_limit >= 10**8:
            return False, "", close
        uret = close / pos.entry_price - 1.0 if pos.entry_price > 0 else 0.0
        atr_n = atr_norm if atr_norm is not None else (atr / close if close > 0 else 0.0)
        can_adapt = (
            cfg.adaptive_exit
            and uret >= cfg.adaptive_min_profit
            and (not cfg.adaptive_require_regime or regime_on)
            and atr_n >= cfg.adaptive_min_atr_norm
        )
        if can_adapt:
            mode = (cfg.adaptive_mode or "extend").lower()
            use_trail_only = mode == "trail_only" or (
                mode == "auto" and uret >= cfg.adaptive_trail_only_profit
            )
            if use_trail_only:
                pos.extensions += 1
                pos.horizon_limit = 10**9  # trail/hard only from here
                pos.k_atr_scale = max(pos.k_atr_scale, cfg.adaptive_trail_k_mult)
                trail2 = pos.highest_high - cfg.k_atr * pos.k_atr_scale * atr
                if trail2 < pos.stop:
                    pos.stop = trail2
                return False, "adaptive_trail_only", close
            if pos.extensions < cfg.adaptive_max_extensions:
                pos.extensions += 1
                pos.horizon_limit = pos.bars_held + cfg.adaptive_extend_bars
                pos.k_atr_scale = max(pos.k_atr_scale, cfg.adaptive_trail_k_mult)
                trail2 = pos.highest_high - cfg.k_atr * pos.k_atr_scale * atr
                if trail2 < pos.stop:
                    pos.stop = trail2
                return False, "adaptive_extend", close
        return True, "time_stop", close
    return False, "", close


def generate_signals(
    df: pd.DataFrame,
    model: Any,
    cfg: BacktestConfig,
) -> Tuple[pd.Series, pd.Series]:
    # Strategy objects (trad_research.strategies) provide generate_signals
    if model is not None and hasattr(model, "generate_signals") and callable(model.generate_signals):
        return model.generate_signals(df, cfg)

    names = cfg.feature_names or M2_REL_FEATURE_NAMES
    X = feature_matrix(df, names)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", list(range(proba.shape[1]))))
        # Binary model: P(class=1) is buy. Multiclass: P(buy_class).
        if proba.shape[1] == 2 and 1 in classes:
            buy_i = classes.index(1)
        elif cfg.buy_class in classes:
            buy_i = classes.index(cfg.buy_class)
        elif 1 in classes:
            buy_i = classes.index(1)
        else:
            buy_i = int(np.argmax(classes)) if len(classes) else 0
        p_buy = proba[:, buy_i]
    else:
        pred = model.predict(X)
        p_buy = ((pred == 1) | (pred == cfg.buy_class)).astype(float)

    sig = p_buy >= cfg.min_confidence
    if cfg.require_trend and "sma_50" in df.columns:
        sig = sig & (df["close"].to_numpy() > df["sma_50"].to_numpy())
    if cfg.require_momentum and "ret_1m" in df.columns:
        sig = sig & (df["ret_1m"].to_numpy() >= cfg.momentum_min)
    if "atr_norm" in df.columns and cfg.max_atr_pct is not None:
        sig = sig & (df["atr_norm"].to_numpy() <= cfg.max_atr_pct)
    if "dist_sma_200" in df.columns and cfg.min_dist_sma200 is not None:
        sig = sig & (df["dist_sma_200"].to_numpy() >= cfg.min_dist_sma200)
    if cfg.meta_model is not None and hasattr(cfg.meta_model, "predict_proba"):
        p_meta = cfg.meta_model.predict_proba(X)[:, 1]
        sig = sig & (p_meta >= cfg.meta_threshold)
        # Rank by primary * meta confidence
        p_buy = p_buy * p_meta
    return pd.Series(sig, index=df.index), pd.Series(p_buy, index=df.index)


def run_portfolio_backtest(
    panels: Dict[str, pd.DataFrame],
    model: Any,
    cfg: BacktestConfig,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    panels: ticker -> featured OHLCV with date column
    Returns trades, equity_curve (daily), signals_log
    """
    # Align calendars
    frames: Dict[str, pd.DataFrame] = {}
    for t, df in panels.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], utc=True)
        if start is not None:
            d = d[d["date"] >= start]
        if end is not None:
            d = d[d["date"] <= end]
        d = d.reset_index(drop=True)
        if len(d) < 30:
            continue
        sig, p = generate_signals(d, model, cfg)
        d = d.copy()
        d["signal"] = sig.to_numpy()
        d["p_buy"] = p.to_numpy()
        frames[t] = d

    if not frames:
        empty_eq = pd.Series(dtype=float)
        return pd.DataFrame(), empty_eq, pd.DataFrame()

    # Build global trading days
    all_dates = sorted(set().union(*[set(df["date"]) for df in frames.values()]))
    # Index rows by date for each ticker
    by_date: Dict[str, Dict[pd.Timestamp, int]] = {}
    for t, df in frames.items():
        by_date[t] = {row.date: i for i, row in enumerate(df.itertuples(index=False))}

    cash = cfg.initial_capital
    positions: Dict[str, OpenPosition] = {}
    trades: List[Dict[str, Any]] = []
    equity_points: List[Tuple[pd.Timestamp, float]] = []
    peak_equity = cfg.initial_capital

    # Optional QQQ core sleeve (buy & hold, rebalanced daily to target %)
    qqq_shares = 0.0
    qqq_by_date: Dict[pd.Timestamp, float] = {}
    if cfg.qqq_sleeve_pct > 0 and cfg.qqq_panel is not None and not cfg.qqq_panel.empty:
        qdf = cfg.qqq_panel.copy()
        qdf["date"] = pd.to_datetime(qdf["date"], utc=True)
        qqq_by_date = {row.date: float(row.close) for row in qdf.itertuples(index=False)}

    def qqq_price(day: pd.Timestamp) -> Optional[float]:
        if day in qqq_by_date:
            return qqq_by_date[day]
        prior = [d for d in qqq_by_date if d <= day]
        return qqq_by_date[max(prior)] if prior else None

    def mark_to_market(day: pd.Timestamp) -> float:
        eq = cash
        for t, pos in positions.items():
            df = frames[t]
            idx = by_date[t].get(day)
            if idx is None:
                # use last known
                past = [d for d in by_date[t] if d <= day]
                if not past:
                    eq += pos.capital_used
                    continue
                idx = by_date[t][max(past)]
            px = float(df.iloc[idx]["close"])
            eq += pos.shares * px
        px_q = qqq_price(day)
        if px_q is not None and qqq_shares > 0:
            eq += qqq_shares * px_q
        return eq

    def rebalance_qqq_sleeve(day: pd.Timestamp) -> None:
        nonlocal cash, qqq_shares
        if cfg.qqq_sleeve_pct <= 0:
            return
        px_q = qqq_price(day)
        if px_q is None or px_q <= 0:
            return
        eq = mark_to_market(day)
        target_val = eq * cfg.qqq_sleeve_pct
        current_val = qqq_shares * px_q
        delta = target_val - current_val
        if abs(delta) < eq * 0.005:
            return
        if delta > 0:
            spend = min(delta, cash * 0.95)
            if spend > 0:
                sh = spend / (px_q * (1 + cfg.slippage))
                cost = sh * px_q * (1 + cfg.slippage)
                comm = cost * cfg.commission
                if cost + comm <= cash:
                    cash -= cost + comm
                    qqq_shares += sh
        else:
            sh = min(qqq_shares, abs(delta) / px_q)
            if sh > 0:
                proceeds = sh * px_q * (1 - cfg.slippage)
                comm = proceeds * cfg.commission
                cash += proceeds - comm
                qqq_shares -= sh

    # Realized PnL by ticker (for re-entry cap)
    realized_pnl_by_ticker: Dict[str, float] = {}
    adaptive_extend_count = 0

    def _delist_ts(ticker: str) -> Optional[pd.Timestamp]:
        if not cfg.delist_dates:
            return None
        d = cfg.delist_dates.get(ticker) or cfg.delist_dates.get(ticker.upper())
        if d is None:
            return None
        dd = pd.Timestamp(d)
        if dd.tzinfo is None:
            dd = dd.tz_localize("UTC")
        return dd

    for day in all_dates:
        regime_on = _regime_on_day(day, cfg.regime_ok)

        # 1) Update exits
        to_close: List[str] = []
        pending_rolls: List[Tuple[str, float]] = []  # successor, capital dollars
        for t, pos in positions.items():
            idx = by_date[t].get(day)
            # Delisting: force exit on last trade day (or if bar missing after delist)
            d_end = _delist_ts(t)
            if d_end is not None and day.normalize() >= d_end.normalize():
                if idx is None:
                    # use last available bar
                    past = [d for d in by_date.get(t, {}) if d <= day]
                    if not past:
                        cash += pos.capital_used  # residual unknown → return book cost
                        trades.append(
                            {
                                "ticker": t,
                                "entry_date": pos.entry_date,
                                "exit_date": day,
                                "entry_price": pos.entry_price,
                                "exit_price": pos.entry_price,
                                "shares": pos.shares,
                                "capital_used": pos.capital_used,
                                "net_profit": 0.0,
                                "trade_return": 0.0,
                                "bars_held": pos.bars_held,
                                "exit_reason": "delisting_no_bar",
                                "extensions": pos.extensions,
                            }
                        )
                        to_close.append(t)
                        continue
                    idx = by_date[t][max(past)]
                row = frames[t].iloc[idx]
                fill = float(row["close"]) * (1 - cfg.slippage)
                proceeds = pos.shares * fill
                comm = proceeds * cfg.commission
                net = proceeds - comm
                cash += net
                pnl = net - pos.capital_used
                realized_pnl_by_ticker[t] = realized_pnl_by_ticker.get(t, 0.0) + float(pnl)
                trades.append(
                    {
                        "ticker": t,
                        "entry_date": pos.entry_date,
                        "exit_date": day,
                        "entry_price": pos.entry_price,
                        "exit_price": fill,
                        "shares": pos.shares,
                        "capital_used": pos.capital_used,
                        "net_profit": pnl,
                        "trade_return": pnl / pos.capital_used if pos.capital_used else 0.0,
                        "bars_held": pos.bars_held,
                        "exit_reason": "delisting",
                        "extensions": pos.extensions,
                    }
                )
                to_close.append(t)
                if cfg.roll_on_delist and cfg.delist_successors:
                    succ = cfg.delist_successors.get(t) or cfg.delist_successors.get(t.upper())
                    if succ and succ in frames and succ not in positions:
                        pending_rolls.append((succ.upper(), float(net)))
                continue

            if idx is None:
                continue
            row = frames[t].iloc[idx]
            atr_v = float(row["atr"])
            close_v = float(row["close"])
            atr_n = float(row["atr_norm"]) if "atr_norm" in row.index and pd.notna(row.get("atr_norm", np.nan)) else (
                atr_v / close_v if close_v > 0 else 0.0
            )
            should, reason, px = _chandelier_step(
                pos,
                float(row["high"]),
                float(row["low"]),
                close_v,
                atr_v,
                cfg,
                regime_on=regime_on,
                atr_norm=atr_n,
            )
            if reason == "adaptive_extend":
                adaptive_extend_count += 1
            if should:
                fill = px * (1 - cfg.slippage)
                proceeds = pos.shares * fill
                comm = proceeds * cfg.commission
                net = proceeds - comm
                cash += net
                pnl = net - pos.capital_used
                realized_pnl_by_ticker[t] = realized_pnl_by_ticker.get(t, 0.0) + float(pnl)
                trades.append(
                    {
                        "ticker": t,
                        "entry_date": pos.entry_date,
                        "exit_date": day,
                        "entry_price": pos.entry_price,
                        "exit_price": fill,
                        "shares": pos.shares,
                        "capital_used": pos.capital_used,
                        "net_profit": pnl,
                        "trade_return": pnl / pos.capital_used if pos.capital_used else 0.0,
                        "bars_held": pos.bars_held,
                        "exit_reason": reason,
                        "extensions": pos.extensions,
                    }
                )
                to_close.append(t)
        for t in to_close:
            if t in positions:
                del positions[t]

        # 1b) Optional M&A roll: open successor with delist proceeds
        for succ, dollars in pending_rolls:
            if succ in positions or dollars <= 0:
                continue
            sidx = by_date.get(succ, {}).get(day)
            if sidx is None:
                continue
            srow = frames[succ].iloc[sidx]
            spx = float(srow["close"]) * (1 + cfg.slippage)
            satr = float(srow["atr"]) if float(srow["atr"]) > 0 else spx * 0.02
            if spx <= 0:
                continue
            shares = int(dollars * 0.98 / spx)
            if shares <= 0:
                continue
            cost = shares * spx
            comm = cost * cfg.commission
            total = cost + comm
            if total > cash:
                continue
            cash -= total
            hard = compute_entry_hard_stop(spx, satr, cfg)
            trail = spx - cfg.k_atr * satr
            positions[succ] = OpenPosition(
                ticker=succ,
                entry_date=day,
                entry_idx=sidx,
                entry_price=spx,
                shares=shares,
                stop=trail,
                hard_stop=hard,
                highest_high=float(srow["high"]),
                bars_held=0,
                capital_used=total,
                horizon_limit=int(cfg.max_horizon),
                extensions=0,
                k_atr_scale=1.0,
            )

        # 2) Entries ranked by p_buy (+ optional rotation when full)
        can_enter = len(positions) < cfg.max_positions or bool(cfg.enable_rotation)
        if can_enter:
            # Regime: legacy hard-block OR soft size-scale (bottleneck fix EXP3)
            size_scale = 1.0
            if cfg.require_regime and cfg.regime_ok is not None:
                if not regime_on:
                    if cfg.soft_hard_regime:
                        hard_scale = (
                            cfg.regime_hard_size_scale
                            if cfg.regime_hard_size_scale is not None
                            else cfg.risk_off_scale
                        )
                        size_scale = float(hard_scale)
                    else:
                        equity_points.append((day, mark_to_market(day)))
                        peak_equity = max(peak_equity, equity_points[-1][1])
                        continue
            if cfg.soft_regime_ok is not None:
                soft = True
                if day in cfg.soft_regime_ok:
                    soft = cfg.soft_regime_ok[day]
                else:
                    prior = [d for d in cfg.soft_regime_ok if d <= day]
                    if prior:
                        soft = cfg.soft_regime_ok[max(prior)]
                if not soft:
                    # Stack soft size scale; do not overwrite a smaller hard-regime scale
                    size_scale = min(size_scale, float(cfg.risk_off_scale))

            candidates: List[Tuple[float, str, int]] = []
            for t, df in frames.items():
                if t in positions:
                    continue
                idx = by_date[t].get(day)
                if idx is None:
                    continue
                row = df.iloc[idx]
                if not bool(row["signal"]):
                    continue
                # Sector ETF below MA → do not enter this name
                if cfg.require_sector_trend and cfg.ticker_sector and cfg.sector_etf_maps:
                    if not sector_allows_entry(
                        t,
                        day,
                        ticker_sector=cfg.ticker_sector,
                        etf_maps=cfg.sector_etf_maps,
                        allow_if_unmapped=cfg.sector_allow_unmapped,
                    ):
                        continue
                candidates.append((float(row["p_buy"]), t, idx))
            candidates.sort(reverse=True)
            candidates = candidates[: max(1, cfg.max_entries_per_day)]

            eq_now = mark_to_market(day)
            peak_equity = max(peak_equity, eq_now)
            dd = eq_now / peak_equity - 1.0 if peak_equity > 0 else 0.0
            # Kill-switch: no new risk when portfolio DD breaches threshold
            if dd <= -abs(cfg.max_portfolio_dd):
                equity_points.append((day, eq_now))
                continue
            # Soft de-risk when deep drawdown (halfway to kill-switch)
            if cfg.max_portfolio_dd < 0.9 and dd <= -0.5 * abs(cfg.max_portfolio_dd):
                size_scale *= float(cfg.dd_soft_scale)

            # Optional rotation: free capital/slot if best candidate dominates worst held.
            # "Full" = max_positions hit OR cash too small to open another min_alloc position
            # (common with min_alloc + vol targeting: capital fills before slot count).
            rotations_today = 0
            if cfg.enable_rotation and positions and candidates:
                while rotations_today < max(1, cfg.rotation_max_per_day) and candidates:
                    eq_now = mark_to_market(day)
                    floor_need = (
                        float(cfg.min_alloc_pct) * eq_now if cfg.min_alloc_pct > 0 else eq_now * 0.02
                    )
                    slots_full = len(positions) >= cfg.max_positions
                    cash_full = cash < floor_need * 0.98
                    if not (slots_full or cash_full):
                        break
                    best_p, best_t, _best_idx = candidates[0]
                    # Score held names with today's p_buy if available
                    held_scores: List[Tuple[float, str]] = []
                    for ht, hpos in positions.items():
                        if hpos.bars_held < int(cfg.rotation_min_bars):
                            continue
                        hidx = by_date.get(ht, {}).get(day)
                        if hidx is None:
                            continue
                        hrow = frames[ht].iloc[hidx]
                        held_scores.append((float(hrow["p_buy"]), ht))
                    if not held_scores:
                        break
                    held_scores.sort()  # worst first
                    worst_p, worst_t = held_scores[0]
                    if best_p < worst_p + float(cfg.rotation_min_score_edge):
                        break
                    # Sell worst to free slot + cash
                    wpos = positions[worst_t]
                    widx = by_date[worst_t].get(day)
                    if widx is None:
                        break
                    wpx = float(frames[worst_t].iloc[widx]["close"]) * (1 - cfg.slippage)
                    proceeds = wpos.shares * wpx
                    comm = proceeds * cfg.commission
                    net = proceeds - comm
                    cash += net
                    pnl = net - wpos.capital_used
                    realized_pnl_by_ticker[worst_t] = (
                        realized_pnl_by_ticker.get(worst_t, 0.0) + float(pnl)
                    )
                    trades.append(
                        {
                            "ticker": worst_t,
                            "entry_date": wpos.entry_date,
                            "exit_date": day,
                            "entry_price": wpos.entry_price,
                            "exit_price": wpx,
                            "shares": wpos.shares,
                            "capital_used": wpos.capital_used,
                            "net_profit": pnl,
                            "trade_return": pnl / wpos.capital_used if wpos.capital_used else 0.0,
                            "bars_held": wpos.bars_held,
                            "exit_reason": "rotation",
                            "extensions": wpos.extensions,
                            "rotated_for": best_t,
                            "score_out": worst_p,
                            "score_in": best_p,
                        }
                    )
                    del positions[worst_t]
                    rotations_today += 1
                    eq_now = mark_to_market(day)

            for p_buy, t, idx in candidates:
                if len(positions) >= cfg.max_positions:
                    break
                # Cap re-entry on tickers that already minted outsized realized PnL
                if cfg.ticker_max_realized_pnl_frac < 0.99 and peak_equity > 0:
                    if realized_pnl_by_ticker.get(t, 0.0) >= cfg.ticker_max_realized_pnl_frac * peak_equity:
                        continue
                row = frames[t].iloc[idx]
                price = float(row["close"]) * (1 + cfg.slippage)
                atr = float(row["atr"])
                if price <= 0 or atr <= 0:
                    continue
                vol = max(atr / price, 1e-8)
                pos_cap = min(cfg.max_position_pct, cfg.ticker_max_capital_pct)
                alloc = eq_now * cfg.volatility_target_pct * size_scale / vol
                alloc = min(alloc, eq_now * pos_cap * size_scale, cash * 0.98)
                floor = float(cfg.min_alloc_pct) * eq_now if cfg.min_alloc_pct > 0.0 else 0.0
                # EXP1 pre-check: skip clearly sub-floor vol-target allocs early
                if floor > 0.0 and alloc < floor:
                    continue
                shares = int(alloc / price)
                if shares <= 0:
                    continue
                cost = shares * price
                comm = cost * cfg.commission
                total = cost + comm
                if total > cash:
                    continue
                # EXP1 post-int: capital_used after share rounding must still meet floor
                # (e.g. alloc=$1625, price=$1200 → 1 share → $1200 < 1.5% of equity)
                if floor > 0.0 and total < floor:
                    continue
                cash -= total
                # EXP4: ATR-aware hard stop (see compute_entry_hard_stop docstring)
                hard = compute_entry_hard_stop(price, atr, cfg)
                trail = price - cfg.k_atr * atr
                positions[t] = OpenPosition(
                    ticker=t,
                    entry_date=day,
                    entry_idx=idx,
                    entry_price=price,
                    shares=shares,
                    stop=trail,
                    hard_stop=hard,
                    highest_high=float(row["high"]),
                    bars_held=0,
                    capital_used=total,
                    horizon_limit=int(cfg.max_horizon),
                    extensions=0,
                    k_atr_scale=1.0,
                )

        rebalance_qqq_sleeve(day)
        equity_points.append((day, mark_to_market(day)))

    # Force close remaining at last day
    if positions and all_dates:
        day = all_dates[-1]
        for t, pos in list(positions.items()):
            idx = by_date[t].get(day)
            if idx is None:
                past = [d for d in by_date[t] if d <= day]
                idx = by_date[t][max(past)] if past else None
            if idx is None:
                continue
            px = float(frames[t].iloc[idx]["close"]) * (1 - cfg.slippage)
            proceeds = pos.shares * px
            comm = proceeds * cfg.commission
            net = proceeds - comm
            cash += net
            pnl = net - pos.capital_used
            trades.append(
                {
                    "ticker": t,
                    "entry_date": pos.entry_date,
                    "exit_date": day,
                    "entry_price": pos.entry_price,
                    "exit_price": px,
                    "shares": pos.shares,
                    "capital_used": pos.capital_used,
                    "net_profit": pnl,
                    "trade_return": pnl / pos.capital_used if pos.capital_used else 0.0,
                    "bars_held": pos.bars_held,
                    "exit_reason": "eod_force",
                }
            )
        positions.clear()
        equity_points.append((day, cash))

    trades_df = pd.DataFrame(trades)
    if equity_points:
        eq = pd.Series(
            {d: v for d, v in equity_points},
            dtype=float,
        ).sort_index()
        eq.index = pd.to_datetime(eq.index, utc=True)
    else:
        eq = pd.Series(dtype=float)
    return trades_df, eq, pd.DataFrame()
