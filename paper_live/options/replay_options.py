"""Daily proxy replay for options strategies (BS marks on OHLCV).

LABELS:
  - Option marks: ``proxy_bs`` math on model IV
  - IV source: ``vix_surface`` | ``proxy_hv`` | ``vix_surface_partial``
  - Fills: bid haircut on sells (not NBBO)
  - Assignment: ``assignment_proxy`` (simple equity rules)

Risk: margin-at-risk sizing + hard kill on portfolio DD / single-day drop.
Premium-seller mgmt: 50% credit TP, 2× credit stop, max 1 roll (configurable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from paper_live.options.bs import black_scholes_price, bs_delta
from paper_live.options.management import (
    SHORT_PREMIUM_KINDS,
    apply_bid_haircut,
    can_roll,
    check_assignment,
    management_action,
    management_from_meta,
    structure_mark_to_close,
)
from paper_live.options.metrics import metrics_from_curve
from paper_live.options.risk import (
    OptionsRiskConfig,
    check_hard_kill,
    size_contracts,
)
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.ta_gates import evaluate_ta_gates
from paper_live.options.vol_proxy import historical_vol
from paper_live.options.vol_surface import (
    aggregate_surface_label,
    iv_from_surface,
    resolve_vix_level,
    VIX_TICKERS,
    VIX3M_TICKERS,
    VXST_TICKERS,
)


@dataclass
class OptionsRunResult:
    strategy_id: str
    label: str
    kind: str
    underlying: str
    days_run: int
    final_equity: float
    total_return: float
    max_dd: float
    n_rolls: int  # legacy alias of n_opens (total successful structure opens)
    n_opens: int = 0  # every successful open (initial + post-roll)
    n_dte_rolls: int = 0  # DTE rolls only (capped by max_rolls per structure)
    data_label: str = "proxy_bs"
    iv_source: str = "proxy_hv"
    notes: List[str] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    cvar_5pct: Optional[float] = None
    calmar_like: Optional[float] = None
    worst_day: Optional[float] = None
    worst_month: Optional[float] = None
    hard_kill: bool = False
    hard_kill_reason: str = ""
    contracts_used: int = 0
    margin_at_risk: float = 0.0
    defined_risk: bool = False
    risk_config: Dict[str, Any] = field(default_factory=dict)
    vs_spy_bh: Optional[float] = None
    vs_qqq_bh: Optional[float] = None
    approx_delta_end: Optional[float] = None
    approx_delta_avg: Optional[float] = None
    n_tp: int = 0
    n_sl: int = 0
    n_time_exit: int = 0
    n_assign: int = 0
    management: Dict[str, Any] = field(default_factory=dict)
    exit_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "label": self.label,
            "kind": self.kind,
            "underlying": self.underlying,
            "days_run": self.days_run,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "max_dd": self.max_dd,
            "cvar_5pct": self.cvar_5pct,
            "calmar_like": self.calmar_like,
            "worst_day": self.worst_day,
            "worst_month": self.worst_month,
            "n_rolls": self.n_rolls,  # legacy = n_opens
            "n_opens": self.n_opens,
            "n_dte_rolls": self.n_dte_rolls,
            "hard_kill": self.hard_kill,
            "hard_kill_reason": self.hard_kill_reason,
            "contracts_used": self.contracts_used,
            "margin_at_risk": self.margin_at_risk,
            "defined_risk": self.defined_risk,
            "risk_config": self.risk_config,
            "vs_spy_bh": self.vs_spy_bh,
            "vs_qqq_bh": self.vs_qqq_bh,
            "data_label": self.data_label,
            "iv_source": self.iv_source,
            "approx_delta_end": self.approx_delta_end,
            "approx_delta_avg": self.approx_delta_avg,
            "n_tp": self.n_tp,
            "n_sl": self.n_sl,
            "n_time_exit": self.n_time_exit,
            "n_assign": self.n_assign,
            "exit_breakdown": self.exit_breakdown,
            "management": self.management,
            "notes": self.notes,
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }


def _closes_series(feed, ticker: str, through: date) -> pd.Series:
    # Prefer fast numpy path on DailyReplayFeed
    if hasattr(feed, "closes_through"):
        arr = feed.closes_through(ticker, through, include_through=True)
        if arr is not None and len(arr) > 0:
            return pd.Series(arr, dtype=float)
    hist = feed.history(ticker, through=through, include_through=True)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    return hist.set_index("date")["close"].astype(float)


def _precompute_day_macro(
    feed,
    und: str,
    days: Sequence[date],
) -> Tuple[Dict[date, float], Dict[date, Optional[float]], Dict[date, Optional[float]], Dict[date, Optional[float]], Dict[date, np.ndarray]]:
    """Precompute HV20 + VIX levels + close prefixes for a session range."""
    hv_map: Dict[date, float] = {}
    vix_map: Dict[date, Optional[float]] = {}
    vix3m_map: Dict[date, Optional[float]] = {}
    vxst_map: Dict[date, Optional[float]] = {}
    closes_map: Dict[date, np.ndarray] = {}

    # Full closes once; rolling HV via numpy
    if hasattr(feed, "closes_through") and days:
        full = np.asarray(
            feed.closes_through(und, days[-1], include_through=True), dtype=float
        )
    else:
        ser = _closes_series(feed, und, days[-1] if days else date.today())
        full = ser.to_numpy(dtype=float) if len(ser) else np.asarray([], dtype=float)

    # Map day -> end exclusive index into full using feed day list if present
    day_to_idx: Dict[date, int] = {}
    if hasattr(feed, "_day_list") and und.upper() in getattr(feed, "_day_list", {}):
        for i, d in enumerate(feed._day_list[und.upper()]):
            day_to_idx[d] = i + 1  # exclusive
    else:
        # rebuild from history dates
        hist = feed.history(und, through=days[-1], include_through=True) if days else None
        if hist is not None and not hist.empty:
            for i, raw_d in enumerate(hist["date"]):
                dd = pd.Timestamp(raw_d).date()
                day_to_idx[dd] = i + 1

    logret = np.full(len(full), np.nan, dtype=float)
    if len(full) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            logret[1:] = np.log(full[1:] / full[:-1])

    for day in days:
        idx = day_to_idx.get(day)
        if idx is None:
            # nearest prior
            candidates = [d for d in day_to_idx if d <= day]
            if not candidates:
                hv_map[day] = 0.20
                closes_map[day] = np.asarray([], dtype=float)
                vix_map[day] = resolve_vix_level(feed, day, aliases=VIX_TICKERS)
                vix3m_map[day] = resolve_vix_level(feed, day, aliases=VIX3M_TICKERS)
                vxst_map[day] = resolve_vix_level(feed, day, aliases=VXST_TICKERS)
                continue
            idx = day_to_idx[max(candidates)]
        prefix = full[:idx]
        closes_map[day] = prefix
        if idx >= 21:
            window = logret[idx - 20 : idx]
            s = float(np.nanstd(window, ddof=1))
            hv = s * math.sqrt(252.0) if math.isfinite(s) and s > 0 else 0.20
        else:
            hv = 0.20
        hv_map[day] = hv if math.isfinite(hv) and hv > 0 else 0.20
        # VIX from bars only (avoid history rebuild)
        vix_map[day] = None
        for t in VIX_TICKERS:
            b = feed.bar(t, day)
            if b is not None and float(b.close) > 0:
                vix_map[day] = float(b.close)
                break
        if vix_map[day] is None:
            vix_map[day] = resolve_vix_level(feed, day, aliases=VIX_TICKERS)
        vix3m_map[day] = None
        for t in VIX3M_TICKERS:
            b = feed.bar(t, day)
            if b is not None and float(b.close) > 0:
                vix3m_map[day] = float(b.close)
                break
        if vix3m_map[day] is None:
            vix3m_map[day] = resolve_vix_level(feed, day, aliases=VIX3M_TICKERS)
        vxst_map[day] = None
        for t in VXST_TICKERS:
            b = feed.bar(t, day)
            if b is not None and float(b.close) > 0:
                vxst_map[day] = float(b.close)
                break
        if vxst_map[day] is None:
            vxst_map[day] = resolve_vix_level(feed, day, aliases=VXST_TICKERS)

    return hv_map, vix_map, vix3m_map, vxst_map, closes_map


def _is_defined_risk(kind: str) -> bool:
    return kind in (
        "put_credit_spread",
        "call_credit_spread",
        "iron_condor",
        "collar",
        "cash",
        "call_debit_spread",
        "put_debit_spread",
        "long_call",
        "long_put",
        "pmcc",
    )


def _risk_for_spec(
    spec: OptionStrategySpec,
    base: Optional[OptionsRiskConfig],
) -> OptionsRiskConfig:
    """
    Merge global risk defaults with per-strategy overrides.

    Spec risk fields default to None → inherit from base.
    Kind-aware margin floors apply only when max_margin_fraction is None
    (so explicit zoo values always bind; no silent 1-lot bypass).
    """
    b = base or OptionsRiskConfig()
    frac = (
        float(spec.max_margin_fraction)
        if spec.max_margin_fraction is not None
        else float(b.max_margin_fraction)
    )
    if spec.max_margin_fraction is None:
        if spec.kind in ("covered_call", "collar"):
            frac = max(frac, 0.95)
        elif spec.kind == "cash_secured_put":
            frac = max(frac, 0.75)

    hard = (
        bool(spec.hard_kill_enabled)
        if spec.hard_kill_enabled is not None
        else bool(b.hard_kill_enabled)
    )

    return OptionsRiskConfig(
        max_portfolio_dd=float(
            spec.max_portfolio_dd if spec.max_portfolio_dd is not None else b.max_portfolio_dd
        ),
        max_single_day_drop=float(
            spec.max_single_day_drop
            if spec.max_single_day_drop is not None
            else b.max_single_day_drop
        ),
        max_margin_fraction=float(frac),
        hard_kill_enabled=hard,
        cvar_alpha=float(b.cvar_alpha),
        min_contracts=int(b.min_contracts),
        max_contracts=int(b.max_contracts),
        notes=b.notes,
    )


def _surface_iv(
    feed,
    day: date,
    *,
    spot: float,
    strike: float,
    expiry: date,
    option_type: str,
    hv: float,
    premium_mult: float,
    vix: Optional[float],
    vix3m: Optional[float],
    vxst: Optional[float],
) -> Tuple[float, str]:
    t_years = max((expiry - day).days, 0) / 365.0
    q = iv_from_surface(
        t_years=t_years,
        spot=spot,
        strike=strike,
        option_type=option_type,
        vix=vix,
        vix3m=vix3m,
        vxst=vxst,
        hv=hv,
        premium_mult=premium_mult,
    )
    return float(q.iv), str(q.source)


def run_options_strategy(
    feed,
    spec: OptionStrategySpec,
    *,
    start: date,
    end: date,
    capital0: float = 100_000.0,
    risk: Optional[OptionsRiskConfig] = None,
    data_label: str = "proxy_bs",
    compute_delta: bool = True,
    store_curve: bool = True,
) -> OptionsRunResult:
    """Replay a single options strategy with BS proxy marks + risk gates.

    ``compute_delta=False`` skips daily BS-delta (large speedup for bulk studies).
    ``store_curve=False`` keeps only lightweight metrics (less memory/JSON bloat).
    """
    risk_cfg = _risk_for_spec(spec, risk)
    mgmt = management_from_meta(spec.meta, kind=spec.kind)
    days = feed.session_days(start, end)
    defined = _is_defined_risk(spec.kind)
    notes = [
        "OPTION MARKS ARE model BS (proxy_bs math). Not real chain fills.",
        f"kind={spec.kind} underlying={spec.underlying} dte={spec.dte_days} otm={spec.otm_pct}",
        f"defined_risk={defined} max_portfolio_dd={risk_cfg.max_portfolio_dd} "
        f"max_day_drop={risk_cfg.max_single_day_drop} max_margin_frac={risk_cfg.max_margin_fraction}",
        "Margin sizing is strict: contracts ≤ capital0 * max_margin_fraction / margin_at_risk "
        "(no 1-lot fallback).",
        f"IV surface: VIX/VIX3M when available else proxy_hv; bid_haircut={mgmt.bid_haircut:.2%}; "
        f"TP={mgmt.take_profit_credit_frac:.0%} credit, SL={mgmt.stop_loss_credit_mult}× credit, "
        f"max_rolls={mgmt.max_rolls}; assignment_proxy={mgmt.enable_assignment_proxy}.",
    ]
    if not days:
        return OptionsRunResult(
            strategy_id=spec.id,
            label=spec.label,
            kind=spec.kind,
            underlying=spec.underlying,
            days_run=0,
            final_equity=capital0,
            total_return=0.0,
            max_dd=0.0,
            n_rolls=0,
            n_opens=0,
            n_dte_rolls=0,
            data_label=data_label,
            iv_source="proxy_hv",
            notes=notes,
            defined_risk=defined,
            risk_config=risk_cfg.to_dict(),
            management=mgmt.to_dict(),
        )

    cash = float(capital0)
    stock_qty = 0.0
    short_call_k: Optional[float] = None
    short_put_k: Optional[float] = None
    long_put_k: Optional[float] = None
    long_call_k: Optional[float] = None
    expiry: Optional[date] = None
    long_expiry: Optional[date] = None  # LEAP / far leg (PMCC); None → use expiry
    open_contracts = 0
    initial_debit = 0.0  # cash paid for long-premium structures
    max_contracts_used = 0
    last_margin = 0.0
    n_opens = 0  # every successful open
    n_dte_rolls = 0  # DTE rolls only
    n_rolls = 0  # legacy alias kept in sync with n_opens
    rolls_this_structure = 0  # completed DTE rolls on current structure lifetime
    initial_credit = 0.0  # cash credit received at open (after haircut), dollars
    n_tp = 0
    n_sl = 0
    n_time_exit = 0
    n_assign = 0
    curve: List[Dict[str, Any]] = []
    peak = float(capital0)
    max_dd = 0.0
    hard_kill = False
    hard_kill_reason = ""
    prev_eq: Optional[float] = None
    skip_open_logged = False
    gap_pending = False
    iv_sources_seen: List[str] = []
    delta_samples: List[float] = []

    und = spec.underlying.upper()
    requested = max(int(spec.contracts), 1)
    haircut = float(mgmt.bid_haircut)

    def _log_skip(msg: str) -> None:
        nonlocal skip_open_logged
        if not skip_open_logged:
            notes.append(msg + " (further skips suppressed)")
            skip_open_logged = True

    def _quote_iv(
        spot: float,
        k: float,
        exp: date,
        day: date,
        otype: str,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
    ) -> float:
        iv, src = _surface_iv(
            feed,
            day,
            spot=spot,
            strike=k,
            expiry=exp,
            option_type=otype,
            hv=hv,
            premium_mult=spec.premium_mult,
            vix=vix,
            vix3m=vix3m,
            vxst=vxst,
        )
        iv_sources_seen.append(src)
        if not math.isfinite(iv):
            return 0.22
        return iv

    def mark_option(
        spot: float,
        k: float,
        exp: date,
        day: date,
        otype: str,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
        *,
        side: str = "mid",
    ) -> float:
        """Per-unit option mid (or haircut mid) via BS + surface IV."""
        iv = _quote_iv(spot, k, exp, day, otype, hv, vix, vix3m, vxst)
        t_years = max((exp - day).days, 0) / 365.0
        mid = black_scholes_price(
            spot, k, t_years, iv, r=spec.r, option_type=otype  # type: ignore[arg-type]
        )
        if side == "sell":
            return apply_bid_haircut(mid, side="sell", haircut=haircut)
        if side == "buy":
            return apply_bid_haircut(mid, side="buy", haircut=haircut)
        return float(mid)

    def mid_leg(
        spot: float,
        k: Optional[float],
        exp: Optional[date],
        day: date,
        otype: str,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
    ) -> float:
        if k is None or exp is None:
            return 0.0
        return mark_option(spot, k, exp, day, otype, hv, vix, vix3m, vxst, side="mid")

    def equity(
        spot: float,
        day: date,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
    ) -> float:
        eq = cash + stock_qty * spot
        n = max(open_contracts, 0)
        if n <= 0 or expiry is None:
            return eq
        if short_call_k is not None:
            eq -= mark_option(spot, short_call_k, expiry, day, "call", hv, vix, vix3m, vxst) * 100.0 * n
        if short_put_k is not None:
            eq -= mark_option(spot, short_put_k, expiry, day, "put", hv, vix, vix3m, vxst) * 100.0 * n
        lexp = long_expiry or expiry
        if long_put_k is not None:
            eq += mark_option(spot, long_put_k, lexp, day, "put", hv, vix, vix3m, vxst) * 100.0 * n
        if long_call_k is not None:
            eq += mark_option(spot, long_call_k, lexp, day, "call", hv, vix, vix3m, vxst) * 100.0 * n
        return eq

    def book_delta(
        spot: float,
        day: date,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
    ) -> float:
        """Approximate share-equivalent delta (stock + 100×contracts×BS delta)."""
        d = float(stock_qty)
        n = max(open_contracts, 0)
        if n <= 0 or expiry is None:
            return d
        t_near = max((expiry - day).days, 0) / 365.0
        lexp = long_expiry or expiry
        t_long = max((lexp - day).days, 0) / 365.0 if lexp else t_near

        def _d(k: Optional[float], otype: str, sign: float, *, far: bool = False) -> float:
            if k is None:
                return 0.0
            exp_u = lexp if far and lexp is not None else expiry
            if exp_u is None:
                return 0.0
            t_y = t_long if far else t_near
            iv = _quote_iv(spot, k, exp_u, day, otype, hv, vix, vix3m, vxst)
            return sign * bs_delta(
                spot, k, t_y, iv, r=spec.r, option_type=otype  # type: ignore[arg-type]
            ) * 100.0 * n

        d += _d(short_call_k, "call", -1.0, far=False)
        d += _d(short_put_k, "put", -1.0, far=False)
        d += _d(long_put_k, "put", +1.0, far=True)
        d += _d(long_call_k, "call", +1.0, far=True)
        return d

    def mark_close_debit(
        spot: float,
        day: date,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
    ) -> float:
        lexp = long_expiry or expiry
        return structure_mark_to_close(
            short_call_mid=mid_leg(spot, short_call_k, expiry, day, "call", hv, vix, vix3m, vxst),
            short_put_mid=mid_leg(spot, short_put_k, expiry, day, "put", hv, vix, vix3m, vxst),
            long_call_mid=mid_leg(spot, long_call_k, lexp, day, "call", hv, vix, vix3m, vxst),
            long_put_mid=mid_leg(spot, long_put_k, lexp, day, "put", hv, vix, vix3m, vxst),
            contracts=open_contracts,
        )

    def clear_option_legs() -> None:
        nonlocal short_call_k, short_put_k, long_put_k, long_call_k
        nonlocal expiry, long_expiry, open_contracts, initial_credit, initial_debit
        nonlocal rolls_this_structure
        short_call_k = None
        short_put_k = None
        long_put_k = None
        long_call_k = None
        open_contracts = 0
        expiry = None
        long_expiry = None
        initial_credit = 0.0
        initial_debit = 0.0
        rolls_this_structure = 0

    def close_structure(
        spot: float,
        day: date,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
        *,
        keep_structure_count: bool = False,
    ) -> None:
        nonlocal cash, short_call_k, short_put_k, long_put_k, long_call_k
        nonlocal expiry, long_expiry, open_contracts, initial_credit, initial_debit
        nonlocal rolls_this_structure
        if (
            expiry is None
            and long_expiry is None
            and short_call_k is None
            and short_put_k is None
            and long_put_k is None
            and long_call_k is None
        ):
            return
        n = max(open_contracts, 0)
        lexp = long_expiry or expiry
        if n > 0:
            # Close at mid (research default); haircut already hit entry credit
            if short_call_k is not None and expiry is not None:
                cash -= mark_option(
                    spot, short_call_k, expiry, day, "call", hv, vix, vix3m, vxst, side="mid"
                ) * 100.0 * n
            if short_put_k is not None and expiry is not None:
                cash -= mark_option(
                    spot, short_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="mid"
                ) * 100.0 * n
            if long_put_k is not None and lexp is not None:
                cash += mark_option(
                    spot, long_put_k, lexp, day, "put", hv, vix, vix3m, vxst, side="mid"
                ) * 100.0 * n
            if long_call_k is not None and lexp is not None:
                cash += mark_option(
                    spot, long_call_k, lexp, day, "call", hv, vix, vix3m, vxst, side="mid"
                ) * 100.0 * n
        short_call_k = None
        short_put_k = None
        long_put_k = None
        long_call_k = None
        open_contracts = 0
        expiry = None
        long_expiry = None
        initial_credit = 0.0
        initial_debit = 0.0
        if not keep_structure_count:
            rolls_this_structure = 0

    def liquidate_all(
        spot: float,
        day: date,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
    ) -> None:
        nonlocal cash, stock_qty
        close_structure(spot, day, hv, vix, vix3m, vxst)
        if stock_qty > 0:
            cash += stock_qty * spot
            stock_qty = 0.0

    def apply_assignment(
        spot: float,
        day: date,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
        *,
        at_expiry: bool,
    ) -> bool:
        """
        Apply assignment proxy and settle remaining legs in cash.

        Assigned shorts: stock/cash from AssignmentEvent (no mid buyback).
        Remaining shorts (e.g. unassigned IC wing): closed at mid.
        Long wings: always settled at mid (or intrinsic at expiry via BS t→0)
        so defined-risk structures do not forfeit paid long-option value.
        After any assignment, the structure is fully flattened (no half-books).
        Label: assignment_proxy (not OCC / index cash-settlement fidelity).
        """
        nonlocal cash, stock_qty, n_assign, short_call_k, short_put_k
        nonlocal long_put_k, long_call_k, open_contracts, expiry, initial_credit
        nonlocal rolls_this_structure
        n = max(open_contracts, 0)
        events = check_assignment(
            spot=spot,
            short_put_k=short_put_k,
            short_call_k=short_call_k,
            contracts=n,
            stock_qty=stock_qty,
            at_expiry=at_expiry,
            deep_itm_pct=mgmt.deep_itm_assign_pct,
            enabled=mgmt.enable_assignment_proxy,
        )
        if not events:
            return False

        assigned_put = False
        assigned_call = False
        for ev in events:
            cash += ev.cash_delta
            stock_qty += ev.shares_delta
            n_assign += 1
            notes.append(
                f"ASSIGNMENT_PROXY {day.isoformat()} {ev.leg} K={ev.strike:.2f} "
                f"n={ev.contracts} reason={ev.reason} label={ev.label}"
            )
            if ev.leg == "short_put":
                short_put_k = None
                assigned_put = True
            if ev.leg == "short_call":
                short_call_k = None
                assigned_call = True

        # Settle any *unassigned* remaining shorts at mid (buy to close)
        if n > 0 and expiry is not None:
            if short_put_k is not None and not assigned_put:
                cash -= (
                    mark_option(
                        spot, short_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="mid"
                    )
                    * 100.0
                    * n
                )
                short_put_k = None
            if short_call_k is not None and not assigned_call:
                cash -= (
                    mark_option(
                        spot, short_call_k, expiry, day, "call", hv, vix, vix3m, vxst, side="mid"
                    )
                    * 100.0
                    * n
                )
                short_call_k = None

            # Always credit long wings (mid ≈ intrinsic near/at expiry) — never wipe free
            if long_put_k is not None:
                long_val = (
                    mark_option(
                        spot, long_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="mid"
                    )
                    * 100.0
                    * n
                )
                cash += long_val
                notes.append(
                    f"ASSIGNMENT_PROXY settle long_put K={long_put_k:.2f} "
                    f"cash+={long_val:.2f} (not forfeited)"
                )
                long_put_k = None
            if long_call_k is not None:
                long_val = (
                    mark_option(
                        spot, long_call_k, expiry, day, "call", hv, vix, vix3m, vxst, side="mid"
                    )
                    * 100.0
                    * n
                )
                cash += long_val
                notes.append(
                    f"ASSIGNMENT_PROXY settle long_call K={long_call_k:.2f} "
                    f"cash+={long_val:.2f} (not forfeited)"
                )
                long_call_k = None

        open_contracts = 0
        expiry = None
        initial_credit = 0.0
        rolls_this_structure = 0
        return True

    def _record_open(*, is_roll: bool) -> None:
        """Increment open/roll counters after a successful structure open."""
        nonlocal n_opens, n_dte_rolls, n_rolls, rolls_this_structure
        n_opens += 1
        n_rolls = n_opens  # legacy alias
        if is_roll:
            n_dte_rolls += 1
            rolls_this_structure += 1
        else:
            rolls_this_structure = 0

    def open_structure(
        spot: float,
        day: date,
        hv: float,
        vix: Optional[float],
        vix3m: Optional[float],
        vxst: Optional[float],
        *,
        is_roll: bool = False,
    ) -> None:
        nonlocal cash, stock_qty, short_call_k, short_put_k, long_put_k, long_call_k
        nonlocal expiry, long_expiry, open_contracts, max_contracts_used, last_margin
        nonlocal initial_credit, initial_debit, rolls_this_structure
        if hard_kill:
            return
        expiry = day + timedelta(days=int(spec.dte_days))
        long_expiry = None
        initial_debit = 0.0
        if spec.kind == "cash":
            open_contracts = 0
            return

        if spec.kind in ("covered_call", "collar"):
            lot_cost = spot * 100.0
            held_lots = int(stock_qty // 100.0) if stock_qty >= 100 else 0
            budget = float(capital0) * float(risk_cfg.max_margin_fraction)
            n_budget = int(budget // lot_cost) if lot_cost > 0 else 0
            n_cash = int(cash // lot_cost) if lot_cost > 0 else 0
            if held_lots > 0:
                n = min(requested, held_lots, int(risk_cfg.max_contracts))
            else:
                n = min(requested, n_budget, n_cash, int(risk_cfg.max_contracts))
            if n <= 0:
                _log_skip(f"{day.isoformat()}: skip open — margin budget 0 contracts")
                expiry = None
                return
            need = 100.0 * float(n)
            stock_bought_this_open = 0.0
            if stock_qty + 1e-9 < need:
                buy = need - stock_qty
                cost = buy * spot
                if cash < cost:
                    afford = int((stock_qty + cash / spot) // 100.0) if spot > 0 else 0
                    n = min(n, afford)
                    if n <= 0:
                        expiry = None
                        return
                    need = 100.0 * float(n)
                    buy = need - stock_qty
                    cost = buy * spot
                if buy > 0:
                    cash -= cost
                    stock_qty += buy
                    stock_bought_this_open = buy

            def _unwind_stock_on_refuse() -> None:
                nonlocal cash, stock_qty
                if stock_bought_this_open > 0 and stock_qty + 1e-9 >= stock_bought_this_open:
                    cash += stock_bought_this_open * spot
                    stock_qty -= stock_bought_this_open
                if spec.kind == "collar" and stock_qty > 0:
                    cash += stock_qty * spot
                    stock_qty = 0.0

            def _trim_excess_stock(target_shares: float) -> None:
                nonlocal cash, stock_qty
                excess = stock_qty - target_shares
                if excess > 1e-9:
                    cash += excess * spot
                    stock_qty = target_shares

            short_call_k = spot * (1.0 + abs(spec.otm_pct))
            call_prem = (
                mark_option(
                    spot, short_call_k, expiry, day, "call", hv, vix, vix3m, vxst, side="sell"
                )
                * 100.0
                * n
            )

            if spec.kind == "collar":
                long_put_k = spot * (1.0 - abs(spec.wing_otm_pct))
                put_prem = (
                    mark_option(
                        spot, long_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="buy"
                    )
                    * 100.0
                    * n
                )
                if cash + call_prem + 1e-9 < put_prem:
                    while n > 0 and cash + (
                        mark_option(
                            spot, short_call_k, expiry, day, "call", hv, vix, vix3m, vxst, side="sell"
                        )
                        * 100.0
                        * n
                    ) + 1e-9 < (
                        mark_option(
                            spot, long_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="buy"
                        )
                        * 100.0
                        * n
                    ):
                        n -= 1
                    if n <= 0:
                        _log_skip(f"{day.isoformat()}: collar skip — put debit not affordable")
                        short_call_k = None
                        long_put_k = None
                        expiry = None
                        _unwind_stock_on_refuse()
                        return
                    need = 100.0 * float(n)
                    _trim_excess_stock(need)
                    call_prem = (
                        mark_option(
                            spot, short_call_k, expiry, day, "call", hv, vix, vix3m, vxst, side="sell"
                        )
                        * 100.0
                        * n
                    )
                    put_prem = (
                        mark_option(
                            spot, long_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="buy"
                        )
                        * 100.0
                        * n
                    )
                cash += call_prem
                cash -= put_prem
                if cash < -1e-6:
                    cash -= call_prem
                    cash += put_prem
                    short_call_k = None
                    long_put_k = None
                    expiry = None
                    _log_skip(f"{day.isoformat()}: collar skip — cash would go negative")
                    _unwind_stock_on_refuse()
                    return
                initial_credit = max(call_prem - put_prem, 0.0)
            else:
                cash += call_prem
                long_put_k = None
                _trim_excess_stock(100.0 * float(n))
                initial_credit = call_prem

            open_contracts = n
            last_margin = lot_cost * n
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "cash_secured_put":
            sk = spot * (1.0 - abs(spec.otm_pct))
            n = size_contracts(
                "cash_secured_put",
                capital0=capital0,
                spot=spot,
                risk=risk_cfg,
                short_strike=sk,
                requested=requested,
            )
            if n <= 0:
                _log_skip(f"{day.isoformat()}: CSP skip — insufficient margin budget")
                expiry = None
                return
            coll = sk * 100.0 * n
            if cash < coll:
                n = int(cash // (sk * 100.0)) if sk > 0 else 0
                if n <= 0:
                    expiry = None
                    return
                coll = sk * 100.0 * n
            short_put_k = sk
            prem = (
                mark_option(
                    spot, short_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="sell"
                )
                * 100.0
                * n
            )
            cash += prem
            initial_credit = prem
            open_contracts = n
            last_margin = coll
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "put_credit_spread":
            sk = spot * (1.0 - abs(spec.otm_pct))
            lk = spot * (1.0 - abs(spec.wing_otm_pct))
            if lk >= sk:
                lk = sk * (1.0 - 0.05)
            n = size_contracts(
                "put_credit_spread",
                capital0=capital0,
                spot=spot,
                risk=risk_cfg,
                short_strike=sk,
                long_strike=lk,
                requested=requested,
            )
            if n <= 0:
                _log_skip(f"{day.isoformat()}: PCS skip — insufficient defined-risk budget")
                expiry = None
                return
            width = abs(sk - lk)
            max_loss = width * 100.0 * n
            if cash < max_loss:
                n = int(cash // (width * 100.0)) if width > 0 else 0
                if n <= 0:
                    expiry = None
                    return
                max_loss = width * 100.0 * n
            short_p = (
                mark_option(spot, sk, expiry, day, "put", hv, vix, vix3m, vxst, side="sell")
                * 100.0
                * n
            )
            long_p = (
                mark_option(spot, lk, expiry, day, "put", hv, vix, vix3m, vxst, side="buy")
                * 100.0
                * n
            )
            credit = short_p - long_p
            if credit <= 0:
                _log_skip(f"{day.isoformat()}: PCS skip — non-positive credit (proxy)")
                expiry = None
                return
            short_put_k = sk
            long_put_k = lk
            cash += credit
            initial_credit = credit
            open_contracts = n
            last_margin = max_loss
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "call_credit_spread":
            sk = spot * (1.0 + abs(spec.otm_pct))
            lk = spot * (1.0 + abs(spec.wing_otm_pct))
            if lk <= sk:
                lk = sk * (1.0 + 0.05)
            n = size_contracts(
                "call_credit_spread",
                capital0=capital0,
                spot=spot,
                risk=risk_cfg,
                short_strike=sk,
                long_strike=lk,
                requested=requested,
            )
            if n <= 0:
                _log_skip(f"{day.isoformat()}: CCS skip — insufficient budget")
                expiry = None
                return
            width = abs(lk - sk)
            max_loss = width * 100.0 * n
            if cash < max_loss:
                n = int(cash // (width * 100.0)) if width > 0 else 0
                if n <= 0:
                    expiry = None
                    return
                max_loss = width * 100.0 * n
            short_c = (
                mark_option(spot, sk, expiry, day, "call", hv, vix, vix3m, vxst, side="sell")
                * 100.0
                * n
            )
            long_c = (
                mark_option(spot, lk, expiry, day, "call", hv, vix, vix3m, vxst, side="buy")
                * 100.0
                * n
            )
            credit = short_c - long_c
            if credit <= 0:
                _log_skip(f"{day.isoformat()}: CCS skip — non-positive credit")
                expiry = None
                return
            short_call_k = sk
            long_call_k = lk
            cash += credit
            initial_credit = credit
            open_contracts = n
            last_margin = max_loss
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "iron_condor":
            put_short = spot * (1.0 - abs(spec.otm_pct))
            put_long = spot * (1.0 - abs(spec.wing_otm_pct))
            call_short = spot * (1.0 + abs(spec.otm_pct))
            call_long = spot * (1.0 + abs(spec.wing_otm_pct))
            if put_long >= put_short:
                put_long = put_short * 0.95
            if call_long <= call_short:
                call_long = call_short * 1.05
            wing_w = max(abs(put_short - put_long), abs(call_long - call_short))
            n = size_contracts(
                "iron_condor",
                capital0=capital0,
                spot=spot,
                risk=risk_cfg,
                short_strike=put_short,
                long_strike=put_long,
                requested=requested,
            )
            if n <= 0:
                _log_skip(f"{day.isoformat()}: IC skip — insufficient budget")
                expiry = None
                return
            max_loss = wing_w * 100.0 * n
            if cash < max_loss:
                n = int(cash // (wing_w * 100.0)) if wing_w > 0 else 0
                if n <= 0:
                    expiry = None
                    return
                max_loss = wing_w * 100.0 * n
            sp = (
                mark_option(spot, put_short, expiry, day, "put", hv, vix, vix3m, vxst, side="sell")
                * 100.0
                * n
            )
            lp = (
                mark_option(spot, put_long, expiry, day, "put", hv, vix, vix3m, vxst, side="buy")
                * 100.0
                * n
            )
            sc = (
                mark_option(spot, call_short, expiry, day, "call", hv, vix, vix3m, vxst, side="sell")
                * 100.0
                * n
            )
            lc = (
                mark_option(spot, call_long, expiry, day, "call", hv, vix, vix3m, vxst, side="buy")
                * 100.0
                * n
            )
            credit = (sp - lp) + (sc - lc)
            if credit <= 0:
                _log_skip(f"{day.isoformat()}: IC skip — non-positive credit")
                expiry = None
                return
            short_put_k = put_short
            long_put_k = put_long
            short_call_k = call_short
            long_call_k = call_long
            cash += credit
            initial_credit = credit
            open_contracts = n
            last_margin = max_loss
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "protective_put":
            lot_cost = spot * 100.0
            budget = float(capital0) * float(risk_cfg.max_margin_fraction)
            n_budget = int(budget // lot_cost) if lot_cost > 0 else 0
            n_cash = int(cash // lot_cost) if lot_cost > 0 else 0
            n = min(requested, n_budget, n_cash, int(risk_cfg.max_contracts))
            if n <= 0:
                _log_skip(f"{day.isoformat()}: protective put skip — no stock budget")
                expiry = None
                return
            need = 100.0 * float(n)
            if stock_qty + 1e-9 < need:
                buy = need - stock_qty
                cost = buy * spot
                if cash < cost:
                    n = int((stock_qty + cash / spot) // 100.0) if spot > 0 else 0
                    if n <= 0:
                        expiry = None
                        return
                    need = 100.0 * float(n)
                    buy = need - stock_qty
                    cost = buy * spot
                if buy > 0:
                    cash -= cost
                    stock_qty += buy
            long_put_k = spot * (1.0 - abs(spec.otm_pct))
            pprem = (
                mark_option(
                    spot, long_put_k, expiry, day, "put", hv, vix, vix3m, vxst, side="buy"
                )
                * 100.0
                * n
            )
            if cash < pprem:
                cash += stock_qty * spot
                stock_qty = 0.0
                long_put_k = None
                expiry = None
                _log_skip(f"{day.isoformat()}: protective put skip — put unaffordable")
                return
            cash -= pprem
            initial_credit = 0.0  # long premium
            initial_debit = pprem
            long_expiry = expiry
            open_contracts = n
            last_margin = lot_cost * n
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        # --- Amplify family: long premium / debit spreads / PMCC ---
        budget_frac = float(spec.meta.get("max_premium_budget_frac") or 0.10)

        if spec.kind == "long_call":
            k = spot * (1.0 + abs(spec.otm_pct))
            # size by premium budget
            unit = mark_option(spot, k, expiry, day, "call", hv, vix, vix3m, vxst, side="buy") * 100.0
            if unit <= 0 or not math.isfinite(unit):
                expiry = None
                return
            budget = float(capital0) * budget_frac
            n = min(
                requested,
                int(risk_cfg.max_contracts),
                int(budget // unit) if unit > 0 else 0,
                int(cash // unit) if unit > 0 else 0,
            )
            if n <= 0:
                _log_skip(f"{day.isoformat()}: long_call skip — budget/cash")
                expiry = None
                return
            debit = unit * n
            cash -= debit
            long_call_k = k
            long_expiry = expiry
            initial_debit = debit
            initial_credit = 0.0
            open_contracts = n
            last_margin = debit
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "long_put":
            k = spot * (1.0 - abs(spec.otm_pct))
            unit = mark_option(spot, k, expiry, day, "put", hv, vix, vix3m, vxst, side="buy") * 100.0
            if unit <= 0 or not math.isfinite(unit):
                expiry = None
                return
            budget = float(capital0) * budget_frac
            n = min(
                requested,
                int(risk_cfg.max_contracts),
                int(budget // unit) if unit > 0 else 0,
                int(cash // unit) if unit > 0 else 0,
            )
            if n <= 0:
                _log_skip(f"{day.isoformat()}: long_put skip — budget/cash")
                expiry = None
                return
            debit = unit * n
            cash -= debit
            long_put_k = k
            long_expiry = expiry
            initial_debit = debit
            initial_credit = 0.0
            open_contracts = n
            last_margin = debit
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "call_debit_spread":
            # long lower strike call, short higher (bull call)
            lk = spot * (1.0 + abs(spec.otm_pct) * 0.5)  # closer ATM long
            sk = spot * (1.0 + abs(spec.wing_otm_pct))
            if sk <= lk:
                sk = lk * 1.05
            long_u = mark_option(spot, lk, expiry, day, "call", hv, vix, vix3m, vxst, side="buy")
            short_u = mark_option(spot, sk, expiry, day, "call", hv, vix, vix3m, vxst, side="sell")
            net = (long_u - short_u) * 100.0
            if net <= 0 or not math.isfinite(net):
                _log_skip(f"{day.isoformat()}: CDS skip — non-positive debit")
                expiry = None
                return
            width = abs(sk - lk) * 100.0
            budget = float(capital0) * budget_frac
            n = min(
                requested,
                int(risk_cfg.max_contracts),
                int(budget // net) if net > 0 else 0,
                int(cash // net) if net > 0 else 0,
            )
            if n <= 0:
                expiry = None
                return
            debit = net * n
            cash -= debit
            long_call_k = lk
            short_call_k = sk
            long_expiry = expiry
            initial_debit = debit
            initial_credit = 0.0
            open_contracts = n
            last_margin = width * n
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "put_debit_spread":
            # long higher put, short lower put (bear put)
            lk = spot * (1.0 - abs(spec.otm_pct) * 0.5)
            sk = spot * (1.0 - abs(spec.wing_otm_pct))
            if sk >= lk:
                sk = lk * 0.95
            long_u = mark_option(spot, lk, expiry, day, "put", hv, vix, vix3m, vxst, side="buy")
            short_u = mark_option(spot, sk, expiry, day, "put", hv, vix, vix3m, vxst, side="sell")
            net = (long_u - short_u) * 100.0
            if net <= 0 or not math.isfinite(net):
                expiry = None
                return
            width = abs(lk - sk) * 100.0
            budget = float(capital0) * budget_frac
            n = min(
                requested,
                int(risk_cfg.max_contracts),
                int(budget // net) if net > 0 else 0,
                int(cash // net) if net > 0 else 0,
            )
            if n <= 0:
                expiry = None
                return
            debit = net * n
            cash -= debit
            long_put_k = lk
            short_put_k = sk
            long_expiry = expiry
            initial_debit = debit
            initial_credit = 0.0
            open_contracts = n
            last_margin = width * n
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        if spec.kind == "pmcc":
            # Poor man's covered call: long LEAP call + short near call
            leap_dte = int(spec.meta.get("leap_dte_days") or 180)
            leap_exp = day + timedelta(days=leap_dte)
            near_k = spot * (1.0 + abs(spec.otm_pct))
            far_k = spot * (1.0 + float(spec.meta.get("leap_otm_pct") or 0.05))
            long_u = mark_option(spot, far_k, leap_exp, day, "call", hv, vix, vix3m, vxst, side="buy")
            short_u = mark_option(spot, near_k, expiry, day, "call", hv, vix, vix3m, vxst, side="sell")
            net = (long_u - short_u) * 100.0
            if net <= 0 or not math.isfinite(net):
                # still allow pure long LEAP if short worthless
                net = long_u * 100.0
            budget = float(capital0) * max(budget_frac, 0.15)
            n = min(
                requested,
                int(risk_cfg.max_contracts),
                int(budget // net) if net > 0 else 0,
                int(cash // net) if net > 0 else 0,
            )
            if n <= 0:
                _log_skip(f"{day.isoformat()}: pmcc skip — budget")
                expiry = None
                return
            debit = net * n
            cash -= debit
            long_call_k = far_k
            short_call_k = near_k
            long_expiry = leap_exp
            # near expiry stays as expiry for short roll
            initial_debit = debit
            initial_credit = max(short_u * 100.0 * n, 0.0)  # short credit portion for mgmt
            open_contracts = n
            last_margin = debit
            max_contracts_used = max(max_contracts_used, n)
            _record_open(is_roll=is_roll)
            return

        notes.append(f"unknown kind={spec.kind}; no open")
        expiry = None

    hv_map, vix_map, vix3m_map, vxst_map, closes_map = _precompute_day_macro(
        feed, und, days
    )
    # Warm featured cache once if any TA gate is active (avoids per-day re-engineer)
    _ta_keys = (
        "require_uptrend",
        "require_sma200",
        "require_volume_confirm",
        "require_volume_dryup",
        "require_rsi_oversold",
        "require_rsi_overbought",
        "require_low_atr",
        "require_range_regime",
        "require_vol_climax",
        "require_compression_after_vol",
        "require_pullback_uptrend",
        "require_iv_rank_above",
        "require_iv_rank_below",
        "require_vrp_proxy_above",
        "require_vrp_proxy_below",
        "require_vix_term_contango",
    )
    has_ta = bool(spec.meta) and any(spec.meta.get(k) for k in _ta_keys)
    if has_ta and hasattr(feed, "featured") and days:
        try:
            feed.featured(und, through=days[-1])
        except Exception:
            pass

    for day in days:
        bar = feed.bar(und, day)
        if bar is None:
            prev_eq = None
            gap_pending = True
            continue
        spot = float(bar.close)
        session_gap = bool(gap_pending)
        gap_pending = False
        closes_arr = closes_map.get(day)
        if closes_arr is None:
            closes = _closes_series(feed, und, day)
            closes_arr = closes.to_numpy(dtype=float) if len(closes) else np.asarray([], dtype=float)
        else:
            closes = pd.Series(closes_arr, dtype=float)
        hv = float(hv_map.get(day) or 0.20)
        if not math.isfinite(hv) or hv <= 0:
            hv = 0.20

        vix = vix_map.get(day)
        vix3m = vix3m_map.get(day)
        vxst = vxst_map.get(day)

        skip_new = False
        if spec.meta.get("require_hv_above_median") and len(closes_arr) > 60:
            hv_long = historical_vol(closes, window=60)
            if math.isfinite(hv_long) and hv < hv_long * 0.9:
                skip_new = True
        if not skip_new and has_ta:
            ta = evaluate_ta_gates(feed, und, day, spec.meta)
            if not ta.allow:
                skip_new = True
                if not skip_open_logged and ta.reason not in ("no_ta_gates",):
                    _log_skip(f"{day.isoformat()}: TA gate skip — {ta.reason}")

        if spec.kind == "cash":
            eq = cash
            peak = max(peak, eq)
            max_dd = min(max_dd, eq / peak - 1.0 if peak > 0 else 0.0)
            curve.append(
                {
                    "date": day.isoformat(),
                    "equity": eq,
                    "contracts": 0,
                    "session_gap": session_gap,
                    "delta": 0.0,
                }
            )
            prev_eq = eq
            continue

        if hard_kill:
            eq = cash + stock_qty * spot
            if short_call_k or short_put_k or long_put_k or long_call_k or stock_qty > 0:
                liquidate_all(spot, day, hv, vix, vix3m, vxst)
                eq = cash
            peak = max(peak, eq)
            max_dd = min(max_dd, eq / peak - 1.0 if peak > 0 else 0.0)
            curve.append(
                {
                    "date": day.isoformat(),
                    "equity": eq,
                    "contracts": 0,
                    "hard_kill": True,
                    "session_gap": session_gap,
                    "delta": 0.0,
                }
            )
            prev_eq = eq
            continue

        # --- Expiry / deep ITM assignment proxy ---
        # Near-leg expiry (short premium / debit spreads); LEAP long may remain via long_expiry
        if expiry is not None and open_contracts > 0:
            at_exp = day >= expiry
            if at_exp or mgmt.enable_assignment_proxy:
                # Deep ITM / expiry assignment; multi-leg longs settled inside apply_assignment
                assigned = apply_assignment(
                    spot, day, hv, vix, vix3m, vxst, at_expiry=at_exp
                )
                if at_exp and not assigned:
                    # expire worthless / settle mid (no assignment triggered)
                    close_structure(spot, day, hv, vix, vix3m, vxst)
                # If assigned: structure already fully flattened with long-wing cash settle

        # --- Premium seller TP / SL / time exit ---
        if (
            expiry is not None
            and open_contracts > 0
            and initial_credit > 0
            and spec.kind in SHORT_PREMIUM_KINDS
        ):
            mtc = mark_close_debit(spot, day, hv, vix, vix3m, vxst)
            dte_now = max((expiry - day).days, 0)
            act = management_action(
                kind=spec.kind,
                initial_credit=initial_credit,
                mark_to_close=mtc,
                cfg=mgmt,
                dte=dte_now,
            )
            if act == "take_profit":
                close_structure(spot, day, hv, vix, vix3m, vxst)
                n_tp += 1
                notes.append(f"TP {day.isoformat()}: credit captured ≥ {mgmt.take_profit_credit_frac:.0%}")
            elif act == "stop_loss":
                close_structure(spot, day, hv, vix, vix3m, vxst)
                n_sl += 1
                notes.append(
                    f"SL {day.isoformat()}: loss ≥ {mgmt.stop_loss_credit_mult}× initial credit"
                )
            elif act == "time_exit":
                close_structure(spot, day, hv, vix, vix3m, vxst)
                n_time_exit += 1
                notes.append(
                    f"TIME_EXIT {day.isoformat()}: DTE≤{mgmt.time_exit_dte} "
                    f"residual≤{mgmt.time_exit_residual_credit_frac:.0%}"
                )

        # --- Roll / open ---
        need_open = expiry is None and not (
            short_call_k or short_put_k or long_put_k or long_call_k
        )
        is_roll = False
        if expiry is not None and (expiry - day).days <= int(spec.roll_when_dte_below):
            if can_roll(rolls_this_structure, mgmt.max_rolls):
                close_structure(spot, day, hv, vix, vix3m, vxst, keep_structure_count=True)
                need_open = True
                is_roll = True
            else:
                # Max rolls exhausted: hold to expiry (no reopen); only close when DTE<=0 handled above
                pass

        if need_open and not skip_new:
            open_structure(spot, day, hv, vix, vix3m, vxst, is_roll=is_roll)

        eq = equity(spot, day, hv, vix, vix3m, vxst)
        dlt = 0.0
        if compute_delta:
            dlt = book_delta(spot, day, hv, vix, vix3m, vxst)
            delta_samples.append(dlt)
        peak = max(peak, eq)
        max_dd = min(max_dd, eq / peak - 1.0 if peak > 0 else 0.0)

        kill, reason = check_hard_kill(
            equity=eq, peak=peak, prev_equity=prev_eq, risk=risk_cfg
        )
        if kill:
            hard_kill = True
            hard_kill_reason = reason
            liquidate_all(spot, day, hv, vix, vix3m, vxst)
            eq = cash + stock_qty * spot
            notes.append(f"HARD_KILL {day.isoformat()}: {reason}")

        if store_curve:
            curve.append(
                {
                    "date": day.isoformat(),
                    "equity": eq,
                    "contracts": open_contracts if not hard_kill else 0,
                    "hard_kill": hard_kill,
                    "session_gap": session_gap,
                    "delta": dlt if not hard_kill else 0.0,
                    "vix": vix,
                }
            )
        else:
            # Minimal curve points for metrics_from_curve (start + each day ret)
            if not curve:
                curve.append({"date": day.isoformat(), "equity": float(capital0)})
            curve.append({"date": day.isoformat(), "equity": eq})
        prev_eq = eq

    iv_src = aggregate_surface_label(iv_sources_seen)
    # Compose data_label: model marks + IV source
    if data_label in ("proxy_bs", "proxy_bs_stress") or data_label.startswith("proxy"):
        run_label = f"{data_label}|{iv_src}" if data_label else iv_src
    else:
        run_label = f"{data_label}|{iv_src}"

    m = metrics_from_curve(curve, capital0=capital0, cvar_alpha=risk_cfg.cvar_alpha)
    delta_end = delta_samples[-1] if delta_samples else 0.0
    delta_avg = float(sum(delta_samples) / len(delta_samples)) if delta_samples else 0.0

    return OptionsRunResult(
        strategy_id=spec.id,
        label=spec.label,
        kind=spec.kind,
        underlying=und,
        days_run=int(m["n_days"]),
        final_equity=float(m["final_equity"]),
        total_return=float(m["total_return"]),
        max_dd=float(m["max_dd"]),
        n_rolls=n_opens,  # legacy alias of n_opens
        n_opens=n_opens,
        n_dte_rolls=n_dte_rolls,
        data_label=run_label,
        iv_source=iv_src,
        notes=notes,
        equity_curve=curve,
        cvar_5pct=m.get("cvar_5pct"),
        calmar_like=m.get("calmar_like"),
        worst_day=m.get("worst_day"),
        worst_month=m.get("worst_month"),
        hard_kill=hard_kill,
        hard_kill_reason=hard_kill_reason,
        contracts_used=max_contracts_used,
        margin_at_risk=float(last_margin),
        defined_risk=defined,
        risk_config=risk_cfg.to_dict(),
        approx_delta_end=delta_end,
        approx_delta_avg=delta_avg,
        n_tp=n_tp,
        n_sl=n_sl,
        n_time_exit=n_time_exit,
        n_assign=n_assign,
        exit_breakdown={
            "take_profit": n_tp,
            "stop_loss": n_sl,
            "time_exit": n_time_exit,
            "assignment": n_assign,
            "dte_rolls": n_dte_rolls,
            "opens": n_opens,
        },
        management=mgmt.to_dict(),
    )


def run_options_batch(
    feed,
    specs: Sequence[OptionStrategySpec],
    *,
    start: date,
    end: date,
    capital0: float = 100_000.0,
    risk: Optional[OptionsRiskConfig] = None,
    data_label: str = "proxy_bs",
    spy_bh: Optional[float] = None,
    qqq_bh: Optional[float] = None,
) -> List[OptionsRunResult]:
    out: List[OptionsRunResult] = []
    for sp in specs:
        r = run_options_strategy(
            feed,
            sp,
            start=start,
            end=end,
            capital0=capital0,
            risk=risk,
            data_label=data_label,
        )
        if spy_bh is not None:
            r.vs_spy_bh = float(r.total_return) - float(spy_bh)
        if qqq_bh is not None:
            r.vs_qqq_bh = float(r.total_return) - float(qqq_bh)
        out.append(r)
    return out


def book_delta_report(results: Sequence[OptionsRunResult]) -> Dict[str, Any]:
    """Aggregate approximate portfolio delta across strategies (paper book)."""
    end_sum = 0.0
    avg_sum = 0.0
    n = 0
    per: List[Dict[str, Any]] = []
    for r in results:
        de = r.approx_delta_end
        da = r.approx_delta_avg
        if de is None and da is None:
            continue
        end_sum += float(de or 0.0)
        avg_sum += float(da or 0.0)
        n += 1
        per.append(
            {
                "strategy_id": r.strategy_id,
                "underlying": r.underlying,
                "approx_delta_end": de,
                "approx_delta_avg": da,
            }
        )
    return {
        "n_strategies": n,
        "sum_delta_end": end_sum,
        "sum_delta_avg": avg_sum,
        "mean_delta_end": end_sum / n if n else 0.0,
        "mean_delta_avg": avg_sum / n if n else 0.0,
        "label": "approx_bs_delta_book",
        "note": "Sum of per-strategy share-equivalent BS deltas (not beta-weighted).",
        "strategies": per,
    }
