"""Daily proxy replay for simple options strategies (BS marks on OHLCV).

LABEL: all option marks are ``proxy_bs`` — not exchange fills.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from paper_live.options.bs import black_scholes_price
from paper_live.options.strategies import OptionStrategySpec
from paper_live.options.vol_proxy import historical_vol, iv_proxy_from_hv


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
    n_rolls: int
    data_label: str = "proxy_bs"
    notes: List[str] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

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
            "n_rolls": self.n_rolls,
            "data_label": self.data_label,
            "notes": self.notes,
            "mode": "paper",
            "capital_label": "VIRTUAL",
        }


def _closes_series(feed, ticker: str, through: date) -> pd.Series:
    hist = feed.history(ticker, through=through, include_through=True)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    return hist.set_index("date")["close"].astype(float)


def run_options_strategy(
    feed,
    spec: OptionStrategySpec,
    *,
    start: date,
    end: date,
    capital0: float = 100_000.0,
) -> OptionsRunResult:
    """Replay a single options strategy with BS proxy marks."""
    days = feed.session_days(start, end)
    notes = [
        "OPTION MARKS ARE proxy_bs (Black–Scholes on HV/IV proxy). Not real chain fills.",
        f"kind={spec.kind} underlying={spec.underlying} dte={spec.dte_days} otm={spec.otm_pct}",
    ]
    if not days:
        return OptionsRunResult(
            spec.id, spec.label, spec.kind, spec.underlying, 0, capital0, 0.0, 0.0, 0, notes=notes
        )

    cash = float(capital0)
    stock_qty = 0.0
    # short call/put open
    short_call_k = None
    short_put_k = None
    long_put_k = None  # wing / collar
    expiry: Optional[date] = None
    n_rolls = 0
    curve: List[Dict[str, Any]] = []
    peak = capital0
    max_dd = 0.0

    und = spec.underlying.upper()

    def mark_option(spot: float, k: float, exp: date, day: date, otype: str, iv: float) -> float:
        t_years = max((exp - day).days, 0) / 365.0
        return black_scholes_price(
            spot, k, t_years, iv, r=spec.r, option_type=otype  # type: ignore[arg-type]
        )

    def equity(spot: float, day: date, iv: float) -> float:
        eq = cash + stock_qty * spot
        # short options: liability = +premium already in cash; mark short as -value
        if short_call_k is not None and expiry is not None:
            eq -= mark_option(spot, short_call_k, expiry, day, "call", iv) * 100.0 * spec.contracts
        if short_put_k is not None and expiry is not None:
            eq -= mark_option(spot, short_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
        if long_put_k is not None and expiry is not None:
            eq += mark_option(spot, long_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
        return eq

    def open_structure(spot: float, day: date, iv: float) -> None:
        nonlocal cash, stock_qty, short_call_k, short_put_k, long_put_k, expiry, n_rolls
        expiry = day + timedelta(days=int(spec.dte_days))
        # align expiry to calendar (ok if weekend — mark by days)
        if spec.kind == "cash":
            return
        if spec.kind in ("covered_call", "collar"):
            # buy shares if needed
            need = float(spec.stock_shares)
            cost = need * spot
            if stock_qty < need and cash >= cost:
                cash -= cost
                stock_qty = need
            short_call_k = spot * (1.0 + abs(spec.otm_pct))
            prem = mark_option(spot, short_call_k, expiry, day, "call", iv) * 100.0 * spec.contracts
            cash += prem  # sell call
            if spec.kind == "collar":
                long_put_k = spot * (1.0 - abs(spec.wing_otm_pct))
                pprem = mark_option(spot, long_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
                cash -= pprem
            n_rolls += 1
        elif spec.kind == "cash_secured_put":
            short_put_k = spot * (1.0 - abs(spec.otm_pct))
            # collateral: strike * 100
            coll = short_put_k * 100.0 * spec.contracts
            if cash < coll:
                return
            prem = mark_option(spot, short_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
            cash += prem
            # cash remains but reserved conceptually
            n_rolls += 1
        elif spec.kind == "put_credit_spread":
            short_put_k = spot * (1.0 - abs(spec.otm_pct))
            long_put_k = spot * (1.0 - abs(spec.wing_otm_pct))
            short_p = mark_option(spot, short_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
            long_p = mark_option(spot, long_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
            credit = short_p - long_p
            if credit > 0:
                cash += credit
                n_rolls += 1

    def close_structure(spot: float, day: date, iv: float) -> None:
        nonlocal cash, stock_qty, short_call_k, short_put_k, long_put_k, expiry
        if expiry is None:
            return
        if short_call_k is not None:
            # buy back call
            cash -= mark_option(spot, short_call_k, expiry, day, "call", iv) * 100.0 * spec.contracts
            short_call_k = None
        if short_put_k is not None:
            cash -= mark_option(spot, short_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
            short_put_k = None
        if long_put_k is not None:
            cash += mark_option(spot, long_put_k, expiry, day, "put", iv) * 100.0 * spec.contracts
            long_put_k = None
        expiry = None

    for day in days:
        bar = feed.bar(und, day)
        if bar is None:
            continue
        spot = float(bar.close)
        closes = _closes_series(feed, und, day)
        hv = historical_vol(closes, window=20)
        if not math.isfinite(hv) or hv <= 0:
            hv = 0.20
        # VRP gate stub for OPT06
        if spec.meta.get("require_hv_above_median") and len(closes) > 60:
            hv_long = historical_vol(closes, window=60)
            if math.isfinite(hv_long) and hv < hv_long * 0.9:
                # stay flat / don't open new — still mark existing
                iv = iv_proxy_from_hv(hv, premium_mult=spec.premium_mult)
                eq = equity(spot, day, iv)
                peak = max(peak, eq)
                max_dd = min(max_dd, eq / peak - 1.0)
                curve.append({"date": day.isoformat(), "equity": eq})
                continue

        iv = iv_proxy_from_hv(hv, premium_mult=spec.premium_mult)
        if not math.isfinite(iv):
            iv = 0.22

        if spec.kind == "cash":
            eq = cash
            curve.append({"date": day.isoformat(), "equity": eq})
            continue

        # roll if no structure or near expiry
        need_open = expiry is None
        if expiry is not None and (expiry - day).days <= int(spec.roll_when_dte_below):
            close_structure(spot, day, iv)
            need_open = True
        if need_open:
            open_structure(spot, day, iv)

        eq = equity(spot, day, iv)
        peak = max(peak, eq)
        max_dd = min(max_dd, eq / peak - 1.0 if peak > 0 else 0.0)
        curve.append({"date": day.isoformat(), "equity": eq})

    final = curve[-1]["equity"] if curve else capital0
    return OptionsRunResult(
        strategy_id=spec.id,
        label=spec.label,
        kind=spec.kind,
        underlying=und,
        days_run=len(curve),
        final_equity=float(final),
        total_return=float(final / capital0 - 1.0),
        max_dd=float(max_dd),
        n_rolls=n_rolls,
        data_label="proxy_bs",
        notes=notes,
        equity_curve=curve,
    )


def run_options_batch(
    feed,
    specs: Sequence[OptionStrategySpec],
    *,
    start: date,
    end: date,
    capital0: float = 100_000.0,
) -> List[OptionsRunResult]:
    out: List[OptionsRunResult] = []
    for sp in specs:
        out.append(run_options_strategy(feed, sp, start=start, end=end, capital0=capital0))
    return out
