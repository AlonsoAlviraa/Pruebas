"""Risk gates and margin-at-risk sizing for paper options (short premium).

VIRTUAL capital only. All marks remain ``proxy_bs`` unless a chain label is set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class OptionsRiskConfig:
    """Stricter gates for short-premium / defined-risk options books."""

    max_portfolio_dd: float = 0.15
    """Hard kill when peak-to-trough DD reaches this (absolute fraction)."""

    max_single_day_drop: float = 0.08
    """Hard kill when one-day portfolio drop exceeds this (consecutive sessions only)."""

    max_margin_fraction: float = 0.75
    """Max fraction of capital0 allocated as margin-at-risk when opening legs."""

    hard_kill_enabled: bool = True
    """If True, close structures and block new opens after a breach."""

    cvar_alpha: float = 0.05
    """Tail mass for CVaR reporting (not a live gate)."""

    min_contracts: int = 0
    """
    Minimum contracts required to open a structure.

    If margin budget cannot fund at least this many contracts, skip the open
    (return 0). Default 0 means "any size including zero / skip" — it is a
    soft require-at-least gate when set >= 1, not a forced upsize.
    """

    max_contracts: int = 20
    """Hard cap on contracts per structure (capacity / sanity)."""

    notes: str = "Short-vol gates: size by margin-at-risk; hard kill on DD / gap day."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Optional[Dict[str, Any]]) -> "OptionsRiskConfig":
        if not raw:
            return cls()
        return cls(
            max_portfolio_dd=float(raw.get("max_portfolio_dd", 0.15)),
            max_single_day_drop=float(raw.get("max_single_day_drop", 0.08)),
            max_margin_fraction=float(raw.get("max_margin_fraction", 0.75)),
            hard_kill_enabled=bool(raw.get("hard_kill_enabled", True)),
            cvar_alpha=float(raw.get("cvar_alpha", 0.05)),
            min_contracts=int(raw.get("min_contracts", 0)),
            max_contracts=int(raw.get("max_contracts", 20)),
            notes=str(raw.get("notes") or cls.notes),
        )


def margin_at_risk_per_contract(
    kind: str,
    *,
    spot: float,
    short_strike: Optional[float] = None,
    long_strike: Optional[float] = None,
    stock_shares: int = 100,
) -> float:
    """
    Margin / capital-at-risk for **one** option contract (or 100-share stock sleeve).

    - cash_secured_put: full cash collateral at short put strike
    - put_credit_spread: defined risk = width * 100 (max loss before credit)
    - covered_call / collar: stock notional for 100 shares (options are covered/hedged)
    - cash: 0
    """
    k = (kind or "").lower()
    if k == "cash":
        return 0.0
    if k in ("put_credit_spread", "call_credit_spread"):
        if short_strike is None or long_strike is None:
            width = abs(float(spot)) * 0.10
        else:
            width = abs(float(short_strike) - float(long_strike))
        return max(width, 0.0) * 100.0
    if k == "iron_condor":
        # Defined-risk: max loss ≈ larger wing width × 100 (proxy)
        if short_strike is None or long_strike is None:
            width = abs(float(spot)) * 0.10
        else:
            width = abs(float(short_strike) - float(long_strike))
        return max(width, 0.0) * 100.0
    if k == "cash_secured_put":
        sk = float(short_strike) if short_strike is not None else float(spot) * 0.95
        return max(sk, 0.0) * 100.0
    if k in ("covered_call", "collar", "protective_put"):
        sh = max(int(stock_shares), 100)
        return float(spot) * float(sh)
    sk = float(short_strike) if short_strike is not None else float(spot)
    return max(sk, 0.0) * 100.0


def size_contracts(
    kind: str,
    *,
    capital0: float,
    spot: float,
    risk: OptionsRiskConfig,
    short_strike: Optional[float] = None,
    long_strike: Optional[float] = None,
    stock_shares: int = 100,
    requested: int = 1,
) -> int:
    """
    Choose contract count so margin-at-risk ≤ capital0 * max_margin_fraction.

    Returns 0 if budget cannot fund one contract (or cannot meet min_contracts).
    Strict: no 1-lot fallback — callers must not bypass this gate silently.
    """
    mar = margin_at_risk_per_contract(
        kind,
        spot=spot,
        short_strike=short_strike,
        long_strike=long_strike,
        stock_shares=stock_shares,
    )
    if mar <= 0:
        return 0
    budget = float(capital0) * float(risk.max_margin_fraction)
    max_by_budget = int(budget // mar) if mar > 0 else 0
    n = min(int(requested), max_by_budget, int(risk.max_contracts))
    n = max(0, n)
    # require-at-least: if min_contracts >= 1 and we cannot fund that many, skip
    if int(risk.min_contracts) >= 1 and n < int(risk.min_contracts):
        return 0
    return n


def check_hard_kill(
    *,
    equity: float,
    peak: float,
    prev_equity: Optional[float],
    risk: OptionsRiskConfig,
) -> tuple[bool, str]:
    """
    Return (kill, reason) for portfolio DD or single-day drop breaches.

    Pass ``prev_equity=None`` after a multi-session gap so gap moves are not
    treated as a single-day drop (caller responsibility).
    """
    if not risk.hard_kill_enabled:
        return False, ""
    if peak > 0:
        dd = equity / peak - 1.0
        if dd <= -abs(float(risk.max_portfolio_dd)):
            return True, f"max_portfolio_dd={dd:.2%} <= -{abs(risk.max_portfolio_dd):.0%}"
    if prev_equity is not None and prev_equity > 0:
        day_ret = equity / prev_equity - 1.0
        if day_ret <= -abs(float(risk.max_single_day_drop)):
            return (
                True,
                f"max_single_day_drop={day_ret:.2%} <= -{abs(risk.max_single_day_drop):.0%}",
            )
    return False, ""
