"""Option strategy specifications (paper / research)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OptionStrategySpec:
    id: str
    label: str
    kind: str  # covered_call | cash_secured_put | put_credit_spread | collar | qqq_hold_control
    underlying: str = "SPY"
    dte_days: int = 30
    otm_pct: float = 0.05  # 5% OTM
    wing_otm_pct: float = 0.15  # for spreads
    target_delta: Optional[float] = None
    premium_mult: float = 1.15  # IV = HV * mult
    contracts: int = 1  # per 100 shares notionally
    stock_shares: int = 100
    r: float = 0.02
    roll_when_dte_below: int = 7
    notes: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def list_builtin_specs() -> List[OptionStrategySpec]:
    """Catalog aligned with design doc OPT01–OPT08 (v0 implementable set)."""
    return [
        OptionStrategySpec(
            id="OPT01_covered_call",
            label="Covered call ~5% OTM 30DTE (proxy BS)",
            kind="covered_call",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            notes="Long 100 SPY + short OTM call; VRP income, capped upside.",
        ),
        OptionStrategySpec(
            id="OPT02_csp",
            label="Cash-secured put ~5% OTM 30DTE (proxy BS)",
            kind="cash_secured_put",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            notes="Short OTM put fully cash-secured; classic VRP / PUT-like.",
        ),
        OptionStrategySpec(
            id="OPT02b_csp_10otm",
            label="CSP ~10% OTM 45DTE (proxy BS)",
            kind="cash_secured_put",
            underlying="SPY",
            dte_days=45,
            otm_pct=0.10,
            notes="Farther OTM put-write; literature often favors 5–10% OTM.",
        ),
        OptionStrategySpec(
            id="OPT03_put_credit_spread",
            label="Bull put credit spread 5%/15% OTM 30DTE",
            kind="put_credit_spread",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            wing_otm_pct=0.15,
            notes="Defined-risk VRP: short put + long lower put.",
        ),
        OptionStrategySpec(
            id="OPT04_collar",
            label="Collar long stock + long put + short call",
            kind="collar",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            wing_otm_pct=0.08,
            notes="Defensive equity sleeve.",
        ),
        OptionStrategySpec(
            id="OPT06_csp_vrp_gate",
            label="CSP only if HV elevated vs long-run (VRP gate stub)",
            kind="cash_secured_put",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            premium_mult=1.20,
            meta={"require_hv_above_median": True},
            notes="Sell premium only when short-term HV regime allows richer IV proxy.",
        ),
        OptionStrategySpec(
            id="OPT08_cash",
            label="Cash control (no options)",
            kind="cash",
            underlying="SPY",
            notes="Floor benchmark: idle virtual cash.",
        ),
        OptionStrategySpec(
            id="OPT_QQQ_cc",
            label="Covered call on QQQ ~5% OTM",
            kind="covered_call",
            underlying="QQQ",
            dte_days=30,
            otm_pct=0.05,
            notes="Same as OPT01 on Nasdaq proxy.",
        ),
    ]
