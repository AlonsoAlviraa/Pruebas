"""Option strategy specifications (paper / research)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OptionStrategySpec:
    id: str
    label: str
    kind: str  # covered_call | cash_secured_put | put_credit_spread | collar | cash
    underlying: str = "SPY"
    dte_days: int = 30
    otm_pct: float = 0.05  # 5% OTM short strike
    wing_otm_pct: float = 0.15  # for spreads / collar long put
    target_delta: Optional[float] = None
    premium_mult: float = 1.15  # HV fallback only: IV = HV * mult when VIX surface missing
    contracts: int = 1  # requested contracts (capped by margin budget)
    stock_shares: int = 100
    r: float = 0.02
    roll_when_dte_below: int = 7
    # Per-strategy risk overrides (None → inherit OptionsRiskConfig + kind floors)
    max_portfolio_dd: Optional[float] = None
    max_single_day_drop: Optional[float] = None
    max_margin_fraction: Optional[float] = None
    hard_kill_enabled: Optional[bool] = None
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
            max_portfolio_dd=0.18,
            max_margin_fraction=0.95,
            notes="Long 100 SPY + short OTM call; VRP income, capped upside.",
        ),
        OptionStrategySpec(
            id="OPT02_csp",
            label="Cash-secured put ~5% OTM 30DTE (proxy BS)",
            kind="cash_secured_put",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            max_portfolio_dd=0.15,
            max_margin_fraction=0.80,
            notes="Short OTM put fully cash-secured; classic VRP / PUT-like.",
        ),
        OptionStrategySpec(
            id="OPT02b_csp_10otm",
            label="CSP ~10% OTM 45DTE (proxy BS)",
            kind="cash_secured_put",
            underlying="SPY",
            dte_days=45,
            otm_pct=0.10,
            max_portfolio_dd=0.15,
            max_margin_fraction=0.80,
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
            max_portfolio_dd=0.12,
            max_single_day_drop=0.06,
            max_margin_fraction=0.40,
            notes="Defined-risk VRP: short put + long lower put; size by width margin.",
        ),
        OptionStrategySpec(
            id="OPT04_collar",
            label="Collar long stock + long put + short call",
            kind="collar",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            wing_otm_pct=0.08,
            max_portfolio_dd=0.12,
            max_margin_fraction=0.95,
            notes="Defensive equity sleeve; defined-ish risk via long put wing.",
        ),
        OptionStrategySpec(
            id="OPT06_csp_vrp_gate",
            label="CSP only if HV elevated vs long-run (VRP gate stub)",
            kind="cash_secured_put",
            underlying="SPY",
            dte_days=30,
            otm_pct=0.05,
            premium_mult=1.20,
            max_portfolio_dd=0.15,
            max_margin_fraction=0.80,
            meta={"require_hv_above_median": True},
            notes="Sell premium only when short-term HV regime allows richer IV proxy.",
        ),
        OptionStrategySpec(
            id="OPT08_cash",
            label="Cash control (no options)",
            kind="cash",
            underlying="SPY",
            hard_kill_enabled=False,
            notes="Floor benchmark: idle virtual cash.",
        ),
        OptionStrategySpec(
            id="OPT_QQQ_cc",
            label="Covered call on QQQ ~5% OTM",
            kind="covered_call",
            underlying="QQQ",
            dte_days=30,
            otm_pct=0.05,
            max_portfolio_dd=0.18,
            max_margin_fraction=0.95,
            notes="Same as OPT01 on Nasdaq proxy.",
        ),
    ]
