"""RSK-02: Deployment policies — sizing/regime overlays without ML retrain.

Policies are pre-registered. Do not search or retune constants on foreign OOS
(e.g. ES 2018–2025). product_mode is an *evaluation outcome*, not a policy field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DeploymentPolicy:
    """Execution/risk overlay applied after strategy.backtest_overrides()."""

    policy_id: str
    region: str  # "US" | "ES" | "DE" | ...
    preferred_index: tuple[str, ...]
    regime_filter: Optional[str] = None  # None = inherit strategy.regime_filter
    vol_target_scale: float = 1.0
    max_position_scale: float = 1.0
    min_confidence_delta: float = 0.0
    max_portfolio_dd: float = 0.99
    commission: Optional[float] = None
    slippage: Optional[float] = None
    require_trend: Optional[bool] = None  # None = inherit strategy/cfg

    def to_backtest_overrides(self, base_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge policy scales onto strategy overrides (policy wins on keys it sets)."""
        o = dict(base_overrides or {})
        if "volatility_target_pct" in o:
            o["volatility_target_pct"] = float(o["volatility_target_pct"]) * self.vol_target_scale
        if "max_position_pct" in o:
            o["max_position_pct"] = float(o["max_position_pct"]) * self.max_position_scale
        if "min_confidence" in o:
            o["min_confidence"] = float(o["min_confidence"]) + self.min_confidence_delta
        o["max_portfolio_dd"] = self.max_portfolio_dd
        if self.commission is not None:
            o["commission"] = self.commission
        if self.slippage is not None:
            o["slippage"] = self.slippage
        if self.require_trend is not None:
            o["require_trend"] = self.require_trend
        return o


# Pre-registered policies (user 2026-07-17: portable_conservative = 0.6× + portable regime)
POLICIES: Dict[str, DeploymentPolicy] = {
    "us_research_default": DeploymentPolicy(
        policy_id="us_research_default",
        region="US",
        preferred_index=("QQQ", "SPY"),
        regime_filter=None,
        vol_target_scale=1.0,
        max_position_scale=1.0,
    ),
    "us_turbo_strict": DeploymentPolicy(
        policy_id="us_turbo_strict",
        region="US",
        preferred_index=("QQQ", "SPY"),
        regime_filter="strict_dual_golden",
        vol_target_scale=1.0,
        max_position_scale=1.0,
    ),
    "portable_conservative": DeploymentPolicy(
        policy_id="portable_conservative",
        region="ES",
        preferred_index=("IBEX",),
        # Economic prior / not-deep-bear; informed by IBEX design 2010–17, frozen, not OOS-ranked
        regime_filter="portable_not_deep_bear",
        vol_target_scale=0.6,
        max_position_scale=0.6,
        min_confidence_delta=0.05,
        max_portfolio_dd=0.35,
    ),
    "portable_defensive": DeploymentPolicy(
        policy_id="portable_defensive",
        region="ES",
        preferred_index=("IBEX",),
        regime_filter="portable_sma200",
        vol_target_scale=0.5,
        max_position_scale=0.5,
        min_confidence_delta=0.08,
        max_portfolio_dd=0.20,
    ),
}


def get_policy(policy_id: str) -> DeploymentPolicy:
    if policy_id not in POLICIES:
        raise KeyError(f"Unknown policy {policy_id!r}. Available: {list(POLICIES)}")
    return POLICIES[policy_id]


def list_policies() -> List[str]:
    return list(POLICIES.keys())
