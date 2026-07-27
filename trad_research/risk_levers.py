"""Pure risk levers for research A/B (MDD attack) — no ML retrain.

Primary week-plan lever: portfolio drawdown circuit (max_portfolio_dd).
Alt-loop combos: DD circuit + vol-target scale + optional position cap scale.

Research only. Not financial advice. Does not claim live edge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RiskMddLever:
    """Single-knob (or frozen combo) risk overlay on strategy backtest overrides.

    ``lever_id`` conventions:
      - ``baseline``: circuit disabled (max_portfolio_dd ~ 0.99)
      - ``dd_circuit_25``: block new entries when portfolio DD ≤ −25%
      - ``vol_target_tight_70``: scale volatility_target_pct × 0.70 (helper only)
      - ``dd25_vt70``: DD 25% circuit + vol target ×0.70 (alt-loop primary)
      - ``dd20_vt60``: DD 20% + vol ×0.60 (stronger MDD attack)
      - ``dd18_vt70_pos75``: DD 18% + vol ×0.70 + max_position ×0.75
    """

    lever_id: str
    max_portfolio_dd: float = 0.99
    vol_target_scale: float = 1.0
    max_position_scale: float = 1.0
    dd_soft_scale: float = 0.55
    # None = hard block on breach; float = size scale when DD ≤ −max_portfolio_dd
    dd_breach_size_scale: Optional[float] = None
    risk_off_scale: Optional[float] = None  # None = leave strategy default
    # Mega-study peak carry: continuous | yearly (yearly avoids permanent-cash trap)
    peak_mode: str = "continuous"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Pre-registered week-plan lever (do not search/retune on OOS)
WEEK_PRIMARY_LEVER_ID = "dd_circuit_25"
# Alt-loop primary combo (MDD attack)
ALT_PRIMARY_LEVER_ID = "dd25_vt70"

LEVERS: Dict[str, RiskMddLever] = {
    "baseline": RiskMddLever(
        lever_id="baseline",
        max_portfolio_dd=0.99,
        vol_target_scale=1.0,
        max_position_scale=1.0,
        dd_soft_scale=0.55,
        description="No portfolio DD circuit (legacy research default)",
    ),
    "dd_circuit_25": RiskMddLever(
        lever_id="dd_circuit_25",
        max_portfolio_dd=0.25,
        vol_target_scale=1.0,
        max_position_scale=1.0,
        dd_soft_scale=0.50,
        description="MDD attack: kill new entries at −25% peak-to-trough; soft scale halfway",
    ),
    "vol_target_tight_70": RiskMddLever(
        lever_id="vol_target_tight_70",
        max_portfolio_dd=0.99,
        vol_target_scale=0.70,
        max_position_scale=1.0,
        dd_soft_scale=0.55,
        description="Helper: tighten vol target to 70% of strategy default (not week primary A/B)",
    ),
    "dd25_vt70": RiskMddLever(
        lever_id="dd25_vt70",
        max_portfolio_dd=0.25,
        vol_target_scale=0.70,
        max_position_scale=1.0,
        dd_soft_scale=0.50,
        description="Alt-loop primary: DD circuit −25% + vol-target ×0.70",
    ),
    "dd20_vt60": RiskMddLever(
        lever_id="dd20_vt60",
        max_portfolio_dd=0.20,
        vol_target_scale=0.60,
        max_position_scale=1.0,
        dd_soft_scale=0.45,
        description="Stronger MDD pack: DD circuit −20% + vol-target ×0.60",
    ),
    "dd18_vt70_pos75": RiskMddLever(
        lever_id="dd18_vt70_pos75",
        max_portfolio_dd=0.18,
        vol_target_scale=0.70,
        max_position_scale=0.75,
        dd_soft_scale=0.45,
        description="Tight pack: DD −18% + vol ×0.70 + max_position ×0.75",
    ),
    # --- Loop2: escape continuous-peak permanent-cash trap ---
    "dd25_vt70_yr": RiskMddLever(
        lever_id="dd25_vt70_yr",
        max_portfolio_dd=0.25,
        vol_target_scale=0.70,
        max_position_scale=1.0,
        dd_soft_scale=0.50,
        peak_mode="yearly",
        description="DD −25% + vol ×0.70 with yearly peak reset (no multi-year cash trap)",
    ),
    "dd25_vt70_soft": RiskMddLever(
        lever_id="dd25_vt70_soft",
        max_portfolio_dd=0.25,
        vol_target_scale=0.70,
        max_position_scale=1.0,
        dd_soft_scale=0.50,
        dd_breach_size_scale=0.30,
        peak_mode="continuous",
        description="DD −25% + vol ×0.70; breach → 30% size (soft recovery, continuous peak)",
    ),
    "vt60_only": RiskMddLever(
        lever_id="vt60_only",
        max_portfolio_dd=0.99,
        vol_target_scale=0.60,
        max_position_scale=1.0,
        dd_soft_scale=0.55,
        peak_mode="continuous",
        description="Vol-target ×0.60 only (no DD circuit)",
    ),
    "dd35_vt80_yr": RiskMddLever(
        lever_id="dd35_vt80_yr",
        max_portfolio_dd=0.35,
        vol_target_scale=0.80,
        max_position_scale=1.0,
        dd_soft_scale=0.55,
        peak_mode="yearly",
        description="Milder DD −35% + vol ×0.80, yearly peak (less aggressive)",
    ),
    # --- Loop F: soft size-scale on book DD (audit k100: hard skip DD kills edge) ---
    "dd25_soft35": RiskMddLever(
        lever_id="dd25_soft35",
        max_portfolio_dd=0.25,
        vol_target_scale=1.0,
        max_position_scale=1.0,
        dd_soft_scale=0.55,
        dd_breach_size_scale=0.35,
        peak_mode="continuous",
        description="Loop F: at book DD≤−25% new entries size×0.35 (no vol cut; continuous peak)",
    ),
    "dd25_soft35_yr": RiskMddLever(
        lever_id="dd25_soft35_yr",
        max_portfolio_dd=0.25,
        vol_target_scale=1.0,
        max_position_scale=1.0,
        dd_soft_scale=0.55,
        dd_breach_size_scale=0.35,
        peak_mode="yearly",
        description="Loop F: DD≤−25% size×0.35 with yearly peak reset",
    ),
    "dd30_soft40": RiskMddLever(
        lever_id="dd30_soft40",
        max_portfolio_dd=0.30,
        vol_target_scale=1.0,
        max_position_scale=1.0,
        dd_soft_scale=0.55,
        dd_breach_size_scale=0.40,
        peak_mode="continuous",
        description="Loop F mild: DD≤−30% size×0.40 continuous",
    ),
}


def loop_f_lever_ids() -> List[str]:
    """Loop F risk arms (soft DD size-scale; excludes pure baseline)."""
    return ["dd25_soft35", "dd25_soft35_yr", "dd30_soft40"]


def get_lever(lever_id: str) -> RiskMddLever:
    if lever_id not in LEVERS:
        raise KeyError(f"Unknown risk lever {lever_id!r}. Available: {list(LEVERS)}")
    return LEVERS[lever_id]


def list_levers() -> List[str]:
    return list(LEVERS.keys())


def apply_risk_mdd_lever(
    base_overrides: Optional[Dict[str, Any]],
    lever: RiskMddLever | str,
) -> Dict[str, Any]:
    """Merge lever onto strategy/backtest overrides (pure; no market data).

    - Scales ``volatility_target_pct`` by ``vol_target_scale`` when present.
    - Scales ``max_position_pct`` by ``max_position_scale`` when present.
    - Always sets ``max_portfolio_dd`` and ``dd_soft_scale`` from lever.
    - Optionally sets ``risk_off_scale`` when lever specifies it.
    """
    if isinstance(lever, str):
        lever = get_lever(lever)
    o = dict(base_overrides or {})
    if "volatility_target_pct" in o and lever.vol_target_scale != 1.0:
        o["volatility_target_pct"] = float(o["volatility_target_pct"]) * float(
            lever.vol_target_scale
        )
    if "max_position_pct" in o and lever.max_position_scale != 1.0:
        o["max_position_pct"] = float(o["max_position_pct"]) * float(
            lever.max_position_scale
        )
    o["max_portfolio_dd"] = float(lever.max_portfolio_dd)
    o["dd_soft_scale"] = float(lever.dd_soft_scale)
    # Always overwrite — never inherit stale soft-breach from contaminated base_overrides
    o["dd_breach_size_scale"] = (
        float(lever.dd_breach_size_scale)
        if lever.dd_breach_size_scale is not None
        else None
    )
    if lever.risk_off_scale is not None:
        o["risk_off_scale"] = float(lever.risk_off_scale)
    # peak_mode is consumed by mega study (not a BacktestConfig field)
    o["_peak_mode"] = str(lever.peak_mode or "continuous")
    return o


def resolve_peak_equity_seed(
    peak_mode: str,
    stored_peak: Optional[float],
) -> Optional[float]:
    """Seed for next OOS segment: yearly → None; continuous → prior HWM."""
    mode = str(peak_mode or "continuous").strip().lower()
    if mode == "yearly":
        return None
    if stored_peak is None:
        return None
    try:
        v = float(stored_peak)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def update_peak_equity_state(
    peak_mode: str,
    prior_peak: Optional[float],
    segment_hi: float,
    ending_capital: float,
    initial_capital: float,
) -> float:
    """Update stored multi-year peak after a segment (pure; no market data).

    - yearly: track segment HWM for logs only (next seed is always None)
    - continuous: ratchet max(prior, segment_hi, ending_capital)
    """
    mode = str(peak_mode or "continuous").strip().lower()
    seg = float(segment_hi)
    end = float(ending_capital)
    init = float(initial_capital)
    if mode == "yearly":
        return max(seg, end)
    if prior_peak is None:
        return max(seg, init, end)
    return max(float(prior_peak), seg, end)


def week_risk_ab_extra_bt() -> Dict[str, Dict[str, Any]]:
    """Return extra_bt dicts for control vs primary treatment arms."""
    base = get_lever("baseline")
    treat = get_lever(WEEK_PRIMARY_LEVER_ID)
    return {
        "baseline": apply_risk_mdd_lever({}, base),
        WEEK_PRIMARY_LEVER_ID: apply_risk_mdd_lever({}, treat),
    }


def alt_mdd_lever_ids() -> List[str]:
    """Ordered alt-loop MDD experiment lever ids (excludes pure baseline)."""
    return [
        "dd_circuit_25",
        ALT_PRIMARY_LEVER_ID,  # dd25_vt70
        "dd20_vt60",
        "dd18_vt70_pos75",
    ]


def alt_mdd_v2_lever_ids() -> List[str]:
    """Loop2 levers: yearly peak / soft breach / vol-only (escape cash trap)."""
    return [
        "dd25_vt70_yr",
        "dd25_vt70_soft",
        "vt60_only",
        "dd35_vt80_yr",
    ]


def alt_mdd_extra_bt_for_strategy(
    strategy_overrides: Optional[Dict[str, Any]],
    lever_id: str,
) -> Dict[str, Any]:
    """Apply registered lever on top of strategy.backtest_overrides()."""
    return apply_risk_mdd_lever(strategy_overrides, lever_id)


def is_control_like_name(name: str, control_strategy_id: str = "turbo_highvol_minalloc") -> bool:
    """True if ADVANCE id is the control book or a pure baseline arm."""
    n = str(name or "").strip()
    if not n:
        return False
    ctrl = str(control_strategy_id or "").strip()
    if n == ctrl:
        return True
    # mega config ids: turbo_highvol_minalloc__baseline
    if n.endswith("__baseline"):
        return True
    # scorecard prefixes: modern::turbo_highvol_minalloc
    if "::" in n:
        tail = n.split("::")[-1]
        if tail == ctrl or tail.endswith("__baseline"):
            return True
    if n.endswith(f"::{ctrl}"):
        return True
    return False


def decide_freeze_path(
    *,
    advance_names: List[str],
    control_strategy_id: str = "turbo_highvol_minalloc",
    winner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase D decision helper: ADVANCE list → freeze action (no file I/O).

    Rules:
      - 0 ADVANCE → keep_control
      - Explicit ``winner_id`` in ADVANCE and control-like → keep_control
      - Prefer first non-control-like ADVANCE (or non-control winner if in list)
      - keep_control only when every ADVANCE is control-like (or winner is control)

    ``write_shadow_candidate`` (not live registration): True only when action is
    register_shadow — orchestrator may dump a *report-side* candidate JSON; never
    overwrites ``paper_live/config/strategy_freeze.json``.
    """
    advances = [str(x) for x in (advance_names or []) if x]
    if not advances:
        return {
            "action": "keep_control",
            "strategy_id": control_strategy_id,
            "shadow_enabled": False,
            "shadow_strategy_id": None,
            "reason": "0 ADVANCE from promotion funnel; paper stays pure control",
            "write_shadow_candidate": False,
            # legacy alias — same meaning as write_shadow_candidate
            "register_new_freeze": False,
        }

    # Explicit human winner that is control-like → do not shadow-promote peers
    if (
        winner_id
        and winner_id in advances
        and is_control_like_name(winner_id, control_strategy_id)
    ):
        return {
            "action": "keep_control",
            "strategy_id": control_strategy_id,
            "shadow_enabled": False,
            "shadow_strategy_id": None,
            "reason": (
                f"Explicit winner_id {winner_id!r} is control-like; keep baseline freeze"
            ),
            "write_shadow_candidate": False,
            "register_new_freeze": False,
            "advance_names": advances,
        }

    non_control = [a for a in advances if not is_control_like_name(a, control_strategy_id)]
    if not non_control:
        return {
            "action": "keep_control",
            "strategy_id": control_strategy_id,
            "shadow_enabled": False,
            "shadow_strategy_id": None,
            "reason": (
                "All ADVANCE names are control-like baselines; keep pure control freeze"
            ),
            "write_shadow_candidate": False,
            "register_new_freeze": False,
            "advance_names": advances,
        }

    # Prefer non-control winner when provided; else first non-control ADVANCE
    if winner_id and winner_id in non_control:
        pick = winner_id
    else:
        # winner missing / not in list / was control-like → fallback first non-control
        pick = non_control[0]

    return {
        "action": "register_shadow",
        "strategy_id": control_strategy_id,
        "shadow_enabled": True,
        "shadow_strategy_id": pick,
        "reason": (
            "Non-control ADVANCE present; write report-side shadow candidate only "
            "(human copy after review; do not overwrite live paper freeze)"
        ),
        "write_shadow_candidate": True,
        "register_new_freeze": True,  # legacy alias of write_shadow_candidate
        "advance_names": advances,
        "notes": [
            "Research only. Shadow freeze path; human review before paper knobs change.",
            "Candidate lives under reports/.../phase_d_freeze/ only.",
            "Do not claim live/OPRA edge.",
        ],
    }
