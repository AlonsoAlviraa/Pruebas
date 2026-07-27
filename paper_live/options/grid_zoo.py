"""Combinatorial options strategy grid (anti single-name leverage bias).

Generates thousands of paper options specs for portfolio / meta studies.
Bans: leverage>1, single-name lottery product ids, QQQ×2 / NVDA×2 styles.

Under proxy marks (``proxy_bs`` / ``vix_surface``), short-vol pure premium
kinds (IC/CCS/PCS/CSP) are filtered — real chain marks required for short-vol.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from paper_live.options.marks_policy import (
    BAN_RULE_NO_OPRA_FROM_PROXY,
    BAN_RULE_PROXY_SHORT_VOL,
    MARKS_PROXY_BS,
    PROXY_ZOO_BAN_KINDS,
    filter_specs_by_marks_mode,
    is_proxy_marks,
    normalize_marks_mode,
    short_vol_allowed,
)

# Diversified liquid underlyings (no lone-name leverage products)
INDEX_UNDS = ("SPY", "QQQ", "IWM")
NAME_UNDS = (
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "UNH",
    "V",
    "MA",
    "COST",
    "NVDA",  # allowed only as diversified name with budget cap, not levered
    "AMD",
    "TSLA",
)

# Short premium + defined-risk + small debit budgets
KINDS_CREDIT = (
    "put_credit_spread",
    "call_credit_spread",
    "iron_condor",
    "cash_secured_put",
    "covered_call",
)
KINDS_DEBIT = (
    "call_debit_spread",
    "put_debit_spread",
    "long_call",
    "long_put",
)

# Structural filter list (known negative mean under proxy_bs failure analysis)
STRUCTURAL_PROXY_NEGATIVE_KINDS = frozenset(PROXY_ZOO_BAN_KINDS)

DTE = (21, 30, 45, 60)
OTM = (0.03, 0.05, 0.08, 0.10)
WING = (0.10, 0.12, 0.15)
GATES: Sequence[Dict[str, Any]] = (
    {},
    {"require_uptrend": True},
    {"require_sma200": True},
    {"require_range_regime": True, "max_atr_pctile": 0.45},
    {"require_vrp_proxy_above": True, "min_vrp_proxy": 0.015},
    {"require_volume_dryup": True, "max_volume_ratio": 0.90},
    {"require_low_atr": True, "max_atr_pctile": 0.40},
)
BUDGETS = (0.05, 0.08, 0.10)


def _sid(*parts: Any) -> str:
    raw = "_".join(str(p) for p in parts)
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    # human prefix
    prefix = "_".join(str(p) for p in parts[:3])[:48]
    return f"G_{prefix}_{h}".replace(".", "p").replace(" ", "")


def _gate_name(g: Dict[str, Any]) -> str:
    if not g:
        return "nogate"
    for k in g:
        if k.startswith("require_"):
            return k.replace("require_", "")[:12]
    return "gate"


def is_banned_spec(spec: Dict[str, Any]) -> bool:
    """Hard bans: leverage products, naked lottery narrative."""
    sid = str(spec.get("id") or "").upper()
    notes = str(spec.get("notes") or "").upper()
    meta = spec.get("meta") or {}
    if meta.get("leverage") and float(meta.get("leverage") or 1) > 1.01:
        return True
    if "X2" in sid or "2X" in sid or "3X" in sid or "LEVER" in sid:
        return True
    if "NVDA2" in sid or "QQQ2" in sid or "TQQQ" in sid:
        return True
    if "LOTTERY" in notes:
        return True
    # single-name long_call with budget > 15% banned
    if spec.get("kind") == "long_call" and float(meta.get("max_premium_budget_frac") or 0) > 0.12:
        und = str(spec.get("underlying") or "")
        if und not in INDEX_UNDS and float(meta.get("max_premium_budget_frac") or 0) > 0.10:
            return True
    return False


def is_proxy_short_vol_banned(spec: Dict[str, Any], *, marks_mode: Optional[str] = None) -> bool:
    """True if kind is structural short-vol banned under proxy marks."""
    if short_vol_allowed(marks_mode or MARKS_PROXY_BS):
        return False
    kind = str(spec.get("kind") or "").lower()
    return kind in STRUCTURAL_PROXY_NEGATIVE_KINDS


def filter_zoo_for_marks(
    strategies: Sequence[Dict[str, Any]],
    marks_mode: Optional[str] = None,
    *,
    ban_kinds: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Post-filter zoo strategies for marks honesty (proxy → drop short-vol pure)."""
    mode = normalize_marks_mode(marks_mode or MARKS_PROXY_BS)
    extra = list(ban_kinds) if ban_kinds else None
    return filter_specs_by_marks_mode(strategies, mode, also_ban_kinds=extra, keep_cash=True)


def default_ban_rules(*, marks_mode: Optional[str] = None) -> List[str]:
    rules = [
        "no leverage>1",
        "no NVDA×2 / QQQ×2 product ids",
        "name debit budget <= 8%",
        BAN_RULE_NO_OPRA_FROM_PROXY,
    ]
    if is_proxy_marks(marks_mode or MARKS_PROXY_BS):
        rules.append(BAN_RULE_PROXY_SHORT_VOL)
        rules.append(
            "proxy_zoo_ban_kinds=" + ",".join(sorted(STRUCTURAL_PROXY_NEGATIVE_KINDS))
        )
    return rules


def _make_spec(
    kind: str,
    und: str,
    dte: int,
    otm: float,
    wing: float,
    gate: Dict[str, Any],
    *,
    bud: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Build one strategy dict or None if banned/invalid."""
    if kind in (
        "put_credit_spread",
        "call_credit_spread",
        "call_debit_spread",
        "put_debit_spread",
        "iron_condor",
    ):
        if wing <= otm:
            return None
    meta = dict(gate)
    if bud is not None:
        meta["max_premium_budget_frac"] = (
            min(bud, 0.08) if und not in INDEX_UNDS else bud
        )
        sid = _sid(kind, und, dte, otm, wing, _gate_name(gate), bud)
        label = f"{kind} {und} dte{dte} otm{otm} {_gate_name(gate)} b{bud}"
        max_dd = 0.25 if kind.endswith("spread") or kind == "iron_condor" else 0.35
        max_m = (
            0.35
            if "credit" in kind or kind in ("iron_condor", "cash_secured_put")
            else 0.20
        )
        day_drop = 0.12
    else:
        sid = _sid(kind, und, dte, otm, wing, _gate_name(gate))
        label = f"{kind} {und} dte{dte} otm{otm} {_gate_name(gate)}"
        max_dd = 0.22 if und in INDEX_UNDS else 0.30
        max_m = (
            0.40
            if kind in ("put_credit_spread", "call_credit_spread", "iron_condor")
            else 0.75
        )
        day_drop = 0.10
    spec = {
        "id": sid,
        "label": label,
        "kind": kind,
        "underlying": und,
        "dte_days": dte,
        "otm_pct": otm,
        "wing_otm_pct": wing,
        "contracts": 3 if und in INDEX_UNDS else 2,
        "max_portfolio_dd": max_dd,
        "max_margin_fraction": max_m,
        "max_single_day_drop": day_drop,
        "meta": meta,
        "notes": "grid_zoo anti-leverage; proxy_bs|vix_surface",
    }
    if is_banned_spec(spec):
        return None
    return spec


def iter_grid_specs(
    *,
    max_strategies: int = 3000,
    include_names: bool = True,
    seed_kinds: Optional[Sequence[str]] = None,
    marks_mode: Optional[str] = None,
    apply_proxy_short_vol_filter: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield strategy dicts for zoo JSON (round-robin by kind for diversity).

    When ``marks_mode`` is proxy and ``apply_proxy_short_vol_filter``, kinds in
    ``STRUCTURAL_PROXY_NEGATIVE_KINDS`` are not generated.
    """
    unds: List[str] = list(INDEX_UNDS)
    if include_names:
        unds.extend(NAME_UNDS)

    kinds = list(seed_kinds) if seed_kinds else list(KINDS_CREDIT) + list(KINDS_DEBIT)
    mode = normalize_marks_mode(marks_mode or MARKS_PROXY_BS)
    if apply_proxy_short_vol_filter and is_proxy_marks(mode):
        kinds = [k for k in kinds if k not in STRUCTURAL_PROXY_NEGATIVE_KINDS]
    n = 0

    # Always include cash control first
    yield {
        "id": "G_CASH_CTRL",
        "label": "Cash control",
        "kind": "cash",
        "underlying": "SPY",
        "hard_kill_enabled": False,
        "notes": "Portfolio residual benchmark",
        "meta": {},
    }
    n += 1

    # Pools keyed by (kind, und) so round-robin diversifies both axes
    pools: List[List[Dict[str, Any]]] = []
    for kind in kinds:
        for und in unds:
            pool: List[Dict[str, Any]] = []
            for dte, otm, wing, gate in itertools.product(DTE, OTM, WING, GATES):
                if kind in (
                    "long_call",
                    "long_put",
                    "call_debit_spread",
                    "put_debit_spread",
                ):
                    for bud in BUDGETS:
                        sp = _make_spec(kind, und, dte, otm, wing, gate, bud=bud)
                        if sp is not None:
                            pool.append(sp)
                else:
                    sp = _make_spec(kind, und, dte, otm, wing, gate)
                    if sp is not None:
                        pool.append(sp)
            if pool:
                pools.append(pool)

    # Round-robin across kind×underlying pools
    idxs = [0] * len(pools)
    while n < max_strategies:
        progressed = False
        for ki, pool in enumerate(pools):
            if n >= max_strategies:
                break
            i = idxs[ki]
            if i >= len(pool):
                continue
            yield pool[i]
            idxs[ki] = i + 1
            n += 1
            progressed = True
        if not progressed:
            break


def build_grid_zoo(
    *,
    max_strategies: int = 3000,
    include_names: bool = True,
    capital0: float = 100_000.0,
    marks_mode: Optional[str] = None,
    apply_proxy_short_vol_filter: bool = True,
) -> Dict[str, Any]:
    mode = normalize_marks_mode(marks_mode or MARKS_PROXY_BS)
    # Zoo data_label follows engine honesty: real only if short_vol_allowed
    data_label = "real_chain" if short_vol_allowed(mode) else "proxy_bs"
    specs = list(
        iter_grid_specs(
            max_strategies=max_strategies,
            include_names=include_names,
            marks_mode=mode,
            apply_proxy_short_vol_filter=apply_proxy_short_vol_filter,
        )
    )
    # Post-filter safety net
    if apply_proxy_short_vol_filter:
        specs = filter_zoo_for_marks(specs, mode)
    return {
        "version": "options-grid-zoo-v2",
        "capital0": capital0,
        "marks_mode": mode,
        "data_label": data_label,
        "notes": (
            "Combinatorial options grid for portfolio meta study. "
            "No single-name leverage. "
            f"marks_mode={mode}. "
            "Never claim OPRA edge from proxy_bs. VIRTUAL."
        ),
        "risk": {
            "max_portfolio_dd": 0.30,
            "max_single_day_drop": 0.12,
            "max_margin_fraction": 0.40,
            "hard_kill_enabled": True,
            "max_contracts": 12,
        },
        "ban_rules": default_ban_rules(marks_mode=mode),
        "proxy_zoo_ban_kinds": sorted(STRUCTURAL_PROXY_NEGATIVE_KINDS)
        if is_proxy_marks(mode)
        else [],
        "n_strategies": len(specs),
        "strategies": specs,
    }


def write_grid_zoo(path: Path, **kwargs: Any) -> Dict[str, Any]:
    zoo = build_grid_zoo(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zoo, indent=2), encoding="utf-8")
    return zoo
