"""Option marks honesty policy (research-grade).

**Permanent norm:** real marks always for short-vol claims.
Never claim OPRA / exchange edge from ``proxy_bs`` or VIX-surface model marks.

- ``real_chain`` — exchange / marketplace / OPRA-like fills or mid quotes
- ``proxy_bs`` / ``vix_surface`` — model Black–Scholes on proxy IV (not fills)

When marks are proxy, short-premium structures must be **excluded** from
portfolio meta-study evaluation and research claims about short-vol edge.

**Fail closed:** until a real chain pricing path is wired into
``run_options_strategy``, study code must not claim ``real_chain`` marks or
allow short-vol evaluation merely because a CLI flag says so.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set

# Canonical modes
MARKS_REAL_CHAIN = "real_chain"
MARKS_PROXY_BS = "proxy_bs"
MARKS_VIX_SURFACE = "vix_surface"
MARKS_PROXY_COMBO = "proxy_bs|vix_surface"

# Set True only when run_options_strategy (or study path) actually prices
# from exchange/marketplace chain quotes — not model BS.
CHAIN_PRICING_ENGINE_AVAILABLE: bool = False

# Labels treated as real option chain marks (research-grade, not model BS)
REAL_MARKS_LABELS: FrozenSet[str] = frozenset(
    {
        "real_chain",
        "yahoo_chain",
        "eodhd_options_eod",
        "opra",
        "unicornbay",
        "exchange_chain",
    }
)

PROXY_MARKS_LABELS: FrozenSet[str] = frozenset(
    {
        MARKS_PROXY_BS,
        MARKS_VIX_SURFACE,
        MARKS_PROXY_COMBO,
        "proxy",
        "proxy_bs_stress",
        "model_bs",
    }
)

# Short-premium family taxonomy for portfolio CAPS only — NOT the proxy meta ban set.
# For evaluation gates under proxy marks, use PROXY_META_EXCLUDE_KINDS.
SHORT_PREMIUM_KINDS: FrozenSet[str] = frozenset(
    {
        "put_credit_spread",
        "call_credit_spread",
        "iron_condor",
        "cash_secured_put",
        "covered_call",
    }
)

# Hard-exclude from meta study evaluation under proxy marks.
# covered_call kept out of this set: equity beta + mild positive mean on proxy;
# still short-vol for family caps, but not structural-negative like pure short premium.
PROXY_META_EXCLUDE_KINDS: FrozenSet[str] = frozenset(
    {
        "put_credit_spread",
        "call_credit_spread",
        "iron_condor",
        "cash_secured_put",
    }
)

# Default grid zoo ban under proxy (structural negative mean on proxy_bs)
PROXY_ZOO_BAN_KINDS: FrozenSet[str] = frozenset(
    {
        "iron_condor",
        "call_credit_spread",
        "put_credit_spread",
        "cash_secured_put",
    }
)

BAN_RULE_PROXY_SHORT_VOL = "proxy_marks: exclude short-vol pure premium (IC/CCS/PCS/CSP)"
BAN_RULE_NO_OPRA_FROM_PROXY = "never claim OPRA edge from proxy_bs|vix_surface"
BAN_RULE_NORM_VIOLATION = (
    "NORM VIOLATION: short-vol pure kinds included under proxy pricing (debug only)"
)


def _token_parts(marks_mode: str) -> Set[str]:
    return {p.strip() for p in marks_mode.replace("|", ",").split(",") if p.strip()}


def normalize_marks_mode(marks_mode: Optional[str]) -> str:
    """Collapse free-form data_label into real_chain vs proxy family.

    Rules (fail closed)
    -------------------
    - Only exact known tokens count as real (no substring ``\"real\" in m``).
    - If both real and proxy tokens appear → **proxy** (never open short-vol gate).
    - Unknown strings stay opaque (not promoted to real_chain).
    """
    m = (marks_mode or MARKS_PROXY_BS).strip().lower()
    if not m:
        return MARKS_PROXY_BS

    parts = _token_parts(m)
    has_real = bool(parts & REAL_MARKS_LABELS) or m in REAL_MARKS_LABELS
    has_proxy = bool(parts & PROXY_MARKS_LABELS) or any(
        p.startswith("proxy") for p in parts
    )
    if "vix_surface" in parts or m == MARKS_VIX_SURFACE:
        has_proxy = True
    if "proxy_bs" in m or m.startswith("proxy"):
        has_proxy = True

    # Mixed real+proxy → proxy (fail closed)
    if has_real and has_proxy:
        if "vix_surface" in m or MARKS_VIX_SURFACE in parts:
            return MARKS_PROXY_COMBO
        return MARKS_PROXY_BS

    if has_real:
        return MARKS_REAL_CHAIN

    if has_proxy:
        if "vix_surface" in m or MARKS_VIX_SURFACE in parts:
            return MARKS_PROXY_COMBO
        return MARKS_PROXY_BS

    # Unknown: do not promote to real
    return m


def is_real_marks(marks_mode: Optional[str]) -> bool:
    return normalize_marks_mode(marks_mode) == MARKS_REAL_CHAIN


def is_proxy_marks(marks_mode: Optional[str]) -> bool:
    mode = normalize_marks_mode(marks_mode)
    if mode == MARKS_REAL_CHAIN:
        return False
    # treat unknown as proxy-family for honesty (fail closed on short-vol claims)
    return True


def short_vol_allowed(
    marks_mode: Optional[str],
    *,
    chain_engine_available: Optional[bool] = None,
    pricing_backend: Optional[str] = None,
) -> bool:
    """Short-premium research evaluation allowed only with **actual** real chain marks.

    Requires:
    1. Requested / effective mode is real_chain family
    2. Chain pricing engine is available (wired into replay)
    3. Pricing backend is not proxy BS (if provided)
    """
    engine_ok = (
        CHAIN_PRICING_ENGINE_AVAILABLE
        if chain_engine_available is None
        else bool(chain_engine_available)
    )
    if not engine_ok:
        return False
    if pricing_backend is not None and not is_real_marks(pricing_backend):
        return False
    return is_real_marks(marks_mode)


def resolve_study_marks_context(
    requested_mode: Optional[str],
    *,
    chain_engine_available: Optional[bool] = None,
    pricing_backend: str = MARKS_PROXY_BS,
) -> Dict[str, Any]:
    """Resolve study honesty fields; never claim real marks without a chain engine.

    Returns
    -------
    dict with keys:
      requested_mode, effective_mode, option_marks_label, pricing_backend,
      short_vol_allowed, forced_proxy, forced_proxy_reason
    """
    engine_ok = (
        CHAIN_PRICING_ENGINE_AVAILABLE
        if chain_engine_available is None
        else bool(chain_engine_available)
    )
    requested = normalize_marks_mode(requested_mode)
    backend = normalize_marks_mode(pricing_backend)

    can_claim_real = (
        engine_ok and is_real_marks(requested) and is_real_marks(backend)
    )
    if can_claim_real:
        return {
            "requested_mode": requested,
            "effective_mode": MARKS_REAL_CHAIN,
            "option_marks_label": MARKS_REAL_CHAIN,
            "pricing_backend": backend,
            "short_vol_allowed": True,
            "forced_proxy": False,
            "forced_proxy_reason": None,
        }

    # Fail closed → proxy reporting
    if is_proxy_marks(backend):
        option_label = (
            MARKS_PROXY_COMBO
            if "vix" in backend or backend == MARKS_PROXY_COMBO
            else MARKS_PROXY_COMBO  # study default honesty label for model marks
        )
        # Prefer standard study label
        option_label = "proxy_bs|vix_surface"
        eff = backend if is_proxy_marks(backend) else MARKS_PROXY_BS
        if eff == MARKS_REAL_CHAIN:
            eff = MARKS_PROXY_BS
    else:
        option_label = "proxy_bs|vix_surface"
        eff = MARKS_PROXY_BS

    reason = None
    forced_proxy = False
    if is_real_marks(requested) and not engine_ok:
        forced_proxy = True
        reason = (
            "marks_mode=real_chain requested but chain pricing engine is not wired "
            "into run_options_strategy; forced proxy_bs|vix_surface (fail closed)"
        )
    elif is_real_marks(requested) and not is_real_marks(backend):
        forced_proxy = True
        reason = (
            "marks_mode=real_chain requested but pricing_backend is proxy; "
            "forced proxy honesty labels (fail closed)"
        )

    # effective_mode for filters: always proxy when we cannot claim real
    if not is_proxy_marks(eff):
        eff = MARKS_PROXY_BS

    return {
        "requested_mode": requested,
        "effective_mode": eff if is_proxy_marks(eff) else MARKS_PROXY_BS,
        "option_marks_label": option_label,
        "pricing_backend": MARKS_PROXY_BS if not is_real_marks(backend) else backend,
        "short_vol_allowed": False,
        "forced_proxy": forced_proxy,
        "forced_proxy_reason": reason,
    }


def is_short_premium_kind(kind: Optional[str]) -> bool:
    """True if kind is in short-premium family (caps taxonomy, not proxy ban)."""
    k = (kind or "").strip().lower()
    return k in SHORT_PREMIUM_KINDS


def is_proxy_meta_excluded_kind(kind: Optional[str]) -> bool:
    k = (kind or "").strip().lower()
    return k in PROXY_META_EXCLUDE_KINDS


def allow_kind_for_marks(
    kind: Optional[str],
    marks_mode: Optional[str],
    *,
    chain_engine_available: Optional[bool] = None,
    pricing_backend: Optional[str] = None,
) -> bool:
    """False if kind must not enter portfolio meta evaluation under this marks mode."""
    if short_vol_allowed(
        marks_mode,
        chain_engine_available=chain_engine_available,
        pricing_backend=pricing_backend,
    ):
        return True
    return not is_proxy_meta_excluded_kind(kind)


def filter_specs_by_marks_mode(
    specs: Sequence[Any],
    marks_mode: Optional[str],
    *,
    also_ban_kinds: Optional[Iterable[str]] = None,
    keep_cash: bool = True,
    chain_engine_available: Optional[bool] = None,
    pricing_backend: Optional[str] = None,
    apply_filter: bool = True,
) -> List[Any]:
    """Drop proxy-banned short-vol (and optional extra kinds) from a zoo/spec list."""
    extra = {str(k).lower() for k in (also_ban_kinds or [])}
    ban = set(PROXY_ZOO_BAN_KINDS) | extra
    if not apply_filter:
        ban = set(extra)
    elif short_vol_allowed(
        marks_mode,
        chain_engine_available=chain_engine_available,
        pricing_backend=pricing_backend,
    ):
        ban = set(extra)  # only explicit extras when real marks + engine

    out: List[Any] = []
    for s in specs:
        kind = str(
            getattr(s, "kind", None)
            or (s.get("kind") if isinstance(s, Mapping) else "")
            or ""
        ).lower()
        sid = str(
            getattr(s, "id", None)
            or (s.get("id") if isinstance(s, Mapping) else "")
            or ""
        )
        if keep_cash and (kind == "cash" or sid == "G_CASH_CTRL"):
            out.append(s)
            continue
        if kind in ban:
            continue
        if apply_filter and not allow_kind_for_marks(
            kind,
            marks_mode,
            chain_engine_available=chain_engine_available,
            pricing_backend=pricing_backend,
        ):
            continue
        out.append(s)
    return out


def kind_from_sleeve_ymap(
    ymap: Mapping[str, Any],
    sid: str = "",
) -> str:
    """Extract strategy kind from a sleeve_year_returns cell map."""
    if sid == "G_CASH_CTRL":
        return "cash"
    for cell in ymap.values():
        if isinstance(cell, dict) and cell.get("kind"):
            return str(cell["kind"])
    return ""


def filter_sleeve_years_for_marks(
    sleeve_years: Mapping[str, Mapping[str, Any]],
    marks_mode: Optional[str],
    *,
    apply_filter: bool = True,
    restrict_to_ids: Optional[Set[str]] = None,
    chain_engine_available: Optional[bool] = None,
    pricing_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Filter sleeve cache for meta evaluation.

    Always re-checks **kind** (never trust ID membership alone).
    When ``restrict_to_ids`` is None, keeps **all** cache IDs that pass the
    kind filter (full rescore universe).
    """
    out: Dict[str, Any] = {}
    for sid, ymap in sleeve_years.items():
        kind = kind_from_sleeve_ymap(ymap, sid)
        if apply_filter and not allow_kind_for_marks(
            kind,
            marks_mode,
            chain_engine_available=chain_engine_available,
            pricing_backend=pricing_backend,
        ):
            continue
        if restrict_to_ids is not None and sid not in restrict_to_ids and sid != "G_CASH_CTRL":
            continue
        out[str(sid)] = ymap
    return out


def honesty_disclaimer(
    marks_mode: Optional[str],
    *,
    option_marks_label: Optional[str] = None,
    proxy_filter_applied: bool = True,
    short_vol_evaluated: bool = False,
    forced_proxy_reason: Optional[str] = None,
) -> str:
    label = option_marks_label or (
        "real_chain" if is_real_marks(marks_mode) and CHAIN_PRICING_ENGINE_AVAILABLE
        else "proxy_bs|vix_surface"
    )
    if label == "real_chain" or (
        is_real_marks(marks_mode) and CHAIN_PRICING_ENGINE_AVAILABLE
    ):
        return (
            "Option marks labeled real_chain (exchange/marketplace). "
            "Still VIRTUAL capital; not financial advice."
        )
    parts = [
        "VIRTUAL. Option marks are model BS on VIX surface (not OPRA).",
        "Never claim OPRA edge from proxy_bs.",
    ]
    if forced_proxy_reason:
        parts.append(f"Forced proxy: {forced_proxy_reason}")
    if short_vol_evaluated and not proxy_filter_applied:
        parts.append(
            "NORM VIOLATION: short-vol pure kinds were included under proxy pricing (debug)."
        )
    elif proxy_filter_applied:
        parts.append(
            "Short-premium pure kinds excluded from meta evaluation under proxy marks."
        )
    parts.append("Not financial advice.")
    return " ".join(parts)
