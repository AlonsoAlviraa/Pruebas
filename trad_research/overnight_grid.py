"""Smart overnight research grid (risk/MDD first) — not random CPU burn.

Design principles:
  - Only STYLE-US base ``turbo_highvol_minalloc`` (train once per year).
  - Prefer **yearly** peak + hard circuit (Loop2 lesson: continuous+hard = cash trap).
  - Continuous peak only with **soft breach** (recovery path).
  - Dense lattice around HOLD region (dd≈0.30–0.40, vt≈0.70–0.90).
  - Sparse secondary axes (pos, soft_scale, breach, risk_off).
  - Exclude exact already-tested lever ids / fingerprints when provided.
  - Phase-2 overlays (wr_pack / crash) applied only to Phase-1 survivors (caller).

Research only. Not financial advice. No guaranteed alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from trad_research.risk_levers import RiskMddLever, apply_risk_mdd_lever


# Anchors already studied (include in grid once for baseline compare, flag as known)
KNOWN_DONE_LABELS: frozenset[str] = frozenset(
    {
        "baseline",
        "dd_circuit_25",
        "dd25_vt70",
        "dd20_vt60",
        "dd18_vt70_pos75",
        "dd25_vt70_yr",
        "dd25_vt70_soft",
        "vt60_only",
        "dd35_vt80_yr",
    }
)


@dataclass(frozen=True)
class OvernightCell:
    """One research arm (risk overlay on minalloc)."""

    label: str
    max_portfolio_dd: float
    vol_target_scale: float
    max_position_scale: float = 1.0
    dd_soft_scale: float = 0.55
    dd_breach_size_scale: Optional[float] = None
    risk_off_scale: Optional[float] = None
    peak_mode: str = "yearly"
    family: str = "risk_lattice"
    known: bool = False

    def fingerprint(self) -> str:
        b = (
            f"dd{self.max_portfolio_dd:.2f}"
            f"_vt{self.vol_target_scale:.2f}"
            f"_pos{self.max_position_scale:.2f}"
            f"_soft{self.dd_soft_scale:.2f}"
            f"_br{self.dd_breach_size_scale if self.dd_breach_size_scale is not None else 'hard'}"
            f"_ro{self.risk_off_scale if self.risk_off_scale is not None else 'na'}"
            f"_pk{self.peak_mode}"
        )
        return b

    def to_lever(self) -> RiskMddLever:
        return RiskMddLever(
            lever_id=self.label,
            max_portfolio_dd=float(self.max_portfolio_dd),
            vol_target_scale=float(self.vol_target_scale),
            max_position_scale=float(self.max_position_scale),
            dd_soft_scale=float(self.dd_soft_scale),
            dd_breach_size_scale=self.dd_breach_size_scale,
            risk_off_scale=self.risk_off_scale,
            peak_mode=str(self.peak_mode),
            description=f"overnight {self.family}: {self.fingerprint()}",
        )


def _label_from_fp(fp: str, family: str) -> str:
    # Short stable id for paths (Windows max path caution)
    return f"{family}_{fp}".replace(".", "p").replace("-", "m")[:90]


def build_phase1_risk_cells(
    *,
    mode: str = "full",
    exclude_fps: Optional[Set[str]] = None,
) -> List[OvernightCell]:
    """Phase-1 risk lattice. ``mode``: smoke | medium | full."""
    exclude_fps = set(exclude_fps or set())
    cells: List[OvernightCell] = []
    seen: Set[str] = set()

    def add(c: OvernightCell) -> None:
        fp = c.fingerprint()
        if fp in seen or fp in exclude_fps:
            return
        # Kill continuous + hard (permanent cash trap)
        if c.peak_mode == "continuous" and c.dd_breach_size_scale is None:
            if c.max_portfolio_dd < 0.9:
                return
        seen.add(fp)
        cells.append(c)

    # Always include pure baseline
    add(
        OvernightCell(
            label="baseline",
            max_portfolio_dd=0.99,
            vol_target_scale=1.0,
            peak_mode="yearly",
            family="anchor",
            known=True,
        )
    )
    # Anchor HOLD
    add(
        OvernightCell(
            label="dd35_vt80_yr",
            max_portfolio_dd=0.35,
            vol_target_scale=0.80,
            peak_mode="yearly",
            family="anchor",
            known=True,
        )
    )

    if mode == "smoke":
        for dd, vt in [(0.30, 0.75), (0.35, 0.80), (0.40, 0.85)]:
            add(
                OvernightCell(
                    label=_label_from_fp(
                        f"dd{dd:.2f}_vt{vt:.2f}_pos1.00_soft0.55_brhard_rona_pkyearly",
                        "yr",
                    ),
                    max_portfolio_dd=dd,
                    vol_target_scale=vt,
                    peak_mode="yearly",
                    family="risk_yr_hard",
                )
            )
        return cells

    # --- Family A: yearly hard lattice (primary MDD attack surface) ---
    if mode == "medium":
        dds = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45]
        vts = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
        poss = [0.80, 0.90, 1.0]
        softs = [0.50, 0.55]
    else:  # full overnight (~2k risk arms, dense near HOLD)
        dds = [
            0.20,
            0.22,
            0.25,
            0.28,
            0.30,
            0.32,
            0.34,
            0.35,
            0.36,
            0.38,
            0.40,
            0.42,
            0.45,
            0.48,
        ]
        vts = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        poss = [0.70, 0.80, 0.90, 1.0]
        softs = [0.45, 0.50, 0.55, 0.60]

    for dd, vt, pos, soft in product(dds, vts, poss, softs):
        # Dense near HOLD; mild subsample only on far tails (keep sense, not burn)
        if mode == "full":
            near = 0.28 <= dd <= 0.42 and 0.65 <= vt <= 0.95
            if not near:
                # keep 1/2 of far-tail cells (still structured, not random)
                if int(round(dd * 100 + vt * 100 + pos * 10 + soft * 100)) % 2 != 0:
                    continue
        fp = f"dd{dd:.2f}_vt{vt:.2f}_pos{pos:.2f}_soft{soft:.2f}_brhard_rona_pkyearly"
        add(
            OvernightCell(
                label=_label_from_fp(fp, "yr"),
                max_portfolio_dd=dd,
                vol_target_scale=vt,
                max_position_scale=pos,
                dd_soft_scale=soft,
                peak_mode="yearly",
                family="risk_yr_hard",
            )
        )

    # --- Family B: soft breach + yearly (recovery without permanent cash) ---
    if mode == "medium":
        b_dds = [0.28, 0.32, 0.35, 0.40]
        b_vts = [0.70, 0.80, 0.90]
        breaches = [0.20, 0.30, 0.45]
        b_poss = [0.85, 1.0]
    else:
        b_dds = [0.25, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45]
        b_vts = [0.65, 0.75, 0.80, 0.85, 0.95]
        breaches = [0.15, 0.25, 0.35, 0.50]
        b_poss = [0.80, 1.0]

    for dd, vt, br, pos in product(b_dds, b_vts, breaches, b_poss):
        fp = f"dd{dd:.2f}_vt{vt:.2f}_pos{pos:.2f}_soft0.55_br{br:.2f}_rona_pkyearly"
        add(
            OvernightCell(
                label=_label_from_fp(fp, "yrs"),
                max_portfolio_dd=dd,
                vol_target_scale=vt,
                max_position_scale=pos,
                dd_soft_scale=0.55,
                dd_breach_size_scale=br,
                peak_mode="yearly",
                family="risk_yr_soft",
            )
        )

    # --- Family C: continuous + soft only (no hard continuous) ---
    c_dds = [0.30, 0.35, 0.40] if mode == "medium" else [0.28, 0.32, 0.35, 0.40]
    c_vts = [0.75, 0.85] if mode == "medium" else [0.70, 0.80, 0.90]
    c_brs = [0.25, 0.40] if mode == "medium" else [0.20, 0.30, 0.45]
    for dd, vt, br in product(c_dds, c_vts, c_brs):
        fp = f"dd{dd:.2f}_vt{vt:.2f}_pos1.00_soft0.55_br{br:.2f}_rona_pkcontinuous"
        add(
            OvernightCell(
                label=_label_from_fp(fp, "cs"),
                max_portfolio_dd=dd,
                vol_target_scale=vt,
                dd_breach_size_scale=br,
                peak_mode="continuous",
                family="risk_cont_soft",
            )
        )

    # --- Family D: risk_off_scale around HOLD region ---
    ro_dds = [0.32, 0.35, 0.38]
    ro_vts = [0.75, 0.80, 0.85]
    ros = [0.40, 0.55, 0.70] if mode == "medium" else [0.35, 0.45, 0.55, 0.70, 0.85]
    for dd, vt, ro in product(ro_dds, ro_vts, ros):
        fp = f"dd{dd:.2f}_vt{vt:.2f}_pos1.00_soft0.55_brhard_ro{ro:.2f}_pkyearly"
        add(
            OvernightCell(
                label=_label_from_fp(fp, "ro"),
                max_portfolio_dd=dd,
                vol_target_scale=vt,
                risk_off_scale=ro,
                peak_mode="yearly",
                family="risk_off",
            )
        )

    # --- Family E: vol-only (no circuit) sparse ---
    for vt in ([0.50, 0.60, 0.70, 0.80] if mode != "full" else [0.45, 0.55, 0.60, 0.70, 0.80, 0.90]):
        fp = f"dd0.99_vt{vt:.2f}_pos1.00_soft0.55_brhard_rona_pkyearly"
        add(
            OvernightCell(
                label=_label_from_fp(fp, "vt"),
                max_portfolio_dd=0.99,
                vol_target_scale=vt,
                peak_mode="yearly",
                family="vol_only",
            )
        )

    return cells


def build_phase2_overlay_cells(
    survivors: Sequence[OvernightCell],
    *,
    max_survivors: int = 40,
) -> List[Tuple[OvernightCell, str]]:
    """Expand top Phase-1 cells with wr_pack / crash_rsi30 / both.

    Returns list of (base_cell, overlay_tag) where overlay_tag in
    ``wr``, ``crash``, ``crash_wr``.
    """
    top = list(survivors)[: max(1, int(max_survivors))]
    out: List[Tuple[OvernightCell, str]] = []
    for cell in top:
        if cell.label == "baseline" and cell.max_portfolio_dd >= 0.9:
            # still allow overlays on baseline
            pass
        for tag in ("wr", "crash", "crash_wr"):
            out.append((cell, tag))
    return out


def cells_to_mega_configs(
    cells: Sequence[OvernightCell],
    *,
    strategy_overrides: Optional[Dict[str, Any]] = None,
    base: str = "turbo_highvol_minalloc",
) -> List[Dict[str, Any]]:
    """Map overnight cells → mega-study config dicts (no crash/wr)."""
    from trad_research.crash_entry import CrashEntryConfig, WinRateFilterConfig  # noqa: F401

    configs: List[Dict[str, Any]] = []
    for c in cells:
        lever = c.to_lever()
        extra = apply_risk_mdd_lever(strategy_overrides, lever)
        pm = str(extra.pop("_peak_mode", None) or c.peak_mode)
        configs.append(
            {
                "id": f"{base}__{c.label}",
                "base": base,
                "label": c.label,
                "crash": None,
                "wr": None,
                "extra_bt": extra,
                "breadth": None,
                "regime_key": None,
                "peak_mode": pm,
                "family": c.family,
                "fingerprint": c.fingerprint(),
                "known": c.known,
            }
        )
    return configs


def overlay_to_mega_configs(
    pairs: Sequence[Tuple[OvernightCell, str]],
    *,
    strategy_overrides: Optional[Dict[str, Any]] = None,
    base: str = "turbo_highvol_minalloc",
) -> List[Dict[str, Any]]:
    from trad_research.crash_entry import CrashEntryConfig, WinRateFilterConfig

    wr_pack = WinRateFilterConfig(
        hard_stop_cooldown_days=10,
        max_atr_pct_tight=0.16,
        soft_trend_non_crash=True,
    )
    crash_rsi30 = CrashEntryConfig(
        enabled=True,
        mode="rsi",
        rsi_threshold=30.0,
        crash_min_confidence=0.22,
        relax_regime=True,
        crash_relax_trend=True,
    )
    configs: List[Dict[str, Any]] = []
    for cell, tag in pairs:
        lever = cell.to_lever()
        extra = apply_risk_mdd_lever(strategy_overrides, lever)
        pm = str(extra.pop("_peak_mode", None) or cell.peak_mode)
        wr = wr_pack if tag in ("wr", "crash_wr") else None
        crash = crash_rsi30 if tag in ("crash", "crash_wr") else None
        label = f"{cell.label}__{tag}"
        configs.append(
            {
                "id": f"{base}__{label}",
                "base": base,
                "label": label,
                "crash": crash,
                "wr": wr,
                "extra_bt": extra,
                "breadth": None,
                "regime_key": None,
                "peak_mode": pm,
                "family": f"overlay_{tag}",
                "fingerprint": f"{cell.fingerprint()}|{tag}",
                "known": False,
            }
        )
    return configs


def estimate_grid_sizes() -> Dict[str, int]:
    return {
        "smoke": len(build_phase1_risk_cells(mode="smoke")),
        "medium": len(build_phase1_risk_cells(mode="medium")),
        "full": len(build_phase1_risk_cells(mode="full")),
    }
