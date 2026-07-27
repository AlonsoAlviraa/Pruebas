"""Multi-stage promotion funnel — only best models ADVANCE (PROMO-01).

Stages (pre-registered in docs/design/2026-07-23_metrics_montecarlo_promotion.md):
  0 eligibility → 1 edge (Sortino/Sharpe/residual) → 2 Monte Carlo → 3 structural
  → 4 multi-test / top-K

Labels: KILL | HOLD | ADVANCE_STYLE | ADVANCE_ALPHA
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from trad_research.alpha_attribution import compare_to_benchmark
from trad_research.monte_carlo import (
    MCResult,
    mc_block_bootstrap_returns,
    mc_bootstrap_trades,
    mc_shuffle_trades,
    trade_pnls_from_equity_diff,
)
from trad_research.risk_metrics import extended_risk_from_equity
from trad_research.zoo import deflated_sharpe_note


@dataclass
class PromotionThresholds:
    """Frozen design thresholds."""

    min_trades: int = 50
    min_trades_smoke: int = 20
    pathology_cagr_abs: float = 1.0
    sortino_min: float = 0.50
    sharpe_min: float = 0.40
    residual_excess_min: float = 0.0
    residual_sharpe_min: float = 0.0
    mdd_min: float = -0.50
    profit_factor_min: float = 1.05
    expectancy_min: float = 0.0
    mc_n_sims: int = 2000
    mc_sortino_p5_min: float = 0.20
    mc_mdd_p5_max: float = -0.60  # worst allowed (more negative = worse); gate: mdd_p5 >= this
    mc_shuffle_sortino_p50_ratio: float = 0.50
    mc_min_trades_advance: int = 50
    max_advance: int = 3
    dsr_n_trials_force: int = 20


DEFAULT_THRESHOLDS = PromotionThresholds()


@dataclass
class CandidateInput:
    name: str
    equity: pd.Series
    style_equity: Optional[pd.Series] = None
    trade_pnls: Optional[np.ndarray] = None
    product: str = "STYLE-US"  # STYLE-US | ALPHA-PORTABLE
    n_trades: Optional[int] = None
    geo_p3_confirmed: Optional[bool] = None  # True = geo stress failed (P3)
    early_residual_ok: Optional[bool] = None
    smoke: bool = False


@dataclass
class StageResult:
    stage: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionCard:
    name: str
    product: str
    label: str
    stages: List[StageResult]
    metrics: Dict[str, Any]
    mc_bootstrap: Optional[Dict[str, Any]] = None
    mc_shuffle: Optional[Dict[str, Any]] = None
    residual: Optional[Dict[str, Any]] = None
    dsr: Optional[Dict[str, Any]] = None
    kill_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "product": self.product,
            "label": self.label,
            "stages": [{"stage": s.stage, "passed": s.passed, "details": s.details} for s in self.stages],
            "metrics": self.metrics,
            "mc_bootstrap": self.mc_bootstrap,
            "mc_shuffle": self.mc_shuffle,
            "residual": self.residual,
            "dsr": self.dsr,
            "kill_reasons": self.kill_reasons,
        }


def _eq_series(s: pd.Series) -> pd.Series:
    out = s.dropna().astype(float)
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, utc=True)
        except Exception:
            pass
    try:
        out.index = out.index.normalize()
    except Exception:
        pass
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def evaluate_candidate(
    cand: CandidateInput,
    *,
    thr: PromotionThresholds = DEFAULT_THRESHOLDS,
    n_sims: Optional[int] = None,
    seed: int = 42,
    n_trials_zoo: int = 1,
) -> PromotionCard:
    thr = thr or DEFAULT_THRESHOLDS
    n_sims = int(n_sims if n_sims is not None else thr.mc_n_sims)
    if cand.smoke:
        n_sims = min(n_sims, 200)
        min_tr = thr.min_trades_smoke
    else:
        min_tr = thr.min_trades

    eq = _eq_series(cand.equity)
    pnls = cand.trade_pnls
    if pnls is None or len(np.asarray(pnls).ravel()) < 2:
        pnls = trade_pnls_from_equity_diff(eq.to_numpy())
    else:
        pnls = np.asarray(pnls, dtype=float).ravel()
    n_trades = int(cand.n_trades if cand.n_trades is not None else pnls.size)

    risk = extended_risk_from_equity(eq.to_numpy(), trade_pnls=pnls)
    metrics = risk.to_dict()
    metrics["n_trades_used"] = n_trades

    stages: List[StageResult] = []
    kill_reasons: List[str] = []

    # ----- Stage 0 -----
    s0_ok = True
    s0: Dict[str, Any] = {}
    if eq.size < 5 or not np.isfinite(eq.iloc[-1]):
        s0_ok = False
        kill_reasons.append("invalid_equity")
    if abs(risk.cagr) > thr.pathology_cagr_abs:
        s0_ok = False
        kill_reasons.append("pathology_cagr")
        s0["pathology_cagr"] = risk.cagr
    if n_trades < min_tr and pnls.size < min_tr:
        # using daily rets as pseudo trades — allow stage0 if equity long enough
        s0["low_trade_count"] = n_trades
        if eq.size < min_tr:
            s0_ok = False
            kill_reasons.append("too_few_trades")
    s0["n_trades"] = n_trades
    stages.append(StageResult("stage0_eligibility", s0_ok, s0))
    if not s0_ok:
        return PromotionCard(
            name=cand.name,
            product=cand.product,
            label="KILL",
            stages=stages,
            metrics=metrics,
            kill_reasons=kill_reasons,
        )

    # ----- Stage 1 -----
    s1: Dict[str, Any] = {
        "sortino": risk.sortino,
        "sharpe": risk.sharpe,
        "mdd": risk.max_drawdown,
        "pf": risk.profit_factor,
        "expectancy": risk.expectancy,
    }
    s1_ok = True
    if risk.sortino < thr.sortino_min:
        s1_ok = False
        kill_reasons.append("sortino_below_min")
    if risk.sharpe < thr.sharpe_min:
        s1_ok = False
        kill_reasons.append("sharpe_below_min")
    if risk.max_drawdown < thr.mdd_min:
        s1_ok = False
        kill_reasons.append("mdd_too_deep")

    residual_dict = None
    if cand.style_equity is not None and len(cand.style_equity.dropna()) > 5:
        try:
            res = compare_to_benchmark(eq, _eq_series(cand.style_equity), label="vs_style")
            residual_dict = res.to_dict()
            s1["residual_excess_cagr"] = res.excess_cagr
            s1["residual_sharpe"] = res.residual_sharpe
            if res.excess_cagr <= thr.residual_excess_min:
                s1_ok = False
                kill_reasons.append("residual_excess_fail")
            if res.residual_sharpe <= thr.residual_sharpe_min:
                s1_ok = False
                kill_reasons.append("residual_sharpe_fail")
        except Exception as exc:  # noqa: BLE001
            s1["residual_error"] = str(exc)
            if cand.product == "ALPHA-PORTABLE":
                s1_ok = False
                kill_reasons.append("residual_error")
    else:
        s1["residual"] = "not_provided"
        if cand.product == "ALPHA-PORTABLE":
            s1_ok = False
            kill_reasons.append("residual_required_for_alpha")

    # PF/expectancy only if real trade-like pnls (not pure daily)
    if n_trades >= min_tr and risk.n_trades >= min_tr:
        if risk.profit_factor < thr.profit_factor_min and risk.profit_factor < 900:
            s1_ok = False
            kill_reasons.append("profit_factor_low")
        if risk.expectancy <= thr.expectancy_min:
            s1_ok = False
            kill_reasons.append("expectancy_non_positive")

    stages.append(StageResult("stage1_edge", s1_ok, s1))
    if not s1_ok:
        return PromotionCard(
            name=cand.name,
            product=cand.product,
            label="KILL",
            stages=stages,
            metrics=metrics,
            residual=residual_dict,
            kill_reasons=kill_reasons,
        )

    # ----- Stage 2 Monte Carlo -----
    boot = mc_bootstrap_trades(pnls, n_sims=n_sims, seed=seed, min_trades_full=thr.mc_min_trades_advance)
    shuf = mc_shuffle_trades(pnls, n_sims=n_sims, seed=seed, min_trades_full=thr.mc_min_trades_advance)
    # Also run block bootstrap on daily rets as secondary
    daily = trade_pnls_from_equity_diff(eq.to_numpy())
    _ = mc_block_bootstrap_returns(daily, n_sims=min(n_sims, 500), seed=seed)

    s2: Dict[str, Any] = {
        "bootstrap_sortino_p5": boot.sortino_p5,
        "bootstrap_mdd_p5": boot.mdd_p5,
        "shuffle_sortino_p50": shuf.sortino_p50,
        "hist_sortino": risk.sortino,
        "mc_diagnostic_only": boot.diagnostic_only or shuf.diagnostic_only,
    }
    s2_ok = True
    if boot.diagnostic_only or n_trades < thr.mc_min_trades_advance:
        s2_ok = False
        kill_reasons.append("mc_diagnostic_only_cannot_advance")
        s2["note"] = "too few trades for MC advance"
    else:
        if boot.sortino_p5 < thr.mc_sortino_p5_min:
            s2_ok = False
            kill_reasons.append("mc_sortino_p5_fail")
        # mdd_p5 is severe tail (most negative); require not worse than mc_mdd_p5_max
        if boot.mdd_p5 < thr.mc_mdd_p5_max:
            s2_ok = False
            kill_reasons.append("mc_mdd_tail_fail")
        if risk.sortino > 1e-9:
            ratio = shuf.sortino_p50 / risk.sortino
            s2["shuffle_sortino_p50_ratio"] = ratio
            if ratio < thr.mc_shuffle_sortino_p50_ratio:
                s2_ok = False
                kill_reasons.append("mc_shuffle_sortino_collapse")

    stages.append(StageResult("stage2_monte_carlo", s2_ok, s2))

    # ----- Stage 3 structural -----
    s3: Dict[str, Any] = {
        "geo_p3_confirmed": cand.geo_p3_confirmed,
        "early_residual_ok": cand.early_residual_ok,
    }
    s3_ok = True
    if cand.product == "ALPHA-PORTABLE":
        if cand.geo_p3_confirmed is True:
            s3_ok = False
            kill_reasons.append("geo_p3_blocks_alpha")
        if cand.early_residual_ok is False:
            # HOLD rather than hard kill if only early fails — mark hold later
            s3["early_warning"] = True
    stages.append(StageResult("stage3_structural", s3_ok, s3))

    # ----- Stage 4 DSR note -----
    dsr = deflated_sharpe_note(risk.sharpe, n_trials=max(n_trials_zoo, 1))
    s4_ok = True
    if n_trials_zoo >= thr.dsr_n_trials_force and float(dsr.get("deflated_sharpe_approx", 0)) <= 0:
        s4_ok = False
        kill_reasons.append("dsr_approx_non_positive")
    stages.append(StageResult("stage4_multitest", s4_ok, {"n_trials": n_trials_zoo}))

    # Label
    all_hard = s0_ok and s1_ok and s2_ok and s4_ok
    if not all_hard:
        # Stage1 passed (we returned early if not); failed MC/structural → HOLD if edge ok
        label = "HOLD" if s1_ok else "KILL"
        if not s2_ok and s1_ok:
            label = "HOLD"
        if not s0_ok or not s1_ok:
            label = "KILL"
        if not s4_ok and s1_ok and s2_ok:
            label = "HOLD"
    else:
        if not s3_ok and cand.product == "ALPHA-PORTABLE":
            label = "HOLD"
        elif cand.product == "ALPHA-PORTABLE":
            label = "ADVANCE_ALPHA"
        else:
            label = "ADVANCE_STYLE"

    return PromotionCard(
        name=cand.name,
        product=cand.product,
        label=label,
        stages=stages,
        metrics=metrics,
        mc_bootstrap=boot.to_dict(),
        mc_shuffle=shuf.to_dict(),
        residual=residual_dict,
        dsr=dsr,
        kill_reasons=kill_reasons,
    )


def apply_top_k(
    cards: Sequence[PromotionCard],
    *,
    k: int = 3,
) -> List[PromotionCard]:
    """Keep at most K ADVANCE_*; demote extras to HOLD."""
    advances = [c for c in cards if c.label.startswith("ADVANCE")]
    others = [c for c in cards if not c.label.startswith("ADVANCE")]
    # Rank advances by residual excess then sortino
    def score(c: PromotionCard) -> float:
        res = c.residual or {}
        return float(res.get("excess_cagr") or 0.0) + 0.01 * float(c.metrics.get("sortino") or 0.0)

    advances_sorted = sorted(advances, key=score, reverse=True)
    kept = advances_sorted[:k]
    demoted = advances_sorted[k:]
    for c in demoted:
        c.label = "HOLD"
        c.kill_reasons = list(c.kill_reasons) + ["top_k_demoted"]
    return kept + demoted + list(others)


def scorecard_table(cards: Sequence[PromotionCard]) -> str:
    lines = [
        "| Name | Product | Label | Sortino | Sharpe | MDD | Resid excess | MC Sortino p5 | Reasons |",
        "|------|---------|-------|---------|--------|-----|--------------|---------------|---------|",
    ]
    for c in cards:
        res_ex = (c.residual or {}).get("excess_cagr")
        res_s = f"{res_ex:.2%}" if isinstance(res_ex, (int, float)) else "n/a"
        mc_s = (c.mc_bootstrap or {}).get("sortino_p5")
        mc_ss = f"{mc_s:.2f}" if isinstance(mc_s, (int, float)) else "n/a"
        lines.append(
            f"| `{c.name}` | {c.product} | **{c.label}** | "
            f"{c.metrics.get('sortino', 0):.2f} | {c.metrics.get('sharpe', 0):.2f} | "
            f"{c.metrics.get('max_drawdown', 0):.2%} | {res_s} | {mc_ss} | "
            f"{','.join(c.kill_reasons[:4]) or '—'} |"
        )
    return "\n".join(lines)
