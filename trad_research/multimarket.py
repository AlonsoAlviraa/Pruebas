"""Multi-market registry + global scoring (anti-overfit transfer).

Protocol (research only)
------------------------
1. **Screen** knobs on **US only** (never fit thresholds on ES/DE/UK/FR).
2. **Transfer** frozen knobs to foreign markets (local train OK for risk
   overlays; no re-ranking of grid on geo).
3. **Global score** = mean rank across markets with **min-market penalty**
   (configs that only work in one market lose).

Not financial advice. No guaranteed alpha. Geo transfer may fail (prior FROZEN
ES/DE evidence); global winner may still be US-only paper if transfer fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MarketSpec:
    market_id: str
    data_root: Path
    ticker_file: Path
    preferred_index: tuple[str, ...]
    regime_key: str
    universe_limit: Optional[int]
    role: str  # screen | transfer


def default_markets(
    *,
    us_univ_limit: int = 40,
    geo_univ_limit: int = 40,
) -> List[MarketSpec]:
    """US screen + ES/DE/FR/UK transfer (local indices present in repo)."""
    return [
        MarketSpec(
            market_id="US",
            data_root=ROOT / "data",
            ticker_file=ROOT / "universe_highvol80.txt",
            preferred_index=("QQQ", "SPY"),
            regime_key="strict_dual_golden",
            universe_limit=us_univ_limit,
            role="screen",
        ),
        MarketSpec(
            market_id="ES",
            data_root=ROOT / "data_es",
            ticker_file=ROOT / "spain_wf_universe.txt",
            preferred_index=("IBEX",),
            regime_key="portable_not_deep_bear",
            universe_limit=geo_univ_limit,
            role="transfer",
        ),
        MarketSpec(
            market_id="DE",
            data_root=ROOT / "data_de",
            ticker_file=ROOT / "germany_wf_universe.txt",
            preferred_index=("DAX",),
            regime_key="portable_not_deep_bear",
            universe_limit=geo_univ_limit,
            role="transfer",
        ),
        MarketSpec(
            market_id="FR",
            data_root=ROOT / "data_fr",
            ticker_file=ROOT / "france_wf_universe.txt",
            preferred_index=("CAC",),
            regime_key="portable_not_deep_bear",
            universe_limit=geo_univ_limit,
            role="transfer",
        ),
        MarketSpec(
            market_id="UK",
            data_root=ROOT / "data_uk",
            ticker_file=ROOT / "uk_wf_universe.txt",
            preferred_index=("FTSE",),
            regime_key="portable_not_deep_bear",
            universe_limit=geo_univ_limit,
            role="transfer",
        ),
    ]


def available_markets(specs: Sequence[MarketSpec]) -> List[MarketSpec]:
    out: List[MarketSpec] = []
    for m in specs:
        if not m.data_root.is_dir() or not m.ticker_file.is_file():
            continue
        # need at least one preferred index history if listed
        if m.preferred_index:
            if not any(
                (m.data_root / f"{name}_history.csv").exists()
                for name in m.preferred_index
            ):
                continue
        out.append(m)
    return out


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError):
        return default


def market_row_score(row: Dict[str, Any]) -> float:
    """Higher is better. Single-market research score (not promotion)."""
    mdd = _safe_float(row.get("max_drawdown"), -1.0)
    excess = _safe_float(row.get("excess_total_vs_spy"), -9.0)
    sharpe = _safe_float(row.get("sharpe"), -9.0)
    cagr = _safe_float(row.get("cagr"), -9.0)
    n = _safe_float(row.get("n_trades"), 0.0)
    if n < 20:
        return -100.0
    mdd_ok = 1.0 if mdd >= -0.50 else (0.5 if mdd >= -0.60 else 0.0)
    # excess_total is total period excess vs local index (field name historical)
    return (
        50.0 * mdd_ok
        + 10.0 * mdd  # less negative better
        + 2.0 * min(excess, 5.0)  # cap extreme
        + 5.0 * sharpe
        + 3.0 * cagr
    )


def global_rank_table(
    per_market: Dict[str, List[Dict[str, Any]]],
    *,
    id_key: str = "label",
) -> List[Dict[str, Any]]:
    """Combine per-market rows into global ranking.

    ``per_market``: market_id -> list of metric rows (must share labels).
    Score = mean market score − 0.5 * (mean − min)  [penalize uneven transfer].
    """
    # label -> market -> row
    by: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mid, rows in per_market.items():
        for r in rows:
            if r.get("error"):
                continue
            lab = str(r.get(id_key) or r.get("id") or "")
            if not lab:
                continue
            by.setdefault(lab, {})[mid] = r

    markets = list(per_market.keys())
    out: List[Dict[str, Any]] = []
    for lab, mrows in by.items():
        scores = []
        detail: Dict[str, Any] = {}
        for mid in markets:
            r = mrows.get(mid)
            if r is None:
                scores.append(-100.0)
                detail[mid] = {"missing": True}
                continue
            sc = market_row_score(r)
            scores.append(sc)
            detail[mid] = {
                "score": sc,
                "cagr": r.get("cagr"),
                "sharpe": r.get("sharpe"),
                "max_drawdown": r.get("max_drawdown"),
                "excess_total_vs_spy": r.get("excess_total_vs_spy"),
                "n_trades": r.get("n_trades"),
            }
        mean_s = sum(scores) / max(len(scores), 1)
        min_s = min(scores)
        global_s = mean_s - 0.5 * (mean_s - min_s)
        n_ok = sum(1 for s in scores if s > -50)
        out.append(
            {
                "label": lab,
                "global_score": global_s,
                "mean_score": mean_s,
                "min_score": min_s,
                "n_markets_ok": n_ok,
                "n_markets": len(markets),
                "per_market": detail,
            }
        )
    out.sort(key=lambda x: float(x["global_score"]), reverse=True)
    return out
