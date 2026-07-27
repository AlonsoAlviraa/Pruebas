#!/usr/bin/env python
"""Read-only mega-audit of overnight multi-market results.

Computes honest counters the global score hides:
  - n_markets_beat_index (excess_total_vs_spy > 0)
  - n_markets_mdd_pass (MDD >= -0.50)
  - score decomposition (mdd_ok vs rest)
  - leave-one-year-out wealth (drop 2020) when year_results available
  - twin-config equity correlation if equities on disk

Usage::

    python scripts/audit_multimarket_results.py \\
        --root reports/redesign/overnight_multimarket_2026-07-23

Research only. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.multimarket import market_row_score  # noqa: E402


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def score_decomposition(row: Dict[str, Any]) -> Dict[str, float]:
    """Break market_row_score into components (mirrors multimarket.market_row_score)."""
    mdd = _f(row.get("max_drawdown"), -1.0)
    excess = _f(row.get("excess_total_vs_spy"), -9.0)
    sharpe = _f(row.get("sharpe"), -9.0)
    cagr = _f(row.get("cagr"), -9.0)
    n = _f(row.get("n_trades"), 0.0)
    if n < 20:
        return {
            "total": -100.0,
            "mdd_ok_bonus": 0.0,
            "mdd_term": 0.0,
            "excess_term": 0.0,
            "sharpe_term": 0.0,
            "cagr_term": 0.0,
        }
    mdd_ok = 1.0 if mdd >= -0.50 else (0.5 if mdd >= -0.60 else 0.0)
    return {
        "total": market_row_score(row),
        "mdd_ok_bonus": 50.0 * mdd_ok,
        "mdd_term": 10.0 * mdd,
        "excess_term": 2.0 * min(excess, 5.0),
        "sharpe_term": 5.0 * sharpe,
        "cagr_term": 3.0 * cagr,
    }


def load_market_rows(root: Path, market: str) -> List[Dict[str, Any]]:
    p = root / market / "all_rows.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def find_row(rows: List[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    for r in rows:
        if str(r.get("label") or "") == label:
            return r
    return None


def wealth_path(year_returns: List[float]) -> float:
    eq = 1.0
    for r in year_returns:
        eq *= 1.0 + float(r)
    return eq


def leave_one_year_out_cagr(
    year_results: List[Dict[str, Any]], drop_year: int
) -> Optional[float]:
    rets = []
    for y in year_results:
        if int(y.get("year") or 0) == drop_year:
            continue
        rets.append(float(y.get("year_return") or 0.0))
    if len(rets) < 2:
        return None
    w = wealth_path(rets)
    n = len(rets)
    if w <= 0:
        return None
    return float(w ** (1.0 / n) - 1.0)


def twin_equity_corr(root: Path, id_a: str, id_b: str) -> Optional[float]:
    try:
        import pandas as pd
    except ImportError:
        return None
    paths_a = list(root.rglob(f"**/configs/{id_a}/equity.csv"))
    paths_b = list(root.rglob(f"**/configs/{id_b}/equity.csv"))
    if not paths_a or not paths_b:
        return None
    a = pd.read_csv(paths_a[0], index_col=0).iloc[:, 0]
    b = pd.read_csv(paths_b[0], index_col=0).iloc[:, 0]
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 50:
        return None
    j.columns = ["a", "b"]
    return float(j["a"].pct_change().corr(j["b"].pct_change()))


def audit_winner(
    root: Path,
    *,
    winner_label: str,
    markets: Tuple[str, ...] = ("US", "ES", "DE", "FR", "UK"),
) -> Dict[str, Any]:
    per: Dict[str, Dict[str, Any]] = {}
    beat = 0
    mdd_pass = 0
    for mid in markets:
        rows = load_market_rows(root, mid)
        w = find_row(rows, winner_label)
        b = find_row(rows, "baseline")
        if w is None:
            per[mid] = {"missing": True}
            continue
        xs = _f(w.get("excess_total_vs_spy"))
        mdd = _f(w.get("max_drawdown"))
        if xs > 0:
            beat += 1
        if mdd >= -0.50:
            mdd_pass += 1
        per[mid] = {
            "winner": {
                "cagr": w.get("cagr"),
                "sharpe": w.get("sharpe"),
                "max_drawdown": w.get("max_drawdown"),
                "excess_total_vs_spy": w.get("excess_total_vs_spy"),
                "n_trades": w.get("n_trades"),
                "win_rate": w.get("win_rate"),
                "score": market_row_score(w),
                "score_decomp": score_decomposition(w),
            },
            "baseline": (
                {
                    "cagr": b.get("cagr"),
                    "max_drawdown": b.get("max_drawdown"),
                    "excess_total_vs_spy": b.get("excess_total_vs_spy"),
                    "score": market_row_score(b),
                }
                if b
                else None
            ),
            "year_results": w.get("year_results"),
        }

    us = per.get("US", {}).get("winner") or {}
    us_row = find_row(load_market_rows(root, "US"), winner_label) or {}
    years = us_row.get("year_results") or []
    loo_2020 = leave_one_year_out_cagr(years, 2020) if years else None

    # twin: vt 1.00 vs 0.90 same family
    twin_label = winner_label.replace("vt1p00", "vt0p90")
    twin_corr = None
    if twin_label != winner_label:
        id_a = f"turbo_highvol_minalloc__{winner_label}"
        id_b = f"turbo_highvol_minalloc__{twin_label}"
        twin_corr = twin_equity_corr(root, id_a, id_b)

    n_m = sum(1 for mid in markets if not per.get(mid, {}).get("missing"))
    return {
        "disclaimer": "Research only. Not financial advice. Audit of reported overnight results.",
        "root": str(root),
        "winner_label": winner_label,
        "n_markets_present": n_m,
        "n_markets_beat_index": beat,
        "n_markets_mdd_pass_50": mdd_pass,
        "claim_global_ok_is_misleading": beat < n_m,
        "us_cagr": us.get("cagr"),
        "us_mdd": us.get("max_drawdown"),
        "leave_one_year_out_cagr_drop_2020": loo_2020,
        "twin_vt_daily_return_corr": twin_corr,
        "per_market": per,
        "score_design_note": (
            "market_row_score awards +50 for MDD>=-50% and caps excess at +5; "
            "n_markets_ok uses score>-50, NOT excess>0."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Audit multi-market overnight results")
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT / "reports/redesign/overnight_multimarket_2026-07-23",
    )
    p.add_argument(
        "--winner",
        type=str,
        default=None,
        help="Label; default from summary.json global_winner",
    )
    args = p.parse_args(argv)

    root: Path = args.root
    winner = args.winner
    if not winner:
        sj = root / "summary.json"
        if sj.exists():
            winner = json.loads(sj.read_text(encoding="utf-8")).get("global_winner")
        if not winner:
            gr = root / "global_rank.json"
            if gr.exists():
                arr = json.loads(gr.read_text(encoding="utf-8"))
                if arr:
                    winner = arr[0].get("label")
    if not winner:
        print("ERROR: no winner label", file=sys.stderr)
        return 2

    report = audit_winner(root, winner_label=str(winner))
    out = root / "AUDIT_AUTO.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== Multi-market audit ===")
    print(f"winner: {winner}")
    print(
        f"beat_index: {report['n_markets_beat_index']}/{report['n_markets_present']}  "
        f"mdd_pass: {report['n_markets_mdd_pass_50']}/{report['n_markets_present']}"
    )
    print(f"claim_global_ok_misleading: {report['claim_global_ok_is_misleading']}")
    loo = report.get("leave_one_year_out_cagr_drop_2020")
    if loo is not None:
        print(f"leave-one-year-out CAGR (drop 2020): {100*loo:.1f}%")
    if report.get("twin_vt_daily_return_corr") is not None:
        print(f"twin vt corr: {report['twin_vt_daily_return_corr']:.6f}")
    for mid, block in (report.get("per_market") or {}).items():
        if block.get("missing"):
            print(f"  {mid}: missing")
            continue
        w = block["winner"]
        print(
            f"  {mid}: CAGR {100*_f(w.get('cagr')):.1f}% "
            f"MDD {100*_f(w.get('max_drawdown')):.1f}% "
            f"excess {100*_f(w.get('excess_total_vs_spy')):.1f}% "
            f"score { _f(w.get('score')):.1f} "
            f"(mdd_ok_bonus {_f((w.get('score_decomp') or {}).get('mdd_ok_bonus')):.0f})"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
