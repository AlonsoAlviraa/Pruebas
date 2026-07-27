#!/usr/bin/env python3
"""Comparative study: baseline vol_target_hold vs expert variants.

Focus: QQQ/SPY vol targets 12/15/20% with costs + financing.
Expert adds: EWMA vol, VIX level/rank, SMA200 risk-off, circuit breaker, deadband.

VIRTUAL research. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.equity.cost_drag import CostDragConfig
from paper_live.equity.signal_backtest import run_equity_spec
from paper_live.equity.vol_target_expert import (
    expert_feature_gap_report,
    iter_study_specs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vt_improve")


def _bh(feed, ticker: str, start: date, end: date) -> Optional[float]:
    days = feed.session_days(start, end)
    if not days:
        return None
    b0 = feed.bar(ticker, days[0])
    b1 = feed.bar(ticker, days[-1])
    if b0 is None or b1 is None or float(b0.close) <= 0:
        return None
    return float(b1.close) / float(b0.close) - 1.0


def _sharpe_from_years(year_rets: Dict[str, float]) -> float:
    if not year_rets:
        return float("nan")
    xs = np.array(list(year_rets.values()), dtype=float)
    if len(xs) < 2:
        return float("nan")
    mu = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return mu / sd  # already annual-ish year returns


def _calmar(total_ret: float, max_dd: float, n_years: float) -> float:
    if n_years <= 0 or max_dd >= 0 or abs(max_dd) < 1e-12:
        return float("nan")
    # CAGR
    if total_ret <= -1:
        return float("nan")
    cagr = (1.0 + total_ret) ** (1.0 / n_years) - 1.0
    return cagr / abs(max_dd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Vol-target improve study")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--capital0", type=float, default=100_000.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "vol_target_improve" / "latest",
    )
    ap.add_argument("--synthetic", action="store_true", help="No network; synthetic prices")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "eodhd_cache")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    unds = ["SPY", "QQQ", "VIX"]

    if args.synthetic:
        from paper_live.datafeed.replay import DailyReplayFeed

        feed = DailyReplayFeed.from_synthetic(
            unds, n_days=2800, seed=42, start="2015-01-02"
        )
        data_label = "synthetic"
    else:
        from paper_live.data.eodhd_client import build_eodhd_feed

        feed, info = build_eodhd_feed(
            unds, start="2010-01-01", cache_dir=args.cache_dir, min_history=60
        )
        data_label = f"eodhd:{info}"
        logger.info("Feed built: %s", info)

    specs = iter_study_specs(
        underlyings=("QQQ", "SPY"),
        vol_targets=(0.12, 0.15, 0.20),
        max_leverages=(1.5, 2.0),
        presets=(
            "ewma_only",
            "ewma_trend",
            "ewma_vix",
            "full_expert",
            "full_expert_half_trend",
        ),
    )
    logger.info("Running %d specs %s → %s", len(specs), start, end)

    cost = CostDragConfig()
    rows: List[Dict[str, Any]] = []
    for i, sp in enumerate(specs):
        r = run_equity_spec(
            feed, sp, start=start, end=end, capital0=args.capital0, cost=cost
        )
        d = r.to_dict()
        d["preset"] = sp.get("preset")
        d["label"] = sp.get("label")
        d["vol_target"] = (sp.get("meta") or {}).get("vol_target")
        d["max_leverage"] = (sp.get("meta") or {}).get("max_leverage")
        n_years = max(1.0, r.n_days / 252.0)
        d["sharpe_year"] = _sharpe_from_years(r.year_returns)
        d["calmar"] = _calmar(r.total_return, r.max_dd, n_years)
        if r.total_return > -1 and n_years > 0:
            d["cagr"] = (1.0 + r.total_return) ** (1.0 / n_years) - 1.0
        else:
            d["cagr"] = float("nan")
        rows.append(d)
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "[%d/%d] %s ret=%.1f%% dd=%.1f%% L=%.2f wiped=%s",
                i + 1,
                len(specs),
                r.strategy_id,
                100 * r.total_return,
                100 * r.max_dd,
                r.mean_leverage,
                r.wiped,
            )

    spy_bh = _bh(feed, "SPY", start, end)
    qqq_bh = _bh(feed, "QQQ", start, end)

    # Rank by calmar then return among non-wiped; also by return
    alive = [x for x in rows if not x.get("wiped")]
    by_ret = sorted(alive, key=lambda x: float(x.get("total_return") or -9), reverse=True)
    by_calmar = sorted(
        [x for x in alive if math.isfinite(float(x.get("calmar") or float("nan")))],
        key=lambda x: float(x["calmar"]),
        reverse=True,
    )
    by_sharpe = sorted(
        [x for x in alive if math.isfinite(float(x.get("sharpe_year") or float("nan")))],
        key=lambda x: float(x["sharpe_year"]),
        reverse=True,
    )

    # Baseline vs best expert per (und, vt)
    comparisons: List[Dict[str, Any]] = []
    for und in ("QQQ", "SPY"):
        for vt in (0.12, 0.15, 0.20):
            base = [
                x
                for x in rows
                if x.get("kind") == "vol_target_hold"
                and x.get("underlying") == und
                and abs(float(x.get("vol_target") or 0) - vt) < 1e-9
                and not x.get("wiped")
            ]
            exp = [
                x
                for x in rows
                if x.get("kind") == "vol_target_expert"
                and x.get("underlying") == und
                and abs(float(x.get("vol_target") or 0) - vt) < 1e-9
                and not x.get("wiped")
            ]
            if not base:
                continue
            bbest = max(base, key=lambda x: float(x.get("total_return") or -9))
            if exp:
                ebest_ret = max(exp, key=lambda x: float(x.get("total_return") or -9))
                ebest_cal = max(
                    exp,
                    key=lambda x: float(x.get("calmar") or -9)
                    if math.isfinite(float(x.get("calmar") or float("nan")))
                    else -9,
                )
            else:
                ebest_ret = ebest_cal = None
            comparisons.append(
                {
                    "underlying": und,
                    "vol_target": vt,
                    "baseline_best_id": bbest["strategy_id"],
                    "baseline_ret": bbest["total_return"],
                    "baseline_dd": bbest["max_dd"],
                    "baseline_calmar": bbest.get("calmar"),
                    "baseline_mean_L": bbest.get("mean_leverage"),
                    "baseline_cost": bbest.get("cost_drag_total"),
                    "expert_best_ret_id": (ebest_ret or {}).get("strategy_id"),
                    "expert_best_ret": (ebest_ret or {}).get("total_return"),
                    "expert_best_ret_dd": (ebest_ret or {}).get("max_dd"),
                    "expert_best_ret_preset": (ebest_ret or {}).get("preset"),
                    "expert_best_calmar_id": (ebest_cal or {}).get("strategy_id"),
                    "expert_best_calmar": (ebest_cal or {}).get("calmar"),
                    "expert_best_calmar_preset": (ebest_cal or {}).get("preset"),
                    "delta_ret_pp": (
                        100.0
                        * (
                            float((ebest_ret or {}).get("total_return") or 0)
                            - float(bbest["total_return"])
                        )
                        if ebest_ret
                        else None
                    ),
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sleeve_results.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "period": {"start": str(start), "end": str(end)},
        "data_label": data_label,
        "capital_label": "VIRTUAL",
        "n_specs": len(specs),
        "spy_bh": spy_bh,
        "qqq_bh": qqq_bh,
        "feature_gaps": expert_feature_gap_report(),
        "comparisons": comparisons,
        "top10_return": [
            {
                "id": x["strategy_id"],
                "kind": x["kind"],
                "preset": x.get("preset"),
                "und": x["underlying"],
                "vt": x.get("vol_target"),
                "ret": x["total_return"],
                "dd": x["max_dd"],
                "calmar": x.get("calmar"),
                "sharpe_year": x.get("sharpe_year"),
                "mean_L": x.get("mean_leverage"),
                "cost": x.get("cost_drag_total"),
            }
            for x in by_ret[:10]
        ],
        "top10_calmar": [
            {
                "id": x["strategy_id"],
                "kind": x["kind"],
                "preset": x.get("preset"),
                "und": x["underlying"],
                "vt": x.get("vol_target"),
                "ret": x["total_return"],
                "dd": x["max_dd"],
                "calmar": x.get("calmar"),
                "mean_L": x.get("mean_leverage"),
            }
            for x in by_calmar[:10]
        ],
        "top10_sharpe_year": [
            {
                "id": x["strategy_id"],
                "preset": x.get("preset"),
                "und": x["underlying"],
                "vt": x.get("vol_target"),
                "sharpe_year": x.get("sharpe_year"),
                "ret": x["total_return"],
                "dd": x["max_dd"],
            }
            for x in by_sharpe[:10]
        ],
    }
    (out / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Markdown report
    lines = [
        "# Vol-target improve study",
        "",
        f"**Period:** {start} → {end}",
        f"**Data:** `{data_label}`",
        f"**Capital:** VIRTUAL",
        "",
        f"| Benchmark | Total return |",
        f"|-----------|-------------|",
        f"| SPY BH | {100*(spy_bh or 0):+.1f}% |",
        f"| QQQ BH | {100*(qqq_bh or 0):+.1f}% |",
        "",
        "## Feature gaps (baseline vs experts)",
        "",
    ]
    for k, v in expert_feature_gap_report().items():
        lines.append(f"- **{k}**: {v}")
    lines.extend(
        [
            "",
            "## Baseline vs best expert (by underlying × vol target)",
            "",
            "| Und | VT | Baseline ret | Baseline DD | Expert best ret | Expert DD | Preset | Δ ret (pp) |",
            "|-----|----|--------------|-------------|-----------------|-----------|--------|------------|",
        ]
    )
    for c in comparisons:
        br = 100 * float(c["baseline_ret"])
        bd = 100 * float(c["baseline_dd"])
        er = c.get("expert_best_ret")
        ed = c.get("expert_best_ret_dd")
        dpp = c.get("delta_ret_pp")
        lines.append(
            f"| {c['underlying']} | {c['vol_target']} | "
            f"{br:+.1f}% | {bd:.1f}% | "
            f"{(100*float(er)):+.1f}% | "
            f"{(100*float(ed) if ed is not None else float('nan')):.1f}% | "
            f"{c.get('expert_best_ret_preset')} | "
            f"{(f'{dpp:+.1f}' if dpp is not None else 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 by total return (non-wiped)",
            "",
            "| # | ID | Kind/preset | Und | VT | Ret | MaxDD | Calmar | Mean L | Cost |",
            "|---|----|-------------|-----|----|-----|-------|--------|--------|------|",
        ]
    )
    for i, x in enumerate(by_ret[:10], 1):
        lines.append(
            f"| {i} | `{x['strategy_id']}` | {x['kind']}/{x.get('preset')} | "
            f"{x['underlying']} | {x.get('vol_target')} | "
            f"{100*float(x['total_return']):+.1f}% | "
            f"{100*float(x['max_dd']):.1f}% | "
            f"{float(x.get('calmar') or float('nan')):.2f} | "
            f"{float(x.get('mean_leverage') or 0):.2f} | "
            f"{float(x.get('cost_drag_total') or 0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 by Calmar (risk-adjusted)",
            "",
            "| # | ID | Preset | Und | VT | Ret | MaxDD | Calmar |",
            "|---|----|--------|-----|----|-----|-------|--------|",
        ]
    )
    for i, x in enumerate(by_calmar[:10], 1):
        lines.append(
            f"| {i} | `{x['strategy_id']}` | {x.get('preset')} | "
            f"{x['underlying']} | {x.get('vol_target')} | "
            f"{100*float(x['total_return']):+.1f}% | "
            f"{100*float(x['max_dd']):.1f}% | "
            f"{float(x.get('calmar') or float('nan')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- **Moreira/Muir-style vol targeting** tends to improve *risk-adjusted* returns "
            "(Sharpe/Calmar) more reliably than raw total return vs buy-and-hold.",
            "- **EWMA** reacts faster than 20d equal-weight std → quicker de-risk in crises.",
            "- **VIX level/rank** are the main missing *financial* features in baseline; "
            "they gate leverage when implied fear is elevated (causal lag 1 day).",
            "- **SMA200 risk-off** cuts bull-market leverage capture (often lower total return "
            "on QQQ 2015–2025) but improves crisis DD — compare `full_expert` vs `ewma_only`.",
            "- **Deadband** reduces rebalance cost drag (baseline ~6% cumulative cost on top QQQ).",
            "- Hard DD wipe and financing (L−1) still apply. Not live trading advice.",
            "",
        ]
    )
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
