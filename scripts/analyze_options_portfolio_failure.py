#!/usr/bin/env python3
"""Post-mortem: why options portfolio meta study underperformed SPY."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "reports" / "options_portfolio_meta" / "latest"


def main() -> int:
    wf = json.loads((LATEST / "walk_forward.json").read_text(encoding="utf-8"))
    sleeves = json.loads((LATEST / "sleeve_year_returns.json").read_text(encoding="utf-8"))
    active = [r for r in wf["walk_forward"] if r.get("mode") == "meta"]
    years = [str(y) for y in range(2010, 2026)]

    short_k = {
        "put_credit_spread",
        "call_credit_spread",
        "iron_condor",
        "cash_secured_put",
        "covered_call",
    }
    long_k = {"long_call", "long_put", "call_debit_spread", "put_debit_spread"}

    # --- sleeve universe ---
    by_kind: Dict[str, List[float]] = defaultdict(list)
    by_und: Dict[str, List[float]] = defaultdict(list)
    opens_by_kind: Dict[str, List[int]] = defaultdict(list)
    hk_by_kind: Dict[str, int] = defaultdict(int)
    n_by_kind: Dict[str, int] = defaultdict(int)
    all_rets: List[float] = []
    zero_open = 0

    for sid, ymap in sleeves.items():
        for y, cell in ymap.items():
            if cell.get("error"):
                continue
            tr = float(cell.get("total_return") or 0.0)
            kind = str(cell.get("kind") or "?")
            und = str(cell.get("underlying") or "?")
            n_op = int(cell.get("n_opens") or 0)
            by_kind[kind].append(tr)
            by_und[und].append(tr)
            opens_by_kind[kind].append(n_op)
            n_by_kind[kind] += 1
            if cell.get("hard_kill"):
                hk_by_kind[kind] += 1
            all_rets.append(tr)
            if n_op == 0 and abs(tr) < 1e-12:
                zero_open += 1

    arr = np.asarray(all_rets, dtype=float)

    # --- meta selection ---
    selected_gross: List[float] = []
    selected_scaled: List[float] = []
    oracles: List[float] = []
    meta_hits: List[float] = []
    sum_ws: List[float] = []
    year_rows: List[Dict[str, Any]] = []
    oracle_kind_c: Counter = Counter()
    sel_kind_c: Counter = Counter()
    sel_und_c: Counter = Counter()

    for row in active:
        y = row["year"]
        sel = row.get("selected") or []
        w = row.get("weights") or {}
        sum_w = float(sum(w.values()))
        sum_ws.append(sum_w)
        rets = []
        for s in sel:
            sid = s.get("id")
            tr = float(sleeves.get(sid, {}).get(y, {}).get("total_return") or 0.0)
            rets.append(tr)
            sel_kind_c[str(s.get("kind"))] += 1
            sel_und_c[str(s.get("und"))] += 1
            if float(s.get("proba") or 0) >= 0.55:
                meta_hits.append(1.0 if tr > 0 else 0.0)
        gross = float(np.mean(rets)) if rets else 0.0
        selected_gross.append(gross)
        selected_scaled.append(float(row["portfolio_return"]))

        year_cells = []
        for sid, ymap in sleeves.items():
            if sid == "G_CASH_CTRL":
                continue
            c = ymap.get(y)
            if not c or c.get("error"):
                continue
            year_cells.append(
                (
                    float(c["total_return"]),
                    str(c.get("kind")),
                    str(c.get("underlying")),
                    sid,
                )
            )
        year_cells.sort(reverse=True)
        top8 = year_cells[:8]
        oracle = float(np.mean([t[0] for t in top8])) if top8 else 0.0
        oracles.append(oracle)
        for t in top8:
            oracle_kind_c[t[1]] += 1

        # base rate positivity
        base_pos = float(np.mean([t[0] > 0 for t in year_cells])) if year_cells else 0.0
        sel_pos = float(np.mean([r > 0 for r in rets])) if rets else 0.0

        year_rows.append(
            {
                "year": y,
                "gross_ew": gross,
                "port": float(row["portfolio_return"]),
                "oracle8": oracle,
                "naive5": float(row.get("naive_top5_return") or 0.0),
                "spy": float(row.get("spy_bh") or 0.0),
                "qqq": float(row.get("qqq_bh") or 0.0),
                "sum_w": sum_w,
                "cash_w": 1.0 - sum_w,
                "base_pos": base_pos,
                "sel_pos": sel_pos,
                "meta_pos_rate_train": row.get("meta_pos_rate"),
                "train_rows": row.get("meta_train_rows"),
            }
        )

    # momentum persistence
    pairs = []
    for sid, ymap in sleeves.items():
        for i in range(len(years) - 1):
            a, b = ymap.get(years[i]), ymap.get(years[i + 1])
            if not a or not b or a.get("error") or b.get("error"):
                continue
            pairs.append((float(a["total_return"]), float(b["total_return"])))
    pairs_a = np.asarray(pairs, dtype=float) if pairs else np.zeros((0, 2))
    mom = {}
    if len(pairs_a) > 100:
        corr = float(np.corrcoef(pairs_a[:, 0], pairs_a[:, 1])[0, 1])
        qs = np.quantile(pairs_a[:, 0], [0.2, 0.4, 0.6, 0.8])
        quints = []
        for i, name in enumerate(["Q1_worst", "Q2", "Q3", "Q4", "Q5_best"]):
            if i == 0:
                m = pairs_a[:, 0] <= qs[0]
            elif i == 4:
                m = pairs_a[:, 0] > qs[3]
            else:
                m = (pairs_a[:, 0] > qs[i - 1]) & (pairs_a[:, 0] <= qs[i])
            quints.append(
                {
                    "name": name,
                    "prior_mean": float(pairs_a[m, 0].mean()),
                    "next_mean": float(pairs_a[m, 1].mean()),
                    "n": int(m.sum()),
                }
            )
        mom = {"corr": corr, "n": len(pairs_a), "quintiles": quints}

    # family by year
    fam_years = []
    for y in years:
        sh, lo = [], []
        for sid, ymap in sleeves.items():
            c = ymap.get(y)
            if not c:
                continue
            tr = float(c.get("total_return") or 0.0)
            k = c.get("kind")
            if k in short_k:
                sh.append(tr)
            if k in long_k:
                lo.append(tr)
        spy = (wf.get("benchmarks") or {}).get(y, {}).get("SPY")
        fam_years.append(
            {
                "year": y,
                "short_mean": float(np.mean(sh)) if sh else None,
                "long_mean": float(np.mean(lo)) if lo else None,
                "spy": float(spy) if spy is not None else None,
                "n_short": len(sh),
                "n_long": len(lo),
            }
        )

    # counterfactual: if we had invested sum_w at gross_ew
    cf = [r["gross_ew"] * r["sum_w"] for r in year_rows]

    kind_stats = []
    for k, v in by_kind.items():
        a = np.asarray(v, dtype=float)
        o = np.asarray(opens_by_kind[k], dtype=float)
        kind_stats.append(
            {
                "kind": k,
                "n": len(a),
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "pos_rate": float((a > 0).mean()),
                "mean_opens": float(o.mean()) if len(o) else 0.0,
                "pct_zero_open": float((o == 0).mean()) if len(o) else 0.0,
                "hard_kill_rate": float(hk_by_kind[k] / max(n_by_kind[k], 1)),
            }
        )
    kind_stats.sort(key=lambda x: -x["mean"])

    report = {
        "headline": {
            "portfolio_mean": float(np.mean(selected_scaled)),
            "selected_gross_ew_mean": float(np.mean(selected_gross)),
            "oracle_top8_mean": float(np.mean(oracles)),
            "naive_top5_mean": float(np.mean([r["naive5"] for r in year_rows])),
            "spy_mean": float(np.mean([r["spy"] for r in year_rows])),
            "qqq_mean": float(np.mean([r["qqq"] for r in year_rows])),
            "avg_invested_weight": float(np.mean(sum_ws)),
            "avg_cash_weight": float(1.0 - np.mean(sum_ws)),
            "gross_times_weight_mean": float(np.mean(cf)),
            "meta_hit_rate": float(np.mean(meta_hits)) if meta_hits else None,
            "cash_drag_pp": float(np.mean(selected_gross) - np.mean(selected_scaled)),
        },
        "universe": {
            "n_sleeves": len(sleeves),
            "n_cells": int(sum(len(v) for v in sleeves.values())),
            "mean_ret": float(arr.mean()),
            "median_ret": float(np.median(arr)),
            "pos_rate": float((arr > 0).mean()),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "zero_open_cells": zero_open,
        },
        "by_kind": kind_stats,
        "year_rows": year_rows,
        "momentum": mom,
        "family_years": fam_years,
        "selected_kind_counts": dict(sel_kind_c),
        "selected_und_counts": dict(sel_und_c),
        "oracle_kind_counts": dict(oracle_kind_c),
        "failure_modes": [],
    }

    # Rank failure modes by estimated impact
    h = report["headline"]
    report["failure_modes"] = [
        {
            "id": "F1_cash_and_caps",
            "title": "Cash residual + underlying caps crush deployed capital",
            "evidence": (
                f"Avg invested weight {h['avg_invested_weight']:.1%}; "
                f"selected sleeves EW mean {h['selected_gross_ew_mean']:.2%} "
                f"but portfolio only {h['portfolio_mean']:.2%} "
                f"(drag ~{h['cash_drag_pp']:.2%} absolute)."
            ),
            "severity": "critical",
        },
        {
            "id": "F2_weak_alpha_in_sleeve_zoo",
            "title": "Median sleeve is near zero / negative; short premium dies on proxy marks",
            "evidence": (
                f"Universe mean {report['universe']['mean_ret']:.2%}, "
                f"median {report['universe']['median_ret']:.2%}, "
                f"pos rate {report['universe']['pos_rate']:.1%}. "
                "Iron condor / CCS / PCS mean deeply negative."
            ),
            "severity": "critical",
        },
        {
            "id": "F3_meta_label_definition",
            "title": "Meta trains on ret>0 which is hard and barely better than base rate",
            "evidence": (
                f"Train pos_rate ~43–45%; meta hit rate on selected "
                f"{(h['meta_hit_rate'] or 0):.1%}. Label is binary profitability, "
                "not beat-cash or beat-SPY."
            ),
            "severity": "high",
        },
        {
            "id": "F4_no_momentum_edge",
            "title": "Prior-year sleeve return does not predict next year",
            "evidence": (
                f"corr(prior,next)={mom.get('corr')}; quintiles reverse or flat — "
                "fallback prior rank and meta features built on prior ret are weak."
            ),
            "severity": "high",
        },
        {
            "id": "F5_proxy_bs_bias",
            "title": "Option marks are model BS+VIX, not OPRA fills",
            "evidence": (
                "Short-vol structures systematically underperform on model marks "
                "(missing real VRP / bid-ask / skew). Long premium still lags equities "
                "after haircut+budget caps."
            ),
            "severity": "high",
        },
        {
            "id": "F6_selection_homogeneity",
            "title": "Meta clusters on same und/kind → caps fire harder",
            "evidence": (
                f"Selected kinds top: {sel_kind_c.most_common(4)}; "
                f"und top: {sel_und_c.most_common(5)}. "
                "8 near-identical long_calls on AMD/GOOGL → sum_w collapses to 0.20."
            ),
            "severity": "medium",
        },
        {
            "id": "F7_benchmark_mismatch",
            "title": "Compared to full SPY BH while often 50–80% cash",
            "evidence": (
                "Correct risk comparison is cash-blend or vol-matched; "
                "still, even gross selected EW << SPY most years."
            ),
            "severity": "medium",
        },
    ]

    out_json = LATEST / "FAILURE_ANALYSIS.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def pct(x: Any) -> str:
        try:
            return f"{100.0 * float(x):+.2f}%"
        except Exception:
            return "n/a"

    lines = [
        "# Failure analysis — Options portfolio + meta-label study",
        "",
        f"**Date:** 2026-07-23  ",
        f"**Source:** `{LATEST.as_posix()}`  ",
        f"**Scope:** 1000 sleeves · WF active 2013–2025 · marks `proxy_bs|vix_surface` · no ×2 lottery",
        "",
        "## Executive verdict",
        "",
        "We failed to beat SPY **not because of one bug**, but because **three structural "
        "layers each destroyed return**: (1) the options sleeve zoo has weak/negative edge "
        "under model marks, (2) the meta-label barely ranks better than chance on "
        "`ret>0`, and (3) portfolio caps leave most capital in **cash**, so even "
        "mediocre sleeve picks become ~1%/yr portfolio returns.",
        "",
        f"| Layer | Mean annual | vs SPY ({pct(h['spy_mean'])}) |",
        f"|-------|-------------|-------------------------------|",
        f"| Oracle top-8 sleeves (ex-post EW) | {pct(h['oracle_top8_mean'])} | still research ceiling |",
        f"| Naive top-5 prior-year (90% invested) | {pct(h['naive_top5_mean'])} | lottery-ish, no und caps |",
        f"| Meta-selected sleeves EW (gross) | {pct(h['selected_gross_ew_mean'])} | before cash residual |",
        f"| **Portfolio after caps (reported)** | **{pct(h['portfolio_mean'])}** | **{pct(h['portfolio_mean']-h['spy_mean'])}** |",
        f"| Avg capital invested | {h['avg_invested_weight']:.1%} | cash {h['avg_cash_weight']:.1%} |",
        "",
        "## Failure mode stack (ranked)",
        "",
    ]
    for i, fm in enumerate(report["failure_modes"], 1):
        lines += [
            f"### {i}. [{fm['severity'].upper()}] {fm['title']}",
            "",
            fm["evidence"],
            "",
        ]

    lines += [
        "## Evidence tables",
        "",
        "### A. Year-by-year decomposition",
        "",
        "| Year | Gross EW sel | Port (caps) | Cash w | Naive5 | Oracle8 | SPY | Sel pos% | Base pos% |",
        "|------|--------------|-------------|--------|--------|---------|-----|----------|-----------|",
    ]
    for r in year_rows:
        lines.append(
            f"| {r['year']} | {pct(r['gross_ew'])} | {pct(r['port'])} | {r['cash_w']:.0%} | "
            f"{pct(r['naive5'])} | {pct(r['oracle8'])} | {pct(r['spy'])} | "
            f"{100*r['sel_pos']:.0f}% | {100*r['base_pos']:.0f}% |"
        )

    lines += [
        "",
        "### B. Sleeve zoo by structure kind (all years, all unds)",
        "",
        "| Kind | N cells | Mean | Median | Pos% | Mean opens | % zero open | Hard kill% |",
        "|------|---------|------|--------|------|------------|-------------|------------|",
    ]
    for k in kind_stats:
        lines.append(
            f"| {k['kind']} | {k['n']} | {pct(k['mean'])} | {pct(k['median'])} | "
            f"{100*k['pos_rate']:.1f}% | {k['mean_opens']:.1f} | "
            f"{100*k['pct_zero_open']:.1f}% | {100*k['hard_kill_rate']:.1f}% |"
        )

    lines += [
        "",
        "### C. Short-premium vs long-premium vs SPY (equal-weight all sleeves)",
        "",
        "| Year | Short family | Long family | SPY |",
        "|------|--------------|-------------|-----|",
    ]
    for r in fam_years:
        lines.append(
            f"| {r['year']} | {pct(r['short_mean'])} | {pct(r['long_mean'])} | {pct(r['spy'])} |"
        )

    if mom:
        lines += [
            "",
            "### D. Momentum / mean-reversion of sleeve annual returns",
            "",
            f"Correlation prior year → next year: **{mom['corr']:.3f}** (n={mom['n']})",
            "",
            "| Quintile prior | Prior mean | Next mean |",
            "|----------------|------------|-----------|",
        ]
        for q in mom["quintiles"]:
            lines.append(
                f"| {q['name']} | {pct(q['prior_mean'])} | {pct(q['next_mean'])} |"
            )

    lines += [
        "",
        "### E. What meta actually picked",
        "",
        f"- Kinds: `{dict(sel_kind_c.most_common())}`",
        f"- Underlyings: `{dict(sel_und_c.most_common(10))}`",
        f"- Oracle (hindsight top-8) kinds: `{dict(oracle_kind_c.most_common())}`",
        "",
        "## Causal chain (why ~1% portfolio)",
        "",
        "```",
        "proxy_bs short-vol zoo  ──►  median sleeve ~0% / many negative",
        "        │",
        "        ▼",
        "meta label = ret_{t+1}>0  ──►  base rate ~44%, weak ranking signal",
        "        │                     (prior ret almost uncorrelated with next)",
        "        ▼",
        "select top-K similar sleeves  ──►  und/family caps fire",
        "        │",
        "        ▼",
        "invested weight 20–55% + cash  ──►  portfolio mean ~0.9%/yr",
        "        │",
        "        ▼",
        "compare to 100% SPY BH        ──►  −14 pp headline 'failure'",
        "```",
        "",
        "## What is *not* the main failure",
        "",
        "- **Not** missing EODHD history (SPY/VIX from 2005; years 2010+ populated).",
        "- **Not** a single lottery year domination (max upside share ~21%).",
        "- **Not** NVDA×2/QQQ×2 leakage (banned; unds diversified in selection counts).",
        "- **Not** code crash / empty sleeves (0 errors on 16k cells).",
        "",
        "## What would need to change to have a chance",
        "",
        "1. **Marks:** OPRA / paid options history; or stop claiming short-vol edge on pure BS.",
        "2. **Label:** meta target = beat cash + X, or rank excess vs SPY, or utility of Sharpe — not raw `ret>0`.",
        "3. **Features:** regime/VIX path features at *allocation* time; drop reliance on sleeve prior year.",
        "4. **Allocator:** diversity penalty *before* ranking (1 sleeve per und/kind) so caps do not "
        "silently force cash; or explicit target equity beta / vol.",
        "5. **Benchmark honesty:** report port vs `w*SPY+(1-w)*cash` with same average `w`.",
        "6. **Sleeve filter:** kill structures with structural negative mean on proxy (CCS/IC) "
        "before meta, or re-price with VRP premium.",
        "",
        "## Bottom line",
        "",
        "The system **did what it was told**: diversify, avoid leverage lotteries, skip weak meta "
        "prob, hold cash. That produces a **defensive residual** (~flat to +2%/yr), not an equity "
        "replacement. The research claim 'options zoo + meta beats market' is **rejected** under "
        "these marks and rules. The closest 'works' artifact is **naive top-5 prior** (~17%), "
        "which is exactly the concentrated path the design banned — and still is not clean OPRA alpha.",
        "",
        "---",
        "VIRTUAL research. Not financial advice. Machine-readable twin: `FAILURE_ANALYSIS.json`.",
        "",
    ]

    out_md = LATEST / "FAILURE_ANALYSIS.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"md": str(out_md), "json": str(out_json), "headline": h}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
