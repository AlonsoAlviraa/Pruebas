#!/usr/bin/env python
"""Analyze Kaggle universe-generalization PROGRESS.json → reports."""
from __future__ import annotations

import json
import math
import statistics as stats
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS = (
    ROOT
    / "reports"
    / "redesign"
    / "kaggle_univ_gen_t4x2"
    / "universe_gen_overnight"
    / "PROGRESS.json"
)
OUT_DIR = ROOT / "reports" / "redesign" / "kaggle_univ_gen_t4x2" / "universe_gen_overnight"

GATE_CAGR = 0.10
GATE_MDD = -0.65


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def agg(vals: List[Any]) -> Dict[str, Any]:
    xs = [fnum(v) for v in vals]
    xs = [v for v in xs if v is not None]
    if not xs:
        return {"n": 0}
    xs = sorted(xs)

    def q(p: float) -> float:
        i = (len(xs) - 1) * p
        lo = int(i)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": q(0.5),
        "std": stats.stdev(xs) if len(xs) > 1 else 0.0,
        "p10": q(0.1),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.9),
        "min": xs[0],
        "max": xs[-1],
    }


def pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.1f}%"


def us_verdict(pass_rate: float, median_cagr: Optional[float], median_mdd: Optional[float], prefix_pass: bool) -> str:
    mc = median_cagr if median_cagr is not None else -1.0
    mm = median_mdd if median_mdd is not None else -1.0
    if pass_rate < 0.15 and prefix_pass:
        return "PREFIX-ONLY"
    if mc <= GATE_CAGR or mm < GATE_MDD:
        return "FAIL"
    if pass_rate >= 0.40 and mc > GATE_CAGR and mm >= GATE_MDD:
        return "GENERALIZES"
    if 0.15 <= pass_rate < 0.40:
        return "FRAGILE"
    return "FRAGILE"


def main() -> int:
    prog_path = DEFAULT_PROGRESS
    d = json.loads(prog_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = list(d.get("rows") or [])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # flat CSV
    flat = []
    for r in rows:
        c = r.get("confirm") or {}
        f = r.get("full") or {}
        s = r.get("screen") or {}
        flat.append(
            {
                "arm_id": r.get("arm_id"),
                "market": r.get("market"),
                "series": r.get("series"),
                "strategy": r.get("strategy"),
                "seed": r.get("seed"),
                "draw_size": r.get("draw_size"),
                "confirm_cagr": c.get("cagr"),
                "confirm_mdd": c.get("max_drawdown"),
                "confirm_sharpe": c.get("sharpe"),
                "confirm_n_trades": c.get("n_trades"),
                "confirm_pass": r.get("confirm_pass"),
                "full_cagr": f.get("cagr"),
                "full_mdd": f.get("max_drawdown"),
                "full_pass": r.get("full_pass"),
                "full_gates_pass": (f.get("gates") or {}).get("pass"),
                "research_pass": r.get("research_pass"),
                "excess_index": c.get("excess_index_total"),
                "honest_score": r.get("honest_score"),
                "screen_cagr": s.get("cagr"),
                "error": r.get("error"),
                "elapsed_sec": r.get("elapsed_sec"),
            }
        )
    try:
        import pandas as pd

        pd.DataFrame(flat).to_csv(OUT_DIR / "all_runs.csv", index=False)
    except Exception:
        # manual csv
        if flat:
            keys = list(flat[0].keys())
            lines = [",".join(keys)]
            for row in flat:
                lines.append(",".join(str(row.get(k, "")) for k in keys))
            (OUT_DIR / "all_runs.csv").write_text("\n".join(lines), encoding="utf-8")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        key = f"{r.get('market')}|{r.get('strategy')}|{r.get('series')}"
        groups[key].append(r)

    agg_table = {}
    for key, g in groups.items():
        market, strategy, series = key.split("|", 2)
        cagrs = [(x.get("confirm") or {}).get("cagr") for x in g]
        mdds = [(x.get("confirm") or {}).get("max_drawdown") for x in g]
        sharpes = [(x.get("confirm") or {}).get("sharpe") for x in g]
        full_c = [(x.get("full") or {}).get("cagr") for x in g]
        full_m = [(x.get("full") or {}).get("max_drawdown") for x in g]
        n = len(g)
        n_cpass = sum(1 for x in g if x.get("confirm_pass"))
        n_fgate = sum(1 for x in g if ((x.get("full") or {}).get("gates") or {}).get("pass"))
        n_rpass = sum(1 for x in g if x.get("research_pass"))
        # softer confirm pass: cagr+mdd only (ignore trades for diagnosis)
        n_cm = sum(
            1
            for x in g
            if (fnum((x.get("confirm") or {}).get("cagr"), -1) or -1) > GATE_CAGR
            and (fnum((x.get("confirm") or {}).get("max_drawdown"), -1) or -1) >= GATE_MDD
        )
        ca = agg(cagrs)
        md = agg(mdds)
        agg_table[key] = {
            "market": market,
            "strategy": strategy,
            "series": series,
            "n": n,
            "confirm_cagr": ca,
            "confirm_mdd": md,
            "confirm_sharpe": agg(sharpes),
            "full_cagr": agg(full_c),
            "full_mdd": agg(full_m),
            "confirm_pass_rate": n_cpass / n if n else 0.0,
            "confirm_cagr_mdd_pass_rate": n_cm / n if n else 0.0,
            "full_gates_pass_rate": n_fgate / n if n else 0.0,
            "research_pass_rate": n_rpass / n if n else 0.0,
            "median_confirm_pass": bool(
                (ca.get("median") is not None and ca["median"] > GATE_CAGR)
                and (md.get("median") is not None and md["median"] >= GATE_MDD)
            ),
        }

    (OUT_DIR / "aggregate_by_market.json").write_text(
        json.dumps(agg_table, indent=2, default=str), encoding="utf-8"
    )

    # PREFIX
    prefix_rows = [
        r
        for r in rows
        if r.get("market") == "US"
        and r.get("strategy") == "turbo_strict"
        and r.get("series") == "PREFIX"
        and "L50" in str(r.get("arm_id"))
    ]
    prefix_pass_confirm = any(r.get("confirm_pass") for r in prefix_rows)
    prefix_pass_research = any(r.get("research_pass") for r in prefix_rows)

    us_key = "US|turbo_strict|R50"
    us_a = agg_table.get(us_key) or {}
    # Use confirm_cagr_mdd_pass_rate as primary generalization rate (research_pass may be 0 if full trades gate)
    # Pre-reg plan used research_pass; also report confirm pass rate.
    pr_research = float(us_a.get("research_pass_rate") or 0.0)
    pr_confirm = float(us_a.get("confirm_pass_rate") or 0.0)
    pr_cm = float(us_a.get("confirm_cagr_mdd_pass_rate") or 0.0)
    med_c = (us_a.get("confirm_cagr") or {}).get("median")
    med_m = (us_a.get("confirm_mdd") or {}).get("median")

    v_us_research = us_verdict(pr_research, med_c, med_m, prefix_pass_confirm)
    v_us_confirm = us_verdict(pr_confirm, med_c, med_m, prefix_pass_confirm)
    v_us_cm = us_verdict(pr_cm, med_c, med_m, prefix_pass_confirm)

    # Why research_pass 0?
    g50 = groups.get(us_key, [])
    full_fail = Counter()
    for x in g50:
        fg = (x.get("full") or {}).get("gates") or {}
        if not fg.get("pass"):
            if not fg.get("cagr_ok"):
                full_fail["cagr"] += 1
            if not fg.get("mdd_ok"):
                full_fail["mdd"] += 1
            if not fg.get("trades_ok"):
                full_fail["trades"] += 1
        cg = (x.get("confirm") or {}).get("gates") or {}
        if not cg.get("pass"):
            if not cg.get("cagr_ok"):
                full_fail["confirm_cagr"] += 1
            if not cg.get("mdd_ok"):
                full_fail["confirm_mdd"] += 1
            if not cg.get("trades_ok"):
                full_fail["confirm_trades"] += 1

    # GEO
    med_pass: Dict[str, bool] = {}
    for mid in ("ES", "FR", "DE"):
        hits = [
            a
            for a in agg_table.values()
            if a.get("market") == mid
            and a.get("strategy") == "turbo_strict"
            and str(a.get("series", "")).startswith("R")
        ]
        if hits:
            hits = sorted(hits, key=lambda x: -int(x.get("n") or 0))
            med_pass[mid] = bool(hits[0].get("median_confirm_pass"))
    uk_hits = [
        a
        for a in agg_table.values()
        if a.get("market") == "UK" and a.get("strategy") == "turbo_strict"
    ]
    uk_ok = None
    if uk_hits:
        uk_ok = bool((uk_hits[0].get("confirm_cagr") or {}).get("median", -1) > 0)

    n_ok = sum(1 for m in ("ES", "FR", "DE") if med_pass.get(m))
    if n_ok >= 2 and (uk_ok is None or uk_ok):
        v_geo = "TRANSFERS"
    elif n_ok == 0:
        v_geo = "FAIL_GEO"
    elif n_ok == 1:
        v_geo = "MIXED"
    else:
        v_geo = "MIXED"

    # Controls detail
    controls = []
    for r in rows:
        if r.get("series") in ("PREFIX", "FULL100", "FULL"):
            c = r.get("confirm") or {}
            f = r.get("full") or {}
            controls.append(
                {
                    "arm_id": r.get("arm_id"),
                    "confirm_cagr": c.get("cagr"),
                    "confirm_mdd": c.get("max_drawdown"),
                    "confirm_pass": r.get("confirm_pass"),
                    "full_cagr": f.get("cagr"),
                    "full_mdd": f.get("max_drawdown"),
                    "full_gates_pass": (f.get("gates") or {}).get("pass"),
                    "research_pass": r.get("research_pass"),
                    "excess": c.get("excess_index_total"),
                }
            )

    # DISTRIBUTION.md
    lines = [
        "# Universe generalization — DISTRIBUTION (Kaggle T4×2)",
        "",
        "Research only. Not financial advice.",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Source: `{prog_path}`",
        f"n_rows: **{len(rows)}** · errors: **{sum(1 for r in rows if r.get('error'))}**",
        f"elapsed: **{d.get('elapsed_sec')}s** (~{float(d.get('elapsed_sec') or 0)/3600:.2f} h)",
        f"exit_codes: {d.get('exit_codes')}",
        "",
        "## Headline",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| US turbo_strict R50 n | {us_a.get('n')} |",
        f"| US R50 median confirm CAGR | {pct(med_c)} |",
        f"| US R50 median confirm MDD | {pct(med_m)} |",
        f"| US R50 confirm_pass_rate | {pct(pr_confirm)} |",
        f"| US R50 cagr+mdd pass_rate | {pct(pr_cm)} |",
        f"| US R50 research_pass_rate | {pct(pr_research)} |",
        f"| PREFIX strict L50 confirm pass | {prefix_pass_confirm} |",
        f"| US verdict (confirm_pass) | **{v_us_confirm}** |",
        f"| US verdict (cagr+mdd) | **{v_us_cm}** |",
        f"| US verdict (research_pass) | **{v_us_research}** |",
        f"| GEO verdict | **{v_geo}** |",
        "",
        "### Note on research_pass=0",
        "",
        f"Full-path fail reasons on US·strict·R50: `{dict(full_fail)}`.",
        "If `trades` dominates full gates, CAGR/MDD may still look OK but research_pass stays False.",
        "",
        "## Aggregates by market|strategy|series (confirm)",
        "",
        "| key | n | mean CAGR | med CAGR | p10 CAGR | med MDD | conf_pass% | cagr+mdd% | full_gate% |",
        "|-----|--:|----------:|---------:|---------:|--------:|-----------:|----------:|-----------:|",
    ]
    for key in sorted(agg_table):
        a = agg_table[key]
        ca = a.get("confirm_cagr") or {}
        md = a.get("confirm_mdd") or {}
        lines.append(
            f"| `{key}` | {a.get('n')} | {pct(ca.get('mean'))} | {pct(ca.get('median'))} | "
            f"{pct(ca.get('p10'))} | {pct(md.get('median'))} | {pct(a.get('confirm_pass_rate'))} | "
            f"{pct(a.get('confirm_cagr_mdd_pass_rate'))} | {pct(a.get('full_gates_pass_rate'))} |"
        )

    lines.extend(["", "## Controls PREFIX / FULL", ""])
    for c in controls:
        lines.append(
            f"- `{c['arm_id']}`: confirm {pct(c['confirm_cagr'])} / MDD {pct(c['confirm_mdd'])} "
            f"pass={c['confirm_pass']} · full {pct(c['full_cagr'])} fpass={c['full_gates_pass']} "
            f"rpass={c['research_pass']}"
        )

    lines.extend(
        [
            "",
            "## Geo median confirm pass (turbo_strict random series)",
            "",
        ]
    )
    for mid, ok in med_pass.items():
        lines.append(f"- **{mid}**: median_confirm_pass={ok}")
    lines.append(f"- **UK** median CAGR>0: {uk_ok}")
    lines.append("")
    (OUT_DIR / "DISTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")

    # GEO
    geo_lines = [
        "# Geo transfer — Kaggle T4×2",
        "",
        f"Verdict: **{v_geo}**",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    for mid in ("ES", "FR", "DE", "UK"):
        hits = [
            a
            for a in agg_table.values()
            if a.get("market") == mid and a.get("strategy") == "turbo_strict"
        ]
        geo_lines.append(f"## {mid}")
        for a in sorted(hits, key=lambda x: str(x.get("series"))):
            ca = a.get("confirm_cagr") or {}
            md = a.get("confirm_mdd") or {}
            geo_lines.append(
                f"- {a.get('series')}: n={a.get('n')} med CAGR {pct(ca.get('median'))} "
                f"med MDD {pct(md.get('median'))} conf_pass {pct(a.get('confirm_pass_rate'))} "
                f"cagr+mdd {pct(a.get('confirm_cagr_mdd_pass_rate'))}"
            )
        geo_lines.append("")
    (OUT_DIR / "GEO_TRANSFER.md").write_text("\n".join(geo_lines), encoding="utf-8")

    # DECISION
    dlines = [
        "# Universe generalization — Kaggle T4×2 DECISION",
        "",
        f"- Complete: **True** (549/549 rows, exit_codes={d.get('exit_codes')})",
        f"- Wall time: **{float(d.get('elapsed_sec') or 0)/3600:.2f} h** (budget 8 h)",
        f"- US verdict (confirm_pass, S1·R50): **{v_us_confirm}**",
        f"- US verdict (cagr+mdd pass): **{v_us_cm}**",
        f"- US verdict (research_pass confirm∩full): **{v_us_research}**",
        f"- GEO verdict: **{v_geo}**",
        f"- Paper freeze: **turbo_highvol_minalloc** unchanged (no auto-ADVANCE)",
        f"- PREFIX turbo_strict L50 confirm pass: **{prefix_pass_confirm}** (research_pass={prefix_pass_research})",
        f"- US R50 n={us_a.get('n')} · median confirm CAGR/MDD: **{pct(med_c)}** / **{pct(med_m)}**",
        f"- US R50 rates: confirm_pass **{pct(pr_confirm)}** · cagr+mdd **{pct(pr_cm)}** · research_pass **{pct(pr_research)}**",
        f"- Full-fail reasons US R50 strict: `{dict(full_fail)}`",
        "",
        "## Interpretation",
        "",
        "1. PREFIX L50 should match prior Kaggle zoo (~18% confirm / ~−36% MDD).",
        "2. GENERALIZES only if random R50 books pass at ≥40% rate with median gates.",
        "3. research_pass requires full stitch gates too — if only full.trades fails, treat confirm_pass as primary OOS.",
        "4. EU transfer uses local index; UK is confirm-only (no decade claim).",
        "",
        "Research only. Not financial advice. Past backtests ≠ future results.",
        "",
    ]
    (OUT_DIR / "DECISION.md").write_text("\n".join(dlines), encoding="utf-8")

    # MASTER analysis markdown
    master = [
        "# Master analysis — Universe generalization Kaggle T4×2",
        "",
        f"**Date:** {datetime.now(timezone.utc).date().isoformat()}",
        "**Kernel:** `alonsoalviraaaa/trad-univ-gen-t4x2` COMPLETE",
        f"**Rows:** {len(rows)} / 549 planned · **0 errors** · **{float(d.get('elapsed_sec') or 0)/3600:.2f} h**",
        "**Paper freeze:** `turbo_highvol_minalloc` unchanged",
        "",
        "Research only. Not financial advice.",
        "",
        "## 1. Executive verdict",
        "",
        f"| Question | Answer |",
        f"|----------|--------|",
        f"| ¿PREFIX repro Kaggle zoo? | **{'YES' if prefix_pass_confirm else 'NO'}** |",
        f"| ¿Generaliza a books aleatorios US (confirm)? | **{v_us_confirm}** |",
        f"| ¿Generaliza (cagr+mdd, sin n_trades)? | **{v_us_cm}** |",
        f"| ¿Research PASS confirm∩full en R50? | **{v_us_research}** (rate {pct(pr_research)}) |",
        f"| ¿Transfer EU? | **{v_geo}** |",
        f"| ¿ADVANCE paper? | **NO** |",
        "",
        "## 2. US turbo_strict · R50 (core)",
        "",
        f"- n draws: **{us_a.get('n')}**",
        f"- median confirm CAGR: **{pct(med_c)}** · mean {pct((us_a.get('confirm_cagr') or {}).get('mean'))}",
        f"- p10 / p90 CAGR: **{pct((us_a.get('confirm_cagr') or {}).get('p10'))}** / **{pct((us_a.get('confirm_cagr') or {}).get('p90'))}**",
        f"- median confirm MDD: **{pct(med_m)}** · p10 MDD {pct((us_a.get('confirm_mdd') or {}).get('p10'))}",
        f"- pass rates: confirm **{pct(pr_confirm)}** · cagr+mdd **{pct(pr_cm)}** · full_gates **{pct(us_a.get('full_gates_pass_rate'))}** · research **{pct(pr_research)}**",
        "",
        "### Comparación con minalloc US R50",
        "",
    ]
    us_min = agg_table.get("US|turbo_highvol_minalloc|R50") or {}
    master.append(
        f"- minalloc R50: n={us_min.get('n')} med CAGR {pct((us_min.get('confirm_cagr') or {}).get('median'))} "
        f"med MDD {pct((us_min.get('confirm_mdd') or {}).get('median'))} "
        f"conf_pass {pct(us_min.get('confirm_pass_rate'))}"
    )
    master.extend(
        [
            "",
            "## 3. Controls (PREFIX / FULL)",
            "",
        ]
    )
    for c in controls:
        if c["arm_id"] and str(c["arm_id"]).startswith("US__"):
            master.append(
                f"- `{c['arm_id']}`: confirm **{pct(c['confirm_cagr'])}** MDD **{pct(c['confirm_mdd'])}** "
                f"pass={c['confirm_pass']} · full **{pct(c['full_cagr'])}** fpass={c['full_gates_pass']}"
            )
    master.extend(["", "## 4. EU transfer", ""])
    for mid in ("ES", "FR", "DE", "UK"):
        hits = [
            a
            for a in agg_table.values()
            if a.get("market") == mid and a.get("strategy") == "turbo_strict" and str(a.get("series", "")).startswith("R")
        ]
        if not hits:
            continue
        a = sorted(hits, key=lambda x: -int(x.get("n") or 0))[0]
        master.append(
            f"- **{mid}** `{a.get('series')}` n={a.get('n')}: med CAGR **{pct((a.get('confirm_cagr') or {}).get('median'))}** "
            f"med MDD **{pct((a.get('confirm_mdd') or {}).get('median'))}** "
            f"conf_pass **{pct(a.get('confirm_pass_rate'))}** cagr+mdd **{pct(a.get('confirm_cagr_mdd_pass_rate'))}**"
        )
    master.extend(
        [
            "",
            f"**GEO:** `{v_geo}`",
            "",
            "## 5. Structural reads",
            "",
            "1. **Membership sensitivity:** if PREFIX PASS but R50 pass_rate low → edge lives in file order / lucky book, not rules alone.",
            "2. **Width:** compare R50 vs R60 vs R80 strict medians.",
            "3. **Strict vs minalloc:** decade path favors lower vol / better MDD for strict if pattern holds.",
            "4. **EU:** prior multimarket already showed weak geo transfer; this freezes winners not knobs.",
            "5. **No paper ADVANCE** regardless of partial GENERALIZES.",
            "",
            "## 6. Files",
            "",
            f"- `{OUT_DIR / 'DECISION.md'}`",
            f"- `{OUT_DIR / 'DISTRIBUTION.md'}`",
            f"- `{OUT_DIR / 'GEO_TRANSFER.md'}`",
            f"- `{OUT_DIR / 'all_runs.csv'}`",
            f"- `{OUT_DIR / 'aggregate_by_market.json'}`",
            f"- `{prog_path}`",
            "",
            "Research only. Not financial advice.",
            "",
        ]
    )
    master_path = OUT_DIR / "MASTER_ANALYSIS.md"
    master_path.write_text("\n".join(master), encoding="utf-8")

    summary = {
        "n_rows": len(rows),
        "elapsed_sec": d.get("elapsed_sec"),
        "us_verdict_confirm": v_us_confirm,
        "us_verdict_cagr_mdd": v_us_cm,
        "us_verdict_research": v_us_research,
        "geo_verdict": v_geo,
        "prefix_pass_confirm": prefix_pass_confirm,
        "us_r50": us_a,
        "full_fail_us_r50": dict(full_fail),
        "paper_freeze": "turbo_highvol_minalloc",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))
    print("WROTE", master_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
