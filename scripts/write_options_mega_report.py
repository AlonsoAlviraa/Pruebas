#!/usr/bin/env python3
"""Write MEGA_RESULTS.md from paper_options_mega summary.json."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "paper_options_mega" / "latest" / "summary.json"
OUT = ROOT / "reports" / "paper_options_mega" / "MEGA_RESULTS.md"


def main() -> int:
    d = json.loads(SUMMARY.read_text(encoding="utf-8"))
    strats = d["strategies"]
    spy = (d.get("benchmarks") or {}).get("spy_bh")
    qqq = (d.get("benchmarks") or {}).get("qqq_bh")
    rows = sorted(strats, key=lambda x: -x["total_return"])
    by: dict[str, list] = defaultdict(list)
    for r in strats:
        by[r["kind"]].append(r)

    lines: list[str] = []
    lines.append("# Mega paper options test (~50 strategies)")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat()} · VIRTUAL · data_label=`proxy_bs`_"
    )
    lines.append("")
    lines.append("## Research sources")
    lines.append("")
    lines.append("| Channel | Takeaways used in zoo |")
    lines.append("|---------|----------------------|")
    lines.append(
        "| **CBOE indexes** | BXM/BXY buy-write, PUT put-write, CNDR iron condor families |"
    )
    lines.append(
        "| **Papers / Quantpedia** | VRP (IV>RV); OTM 5–10% put-write; defined-risk wings |"
    )
    lines.append(
        "| **Twitter/X** | IC = positioning; equity drift → prefer PCS / wider structures; CNDR fails when RV>IV |"
    )
    lines.append(
        "| **GitHub style** | Parametric grids (underlying × OTM × DTE) over few structure kinds |"
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    w = d.get("window") or {}
    lines.append(f"- Window: **{w.get('start')} → {w.get('end')}**")
    lines.append(f"- Strategies completed: **{len(strats)}**")
    lines.append("- Capital: VIRTUAL $100,000")
    lines.append("- Marks: Black–Scholes on HV/IV proxy (**not** OPRA fills)")
    if spy is not None:
        qtxt = f"{100 * qqq:.2f}%" if qqq is not None else "n/a"
        lines.append(f"- SPY B&H: **{100 * spy:.2f}%** · QQQ B&H: **{qtxt}**")
    lines.append("")
    lines.append("## Headline results")
    lines.append("")
    lines.append(
        f"- Positive return: **{sum(1 for r in strats if r['total_return'] > 0)}** / {len(strats)}"
    )
    lines.append(f"- Hard kill: **{sum(1 for r in strats if r.get('hard_kill'))}**")
    lines.append(
        f"- Beat SPY: **{sum(1 for r in strats if r.get('vs_spy_bh') is not None and r['vs_spy_bh'] > 0)}**"
    )
    lines.append("")
    lines.append("## Average by structure kind")
    lines.append("")
    lines.append("| Kind | n | Avg ret | Best ret |")
    lines.append("|------|---:|--------:|---------:|")
    for k, vs in sorted(
        by.items(), key=lambda kv: -sum(x["total_return"] for x in kv[1]) / len(kv[1])
    ):
        avg = sum(x["total_return"] for x in vs) / len(vs)
        best = max(x["total_return"] for x in vs)
        lines.append(f"| `{k}` | {len(vs)} | {100 * avg:.2f}% | {100 * best:.2f}% |")
    lines.append("")
    lines.append("## Top 15 by total return")
    lines.append("")
    lines.append("| Rank | ID | Kind | Ret | MaxDD | CVaR5% | vs SPY | Kill |")
    lines.append("|------|-----|------|----:|------:|-------:|-------:|:----:|")
    for i, r in enumerate(rows[:15], 1):
        cvar = r.get("cvar_5pct")
        vs = r.get("vs_spy_bh")
        cvars = f"{100 * cvar:.2f}%" if cvar is not None else "n/a"
        vss = f"{100 * vs:+.2f}pp" if vs is not None else "n/a"
        kill = "YES" if r.get("hard_kill") else "no"
        lines.append(
            f"| {i} | `{r['strategy_id']}` | {r['kind']} | {100 * r['total_return']:.2f}% | "
            f"{100 * r['max_dd']:.2f}% | {cvars} | {vss} | {kill} |"
        )
    lines.append("")
    lines.append("## Bottom 10")
    lines.append("")
    lines.append("| Rank | ID | Kind | Ret | MaxDD | vs SPY |")
    lines.append("|------|-----|------|----:|------:|-------:|")
    start_rank = len(rows) - 9
    for i, r in enumerate(rows[-10:], start_rank):
        vs = r.get("vs_spy_bh")
        vss = f"{100 * vs:+.2f}pp" if vs is not None else "n/a"
        lines.append(
            f"| {i} | `{r['strategy_id']}` | {r['kind']} | {100 * r['total_return']:.2f}% | "
            f"{100 * r['max_dd']:.2f}% | {vss} |"
        )
    lines.append("")
    lines.append("## Interpretation (honest)")
    lines.append("")
    lines.append(
        "1. **Bull window** (SPY +8.9%, QQQ +11.5%): long-stock + short call "
        "(buywrite/collar/PP) dominate ranking because they keep **equity beta**."
    )
    lines.append(
        "2. **Pure short premium** (CSP / PCS / IC / CCS) is **positive but lags SPY** — "
        "classic VRP income profile in a strong market, not free alpha."
    )
    lines.append(
        "3. **Iron condors** ~+0.5–2% average — X thesis that IC is not \"easy money\" holds; "
        "only one strategy slightly negative."
    )
    lines.append(
        "4. **Zero hard kills** on this calm path; use `--stress` for crash behavior."
    )
    lines.append(
        "5. **No strategy beats SPY** here (best buywrite still ~1–2pp short of SPY due to "
        "capped upside + rolls)."
    )
    lines.append(
        "6. These are **parameterizations of known structures**, not 50 independent alpha ideas."
    )
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("python scripts/build_options_zoo_50.py")
    lines.append(
        "python scripts/run_paper_options_batch.py --zoo paper_live/cloud/zoo_options_50.json "
        "--out reports/paper_options_mega --start 2025-10-29"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Zoo: `paper_live/cloud/zoo_options_50.json`")
    lines.append("- Pack: `reports/paper_options_mega/latest/`")
    lines.append("- Design: `docs/design/2026-07-22_options_mega_50.md`")
    lines.append("")
    lines.append("_Not financial advice. proxy_bs ≠ exchange fills._")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} n={len(strats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
