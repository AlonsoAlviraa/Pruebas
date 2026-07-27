#!/usr/bin/env python3
"""Print top-10 equity mega strategies by total return (unique IDs)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
latest = ROOT / "reports" / "equity_mega_lever" / "latest"
rows = json.loads((latest / "sleeve_results.json").read_text(encoding="utf-8"))
summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
spy = float(summary.get("spy_bh") or 0)
qqq = float(summary.get("qqq_bh") or 0)

# unique by strategy_id
by_id = {}
for r in rows:
    if r.get("error"):
        continue
    sid = str(r.get("strategy_id") or "")
    tr = float(r.get("total_return") or 0)
    if sid not in by_id or tr > float(by_id[sid].get("total_return") or 0):
        by_id[sid] = r
uniq = list(by_id.values())


def fmt(r, rank: int) -> str:
    tr = float(r.get("total_return") or 0)
    dd = float(r.get("max_dd") or 0)
    L = float(r.get("mean_leverage") or 0)
    cost = float(r.get("cost_drag_total") or 0)
    w = "Y" if r.get("wiped") else ""
    return (
        f"| {rank} | `{r.get('strategy_id')}` | {r.get('kind')} | {r.get('underlying')} | "
        f"{100*tr:+.1f}% | {100*dd:+.1f}% | {L:.2f} | {cost:.4f} | {w} | "
        f"{100*(tr-spy):+.1f} pp | {100*(tr-qqq):+.1f} pp |"
    )


top_all = sorted(uniq, key=lambda r: float(r.get("total_return") or 0), reverse=True)[:10]
top_nw = sorted(
    [r for r in uniq if not r.get("wiped")],
    key=lambda r: float(r.get("total_return") or 0),
    reverse=True,
)[:10]

lines = [
    "# Top 10 estrategias por retorno total",
    "",
    f"Periodo study: 2015–2025 · SPY BH **{100*spy:+.1f}%** · QQQ BH **{100*qqq:+.1f}%**",
    f"IDs únicos en zoo: {len(uniq)}",
    "",
    "## A) Top 10 por retorno total (incluye wiped por hard DD)",
    "",
    "| # | ID | Kind | Und | Retorno | MaxDD | Mean L | Cost drag | Wiped | vs SPY | vs QQQ |",
    "|---|----|------|-----|---------|-------|--------|-----------|-------|--------|--------|",
]
for i, r in enumerate(top_all, 1):
    lines.append(fmt(r, i))

lines += [
    "",
    "_Wiped = la estrategia tocó el tope de drawdown y dejó de arriesgar; el retorno "
    "total puede quedar 'congelado' y no es comparable a un path vivo hasta 2025._",
    "",
    "## B) Top 10 por retorno total **sin wiped** (recomendado)",
    "",
    "| # | ID | Kind | Und | Retorno | MaxDD | Mean L | Cost drag | Wiped | vs SPY | vs QQQ |",
    "|---|----|------|-----|---------|-------|--------|-----------|-------|--------|--------|",
]
for i, r in enumerate(top_nw, 1):
    lines.append(fmt(r, i))

lines += [
    "",
    "### Año a año — Top 5 no wiped",
    "",
]
for i, r in enumerate(top_nw[:5], 1):
    lines.append(f"#### {i}. `{r.get('strategy_id')}` ({r.get('kind')} / {r.get('underlying')})")
    lines.append("")
    lines.append("| Año | Ret |")
    lines.append("|-----|-----|")
    for y, v in sorted((r.get("year_returns") or {}).items()):
        lines.append(f"| {y} | {100*float(v):+.1f}% |")
    lines.append("")

lines += [
    "---",
    "Costes ya incluidos (financiación L−1 + comisiones/slippage). VIRTUAL.",
    "",
]
text = "\n".join(lines)
(latest / "TOP10_BY_RETURN.md").write_text(text, encoding="utf-8")
print(text)
