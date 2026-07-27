#!/usr/bin/env python3
"""Year-by-year: which equity mega sleeves beat SPY (and optional QQQ)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):+.1f}%"
    except Exception:
        return "n/a"


def main() -> int:
    latest = ROOT / "reports" / "equity_mega_lever" / "latest"
    summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((latest / "sleeve_results.json").read_text(encoding="utf-8"))
    year_spy: Dict[str, Optional[float]] = summary.get("year_spy") or {}

    # Reconstruct year QQQ from feed if possible
    year_qqq: Dict[str, Optional[float]] = {}
    try:
        from datetime import date
        from paper_live.data.eodhd_client import build_eodhd_feed

        cache = ROOT / "reports" / "equity_mega_lever" / "eodhd_cache"
        feed, _ = build_eodhd_feed(
            ["SPY", "QQQ"], start="2010-01-01", cache_dir=cache, min_history=40
        )
        days = list(feed.days)

        def bh(ticker: str, y: int) -> Optional[float]:
            ys, ye = date(y, 1, 2), date(y, 12, 31)
            s = next((d for d in days if d >= ys), None)
            e = next((d for d in reversed(days) if d <= ye), None)
            if s is None or e is None:
                return None
            b0, b1 = feed.bar(ticker, s), feed.bar(ticker, e)
            if not b0 or not b1 or float(b0.close) <= 0:
                return None
            return float(b1.close) / float(b0.close) - 1.0

        for y in sorted(year_spy.keys()):
            year_qqq[y] = bh("QQQ", int(y))
            if year_spy.get(y) is None:
                year_spy[y] = bh("SPY", int(y))
    except Exception as e:
        print("QQQ year BH skipped:", e, file=sys.stderr)

    valid = [r for r in rows if r.get("year_returns") and not r.get("error")]
    years = sorted(set(year_spy.keys()) | set(year_qqq.keys()))

    report: Dict[str, Any] = {
        "n_strategies": len(rows),
        "n_with_years": len(valid),
        "years": {},
    }

    md: List[str] = [
        "# Equity mega — estrategias que superan al índice (año a año)",
        "",
        f"**Universo:** {len(valid)} sleeves con `year_returns` · ventana estudio 2015–2025",
        f"**Índices:** SPY (principal) y QQQ (referencia growth)",
        f"**Nota:** retornos ya netos de financiación (L−1) + comisiones/slippage de rebalance.",
        f"**Wiped:** si `hard_dd` mató el sleeve a mitad de muestra, años posteriores pueden ser 0%.",
        "",
        "## Resumen por año vs SPY",
        "",
        "| Año | SPY | QQQ | # con dato | # > SPY | % > SPY | Mediana sleeves | Mejor sleeve | Ret mejor | Kind |",
        "|-----|-----|-----|------------|---------|---------|-----------------|--------------|-----------|------|",
    ]

    for y in years:
        spy = year_spy.get(y)
        qqq = year_qqq.get(y)
        if spy is None:
            continue
        beat_spy: List[Dict[str, Any]] = []
        beat_qqq: List[Dict[str, Any]] = []
        all_rets: List[float] = []
        for r in valid:
            yr = r.get("year_returns") or {}
            if y not in yr:
                continue
            tr = float(yr[y])
            all_rets.append(tr)
            row = {
                "id": r.get("strategy_id"),
                "kind": r.get("kind"),
                "und": r.get("underlying"),
                "ret": tr,
                "excess_spy": tr - float(spy),
                "wiped_flag": bool(r.get("wiped")),
                "mean_L": r.get("mean_leverage"),
            }
            if tr > float(spy):
                beat_spy.append(row)
            if qqq is not None and tr > float(qqq):
                beat_qqq.append({**row, "excess_qqq": tr - float(qqq)})
        beat_spy.sort(key=lambda x: -x["ret"])
        beat_qqq.sort(key=lambda x: -x["ret"])
        top = beat_spy[0] if beat_spy else None
        report["years"][y] = {
            "spy": float(spy),
            "qqq": float(qqq) if qqq is not None else None,
            "n_with_year": len(all_rets),
            "n_beat_spy": len(beat_spy),
            "pct_beat_spy": len(beat_spy) / max(len(all_rets), 1),
            "n_beat_qqq": len(beat_qqq),
            "pct_beat_qqq": len(beat_qqq) / max(len(all_rets), 1) if qqq is not None else None,
            "median_strat": float(np.median(all_rets)) if all_rets else None,
            "kind_counts_beat_spy": dict(Counter(b["kind"] for b in beat_spy).most_common(12)),
            "und_counts_beat_spy": dict(Counter(b["und"] for b in beat_spy).most_common(12)),
            "top15_beat_spy": beat_spy[:15],
            "top10_beat_qqq": beat_qqq[:10],
        }
        md.append(
            f"| {y} | {pct(spy)} | {pct(qqq)} | {len(all_rets)} | {len(beat_spy)} | "
            f"{100 * len(beat_spy) / max(len(all_rets), 1):.1f}% | {pct(np.median(all_rets) if all_rets else None)} | "
            f"{(str(top['id'])[:36] if top else '—')} | "
            f"{pct(top['ret']) if top else '—'} | {top['kind'] if top else '—'} |"
        )

    md += [
        "",
        "## Detalle: top 10 que batieron SPY cada año",
        "",
    ]
    for y in years:
        block = report["years"].get(y)
        if not block:
            continue
        md += [
            f"### {y} — SPY {pct(block['spy'])} · QQQ {pct(block.get('qqq'))} · "
            f"**{block['n_beat_spy']}/{block['n_with_year']}** batieron SPY "
            f"({100 * block['pct_beat_spy']:.1f}%)",
            "",
            f"Kinds que más batieron SPY: `{block['kind_counts_beat_spy']}`",
            f"Underlyings: `{block['und_counts_beat_spy']}`",
            "",
            "| # | Kind | Und | Ret | vs SPY | Mean L | ID |",
            "|---|------|-----|-----|--------|--------|----|",
        ]
        for i, b in enumerate(block["top15_beat_spy"][:10], 1):
            md.append(
                f"| {i} | {b['kind']} | {b['und']} | {pct(b['ret'])} | "
                f"{pct(b['excess_spy'])} | {float(b.get('mean_L') or 0):.2f} | "
                f"`{str(b['id'])[:42]}` |"
            )
        md.append("")
        if block.get("top10_beat_qqq"):
            md += [
                f"**También > QQQ ({pct(block.get('qqq'))}):** "
                f"{block['n_beat_qqq']} sleeves ({100 * (block.get('pct_beat_qqq') or 0):.1f}%)",
                "",
            ]

    md += [
        "## Lectura rápida",
        "",
        "- **SPY** es el benchmark principal (mercado amplio).",
        "- **QQQ** es más exigente (growth); menos sleeves lo baten.",
        "- Muchos tops son **QQQ trend / vol-target / defensive SMA** con L media ~1.1–1.3 (no 3×).",
        "- Un % alto de sleeves “> SPY” en un año no implica edge estable: hay que mirar consistencia multi-año.",
        "",
        "---",
        "Fuente: `sleeve_results.json` + `year_spy`. VIRTUAL. No es consejo financiero.",
        "",
    ]

    out_json = latest / "year_by_year_beat_index.json"
    out_md = latest / "YEAR_BY_YEAR_BEAT_INDEX.md"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:80]))
    print(f"\n... full report → {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
