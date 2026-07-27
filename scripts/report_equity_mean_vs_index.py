#!/usr/bin/env python3
"""Mean/median of ALL equity mega strategies vs SPY & QQQ (period + year-by-year)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def p(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and x != x):
            return "n/a"
        return f"{100.0 * float(x):+.1f}%"
    except Exception:
        return "n/a"


def pp(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and x != x):
            return "n/a"
        return f"{100.0 * float(x):+.1f} pp"
    except Exception:
        return "n/a"


def main() -> int:
    latest = ROOT / "reports" / "equity_mega_lever" / "latest"
    s = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((latest / "sleeve_results.json").read_text(encoding="utf-8"))
    valid = [r for r in rows if r.get("year_returns") and not r.get("error")]

    year_spy: Dict[str, float] = {
        k: float(v) for k, v in (s.get("year_spy") or {}).items() if v is not None
    }
    year_qqq: Dict[str, Optional[float]] = {}

    try:
        from paper_live.data.eodhd_client import build_eodhd_feed

        cache = ROOT / "reports" / "equity_mega_lever" / "eodhd_cache"
        feed, _ = build_eodhd_feed(
            ["SPY", "QQQ"], start="2010-01-01", cache_dir=cache, min_history=40
        )
        days = list(feed.days)

        def bh(ticker: str, y: int) -> Optional[float]:
            ys, ye = date(y, 1, 2), date(y, 12, 31)
            sd = next((d for d in days if d >= ys), None)
            ed = next((d for d in reversed(days) if d <= ye), None)
            if sd is None or ed is None:
                return None
            b0, b1 = feed.bar(ticker, sd), feed.bar(ticker, ed)
            if not b0 or not b1 or float(b0.close) <= 0:
                return None
            return float(b1.close) / float(b0.close) - 1.0

        for y in year_spy:
            year_qqq[y] = bh("QQQ", int(y))
    except Exception as e:
        print("QQQ years skip:", e, file=sys.stderr)

    by_year: Dict[str, List[float]] = defaultdict(list)
    total_rets: List[float] = []
    for r in valid:
        total_rets.append(float(r.get("total_return") or 0.0))
        for y, v in (r.get("year_returns") or {}).items():
            by_year[y].append(float(v))

    years = sorted(year_spy.keys())
    spy_t = float(s.get("spy_bh") or 0.0)
    qqq_t = float(s.get("qqq_bh") or 0.0)
    mean_t = float(np.mean(total_rets)) if total_rets else 0.0
    med_t = float(np.median(total_rets)) if total_rets else 0.0

    lines: List[str] = [
        "# Comparación por media: TODAS las estrategias vs índices",
        "",
        f"**Universo:** {len(valid)} sleeves equity (mega lever study)",
        f"**Periodo total (BH summary):** ver SPY/QQQ full-window del study",
        "",
        "## 1. Resultado total del periodo (todas a la vez)",
        "",
        "| Métrica | Retorno total del periodo |",
        "|---------|---------------------------|",
        f"| **SPY buy & hold** | **{p(spy_t)}** |",
        f"| **QQQ buy & hold** | **{p(qqq_t)}** |",
        f"| **Media de todas las estrategias** | **{p(mean_t)}** |",
        f"| Mediana de todas las estrategias | {p(med_t)} |",
        f"| Percentil 10 estrategias | {p(float(np.percentile(total_rets, 10)))} |",
        f"| Percentil 90 estrategias | {p(float(np.percentile(total_rets, 90)))} |",
        f"| Media − SPY | {pp(mean_t - spy_t)} |",
        f"| Mediana − SPY | {pp(med_t - spy_t)} |",
        f"| Media − QQQ | {pp(mean_t - qqq_t)} |",
        "",
        "## 2. Cada año: media (y mediana) de TODAS las estrategias vs SPY y QQQ",
        "",
        "| Año | Media strats | Mediana strats | SPY | QQQ | Media−SPY | Med−SPY | Media−QQQ | % strats > SPY | % > QQQ |",
        "|-----|--------------|----------------|-----|-----|-----------|---------|-----------|----------------|---------|",
    ]

    ann_means: List[float] = []
    ann_spy: List[float] = []
    ann_qqq: List[float] = []

    for y in years:
        arr = np.asarray(by_year.get(y) or [], dtype=float)
        if len(arr) == 0:
            continue
        spy = year_spy.get(y)
        qqq = year_qqq.get(y)
        m = float(np.mean(arr))
        med = float(np.median(arr))
        ann_means.append(m)
        if spy is not None:
            ann_spy.append(float(spy))
        if qqq is not None:
            ann_qqq.append(float(qqq))
        pct_spy = float(np.mean(arr > float(spy))) if spy is not None else float("nan")
        pct_qqq = float(np.mean(arr > float(qqq))) if qqq is not None else float("nan")
        lines.append(
            f"| {y} | {p(m)} | {p(med)} | {p(spy)} | {p(qqq)} | "
            f"{pp(m - float(spy)) if spy is not None else 'n/a'} | "
            f"{pp(med - float(spy)) if spy is not None else 'n/a'} | "
            f"{pp(m - float(qqq)) if qqq is not None else 'n/a'} | "
            f"{100 * pct_spy:.1f}% | {100 * pct_qqq:.1f}% |"
        )

    mean_ann = float(np.mean(ann_means)) if ann_means else 0.0
    mean_spy_a = float(np.mean(ann_spy)) if ann_spy else 0.0
    mean_qqq_a = float(np.mean(ann_qqq)) if ann_qqq else 0.0

    # compound equal-weight zoo each year
    eq = 1.0
    for r in ann_means:
        eq *= 1.0 + r

    lines += [
        "",
        "## 3. Media de las medias anuales (cada año pesa igual)",
        "",
        "| | Media anual |",
        "|--|-------------|",
        f"| **Media del zoo (promedio de medias anuales)** | **{p(mean_ann)}** |",
        f"| Media anual SPY | {p(mean_spy_a)} |",
        f"| Media anual QQQ | {p(mean_qqq_a)} |",
        f"| Gap zoo vs SPY | {pp(mean_ann - mean_spy_a)} |",
        f"| Gap zoo vs QQQ | {pp(mean_ann - mean_qqq_a)} |",
        "",
        "## 4. Si hubieras invertido cada año en el *promedio* del zoo",
        "",
        f"| Compound de la media anual del zoo | **{p(eq - 1.0)}** |",
        f"| SPY total periodo | {p(spy_t)} |",
        f"| QQQ total periodo | {p(qqq_t)} |",
        "",
        "_Interpretación: cartera conceptual equal-weight de **todas** las reglas, "
        "rebalanceada cada año al promedio. No es un solo sleeve óptimo._",
        "",
        "## 5. Media total por **tipo** de estrategia (kind) vs SPY del periodo",
        "",
        "| Kind | N | Media total | Mediana | vs SPY (media) |",
        "|------|---|-------------|---------|----------------|",
    ]

    kind_stats = []
    for k in sorted({str(r.get("kind")) for r in valid}):
        rets = [
            float(r.get("total_return") or 0.0)
            for r in valid
            if str(r.get("kind")) == k
        ]
        if not rets:
            continue
        kind_stats.append(
            (k, len(rets), float(np.mean(rets)), float(np.median(rets)))
        )
    kind_stats.sort(key=lambda x: -x[2])
    for k, n, m, med in kind_stats:
        lines.append(
            f"| {k} | {n} | {p(m)} | {p(med)} | {pp(m - spy_t)} |"
        )

    lines += [
        "",
        "## Lectura en una frase",
        "",
        f"La **media de las ~{len(valid)} estrategias** del zoo "
        f"({p(mean_t)} total / ~{p(mean_ann)} anual media) "
        f"**queda por debajo de SPY** ({p(spy_t)} total / ~{p(mean_spy_a)} anual) "
        f"y **muy por debajo de QQQ** ({p(qqq_t)}).",
        "",
        "---",
        "Fuente: `reports/equity_mega_lever/latest/sleeve_results.json`. VIRTUAL.",
        "",
    ]

    out = latest / "MEAN_ALL_VS_INDEX.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "n": len(valid),
        "period": {
            "spy": spy_t,
            "qqq": qqq_t,
            "strat_mean": mean_t,
            "strat_median": med_t,
            "strat_mean_minus_spy": mean_t - spy_t,
            "strat_mean_minus_qqq": mean_t - qqq_t,
        },
        "annual_equal_weight_years": {
            "zoo_mean_of_annual_means": mean_ann,
            "spy_mean_annual": mean_spy_a,
            "qqq_mean_annual": mean_qqq_a,
            "compound_zoo_annual_means": eq - 1.0,
        },
        "by_kind": [
            {"kind": k, "n": n, "mean": m, "median": med, "vs_spy": m - spy_t}
            for k, n, m, med in kind_stats
        ],
    }
    (latest / "mean_all_vs_index.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
