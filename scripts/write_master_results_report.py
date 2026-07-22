#!/usr/bin/env python3
"""Consolidate all paper result packs into one master markdown report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "MASTER_ALL_RESULTS.md"

PACKS = [
    ("equity_cloud_2026", "Equity cloud GitHub (solo 2026)", "reports/paper_cloud/latest/summary.json"),
    ("equity_ab", "Equity A/B signal modes", "reports/paper_cloud_ab/latest/summary.json"),
    ("equity_ta", "Equity TA/volume (Yahoo real)", "reports/paper_cloud_ta/latest/summary.json"),
    ("equity_ta_synth", "Equity TA/volume (synthetic smoke)", "reports/paper_cloud_ta_smoke/latest/summary.json"),
    ("opt_base", "Options base zoo", "reports/paper_options/latest/summary.json"),
    ("opt_mega", "Options mega ~50–56", "reports/paper_options_mega/latest/summary.json"),
    ("opt_ta", "Options TA-gated", "reports/paper_options_ta_smoke/latest/summary.json"),
]


def _load(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt_pct(x: Optional[float], pp: bool = False) -> str:
    if x is None:
        return "n/a"
    if pp:
        return f"{100 * x:+.2f}pp"
    return f"{100 * x:.2f}%"


def _table(strats: List[Dict[str, Any]], limit: int = 99) -> List[str]:
    rows = sorted(strats, key=lambda s: -(s.get("total_return") or 0))
    lines = [
        "| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |",
        "|------|-----|-----------|-------:|-------:|---------|:----:|",
    ]
    for i, r in enumerate(rows[:limit], 1):
        mid = r.get("signal_mode") or r.get("kind") or ""
        vs = r.get("vs_spy") if r.get("vs_spy") is not None else r.get("vs_spy_bh")
        pf = r.get("profit_factor")
        cvar = r.get("cvar_5pct")
        extra = ""
        if pf is not None:
            extra = f"PF {pf:.2f}"
        elif cvar is not None:
            extra = f"CVaR {_fmt_pct(cvar)}"
        else:
            extra = "—"
        kill = "YES" if r.get("hard_kill") else "no"
        lines.append(
            f"| {i} | `{r.get('strategy_id')}` | {mid} | {_fmt_pct(r.get('total_return'))} | "
            f"{_fmt_pct(vs, pp=True)} | {extra} | {kill} |"
        )
    return lines


def main() -> int:
    lines: List[str] = []
    lines.append("# MASTER — todos los resultados paper (equity + opciones + TA)")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat()} · capital VIRTUAL · no es consejo financiero_"
    )
    lines.append("")
    lines.append("## 0. Mapa de packs")
    lines.append("")
    lines.append("| Pack | Descripción | Path |")
    lines.append("|------|-------------|------|")
    for key, title, rel in PACKS:
        p = ROOT / rel
        status = "OK" if p.is_file() else "MISSING"
        lines.append(f"| `{key}` | {title} | `{rel}` ({status}) |")
    lines.append("")
    lines.append("### Otros docs de auditoría")
    lines.append("")
    lines.append("- `reports/paper_cloud/audits/LATEST_loss_audit.md` — por qué el zoo viejo iba en rojo")
    lines.append("- `reports/paper_cloud_ab/audits/LOOP_AUD_AB_RESULTS.md` — A/B AUD-A/B")
    lines.append("- `reports/paper_options_mega/MEGA_RESULTS.md` — mega 56 opciones")
    lines.append("- `docs/design/2026-07-22_*.md` — diseños AUD, opciones, mega, TA")
    lines.append("")
    lines.append("## 1. Qué se construyó (stack completo)")
    lines.append("")
    lines.append("### Infra paper cloud (acciones)")
    lines.append("- GitHub Actions diario (lun–vie), datos Yahoo reales, anti-synthetic gate")
    lines.append("- Kill switch ajustado (sin false positives en sample corto)")
    lines.append("- Ventana configurable `--start`/`--end`")
    lines.append("- Instrumentación: closed_trades, exit_reason, SPY/eq-weight BH, WR, PF")
    lines.append("")
    lines.append("### Señales equity")
    lines.append("- Legacy: trend_mom, no_extension, pullback, topk, qqq_gate, qqq_hold")
    lines.append("- **TA/volumen:** vol_confirm, rsi_mr, vol_dryup, vol_expand, rvol_trend, vol_pullback, combined_ta_v1")
    lines.append("")
    lines.append("### Opciones (proxy_bs)")
    lines.append("- Kinds: CC, CSP, PCS, CCS, iron_condor, collar, protective_put, cash")
    lines.append("- Risk: DD, day-drop, margin-at-risk, hard kill, CVaR, multi-window, stress −30%")
    lines.append("- Yahoo chain “hoy” (label real vs failed)")
    lines.append("- Gates TA: uptrend, volume, RSI, ATR/range, climax, compression")
    lines.append("- Mega zoo ~56 estrategias (CBOE/VRP/X/GitHub grid)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for key, title, rel in PACKS:
        data = _load(ROOT / rel)
        lines.append(f"## Pack: {title}")
        lines.append("")
        if not data:
            lines.append(f"_Missing `{rel}`_")
            lines.append("")
            continue
        w = data.get("window") or {}
        b = data.get("benchmarks") or {}
        src = data.get("data_sources") or {}
        strats = data.get("strategies") or []
        lines.append(f"- **Window:** {w.get('start')} → {w.get('end')}")
        lines.append(f"- **N strategies:** {len(strats)}")
        if b.get("spy_bh") is not None:
            lines.append(f"- **SPY B&H:** {_fmt_pct(b.get('spy_bh'))}")
        if b.get("qqq_bh") is not None:
            lines.append(f"- **QQQ B&H:** {_fmt_pct(b.get('qqq_bh'))}")
        if b.get("eq_weight_bh") is not None:
            lines.append(f"- **Eq-weight BH:** {_fmt_pct(b.get('eq_weight_bh'))}")
        if data.get("data_label"):
            lines.append(f"- **Data label:** `{data.get('data_label')}`")
        if src:
            real = sum(
                1
                for v in src.values()
                if not str(v).startswith("synthetic") and v != "missing"
            )
            lines.append(f"- **Sources real/total:** {real}/{len(src)}")
        pos = sum(1 for s in strats if (s.get("total_return") or 0) > 0)
        kills = sum(1 for s in strats if s.get("hard_kill"))
        beat = sum(
            1
            for s in strats
            if (s.get("vs_spy") if s.get("vs_spy") is not None else s.get("vs_spy_bh") or -1) > 0
        )
        lines.append(f"- **Positive / kill / beat SPY:** {pos} / {kills} / {beat}")
        lines.append("")
        lines.extend(_table(strats))
        lines.append("")

    # Stress snippet from options if present
    stress = ROOT / "reports" / "paper_options" / "latest" / "stress.json"
    if not stress.is_file():
        stress = ROOT / "reports" / "paper_options_mega" / "latest" / "stress.json"
    # also check base summary markdown for stress table - load stress from opt summary path
    opt_summary_md = ROOT / "reports" / "paper_options" / "latest" / "SUMMARY.md"
    if opt_summary_md.is_file() and "stress" in opt_summary_md.read_text(encoding="utf-8").lower():
        lines.append("## Stress crash sintético (opciones base pack)")
        lines.append("")
        lines.append(
            "Ver tabla completa en `reports/paper_options/latest/SUMMARY.md` sección Synthetic crash stress."
        )
        lines.append("")
        lines.append(
            "Resumen: cash 0%; collar ~−2.6%; PCS ~−6%; CSP/CC hard kill ~−15%…−18% con shock −30%."
        )
        lines.append("")

    lines.append("## 2. Conclusiones cruzadas (honestas)")
    lines.append("")
    lines.append(
        "1. **Mercado alcista (SPY ~+9%, QQQ ~+11.5% en 2025-10→2026-07):** "
        "cualquier long-stock/buywrite gana; short premium puro gana poco y **no bate SPY**."
    )
    lines.append(
        "2. **Zoo equity viejo (2026 YTD):** 10/10 en rojo (−1.6%…−5.9%) con Yahoo real — "
        "comprar momentum/extensión + stops."
    )
    lines.append(
        "3. **A/B equity:** QQQ hold ~+11% (control índice); no_extension PF>1 y leve verde; "
        "pullback +1.2%; baseline ~−0.7%."
    )
    lines.append(
        "4. **TA/volume equity (Yahoo real):** ver pack `equity_ta` — filtros de volumen mejoran "
        "estructura de trades vs legacy en paper; no garantiza edge vs SPY."
    )
    lines.append(
        "5. **Opciones mega 56:** best ~+8.5% protective put QQQ / CC ~+7.6% — todos ≤ SPY/QQQ B&H."
    )
    lines.append(
        "6. **Opciones TA-gated:** covered call con uptrend+vol ~+8.2% (mejor que CC ciego en el base pack); "
        "short premium filtrado más bajo y selectivo."
    )
    lines.append(
        "7. **proxy_bs ≠ OPRA.** Paper only. No dinero real. Multi-ventana 2022–24 incompleta si no hay tape denso."
    )
    lines.append("")
    lines.append("## 3. Cómo regenerar todo")
    lines.append("")
    lines.append("```powershell")
    lines.append("# Equity A/B")
    lines.append(
        "python scripts/run_paper_cloud_batch.py --zoo paper_live/cloud/strategy_zoo_ab.json "
        "--out reports/paper_cloud_ab --start 2025-10-29"
    )
    lines.append("# Equity TA")
    lines.append(
        "python scripts/run_paper_cloud_batch.py --zoo paper_live/cloud/strategy_zoo_ta.json "
        "--out reports/paper_cloud_ta --start 2025-10-29"
    )
    lines.append("# Options mega")
    lines.append("python scripts/build_options_zoo_50.py")
    lines.append(
        "python scripts/run_paper_options_batch.py --zoo paper_live/cloud/zoo_options_50.json "
        "--out reports/paper_options_mega --start 2025-10-29"
    )
    lines.append("# Options TA")
    lines.append(
        "python scripts/run_paper_options_batch.py --zoo paper_live/cloud/zoo_options_ta.json "
        "--out reports/paper_options_ta --start 2025-10-29"
    )
    lines.append("python scripts/write_master_results_report.py")
    lines.append("```")
    lines.append("")
    lines.append("_Research software. Past paper ≠ future results._")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
