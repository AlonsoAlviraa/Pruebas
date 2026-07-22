#!/usr/bin/env python3
"""Focus report: innovative options TA results + math + blind spots vs Twitter/GitHub."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "OPTIONS_FOCUS_BLINDSPOT.md"


def load(p: str) -> Optional[Dict[str, Any]]:
    path = ROOT / p
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: Optional[float], signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:+.2f}%" if signed else f"{100 * x:.2f}%"


def table(strats: List[Dict[str, Any]]) -> List[str]:
    rows = sorted(strats, key=lambda s: -(s.get("total_return") or 0))
    lines = [
        "| Rank | ID | Kind | Return | MaxDD | CVaR5% | Rolls | vs SPY | Kill |",
        "|------|-----|------|-------:|------:|-------:|------:|-------:|:----:|",
    ]
    for i, r in enumerate(rows, 1):
        cvar = r.get("cvar_5pct")
        vs = r.get("vs_spy_bh")
        lines.append(
            f"| {i} | `{r.get('strategy_id')}` | {r.get('kind')} | {pct(r.get('total_return'), True)} | "
            f"{pct(r.get('max_dd'))} | {pct(cvar) if cvar is not None else 'n/a'} | "
            f"{r.get('n_rolls', 0)} | {pct(vs, True) if vs is not None else 'n/a'} | "
            f"{'YES' if r.get('hard_kill') else 'no'} |"
        )
    return lines


def main() -> int:
    ta = load("reports/paper_options_ta_smoke/latest/summary.json")
    base = load("reports/paper_options/latest/summary.json")
    mega = load("reports/paper_options_mega/latest/summary.json")

    lines: List[str] = []
    lines.append("# Opciones — estrategias innovadoras (TA/volumen) + punto ciego")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat()} · VIRTUAL · marks `proxy_bs`_"
    )
    lines.append("")
    lines.append("## 1. Qué son las “últimas innovadoras”")
    lines.append("")
    lines.append(
        "No inventan un payoff nuevo de la nada: **combinan estructuras clásicas "
        "(CSP, PCS, IC, CC, protective put) con gates de análisis técnico y volumen causales** "
        "antes de abrir riesgo. Implementación: `paper_live/options/ta_gates.py` + "
        "`zoo_options_ta.json`."
    )
    lines.append("")
    lines.append("| ID | Estructura | Gate innovador (causal) | Tesis |")
    lines.append("|----|------------|-------------------------|-------|")
    lines.append(
        "| OPT_TA01 | CSP | low ATR + cerca SMA50 (range) | Vender put solo en rango / IV-rich proxy |"
    )
    lines.append(
        "| OPT_TA02 | PCS | volume spike → compression | Tras actividad, secar vol y vender defined-risk |"
    )
    lines.append(
        "| OPT_TA03 | Iron condor | ATR percentil bajo | CNDR-like: short vol solo en ATR comprimido |"
    )
    lines.append(
        "| OPT_TA04 | Covered call | uptrend + volume confirm | CC solo con tendencia y participación |"
    )
    lines.append(
        "| OPT_TA05 | Protective put | RSI overbought + volume climax | Seguro cuando “todo el mundo empuja” |"
    )
    lines.append(
        "| OPT_TA06 | PCS | pullback uptrend + dry-up volume | Comprar dip de vol, vender put credit |"
    )
    lines.append(
        "| OPT_TA07 | CSP | HV gate + range | Solo si HV reciente “rico” vs mediana |"
    )
    lines.append(
        "| OPT_TA08 | Call credit | RSI overbought | Bear call cuando estirado |"
    )
    lines.append(
        "| OPT_TA09 | CC QQQ | uptrend + volume | Mismo que TA04 en Nasdaq |"
    )
    lines.append(
        "| OPT_TA10 | CSP | SMA200 + dry-up | Put write solo en tendencia larga quieta |"
    )
    lines.append(
        "| OPT_TA11 | Protective put | RSI only | Seguro “barato” de lógica (pocas aperturas) |"
    )
    lines.append("| OPT_TA12 | Cash | — | Control |")
    lines.append("")

    if ta:
        w = ta.get("window") or {}
        b = ta.get("benchmarks") or {}
        lines.append("## 2. Resultados opciones TA (innovadoras)")
        lines.append("")
        lines.append(f"- **Window:** {w.get('start')} → {w.get('end')}")
        lines.append(f"- **SPY B&H:** {pct(b.get('spy_bh'))} · **QQQ B&H:** {pct(b.get('qqq_bh'))}")
        lines.append(f"- **Data:** `{ta.get('data_label', 'proxy_bs')}` — **no** fills de cadena real")
        lines.append(f"- **N:** {len(ta.get('strategies') or [])}")
        pos = sum(1 for s in ta["strategies"] if (s.get("total_return") or 0) > 0)
        beat = sum(1 for s in ta["strategies"] if (s.get("vs_spy_bh") or -1) > 0)
        lines.append(f"- **Positivas / baten SPY / kills:** {pos} / {beat} / "
                     f"{sum(1 for s in ta['strategies'] if s.get('hard_kill'))}")
        lines.append("")
        lines.extend(table(ta["strategies"]))
        lines.append("")
        lines.append("### Lectura de este pack")
        lines.append("")
        lines.append(
            "1. **Ganan los que mantienen beta de equity** (CC con uptrend+vol ~+8.2%, "
            "CC QQQ ~+7.4%) — se acercan a SPY pero **no lo baten**."
        )
        lines.append(
            "2. **Short premium filtrado** (CSP/PCS/IC) gana **poco** (+0.2%…+1%) con CVaR pequeño: "
            "el gate reduce trades; en bull el “edge” de VRP no compensa el upside perdido."
        )
        lines.append(
            "3. **Protective put por climax** (OPT_TA05) es el peor (~−0.8%): pagas seguro cuando "
            "el mercado sigue subiendo (típico en rally)."
        )
        lines.append(
            "4. Los logs de smoke muestran skips tipo `atr_not_low`, `volume_not_elevated` — "
            "los gates **sí filtran**, no son decorativos."
        )
        lines.append("")

    if base:
        lines.append("## 3. Comparación: zoo opciones **sin** TA (base)")
        lines.append("")
        b = base.get("benchmarks") or {}
        lines.append(f"SPY B&H {pct(b.get('spy_bh'))} · misma filosofía de marks.")
        lines.append("")
        lines.extend(table(base.get("strategies") or []))
        lines.append("")
        lines.append(
            "**Delta innovador:** OPT_TA04 (+8.21%) > OPT01 CC ciego (+7.14%) en este tramo — "
            "el filtro uptrend+volumen **ayudó un poco** al buywrite. CSP ciego (+1.42%) > "
            "CSP range-gated (+0.23%): en bull, **filtrar de más** deja prima sobre la mesa."
        )
        lines.append("")

    if mega:
        lines.append("## 4. Contexto “opciones de acciones” (mega 56)")
        lines.append("")
        rows = sorted(mega.get("strategies") or [], key=lambda s: -(s.get("total_return") or 0))
        lines.append(
            f"SPY {pct((mega.get('benchmarks') or {}).get('spy_bh'))} · "
            f"QQQ {pct((mega.get('benchmarks') or {}).get('qqq_bh'))} · "
            f"best `{rows[0].get('strategy_id')}` {pct(rows[0].get('total_return'), True)} · "
            f"beat SPY count: "
            f"{sum(1 for s in rows if (s.get('vs_spy_bh') or -1) > 0)}"
        )
        lines.append("")
        lines.append("Top 8 del mega (buywrite / PP dominan):")
        lines.append("")
        lines.extend(table(rows[:8]))
        lines.append("")
        lines.append(
            "Familias: covered call / collar / protective put ≈ **beta de acciones + overlay**; "
            "CSP/PCS/IC ≈ **vender prima** (ingresos modestos en bull)."
        )
        lines.append("")

    lines.append("## 5. Matemáticas: cómo se piensa el edge (papers / X / repos)")
    lines.append("")
    lines.append("### 5.1 Black–Scholes y el “balance” (lo que modelamos)")
    lines.append("")
    lines.append(
        "Idea central (PDE / BS): **theta ≈ −½ S² σ² Γ** (en el mundo BS sin tasas/div). "
        "Vender opciones cobra **theta**; el riesgo es **gamma × movimiento realizado**. "
        "Si la **volatilidad implícita σ_imp** que usas para cobrar > **volatilidad realizada σ_real**, "
        "en media ganas el **volatility risk premium (VRP)**."
    )
    lines.append("")
    lines.append("```")
    lines.append("VRP ≈ E[σ_imp − σ_real]  (o E[PnL short option under RV path])")
    lines.append("Expected move ~1m (rule of thumb) ≈ SPX × (VIX/100) / √12")
    lines.append("Delta ≈ ∂V/∂S   (también se usa como proxy de P(ITM))")
    lines.append("Lambda = Δ × (S / Premium)  → sensibilidad % del premium al % del subyacente")
    lines.append("```")
    lines.append("")
    lines.append("### 5.2 Qué dice Twitter (práctica)")
    lines.append("")
    lines.append(
        "- **Greeks:** delta/theta/vega/gamma como lenguaje de riesgo "
        "(vender ~0.20Δ puts ≈ ~80% “prob” naive)."
    )
    lines.append(
        "- **Beta-weighted delta** del libro entero + hedge con /ES /MES "
        "(no solo delta de un strike)."
    )
    lines.append(
        "- **Iron condor:** no es “seguro”; es **posicionamiento** de strikes; "
        "drift alcista castiga el lado call simétrico → muchos prefieren **PCS** o call wing más ancha."
    )
    lines.append(
        "- **CNDR / realized vs implied:** el condor muere cuando el índice se mueve **más** "
        "de lo que la IV preció (chop / RV↑), no solo cuando sube el VIX."
    )
    lines.append("")
    lines.append("### 5.3 Qué hacen repos GitHub (estilo vectorbt/backtest)")
    lines.append("")
    lines.append(
        "- Grids de parámetros (DTE, OTM%, wings) sobre **pocas** estructuras."
    )
    lines.append(
        "- A menudo BS o chains históricas; **pocos** modelan: bid/ask, assignment, "
        "early exercise, margin real, GEX/dealers."
    )
    lines.append(
        "- El “punto ciego” típico del open-source: **misma ilusión de VRP** si IV se inventa "
        "como HV×1.15 (exactamente nuestro proxy)."
    )
    lines.append("")

    lines.append("## 6. EL PUNTO CIEGO (nuestro stack vs realidad)")
    lines.append("")
    lines.append("| # | Punto ciego | Por qué duele | Qué haría un pro / buen repo |")
    lines.append("|---|-------------|---------------|------------------------------|")
    lines.append(
        "| **1** | **IV inventada = HV × mult** | No existe surface real; el “edge VRP” "
        "está **metido a mano** en el mark | Cadena histórica (ORATS/OPRA) o al menos "
        "IV rank / VIX term structure |"
    )
    lines.append(
        "| **2** | **Sin bid/ask ni mid** | Premios de entrada irreales; short premium "
        "sobrestimado | Slippage en mid, o fills al bid al vender |"
    )
    lines.append(
        "| **3** | **Sin vega shock / skew** | Crash de put skew no se ve; collars/PP mal valorados | "
        "Skew sticky / jump vol |"
    )
    lines.append(
        "| **4** | **Sin gamma intradía / 0DTE** | No hay dealer gamma, pin, expected move del día | "
        "GEX / 0DTE models (complejo) |"
    )
    lines.append(
        "| **5** | **Assignment / early exercise** | CC y puts no simulan asignación real | "
        "Reglas de exercise americano (equity options) |"
    )
    lines.append(
        "| **6** | **TA gates ≠ vol surface gates** | Filtramos ATR/volumen del **subyacente**, "
        "no “IV rica vs RV” de verdad | Gate: IV/HV, VRP percentile, term structure |"
    )
    lines.append(
        "| **7** | **Un solo régimen (bull 9m)** | Short vol “gana” de mentira por no haber crash | "
        "2020/2022 + stress + multi-year obligatorio |"
    )
    lines.append(
        "| **8** | **Beta-weighted book risk** | Cada estrategia en silo $100k; no hay cartera "
        "multi-estrategia | Agregar delta/vega de libro |"
    )
    lines.append(
        "| **9** | **Equity options vs index** | Single-name (AAPL…) no están en el zoo opciones; "
        "solo SPY/QQQ | Extender underlying a mega-caps con volume gates |"
    )
    lines.append(
        "| **10** | **Management rules de Twitter** | No hay “close at 50% profit / 2× loss / one roll max” | "
        "Reglas de take-profit de premium sellers |"
    )
    lines.append("")
    lines.append("### Punto ciego #1 en una frase")
    lines.append("")
    lines.append(
        "> **Estamos midiendo “estrategias de opciones” con un simulador de acciones + BS, "
        "donde el premium que cobramos es una función de la HV que nosotros mismos inflamos. "
        "Eso valida *gates y risk*, no un edge de mercado de opciones.**"
    )
    lines.append("")
    lines.append("## 7. Qué SÍ validan estos resultados (útil)")
    lines.append("")
    lines.append(
        "- Orden relativo en **bull**: buywrite > short premium filtrado > cash."
    )
    lines.append(
        "- Gates TA **cambian el número de rolls** y a veces mejoran un poco el CC "
        "(TA04 vs OPT01)."
    )
    lines.append(
        "- Risk engine (DD/kill/margin) se comporta de forma creíble bajo **stress sintético**."
    )
    lines.append(
        "- Diseño causal de features (sin look-ahead en ATR/volumen/RSI)."
    )
    lines.append("")
    lines.append("## 8. Siguiente loop (para cerrar el punto ciego)")
    lines.append("")
    lines.append("| Prioridad | Acción |")
    lines.append("|-----------|--------|")
    lines.append("| P0 | IV real o al menos VIX-based surface proxy (no HV×1.15 fijo) |")
    lines.append("| P0 | Multi-ventana 2022 bear + 2023 + stress en **cada** OPT_TA |")
    lines.append("| P1 | Management: 50% credit take-profit, 2× stop, max 1 roll |")
    lines.append("| P1 | Single-name options sleeve (AAPL/NVDA) con volume gates equity |")
    lines.append("| P2 | Bid/ask haircut; assignment model |")
    lines.append("| P2 | Book-level beta-weighted delta report |")
    lines.append("")
    lines.append("## 9. Artefactos")
    lines.append("")
    lines.append("| Pack | Path |")
    lines.append("|------|------|")
    lines.append("| Opciones TA innovadoras | `reports/paper_options_ta_smoke/latest/` |")
    lines.append("| Opciones base | `reports/paper_options/latest/` |")
    lines.append("| Mega 56 | `reports/paper_options_mega/` |")
    lines.append("| Zoo TA | `paper_live/cloud/zoo_options_ta.json` |")
    lines.append("| Gates | `paper_live/options/ta_gates.py` |")
    lines.append("| Math BS | `paper_live/options/bs.py` |")
    lines.append("")
    lines.append("_Not financial advice. Research paper only._")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
