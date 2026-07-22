# Opciones — estrategias innovadoras (TA/volumen) + punto ciego

_Generated 2026-07-22T11:19:02.354814+00:00 · VIRTUAL · marks `proxy_bs`_

## 1. Qué son las “últimas innovadoras”

No inventan un payoff nuevo de la nada: **combinan estructuras clásicas (CSP, PCS, IC, CC, protective put) con gates de análisis técnico y volumen causales** antes de abrir riesgo. Implementación: `paper_live/options/ta_gates.py` + `zoo_options_ta.json`.

| ID | Estructura | Gate innovador (causal) | Tesis |
|----|------------|-------------------------|-------|
| OPT_TA01 | CSP | low ATR + cerca SMA50 (range) | Vender put solo en rango / IV-rich proxy |
| OPT_TA02 | PCS | volume spike → compression | Tras actividad, secar vol y vender defined-risk |
| OPT_TA03 | Iron condor | ATR percentil bajo | CNDR-like: short vol solo en ATR comprimido |
| OPT_TA04 | Covered call | uptrend + volume confirm | CC solo con tendencia y participación |
| OPT_TA05 | Protective put | RSI overbought + volume climax | Seguro cuando “todo el mundo empuja” |
| OPT_TA06 | PCS | pullback uptrend + dry-up volume | Comprar dip de vol, vender put credit |
| OPT_TA07 | CSP | HV gate + range | Solo si HV reciente “rico” vs mediana |
| OPT_TA08 | Call credit | RSI overbought | Bear call cuando estirado |
| OPT_TA09 | CC QQQ | uptrend + volume | Mismo que TA04 en Nasdaq |
| OPT_TA10 | CSP | SMA200 + dry-up | Put write solo en tendencia larga quieta |
| OPT_TA11 | Protective put | RSI only | Seguro “barato” de lógica (pocas aperturas) |
| OPT_TA12 | Cash | — | Control |

## 2. Resultados opciones TA (innovadoras)

- **Window:** 2025-10-29 → 2026-07-21
- **SPY B&H:** 8.86% · **QQQ B&H:** 11.51%
- **Data:** `proxy_bs` — **no** fills de cadena real
- **N:** 12
- **Positivas / baten SPY / kills:** 10 / 0 / 0

| Rank | ID | Kind | Return | MaxDD | CVaR5% | Rolls | vs SPY | Kill |
|------|-----|------|-------:|------:|-------:|------:|-------:|:----:|
| 1 | `OPT_TA04_cc_uptrend_vol` | covered_call | +8.21% | -6.02% | -1.19% | 6 | -0.65% | no |
| 2 | `OPT_TA09_qqq_cc_trend_vol` | covered_call | +7.38% | -7.23% | -1.53% | 5 | -1.48% | no |
| 3 | `OPT_TA11_pp_rsi_only` | protective_put | +3.36% | -3.27% | -0.90% | 1 | -5.50% | no |
| 4 | `OPT_TA06_pcs_pullback_dry` | put_credit_spread | +1.04% | -0.80% | -0.19% | 7 | -7.82% | no |
| 5 | `OPT_TA10_csp_sma200_dry` | cash_secured_put | +0.96% | -0.29% | -0.09% | 6 | -7.90% | no |
| 6 | `OPT_TA07_csp_hv_range` | cash_secured_put | +0.70% | -0.16% | -0.06% | 4 | -8.15% | no |
| 7 | `OPT_TA03_ic_low_atr` | iron_condor | +0.43% | -0.18% | -0.05% | 4 | -8.43% | no |
| 8 | `OPT_TA01_csp_range` | cash_secured_put | +0.23% | -0.22% | -0.06% | 3 | -8.63% | no |
| 9 | `OPT_TA02_pcs_compress` | put_credit_spread | +0.18% | -0.14% | -0.04% | 3 | -8.68% | no |
| 10 | `OPT_TA08_ccs_overbought` | call_credit_spread | +0.13% | -0.48% | -0.13% | 3 | -8.73% | no |
| 11 | `OPT_TA12_cash` | cash | +0.00% | 0.00% | 0.00% | 0 | -8.86% | no |
| 12 | `OPT_TA05_pp_climax` | protective_put | -0.82% | -3.02% | -0.79% | 1 | -9.67% | no |

### Lectura de este pack

1. **Ganan los que mantienen beta de equity** (CC con uptrend+vol ~+8.2%, CC QQQ ~+7.4%) — se acercan a SPY pero **no lo baten**.
2. **Short premium filtrado** (CSP/PCS/IC) gana **poco** (+0.2%…+1%) con CVaR pequeño: el gate reduce trades; en bull el “edge” de VRP no compensa el upside perdido.
3. **Protective put por climax** (OPT_TA05) es el peor (~−0.8%): pagas seguro cuando el mercado sigue subiendo (típico en rally).
4. Los logs de smoke muestran skips tipo `atr_not_low`, `volume_not_elevated` — los gates **sí filtran**, no son decorativos.

## 3. Comparación: zoo opciones **sin** TA (base)

SPY B&H 8.86% · misma filosofía de marks.

| Rank | ID | Kind | Return | MaxDD | CVaR5% | Rolls | vs SPY | Kill |
|------|-----|------|-------:|------:|-------:|------:|-------:|:----:|
| 1 | `OPT01_covered_call` | covered_call | +7.14% | -5.80% | -1.20% | 12 | -1.72% | no |
| 2 | `OPT_QQQ_cc` | covered_call | +7.08% | -6.39% | -1.54% | 12 | -1.77% | no |
| 3 | `OPT04_collar` | collar | +6.71% | -5.65% | -1.15% | 12 | -2.15% | no |
| 4 | `OPT06_csp_vrp_gate` | cash_secured_put | +1.75% | -0.78% | -0.23% | 9 | -7.11% | no |
| 5 | `OPT02_csp` | cash_secured_put | +1.42% | -0.72% | -0.25% | 12 | -7.43% | no |
| 6 | `OPT03_put_credit_spread` | put_credit_spread | +1.41% | -0.72% | -0.25% | 12 | -7.44% | no |
| 7 | `OPT02b_csp_10otm` | cash_secured_put | +0.36% | -0.17% | -0.06% | 7 | -8.50% | no |
| 8 | `OPT08_cash` | cash | +0.00% | 0.00% | 0.00% | 0 | -8.86% | no |

**Delta innovador:** OPT_TA04 (+8.21%) > OPT01 CC ciego (+7.14%) en este tramo — el filtro uptrend+volumen **ayudó un poco** al buywrite. CSP ciego (+1.42%) > CSP range-gated (+0.23%): en bull, **filtrar de más** deja prima sobre la mesa.

## 4. Contexto “opciones de acciones” (mega 56)

SPY 8.86% · QQQ 11.51% · best `M_pp_QQQ` +8.53% · beat SPY count: 0

Top 8 del mega (buywrite / PP dominan):

| Rank | ID | Kind | Return | MaxDD | CVaR5% | Rolls | vs SPY | Kill |
|------|-----|------|-------:|------:|-------:|------:|-------:|:----:|
| 1 | `M_pp_QQQ` | protective_put | +8.53% | -7.42% | -1.60% | 1 | -0.33% | no |
| 2 | `M_cc_QQQ_cc_7otm` | covered_call | +7.61% | -6.94% | -1.59% | 12 | -1.24% | no |
| 3 | `M_cc_SPY_bxm_atm` | covered_call | +7.14% | -5.80% | -1.20% | 12 | -1.72% | no |
| 4 | `M_cc_SPY_cc_5otm` | covered_call | +7.14% | -5.80% | -1.20% | 12 | -1.72% | no |
| 5 | `M_cc_QQQ_bxm_atm` | covered_call | +7.08% | -6.39% | -1.54% | 12 | -1.77% | no |
| 6 | `M_cc_QQQ_cc_5otm` | covered_call | +7.08% | -6.39% | -1.54% | 12 | -1.77% | no |
| 7 | `M_cc_SPY_cc_7otm` | covered_call | +6.94% | -6.12% | -1.22% | 12 | -1.92% | no |
| 8 | `M_collar_SPY` | collar | +6.71% | -5.65% | -1.15% | 12 | -2.15% | no |

Familias: covered call / collar / protective put ≈ **beta de acciones + overlay**; CSP/PCS/IC ≈ **vender prima** (ingresos modestos en bull).

## 5. Matemáticas: cómo se piensa el edge (papers / X / repos)

### 5.1 Black–Scholes y el “balance” (lo que modelamos)

Idea central (PDE / BS): **theta ≈ −½ S² σ² Γ** (en el mundo BS sin tasas/div). Vender opciones cobra **theta**; el riesgo es **gamma × movimiento realizado**. Si la **volatilidad implícita σ_imp** que usas para cobrar > **volatilidad realizada σ_real**, en media ganas el **volatility risk premium (VRP)**.

```
VRP ≈ E[σ_imp − σ_real]  (o E[PnL short option under RV path])
Expected move ~1m (rule of thumb) ≈ SPX × (VIX/100) / √12
Delta ≈ ∂V/∂S   (también se usa como proxy de P(ITM))
Lambda = Δ × (S / Premium)  → sensibilidad % del premium al % del subyacente
```

### 5.2 Qué dice Twitter (práctica)

- **Greeks:** delta/theta/vega/gamma como lenguaje de riesgo (vender ~0.20Δ puts ≈ ~80% “prob” naive).
- **Beta-weighted delta** del libro entero + hedge con /ES /MES (no solo delta de un strike).
- **Iron condor:** no es “seguro”; es **posicionamiento** de strikes; drift alcista castiga el lado call simétrico → muchos prefieren **PCS** o call wing más ancha.
- **CNDR / realized vs implied:** el condor muere cuando el índice se mueve **más** de lo que la IV preció (chop / RV↑), no solo cuando sube el VIX.

### 5.3 Qué hacen repos GitHub (estilo vectorbt/backtest)

- Grids de parámetros (DTE, OTM%, wings) sobre **pocas** estructuras.
- A menudo BS o chains históricas; **pocos** modelan: bid/ask, assignment, early exercise, margin real, GEX/dealers.
- El “punto ciego” típico del open-source: **misma ilusión de VRP** si IV se inventa como HV×1.15 (exactamente nuestro proxy).

## 6. EL PUNTO CIEGO (nuestro stack vs realidad)

| # | Punto ciego | Por qué duele | Qué haría un pro / buen repo |
|---|-------------|---------------|------------------------------|
| **1** | **IV inventada = HV × mult** | No existe surface real; el “edge VRP” está **metido a mano** en el mark | Cadena histórica (ORATS/OPRA) o al menos IV rank / VIX term structure |
| **2** | **Sin bid/ask ni mid** | Premios de entrada irreales; short premium sobrestimado | Slippage en mid, o fills al bid al vender |
| **3** | **Sin vega shock / skew** | Crash de put skew no se ve; collars/PP mal valorados | Skew sticky / jump vol |
| **4** | **Sin gamma intradía / 0DTE** | No hay dealer gamma, pin, expected move del día | GEX / 0DTE models (complejo) |
| **5** | **Assignment / early exercise** | CC y puts no simulan asignación real | Reglas de exercise americano (equity options) |
| **6** | **TA gates ≠ vol surface gates** | Filtramos ATR/volumen del **subyacente**, no “IV rica vs RV” de verdad | Gate: IV/HV, VRP percentile, term structure |
| **7** | **Un solo régimen (bull 9m)** | Short vol “gana” de mentira por no haber crash | 2020/2022 + stress + multi-year obligatorio |
| **8** | **Beta-weighted book risk** | Cada estrategia en silo $100k; no hay cartera multi-estrategia | Agregar delta/vega de libro |
| **9** | **Equity options vs index** | Single-name (AAPL…) no están en el zoo opciones; solo SPY/QQQ | Extender underlying a mega-caps con volume gates |
| **10** | **Management rules de Twitter** | No hay “close at 50% profit / 2× loss / one roll max” | Reglas de take-profit de premium sellers |

### Punto ciego #1 en una frase

> **Estamos midiendo “estrategias de opciones” con un simulador de acciones + BS, donde el premium que cobramos es una función de la HV que nosotros mismos inflamos. Eso valida *gates y risk*, no un edge de mercado de opciones.**

## 7. Qué SÍ validan estos resultados (útil)

- Orden relativo en **bull**: buywrite > short premium filtrado > cash.
- Gates TA **cambian el número de rolls** y a veces mejoran un poco el CC (TA04 vs OPT01).
- Risk engine (DD/kill/margin) se comporta de forma creíble bajo **stress sintético**.
- Diseño causal de features (sin look-ahead en ATR/volumen/RSI).

## 8. Siguiente loop (para cerrar el punto ciego)

| Prioridad | Acción |
|-----------|--------|
| P0 | IV real o al menos VIX-based surface proxy (no HV×1.15 fijo) |
| P0 | Multi-ventana 2022 bear + 2023 + stress en **cada** OPT_TA |
| P1 | Management: 50% credit take-profit, 2× stop, max 1 roll |
| P1 | Single-name options sleeve (AAPL/NVDA) con volume gates equity |
| P2 | Bid/ask haircut; assignment model |
| P2 | Book-level beta-weighted delta report |

## 9. Artefactos

| Pack | Path |
|------|------|
| Opciones TA innovadoras | `reports/paper_options_ta_smoke/latest/` |
| Opciones base | `reports/paper_options/latest/` |
| Mega 56 | `reports/paper_options_mega/` |
| Zoo TA | `paper_live/cloud/zoo_options_ta.json` |
| Gates | `paper_live/options/ta_gates.py` |
| Math BS | `paper_live/options/bs.py` |

_Not financial advice. Research paper only._
