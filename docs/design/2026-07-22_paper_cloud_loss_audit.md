# Mega-auditoría paper cloud — por qué estamos en negativo

_Generated 2026-07-22T08:40:37.827456+00:00 · pack `2026-07-22`_

## 0. Scope y honestidad

- **Ventana analizada (latest SUMMARY):** `2025-10-29` → `2026-07-21` · capital virtual $100,000.
- **Datos:** Yahoo free OHLCV (mega-caps + SPY/QQQ). Paper only — no dinero real.
- **Señal:** rule-based `rule_trend_mom_atr` (no XGBoost meta-label del stack research).
- Round-trips reconstruidos por **FIFO** sobre fills diarios; PnL de closed trades ≠ equity final exacto (quedan posiciones abiertas + MTM).

## 1. Veredicto ejecutivo

**10/10 estrategias en negativo.** No es un solo kill-switch ni un bug de datos sintéticos: el **edge de la regla long-momentum no aparece** en este tramo frente a buy&hold, y el zoo solo re-parametriza la misma idea.

- **SPY buy&hold misma ventana:** +8.86% (maxDD -9.13%, 181 sesiones).
- **SPY B&H 2026-01-02→fin pack:** +9.53% (maxDD -9.13%).
- **Mejor zoo:** `S07_high_vol_only` -3.08% · **Peor:** `S05_concentrated` -9.11%.
- **Closed trades (todas las strats):** n=560 · WR=21.8% · avg_ret=-3.49% · PF=0.337 · closed_pnl_sum=$-62181.
- **Subset entries ≥2026-01-01:** n=327 · WR=24.5% · avg_ret=-2.71% · PF=0.487.

## 2. Ranking vs costes y exposición

| Strat | Return | Entries | Closed | WR | Avg ret | PF | Closed PnL | Comm | Avg exp | MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S07_high_vol_only` | -3.08% | 13 | 11 | 0.0% | -6.89% | 0.00 | $-2941 | $24 | 2.1% | -3.18% |
| `S06_diversified` | -3.41% | 62 | 57 | 21.1% | -3.30% | 0.40 | $-3307 | $119 | 6.8% | -3.64% |
| `S08_low_vol_quality` | -5.09% | 58 | 53 | 20.8% | -2.52% | 0.49 | $-4944 | $111 | 13.9% | -5.58% |
| `S01_baseline_minalloc` | -5.23% | 62 | 57 | 21.1% | -3.30% | 0.39 | $-5116 | $119 | 10.3% | -5.58% |
| `S03_tight_stops` | -6.08% | 77 | 76 | 17.1% | -2.94% | 0.31 | $-6024 | $153 | 9.2% | -6.30% |
| `S02_no_regime` | -6.31% | 69 | 63 | 19.0% | -3.12% | 0.39 | $-5850 | $131 | 11.3% | -6.69% |
| `S04_wide_stops` | -7.52% | 35 | 31 | 12.9% | -9.17% | 0.07 | $-7895 | $66 | 11.5% | -8.06% |
| `S09_aggressive_entries` | -8.06% | 104 | 104 | 37.5% | -3.00% | 0.37 | $-8491 | $202 | 10.0% | -8.30% |
| `S10_defensive` | -8.61% | 71 | 70 | 14.3% | -3.35% | 0.23 | $-8573 | $141 | 9.9% | -8.71% |
| `S05_concentrated` | -9.11% | 42 | 38 | 23.7% | -3.17% | 0.42 | $-9043 | $80 | 19.8% | -9.57% |

### Lectura rápida del zoo

- **S07 high_vol_only** menos rojo → **menos trades** (menos veces pagas el edge negativo).
- **S09 aggressive** y **S05 concentrated** peores → más tamaño o más frecuencia **amplifican** el mismo edge negativo.
- **S02 no_regime** no mejora a baseline → el régimen QQQ dual-MA no es el único problema; la selección de nombres también pierde.
- Comisiones (~$24–$202) son **pequeñas** vs miles de $ de equity drag → el rojo es **PnL de mercado**, no solo fricción.

## 3. Anatomía de los trades (agregado)

- Mediana hold (calendario): **13.0 días**
- Avg win / avg loss (ret): +8.34% / -6.79%
- P25 / P75 ret: -7.15% / -1.86%
- Best / worst single trade ret: +31.98% / -24.91%

### Por ticker (peores primero, closed PnL)

| Ticker | n | WR | Avg ret | Closed PnL |
|---|---:|---:|---:|---:|
| NVDA | 98 | 5.1% | -5.90% | $-19410 |
| AMZN | 71 | 19.7% | -3.86% | $-9055 |
| QQQ | 81 | 17.3% | -3.57% | $-7827 |
| AAPL | 70 | 30.0% | -2.94% | $-6803 |
| MSFT | 20 | 0.0% | -7.15% | $-4204 |
| META | 17 | 0.0% | -8.24% | $-4016 |
| JPM | 56 | 44.6% | -2.15% | $-3835 |
| XOM | 68 | 26.5% | -2.01% | $-3829 |
| GOOGL | 79 | 31.6% | -0.87% | $-3203 |

### Por mes de salida

| Month | n | WR | Avg ret | Closed PnL |
|---|---:|---:|---:|---:|
| 2025-10 | 9 | 0.0% | -7.30% | $-2181 |
| 2025-11 | 113 | 7.1% | -6.35% | $-22826 |
| 2025-12 | 55 | 32.7% | -2.53% | $-5389 |
| 2026-01 | 74 | 25.7% | -2.73% | $-6046 |
| 2026-02 | 71 | 14.1% | -4.20% | $-8353 |
| 2026-03 | 20 | 15.0% | -5.59% | $-3936 |
| 2026-04 | 43 | 34.9% | -2.66% | $-4066 |
| 2026-05 | 72 | 47.2% | +2.46% | $5213 |
| 2026-06 | 81 | 7.4% | -5.78% | $-14610 |
| 2026-07 | 22 | 40.9% | -0.75% | $12 |

### Peores 15 round-trips (todas las strats)

| Strat | Ticker | Entry | Exit | Hold d | Ret | PnL |
|---|---|---|---|---:|---:|---:|
| `S09_aggressive_entries` | GOOGL | 2025-12-25 @ 402.68 | 2026-01-20 @ 302.35 | 26 | -24.91% | $-702 |
| `S05_concentrated` | AMZN | 2025-11-03 @ 255.51 | 2025-11-13 @ 234.70 | 10 | -8.15% | $-645 |
| `S05_concentrated` | NVDA | 2025-10-29 @ 208.10 | 2025-11-06 @ 191.15 | 8 | -8.15% | $-644 |
| `S05_concentrated` | JPM | 2025-11-07 @ 312.08 | 2025-11-14 @ 286.65 | 7 | -8.15% | $-636 |
| `S05_concentrated` | META | 2026-04-20 @ 681.77 | 2026-04-30 @ 618.33 | 10 | -9.31% | $-634 |
| `S05_concentrated` | GOOGL | 2025-10-29 @ 267.91 | 2025-11-14 @ 246.08 | 16 | -8.15% | $-633 |
| `S05_concentrated` | AMZN | 2025-11-14 @ 235.20 | 2025-11-20 @ 216.04 | 6 | -8.15% | $-632 |
| `S05_concentrated` | AAPL | 2025-11-26 @ 277.13 | 2025-12-18 @ 254.55 | 22 | -8.15% | $-632 |
| `S05_concentrated` | GOOGL | 2025-11-17 @ 285.95 | 2025-12-15 @ 262.65 | 28 | -8.15% | $-629 |
| `S05_concentrated` | MSFT | 2025-10-29 @ 545.27 | 2025-10-31 @ 500.84 | 2 | -8.15% | $-622 |
| `S05_concentrated` | XOM | 2026-02-26 @ 147.90 | 2026-03-04 @ 135.85 | 6 | -8.15% | $-615 |
| `S05_concentrated` | AMZN | 2026-01-23 @ 235.10 | 2026-02-05 @ 215.95 | 13 | -8.15% | $-613 |
| `S05_concentrated` | NVDA | 2026-01-29 @ 191.45 | 2026-02-03 @ 175.86 | 5 | -8.15% | $-608 |
| `S05_concentrated` | NVDA | 2026-05-01 @ 201.40 | 2026-05-22 @ 184.99 | 21 | -8.15% | $-607 |
| `S05_concentrated` | AAPL | 2026-02-26 @ 275.11 | 2026-03-06 @ 252.70 | 8 | -8.15% | $-605 |

### Rejects de entrada (confirm gap / filtros)

- `gap_down`: 5

## 4. Detalle baseline S01 (referencia)

Return -5.23% · closed n=57 · WR=21.1% · PF=0.390 · avg exp=10.3%.

Top drag tickers S01:

| Ticker | n | WR | PnL |
|---|---:|---:|---:|
| NVDA | 9 | 0.0% | $-1623 |
| QQQ | 9 | 11.1% | $-781 |
| AAPL | 7 | 28.6% | $-575 |
| MSFT | 3 | 0.0% | $-573 |
| AMZN | 7 | 28.6% | $-572 |
| META | 2 | 0.0% | $-462 |

Peores trades S01:
- META 2026-04-20→2026-04-30 -9.31% ($-254)
- XOM 2026-04-09→2026-04-17 -7.85% ($-222)
- AMZN 2025-10-30→2025-11-06 -7.15% ($-211)
- GOOGL 2025-10-29→2025-11-14 -7.15% ($-211)
- NVDA 2025-11-11→2025-11-14 -7.15% ($-209)
- AMZN 2025-11-07→2025-11-17 -7.15% ($-208)
- NVDA 2025-10-29→2025-11-06 -7.15% ($-208)
- XOM 2025-10-30→2025-11-24 -7.15% ($-208)

## 5. Por qué fallan (causas raíz)

1. BENCHMARK: SPY B&H +8.86% while avg strategy -6.25% — system underperformed a passive long index (edge not present in this window).
2. PROFIT FACTOR: 0.337 < 1 — gross losses exceed gross wins on closed round-trips (before open MTM).
3. TICKER DRAG: worst book is NVDA closed PnL $-19410 (n=98, WR=5.1%) — name selection or repeated re-entry into same mega-cap after failed momentum.
4. ENTRY FILTERS: rejects gap_down×5 — gap rules kill some bad entries but also drop continuation days; remaining entries still lose, so filter is not the main alpha problem.
5. SIGNAL DESIGN: rule is long-only close>SMA50/200 + ret_1m>0 + ATR band. That is late-trend entry (buy strength). In range/choppy mega-cap markets this buys local tops; there is no mean-reversion, no short, no meta-label skip, no sector rotation.
6. UNIVERSE: 8 mega-caps + QQQ/SPY — highly correlated; diversification in zoo (S06 vs S05) barely helps if the common factor is 'long NVDA/META/etc after up month'.
7. SIZING/COSTS: fixed commission + entry/exit slippage on many small tickets (aggressive S09 worst on turnover) compounds when expectancy per trade is near zero/negative.
8. NOT ML EDGE: cloud paper is rule-based proxy (LIV-04), not the XGB+meta stack from research — do not interpret paper_cloud red as proof the full research pipeline fails, but also do not claim paper_cloud proves production readiness.

### Mecánica concreta del trade perdedor típico

```
D close: close > SMA50 & SMA200, ret_1m > 0, ATR en banda → score alto
D+1 open: confirm (no gap down >5% / no chase >8%) → buy ~1.5% NAV
In-trade: stop = max(entry*(1-hard%), entry - k*ATR); trail con high
Exit: low toca stop  OR  bars_held >= max_horizon → time_stop al close
```

El patrón de fallo: **compras después de un mes alcista** en nombres correlacionados; si los siguientes 5–15 días son digieren/range, el **stop o el time_stop** cierra en leve rojo; los pocos winners no compensan (PF<1). Amplificar entradas (S09) o tamaño (S05) empeora; **no tradear** (S07 filtro ATR alto) pierde menos.

## 6. Lo que NO es la causa principal

- ~~Datos sintéticos~~ — este pack es Yahoo real.
- ~~Kill switch~~ — hard_kill=false en las 10.
- ~~Solo comisiones~~ — drag de comisiones << pérdida de equity.
- ~~Un bug de un solo parámetro del zoo~~ — todas las variantes rojas.

## 7. Plan de acción (priorizado)

### AUD-01 · Benchmark honesty + attribution pack · **P0**

**Por qué:** Without SPY/equal-weight daily attribution we cannot tell alpha vs beta drag.

**Hacer:**
- Add SPY B&H and equal-weight universe B&H curves to every SUMMARY.
- Log exit_reason distribution (stop vs time_stop vs EOD) into digests.
- Emit closed-trade CSV per strategy (entry/exit/ret/hold/reason).

### AUD-02 · Fix expectancy before more zoo knobs · **P0**

**Por qué:** All 10 variants red ⇒ shared signal is the problem, not stop width alone.

**Hacer:**
- A/B: require ret_1m rank top-k only (not all positive).
- A/B: add pullback entry (close>SMA200 but RSI/near SMA50) vs pure breakout.
- A/B: meta-skip when QQQ 5d ret < 0 even if dual MA on.
- Disable trading QQQ/SPY as names (regime only) — reduce beta double-count.

### AUD-03 · Exit asymmetry · **P1**

**Por qué:** Time-stop + hard stop realizes losses; winners capped by horizon.

**Hacer:**
- Trail winners more aggressively; lengthen max_horizon on strong trends (ATR).
- Partial take-profit at +1R; let runner to 2–3R.
- Measure MAE/MFE on every closed trade in audit CSV.

### AUD-04 · Cost / turnover control · **P1**

**Por qué:** S09 high entries + commissions with negative edge is pure leak.

**Hacer:**
- Cap max_entries_per_day globally; min hold before re-entry same ticker.
- Raise min_alloc so fewer micro tickets; or commission-aware skip if edge < cost.

### AUD-05 · True OOS protocol for paper cloud · **P0**

**Por qué:** Single ~9m window is not walk-forward; 2026-only is still one regime.

**Hacer:**
- Mandatory multi-window: 2022 bear, 2023 bull, 2024, 2025, 2026 YTD separately.
- Kill strategies that lose to SPY in ≥3/5 windows.
- Only promote rules that beat SPY after costs on purged/walk-forward research first.

### AUD-06 · Reconnect research stack (optional) · **P2**

**Por qué:** Rule_trend_mom is a stub; XGB+meta may differ.

**Hacer:**
- Plug frozen signal model into DailySignalPipeline.signal_fn with feature parity.
- Paper cloud becomes validation harness, not the strategy definition.

## 8. Criterios de éxito (antes de decir 'arreglado')

1. Al menos **1** variante con return > SPY B&H **después de costes** en ≥2 ventanas OOS.
2. Profit factor closed trades **> 1.1** y win*avg_win + (1-win)*avg_loss **> 0**.
3. Audit CSV con exit_reason + MAE/MFE commiteado en cada pack cloud.
4. No reclamar edge si solo mejora el tramo 2026-YTD en aislamiento.

## 9. Próximo PR sugerido

1. **PR-AUD-A:** trade log + exit_reason + SPY benchmark en SUMMARY (instrumentación).
2. **PR-AUD-B:** 4 A/B del signal (rank top-k, pullback, meta-skip QQQ weak, no trade index).
3. **PR-AUD-C:** multi-window batch en Actions (`start` por era) + scorecard comparativo.

---
_Research software. Not financial advice._
