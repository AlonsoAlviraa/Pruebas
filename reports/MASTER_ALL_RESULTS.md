# MASTER — todos los resultados paper (equity + opciones + TA)

_Generated 2026-07-22T11:15:23.845020+00:00 · capital VIRTUAL · no es consejo financiero_

## 0. Mapa de packs

| Pack | Descripción | Path |
|------|-------------|------|
| `equity_cloud_2026` | Equity cloud GitHub (solo 2026) | `reports/paper_cloud/latest/summary.json` (OK) |
| `equity_ab` | Equity A/B signal modes | `reports/paper_cloud_ab/latest/summary.json` (OK) |
| `equity_ta` | Equity TA/volume (Yahoo real) | `reports/paper_cloud_ta/latest/summary.json` (OK) |
| `equity_ta_synth` | Equity TA/volume (synthetic smoke) | `reports/paper_cloud_ta_smoke/latest/summary.json` (OK) |
| `opt_base` | Options base zoo | `reports/paper_options/latest/summary.json` (OK) |
| `opt_mega` | Options mega ~50–56 | `reports/paper_options_mega/latest/summary.json` (OK) |
| `opt_ta` | Options TA-gated | `reports/paper_options_ta_smoke/latest/summary.json` (OK) |

### Otros docs de auditoría

- `reports/paper_cloud/audits/LATEST_loss_audit.md` — por qué el zoo viejo iba en rojo
- `reports/paper_cloud_ab/audits/LOOP_AUD_AB_RESULTS.md` — A/B AUD-A/B
- `reports/paper_options_mega/MEGA_RESULTS.md` — mega 56 opciones
- `docs/design/2026-07-22_*.md` — diseños AUD, opciones, mega, TA

## 1. Qué se construyó (stack completo)

### Infra paper cloud (acciones)
- GitHub Actions diario (lun–vie), datos Yahoo reales, anti-synthetic gate
- Kill switch ajustado (sin false positives en sample corto)
- Ventana configurable `--start`/`--end`
- Instrumentación: closed_trades, exit_reason, SPY/eq-weight BH, WR, PF

### Señales equity
- Legacy: trend_mom, no_extension, pullback, topk, qqq_gate, qqq_hold
- **TA/volumen:** vol_confirm, rsi_mr, vol_dryup, vol_expand, rvol_trend, vol_pullback, combined_ta_v1

### Opciones (proxy_bs)
- Kinds: CC, CSP, PCS, CCS, iron_condor, collar, protective_put, cash
- Risk: DD, day-drop, margin-at-risk, hard kill, CVaR, multi-window, stress −30%
- Yahoo chain “hoy” (label real vs failed)
- Gates TA: uptrend, volume, RSI, ATR/range, climax, compression
- Mega zoo ~56 estrategias (CBOE/VRP/X/GitHub grid)

---

## Pack: Equity cloud GitHub (solo 2026)

- **Window:** 2026-01-02 → 2026-07-21
- **N strategies:** 10
- **Sources real/total:** 10/10
- **Positive / kill / beat SPY:** 0 / 0 / 0

| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |
|------|-----|-----------|-------:|-------:|---------|:----:|
| 1 | `S07_high_vol_only` |  | -1.59% | n/a | — | no |
| 2 | `S06_diversified` |  | -1.61% | n/a | — | no |
| 3 | `S08_low_vol_quality` |  | -2.42% | n/a | — | no |
| 4 | `S01_baseline_minalloc` |  | -2.62% | n/a | — | no |
| 5 | `S03_tight_stops` |  | -3.64% | n/a | — | no |
| 6 | `S02_no_regime` |  | -3.84% | n/a | — | no |
| 7 | `S09_aggressive_entries` |  | -4.65% | n/a | — | no |
| 8 | `S10_defensive` |  | -5.02% | n/a | — | no |
| 9 | `S04_wide_stops` |  | -5.36% | n/a | — | no |
| 10 | `S05_concentrated` |  | -5.86% | n/a | — | no |

## Pack: Equity A/B signal modes

- **Window:** 2025-10-29 → 2026-07-21
- **N strategies:** 11
- **SPY B&H:** 8.86%
- **Eq-weight BH:** 7.72%
- **Sources real/total:** 10/10
- **Positive / kill / beat SPY:** 4 / 0 / 2

| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |
|------|-----|-----------|-------:|-------:|---------|:----:|
| 1 | `AB10_qqq_bh_proxy` | qqq_hold | 11.24% | +2.38pp | — | no |
| 2 | `AB08_qqq_hold` | qqq_hold | 10.88% | +2.02pp | — | no |
| 3 | `AB02_pullback` | pullback | 1.23% | -7.63pp | PF 0.85 | no |
| 4 | `AB01_no_extension` | no_extension | 0.46% | -8.40pp | PF 1.15 | no |
| 5 | `AB04_qqq_gate` | qqq_gate | -0.02% | -8.88pp | PF 0.99 | no |
| 6 | `AB03_topk` | topk_mom | -0.51% | -9.37pp | PF 0.86 | no |
| 7 | `AB00_baseline` | trend_mom | -0.66% | -9.52pp | PF 0.89 | no |
| 8 | `AB09_wide_no_ext` | no_extension | -0.83% | -9.69pp | PF 0.66 | no |
| 9 | `AB06_combined_v2` | combined_v2 | -1.25% | -10.11pp | PF 0.50 | no |
| 10 | `AB05_combined_v1` | combined_v1 | -1.57% | -10.43pp | PF 0.69 | no |
| 11 | `AB07_combined_v3` | combined_v3 | -2.15% | -11.01pp | PF 0.51 | no |

## Pack: Equity TA/volume (Yahoo real)

- **Window:** 2025-10-29 → 2026-07-21
- **N strategies:** 10
- **SPY B&H:** 8.86%
- **Eq-weight BH:** 7.72%
- **Sources real/total:** 10/10
- **Positive / kill / beat SPY:** 4 / 0 / 0

| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |
|------|-----|-----------|-------:|-------:|---------|:----:|
| 1 | `TA06_vol_pullback` | vol_pullback | 0.32% | -8.54pp | PF 0.65 | no |
| 2 | `TA03_vol_dryup` | vol_dryup | 0.29% | -8.57pp | PF 0.65 | no |
| 3 | `TA09_rsi_mr_defensive` | rsi_mr | 0.21% | -8.65pp | PF 1.88 | no |
| 4 | `TA02_rsi_mr` | rsi_mr | 0.15% | -8.71pp | PF 1.46 | no |
| 5 | `TA04_vol_expand` | vol_expand | -0.16% | -9.02pp | PF 0.96 | no |
| 6 | `TA07_combined_ta` | combined_ta_v1 | -0.64% | -9.50pp | PF 0.82 | no |
| 7 | `TA05_rvol_trend` | rvol_trend | -0.80% | -9.66pp | PF 0.72 | no |
| 8 | `TA10_rvol_qqq_gate` | rvol_trend | -1.37% | -10.23pp | PF 0.41 | no |
| 9 | `TA01_vol_breakout` | vol_confirm | -1.91% | -10.77pp | PF 0.60 | no |
| 10 | `TA08_vol_breakout_topk` | vol_confirm | -2.94% | -11.80pp | PF 0.50 | no |

## Pack: Equity TA/volume (synthetic smoke)

- **Window:** 2026-03-24 → 2027-03-22
- **N strategies:** 10
- **SPY B&H:** 64.33%
- **Eq-weight BH:** 20.11%
- **Sources real/total:** 0/10
- **Positive / kill / beat SPY:** 9 / 0 / 0

| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |
|------|-----|-----------|-------:|-------:|---------|:----:|
| 1 | `TA07_combined_ta` | combined_ta_v1 | 4.31% | -60.02pp | PF 2.37 | no |
| 2 | `TA05_rvol_trend` | rvol_trend | 2.42% | -61.91pp | PF 1.96 | no |
| 3 | `TA01_vol_breakout` | vol_confirm | 2.26% | -62.07pp | PF 1.66 | no |
| 4 | `TA06_vol_pullback` | vol_pullback | 1.99% | -62.34pp | PF 1.95 | no |
| 5 | `TA03_vol_dryup` | vol_dryup | 1.91% | -62.42pp | PF 2.23 | no |
| 6 | `TA10_rvol_qqq_gate` | rvol_trend | 1.78% | -62.55pp | PF 1.68 | no |
| 7 | `TA04_vol_expand` | vol_expand | 1.66% | -62.67pp | PF 1.59 | no |
| 8 | `TA02_rsi_mr` | rsi_mr | 0.29% | -64.04pp | PF 99.00 | no |
| 9 | `TA09_rsi_mr_defensive` | rsi_mr | 0.25% | -64.08pp | PF 99.00 | no |
| 10 | `TA08_vol_breakout_topk` | vol_confirm | -0.61% | -64.94pp | PF 0.87 | no |

## Pack: Options base zoo

- **Window:** 2025-10-29 → 2026-07-21
- **N strategies:** 8
- **SPY B&H:** 8.86%
- **Data label:** `proxy_bs`
- **Sources real/total:** 2/2
- **Positive / kill / beat SPY:** 7 / 0 / 0

| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |
|------|-----|-----------|-------:|-------:|---------|:----:|
| 1 | `OPT01_covered_call` | covered_call | 7.14% | -1.72pp | CVaR -1.20% | no |
| 2 | `OPT_QQQ_cc` | covered_call | 7.08% | -1.77pp | CVaR -1.54% | no |
| 3 | `OPT04_collar` | collar | 6.71% | -2.15pp | CVaR -1.15% | no |
| 4 | `OPT06_csp_vrp_gate` | cash_secured_put | 1.75% | -7.11pp | CVaR -0.23% | no |
| 5 | `OPT02_csp` | cash_secured_put | 1.42% | -7.43pp | CVaR -0.25% | no |
| 6 | `OPT03_put_credit_spread` | put_credit_spread | 1.41% | -7.44pp | CVaR -0.25% | no |
| 7 | `OPT02b_csp_10otm` | cash_secured_put | 0.36% | -8.50pp | CVaR -0.06% | no |
| 8 | `OPT08_cash` | cash | 0.00% | -8.86pp | CVaR 0.00% | no |

## Pack: Options mega ~50–56

- **Window:** 2025-10-29 → 2026-07-21
- **N strategies:** 56
- **SPY B&H:** 8.86%
- **QQQ B&H:** 11.51%
- **Data label:** `proxy_bs`
- **Sources real/total:** 2/2
- **Positive / kill / beat SPY:** 54 / 0 / 0

| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |
|------|-----|-----------|-------:|-------:|---------|:----:|
| 1 | `M_pp_QQQ` | protective_put | 8.53% | -0.33pp | CVaR -1.60% | no |
| 2 | `M_cc_QQQ_cc_7otm` | covered_call | 7.61% | -1.24pp | CVaR -1.59% | no |
| 3 | `M_cc_SPY_bxm_atm` | covered_call | 7.14% | -1.72pp | CVaR -1.20% | no |
| 4 | `M_cc_SPY_cc_5otm` | covered_call | 7.14% | -1.72pp | CVaR -1.20% | no |
| 5 | `M_cc_QQQ_bxm_atm` | covered_call | 7.08% | -1.77pp | CVaR -1.54% | no |
| 6 | `M_cc_QQQ_cc_5otm` | covered_call | 7.08% | -1.77pp | CVaR -1.54% | no |
| 7 | `M_cc_SPY_cc_7otm` | covered_call | 6.94% | -1.92pp | CVaR -1.22% | no |
| 8 | `M_collar_SPY` | collar | 6.71% | -2.15pp | CVaR -1.15% | no |
| 9 | `M_cc_SPY_bxy_2otm` | covered_call | 6.49% | -2.37pp | CVaR -1.09% | no |
| 10 | `M_cc_QQQ_bxy_2otm` | covered_call | 6.27% | -2.59pp | CVaR -1.39% | no |
| 11 | `M_pp_SPY` | protective_put | 6.23% | -2.63pp | CVaR -1.22% | no |
| 12 | `M_collar_QQQ` | collar | 5.78% | -3.08pp | CVaR -1.25% | no |
| 13 | `M_cc_SPY_cc_10otm_45d` | covered_call | 4.95% | -3.91pp | CVaR -1.23% | no |
| 14 | `M_cc_SPY_cc_5otm_45d` | covered_call | 4.02% | -4.84pp | CVaR -1.21% | no |
| 15 | `M_csp_QQQ_5otm_45` | cash_secured_put | 3.50% | -5.35pp | CVaR -0.65% | no |
| 16 | `M_csp_vrp_QQQ` | cash_secured_put | 3.27% | -5.59pp | CVaR -0.48% | no |
| 17 | `M_csp_QQQ_3otm_30` | cash_secured_put | 3.21% | -5.65pp | CVaR -0.90% | no |
| 18 | `M_cc_QQQ_cc_5otm_45d` | covered_call | 2.77% | -6.08pp | CVaR -1.52% | no |
| 19 | `M_cc_QQQ_cc_10otm_45d` | covered_call | 2.71% | -6.15pp | CVaR -1.64% | no |
| 20 | `M_ic_spy_rich_iv` | iron_condor | 2.70% | -6.16pp | CVaR -0.25% | no |
| 21 | `M_csp_SPY_3otm_30` | cash_secured_put | 2.50% | -6.36pp | CVaR -0.48% | no |
| 22 | `M_csp_QQQ_5otm_30` | cash_secured_put | 2.31% | -6.55pp | CVaR -0.67% | no |
| 23 | `M_ic_SPY_cndrish` | iron_condor | 2.28% | -6.58pp | CVaR -0.27% | no |
| 24 | `M_ic_SPY_sym_3_8` | iron_condor | 2.15% | -6.71pp | CVaR -0.41% | no |
| 25 | `M_pcs_QQQ_5_15_30` | put_credit_spread | 2.10% | -6.76pp | CVaR -0.63% | no |
| 26 | `M_pcs_SPY_3_8_30` | put_credit_spread | 2.06% | -6.79pp | CVaR -0.43% | no |
| 27 | `M_ic_SPY_sym_5_10` | iron_condor | 1.99% | -6.87pp | CVaR -0.25% | no |
| 28 | `M_pcs_QQQ_3_8_30` | put_credit_spread | 1.90% | -6.96pp | CVaR -0.57% | no |
| 29 | `M_csp_spy_rich_iv` | cash_secured_put | 1.89% | -6.96pp | CVaR -0.28% | no |
| 30 | `M_csp_SPY_5otm_45` | cash_secured_put | 1.87% | -6.98pp | CVaR -0.38% | no |
| 31 | `M_csp_vrp_SPY` | cash_secured_put | 1.75% | -7.11pp | CVaR -0.23% | no |
| 32 | `M_pcs_QQQ_7_12_45` | put_credit_spread | 1.68% | -7.17pp | CVaR -0.31% | no |
| 33 | `M_ic_QQQ_sym_5_15_45` | iron_condor | 1.60% | -7.26pp | CVaR -0.53% | no |
| 34 | `M_csp_QQQ_7otm_30` | cash_secured_put | 1.59% | -7.27pp | CVaR -0.46% | no |
| 35 | `M_csp_QQQ_10otm_45` | cash_secured_put | 1.47% | -7.38pp | CVaR -0.29% | no |
| 36 | `M_pcs_QQQ_5_10_30` | put_credit_spread | 1.46% | -7.40pp | CVaR -0.47% | no |
| 37 | `M_csp_SPY_5otm_30` | cash_secured_put | 1.42% | -7.43pp | CVaR -0.25% | no |
| 38 | `M_pcs_SPY_5_15_30` | put_credit_spread | 1.41% | -7.44pp | CVaR -0.25% | no |
| 39 | `M_ic_SPY_sym_7_12` | iron_condor | 1.32% | -7.53pp | CVaR -0.12% | no |
| 40 | `M_pcs_SPY_5_10_30` | put_credit_spread | 1.26% | -7.60pp | CVaR -0.23% | no |
| 41 | `M_pcs_SPY_7_12_45` | put_credit_spread | 0.89% | -7.97pp | CVaR -0.19% | no |
| 42 | `M_csp_QQQ_10otm_30` | cash_secured_put | 0.85% | -8.01pp | CVaR -0.23% | no |
| 43 | `M_ccs_SPY_5_10` | call_credit_spread | 0.73% | -8.13pp | CVaR -0.20% | no |
| 44 | `M_ic_QQQ_cndrish` | iron_condor | 0.71% | -8.15pp | CVaR -0.63% | no |
| 45 | `M_ccs_SPY_7_12` | call_credit_spread | 0.70% | -8.16pp | CVaR -0.12% | no |
| 46 | `M_csp_SPY_7otm_30` | cash_secured_put | 0.68% | -8.18pp | CVaR -0.11% | no |
| 47 | `M_ic_QQQ_sym_3_8` | iron_condor | 0.61% | -8.25pp | CVaR -0.54% | no |
| 48 | `M_ic_QQQ_sym_7_12` | iron_condor | 0.52% | -8.34pp | CVaR -0.41% | no |
| 49 | `M_csp_QQQ_15otm_45` | cash_secured_put | 0.45% | -8.40pp | CVaR -0.11% | no |
| 50 | `M_ic_QQQ_sym_5_10` | iron_condor | 0.37% | -8.49pp | CVaR -0.51% | no |
| 51 | `M_csp_SPY_10otm_45` | cash_secured_put | 0.36% | -8.50pp | CVaR -0.06% | no |
| 52 | `M_csp_SPY_10otm_30` | cash_secured_put | 0.16% | -8.70pp | CVaR -0.02% | no |
| 53 | `M_ccs_SPY_3_8` | call_credit_spread | 0.09% | -8.77pp | CVaR -0.32% | no |
| 54 | `M_csp_SPY_15otm_45` | cash_secured_put | 0.04% | -8.82pp | CVaR -0.00% | no |
| 55 | `M00_cash` | cash | 0.00% | -8.86pp | CVaR 0.00% | no |
| 56 | `M_ic_SPY_sym_5_15_45` | iron_condor | -0.32% | -9.18pp | CVaR -0.59% | no |

## Pack: Options TA-gated

- **Window:** 2025-10-29 → 2026-07-21
- **N strategies:** 12
- **SPY B&H:** 8.86%
- **QQQ B&H:** 11.51%
- **Data label:** `proxy_bs`
- **Sources real/total:** 2/2
- **Positive / kill / beat SPY:** 10 / 0 / 0

| Rank | ID | Mode/Kind | Return | vs SPY | PF/CVaR | Kill |
|------|-----|-----------|-------:|-------:|---------|:----:|
| 1 | `OPT_TA04_cc_uptrend_vol` | covered_call | 8.21% | -0.65pp | CVaR -1.19% | no |
| 2 | `OPT_TA09_qqq_cc_trend_vol` | covered_call | 7.38% | -1.48pp | CVaR -1.53% | no |
| 3 | `OPT_TA11_pp_rsi_only` | protective_put | 3.36% | -5.50pp | CVaR -0.90% | no |
| 4 | `OPT_TA06_pcs_pullback_dry` | put_credit_spread | 1.04% | -7.82pp | CVaR -0.19% | no |
| 5 | `OPT_TA10_csp_sma200_dry` | cash_secured_put | 0.96% | -7.90pp | CVaR -0.09% | no |
| 6 | `OPT_TA07_csp_hv_range` | cash_secured_put | 0.70% | -8.15pp | CVaR -0.06% | no |
| 7 | `OPT_TA03_ic_low_atr` | iron_condor | 0.43% | -8.43pp | CVaR -0.05% | no |
| 8 | `OPT_TA01_csp_range` | cash_secured_put | 0.23% | -8.63pp | CVaR -0.06% | no |
| 9 | `OPT_TA02_pcs_compress` | put_credit_spread | 0.18% | -8.68pp | CVaR -0.04% | no |
| 10 | `OPT_TA08_ccs_overbought` | call_credit_spread | 0.13% | -8.73pp | CVaR -0.13% | no |
| 11 | `OPT_TA12_cash` | cash | 0.00% | -8.86pp | CVaR 0.00% | no |
| 12 | `OPT_TA05_pp_climax` | protective_put | -0.82% | -9.67pp | CVaR -0.79% | no |

## Stress crash sintético (opciones base pack)

Ver tabla completa en `reports/paper_options/latest/SUMMARY.md` sección Synthetic crash stress.

Resumen: cash 0%; collar ~−2.6%; PCS ~−6%; CSP/CC hard kill ~−15%…−18% con shock −30%.

## 2. Conclusiones cruzadas (honestas)

1. **Mercado alcista (SPY ~+9%, QQQ ~+11.5% en 2025-10→2026-07):** cualquier long-stock/buywrite gana; short premium puro gana poco y **no bate SPY**.
2. **Zoo equity viejo (2026 YTD):** 10/10 en rojo (−1.6%…−5.9%) con Yahoo real — comprar momentum/extensión + stops.
3. **A/B equity:** QQQ hold ~+11% (control índice); no_extension PF>1 y leve verde; pullback +1.2%; baseline ~−0.7%.
4. **TA/volume equity (Yahoo real):** ver pack `equity_ta` — filtros de volumen mejoran estructura de trades vs legacy en paper; no garantiza edge vs SPY.
5. **Opciones mega 56:** best ~+8.5% protective put QQQ / CC ~+7.6% — todos ≤ SPY/QQQ B&H.
6. **Opciones TA-gated:** covered call con uptrend+vol ~+8.2% (mejor que CC ciego en el base pack); short premium filtrado más bajo y selectivo.
7. **proxy_bs ≠ OPRA.** Paper only. No dinero real. Multi-ventana 2022–24 incompleta si no hay tape denso.

## 3. Cómo regenerar todo

```powershell
# Equity A/B
python scripts/run_paper_cloud_batch.py --zoo paper_live/cloud/strategy_zoo_ab.json --out reports/paper_cloud_ab --start 2025-10-29
# Equity TA
python scripts/run_paper_cloud_batch.py --zoo paper_live/cloud/strategy_zoo_ta.json --out reports/paper_cloud_ta --start 2025-10-29
# Options mega
python scripts/build_options_zoo_50.py
python scripts/run_paper_options_batch.py --zoo paper_live/cloud/zoo_options_50.json --out reports/paper_options_mega --start 2025-10-29
# Options TA
python scripts/run_paper_options_batch.py --zoo paper_live/cloud/zoo_options_ta.json --out reports/paper_options_ta --start 2025-10-29
python scripts/write_master_results_report.py
```

_Research software. Past paper ≠ future results._
