# Mega paper options test (~50 strategies)

_Generated 2026-07-22T10:17:01.739280+00:00 · VIRTUAL · data_label=`proxy_bs`_

## Research sources

| Channel | Takeaways used in zoo |
|---------|----------------------|
| **CBOE indexes** | BXM/BXY buy-write, PUT put-write, CNDR iron condor families |
| **Papers / Quantpedia** | VRP (IV>RV); OTM 5–10% put-write; defined-risk wings |
| **Twitter/X** | IC = positioning; equity drift → prefer PCS / wider structures; CNDR fails when RV>IV |
| **GitHub style** | Parametric grids (underlying × OTM × DTE) over few structure kinds |

## Protocol

- Window: **2025-10-29 → 2026-07-21**
- Strategies completed: **56**
- Capital: VIRTUAL $100,000
- Marks: Black–Scholes on HV/IV proxy (**not** OPRA fills)
- SPY B&H: **8.86%** · QQQ B&H: **11.51%**

## Headline results

- Positive return: **54** / 56
- Hard kill: **0**
- Beat SPY: **0**

## Average by structure kind

| Kind | n | Avg ret | Best ret |
|------|---:|--------:|---------:|
| `protective_put` | 2 | 7.38% | 8.53% |
| `collar` | 2 | 6.24% | 6.71% |
| `covered_call` | 12 | 5.85% | 7.61% |
| `cash_secured_put` | 17 | 1.61% | 3.50% |
| `put_credit_spread` | 8 | 1.60% | 2.10% |
| `iron_condor` | 11 | 1.27% | 2.70% |
| `call_credit_spread` | 3 | 0.51% | 0.73% |
| `cash` | 1 | 0.00% | 0.00% |

## Top 15 by total return

| Rank | ID | Kind | Ret | MaxDD | CVaR5% | vs SPY | Kill |
|------|-----|------|----:|------:|-------:|-------:|:----:|
| 1 | `M_pp_QQQ` | protective_put | 8.53% | -7.42% | -1.60% | -0.33pp | no |
| 2 | `M_cc_QQQ_cc_7otm` | covered_call | 7.61% | -6.94% | -1.59% | -1.24pp | no |
| 3 | `M_cc_SPY_bxm_atm` | covered_call | 7.14% | -5.80% | -1.20% | -1.72pp | no |
| 4 | `M_cc_SPY_cc_5otm` | covered_call | 7.14% | -5.80% | -1.20% | -1.72pp | no |
| 5 | `M_cc_QQQ_bxm_atm` | covered_call | 7.08% | -6.39% | -1.54% | -1.77pp | no |
| 6 | `M_cc_QQQ_cc_5otm` | covered_call | 7.08% | -6.39% | -1.54% | -1.77pp | no |
| 7 | `M_cc_SPY_cc_7otm` | covered_call | 6.94% | -6.12% | -1.22% | -1.92pp | no |
| 8 | `M_collar_SPY` | collar | 6.71% | -5.65% | -1.15% | -2.15pp | no |
| 9 | `M_cc_SPY_bxy_2otm` | covered_call | 6.49% | -4.58% | -1.09% | -2.37pp | no |
| 10 | `M_cc_QQQ_bxy_2otm` | covered_call | 6.27% | -4.83% | -1.39% | -2.59pp | no |
| 11 | `M_pp_SPY` | protective_put | 6.23% | -6.29% | -1.22% | -2.63pp | no |
| 12 | `M_collar_QQQ` | collar | 5.78% | -6.02% | -1.25% | -3.08pp | no |
| 13 | `M_cc_SPY_cc_10otm_45d` | covered_call | 4.95% | -6.24% | -1.23% | -3.91pp | no |
| 14 | `M_cc_SPY_cc_5otm_45d` | covered_call | 4.02% | -5.81% | -1.21% | -4.84pp | no |
| 15 | `M_csp_QQQ_5otm_45` | cash_secured_put | 3.50% | -1.78% | -0.65% | -5.35pp | no |

## Bottom 10

| Rank | ID | Kind | Ret | MaxDD | vs SPY |
|------|-----|------|----:|------:|-------:|
| 47 | `M_ic_QQQ_sym_3_8` | iron_condor | 0.61% | -3.92% | -8.25pp |
| 48 | `M_ic_QQQ_sym_7_12` | iron_condor | 0.52% | -2.60% | -8.34pp |
| 49 | `M_csp_QQQ_15otm_45` | cash_secured_put | 0.45% | -0.36% | -8.40pp |
| 50 | `M_ic_QQQ_sym_5_10` | iron_condor | 0.37% | -3.68% | -8.49pp |
| 51 | `M_csp_SPY_10otm_45` | cash_secured_put | 0.36% | -0.17% | -8.50pp |
| 52 | `M_csp_SPY_10otm_30` | cash_secured_put | 0.16% | -0.07% | -8.70pp |
| 53 | `M_ccs_SPY_3_8` | call_credit_spread | 0.09% | -2.68% | -8.77pp |
| 54 | `M_csp_SPY_15otm_45` | cash_secured_put | 0.04% | -0.01% | -8.82pp |
| 55 | `M00_cash` | cash | 0.00% | 0.00% | -8.86pp |
| 56 | `M_ic_SPY_sym_5_15_45` | iron_condor | -0.32% | -3.76% | -9.18pp |

## Interpretation (honest)

1. **Bull window** (SPY +8.9%, QQQ +11.5%): long-stock + short call (buywrite/collar/PP) dominate ranking because they keep **equity beta**.
2. **Pure short premium** (CSP / PCS / IC / CCS) is **positive but lags SPY** — classic VRP income profile in a strong market, not free alpha.
3. **Iron condors** ~+0.5–2% average — X thesis that IC is not "easy money" holds; only one strategy slightly negative.
4. **Zero hard kills** on this calm path; use `--stress` for crash behavior.
5. **No strategy beats SPY** here (best buywrite still ~1–2pp short of SPY due to capped upside + rolls).
6. These are **parameterizations of known structures**, not 50 independent alpha ideas.

## How to reproduce

```powershell
python scripts/build_options_zoo_50.py
python scripts/run_paper_options_batch.py --zoo paper_live/cloud/zoo_options_50.json --out reports/paper_options_mega --start 2025-10-29
```

## Artifacts

- Zoo: `paper_live/cloud/zoo_options_50.json`
- Pack: `reports/paper_options_mega/latest/`
- Design: `docs/design/2026-07-22_options_mega_50.md`

_Not financial advice. proxy_bs ≠ exchange fills._
