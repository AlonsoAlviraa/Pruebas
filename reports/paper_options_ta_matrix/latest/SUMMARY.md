# OPT_TA multi-window matrix — `2026-07-22`

**Zoo:** `paper_live\cloud\zoo_options_ta.json`
**Capital:** VIRTUAL $100,000
**VIX in feed:** True
**N strategies:** 12

## Data quality labels

- Marks: Black–Scholes (`proxy_bs` math)
- IV: `vix_surface` when VIX (±VIX3M) available, else `proxy_hv` (HV×premium_mult)
- Fills: bid haircut on sells (default 5%); not NBBO
- Assignment: `assignment_proxy` at expiry / deep ITM
- Management: 50% credit TP, 2× credit SL, max 1 **DTE roll** per structure (meta-overridable)
- Counters: **Opens** = every successful entry; **DTE rolls** = roll-only (capped by max_rolls)
- Stress: equity path shock **and** VIX/VIX3M spike (not spot-only)

## Windows overview

| Window | Requested | Actual | Clamped | Best ID | Best Ret | Worst MaxDD | SPY B&H | Book Δend |
|--------|-----------|--------|---------|---------|----------|-------------|---------|-----------|
| 2022_bear | 2022-01-03→2022-12-30 | 2022-03-07→2022-12-30 | yes | `OPT_TA01_csp_range` | 0.29% | -6.61% | -8.82% | 231.7145438391899 |
| 2023 | 2023-01-03→2023-12-29 | 2023-01-03→2023-12-29 | no | `OPT_TA09_qqq_cc_trend_vol` | 10.53% | -4.62% | 24.81% | 358.30558013668235 |
| 2024 | 2024-01-02→2024-12-31 | 2024-01-02→2024-12-31 | no | `OPT_TA04_cc_uptrend_vol` | 12.18% | -6.10% | 24.00% | 441.75773563840255 |
| 2025_study | 2025-10-29→2026-07-22 | 2025-10-29→2026-07-22 | no | `OPT_TA04_cc_uptrend_vol` | 9.14% | -7.09% | n/a | 438.3725142330807 |

## Per-window detail

### Window `2022_bear`

- **Actual:** 2022-03-07 → 2022-12-30
- **Requested:** 2022-01-03 → 2022-12-30
- **Clamped:** **yes** (history short — labeled honestly)
- **data_label / IV:** `mixed` · sources=['proxy_hv', 'vix_surface']
- **SPY B&H:** -8.82% · **QQQ B&H:** -18.03%

| Rank | ID | Kind | Und | Return | MaxDD | CVaR5% | Opens | DTE rolls | TP/SL | Δend | vsSPY | Kill |
|------|-----|------|-----|--------|-------|--------|-------|-----------|------|------|-------|------|
| 1 | `OPT_TA01_csp_range` | cash_secured_put | SPY | 0.29% | -1.14% | -0.21% | 7 | 0 | 5/1 | 41 | 9.12% | no |
| 2 | `OPT_TA08_ccs_overbought` | call_credit_spread | SPY | 0.22% | -0.30% | -0.04% | 1 | 0 | 1/0 | 0 | 9.04% | no |
| 3 | `OPT_TA02_pcs_compress` | put_credit_spread | SPY | 0.22% | -0.03% | -0.00% | 1 | 0 | 1/0 | 0 | 9.04% | no |
| 4 | `OPT_TA05_pp_climax` | protective_put | SPY | 0.00% | 0.00% | 0.00% | 0 | 0 | 0/0 | 0 | 8.82% | no |
| 5 | `OPT_TA06_pcs_pullback_dry` | put_credit_spread | SPY | 0.00% | 0.00% | 0.00% | 0 | 0 | 0/0 | 0 | 8.82% | no |
| 6 | `OPT_TA09_qqq_cc_trend_vol` | covered_call | QQQ | 0.00% | 0.00% | 0.00% | 0 | 0 | 0/0 | 0 | 8.82% | no |
| 7 | `OPT_TA12_cash` | cash | SPY | 0.00% | 0.00% | 0.00% | 0 | 0 | 0/0 | 0 | 8.82% | no |
| 8 | `OPT_TA10_csp_sma200_dry` | cash_secured_put | SPY | -0.25% | -0.72% | -0.10% | 3 | 0 | 2/1 | 0 | 8.58% | no |
| 9 | `OPT_TA07_csp_hv_range` | cash_secured_put | SPY | -1.13% | -1.67% | -0.30% | 8 | 0 | 5/2 | 0 | 7.69% | no |
| 10 | `OPT_TA04_cc_uptrend_vol` | covered_call | SPY | -2.03% | -6.47% | -0.95% | 4 | 0 | 4/0 | 100 | 6.79% | no |
| 11 | `OPT_TA03_ic_low_atr` | iron_condor | SPY | -2.03% | -2.60% | -0.23% | 7 | 2 | 2/2 | -10 | 6.79% | no |
| 12 | `OPT_TA11_pp_rsi_only` | protective_put | SPY | -3.87% | -6.61% | -0.94% | 1 | 0 | 0/0 | 100 | 4.95% | no |

**Book delta (approx):** sum_end=231.7145438391899 · mean_end=19.309545319932493 · label=`approx_bs_delta_book`

### Window `2023`

- **Actual:** 2023-01-03 → 2023-12-29
- **Requested:** 2023-01-03 → 2023-12-29
- **Clamped:** no
- **data_label / IV:** `mixed` · sources=['proxy_hv', 'vix_surface']
- **SPY B&H:** 24.81% · **QQQ B&H:** 54.84%

| Rank | ID | Kind | Und | Return | MaxDD | CVaR5% | Opens | DTE rolls | TP/SL | Δend | vsSPY | Kill |
|------|-----|------|-----|--------|-------|--------|-------|-----------|------|------|-------|------|
| 1 | `OPT_TA09_qqq_cc_trend_vol` | covered_call | QQQ | 10.53% | -3.70% | -0.64% | 17 | 0 | 10/5 | 90 | -14.28% | no |
| 2 | `OPT_TA04_cc_uptrend_vol` | covered_call | SPY | 8.33% | -4.12% | -0.63% | 13 | 0 | 9/0 | 89 | -16.48% | no |
| 3 | `OPT_TA05_pp_climax` | protective_put | SPY | 5.70% | -4.53% | -0.64% | 3 | 0 | 0/0 | 98 | -19.11% | no |
| 4 | `OPT_TA11_pp_rsi_only` | protective_put | SPY | 3.47% | -4.62% | -0.59% | 4 | 0 | 0/0 | 100 | -21.35% | no |
| 5 | `OPT_TA03_ic_low_atr` | iron_condor | SPY | 1.43% | -0.48% | -0.15% | 17 | 2 | 13/1 | -12 | -23.38% | no |
| 6 | `OPT_TA01_csp_range` | cash_secured_put | SPY | 1.14% | -0.69% | -0.16% | 20 | 0 | 18/2 | 0 | -23.67% | no |
| 7 | `OPT_TA10_csp_sma200_dry` | cash_secured_put | SPY | 0.89% | -0.58% | -0.14% | 25 | 0 | 22/3 | 0 | -23.93% | no |
| 8 | `OPT_TA07_csp_hv_range` | cash_secured_put | SPY | 0.71% | -0.66% | -0.15% | 14 | 0 | 12/2 | 0 | -24.11% | no |
| 9 | `OPT_TA08_ccs_overbought` | call_credit_spread | SPY | 0.20% | -0.25% | -0.07% | 7 | 0 | 5/1 | -7 | -24.62% | no |
| 10 | `OPT_TA02_pcs_compress` | put_credit_spread | SPY | 0.17% | -0.74% | -0.12% | 11 | 0 | 9/2 | 0 | -24.64% | no |
| 11 | `OPT_TA06_pcs_pullback_dry` | put_credit_spread | SPY | 0.04% | -0.66% | -0.15% | 13 | 0 | 10/3 | 0 | -24.78% | no |
| 12 | `OPT_TA12_cash` | cash | SPY | 0.00% | 0.00% | 0.00% | 0 | 0 | 0/0 | 0 | -24.81% | no |

**Book delta (approx):** sum_end=358.30558013668235 · mean_end=29.858798344723528 · label=`approx_bs_delta_book`

### Window `2024`

- **Actual:** 2024-01-02 → 2024-12-31
- **Requested:** 2024-01-02 → 2024-12-31
- **Clamped:** no
- **data_label / IV:** `mixed` · sources=['proxy_hv', 'vix_surface']
- **SPY B&H:** 24.00% · **QQQ B&H:** 26.99%

| Rank | ID | Kind | Und | Return | MaxDD | CVaR5% | Opens | DTE rolls | TP/SL | Δend | vsSPY | Kill |
|------|-----|------|-----|--------|-------|--------|-------|-----------|------|------|-------|------|
| 1 | `OPT_TA04_cc_uptrend_vol` | covered_call | SPY | 12.18% | -4.47% | -0.97% | 17 | 0 | 11/2 | 100 | -11.82% | no |
| 2 | `OPT_TA09_qqq_cc_trend_vol` | covered_call | QQQ | 10.73% | -6.10% | -1.16% | 21 | 0 | 14/6 | 100 | -13.27% | no |
| 3 | `OPT_TA11_pp_rsi_only` | protective_put | SPY | 9.58% | -4.42% | -0.97% | 2 | 0 | 0/0 | 100 | -14.41% | no |
| 4 | `OPT_TA05_pp_climax` | protective_put | SPY | 3.77% | -4.67% | -0.98% | 1 | 0 | 0/0 | 100 | -20.23% | no |
| 5 | `OPT_TA07_csp_hv_range` | cash_secured_put | SPY | 1.13% | -0.57% | -0.12% | 15 | 0 | 13/2 | 0 | -22.87% | no |
| 6 | `OPT_TA01_csp_range` | cash_secured_put | SPY | 0.96% | -0.98% | -0.20% | 19 | 0 | 17/2 | 0 | -23.04% | no |
| 7 | `OPT_TA10_csp_sma200_dry` | cash_secured_put | SPY | 0.78% | -0.76% | -0.18% | 31 | 0 | 26/4 | 17 | -23.22% | no |
| 8 | `OPT_TA03_ic_low_atr` | iron_condor | SPY | 0.74% | -0.77% | -0.13% | 15 | 0 | 13/1 | 0 | -23.26% | no |
| 9 | `OPT_TA08_ccs_overbought` | call_credit_spread | SPY | 0.71% | -0.17% | -0.06% | 10 | 0 | 9/0 | 0 | -23.29% | no |
| 10 | `OPT_TA02_pcs_compress` | put_credit_spread | SPY | 0.28% | -0.36% | -0.10% | 10 | 0 | 9/1 | 0 | -23.72% | no |
| 11 | `OPT_TA06_pcs_pullback_dry` | put_credit_spread | SPY | 0.05% | -1.16% | -0.18% | 11 | 0 | 9/1 | 25 | -23.95% | no |
| 12 | `OPT_TA12_cash` | cash | SPY | 0.00% | 0.00% | 0.00% | 0 | 0 | 0/0 | 0 | -24.00% | no |

**Book delta (approx):** sum_end=441.75773563840255 · mean_end=36.81314463653354 · label=`approx_bs_delta_book`

### Window `2025_study`

- **Actual:** 2025-10-29 → 2026-07-22
- **Requested:** 2025-10-29 → 2026-07-22
- **Clamped:** no
- **data_label / IV:** `mixed` · sources=['proxy_hv', 'vix_surface']
- **SPY B&H:** n/a · **QQQ B&H:** n/a

| Rank | ID | Kind | Und | Return | MaxDD | CVaR5% | Opens | DTE rolls | TP/SL | Δend | vsSPY | Kill |
|------|-----|------|-----|--------|-------|--------|-------|-----------|------|------|-------|------|
| 1 | `OPT_TA04_cc_uptrend_vol` | covered_call | SPY | 9.14% | -5.75% | -1.17% | 8 | 0 | 7/0 | 100 | n/a | no |
| 2 | `OPT_TA09_qqq_cc_trend_vol` | covered_call | QQQ | 7.74% | -7.09% | -1.56% | 7 | 0 | 6/1 | 100 | n/a | no |
| 3 | `OPT_TA11_pp_rsi_only` | protective_put | SPY | 3.49% | -3.26% | -0.90% | 1 | 0 | 0/0 | 100 | n/a | no |
| 4 | `OPT_TA03_ic_low_atr` | iron_condor | SPY | 1.59% | -0.24% | -0.09% | 6 | 0 | 6/0 | 0 | n/a | no |
| 5 | `OPT_TA06_pcs_pullback_dry` | put_credit_spread | SPY | 0.76% | -1.06% | -0.31% | 10 | 0 | 8/1 | 13 | n/a | no |
| 6 | `OPT_TA01_csp_range` | cash_secured_put | SPY | 0.66% | -0.54% | -0.20% | 7 | 0 | 6/1 | 0 | n/a | no |
| 7 | `OPT_TA02_pcs_compress` | put_credit_spread | SPY | 0.45% | -0.06% | -0.02% | 3 | 0 | 3/0 | 0 | n/a | no |
| 8 | `OPT_TA12_cash` | cash | SPY | 0.00% | 0.00% | 0.00% | 0 | 0 | 0/0 | 0 | n/a | no |
| 9 | `OPT_TA08_ccs_overbought` | call_credit_spread | SPY | -0.06% | -0.57% | -0.12% | 3 | 1 | 2/0 | 0 | n/a | no |
| 10 | `OPT_TA07_csp_hv_range` | cash_secured_put | SPY | -0.16% | -0.74% | -0.16% | 5 | 0 | 3/1 | 18 | n/a | no |
| 11 | `OPT_TA10_csp_sma200_dry` | cash_secured_put | SPY | -0.20% | -1.26% | -0.40% | 16 | 0 | 12/3 | 8 | n/a | no |
| 12 | `OPT_TA05_pp_climax` | protective_put | SPY | -1.04% | -3.19% | -0.74% | 1 | 0 | 0/0 | 100 | n/a | no |

**Book delta (approx):** sum_end=438.3725142330807 · mean_end=36.53104285275673 · label=`approx_bs_delta_book`

---
_Generated 2026-07-22T11:46:57.464988+00:00 · paper only · VIRTUAL_
**Data sources:** `{"QQQ": "yahoo", "SPY": "yahoo", "VIX": "yahoo", "VIX3M": "yahoo"}`
