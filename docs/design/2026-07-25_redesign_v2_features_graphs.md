# Redesign v2 — features, graph math, multi-window stress (16h research loop)

**Date:** 2026-07-25  
**Modules:** FEA-02 (ext features), STR-R2 (rule + hybrid strategies), VAL-02 (screen/confirm + full)  
**Product:** research only — paper freeze `turbo_highvol_minalloc` **not** auto-changed  

---

## 1. Why redesign (evidence, not vibes)

| Finding | Source | Implication |
|---------|--------|-------------|
| highvol k100 full 2018–25: high CAGR, path MDD often ≤−50% | Loop E/F | Edge is fat tails; hard to “MDD-tame” without killing edge |
| Soft-ban full-window “PASS” dies when purged | Loop G | Name lists from full OOS are leakage |
| longhist 2010–25 limit80 FAIL CAGR; limit54 accidental PASS | longpath_2010 | Capacity/limit sensitivity, not proven edge |
| Screen 2010–17 strong, confirm 2018–25 FAIL for limits 40/50/60 | universe_limit_sc | **Overfit to early window** on liquid longhist |
| Style residual still positive on highvol modern | S1 style gap | ML/timing residual exists but STYLE-US coupled |

**Frozen rule:** Do not retune soft-ban or invent limit=54 as “the answer.”  
**Redesign rule:** Change **signal structure** (features + ranking + risk geometry), not ticker banlists.

---

## 2. Intent (success metrics)

**Primary gates (confirm 2018–2025, pre-registered):**

- CAGR **> 10%**
- max drawdown **≥ −65%**
- n_trades ≥ 80
- excess vs SPY total **> 0** (soft prefer; report always)

**Screen (2010–2017):** used only for ranking freeze of top-K candidates (≤3), never for claiming edge alone.

**Full path 2010–2025:** report always; research PASS requires **confirm** gates; full path is secondary honesty.

**Kill:** any candidate that passes screen only and fails confirm (same as universe_limit study).

---

## 3. New mathematics & features (causal)

All rolling ops use past+present bars only.

| Family | Feature / object | Use |
|--------|------------------|-----|
| Residual mom | `resid_ret_20` = ret_20 − β̂·mkt_ret_20 (β from 60d rolling OLS vs SPY if available; else 0) | Score for `r2_residual_mom` |
| Risk-adj mom | `mom_sharpe_20` = ret_20 / (vol_20 + ε) | Score / filter |
| Path geometry | `dd_from_peak_252` = close/rollmax_252 − 1 | Avoid deep individual DD entries |
| Vol regime | `vol_of_vol_20` = std of daily vol proxy | Skip explosion |
| Trend stack | `trend_stack` = 1_{close>sma50} + 1_{sma50>sma200} + 1_{ret_1m>0} | Discrete score 0–3 |
| Graph (analysis) | Corr graph of trade co-occurrence / residual series | HTML network + hub scores — **not** live banlist |
| Graph (soft signal) | `avg_peer_corr_proxy` optional later | Out of v1 entry loop if needs panel |

**Banned:** fundamentals that reintroduce growth-hard FAIL; soft-ban from full-window audit.

---

## 4. Strategy zoo (structurally distinct)

| id | Class | Thesis |
|----|-------|--------|
| `turbo_highvol_minalloc` | Control ML | Frozen control |
| `r2_residual_mom` | Rules | Residual momentum vs market |
| `r2_mom_sharpe` | Rules | Risk-adjusted momentum |
| `r2_trend_stack` | Rules | Discrete trend quality |
| `r2_defensive_vt` | Rules+risk | Tight VT + hard regime + mild mom |
| `r2_rsi_reclaim` | Rules | RSI reclaim in uptrend (mean-rev entry) |
| `turbo_strict` | Control | Lower beta shell |
| `champion_ml` | Control | Classic champion |

Universes (pre-reg):  
- `universe_longhist2010_pass.txt` limit ∈ {50, 80}  
- Optional arm: `universe_highvol80_2010_pass.txt` limit 50 (stress only)

---

## 5. Multi-hour loop protocol

```
for hours remaining:
  for universe_arm:
    for strategy in zoo:
      run screen 2010-17
      run confirm 2018-25
      stitch full
      score confirm gates
  freeze top-3 by confirm honest_score
  build graphs + HTML dashboard
  write PROGRESS.json
stop when hours exhausted or all configs done
```

**honest_score** = 2·confirm_cagr + 1·sortino + 0.5·max(0, excess_spy) − 2·max(0, −0.50 − mdd)

No auto freeze change. Shadow candidate only if confirm PASS + full MDD ≥ −65%.

---

## 6. Deliverables

| Path | Content |
|------|---------|
| `trad_research/redesign_v2/` | features_ext, graph_math, (strategies in strategies.py) |
| `scripts/run_redesign_v2_mega.py` | multi-hour orchestrator |
| `reports/redesign/redesign_v2/` | PROGRESS, SUMMARY, DECISION, graphs HTML |
| unit tests | pure feature + graph + ranking |

---

## 7. Honesty

Research software. Past OOS ≠ future. Paper freeze unchanged unless human ADVANCE.

Research only. Not financial advice.
