# Design: Options portfolio + meta-labeling long-history study

**Date:** 2026-07-22  
**Horizon:** multi-year from **2005/2010** (EODHD) through present  
**Capital:** VIRTUAL  
**Loop:** design → implement → verify (loop-engineering)

---

## 1. Problem

Prior studies overfit narrative biases:

- Single-name leverage (NVDA×2) and index leverage (QQQ×2) dominate mean returns  
- Short windows (2022–2025)  
- No **portfolio construction** or **meta-label selection** across sleeves  

User wants:

1. **Thousands** of options strategy variants  
2. **Meta-labeling** (primary signal + secondary take/skip or size)  
3. **Portfolio management** (how the model chooses the best mix)  
4. **Long history** including **2010 and earlier** (EODHD has SPY/VIX from 2005)  
5. **No** NVDA×2 / QQQ×2 style lottery sleeves  

---

## 2. Goals

| Goal | Metric |
|------|--------|
| Diversified options sleeve zoo | ≥1 000 combinatorial defined-risk / budgeted-debit configs |
| Fair portfolio selection | Walk-forward yearly: train meta on past windows, allocate next year |
| Long hist | Calendar years **2010–2025** (+ optional 2005–2009 stress) |
| Benchmarks | SPY BH, QQQ BH, equal-weight cash+SPY 60/40, **not** levered single-name |
| Honesty | Labels `proxy_bs\|vix_surface` / `eodhd_eod` underlyings; no OPRA claim |

**Non-goals:** live trading; UnicornBay full chain (403); claiming OPRA edge.

---

## 3. Architecture

```
paper_live/
  portfolio/
    meta_label_selector.py   # features → take/skip / size
    sleeve_portfolio.py      # allocate capital across sleeves
  options/
    grid_zoo.py              # combinatorial strategy generator
scripts/
  build_options_grid_zoo.py
  run_options_portfolio_meta_study.py
reports/options_portfolio_meta/
  latest/SUMMARY.md
  walk_forward.json
```

### 3.1 Strategy grid (anti-bias rules)

**Allowed kinds:**  
`put_credit_spread`, `call_credit_spread`, `iron_condor`, `call_debit_spread`,  
`put_debit_spread`, `long_call` (budget-capped ≤10%), `long_put` (budget ≤5%),  
`covered_call`, `cash_secured_put`, `collar`, `cash`

**Underlyings (liquid, diversified — no lone-NVDA product):**

- Index: SPY, QQQ, IWM  
- Mega equal-weight baskets via multi-run average OR single names with **max weight cap 15% in portfolio**  
- Never: leverage multiplier > 1.0 on any single name sleeve  
- Never: `pmcc` with leverage narrative; optional PMCC only if budget-capped and diversified  

**Grid axes:**

| Axis | Values |
|------|--------|
| dte_days | 21, 30, 45, 60 |
| otm_pct | 0.03, 0.05, 0.08, 0.10 |
| wing_otm_pct | 0.10, 0.12, 0.15 |
| gate | none, uptrend, sma200, range, vrp_above, volume_dry |
| budget_frac (debit) | 0.05, 0.08, 0.10 |

Product: kinds×underlyings×axes → **thousands** (cap at 3000 for runtime; sample 1500 if needed).

### 3.2 Evaluation protocol

1. **Phase A — Sleeve backtests**  
   For each strategy × calendar year 2010…2025 (and 2005–2009 optional):  
   total_return, max_dd, n_opens, hard_kill  

2. **Phase B — Meta-label dataset (causal)**  
   At year-end *t* for each sleeve *s*:  
   - Features (≤ last day of t): VIX, VIX3M/VIX, HV20 SPY, rolling sleeve ret/sharpe over prior 1–3 years, kind one-hot, otm, dte, gate  
   - Label: **1 if return of s in year t+1 > 0** (or > cash), else 0  
   Train XGBoost meta on years through t (expanding), predict year t+1  

3. **Phase C — Portfolio construction (year t+1)**  
   Among sleeves with meta_prob ≥ τ (default 0.55):  
   - Rank by meta_prob × inverse |max_dd|  
   - Take top K (default 8)  
   - Weights: inverse-vol or equal weight  
   - **Caps:** max 20% per underlying, max 40% short-premium family, max 30% long-premium  
   - Residual → cash  

4. **Phase D — Report**  
   Portfolio equity path annual; vs SPY/QQQ; hit rate of meta; concentration; kill list of lottery sleeves  

### 3.3 Primary vs meta (López de Prado style)

- **Primary:** structure/side implied by kind (PCS bullish, CCS bearish, IC neutral, long call bullish)  
- **Meta:** take-or-skip (and soft size ∈ {0, 0.5, 1.0}× target weight)  

---

## 4. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data | EODHD EOD underlyings + VIX | Token available; long history to 2005 |
| Option marks | proxy_bs + vix_surface | No UnicornBay options sub |
| Ban levered single-name | hard filter | User requirement |
| Selection | expanding WF yearly meta | No look-ahead |
| Scale | up to 3000 sleeves, chunked | 5h compute budget |
| Portfolio | top-K meta + risk caps | Diversification over lottery |

---

## 5. PR Plan

1. **PR-PM-1** — `grid_zoo.py` + build script + ban filters  
2. **PR-PM-2** — sleeve batch runner multi-year EODHD (chunked, resume)  
3. **PR-PM-3** — meta_label_selector + features/labels pure functions + tests  
4. **PR-PM-4** — sleeve_portfolio allocate + walk-forward orchestration  
5. **PR-PM-5** — reports + SUMMARY + docs/11 history  
6. **PR-PM-6** — long run 2010–2025 (background); smoke 2018–2024 first  

---

## 6. Verification

```powershell
python -m pytest tests/ -q --tb=short -k "portfolio_meta or options_grid or metalabel"
python scripts/build_options_grid_zoo.py --max 2000 --out paper_live/cloud/zoo_options_grid.json
python scripts/run_options_portfolio_meta_study.py --from 2010 --to 2025 --max-strategies 500 --smoke
# full:
python scripts/run_options_portfolio_meta_study.py --from 2010 --to 2025 --max-strategies 2000
```

---

## 7. Success criteria

1. ≥1000 strategies generated without NVDA×2 / QQQ×2  
2. WF portfolio report 2010–2025 with annual returns vs SPY  
3. Meta hit-rate and take/skip counts documented  
4. Honest labels; tests green  
5. Evidence that selection ≠ single lottery year (max upside share < 50% preferred)  

---

## 8. Open risks

- Thousands × 16 years of full options replay may exceed 5h → chunk + resume + optional year-subset for first pass  
- Meta with few years of history early (2010–2012) is weak → min train years = 3  
- proxy_bs bias remains without OPRA  
