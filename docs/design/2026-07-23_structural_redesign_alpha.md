# Design — Structural redesign for residual alpha + generalization

**Date:** 2026-07-23  
**Status:** APPROVED (plan rev.2) — implementation in progress  
**Modules:** STR-01 … STR-06, FEA-04, VAL-03 (reuse), DAT-04 (PIT-first path)  
**Product focus:** **ALPHA-PORTABLE** (STYLE-US is control only)

---

## 1. Intent

Stop patching `turbo_highvol_*` knobs. Confirm **structural limiters** (P1–P10), then build a **decoupled L0/L1/L2** path that can claim residual alpha vs a style clone, not CAGR vs SPY alone.

Success this cycle ≠ high CAGR.  
Success = correct root-causes + redesign that attacks them + honest falsification.

---

## 2. Structural problems (hypotheses)

| ID | Name | Redesign lever |
|----|------|----------------|
| P1 | Universe ↔ strategy coupled | L0 ≠ L1 |
| P2 | SPY/QQQ benches don’t falsify style | Style clone + PIT EW primary |
| P3 | US regime overfit by design | Invariant feats + local regime + FROZEN geo |
| P4 | Labels = barrier hit, not beat-style | Residual / beat_style meta |
| P5 | Absolute OHLC features | Rank/z-score SSOT (FEA-04) |
| P6 | Vanity CAGR gates | Residual + block consistency + DSR |
| P7 | PIT not default train/OOS | PIT-first on redesign path |
| P8 | One product for two theses | STYLE-US vs ALPHA-PORTABLE |
| P9 | Options/mega zoo distraction | Freeze this month |
| P10 | Platform debt | Single `run_redesign_eval` contract |

### Diagnostic confirmation (S1)

| ID | Confirmed if |
|----|----------------|
| P1✓ | Style clone captures ≥60% of baseline excess vs SPY **or** residual Sharpe (baseline−clone) ≤ 0.15 |
| P2✓ | Baseline excess vs PIT EW < 0 on long window |
| P3✓ | FROZEN ES/DE: excess A1 < 0 and/or MDD > 1.5× US |
| P4✓ | High corr ML signal vs rule trend/mom on same L0 |
| P5✓ | Absolute features worse on transfer than REL/rank |

### Redesign gates (S3–S4) — frozen

| Gate | Threshold |
|------|-----------|
| R1 Residual | Excess vs style clone > 0 on modern **and** early US |
| R2 PIT | Excess vs PIT EW ≥ −1 pp CAGR; delist on |
| R3 Costs | R1 holds at 1× costs |
| R4 Geo | FROZEN ≥1 of {ES,DE}: non-collapse + preferably beat local style/cash-blend |
| R5 Complexity | ≤1 model + 1 meta; DSR if >12 configs |
| R6 Honesty | Invariant features only on ALPHA path |

### Exit labels

`STRUCTURAL_DIAGNOSIS_ONLY` | `STYLE_US_BOOK` | `US_RESIDUAL_CANDIDATE` | `PORTABLE_ALPHA_CANDIDATE`

---

## 3. Target architecture (ALPHA-PORTABLE v0)

```
as-of t
  L0  Membership (PIT) + liquidity + optional vol bucket (NOT the alpha claim)
  L1  Cross-sectional score: invariant ranks/z only; train label = beat_style / residual H
  L2  Portfolio: top-K / vol target; portable risk overseas
Eval (always dual):
  strategy vs style_clone(same L0, dumb L1)
  vs PIT EW / DVW
  vs local index cash-blend
  blocks: early | modern | crisis
  geo: US train → foreign FROZEN (no foreign retrain)
```

**Baseline `turbo_highvol_minalloc`:** STYLE-US **control** only. Do not retune its knobs in this program.

---

## 4. Feature contract (FEA-04 draft)

**Allowed on ALPHA path:** scale-free / rankable series only, e.g.

- `atr_norm`, `rsi_*`, `dist_sma_*`, `volatility_20`, `volume_ratio`, `volume_zscore`, `ret_1m`
- cross-sectional ranks of the above within day (0–1)

**Banned on ALPHA path:** raw `open`, `high`, `low`, `close` levels (and any absolute price).

Legacy STYLE-US may keep M2 absolute for control runs.

---

## 5. Label contract (STR-03 draft)

- **Barrier labels:** exits / risk only (optional).  
- **Promotion / meta train label:**  
  `y_residual = 1{ r_i,t→t+H − r_style_clone,t→t+H > 0 }`  
  or continuous excess for ranking models.

---

## 6. Implementation map (PRs)

| PR | Deliverable |
|----|-------------|
| PR1 | This design + docs/11 STR-* |
| PR2 | `style_clone.py`, `alpha_attribution.py`, S1 harness |
| PR3 | `portable/cs_features.py`, `residual_labels.py` |
| PR4–5 | L0/L1/L2 + `run_redesign_eval.py` |
| PR6 | Falsification suite |
| PR7 | Product bifurcation + FINAL scorecard |

---

## 7. Out of scope

- Optuna on turbo  
- Foreign retrain  
- OPRA claims from proxy_bs  
- New options/equity mega matrices  

---

## 8. Verification

```powershell
python -m pytest tests/test_style_clone_unit.py tests/test_alpha_attribution_unit.py tests/test_portable_cs_unit.py -q
python scripts/run_style_clone_gap.py --help
```

Heavy WF runs only when user requests full S1 numbers.

---

## 9. Disclaimer

Research software. Not financial advice. Residual gates may reject the redesign; that is a valid scientific outcome.
