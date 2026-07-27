# Spec — Falsification Framework v1 (research evaluation OS)

**Date:** 2026-07-27  
**Status:** APPROVED · scaffold implementation  
**Product:** evaluation infrastructure only · paper freeze `turbo_highvol_minalloc` **unchanged**  
**Module ID:** FALSIFY-01

---

## 1. Purpose

Provide a **pre-registered kill/hold evaluation OS** for research candidates so future cycles cannot advance on in-sample noise, leakage, or style clones of the baseline book.

This is **framework infrastructure**, not a multi-week research campaign and not social/YouTube scraping.

---

## 2. Pre-registered gates (v1)

| Gate | What | Fail → |
|------|------|--------|
| **Purged / combinatorial CV** | Time-ordered folds; purge bars around test; embargo after test | KILL if fold protocol violated or (when scores provided) OOS collapse |
| **Embargo** | `embargo_bars` (or pct) after each test fold excluded from train | Structural |
| **DSR** | Bailey–López de Prado Deflated Sharpe with **required** `n_trials` (from ResearchMemory when available). DSR = Φ[(SR̂−SR₀)/se], SR₀=E[max SR\|N] under null — not P(SR>0) without selection. Default `dsr_min=0.05` is a soft floor (binds near/below SR₀); pair with `sharpe_pathology_abs` | KILL if DSR < threshold or pathology SR |
| **Leakage scan** | Future peek, label-in-features, index order, train/test overlap, null/const features | KILL on any high-severity finding |
| **Book correlation** | Candidate daily equity returns vs baseline book (e.g. turbo_strict / minalloc) | KILL if corr > threshold (style clone) |
| **Costs / capacity** | Cost multiplier stress + ADV/dollar-volume caps | KILL if edge dies under stressed costs |
| **MC sequence** | Optional shuffle/bootstrap via existing `monte_carlo` | HOLD if borderline; KILL if path fragility extreme |
| **Verdict** | **KILL \| HOLD only** | ADVANCE deferred (optional later; not produced by v1) |

Default verdict ceiling: **HOLD**. ADVANCE remains a separate promotion concern (`promotion.py`), not this framework’s output.

---

## 3. Idea sources

| Allowed | Forbidden as edge source |
|---------|---------------------------|
| Peer-reviewed / industry quant literature (LdP, AFML, AQR-style notes) | Retail YouTube “secret setup” as primary edge claim |
| Residual / portable CS research already in-repo | Re-scraping social for new unfalsified patterns |
| Controlled synthetic tests of the framework | Fabricated prices labeled as real |
| Pre-registered hypotheses with n_trials logged | Silent multi-test without DSR / memory |

ORB kill already completed — do not re-run heavy ORB studies under this scaffold.

---

## 4. First research line **after** framework (spec later)

**Residual improvement on turbo_strict / minalloc book** — improve portable residual or reduce book correlation while preserving OOS residual excess. Not implemented in this PR; framework only (one synthetic residual demo OK in smoke).

---

## 5. Package map

```
trad_research/falsify/
  config.py          FalsifyConfig
  purged_cv.py       Combinatorial purged K-fold + embargo
  deflated_sharpe.py Bailey-style DSR (n_trials required)
  leakage.py         Leakage detectors
  book_corr.py       Equity-path correlation vs baseline book
  feature_store.py   Causal feature registry + materialize
  regime_features.py Vol / trend regime named features
  costs_capacity.py  Cost stress + capacity helpers
  research_memory.py JSONL trial ledger (n_trials for DSR)
  scorecard.py       FalsifyReport KILL|HOLD
  pipeline.py        run_falsify_suite orchestration
```

Reuse (do not reinvent weakly): `zoo.py`, `promotion.py`, `monte_carlo.py`, `walk_forward.py`, `regime.py`, `alpha_attribution.py`, `portable/`.

---

## 6. Non-goals

- Full LOB / microstructure platform  
- Heavy Nautilus / Qlib / vectorbt integration  
- New YouTube scraping  
- Changing paper freeze  
- Multi-year residual turbo campaign  

---

## 7. Verification

```powershell
python -m pytest tests/test_falsify_framework_unit.py -q --tb=short
python -m trad_research.falsify
```

---

## 8. Disclaimers

Research software. Past synthetic or historical paths do not guarantee future results. No ADVANCE claim from framework v1.
