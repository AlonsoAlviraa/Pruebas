# Design: Alt strategy loop — MDD attack (2026-07-23)

**Status:** Loop 1–2 complete — **success B met** (`dd35_vt80_yr` HOLD; 0 ADVANCE)  
**Product:** STYLE-US research (control remains `turbo_highvol_minalloc`)  
**Scope:** Risk overlays only — no ML retrain, no 1000-ticker expand, no geo retrain, no OPRA claims  
**Evidence window:** OOS 2018–2025, universe **n=40** highvol (full80 re-score left for user)

## Problem

Promotion Stage1 kills every curated sleeve: multi-year MDD ≈ −50% to −70%.  
Crash/WR overlays improve CAGR/WR but do not clear MDD. Paper freeze stays minalloc.

## Goals

1. Measurable multi-year MDD improvement vs minalloc baseline (target ≥10pp for research HOLD).  
2. Keep excess vs SPY positive and n_trades ≥50.  
3. Prefer ADVANCE; never auto-overwrite live freeze.

---

## Design choices

### Registered risk levers (`risk_levers.py`)

| lever_id | max_portfolio_dd | vol_target_scale | max_position_scale | peak_mode | dd_breach_size_scale |
|----------|------------------|------------------|--------------------|-----------|----------------------|
| baseline | 0.99 (off) | 1.0 | 1.0 | continuous | None (hard N/A) |
| dd_circuit_25 | 0.25 | 1.0 | 1.0 | continuous | None (hard block) |
| dd25_vt70 | 0.25 | 0.70 | 1.0 | continuous | None |
| dd20_vt60 | 0.20 | 0.60 | 1.0 | continuous | None |
| dd18_vt70_pos75 | 0.18 | 0.70 | 0.75 | continuous | None |
| **dd25_vt70_yr** | 0.25 | 0.70 | 1.0 | **yearly** | None |
| **dd25_vt70_soft** | 0.25 | 0.70 | 1.0 | continuous | **0.30** |
| **vt60_only** | 0.99 | 0.60 | 1.0 | continuous | None |
| **dd35_vt80_yr** | 0.35 | 0.80 | 1.0 | **yearly** | None |

Applied via `apply_risk_mdd_lever(strategy.backtest_overrides(), lever_id)` so vol/pos scales hit real strategy knobs.  
`dd_breach_size_scale` is **always** written from the lever (never inherits stale soft scale).

### Peak modes (mega study)

| Mode | `resolve_peak_equity_seed` | Behavior |
|------|----------------------------|----------|
| `continuous` | prior segment HWM | Multi-year peak seed; can create permanent-cash trap with hard block |
| `yearly` | always `None` | Within-year peak from capital; re-risk each January |

Helpers: `resolve_peak_equity_seed`, `update_peak_equity_state` in `risk_levers.py`.

### Soft breach (`backtest.dd_breach_size_scale`)

When DD ≤ −`max_portfolio_dd`:

- `None` → legacy **hard block** (no new entries that day)  
- float (e.g. 0.30) → **size_scale × rec** (recovery path; half-way soft scale is `elif`)

### Peak HWM ratchet (bugfix)

`peak_equity` is ratcheted on **end-of-day MTM** every bar (including full-book days when `can_enter` is false). Without this, DD circuit understated drawdown after long full-book rises.

### Breadth gate (`breadth_gate.py`)

- Causal: daily fraction of names with close > SMA50.  
- Fail-closed if `< min_names` valid SMAs or breadth NaN.  
- AND with `strict_dual_golden` hard map in mega study.  
- Same-bar causality class as index regime maps (not EOD→next-open lag).

---

## Grids

### `alt_mdd` (Loop1)

1. baseline  
2. dd_circuit_25  
3. dd25_vt70  
4. dd20_vt60  
5. dd18_vt70_pos75  
6. breadth40_dd25_vt70  
7. crash_rsi30_wr_dd25  

### `alt_mdd_v2` (Loop2)

1. baseline  
2. dd25_vt70_yr  
3. dd25_vt70_soft  
4. vt60_only  
5. dd35_vt80_yr  
6. breadth40_dd25_vt70_yr  

Default medium run: univ=40, OOS 2018–2025.

---

## Loop results (honest)

### Loop1 trap

Continuous peak + hard DD block after 2018 losses → **permanent cash** (no re-entry path). MDD “improved,” excess destroyed. Kill continuous hard-block as multi-year default.

### Loop2 success B (`dd35_vt80_yr`, n=40)

| Metric | baseline | dd35_vt80_yr |
|--------|----------|--------------|
| CAGR | 25.9% | **30.4%** |
| MDD | −70.4% | **−45.6%** (+24.8pp) |
| excess vs SPY | +243% | **+448%** |
| n_trades | 583 | 607 |
| Promo | KILL mdd | **HOLD** (MC tail / DSR block ADVANCE) |

**Note:** Mega `equity_metrics` Sharpe (~0.74) vs promo Stage1 Sharpe (~0.83) use different pipelines — do not mix. MDD/n/CAGR align.

---

## Validation

- Unit: breadth, alt_mdd grids, peak helpers, soft-breach path, peak full-book ratchet  
- Multi-year mega → equity dumps → promotion scorecard  
- Success A/B/C: **B met**; A not met (0 ADVANCE)

## Non-goals

- Kitchen-sink fundamentals  
- Retuning crash RSI thresholds on OOS  
- Claiming paper/live edge without ADVANCE + human freeze copy  
- Treating n=40 HOLD as full80 paper-ready  

## PR-style file list

- `trad_research/risk_levers.py`  
- `trad_research/breadth_gate.py`  
- `trad_research/backtest.py` (`dd_breach_size_scale`, EOD peak ratchet)  
- `scripts/run_crash_entry_mega_study.py` (`--grid alt_mdd|alt_mdd_v2`)  
- `tests/test_breadth_gate_unit.py`, `tests/test_alt_mdd_unit.py`  
- `reports/redesign/alt_scout/*`, `reports/redesign/alt_loop_2026-07-23/*`  
- `docs/11_plan_implementacion_modular.md` history row  

Research only. Not financial advice.
