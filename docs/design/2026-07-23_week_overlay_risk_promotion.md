# Design: Week plan — highvol80 overlays + promotion + MDD risk lever

**Date:** 2026-07-23  
**Status:** Research implement (not live promotion)  
**Modules:** Crash-entry mega harness extension · PROMO-01 · RSK MDD lever · paper freeze path  
**Reports:** `reports/redesign/week_plan_2026-07-23/`

---

## Context

Mega study on **universe_limit=40** recommended:

- Prefer WR pack / selected crash sleeves for research, **not** 1000-ticker expansion or fundamental kitchen-sink.
- Do **not** promote on n=40 alone.
- STYLE-US paper control remains `turbo_highvol_minalloc` unless multi-stage promotion **ADVANCE**.

This week plan re-runs a **curated 3–5 config** set on **full `universe_highvol80.txt`**, then funnels candidates through promotion gates and a **single** MDD risk A/B.

---

## Phases

### Phase A — Evidence on real universe (highvol80)

**Universe:** full file (`--universe-limit 0` = no cap).  
**Regime:** `strict_dual_golden`.  
**Costs:** commission 0.10%, slippage 0.05% (existing `BacktestConfig` defaults).  
**OOS:** 2018–2025 when data allows (smoke: short window).

| # | Config id | Base | Overlay |
|---|-----------|------|---------|
| 1 | `turbo_highvol_minalloc__baseline` | minalloc | none (STYLE-US control) |
| 2 | `turbo_highvol_minalloc__crash_rsi30_wr` | minalloc | crash RSI thr=30 + WR pack |
| 3 | `turbo_highvol__wr_pack` | highvol | WR pack only |
| 4 | `turbo_highvol__crash_dd15` | highvol | crash DD ≤ −15% |
| 5 | `turbo_highvol__crash_rsi_or_dd15` | highvol | crash RSI30 **or** DD−15% |

**CLI:**

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/run_crash_entry_mega_study.py --grid week --universe-limit 0 --out reports/redesign/week_plan_2026-07-23/phase_a
# smoke:
python scripts/run_crash_entry_mega_study.py --grid week --smoke --out reports/redesign/week_plan_2026-07-23/phase_a_smoke
```

`--smoke` still forces a tiny year/universe for CI unless overridden; prefer:

```powershell
python scripts/run_week_plan_study.py --smoke
```

### Phase B — Promotion scorecard

Load Phase A equity CSVs + trades into `trad_research.promotion.evaluate_candidate`.

- Style residual: vs minalloc baseline equity when present.
- Labels: `KILL` | `HOLD` | `ADVANCE_STYLE` (no ALPHA claim from these long-only overlays).
- Expect **0 ADVANCE** possible (deep MDD / MC honesty).

```powershell
python scripts/run_promotion_scorecard.py --from-configs-dir reports/redesign/week_plan_2026-07-23/phase_a/configs --style-name turbo_highvol_minalloc__baseline --out reports/redesign/week_plan_2026-07-23/phase_b --smoke
```

### Phase C — One risk experiment (MDD attack)

**Primary lever only:** portfolio **DD circuit** (`max_portfolio_dd=0.25`, soft scale halfway) on **minalloc baseline**.

| Arm | Description |
|-----|-------------|
| A control | `turbo_highvol_minalloc` baseline (`max_portfolio_dd=0.99` off) |
| B treatment | same + `max_portfolio_dd=0.25`, `dd_soft_scale=0.50` |

Not a multi-knob retune. Vol-target scale is available as a pure-function helper but **not** the registered week A/B.

### Phase D — Decision freeze path

- Live paper freeze remains `paper_live/config/strategy_freeze.json` (`turbo_highvol_minalloc`) unless a **human** copies a candidate after review.
- On non-control **ADVANCE**: orchestrator writes a **report-only** candidate under  
  `reports/redesign/week_plan_*/phase_d_freeze/strategy_freeze_candidate.json`  
  (`write_shadow_candidate=True`). **Never** auto-writes under `paper_live/config/`.
- On 0 ADVANCE or all-control-like ADVANCE: write `DECISION.md` only; paper stays pure control.
- Multi-ADVANCE: prefer first **non-control-like** name (not order-dependent keep on baseline-first lists).

### Peak carry (Phase C / mega years)

- Mega harness carries `peak_equity` high-water mark across OOS calendar years via  
  `BacktestConfig.peak_equity_seed` so `max_portfolio_dd` is continuous multi-year, not reset each Jan 1.

Kill criteria (any → no freeze promotion):

- Stage 0 pathology / invalid equity  
- Stage 1 Sortino/Sharpe/MDD fail  
- Residual vs style ≤ 0 when residual required for claim  
- Stage 2 MC fail (or diagnostic-only trade count)  
- Geo retrain / OPRA / 1000-universe out of scope experiments  

---

## Non-negotiables

- Causal crash flags (fail-closed); ATR tight **non-crash only**.
- Equity long-only; no OPRA / short-vol claims.
- No look-ahead; labels never as features.
- No fabricated prices; no delete of user data/models.
- Spanish OK for research narrative; code in English.

---

## Explicit out of scope

- 1000-ticker universe expansion  
- New financial/fundamental features  
- Geo retrain (ES/DE FROZEN)  
- Random turbo knob retunes  
- Claiming live/OPRA edge  

---

## Files

| Path | Role |
|------|------|
| `docs/design/2026-07-23_week_overlay_risk_promotion.md` | This design |
| `scripts/run_crash_entry_mega_study.py` | `--grid week`, `--universe-limit 0` |
| `trad_research/risk_levers.py` | Pure MDD lever apply helpers |
| `scripts/run_promotion_scorecard.py` | `--from-configs-dir` Phase B |
| `scripts/run_week_plan_study.py` | Orchestrator A→B→C→D |
| `tests/test_risk_levers_unit.py` | Unit tests |
| `reports/redesign/week_plan_2026-07-23/SUMMARY.md` | Decision |

---

## Verification

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest tests/test_risk_levers_unit.py tests/test_crash_entry_unit.py tests/test_promotion_unit.py -q --tb=short
python scripts/run_week_plan_study.py --smoke
python -m pytest tests/ -q --tb=short
```

Orchestrator requires explicit `--smoke` or `--full` (no accidental heavy default).

Full OOS (heavy):

```powershell
python scripts/run_week_plan_study.py --full --universe-limit 0 --first-oos 2018 --last-oos 2025
```
