# Spec — Sistema A: ORB + HTF bias (falsificación)

**Date:** 2026-07-27  
**Status:** APPROVED (auditor) · implementation v1  
**Product:** research kill-test only · paper freeze `turbo_highvol_minalloc` **unchanged**  
**Origin:** social batch `20260727c` pattern #1 (HTF→breakout); not ICT/MCP STRONG as primary specs.

---

## 1. Objective

Falsify (KILL) or retain HOLD a **fully mechanical (G1)** long-only system inspired by “HTF bias + opening-range break”, with:

- fixed knobs (no re-fit), OOS year blocks + full stitch 2010–2025  
- realistic costs  
- Monte Carlo path/sizing  
- SPY/QQQ (+ cash-aware if idle)  
- scorecard **KILL | HOLD** only (**no ADVANCE** this cycle)

---

## 2. Data honesty

| Mode | Data | Real session ORB? | v1 |
|------|------|-------------------|-----|
| **`orb_htf_daily_proxy_v1`** | EOD OHLCV in `data/` | **No** — prior-day high/low break + dual-MA | **PRIMARY** |
| **`orb_15m_true_v1`** | 1m/5m/15m 09:30–09:45 ET | Yes | **BLOCKED** until minute bars exist |

Reports must set `data_label=eod_proxy`. Do not claim 15m futures ORB edge.

---

## 3. Frozen rules — `orb_htf_daily_proxy_v1`

### Universe / calendar

| Field | Value |
|-------|--------|
| Primary | `universe_longhist100.txt` limit **50** |
| Controls | SPY-only, QQQ-only |
| Benchmark | SPY BH; QQQ secondary; cash blend if mean weight `w < 0.95` |
| Windows | Full 2010–2025; Early 2010–2017; Modern 2018–2025; Stress 2022 |
| Costs | commission **0.10%** + slippage **0.05%** / side |
| Capital | 100_000 virtual |

### Signal (causal, close ≤ t)

```
bias_long[t] = (close[t] > SMA50[t]) AND (close[t] > SMA200[t])   # A0
# A1: bias_long[t] = close[t] > SMA200[t] only

orb_high[t] = high[t-1]
orb_low[t]  = low[t-1]

signal_long[t] = bias_long[t] AND (close[t] > orb_high[t]) AND (close[t] > open[t])
score = (close/orb_high - 1) / max(atr_norm, 1e-6)
```

No retest, FVG, ICT discretion. Long only.

### Exit / risk

| Param | Value |
|-------|--------|
| Hard stop | ~ `entry − max(hard_stop_pct·P, 1.5·ATR)` (engine); orb_low exact stop not wired — documented approx |
| Take profit | **2.0 R** vs hard-stop distance |
| Time stop | **10** bars |
| Risk / trade | **0.75%** equity (`risk_per_trade_pct`); MC also 0.5% / 1.0% |
| max_positions | 8 · max_position_pct 0.12 · min_alloc 1.5% |
| Vol target | **off** when risk_per_trade set |
| Extra regime map | **off** (dual-MA is the HTF filter) |

Notches only: **A0_base**, **A1_sma200_only**.

### Execution note

Research engine marks/fills on **daily close** path (existing portfolio backtest). Label `execution_mode=daily_close_research` — not open t+1 brokerage realism.

---

## 4. Kill / HOLD (no ADVANCE)

KILL if any:

- full path n_trades < 80  
- full MaxDD < −65%  
- full CAGR ≤ 0  
- full excess SPY ≤ 0 **and** Sortino < 0.4  
- full CAGR ≤ 10% **or** excess SPY ≤ 0 (primary longhist50 gate from plan)  
- early 2010–17 fails while modern 2018–25 only looks good (CAGR early < 0 or excess SPY early < −5 pp) → **window cherry**  
- MC Sortino p5 < 0.1 or P(MDD < −60%) > 0.25  

HOLD: middling full path, weak MC/early without total collapse.  
**ADVANCE_*** forbidden this experiment. Paper freeze unchanged.

---

## 5. Out of scope v1

ICT STRONG, MCP/Fable workflow, shorts, options, leverage, ORB minute grids, more YouTube.

**System B** only if A = KILL or HOLD-tie (one of three options in social plan).

---

## 6. Artifacts

- Strategy: `trad_research/orb_htf.py` + register in `strategies.py`  
- Runner: `scripts/run_orb_htf_falsification.py`  
- Out: `reports/redesign/orb_htf_falsification_v1/`  
- Tests: `tests/test_orb_htf_unit.py`
