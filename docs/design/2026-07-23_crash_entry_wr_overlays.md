# Design: Crash / oversold entry overlays + win-rate levers

**Date:** 2026-07-23  
**Status:** Research implement (not live promotion)  
**Module:** `trad_research/crash_entry.py` + backtest / strategy_runner wiring  
**Study:** `scripts/run_crash_entry_mega_study.py` → `reports/redesign/crash_entry_mega_study/`

---

## Problem

`#1` bake-off book `turbo_highvol` (highvol80, long-only cash) shows **high total return** with **low win rate (~35%)**: many `hard_stop` losses near −11%, few large winners. In **index crashes** (esp. 2020), strict dual-golden regime + trend gates **delay re-entry** until risk-on flips back.

Goals (research only):

1. Raise win rate via **filters**, not random turbo knob retunes.
2. Enter **earlier in deep selloffs** using **causal** index oversold metrics (SPY/QQQ RSI, DD).
3. Stress many configs on full OOS + crash windows.

---

## Non-negotiables

- **Causal / fail-closed:** crash flags use only bars ≤ t; missing index data → crash off.
- **No look-ahead** on features or labels.
- **No OPRA / short-vol claims.** Equity long-only overlays only.
- **STYLE-US** (`turbo_highvol_minalloc`) remains paper control unless promotion gates pass.
- Do **not** claim live edge from this study.

---

## Design

### A) Crash entry map (`crash_entry.py`)

Index sources: `data/SPY_history.csv`, `data/QQQ_history.csv` (first available / any / all).

Metrics (all causal):

| Metric | Definition |
|--------|------------|
| RSI(14) | Wilder EWM — same `_wilder_rsi` as `features.py` |
| DD from peak | `close / cummax(close) - 1` |
| below SMA50/200 | optional extra gates |
| RSI rising | `rsi_t > rsi_{t-1}` (recovery mode) |

Modes: `rsi`, `dd`, `rsi_or_dd`, `rsi_and_dd`, `rsi_recover`.

**Actions on crash day:**

- `relax_regime`: allow entries even if hard index regime is risk-off  
- Lower `crash_min_confidence` for signal union  
- Soft trend (SMA50 \| SMA20) instead of hard SMA50  
- Looser `min_dist_sma200`  
- Optional score boost for ranking  

Signal path: base ML mask **OR** crash-relaxed mask on crash days only (`apply_crash_signal_overlay`). Regime path: `BacktestConfig.crash_relax_regime` in day loop.

### B) Win-rate levers

| Lever | Mechanism |
|-------|-----------|
| ATR tight | `max_atr_pct_tight` / `max_atr_pct_entry` skip wild names |
| Hard-stop cooldown | `hard_stop_cooldown_days` block same ticker after `hard_stop` |
| Soft trend non-crash | SMA50 \| SMA20 outside crash |
| Meta conf floor | `min_meta_conf` when meta model present |

### C) Strategy variants

| Name | Notes |
|------|--------|
| `turbo_highvol_crash_rsi` | highvol + RSI thr 30 overlay |
| `turbo_highvol_crash_rsi_wr` | crash RSI + WR pack |
| `turbo_highvol_minalloc_crash_rsi` | minalloc + crash RSI + cooldown |

### D) Mega study

- Train base highvol **once per OOS year**, evaluate many overlay configs without retrain.
- Windows: full OOS; 2018Q4; 2020-02–04; 2022-01–10.
- Metrics: return, CAGR, Sharpe, Sortino, MDD, WR, n_trades, PF, excess vs SPY; crash-slice return/DD/entries.
- Rank: `composite_rank_score` (WR + residual vs SPY + crash survival).
- CLI: `--smoke` (CI), `--grid medium|full`.

---

## Verification

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest tests/test_crash_entry_unit.py -q --tb=short
python scripts/run_crash_entry_mega_study.py --smoke
```

---

## Explicit non-goals

- Live capital / broker deployment  
- Options short-premium with proxy marks  
- Claiming OPRA edge  
- Retuning vol_target / max_position without measurement  

---

## Files

- `trad_research/crash_entry.py`
- `trad_research/backtest.py` (cooldown, crash regime, ATR entry)
- `trad_research/strategy_runner.py` (map wiring)
- `trad_research/strategies.py` (variants)
- `scripts/run_crash_entry_mega_study.py`
- `tests/test_crash_entry_unit.py`
- `reports/redesign/crash_entry_mega_study/`
