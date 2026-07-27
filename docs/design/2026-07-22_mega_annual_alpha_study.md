# Design: Mega annual alpha study (+3pp vs all indices)

**Date:** 2026-07-22  
**Status:** implemented  
**Owner:** research / paper_live  

## Problem

Options PROMOTE sleeves (OPT_TA02/03/06/08) produce income-like returns (~0.2–0.4% mean) and lag SPY by ~13pp on recent study windows. The research goal is to **search annually** for strategies that clear a **strict alpha bar**:

> For each calendar year in the study:  
> `strategy_return ≥ max(SPY_BH, QQQ_BH, IWM_BH) + 0.03`

Equity signal strategies (trend / momentum / pullback / TA) are the primary candidates; short-premium options are retained as **controls** (not expected to clear bull-year bars).

## Scope

| Layer | Content |
|-------|---------|
| Equity | DailySignalPipeline modes: trend_mom, no_extension, pullback, qqq_gate, combined_v*, TA/volume, concentrated allocation, QQQ hold controls |
| Options | Subset of OPT_TA + basic CSP/collar/CC/IC — labels `proxy_bs` / `vix_surface` |
| Years | 2022, 2023, 2024, 2025_study (YTD) |
| Benchmarks | SPY, QQQ, IWM B&H same window |
| Capital | VIRTUAL only |

## Artifacts

| Path | Role |
|------|------|
| `paper_live/cloud/mega_annual_alpha.py` | Pure eval + runner |
| `paper_live/cloud/zoo_mega_alpha.json` | ≥20 equity + options controls |
| `scripts/run_mega_annual_alpha_study.py` | CLI |
| `tests/test_mega_annual_alpha.py` | Synthetic unit + smoke |
| `reports/mega_annual_alpha/latest/` | SUMMARY.md, winners.json, by_year/* |

## Protocol

1. Build free OHLCV feed (`build_cloud_feed`, Yahoo primary; refuse silent synthetic for real claims).
2. For each year window, clamp to available sessions; compute index B&H.
3. Run each equity strategy via ReplaySession (paper fills, causal signals ≤ t).
4. Run options batch (`run_options_batch`) per year.
5. Flag `beat_all_indices_by_3pp` per strategy×year.
6. Aggregate: years_passed, mean excess vs best index, tiers 4/4 … 1/4.
7. **Strict winners** = tier 4/4. If zero, document as valid scientific result and list near-misses (3/4, 2/4).

## Winner filters

**Primary (strict):** clear +3pp over best index **every** study year.

**Secondary:** rank by `years_passed`, then `mean_excess_vs_best`; optional `min_opens`; hard-kill years reported (soft-exclude in strict filter when `allow_hard_kill=False`).

## Honesty rules (non-negotiable)

- No look-ahead features.
- No fabricated returns.
- Options never labeled as OPRA / exchange fills.
- Aggressive equity concentration is long-only paper allocation — **not** leveraged ETF claims unless explicitly modeled.
- Zero strict winners is an acceptable outcome.

## How to run

```powershell
# Full real Yahoo study
python scripts/run_mega_annual_alpha_study.py --out reports/mega_annual_alpha

# Cap strategies if heavy
python scripts/run_mega_annual_alpha_study.py --max-equity 20 --max-options 8

# Offline / CI synthetic
python scripts/run_mega_annual_alpha_study.py --synthetic --max-equity 5 --max-options 2 --lookback-days 900

# Tests
python -m pytest tests/test_mega_annual_alpha.py -q --tb=short
```

## Out of scope

- Live trading promotion gates (scorecard PROMOTE remains separate).
- True option chain fills / OPRA.
- Intraday signals.
- Claiming guaranteed alpha from any paper result.
