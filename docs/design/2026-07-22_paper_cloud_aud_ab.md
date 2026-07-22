# Design: PR-AUD-A + AUD-B — paper cloud loss fix loop

**Status:** implemented + loop verified (see `reports/paper_cloud_ab/audits/LOOP_AUD_AB_RESULTS.md`)  
**Date:** 2026-07-22  
**Scope:** Virtual paper only. No live capital.

## Problem

Audit of paper cloud (window ~2025-10 → 2026-07):

- 10/10 zoo variants negative while **SPY B&H +8.9%**
- Closed-trade **WR ~22%**, **PF ~0.34**
- Shared signal `rule_trend_mom` buys extended strength; zoo knobs only re-scale losses

## Goals

1. **AUD-A (instrumentation):** every pack emits closed-trade CSV with `exit_reason`, exit-reason histogram, SPY + equal-weight B&H in SUMMARY/JSON.
2. **AUD-B (signal A/B):** replace single rule with explicit `signal_mode` variants; re-run audit until **PF moves up** and **vs-SPY gap shrinks** (target: PF > 1.0 and at least one variant not deeply lagging SPY / better than baseline −5%).

## Non-goals

- Live trading, ML retrain, changing free data providers.
- Claiming production alpha after one window.

## Design

### Signal modes (`DailySignalPipeline.signal_mode`)

| Mode | Intent |
|------|--------|
| `trend_mom` | Baseline (legacy) |
| `no_extension` | Trend+mom but reject `dist_sma_50 > cap` / RSI overbought |
| `pullback` | Above SMA200, weak RSI / below-near SMA50 |
| `topk_mom` | Baseline filters + keep top-k scores only |
| `qqq_gate` | Require QQQ ret_1m > 0 in addition to dual-MA regime |
| `combined_v1` | no_extension + qqq_gate + top-k + exclude index names |
| `combined_v2` | pullback + qqq_gate + top-k + longer horizon (via zoo knobs) |

### Session trade log

`ReplaySession.closed_trades[]` on each exit: ticker, entry/exit day+px, qty, ret, pnl, bars_held, exit_reason (`stop`|`time_stop`).

Batch writes `strategies/<id>/closed_trades.csv` and aggregates exit reasons into master summary.

### Benchmarks

From feed OHLCV (causal closes on window endpoints):

- `spy_bh_return`
- `eq_weight_bh_return` (simple average of member total returns; tradeable names)

## PR plan

1. **PR-AUD-A** — trade log + benchmarks + SUMMARY fields  
2. **PR-AUD-B** — signal modes + zoo_ab.json + wire batch  
3. **PR-AUD-C** — run batch + audit script; iterate combined knobs if PF still &lt; 1

## Success criteria

- Instrumentación presente en latest pack.
- At least one A/B variant: **PF_closed ≥ 1.0** OR **total_return > baseline S01** by ≥2pp and closer to SPY than baseline.
- Prefer PF ≥ 1.0 and underperformance vs SPY reduced by half.

## Risks

- Single window still not walk-forward (AUD-05 later).
- Beating SPY with long-only mega-caps may require high beta; honesty over force-fitting.
