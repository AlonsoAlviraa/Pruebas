# Design: TA / volume equity signals + options TA gates

**Status:** implemented v1  
**Date:** 2026-07-22  
**Capital:** VIRTUAL only — research / paper cloud.

## Motivation

Existing paper equity zoo is trend/pullback/no-extension. Volume and RSI features already exist on the causal feed (`volume_ratio`, `volume_zscore`, `rsi_*`, `atr_norm`, `dist_sma_*`) but were under-used. Options premium selling was gated only by a crude HV median check. This module adds **causal TA/volume modes** for equities and **meta TA gates** for options opens.

## Causality (non-negotiable)

- Features at signal day `D` use only OHLCV with `date ≤ D` via `DailyReplayFeed.featured(..., through=D)`.
- Labels that look into the future are **never** used as features.
- Options gates call the same featured row (and trailing ATR percentile on history ≤ D).
- Entry still follows the paper pipeline convention: signal close D → candidate for D+1 open when the equity runner applies it.

## Equity signal modes

Implemented in `paper_live/signals/daily_pipeline.py` and dispatched by `score_row_for_mode`:

| Mode | Rule (summary) | Reason string |
|------|----------------|---------------|
| `vol_confirm` / `volume_breakout` | Baseline trend_mom **and** elevated `volume_ratio` or `volume_zscore` | `rule_volume_breakout` |
| `rsi_mr` | RSI ≤ 32, close near/above SMA200, ATR band | `rule_rsi_mean_reversion` |
| `vol_dryup` | Pullback-in-uptrend **and** quiet volume | `rule_volume_dryup` |
| `vol_expand` | Uptrend + ret_1m>0 + high relative volume, RSI not climax | `rule_volume_expansion` |
| `rvol_trend` | Mild no_extension **and** relative volume confirm | `rule_rvol_trend` |
| `vol_pullback` | Dry-up pullback or soft RSI+dry volume | `rule_vol_pullback` |
| `combined_ta_v1` | Prefer dry-up; else rvol_trend | either above |

Zoo: `paper_live/cloud/strategy_zoo_ta.json` (TA01–TA10).

### Run

```powershell
python scripts/run_paper_cloud_batch.py --zoo paper_live/cloud/strategy_zoo_ta.json --out reports/paper_cloud_ta --synthetic --lookback-days 120 --start 2020-06-01 --end 2020-07-15
# real/Yahoo:
python scripts/run_paper_cloud_batch.py --zoo paper_live/cloud/strategy_zoo_ta.json --out reports/paper_cloud_ta --start 2025-10-29
```

## Options TA gates

Module: `paper_live/options/ta_gates.py`  
Hook: `paper_live/options/replay_options.py` — before `open_structure`, if meta TA keys fail → `skip_new` (existing position marks continue).

| Meta key | Intent |
|----------|--------|
| `require_uptrend` | close > SMA50 & SMA200 |
| `require_sma200` | close > SMA200 |
| `require_volume_confirm` | elevated volume_ratio / z |
| `require_volume_dryup` | quiet volume |
| `require_rsi_oversold` / `require_rsi_overbought` | RSI thresholds |
| `require_low_atr` | ATR-norm percentile ≤ threshold |
| `require_range_regime` | low ATR + not extended vs SMA50 |
| `require_vol_climax` | high volume (risk-off insurance) |
| `require_compression_after_vol` | recent elevated vol → dry + low ATR |
| `require_pullback_uptrend` | above SMA200 + soft intermediate |

Legacy `require_hv_above_median` still applies first.

Zoo: `paper_live/cloud/zoo_options_ta.json` (OPT_TA01–OPT_TA12).  
**Data label remains `proxy_bs`** — not exchange fills.

### Run

```powershell
python scripts/run_paper_options_batch.py --zoo paper_live/cloud/zoo_options_ta.json --out reports/paper_options_ta --start 2025-10-29
```

## What this is not

- Not a claim of live alpha.
- Not real options NBBO fills.
- Not using future bars or same-day open knowledge beyond the paper pipeline’s documented lag.

## Tests

- `tests/test_paper_signal_ta_volume.py` — row-level equity modes (synthetic Series).
- `tests/test_paper_options_ta_gates.py` — gate unit + options replay smoke.
