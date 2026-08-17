# Paper cloud multi-strategy — `2026-08-17`

**Window:** 2025-11-25 → 2026-08-17 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **14.47%** · Equal-weight names B&H **14.13%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 20.41% | +5.94% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.13% | +3.67% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.99% | -12.48% | 43.8% | 1.56 | 16 | 19 | no |
| 4 | `S02_no_extension` | `no_extension` | 1.45% | -13.02% | 45.5% | 1.56 | 22 | 26 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.90% | -13.57% | 42.9% | 1.36 | 28 | 32 | no |
| 6 | `S06_combined_v1` | `combined_v1` | 0.17% | -14.29% | 35.3% | 0.97 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.48% | -14.95% | 33.3% | 0.89 | 36 | 40 | no |
| 8 | `S03_pullback` | `pullback` | -0.54% | -15.01% | 37.0% | 0.94 | 27 | 29 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -0.81% | -15.28% | 35.7% | 0.84 | 42 | 48 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.67% | -16.13% | 29.4% | 0.57 | 17 | 19 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=11, time_stop=5
- `S02_no_extension`: stop=16, time_stop=6
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S06_combined_v1`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=24, time_stop=12
- `S03_pullback`: stop=23, time_stop=4
- `S01_baseline_trend_mom`: stop=30, time_stop=12
- `S07_pullback_long`: stop=12, time_stop=5

## Data sources

- `AAPL`: yahoo
- `AMZN`: yahoo
- `GOOGL`: yahoo
- `JPM`: yahoo
- `META`: yahoo
- `MSFT`: yahoo
- `NVDA`: yahoo
- `QQQ`: yahoo
- `SPY`: yahoo
- `XOM`: yahoo

## Per-strategy digests

See `strategies/<id>/dashboard.html`, `daily/`, and `closed_trades.csv`.

---
_Generated 2026-08-17T21:56:40.257197+00:00 · paper only_
