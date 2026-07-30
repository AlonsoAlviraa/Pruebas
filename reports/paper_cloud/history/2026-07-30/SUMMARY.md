# Paper cloud multi-strategy — `2026-07-30`

**Window:** 2025-11-07 → 2026-07-30 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.54%** · Equal-weight names B&H **8.78%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 12.05% | +1.51% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 11.68% | +1.14% | n/a | n/a | 0 | 1 | no |
| 3 | `S03_pullback` | `pullback` | 0.93% | -9.61% | 34.6% | 0.99 | 26 | 27 | no |
| 4 | `S05_topk_no_ext` | `no_extension` | 0.64% | -9.90% | 31.6% | 1.18 | 19 | 19 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.60% | -9.94% | 38.5% | 1.20 | 26 | 26 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.16% | -10.38% | 39.4% | 1.08 | 33 | 33 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.05% | -10.59% | 33.3% | 0.94 | 39 | 41 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.50% | -11.04% | 35.6% | 0.86 | 45 | 47 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -1.30% | -11.84% | 28.6% | 0.71 | 21 | 21 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.46% | -12.00% | 26.7% | 0.48 | 15 | 17 | no |

## Exit reasons (per strategy)

- `S03_pullback`: stop=21, time_stop=5
- `S05_topk_no_ext`: stop=14, time_stop=5
- `S02_no_extension`: stop=20, time_stop=6
- `S10_defensive_no_ext`: stop=24, time_stop=9
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=32, time_stop=13
- `S06_combined_v1`: stop=18, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=3

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
_Generated 2026-07-30T22:39:58.030796+00:00 · paper only_
