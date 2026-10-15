# Paper cloud multi-strategy — `2026-10-15`

**Window:** 2026-01-26 → 2026-10-13 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.31%** · Equal-weight names B&H **17.54%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 18.84% | +6.53% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.24% | +5.93% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.73% | -11.58% | 41.7% | 1.10 | 24 | 29 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.66% | -11.65% | 45.0% | 1.02 | 20 | 23 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.55% | -11.76% | 39.3% | 1.06 | 28 | 32 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | 0.05% | -12.26% | 41.2% | 0.84 | 17 | 19 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.44% | -12.75% | 40.0% | 0.77 | 40 | 45 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.68% | -12.99% | 40.0% | 0.72 | 45 | 51 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.16% | -13.47% | 42.9% | 0.63 | 14 | 17 | no |
| 10 | `S03_pullback` | `pullback` | -1.30% | -13.61% | 28.6% | 0.49 | 21 | 25 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=16, time_stop=8
- `S06_combined_v1`: stop=15, time_stop=5
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S05_topk_no_ext`: stop=13, time_stop=4
- `S04_qqq_gate`: stop=28, time_stop=12
- `S01_baseline_trend_mom`: stop=31, time_stop=14
- `S07_pullback_long`: stop=9, time_stop=5
- `S03_pullback`: stop=17, time_stop=4

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
_Generated 2026-10-15T00:03:32.076737+00:00 · paper only_
