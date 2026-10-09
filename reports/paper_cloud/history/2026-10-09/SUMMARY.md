# Paper cloud multi-strategy — `2026-10-09`

**Window:** 2026-01-20 → 2026-10-07 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **14.71%** · Equal-weight names B&H **20.71%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 23.49% | +8.79% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 22.76% | +8.05% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.14% | -13.57% | 42.1% | 1.22 | 19 | 23 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.95% | -13.75% | 39.1% | 1.30 | 23 | 27 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.71% | -13.99% | 40.7% | 1.17 | 27 | 31 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | 0.08% | -14.63% | 31.2% | 0.86 | 16 | 19 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.22% | -14.92% | 38.5% | 0.88 | 39 | 45 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.52% | -15.23% | 38.6% | 0.81 | 44 | 50 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.19% | -15.90% | 42.9% | 0.63 | 14 | 16 | no |
| 10 | `S03_pullback` | `pullback` | -1.66% | -16.36% | 28.6% | 0.49 | 21 | 25 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=14, time_stop=5
- `S02_no_extension`: stop=15, time_stop=8
- `S10_defensive_no_ext`: stop=19, time_stop=8
- `S05_topk_no_ext`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=30, time_stop=14
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
_Generated 2026-10-09T01:22:32.243753+00:00 · paper only_
