# Paper cloud multi-strategy — `2026-07-22`

**Window:** 2025-10-29 → 2026-07-21 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **8.86%** · Equal-weight names B&H **7.72%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 11.24% | +2.38% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 10.88% | +2.02% | n/a | n/a | 0 | 1 | no |
| 3 | `S03_pullback` | `pullback` | 1.23% | -7.63% | 33.3% | 0.85 | 24 | 29 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.46% | -8.40% | 42.9% | 1.15 | 28 | 29 | no |
| 5 | `S05_topk_no_ext` | `no_extension` | 0.31% | -8.54% | 33.3% | 1.09 | 21 | 22 | no |
| 6 | `S04_qqq_gate` | `qqq_gate` | -0.02% | -8.88% | 37.5% | 0.99 | 40 | 44 | no |
| 7 | `S10_defensive_no_ext` | `no_extension` | -0.13% | -8.99% | 35.3% | 0.93 | 34 | 36 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.58% | -9.44% | 37.5% | 0.88 | 48 | 52 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.25% | -10.11% | 26.7% | 0.50 | 15 | 19 | no |
| 10 | `S06_combined_v1` | `combined_v1` | -1.57% | -10.43% | 30.4% | 0.69 | 23 | 24 | no |

## Exit reasons (per strategy)

- `S03_pullback`: stop=20, time_stop=4
- `S02_no_extension`: stop=22, time_stop=6
- `S05_topk_no_ext`: stop=16, time_stop=5
- `S04_qqq_gate`: stop=28, time_stop=12
- `S10_defensive_no_ext`: stop=26, time_stop=8
- `S01_baseline_trend_mom`: stop=36, time_stop=12
- `S07_pullback_long`: stop=12, time_stop=3
- `S06_combined_v1`: stop=20, time_stop=3

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
_Generated 2026-07-22T09:30:32.464382+00:00 · paper only_
