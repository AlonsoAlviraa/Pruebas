# Paper cloud multi-strategy — `2026-10-31`

**Window:** 2026-02-11 → 2026-10-29 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.19%** · Equal-weight names B&H **13.26%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 15.58% | +4.39% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 15.30% | +4.12% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.32% | -9.86% | 50.0% | 1.30 | 18 | 20 | no |
| 4 | `S05_topk_no_ext` | `no_extension` | 0.58% | -10.61% | 46.7% | 1.07 | 15 | 18 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.43% | -10.75% | 42.3% | 1.29 | 26 | 29 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.37% | -10.82% | 39.3% | 1.19 | 28 | 30 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.03% | -12.21% | 40.0% | 0.85 | 40 | 41 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.31% | -12.50% | 37.8% | 0.73 | 45 | 49 | no |
| 9 | `S03_pullback` | `pullback` | -1.54% | -12.72% | 25.0% | 0.39 | 20 | 23 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.61% | -12.80% | 38.5% | 0.46 | 13 | 15 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=12, time_stop=6
- `S05_topk_no_ext`: stop=10, time_stop=5
- `S02_no_extension`: stop=16, time_stop=10
- `S10_defensive_no_ext`: stop=17, time_stop=11
- `S04_qqq_gate`: stop=26, time_stop=14
- `S01_baseline_trend_mom`: stop=30, time_stop=15
- `S03_pullback`: stop=16, time_stop=4
- `S07_pullback_long`: stop=7, time_stop=6

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
_Generated 2026-10-31T01:00:08.031631+00:00 · paper only_
