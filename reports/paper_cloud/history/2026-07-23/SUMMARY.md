# Paper cloud multi-strategy — `2026-07-23`

**Window:** 2025-10-31 → 2026-07-23 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **8.23%** · Equal-weight names B&H **6.39%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 8.84% | +0.61% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 8.55% | +0.33% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 0.86% | -7.37% | 36.8% | 1.25 | 19 | 20 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.77% | -7.46% | 42.3% | 1.26 | 26 | 27 | no |
| 5 | `S03_pullback` | `pullback` | 0.65% | -7.58% | 32.0% | 0.80 | 25 | 29 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.22% | -8.01% | 39.4% | 1.09 | 33 | 34 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.83% | -9.06% | 34.1% | 0.83 | 41 | 44 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.00% | -9.22% | 34.0% | 0.78 | 47 | 50 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -1.09% | -9.32% | 33.3% | 0.75 | 21 | 22 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.87% | -10.09% | 25.0% | 0.41 | 16 | 19 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=14, time_stop=5
- `S02_no_extension`: stop=20, time_stop=6
- `S03_pullback`: stop=21, time_stop=4
- `S10_defensive_no_ext`: stop=24, time_stop=9
- `S04_qqq_gate`: stop=28, time_stop=13
- `S01_baseline_trend_mom`: stop=34, time_stop=13
- `S06_combined_v1`: stop=18, time_stop=3
- `S07_pullback_long`: stop=13, time_stop=3

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
_Generated 2026-07-23T22:34:49.316270+00:00 · paper only_
