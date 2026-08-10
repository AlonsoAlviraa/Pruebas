# Paper cloud multi-strategy — `2026-08-10`

**Window:** 2025-11-18 → 2026-08-10 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **17.11%** · Equal-weight names B&H **18.14%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 19.70% | +2.59% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 16.74% | -0.37% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.77% | -15.35% | 43.8% | 1.56 | 16 | 19 | no |
| 4 | `S02_no_extension` | `no_extension` | 1.44% | -15.68% | 45.5% | 1.56 | 22 | 25 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.89% | -16.22% | 42.9% | 1.36 | 28 | 31 | no |
| 6 | `S03_pullback` | `pullback` | 0.39% | -16.73% | 37.0% | 1.10 | 27 | 29 | no |
| 7 | `S06_combined_v1` | `combined_v1` | -0.04% | -17.15% | 35.3% | 0.97 | 17 | 20 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -0.20% | -17.31% | 34.3% | 0.94 | 35 | 39 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -0.51% | -17.62% | 36.6% | 0.88 | 41 | 47 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.57% | -18.68% | 25.0% | 0.34 | 16 | 18 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=11, time_stop=5
- `S02_no_extension`: stop=16, time_stop=6
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S03_pullback`: stop=22, time_stop=5
- `S06_combined_v1`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=23, time_stop=12
- `S01_baseline_trend_mom`: stop=29, time_stop=12
- `S07_pullback_long`: stop=12, time_stop=4

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
_Generated 2026-08-10T22:10:17.787630+00:00 · paper only_
