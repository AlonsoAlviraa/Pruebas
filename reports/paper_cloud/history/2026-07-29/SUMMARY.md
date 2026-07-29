# Paper cloud multi-strategy — `2026-07-29`

**Window:** 2025-11-06 → 2026-07-29 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **8.82%** · Equal-weight names B&H **7.35%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 6.21% | -2.61% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 6.01% | -2.81% | n/a | n/a | 0 | 1 | no |
| 3 | `S03_pullback` | `pullback` | 0.68% | -8.14% | 33.3% | 0.92 | 27 | 28 | no |
| 4 | `S05_topk_no_ext` | `no_extension` | 0.65% | -8.17% | 31.6% | 1.19 | 19 | 19 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.61% | -8.22% | 38.5% | 1.20 | 26 | 26 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.17% | -8.65% | 39.4% | 1.08 | 33 | 33 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.45% | -9.28% | 32.5% | 0.87 | 40 | 42 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.75% | -9.57% | 34.8% | 0.81 | 46 | 48 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -1.29% | -10.11% | 28.6% | 0.71 | 21 | 21 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -2.12% | -10.94% | 25.0% | 0.41 | 16 | 18 | no |

## Exit reasons (per strategy)

- `S03_pullback`: stop=22, time_stop=5
- `S05_topk_no_ext`: stop=14, time_stop=5
- `S02_no_extension`: stop=20, time_stop=6
- `S10_defensive_no_ext`: stop=24, time_stop=9
- `S04_qqq_gate`: stop=27, time_stop=13
- `S01_baseline_trend_mom`: stop=32, time_stop=14
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
_Generated 2026-07-29T22:32:26.658320+00:00 · paper only_
