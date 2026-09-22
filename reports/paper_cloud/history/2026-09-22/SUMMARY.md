# Paper cloud multi-strategy — `2026-09-22`

**Window:** 2025-12-30 → 2026-09-18 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.87%** · Equal-weight names B&H **13.68%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 15.99% | +5.12% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 15.49% | +4.62% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.03% | -9.84% | 41.7% | 1.46 | 24 | 28 | no |
| 4 | `S06_combined_v1` | `combined_v1` | -0.29% | -11.16% | 35.0% | 0.98 | 20 | 21 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | -0.70% | -11.57% | 30.0% | 0.81 | 30 | 32 | no |
| 6 | `S07_pullback_long` | `combined_v2` | -0.75% | -11.62% | 28.6% | 0.57 | 14 | 18 | no |
| 7 | `S05_topk_no_ext` | `no_extension` | -1.27% | -12.14% | 26.3% | 0.73 | 19 | 20 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -1.30% | -12.18% | 33.3% | 0.75 | 39 | 42 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.49% | -12.36% | 35.6% | 0.70 | 45 | 50 | no |
| 10 | `S03_pullback` | `pullback` | -1.65% | -12.52% | 26.1% | 0.61 | 23 | 27 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=6
- `S06_combined_v1`: stop=16, time_stop=4
- `S10_defensive_no_ext`: stop=23, time_stop=7
- `S07_pullback_long`: stop=10, time_stop=4
- `S05_topk_no_ext`: stop=15, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=31, time_stop=14
- `S03_pullback`: stop=20, time_stop=3

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
_Generated 2026-09-22T00:17:19.849600+00:00 · paper only_
