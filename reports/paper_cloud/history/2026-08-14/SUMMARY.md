# Paper cloud multi-strategy — `2026-08-14`

**Window:** 2025-11-24 → 2026-08-14 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **16.09%** · Equal-weight names B&H **15.58%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 22.21% | +6.12% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.32% | +2.23% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.95% | -14.15% | 43.8% | 1.56 | 16 | 19 | no |
| 4 | `S02_no_extension` | `no_extension` | 1.44% | -14.65% | 45.5% | 1.56 | 22 | 26 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.89% | -15.20% | 42.9% | 1.36 | 28 | 32 | no |
| 6 | `S06_combined_v1` | `combined_v1` | 0.13% | -15.96% | 35.3% | 0.97 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.36% | -16.45% | 34.3% | 0.94 | 35 | 40 | no |
| 8 | `S03_pullback` | `pullback` | -0.49% | -16.58% | 37.0% | 0.94 | 27 | 29 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -0.70% | -16.79% | 36.6% | 0.88 | 41 | 48 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.63% | -17.72% | 29.4% | 0.57 | 17 | 19 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=11, time_stop=5
- `S02_no_extension`: stop=16, time_stop=6
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S06_combined_v1`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=23, time_stop=12
- `S03_pullback`: stop=23, time_stop=4
- `S01_baseline_trend_mom`: stop=29, time_stop=12
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
_Generated 2026-08-14T21:52:37.401489+00:00 · paper only_
