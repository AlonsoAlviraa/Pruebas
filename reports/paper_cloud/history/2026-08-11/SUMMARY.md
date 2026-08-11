# Paper cloud multi-strategy — `2026-08-11`

**Window:** 2025-11-19 → 2026-08-11 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **16.29%** · Equal-weight names B&H **16.67%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 19.85% | +3.56% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 16.36% | +0.07% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.70% | -14.59% | 43.8% | 1.56 | 16 | 19 | no |
| 4 | `S02_no_extension` | `no_extension` | 1.26% | -15.03% | 45.5% | 1.56 | 22 | 26 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.71% | -15.58% | 42.9% | 1.36 | 28 | 32 | no |
| 6 | `S03_pullback` | `pullback` | 0.14% | -16.15% | 37.0% | 1.10 | 27 | 29 | no |
| 7 | `S06_combined_v1` | `combined_v1` | -0.11% | -16.39% | 35.3% | 0.97 | 17 | 20 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -0.30% | -16.59% | 34.3% | 0.94 | 35 | 39 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -0.69% | -16.98% | 36.6% | 0.88 | 41 | 48 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.63% | -17.92% | 25.0% | 0.34 | 16 | 18 | no |

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
_Generated 2026-08-11T22:21:15.810593+00:00 · paper only_
