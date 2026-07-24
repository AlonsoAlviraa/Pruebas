# Paper cloud multi-strategy — `2026-07-24`

**Window:** 2025-11-03 → 2026-07-24 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **8.14%** · Equal-weight names B&H **6.15%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 7.50% | -0.63% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 7.26% | -0.87% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.09% | -7.04% | 38.9% | 1.36 | 18 | 19 | no |
| 4 | `S03_pullback` | `pullback` | 0.75% | -7.38% | 30.8% | 0.77 | 26 | 29 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.73% | -7.40% | 42.3% | 1.27 | 26 | 27 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.40% | -7.74% | 40.6% | 1.18 | 32 | 33 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.40% | -8.54% | 35.0% | 0.88 | 40 | 43 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.73% | -8.87% | 34.8% | 0.81 | 46 | 49 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -0.86% | -9.00% | 35.0% | 0.80 | 20 | 21 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.96% | -10.09% | 25.0% | 0.41 | 16 | 19 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=13, time_stop=5
- `S03_pullback`: stop=22, time_stop=4
- `S02_no_extension`: stop=20, time_stop=6
- `S10_defensive_no_ext`: stop=23, time_stop=9
- `S04_qqq_gate`: stop=27, time_stop=13
- `S01_baseline_trend_mom`: stop=33, time_stop=13
- `S06_combined_v1`: stop=17, time_stop=3
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
_Generated 2026-07-24T22:39:11.697007+00:00 · paper only_
