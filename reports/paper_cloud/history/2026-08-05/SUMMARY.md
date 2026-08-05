# Paper cloud multi-strategy — `2026-08-05`

**Window:** 2025-11-13 → 2026-08-05 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **14.55%** · Equal-weight names B&H **14.57%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 15.70% | +1.16% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 15.21% | +0.66% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.43% | -14.12% | 38.5% | 1.13 | 26 | 27 | no |
| 4 | `S03_pullback` | `pullback` | 0.31% | -14.23% | 37.0% | 1.08 | 27 | 28 | no |
| 5 | `S05_topk_no_ext` | `no_extension` | 0.30% | -14.24% | 31.6% | 1.07 | 19 | 20 | no |
| 6 | `S06_combined_v1` | `combined_v1` | 0.19% | -14.35% | 38.1% | 1.04 | 21 | 22 | no |
| 7 | `S10_defensive_no_ext` | `no_extension` | 0.06% | -14.48% | 37.5% | 1.03 | 32 | 33 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.82% | -15.37% | 40.9% | 0.87 | 44 | 49 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -0.88% | -15.43% | 36.8% | 0.88 | 38 | 44 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.86% | -16.40% | 25.0% | 0.34 | 16 | 18 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=19, time_stop=7
- `S03_pullback`: stop=22, time_stop=5
- `S05_topk_no_ext`: stop=13, time_stop=6
- `S06_combined_v1`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=23, time_stop=9
- `S01_baseline_trend_mom`: stop=30, time_stop=14
- `S04_qqq_gate`: stop=26, time_stop=12
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
_Generated 2026-08-05T22:37:15.620860+00:00 · paper only_
