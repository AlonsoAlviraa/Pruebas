# Paper cloud multi-strategy — `2026-07-22`

**Window:** 2025-10-30 → 2026-07-22 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **9.94%** · Equal-weight names B&H **9.36%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 11.28% | +1.34% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 10.92% | +0.98% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.22% | -8.72% | 35.0% | 1.32 | 20 | 21 | no |
| 4 | `S03_pullback` | `pullback` | 1.07% | -8.87% | 33.3% | 0.81 | 24 | 29 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.99% | -8.95% | 40.7% | 1.31 | 27 | 28 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.52% | -9.42% | 38.2% | 1.17 | 34 | 35 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.17% | -10.11% | 37.5% | 0.96 | 40 | 44 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.56% | -10.50% | 38.3% | 0.88 | 47 | 51 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -0.72% | -10.66% | 31.8% | 0.82 | 22 | 23 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.26% | -11.20% | 26.7% | 0.48 | 15 | 19 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=15, time_stop=5
- `S03_pullback`: stop=20, time_stop=4
- `S02_no_extension`: stop=21, time_stop=6
- `S10_defensive_no_ext`: stop=25, time_stop=9
- `S04_qqq_gate`: stop=28, time_stop=12
- `S01_baseline_trend_mom`: stop=35, time_stop=12
- `S06_combined_v1`: stop=19, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=3

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
_Generated 2026-07-22T22:38:43.643580+00:00 · paper only_
