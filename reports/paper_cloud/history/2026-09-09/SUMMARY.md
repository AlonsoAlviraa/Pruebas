# Paper cloud multi-strategy — `2026-09-09`

**Window:** 2025-12-18 → 2026-09-09 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.70%** · Equal-weight names B&H **15.19%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 16.98% | +4.28% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.38% | +1.67% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.81% | -11.89% | 47.8% | 1.48 | 23 | 28 | no |
| 4 | `S06_combined_v1` | `combined_v1` | -0.13% | -12.83% | 35.3% | 0.98 | 17 | 21 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | -0.19% | -12.90% | 39.3% | 0.92 | 28 | 30 | no |
| 6 | `S03_pullback` | `pullback` | -0.96% | -13.66% | 29.2% | 0.82 | 24 | 29 | no |
| 7 | `S05_topk_no_ext` | `no_extension` | -0.97% | -13.67% | 23.5% | 0.73 | 17 | 20 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -1.31% | -14.01% | 33.3% | 0.84 | 39 | 44 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.31% | -14.01% | 38.6% | 0.81 | 44 | 50 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.85% | -14.56% | 31.2% | 0.52 | 16 | 20 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=5
- `S06_combined_v1`: stop=14, time_stop=3
- `S10_defensive_no_ext`: stop=21, time_stop=7
- `S03_pullback`: stop=22, time_stop=2
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=30, time_stop=14
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
_Generated 2026-09-09T23:29:17.840905+00:00 · paper only_
