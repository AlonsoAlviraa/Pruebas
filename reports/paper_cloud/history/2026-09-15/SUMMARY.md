# Paper cloud multi-strategy — `2026-09-15`

**Window:** 2025-12-24 → 2026-09-15 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **9.71%** · Equal-weight names B&H **12.73%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 12.90% | +3.19% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 12.49% | +2.78% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.11% | -8.60% | 48.0% | 1.43 | 25 | 28 | no |
| 4 | `S06_combined_v1` | `combined_v1` | -0.28% | -9.98% | 40.0% | 1.02 | 20 | 21 | no |
| 5 | `S03_pullback` | `pullback` | -0.34% | -10.04% | 30.4% | 0.84 | 23 | 28 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | -0.68% | -10.39% | 35.5% | 0.81 | 31 | 32 | no |
| 7 | `S05_topk_no_ext` | `no_extension` | -0.90% | -10.61% | 33.3% | 0.78 | 18 | 18 | no |
| 8 | `S07_pullback_long` | `combined_v2` | -0.96% | -10.67% | 28.6% | 0.56 | 14 | 18 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -1.07% | -10.77% | 35.0% | 0.78 | 40 | 43 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.24% | -10.95% | 37.0% | 0.72 | 46 | 50 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=7
- `S06_combined_v1`: stop=16, time_stop=4
- `S03_pullback`: stop=21, time_stop=2
- `S10_defensive_no_ext`: stop=23, time_stop=8
- `S05_topk_no_ext`: stop=14, time_stop=4
- `S07_pullback_long`: stop=10, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=13
- `S01_baseline_trend_mom`: stop=31, time_stop=15

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
_Generated 2026-09-15T23:39:14.018213+00:00 · paper only_
