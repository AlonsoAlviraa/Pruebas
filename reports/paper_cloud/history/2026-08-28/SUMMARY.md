# Paper cloud multi-strategy — `2026-08-28`

**Window:** 2025-12-08 → 2026-08-27 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.79%** · Equal-weight names B&H **12.10%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.59% | +1.79% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.12% | +1.33% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.05% | -11.74% | 45.5% | 1.47 | 22 | 27 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.67% | -12.12% | 43.3% | 1.37 | 30 | 31 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.21% | -12.59% | 38.9% | 1.02 | 18 | 21 | no |
| 6 | `S07_pullback_long` | `combined_v2` | -0.02% | -12.81% | 37.5% | 1.01 | 16 | 18 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.60% | -13.40% | 35.1% | 0.88 | 37 | 41 | no |
| 8 | `S05_topk_no_ext` | `no_extension` | -0.77% | -13.57% | 27.8% | 0.74 | 18 | 20 | no |
| 9 | `S03_pullback` | `pullback` | -0.82% | -13.61% | 32.0% | 0.85 | 25 | 27 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.01% | -13.81% | 36.6% | 0.79 | 41 | 47 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=4
- `S10_defensive_no_ext`: stop=20, time_stop=10
- `S06_combined_v1`: stop=15, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=26, time_stop=11
- `S05_topk_no_ext`: stop=15, time_stop=3
- `S03_pullback`: stop=23, time_stop=2
- `S01_baseline_trend_mom`: stop=30, time_stop=11

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
_Generated 2026-08-28T05:35:59.655009+00:00 · paper only_
