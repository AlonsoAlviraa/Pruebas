# Paper cloud multi-strategy — `2026-10-03`

**Window:** 2026-01-13 → 2026-10-01 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.12%** · Equal-weight names B&H **14.76%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 17.84% | +7.72% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 17.27% | +7.15% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.26% | -9.86% | 38.9% | 1.09 | 18 | 22 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.17% | -9.95% | 34.8% | 1.08 | 23 | 27 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | -0.25% | -10.37% | 33.3% | 0.92 | 27 | 31 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.79% | -10.92% | 29.4% | 0.72 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.88% | -11.01% | 34.2% | 0.83 | 38 | 44 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.49% | -11.62% | 31.8% | 0.68 | 44 | 50 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -2.02% | -12.14% | 35.7% | 0.53 | 14 | 17 | no |
| 10 | `S03_pullback` | `pullback` | -2.33% | -12.45% | 27.3% | 0.46 | 22 | 26 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=5
- `S02_no_extension`: stop=15, time_stop=8
- `S10_defensive_no_ext`: stop=20, time_stop=7
- `S05_topk_no_ext`: stop=13, time_stop=4
- `S04_qqq_gate`: stop=26, time_stop=12
- `S01_baseline_trend_mom`: stop=31, time_stop=13
- `S07_pullback_long`: stop=10, time_stop=4
- `S03_pullback`: stop=18, time_stop=4

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
_Generated 2026-10-03T00:41:30.903503+00:00 · paper only_
