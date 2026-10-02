# Paper cloud multi-strategy — `2026-10-02`

**Window:** 2026-01-12 → 2026-09-30 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **9.71%** · Equal-weight names B&H **14.20%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 18.38% | +8.68% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 17.80% | +8.09% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.30% | -9.41% | 34.8% | 1.09 | 23 | 26 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.16% | -9.54% | 36.8% | 1.00 | 19 | 22 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | -0.09% | -9.79% | 33.3% | 0.94 | 27 | 30 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.81% | -10.51% | 29.4% | 0.73 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.22% | -10.93% | 32.5% | 0.76 | 40 | 45 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.29% | -10.99% | 31.8% | 0.71 | 44 | 49 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.93% | -11.64% | 35.7% | 0.53 | 14 | 17 | no |
| 10 | `S03_pullback` | `pullback` | -2.27% | -11.98% | 27.3% | 0.46 | 22 | 26 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=15, time_stop=8
- `S06_combined_v1`: stop=14, time_stop=5
- `S10_defensive_no_ext`: stop=20, time_stop=7
- `S05_topk_no_ext`: stop=13, time_stop=4
- `S04_qqq_gate`: stop=28, time_stop=12
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
_Generated 2026-10-02T01:04:24.292707+00:00 · paper only_
