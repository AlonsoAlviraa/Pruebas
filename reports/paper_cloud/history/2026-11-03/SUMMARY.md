# Paper cloud multi-strategy — `2026-11-03`

**Window:** 2026-02-13 → 2026-11-02 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.32%** · Equal-weight names B&H **13.95%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 13.09% | +2.77% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 10.03% | -0.29% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.96% | -9.36% | 47.4% | 1.08 | 19 | 20 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.06% | -10.26% | 36.7% | 1.04 | 30 | 30 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.05% | -10.27% | 39.3% | 1.07 | 28 | 29 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.03% | -10.34% | 41.2% | 0.79 | 17 | 18 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.11% | -11.43% | 39.0% | 0.80 | 41 | 41 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.58% | -11.90% | 37.5% | 0.69 | 48 | 49 | no |
| 9 | `S03_pullback` | `pullback` | -1.93% | -12.25% | 27.3% | 0.53 | 22 | 23 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.99% | -12.31% | 35.7% | 0.41 | 14 | 15 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=6
- `S10_defensive_no_ext`: stop=19, time_stop=11
- `S02_no_extension`: stop=18, time_stop=10
- `S05_topk_no_ext`: stop=12, time_stop=5
- `S04_qqq_gate`: stop=27, time_stop=14
- `S01_baseline_trend_mom`: stop=32, time_stop=16
- `S03_pullback`: stop=17, time_stop=5
- `S07_pullback_long`: stop=8, time_stop=6

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
_Generated 2026-11-03T03:03:37.812965+00:00 · paper only_
