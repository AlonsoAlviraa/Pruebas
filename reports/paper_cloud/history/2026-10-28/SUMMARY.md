# Paper cloud multi-strategy — `2026-10-28`

**Window:** 2026-02-06 → 2026-10-26 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.82%** · Equal-weight names B&H **14.34%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 18.20% | +7.37% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.82% | +4.00% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.44% | -9.38% | 47.1% | 1.25 | 17 | 20 | no |
| 4 | `S05_topk_no_ext` | `no_extension` | 0.81% | -10.02% | 42.9% | 1.02 | 14 | 17 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.57% | -10.25% | 40.0% | 1.07 | 25 | 27 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.43% | -10.39% | 39.3% | 1.19 | 28 | 28 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.04% | -11.86% | 38.5% | 0.81 | 39 | 41 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.25% | -12.07% | 37.8% | 0.73 | 45 | 47 | no |
| 9 | `S03_pullback` | `pullback` | -1.51% | -12.33% | 26.3% | 0.41 | 19 | 23 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.71% | -12.53% | 38.5% | 0.46 | 13 | 15 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=12, time_stop=5
- `S05_topk_no_ext`: stop=10, time_stop=4
- `S02_no_extension`: stop=16, time_stop=9
- `S10_defensive_no_ext`: stop=17, time_stop=11
- `S04_qqq_gate`: stop=26, time_stop=13
- `S01_baseline_trend_mom`: stop=30, time_stop=15
- `S03_pullback`: stop=15, time_stop=4
- `S07_pullback_long`: stop=7, time_stop=6

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
_Generated 2026-10-28T00:42:13.433390+00:00 · paper only_
