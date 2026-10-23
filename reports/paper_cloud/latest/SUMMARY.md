# Paper cloud multi-strategy — `2026-10-23`

**Window:** 2026-02-03 → 2026-10-21 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.12%** · Equal-weight names B&H **15.30%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.96% | +2.84% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.57% | +2.46% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.85% | -11.27% | 44.4% | 1.05 | 18 | 21 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.52% | -11.60% | 41.7% | 1.05 | 24 | 28 | no |
| 5 | `S05_topk_no_ext` | `no_extension` | 0.18% | -11.94% | 40.0% | 0.84 | 15 | 18 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.16% | -11.95% | 35.7% | 1.03 | 28 | 29 | no |
| 7 | `S07_pullback_long` | `combined_v2` | -1.12% | -13.23% | 45.5% | 0.57 | 11 | 15 | no |
| 8 | `S03_pullback` | `pullback` | -1.31% | -13.42% | 26.3% | 0.41 | 19 | 23 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.46% | -13.57% | 37.0% | 0.62 | 46 | 51 | no |
| 10 | `S04_qqq_gate` | `qqq_gate` | -1.50% | -13.61% | 36.6% | 0.65 | 41 | 45 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=5
- `S02_no_extension`: stop=15, time_stop=9
- `S05_topk_no_ext`: stop=11, time_stop=4
- `S10_defensive_no_ext`: stop=18, time_stop=10
- `S07_pullback_long`: stop=6, time_stop=5
- `S03_pullback`: stop=15, time_stop=4
- `S01_baseline_trend_mom`: stop=32, time_stop=14
- `S04_qqq_gate`: stop=29, time_stop=12

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
_Generated 2026-10-23T00:07:47.028796+00:00 · paper only_
