# Paper cloud multi-strategy — `2026-10-15`

**Window:** 2026-01-28 → 2026-10-15 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.98%** · Equal-weight names B&H **14.58%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.19% | +3.21% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 13.73% | +2.75% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.62% | -10.37% | 43.5% | 1.16 | 23 | 29 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.57% | -10.41% | 47.4% | 1.06 | 19 | 23 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.21% | -10.77% | 37.9% | 1.06 | 29 | 31 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.36% | -11.34% | 41.2% | 0.78 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.99% | -11.97% | 40.0% | 0.74 | 40 | 46 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.17% | -12.16% | 40.0% | 0.69 | 45 | 52 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.18% | -12.16% | 45.5% | 0.57 | 11 | 15 | no |
| 10 | `S03_pullback` | `pullback` | -1.39% | -12.37% | 27.8% | 0.43 | 18 | 23 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=15, time_stop=8
- `S06_combined_v1`: stop=14, time_stop=5
- `S10_defensive_no_ext`: stop=19, time_stop=10
- `S05_topk_no_ext`: stop=13, time_stop=4
- `S04_qqq_gate`: stop=28, time_stop=12
- `S01_baseline_trend_mom`: stop=31, time_stop=14
- `S07_pullback_long`: stop=6, time_stop=5
- `S03_pullback`: stop=14, time_stop=4

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
_Generated 2026-10-15T23:48:17.716275+00:00 · paper only_
