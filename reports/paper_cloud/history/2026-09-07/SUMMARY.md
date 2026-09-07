# Paper cloud multi-strategy — `2026-09-07`

**Window:** 2025-12-16 → 2026-09-04 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **13.45%** · Equal-weight names B&H **15.74%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 17.76% | +4.31% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.78% | +1.33% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.13% | -12.32% | 45.5% | 1.44 | 22 | 27 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.35% | -13.11% | 35.3% | 0.98 | 17 | 21 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.19% | -13.27% | 42.3% | 0.99 | 26 | 30 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.56% | -14.02% | 23.5% | 0.73 | 17 | 20 | no |
| 7 | `S03_pullback` | `pullback` | -0.88% | -14.34% | 29.2% | 0.81 | 24 | 28 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -1.06% | -14.51% | 34.2% | 0.85 | 38 | 44 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.13% | -14.58% | 39.5% | 0.84 | 43 | 50 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.66% | -15.11% | 31.2% | 0.53 | 16 | 20 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=4
- `S06_combined_v1`: stop=14, time_stop=3
- `S10_defensive_no_ext`: stop=19, time_stop=7
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S03_pullback`: stop=22, time_stop=2
- `S04_qqq_gate`: stop=27, time_stop=11
- `S01_baseline_trend_mom`: stop=30, time_stop=13
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
_Generated 2026-09-07T23:44:19.737250+00:00 · paper only_
