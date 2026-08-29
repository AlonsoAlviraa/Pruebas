# Paper cloud multi-strategy — `2026-08-29`

**Window:** 2025-12-09 → 2026-08-28 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.64%** · Equal-weight names B&H **13.07%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.61% | +1.97% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.14% | +1.51% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.13% | -11.50% | 45.5% | 1.45 | 22 | 27 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.78% | -11.86% | 43.3% | 1.32 | 30 | 33 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.53% | -12.11% | 41.2% | 1.10 | 17 | 21 | no |
| 6 | `S07_pullback_long` | `combined_v2` | 0.36% | -12.27% | 37.5% | 1.01 | 16 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.42% | -13.05% | 36.1% | 0.89 | 36 | 40 | no |
| 8 | `S03_pullback` | `pullback` | -0.51% | -13.14% | 32.0% | 0.85 | 25 | 29 | no |
| 9 | `S05_topk_no_ext` | `no_extension` | -0.53% | -13.17% | 29.4% | 0.78 | 17 | 20 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -0.80% | -13.44% | 37.5% | 0.80 | 40 | 46 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=4
- `S10_defensive_no_ext`: stop=21, time_stop=9
- `S06_combined_v1`: stop=14, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=26, time_stop=10
- `S03_pullback`: stop=23, time_stop=2
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S01_baseline_trend_mom`: stop=30, time_stop=10

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
_Generated 2026-08-29T03:18:56.352740+00:00 · paper only_
