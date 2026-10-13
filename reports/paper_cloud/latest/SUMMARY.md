# Paper cloud multi-strategy — `2026-10-13`

**Window:** 2026-01-23 → 2026-10-12 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.89%** · Equal-weight names B&H **18.08%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 19.90% | +7.00% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 19.27% | +6.37% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.86% | -12.04% | 42.1% | 1.07 | 19 | 23 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.83% | -12.06% | 39.1% | 1.15 | 23 | 29 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.52% | -12.37% | 39.3% | 1.09 | 28 | 32 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.26% | -13.15% | 31.2% | 0.75 | 16 | 19 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.52% | -13.42% | 38.5% | 0.78 | 39 | 45 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.74% | -13.64% | 38.6% | 0.73 | 44 | 51 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.11% | -14.00% | 42.9% | 0.63 | 14 | 17 | no |
| 10 | `S03_pullback` | `pullback` | -1.32% | -14.21% | 28.6% | 0.49 | 21 | 25 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=14, time_stop=5
- `S02_no_extension`: stop=15, time_stop=8
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S05_topk_no_ext`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=30, time_stop=14
- `S07_pullback_long`: stop=9, time_stop=5
- `S03_pullback`: stop=17, time_stop=4

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
_Generated 2026-10-13T02:42:56.389388+00:00 · paper only_
