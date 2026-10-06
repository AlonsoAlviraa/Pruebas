# Paper cloud multi-strategy — `2026-10-06`

**Window:** 2026-01-15 → 2026-10-05 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.93%** · Equal-weight names B&H **17.44%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 20.16% | +8.23% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 19.51% | +7.58% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.45% | -11.48% | 36.4% | 1.08 | 22 | 26 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.38% | -11.55% | 38.9% | 0.99 | 18 | 22 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | -0.19% | -12.12% | 32.1% | 0.85 | 28 | 32 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.52% | -12.45% | 31.2% | 0.72 | 16 | 19 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.03% | -12.96% | 35.9% | 0.75 | 39 | 45 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.11% | -13.04% | 36.4% | 0.70 | 44 | 50 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.40% | -13.33% | 38.5% | 0.61 | 13 | 16 | no |
| 10 | `S03_pullback` | `pullback` | -1.84% | -13.77% | 28.6% | 0.49 | 21 | 25 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=14, time_stop=8
- `S06_combined_v1`: stop=13, time_stop=5
- `S10_defensive_no_ext`: stop=20, time_stop=8
- `S05_topk_no_ext`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=26, time_stop=13
- `S01_baseline_trend_mom`: stop=29, time_stop=15
- `S07_pullback_long`: stop=9, time_stop=4
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
_Generated 2026-10-06T01:52:27.225130+00:00 · paper only_
