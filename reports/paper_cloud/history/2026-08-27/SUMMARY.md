# Paper cloud multi-strategy — `2026-08-27`

**Window:** 2025-12-05 → 2026-08-26 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.72%** · Equal-weight names B&H **10.90%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 13.51% | +1.79% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 13.16% | +1.44% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.78% | -10.95% | 42.9% | 1.41 | 21 | 27 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.67% | -11.05% | 43.3% | 1.36 | 30 | 31 | no |
| 5 | `S07_pullback_long` | `combined_v2` | 0.06% | -11.67% | 31.2% | 1.03 | 16 | 18 | no |
| 6 | `S06_combined_v1` | `combined_v1` | -0.23% | -11.96% | 35.3% | 0.97 | 17 | 21 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.76% | -12.49% | 33.3% | 0.89 | 36 | 41 | no |
| 8 | `S03_pullback` | `pullback` | -0.86% | -12.58% | 32.0% | 0.84 | 25 | 27 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.27% | -12.99% | 34.1% | 0.78 | 41 | 48 | no |
| 10 | `S05_topk_no_ext` | `no_extension` | -1.31% | -13.04% | 23.5% | 0.69 | 17 | 20 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=20, time_stop=10
- `S07_pullback_long`: stop=12, time_stop=4
- `S06_combined_v1`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=24, time_stop=12
- `S03_pullback`: stop=23, time_stop=2
- `S01_baseline_trend_mom`: stop=29, time_stop=12
- `S05_topk_no_ext`: stop=14, time_stop=3

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
_Generated 2026-08-27T01:01:13.858120+00:00 · paper only_
