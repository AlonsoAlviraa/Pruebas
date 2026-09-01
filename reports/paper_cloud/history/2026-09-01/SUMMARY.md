# Paper cloud multi-strategy — `2026-09-01`

**Window:** 2025-12-11 → 2026-09-01 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.54%** · Equal-weight names B&H **12.21%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 13.10% | +2.56% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 12.68% | +2.15% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.99% | -9.55% | 42.9% | 1.53 | 21 | 26 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.47% | -10.07% | 41.9% | 1.19 | 31 | 35 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.19% | -10.35% | 35.3% | 1.06 | 17 | 21 | no |
| 6 | `S07_pullback_long` | `combined_v2` | 0.17% | -10.37% | 37.5% | 1.01 | 16 | 20 | no |
| 7 | `S03_pullback` | `pullback` | -0.65% | -11.19% | 32.0% | 0.85 | 25 | 29 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -0.78% | -11.31% | 36.8% | 0.87 | 38 | 42 | no |
| 9 | `S05_topk_no_ext` | `no_extension` | -0.84% | -11.37% | 23.5% | 0.75 | 17 | 20 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.13% | -11.66% | 38.1% | 0.79 | 42 | 47 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=21, time_stop=10
- `S06_combined_v1`: stop=14, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=4
- `S03_pullback`: stop=23, time_stop=2
- `S04_qqq_gate`: stop=26, time_stop=12
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S01_baseline_trend_mom`: stop=30, time_stop=12

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
_Generated 2026-09-01T23:31:31.853671+00:00 · paper only_
