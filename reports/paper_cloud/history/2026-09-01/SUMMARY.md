# Paper cloud multi-strategy — `2026-09-01`

**Window:** 2025-12-10 → 2026-08-31 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.56%** · Equal-weight names B&H **12.17%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.53% | +2.97% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.06% | +2.50% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.10% | -10.46% | 42.9% | 1.55 | 21 | 26 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.58% | -10.98% | 43.3% | 1.22 | 30 | 33 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.28% | -11.28% | 35.3% | 1.07 | 17 | 21 | no |
| 6 | `S07_pullback_long` | `combined_v2` | 0.19% | -11.37% | 37.5% | 1.01 | 16 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.38% | -11.94% | 34.3% | 0.92 | 35 | 39 | no |
| 8 | `S03_pullback` | `pullback` | -0.64% | -12.20% | 32.0% | 0.85 | 25 | 29 | no |
| 9 | `S05_topk_no_ext` | `no_extension` | -0.75% | -12.31% | 23.5% | 0.75 | 17 | 20 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -0.85% | -12.41% | 35.9% | 0.81 | 39 | 45 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=19, time_stop=11
- `S06_combined_v1`: stop=14, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=25, time_stop=10
- `S03_pullback`: stop=23, time_stop=2
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S01_baseline_trend_mom`: stop=29, time_stop=10

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
_Generated 2026-09-01T00:46:17.623073+00:00 · paper only_
