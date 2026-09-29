# Paper cloud multi-strategy — `2026-09-29`

**Window:** 2026-01-07 → 2026-09-25 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.86%** · Equal-weight names B&H **15.84%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 19.01% | +7.15% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.40% | +6.55% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.49% | -10.36% | 40.0% | 1.50 | 20 | 24 | no |
| 4 | `S02_no_extension` | `no_extension` | 1.05% | -10.80% | 36.0% | 1.34 | 25 | 28 | no |
| 5 | `S05_topk_no_ext` | `no_extension` | 1.04% | -10.81% | 33.3% | 1.25 | 18 | 21 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.17% | -11.69% | 34.5% | 1.04 | 29 | 32 | no |
| 7 | `S07_pullback_long` | `combined_v2` | -0.81% | -12.67% | 28.6% | 0.61 | 14 | 18 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -0.91% | -12.77% | 28.9% | 0.72 | 38 | 43 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.14% | -13.00% | 31.8% | 0.66 | 44 | 49 | no |
| 10 | `S03_pullback` | `pullback` | -1.59% | -13.45% | 24.0% | 0.62 | 25 | 29 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=16, time_stop=4
- `S02_no_extension`: stop=17, time_stop=8
- `S05_topk_no_ext`: stop=14, time_stop=4
- `S10_defensive_no_ext`: stop=21, time_stop=8
- `S07_pullback_long`: stop=10, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=11
- `S01_baseline_trend_mom`: stop=31, time_stop=13
- `S03_pullback`: stop=21, time_stop=4

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
_Generated 2026-09-29T01:14:00.330089+00:00 · paper only_
