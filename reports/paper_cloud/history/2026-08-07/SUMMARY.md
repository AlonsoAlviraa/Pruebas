# Paper cloud multi-strategy — `2026-08-07`

**Window:** 2025-11-17 → 2026-08-07 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **16.16%** · Equal-weight names B&H **16.12%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 18.73% | +2.57% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.15% | +1.99% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.31% | -14.86% | 29.4% | 1.40 | 17 | 20 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.98% | -15.18% | 41.7% | 1.33 | 24 | 27 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.49% | -15.67% | 40.0% | 1.17 | 30 | 33 | no |
| 6 | `S03_pullback` | `pullback` | 0.44% | -15.73% | 37.0% | 1.10 | 27 | 28 | no |
| 7 | `S06_combined_v1` | `combined_v1` | -0.14% | -16.31% | 26.3% | 0.93 | 19 | 22 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.60% | -16.77% | 39.5% | 0.88 | 43 | 49 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -0.69% | -16.85% | 35.9% | 0.88 | 39 | 43 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.75% | -17.92% | 25.0% | 0.34 | 16 | 18 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=11, time_stop=6
- `S02_no_extension`: stop=17, time_stop=7
- `S10_defensive_no_ext`: stop=21, time_stop=9
- `S03_pullback`: stop=22, time_stop=5
- `S06_combined_v1`: stop=14, time_stop=5
- `S01_baseline_trend_mom`: stop=31, time_stop=12
- `S04_qqq_gate`: stop=28, time_stop=11
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
_Generated 2026-08-07T22:08:58.884332+00:00 · paper only_
