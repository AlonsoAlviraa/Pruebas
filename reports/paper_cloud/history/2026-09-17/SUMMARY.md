# Paper cloud multi-strategy — `2026-09-17`

**Window:** 2025-12-29 → 2026-09-17 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.87%** · Equal-weight names B&H **13.69%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 15.24% | +4.37% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.76% | +3.89% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.13% | -9.74% | 41.7% | 1.48 | 24 | 27 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.00% | -10.86% | 36.8% | 1.09 | 19 | 20 | no |
| 5 | `S03_pullback` | `pullback` | -0.54% | -11.40% | 29.2% | 0.88 | 24 | 27 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | -0.70% | -11.57% | 30.0% | 0.78 | 30 | 31 | no |
| 7 | `S05_topk_no_ext` | `no_extension` | -0.89% | -11.76% | 27.8% | 0.78 | 18 | 18 | no |
| 8 | `S07_pullback_long` | `combined_v2` | -0.98% | -11.85% | 28.6% | 0.56 | 14 | 18 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -1.30% | -12.17% | 33.3% | 0.75 | 39 | 42 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.39% | -12.26% | 35.6% | 0.69 | 45 | 49 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=6
- `S06_combined_v1`: stop=15, time_stop=4
- `S03_pullback`: stop=21, time_stop=3
- `S10_defensive_no_ext`: stop=23, time_stop=7
- `S05_topk_no_ext`: stop=14, time_stop=4
- `S07_pullback_long`: stop=10, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=31, time_stop=14

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
_Generated 2026-09-17T23:43:04.492699+00:00 · paper only_
