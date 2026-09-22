# Paper cloud multi-strategy — `2026-09-22`

**Window:** 2026-01-02 → 2026-09-22 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **13.20%** · Equal-weight names B&H **16.10%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 20.07% | +6.86% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 19.56% | +6.36% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.90% | -11.31% | 42.1% | 1.66 | 19 | 23 | no |
| 4 | `S02_no_extension` | `no_extension` | 1.16% | -12.04% | 41.7% | 1.44 | 24 | 28 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.22% | -12.98% | 35.7% | 1.07 | 28 | 31 | no |
| 6 | `S07_pullback_long` | `combined_v2` | -0.86% | -14.06% | 33.3% | 0.58 | 15 | 18 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.93% | -14.13% | 29.7% | 0.74 | 37 | 42 | no |
| 8 | `S05_topk_no_ext` | `no_extension` | -0.98% | -14.19% | 27.8% | 0.74 | 18 | 21 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.40% | -14.60% | 33.3% | 0.65 | 45 | 50 | no |
| 10 | `S03_pullback` | `pullback` | -1.75% | -14.95% | 26.9% | 0.61 | 26 | 28 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=15, time_stop=4
- `S02_no_extension`: stop=17, time_stop=7
- `S10_defensive_no_ext`: stop=20, time_stop=8
- `S07_pullback_long`: stop=11, time_stop=4
- `S04_qqq_gate`: stop=26, time_stop=11
- `S05_topk_no_ext`: stop=14, time_stop=4
- `S01_baseline_trend_mom`: stop=32, time_stop=13
- `S03_pullback`: stop=22, time_stop=4

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
_Generated 2026-09-22T23:46:47.721862+00:00 · paper only_
