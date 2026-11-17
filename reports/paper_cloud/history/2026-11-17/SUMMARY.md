# Paper cloud multi-strategy — `2026-11-17`

**Window:** 2026-03-02 → 2026-11-16 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **7.80%** · Equal-weight names B&H **10.18%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 10.19% | +2.39% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 8.04% | +0.24% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.65% | -6.16% | 50.0% | 1.55 | 20 | 20 | no |
| 4 | `S05_topk_no_ext` | `no_extension` | 0.87% | -6.94% | 50.0% | 1.32 | 18 | 18 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.48% | -7.32% | 39.3% | 1.22 | 28 | 28 | no |
| 6 | `S02_no_extension` | `no_extension` | 0.25% | -7.55% | 40.7% | 1.10 | 27 | 27 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.71% | -8.52% | 40.0% | 0.87 | 40 | 40 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.06% | -8.86% | 37.8% | 0.77 | 45 | 45 | no |
| 9 | `S03_pullback` | `pullback` | -1.55% | -9.36% | 27.3% | 0.59 | 22 | 22 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -2.29% | -10.09% | 33.3% | 0.38 | 15 | 15 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=7
- `S05_topk_no_ext`: stop=13, time_stop=5
- `S10_defensive_no_ext`: stop=17, time_stop=11
- `S02_no_extension`: stop=18, time_stop=9
- `S04_qqq_gate`: stop=26, time_stop=14
- `S01_baseline_trend_mom`: stop=29, time_stop=16
- `S03_pullback`: stop=16, time_stop=6
- `S07_pullback_long`: stop=10, time_stop=5

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
_Generated 2026-11-17T04:08:45.890312+00:00 · paper only_
