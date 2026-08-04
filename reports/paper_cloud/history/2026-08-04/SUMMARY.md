# Paper cloud multi-strategy — `2026-08-04`

**Window:** 2025-11-12 → 2026-08-04 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.87%** · Equal-weight names B&H **13.25%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 15.39% | +2.52% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.99% | +2.12% | n/a | n/a | 0 | 1 | no |
| 3 | `S03_pullback` | `pullback` | 0.53% | -12.34% | 37.0% | 1.13 | 27 | 27 | no |
| 4 | `S05_topk_no_ext` | `no_extension` | 0.44% | -12.43% | 31.6% | 1.12 | 19 | 19 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.44% | -12.43% | 38.5% | 1.15 | 26 | 26 | no |
| 6 | `S06_combined_v1` | `combined_v1` | 0.22% | -12.65% | 38.1% | 1.06 | 21 | 21 | no |
| 7 | `S10_defensive_no_ext` | `no_extension` | -0.06% | -12.93% | 36.4% | 1.00 | 33 | 33 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.14% | -14.01% | 37.0% | 0.77 | 46 | 46 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -1.20% | -14.07% | 32.5% | 0.76 | 40 | 41 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.48% | -14.35% | 25.0% | 0.43 | 16 | 17 | no |

## Exit reasons (per strategy)

- `S03_pullback`: stop=22, time_stop=5
- `S05_topk_no_ext`: stop=13, time_stop=6
- `S02_no_extension`: stop=19, time_stop=7
- `S06_combined_v1`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=24, time_stop=9
- `S01_baseline_trend_mom`: stop=33, time_stop=13
- `S04_qqq_gate`: stop=29, time_stop=11
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
_Generated 2026-08-04T22:39:07.642237+00:00 · paper only_
