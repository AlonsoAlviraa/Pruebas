# Paper cloud multi-strategy — `2026-07-27`

**Window:** 2025-11-04 → 2026-07-27 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **9.46%** · Equal-weight names B&H **7.27%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 9.18% | -0.28% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 8.89% | -0.57% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 0.88% | -8.57% | 38.9% | 1.39 | 18 | 19 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.76% | -8.70% | 44.0% | 1.36 | 25 | 26 | no |
| 5 | `S03_pullback` | `pullback` | 0.74% | -8.72% | 32.0% | 0.81 | 25 | 28 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.23% | -9.22% | 39.4% | 1.10 | 33 | 33 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.31% | -9.77% | 35.0% | 0.92 | 40 | 43 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.69% | -10.15% | 34.8% | 0.84 | 46 | 49 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -1.06% | -10.52% | 35.0% | 0.82 | 20 | 21 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -2.02% | -11.48% | 25.0% | 0.41 | 16 | 18 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=13, time_stop=5
- `S02_no_extension`: stop=19, time_stop=6
- `S03_pullback`: stop=21, time_stop=4
- `S10_defensive_no_ext`: stop=24, time_stop=9
- `S04_qqq_gate`: stop=27, time_stop=13
- `S01_baseline_trend_mom`: stop=33, time_stop=13
- `S06_combined_v1`: stop=17, time_stop=3
- `S07_pullback_long`: stop=13, time_stop=3

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
_Generated 2026-07-27T22:36:12.418329+00:00 · paper only_
