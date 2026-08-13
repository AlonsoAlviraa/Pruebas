# Paper cloud multi-strategy — `2026-08-13`

**Window:** 2025-11-21 → 2026-08-13 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **18.03%** · Equal-weight names B&H **17.87%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 23.94% | +5.91% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.47% | +0.44% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 1.88% | -16.15% | 43.8% | 1.56 | 16 | 19 | no |
| 4 | `S02_no_extension` | `no_extension` | 1.40% | -16.63% | 45.5% | 1.56 | 22 | 26 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.85% | -17.18% | 42.9% | 1.36 | 28 | 32 | no |
| 6 | `S06_combined_v1` | `combined_v1` | 0.07% | -17.97% | 35.3% | 0.97 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.34% | -18.38% | 34.3% | 0.94 | 35 | 40 | no |
| 8 | `S03_pullback` | `pullback` | -0.51% | -18.54% | 37.0% | 0.94 | 27 | 29 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -0.69% | -18.72% | 36.6% | 0.88 | 41 | 48 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.63% | -19.67% | 29.4% | 0.57 | 17 | 19 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=11, time_stop=5
- `S02_no_extension`: stop=16, time_stop=6
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S06_combined_v1`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=23, time_stop=12
- `S03_pullback`: stop=23, time_stop=4
- `S01_baseline_trend_mom`: stop=29, time_stop=12
- `S07_pullback_long`: stop=12, time_stop=5

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
_Generated 2026-08-13T22:22:45.729003+00:00 · paper only_
