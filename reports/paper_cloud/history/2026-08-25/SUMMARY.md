# Paper cloud multi-strategy — `2026-08-25`

**Window:** 2025-12-04 → 2026-08-25 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.91%** · Equal-weight names B&H **11.36%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 13.32% | +1.41% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 12.98% | +1.07% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.93% | -10.99% | 42.9% | 1.42 | 21 | 27 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.76% | -11.15% | 42.9% | 1.33 | 28 | 31 | no |
| 5 | `S07_pullback_long` | `combined_v2` | 0.11% | -11.80% | 31.2% | 1.04 | 16 | 18 | no |
| 6 | `S06_combined_v1` | `combined_v1` | -0.10% | -12.01% | 35.3% | 0.97 | 17 | 21 | no |
| 7 | `S03_pullback` | `pullback` | -0.66% | -12.57% | 34.6% | 0.89 | 26 | 28 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -0.67% | -12.58% | 33.3% | 0.89 | 36 | 41 | no |
| 9 | `S05_topk_no_ext` | `no_extension` | -1.19% | -13.10% | 23.5% | 0.69 | 17 | 20 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.38% | -13.29% | 33.3% | 0.74 | 42 | 49 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S07_pullback_long`: stop=12, time_stop=4
- `S06_combined_v1`: stop=14, time_stop=3
- `S03_pullback`: stop=23, time_stop=3
- `S04_qqq_gate`: stop=24, time_stop=12
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S01_baseline_trend_mom`: stop=31, time_stop=11

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
_Generated 2026-08-25T21:56:41.982168+00:00 · paper only_
