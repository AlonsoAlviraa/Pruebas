# Paper cloud multi-strategy — `2026-08-21`

**Window:** 2025-12-02 → 2026-08-21 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.35%** · Equal-weight names B&H **11.28%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.79% | +2.44% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.32% | +1.97% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.90% | -11.45% | 40.9% | 1.37 | 22 | 28 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.79% | -11.57% | 42.9% | 1.32 | 28 | 32 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.05% | -12.30% | 35.3% | 0.97 | 17 | 21 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.58% | -12.93% | 29.4% | 0.79 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.68% | -13.03% | 33.3% | 0.89 | 36 | 41 | no |
| 8 | `S03_pullback` | `pullback` | -0.94% | -13.30% | 30.8% | 0.83 | 26 | 28 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.26% | -13.61% | 31.0% | 0.75 | 42 | 49 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.83% | -14.18% | 29.4% | 0.54 | 17 | 19 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=5
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S06_combined_v1`: stop=14, time_stop=3
- `S05_topk_no_ext`: stop=13, time_stop=4
- `S04_qqq_gate`: stop=24, time_stop=12
- `S03_pullback`: stop=23, time_stop=3
- `S01_baseline_trend_mom`: stop=30, time_stop=12
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
_Generated 2026-08-21T21:52:46.541441+00:00 · paper only_
