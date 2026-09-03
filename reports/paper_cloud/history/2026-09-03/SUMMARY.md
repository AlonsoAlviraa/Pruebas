# Paper cloud multi-strategy — `2026-09-03`

**Window:** 2025-12-15 → 2026-09-03 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **13.58%** · Equal-weight names B&H **16.33%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 15.63% | +2.05% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 15.14% | +1.56% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.51% | -12.07% | 47.6% | 1.58 | 21 | 26 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.97% | -12.61% | 41.2% | 1.14 | 17 | 21 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.92% | -12.66% | 42.9% | 1.25 | 28 | 32 | no |
| 6 | `S07_pullback_long` | `combined_v2` | 0.49% | -13.09% | 37.5% | 1.01 | 16 | 20 | no |
| 7 | `S05_topk_no_ext` | `no_extension` | -0.29% | -13.87% | 29.4% | 0.77 | 17 | 20 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -0.33% | -13.91% | 37.8% | 0.93 | 37 | 41 | no |
| 9 | `S03_pullback` | `pullback` | -0.40% | -13.98% | 32.0% | 0.85 | 25 | 29 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -0.70% | -14.28% | 40.5% | 0.88 | 42 | 45 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S06_combined_v1`: stop=14, time_stop=3
- `S10_defensive_no_ext`: stop=18, time_stop=10
- `S07_pullback_long`: stop=12, time_stop=4
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=26, time_stop=11
- `S03_pullback`: stop=23, time_stop=2
- `S01_baseline_trend_mom`: stop=29, time_stop=13

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
_Generated 2026-09-03T23:29:52.832706+00:00 · paper only_
