# Paper cloud multi-strategy — `2026-09-04`

**Window:** 2025-12-16 → 2026-09-04 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **13.45%** · Equal-weight names B&H **15.74%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 17.76% | +4.31% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 14.78% | +1.33% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.43% | -12.02% | 47.6% | 1.63 | 21 | 26 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 1.09% | -12.37% | 44.4% | 1.39 | 27 | 31 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.80% | -12.65% | 41.2% | 1.14 | 17 | 21 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.28% | -13.73% | 29.4% | 0.79 | 17 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.63% | -14.08% | 37.8% | 0.93 | 37 | 43 | no |
| 8 | `S03_pullback` | `pullback` | -0.72% | -14.18% | 29.2% | 0.84 | 24 | 28 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -0.98% | -14.43% | 40.5% | 0.87 | 42 | 49 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.64% | -15.09% | 31.2% | 0.53 | 16 | 20 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=18, time_stop=9
- `S06_combined_v1`: stop=14, time_stop=3
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=26, time_stop=11
- `S03_pullback`: stop=22, time_stop=2
- `S01_baseline_trend_mom`: stop=29, time_stop=13
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
_Generated 2026-09-04T23:14:17.887093+00:00 · paper only_
