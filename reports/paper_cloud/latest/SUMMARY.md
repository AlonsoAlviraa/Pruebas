# Paper cloud multi-strategy — `2026-10-22`

**Window:** 2026-02-02 → 2026-10-20 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.59%** · Equal-weight names B&H **13.92%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 15.57% | +4.98% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 15.08% | +4.49% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 0.60% | -9.98% | 43.5% | 1.18 | 23 | 28 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.50% | -10.09% | 44.4% | 1.08 | 18 | 22 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.05% | -10.54% | 35.7% | 1.05 | 28 | 30 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.27% | -10.86% | 37.5% | 0.78 | 16 | 19 | no |
| 7 | `S07_pullback_long` | `combined_v2` | -1.17% | -11.76% | 45.5% | 0.57 | 11 | 15 | no |
| 8 | `S03_pullback` | `pullback` | -1.28% | -11.87% | 27.8% | 0.43 | 18 | 23 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.63% | -12.22% | 37.0% | 0.62 | 46 | 52 | no |
| 10 | `S04_qqq_gate` | `qqq_gate` | -1.72% | -12.31% | 36.6% | 0.65 | 41 | 46 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=14, time_stop=9
- `S06_combined_v1`: stop=13, time_stop=5
- `S10_defensive_no_ext`: stop=18, time_stop=10
- `S05_topk_no_ext`: stop=12, time_stop=4
- `S07_pullback_long`: stop=6, time_stop=5
- `S03_pullback`: stop=14, time_stop=4
- `S01_baseline_trend_mom`: stop=32, time_stop=14
- `S04_qqq_gate`: stop=29, time_stop=12

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
_Generated 2026-10-22T00:10:25.867069+00:00 · paper only_
