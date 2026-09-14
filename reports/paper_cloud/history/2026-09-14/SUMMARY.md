# Paper cloud multi-strategy — `2026-09-14`

**Window:** 2025-12-23 → 2026-09-14 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.60%** · Equal-weight names B&H **13.08%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.32% | +3.72% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 13.86% | +3.26% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.05% | -9.55% | 45.8% | 1.51 | 24 | 29 | no |
| 4 | `S06_combined_v1` | `combined_v1` | 0.13% | -10.47% | 41.2% | 1.03 | 17 | 20 | no |
| 5 | `S03_pullback` | `pullback` | -0.32% | -10.92% | 30.4% | 0.84 | 23 | 28 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | -0.58% | -11.18% | 33.3% | 0.83 | 30 | 31 | no |
| 7 | `S05_topk_no_ext` | `no_extension` | -0.84% | -11.44% | 25.0% | 0.72 | 16 | 18 | no |
| 8 | `S07_pullback_long` | `combined_v2` | -0.90% | -11.50% | 28.6% | 0.56 | 14 | 18 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -1.24% | -11.84% | 32.5% | 0.77 | 40 | 44 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.24% | -11.84% | 37.8% | 0.76 | 45 | 51 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=18, time_stop=6
- `S06_combined_v1`: stop=14, time_stop=3
- `S03_pullback`: stop=21, time_stop=2
- `S10_defensive_no_ext`: stop=22, time_stop=8
- `S05_topk_no_ext`: stop=13, time_stop=3
- `S07_pullback_long`: stop=10, time_stop=4
- `S04_qqq_gate`: stop=28, time_stop=12
- `S01_baseline_trend_mom`: stop=31, time_stop=14

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
_Generated 2026-09-14T23:59:00.076079+00:00 · paper only_
