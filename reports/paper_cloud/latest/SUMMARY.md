# Paper cloud multi-strategy — `2026-11-14`

**Window:** 2026-02-27 → 2026-11-13 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **9.97%** · Equal-weight names B&H **13.04%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 12.40% | +2.43% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 10.49% | +0.52% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.87% | -9.10% | 50.0% | 1.27 | 20 | 20 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.30% | -9.67% | 37.9% | 1.13 | 29 | 29 | no |
| 5 | `S05_topk_no_ext` | `no_extension` | 0.27% | -9.70% | 47.1% | 1.10 | 17 | 17 | no |
| 6 | `S02_no_extension` | `no_extension` | 0.20% | -9.77% | 39.3% | 1.08 | 28 | 28 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.11% | -11.08% | 39.0% | 0.80 | 41 | 41 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.33% | -11.29% | 37.0% | 0.72 | 46 | 46 | no |
| 9 | `S03_pullback` | `pullback` | -1.55% | -11.52% | 27.3% | 0.59 | 22 | 22 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -2.23% | -12.20% | 33.3% | 0.39 | 15 | 15 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=7
- `S10_defensive_no_ext`: stop=18, time_stop=11
- `S05_topk_no_ext`: stop=11, time_stop=6
- `S02_no_extension`: stop=18, time_stop=10
- `S04_qqq_gate`: stop=27, time_stop=14
- `S01_baseline_trend_mom`: stop=30, time_stop=16
- `S03_pullback`: stop=16, time_stop=6
- `S07_pullback_long`: stop=9, time_stop=6

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
_Generated 2026-11-14T01:40:41.048923+00:00 · paper only_
