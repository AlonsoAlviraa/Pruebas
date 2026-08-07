# Paper cloud multi-strategy — `2026-08-07`

**Window:** 2025-11-14 → 2026-08-06 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **14.38%** · Equal-weight names B&H **14.95%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 18.70% | +4.32% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.13% | +3.75% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 0.80% | -13.58% | 33.3% | 1.21 | 18 | 21 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.72% | -13.66% | 40.0% | 1.25 | 25 | 28 | no |
| 5 | `S03_pullback` | `pullback` | 0.42% | -13.96% | 37.0% | 1.10 | 27 | 28 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.30% | -14.08% | 38.7% | 1.12 | 31 | 34 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.25% | -14.63% | 39.5% | 1.03 | 38 | 43 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.33% | -14.72% | 41.9% | 0.96 | 43 | 49 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -1.02% | -15.40% | 25.0% | 0.75 | 20 | 23 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.70% | -16.08% | 25.0% | 0.34 | 16 | 18 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=12, time_stop=6
- `S02_no_extension`: stop=18, time_stop=7
- `S03_pullback`: stop=22, time_stop=5
- `S10_defensive_no_ext`: stop=22, time_stop=9
- `S04_qqq_gate`: stop=25, time_stop=13
- `S01_baseline_trend_mom`: stop=29, time_stop=14
- `S06_combined_v1`: stop=16, time_stop=4
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
_Generated 2026-08-07T01:05:36.100229+00:00 · paper only_
