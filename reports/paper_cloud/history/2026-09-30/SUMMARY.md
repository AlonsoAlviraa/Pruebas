# Paper cloud multi-strategy — `2026-09-30`

**Window:** 2026-01-08 → 2026-09-28 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.04%** · Equal-weight names B&H **14.43%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 17.76% | +6.72% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 17.19% | +6.16% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 0.83% | -10.20% | 33.3% | 1.16 | 18 | 21 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.70% | -10.34% | 36.0% | 1.18 | 25 | 28 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.26% | -10.78% | 36.8% | 1.00 | 19 | 22 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | 0.15% | -10.89% | 34.5% | 1.01 | 29 | 32 | no |
| 7 | `S07_pullback_long` | `combined_v2` | -1.07% | -12.11% | 28.6% | 0.61 | 14 | 18 | no |
| 8 | `S04_qqq_gate` | `qqq_gate` | -1.15% | -12.19% | 30.8% | 0.74 | 39 | 43 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.30% | -12.34% | 33.3% | 0.71 | 45 | 49 | no |
| 10 | `S03_pullback` | `pullback` | -1.71% | -12.74% | 24.0% | 0.62 | 25 | 29 | no |

## Exit reasons (per strategy)

- `S05_topk_no_ext`: stop=14, time_stop=4
- `S02_no_extension`: stop=18, time_stop=7
- `S06_combined_v1`: stop=14, time_stop=5
- `S10_defensive_no_ext`: stop=21, time_stop=8
- `S07_pullback_long`: stop=10, time_stop=4
- `S04_qqq_gate`: stop=28, time_stop=11
- `S01_baseline_trend_mom`: stop=32, time_stop=13
- `S03_pullback`: stop=21, time_stop=4

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
_Generated 2026-09-30T00:44:02.553958+00:00 · paper only_
