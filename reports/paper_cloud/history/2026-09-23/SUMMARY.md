# Paper cloud multi-strategy — `2026-09-23`

**Window:** 2026-01-05 → 2026-09-23 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.65%** · Equal-weight names B&H **14.23%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 19.20% | +7.55% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.61% | +6.96% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.41% | -10.23% | 40.0% | 1.49 | 20 | 23 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.85% | -10.80% | 38.5% | 1.27 | 26 | 28 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.05% | -11.60% | 34.5% | 1.02 | 29 | 31 | no |
| 6 | `S07_pullback_long` | `combined_v2` | -1.06% | -12.71% | 28.6% | 0.58 | 14 | 18 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.11% | -12.75% | 28.9% | 0.72 | 38 | 42 | no |
| 8 | `S05_topk_no_ext` | `no_extension` | -1.31% | -12.95% | 26.3% | 0.69 | 19 | 21 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.52% | -13.16% | 32.6% | 0.63 | 46 | 50 | no |
| 10 | `S03_pullback` | `pullback` | -1.88% | -13.53% | 26.9% | 0.60 | 26 | 29 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=16, time_stop=4
- `S02_no_extension`: stop=18, time_stop=8
- `S10_defensive_no_ext`: stop=21, time_stop=8
- `S07_pullback_long`: stop=10, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=11
- `S05_topk_no_ext`: stop=15, time_stop=4
- `S01_baseline_trend_mom`: stop=33, time_stop=13
- `S03_pullback`: stop=22, time_stop=4

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
_Generated 2026-09-23T23:57:22.825406+00:00 · paper only_
