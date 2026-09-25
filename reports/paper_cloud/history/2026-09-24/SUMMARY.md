# Paper cloud multi-strategy — `2026-09-24`

**Window:** 2026-01-06 → 2026-09-24 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.89%** · Equal-weight names B&H **15.13%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 19.20% | +8.30% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 18.59% | +7.69% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 1.37% | -9.52% | 40.0% | 1.50 | 20 | 23 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.82% | -10.07% | 38.5% | 1.27 | 26 | 28 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.02% | -10.87% | 34.5% | 1.02 | 29 | 31 | no |
| 6 | `S07_pullback_long` | `combined_v2` | -0.97% | -11.86% | 28.6% | 0.58 | 14 | 18 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.99% | -11.88% | 28.9% | 0.72 | 38 | 42 | no |
| 8 | `S05_topk_no_ext` | `no_extension` | -1.35% | -12.25% | 26.3% | 0.69 | 19 | 21 | no |
| 9 | `S01_baseline_trend_mom` | `trend_mom` | -1.42% | -12.31% | 32.6% | 0.63 | 46 | 50 | no |
| 10 | `S03_pullback` | `pullback` | -1.83% | -12.72% | 26.9% | 0.61 | 26 | 29 | no |

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
_Generated 2026-09-25T00:00:14.113804+00:00 · paper only_
