# Paper cloud multi-strategy — `2026-10-07`

**Window:** 2026-01-16 → 2026-10-06 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.64%** · Equal-weight names B&H **17.91%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 20.87% | +8.23% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 20.20% | +7.56% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.62% | -12.02% | 38.9% | 1.05 | 18 | 22 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.61% | -12.03% | 36.4% | 1.13 | 22 | 26 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | -0.01% | -12.65% | 33.3% | 0.89 | 27 | 31 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.28% | -12.92% | 31.2% | 0.76 | 16 | 19 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.78% | -13.42% | 35.9% | 0.77 | 39 | 45 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.95% | -13.59% | 36.4% | 0.72 | 44 | 50 | no |
| 9 | `S07_pullback_long` | `combined_v2` | -1.29% | -13.93% | 38.5% | 0.61 | 13 | 16 | no |
| 10 | `S03_pullback` | `pullback` | -1.73% | -14.37% | 28.6% | 0.49 | 21 | 25 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=5
- `S02_no_extension`: stop=14, time_stop=8
- `S10_defensive_no_ext`: stop=19, time_stop=8
- `S05_topk_no_ext`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=30, time_stop=14
- `S07_pullback_long`: stop=9, time_stop=4
- `S03_pullback`: stop=17, time_stop=4

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
_Generated 2026-10-07T00:58:56.971998+00:00 · paper only_
