# Paper cloud multi-strategy — `2026-09-16`

**Window:** 2025-12-26 → 2026-09-16 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **9.23%** · Equal-weight names B&H **11.95%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 12.43% | +3.20% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 12.11% | +2.88% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.07% | -8.17% | 41.7% | 1.53 | 24 | 27 | no |
| 4 | `S06_combined_v1` | `combined_v1` | -0.20% | -9.43% | 36.8% | 1.06 | 19 | 20 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | -0.25% | -9.48% | 33.3% | 0.94 | 30 | 31 | no |
| 6 | `S03_pullback` | `pullback` | -0.64% | -9.87% | 32.0% | 0.87 | 25 | 28 | no |
| 7 | `S05_topk_no_ext` | `no_extension` | -1.23% | -10.46% | 26.3% | 0.72 | 19 | 19 | no |
| 8 | `S07_pullback_long` | `combined_v2` | -1.26% | -10.49% | 28.6% | 0.56 | 14 | 18 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -1.55% | -10.79% | 33.3% | 0.73 | 39 | 42 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.58% | -10.81% | 35.6% | 0.68 | 45 | 49 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=7
- `S06_combined_v1`: stop=15, time_stop=4
- `S10_defensive_no_ext`: stop=22, time_stop=8
- `S03_pullback`: stop=22, time_stop=3
- `S05_topk_no_ext`: stop=15, time_stop=4
- `S07_pullback_long`: stop=10, time_stop=4
- `S04_qqq_gate`: stop=27, time_stop=12
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
_Generated 2026-09-16T23:49:20.506121+00:00 · paper only_
