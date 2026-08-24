# Paper cloud multi-strategy — `2026-08-24`

**Window:** 2025-12-03 → 2026-08-24 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **11.64%** · Equal-weight names B&H **11.46%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 13.64% | +2.00% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 13.21% | +1.57% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.04% | -10.59% | 42.9% | 1.45 | 21 | 27 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.92% | -10.71% | 42.9% | 1.36 | 28 | 31 | no |
| 5 | `S07_pullback_long` | `combined_v2` | -0.03% | -11.66% | 31.2% | 0.99 | 16 | 18 | no |
| 6 | `S06_combined_v1` | `combined_v1` | -0.06% | -11.70% | 35.3% | 0.97 | 17 | 21 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.69% | -12.32% | 33.3% | 0.89 | 36 | 41 | no |
| 8 | `S03_pullback` | `pullback` | -0.70% | -12.34% | 34.6% | 0.87 | 26 | 28 | no |
| 9 | `S05_topk_no_ext` | `no_extension` | -1.09% | -12.73% | 23.5% | 0.71 | 17 | 20 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.33% | -12.97% | 31.7% | 0.73 | 41 | 48 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=19, time_stop=9
- `S07_pullback_long`: stop=12, time_stop=4
- `S06_combined_v1`: stop=14, time_stop=3
- `S04_qqq_gate`: stop=24, time_stop=12
- `S03_pullback`: stop=23, time_stop=3
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S01_baseline_trend_mom`: stop=30, time_stop=11

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
_Generated 2026-08-24T22:01:59.438475+00:00 · paper only_
