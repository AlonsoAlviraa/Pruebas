# Paper cloud multi-strategy — `2026-08-03`

**Window:** 2025-11-11 → 2026-08-03 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.93%** · Equal-weight names B&H **11.47%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 12.42% | +1.49% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 12.11% | +1.17% | n/a | n/a | 0 | 1 | no |
| 3 | `S03_pullback` | `pullback` | 0.53% | -10.40% | 37.0% | 1.13 | 27 | 27 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.42% | -10.52% | 38.5% | 1.14 | 26 | 26 | no |
| 5 | `S05_topk_no_ext` | `no_extension` | 0.37% | -10.57% | 31.6% | 1.10 | 19 | 19 | no |
| 6 | `S10_defensive_no_ext` | `no_extension` | -0.04% | -10.98% | 36.4% | 1.01 | 33 | 33 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.56% | -11.49% | 33.3% | 0.88 | 39 | 40 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -0.81% | -11.75% | 37.0% | 0.84 | 46 | 46 | no |
| 9 | `S06_combined_v1` | `combined_v1` | -1.57% | -12.50% | 28.6% | 0.67 | 21 | 21 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -1.58% | -12.51% | 20.0% | 0.42 | 15 | 17 | no |

## Exit reasons (per strategy)

- `S03_pullback`: stop=22, time_stop=5
- `S02_no_extension`: stop=20, time_stop=6
- `S05_topk_no_ext`: stop=14, time_stop=5
- `S10_defensive_no_ext`: stop=24, time_stop=9
- `S04_qqq_gate`: stop=27, time_stop=12
- `S01_baseline_trend_mom`: stop=32, time_stop=14
- `S06_combined_v1`: stop=18, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=3

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
_Generated 2026-08-03T22:36:05.748397+00:00 · paper only_
