# Paper cloud multi-strategy — `2026-10-20`

**Window:** 2026-01-29 → 2026-10-16 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **9.97%** · Equal-weight names B&H **13.05%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 11.82% | +1.85% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 11.51% | +1.54% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.39% | -9.58% | 44.4% | 1.03 | 18 | 22 | no |
| 4 | `S02_no_extension` | `no_extension` | 0.37% | -9.60% | 43.5% | 1.17 | 23 | 28 | no |
| 5 | `S10_defensive_no_ext` | `no_extension` | 0.01% | -9.96% | 35.7% | 1.02 | 28 | 30 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.39% | -10.37% | 37.5% | 0.74 | 16 | 19 | no |
| 7 | `S07_pullback_long` | `combined_v2` | -1.43% | -11.40% | 45.5% | 0.57 | 11 | 15 | no |
| 8 | `S03_pullback` | `pullback` | -1.47% | -11.45% | 27.8% | 0.43 | 18 | 23 | no |
| 9 | `S04_qqq_gate` | `qqq_gate` | -1.64% | -11.61% | 37.5% | 0.67 | 40 | 46 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -1.69% | -11.67% | 37.8% | 0.64 | 45 | 52 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=5
- `S02_no_extension`: stop=14, time_stop=9
- `S10_defensive_no_ext`: stop=18, time_stop=10
- `S05_topk_no_ext`: stop=12, time_stop=4
- `S07_pullback_long`: stop=6, time_stop=5
- `S03_pullback`: stop=14, time_stop=4
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
_Generated 2026-10-20T00:34:37.859341+00:00 · paper only_
