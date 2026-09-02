# Paper cloud multi-strategy — `2026-09-02`

**Window:** 2025-12-12 → 2026-09-02 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **12.23%** · Equal-weight names B&H **14.24%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 13.62% | +1.39% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 13.19% | +0.96% | n/a | n/a | 0 | 1 | no |
| 3 | `S02_no_extension` | `no_extension` | 1.18% | -11.05% | 47.6% | 1.55 | 21 | 26 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.68% | -11.55% | 43.3% | 1.23 | 30 | 34 | no |
| 5 | `S06_combined_v1` | `combined_v1` | 0.46% | -11.77% | 41.2% | 1.08 | 17 | 21 | no |
| 6 | `S07_pullback_long` | `combined_v2` | 0.19% | -12.04% | 37.5% | 1.01 | 16 | 20 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.43% | -12.66% | 40.5% | 0.94 | 37 | 41 | no |
| 8 | `S05_topk_no_ext` | `no_extension` | -0.61% | -12.84% | 29.4% | 0.76 | 17 | 20 | no |
| 9 | `S03_pullback` | `pullback` | -0.64% | -12.87% | 32.0% | 0.85 | 25 | 29 | no |
| 10 | `S01_baseline_trend_mom` | `trend_mom` | -0.90% | -13.13% | 41.9% | 0.85 | 43 | 46 | no |

## Exit reasons (per strategy)

- `S02_no_extension`: stop=17, time_stop=4
- `S10_defensive_no_ext`: stop=19, time_stop=11
- `S06_combined_v1`: stop=14, time_stop=3
- `S07_pullback_long`: stop=12, time_stop=4
- `S04_qqq_gate`: stop=25, time_stop=12
- `S05_topk_no_ext`: stop=14, time_stop=3
- `S03_pullback`: stop=23, time_stop=2
- `S01_baseline_trend_mom`: stop=29, time_stop=14

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
_Generated 2026-09-02T23:32:47.230652+00:00 · paper only_
