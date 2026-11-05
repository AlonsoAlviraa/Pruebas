# Paper cloud multi-strategy — `2026-11-05`

**Window:** 2026-02-17 → 2026-11-03 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **10.47%** · Equal-weight names B&H **14.30%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S09_qqq_bh_proxy` | `qqq_hold` | 14.11% | +3.63% | n/a | n/a | 0 | 1 | no |
| 2 | `S08_qqq_hold_regime` | `qqq_hold` | 10.67% | +0.20% | n/a | n/a | 0 | 1 | no |
| 3 | `S06_combined_v1` | `combined_v1` | 0.87% | -9.60% | 50.0% | 1.27 | 20 | 20 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.06% | -10.41% | 36.7% | 1.04 | 30 | 30 | no |
| 5 | `S02_no_extension` | `no_extension` | -0.10% | -10.57% | 37.9% | 0.99 | 29 | 29 | no |
| 6 | `S05_topk_no_ext` | `no_extension` | -0.11% | -10.58% | 44.4% | 0.98 | 18 | 18 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -1.11% | -11.58% | 39.0% | 0.80 | 41 | 41 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.62% | -12.09% | 37.5% | 0.69 | 48 | 49 | no |
| 9 | `S03_pullback` | `pullback` | -1.97% | -12.44% | 26.1% | 0.53 | 23 | 23 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -2.04% | -12.51% | 35.7% | 0.41 | 14 | 15 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=7
- `S10_defensive_no_ext`: stop=19, time_stop=11
- `S02_no_extension`: stop=19, time_stop=10
- `S05_topk_no_ext`: stop=12, time_stop=6
- `S04_qqq_gate`: stop=27, time_stop=14
- `S01_baseline_trend_mom`: stop=32, time_stop=16
- `S03_pullback`: stop=17, time_stop=6
- `S07_pullback_long`: stop=8, time_stop=6

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
_Generated 2026-11-05T01:45:11.963024+00:00 · paper only_
