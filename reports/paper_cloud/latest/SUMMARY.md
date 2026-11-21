# Paper cloud multi-strategy — `2026-11-21`

**Window:** 2026-03-06 → 2026-11-20 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

**Benchmarks:** SPY B&H **2.59%** · Equal-weight names B&H **3.14%**

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Mode | Return | vs SPY | WR | PF | Closed | Entries | Kill |
|------|----------|------|--------|--------|----|----|--------|---------|------|
| 1 | `S06_combined_v1` | `combined_v1` | 1.65% | -0.94% | 50.0% | 1.55 | 20 | 20 | no |
| 2 | `S09_qqq_bh_proxy` | `qqq_hold` | 1.27% | -1.33% | n/a | n/a | 0 | 1 | no |
| 3 | `S05_topk_no_ext` | `no_extension` | 0.87% | -1.72% | 50.0% | 1.32 | 18 | 18 | no |
| 4 | `S10_defensive_no_ext` | `no_extension` | 0.48% | -2.11% | 39.3% | 1.22 | 28 | 28 | no |
| 5 | `S02_no_extension` | `no_extension` | 0.25% | -2.34% | 40.7% | 1.10 | 27 | 27 | no |
| 6 | `S08_qqq_hold_regime` | `qqq_hold` | -0.22% | -2.81% | n/a | n/a | 0 | 1 | no |
| 7 | `S04_qqq_gate` | `qqq_gate` | -0.71% | -3.30% | 40.0% | 0.87 | 40 | 40 | no |
| 8 | `S01_baseline_trend_mom` | `trend_mom` | -1.06% | -3.65% | 37.8% | 0.77 | 45 | 45 | no |
| 9 | `S03_pullback` | `pullback` | -1.55% | -4.14% | 27.3% | 0.59 | 22 | 22 | no |
| 10 | `S07_pullback_long` | `combined_v2` | -2.29% | -4.88% | 33.3% | 0.38 | 15 | 15 | no |

## Exit reasons (per strategy)

- `S06_combined_v1`: stop=13, time_stop=7
- `S05_topk_no_ext`: stop=13, time_stop=5
- `S10_defensive_no_ext`: stop=17, time_stop=11
- `S02_no_extension`: stop=18, time_stop=9
- `S04_qqq_gate`: stop=26, time_stop=14
- `S01_baseline_trend_mom`: stop=29, time_stop=16
- `S03_pullback`: stop=16, time_stop=6
- `S07_pullback_long`: stop=10, time_stop=5

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
_Generated 2026-11-21T02:44:56.789094+00:00 · paper only_
