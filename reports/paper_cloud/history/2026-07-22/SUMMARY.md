# Paper cloud multi-strategy — `2026-07-22`

**Window:** 2025-11-28 → 2026-05-29 · **Capital:** VIRTUAL $100,000 · **mode:** paper

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Label | Return | Final $ | Entries | Exits | Commission | Kill |
|------|----------|-------|--------|---------|---------|-------|------------|------|
| 1 | `S07_high_vol_only` | High ATR band only | 0.57% | $100,571 | 2 | 2 | $4.00 | YES |
| 2 | `S10_defensive` | Defensive: regime + tight risk | 0.38% | $100,379 | 2 | 2 | $4.00 | YES |
| 3 | `S03_tight_stops` | Tight hard stop 5% | 0.29% | $100,286 | 2 | 2 | $4.00 | YES |
| 4 | `S08_low_vol_quality` | Lower vol / quality-ish | 0.06% | $100,057 | 2 | 2 | $4.00 | YES |
| 5 | `S04_wide_stops` | Wide stop 12% / long hold | 0.01% | $100,009 | 2 | 2 | $4.00 | YES |
| 6 | `S06_diversified` | Diversified 12 slots | -0.10% | $99,901 | 2 | 2 | $4.00 | YES |
| 7 | `S01_baseline_minalloc` | Baseline minalloc + regime | -0.14% | $99,857 | 2 | 2 | $4.00 | YES |
| 8 | `S05_concentrated` | Concentrated 4 slots | -0.45% | $99,549 | 2 | 2 | $4.00 | YES |
| 9 | `S09_aggressive_entries` | More daily entries / shorter horizon | -0.57% | $99,435 | 14 | 14 | $28.00 | YES |
| 10 | `S02_no_regime` | No regime filter | -0.64% | $99,360 | 8 | 8 | $16.00 | YES |

## Data sources

- `AAPL`: synthetic_gapfill
- `AMZN`: synthetic_gapfill
- `GOOGL`: synthetic_gapfill
- `JPM`: synthetic_gapfill
- `META`: synthetic_gapfill
- `MSFT`: synthetic_gapfill
- `NVDA`: synthetic_gapfill
- `QQQ`: synthetic_gapfill
- `SPY`: synthetic_gapfill
- `XOM`: synthetic_gapfill

## Per-strategy digests

See `strategies/<id>/dashboard.html` and `daily/`.

---
_Generated 2026-07-22T07:06:37.691335+00:00 · paper only_
