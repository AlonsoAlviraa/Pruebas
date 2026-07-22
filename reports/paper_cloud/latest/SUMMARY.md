# Paper cloud multi-strategy — `2026-07-22`

**Window:** 2025-10-29 → 2026-07-21 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Label | Return | Final $ | Entries | Exits | Commission | Kill |
|------|----------|-------|--------|---------|---------|-------|------------|------|
| 1 | `S07_high_vol_only` | High ATR band only | -3.08% | $96,920 | 13 | 11 | $24.00 | no |
| 2 | `S06_diversified` | Diversified 12 slots | -3.41% | $96,588 | 62 | 57 | $119.00 | no |
| 3 | `S08_low_vol_quality` | Lower vol / quality-ish | -5.09% | $94,907 | 58 | 53 | $111.00 | no |
| 4 | `S01_baseline_minalloc` | Baseline minalloc + regime | -5.23% | $94,771 | 62 | 57 | $119.00 | no |
| 5 | `S03_tight_stops` | Tight hard stop 5% | -6.08% | $93,920 | 77 | 76 | $153.00 | no |
| 6 | `S02_no_regime` | No regime filter | -6.31% | $93,695 | 69 | 62 | $131.00 | no |
| 7 | `S04_wide_stops` | Wide stop 12% / long hold | -7.52% | $92,479 | 35 | 31 | $66.00 | no |
| 8 | `S09_aggressive_entries` | More daily entries / shorter horizon | -8.06% | $91,936 | 104 | 98 | $202.00 | no |
| 9 | `S10_defensive` | Defensive: regime + tight risk | -8.61% | $91,395 | 71 | 70 | $141.00 | no |
| 10 | `S05_concentrated` | Concentrated 4 slots | -9.11% | $90,892 | 42 | 38 | $80.00 | no |

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

See `strategies/<id>/dashboard.html` and `daily/`.

---
_Generated 2026-07-22T07:30:05.327147+00:00 · paper only_
