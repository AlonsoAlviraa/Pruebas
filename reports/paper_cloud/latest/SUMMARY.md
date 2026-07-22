# Paper cloud multi-strategy — `2026-07-22`

**Window:** 2026-01-02 → 2026-07-21 · **Capital:** VIRTUAL $100,000 · **mode:** paper

**Data:** REAL free market (10/10 tickers) — `yahoo`

Free cloud batch (GitHub Actions). Not financial advice.

## Ranking by total return

| Rank | Strategy | Label | Return | Final $ | Entries | Exits | Commission | Kill |
|------|----------|-------|--------|---------|---------|-------|------------|------|
| 1 | `S07_high_vol_only` | High ATR band only | -1.59% | $98,407 | 7 | 5 | $12.00 | no |
| 2 | `S06_diversified` | Diversified 12 slots | -1.61% | $98,394 | 39 | 34 | $73.00 | no |
| 3 | `S08_low_vol_quality` | Lower vol / quality-ish | -2.42% | $97,581 | 37 | 32 | $69.00 | no |
| 4 | `S01_baseline_minalloc` | Baseline minalloc + regime | -2.62% | $97,380 | 39 | 34 | $73.00 | no |
| 5 | `S03_tight_stops` | Tight hard stop 5% | -3.64% | $96,361 | 49 | 48 | $97.00 | no |
| 6 | `S02_no_regime` | No regime filter | -3.84% | $96,155 | 47 | 40 | $87.00 | no |
| 7 | `S09_aggressive_entries` | More daily entries / shorter horizon | -4.65% | $95,347 | 72 | 66 | $138.00 | no |
| 8 | `S10_defensive` | Defensive: regime + tight risk | -5.02% | $94,977 | 47 | 46 | $93.00 | no |
| 9 | `S04_wide_stops` | Wide stop 12% / long hold | -5.36% | $94,642 | 25 | 21 | $46.00 | no |
| 10 | `S05_concentrated` | Concentrated 4 slots | -5.86% | $94,142 | 30 | 26 | $56.00 | no |

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
_Generated 2026-07-22T08:07:38.190561+00:00 · paper only_
