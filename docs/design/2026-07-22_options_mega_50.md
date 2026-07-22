# Design: Mega paper test ~50 options strategies

**Status:** implement  
**Date:** 2026-07-22  
**Label:** `proxy_bs` only for historical marks · VIRTUAL capital

## Research harvest

### Academic / industry benchmarks
| Source | Strategy family |
|--------|-----------------|
| CBOE BXM / BXY | Covered call ATM / mild OTM |
| CBOE PUT / WPUT | Cash-secured / ATM put-write |
| CBOE CNDR | Iron condor short ~15Δ / wings ~5Δ |
| Quantpedia VRP | Sell premium when IV > RV; OTM put-write 5–10% |
| arXiv put-writing sizing | OTM depth + Kelly/VIX sizing (sizing simplified here) |

### Twitter / X themes
- Iron condor is **positioning** not magic; strike selection matters
- Equity drift: often **skew call side** or prefer put credit spreads over symmetric IC
- CNDR weakens when realized > implied (chop / range break)
- One-roll max discipline (not modeled; we roll by DTE only)

### GitHub style
- Parametric grids (DTE × OTM × underlying) over few structure kinds — used to reach ~50 configs without inventing fake edges

## Zoo
- File: `paper_live/cloud/zoo_options_50.json` (built by `scripts/build_options_zoo_50.py`)
- Kinds: cash, covered_call, cash_secured_put, put_credit_spread, call_credit_spread, iron_condor, collar, protective_put

## Engine extensions
- `call_credit_spread`, `iron_condor`, `protective_put` in `replay_options.py`
- Margin for IC = larger wing width × 100

## Mega test protocol
```bash
python scripts/build_options_zoo_50.py
python scripts/run_paper_options_batch.py \
  --zoo paper_live/cloud/zoo_options_50.json \
  --out reports/paper_options_mega \
  --start 2025-10-29
```
Optional: `--stress` on a subset later.

## Success
- ≥48 strategies complete without crash
- SUMMARY ranked by return with vs SPY, CVaR, kill
- Report `reports/paper_options_mega/MEGA_RESULTS.md` with families + honesty

## Honesty
Not OPRA. Not live. Not 50 unique alpha ideas — many are **parameterizations** of known CBOE/VRP structures.
