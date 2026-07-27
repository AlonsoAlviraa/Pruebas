# Design: Levered annual study + mean multi-year ranking

**Status:** implemented  
**Date:** 2026-07-22  
**Capital:** VIRTUAL  

## Goal

Research leverage (1.5×–3×) with financing; rank by **mean return across years**; promote robust sleeves.

## Labels

- `levered_proxy` / `etf_levered_proxy` (daily reset)
- `levered_wipe_proxy` on hard DD / zero equity
- Never real TQQQ fills

## Run

```powershell
python -m pytest tests/test_leverage_models.py -q --tb=short
python scripts/run_levered_annual_study.py --out reports/levered_annual --lookback-days 2000
```

## Ranking

Primary: `mean_ret` across 2022–2025_study.  
GOOD filters: mean ≥ QQQ_BH mean +3pp OR xs_SPY ≥5pp; DD > −55%; ≥2 positive years; upside concentration ≤70%.
