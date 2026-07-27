# Design: Options amplify mega study (implemented)

**Date:** 2026-07-22  
**Capital:** VIRTUAL  
**Labels:** historical marks = `proxy_bs|vix_surface` · chain = `yahoo_chain` today-only or `yahoo_chain_failed`

## Data

| Layer | Status |
|-------|--------|
| OHLCV + VIX/VIX3M | **EODHD EOD** (paid plan) → `reports/eodhd_options_pack/` (~27 tickers from 2020) |
| Yahoo options chains | 401 blocked (legacy path) |
| EODHD US options marketplace (UnicornBay) | **403 not subscribed** (`eodhd_options_not_subscribed`) |
| Option marks in backtest | still `proxy_bs\|vix_surface` until UnicornBay add-on |

## Code

- Debit kinds: `long_call`, `long_put`, `call_debit_spread`, `put_debit_spread`, `pmcc`
- Zoo: `zoo_options_amplify.json` (60 strategies)
- Download: `scripts/download_options_research_data.py`
- Matrix: `scripts/run_options_amplify_matrix.py`
- Report: `reports/options_amplify/latest/SUMMARY.md`

## Run

```powershell
python scripts/download_options_research_data.py --out reports/options_data_pack --lookback-days 2000
python scripts/build_options_amplify_zoo.py
python scripts/run_options_amplify_matrix.py --out reports/options_amplify --lookback-days 2000
```
