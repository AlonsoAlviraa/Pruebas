# Design: Options next cycle — scorecard + VRP gates + book (PR-OPT-N1…N6)

**Status:** implemented (research paper stack)  
**Date:** 2026-07-22  
**Capital:** VIRTUAL only  

## Goal

Move from “credible options harness” to **decision protocol**: which sleeves deserve more research.

| PR | Deliverable |
|----|-------------|
| **N1** | `scorecard.py` + `scripts/score_options_matrix.py` → promote / watch / kill |
| **N2** | IV rank / VRP proxy / VIX term contango gates + zoo OPT_TA13–18 |
| **N3** | Names zoo default in matrix; defined-risk AAPL/NVDA VRP variants |
| **N4** | Time exit (DTE + residual credit) + exit breakdown |
| **N5** | Beta-weighted delta + sleeve portfolio paper |
| **N6** | Optional chain vs model diagnostic (`--chain-diag`) |

## Labels (unchanged honesty)

| Label | Meaning |
|-------|---------|
| `proxy_bs\|vix_surface` | BS marks on VIX surface IV |
| `proxy_hv` | HV×mult fallback |
| `vrp_proxy` | surface ATM IV − HV20 (**not** true exchange VRP) |
| `iv_rank` | VIX (or HV) percentile lookback 252 |
| `beta_weighted_delta` | approx BS delta × rolling 60d beta to SPY |
| `assignment_proxy` | simplified equity assignment |
| `yahoo_chain_failed` | chain diag network miss — never invent |

## Scorecard rules (default)

Config: `paper_live/cloud/scorecard_options_config.json`

- **KILL:** stress return < cash **and** worse than cash in ≥2 bull windows; or worst maxDD ≤ −25%; or hard_kill ≥ 2 windows
- **PROMOTE_RESEARCH:** defined-risk **and** beat cash in ≥3 calendar windows **and** stress maxDD > −20%
- **WATCH:** bear OK + stress contained + lags SPY in bull (income profile)
- **HOLD:** cash control or no strong signal

Output: `reports/paper_options_ta_matrix/SCORECARD.md` + `.json`

## Verification

```powershell
python -m pytest tests/ -q --tb=short -k "paper_options"
python scripts/score_options_matrix.py --in reports/paper_options_ta_matrix/latest
# full matrix (heavy):
# python scripts/run_options_ta_matrix.py --out reports/paper_options_ta_matrix
```

## Non-goals

OPRA history, live routing, claiming true single-name IV rank from VIX alone without disclaimer.
