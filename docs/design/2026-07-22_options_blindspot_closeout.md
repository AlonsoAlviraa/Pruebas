# Design: Options blind-spot closeout (P0–P2)

**Status:** implemented (research paper stack)  
**Date:** 2026-07-22  
**Capital:** VIRTUAL only  

## Problem

Paper options marks used **IV = HV × 1.15** (`iv_proxy_from_hv`). That embeds a hand-crafted VRP and cannot validate market IV edges. Blind spots listed in `reports/OPTIONS_FOCUS_BLINDSPOT.md`.

## Goals

| Priority | Deliverable |
|----------|-------------|
| **P0** | VIX / term-structure IV surface proxy with honest labels |
| **P0** | Multi-window OPT_TA matrix (2022 bear + 2023 + 2024 + study + stress) |
| **P1** | Premium-seller management: 50% credit TP, 2× SL, max 1 roll |
| **P1** | Single-name sleeve AAPL/NVDA with TA volume gates |
| **P2** | Bid haircut on sells; assignment proxy; book delta report |

## Data quality labels (non-negotiable)

| Label | Meaning |
|-------|---------|
| `proxy_bs` | Black–Scholes pricing math on model IV (not exchange fills) |
| `vix_surface` | IV from VIX (± VIX3M / VXST) + mild tenor + put skew |
| `proxy_hv` | Fallback IV = HV20 × `premium_mult` when VIX missing |
| `vix_surface_partial` | Mixed legs/days across surface and HV |
| `proxy_bs\|vix_surface` | Combined run `data_label` in results |
| `assignment_proxy` | Simplified equity assignment (not OCC) |
| `approx_bs_delta_book` | Sum of share-equivalent BS deltas |

## Architecture

```
paper_live/options/
  vol_surface.py     # VIX term structure + skew + labels
  management.py      # TP/SL, max rolls, haircut, assignment
  vol_proxy.py       # HV + legacy HV×mult (fallback only)
  replay_options.py  # wires surface + mgmt + delta
  bs.py              # BS price + bs_delta

paper_live/cloud/
  zoo_options_ta.json         # index TA zoo (unchanged IDs)
  zoo_options_ta_names.json   # AAPL/NVDA + index controls

scripts/
  run_options_ta_matrix.py    # multi-window + stress pack
  run_paper_options_batch.py  # pulls VIX/VIX3M; book delta in SUMMARY
```

### IV surface model (research proxy)

1. **Anchors:** ~9d (VXST or VIX×1.05), ~30d (VIX/100), ~90d (VIX3M/100 or mild contango of VIX).
2. **Interpolation:** piecewise linear in calendar days.
3. **Skew:** OTM puts richer (`put_slope` × |log(K/S)|); mild OTM call wing.
4. **Fallback:** if no VIX bar causal-available → `proxy_hv`.

Still **not** OPRA / live chain IV. Does not claim true VRP measurement.

### Bid haircut

On **sell** entry: credit = mid × (1 − h).  
On **buy** entry (long wings / protective): debit = mid × (1 + h).  
Default h = 0.05 (meta `bid_haircut`). Documented as bid/ask stub — not NBBO.

### Premium-seller management

For short-premium kinds (CSP, PCS, CCS, IC, CC short leg):

- **TP:** close when `(initial_credit − mark_to_close) / initial_credit ≥ take_profit_credit_frac` (default 0.50).
- **SL:** close when `mark_to_close ≥ initial_credit × (1 + stop_loss_credit_mult)` (default mult 2.0).
- **Max rolls:** DTE roll only if `rolls_this_structure < max_rolls` (default 1). After that, hold to expiry / TP / SL.
- **Counters:** `n_opens` = every successful entry; `n_dte_rolls` = DTE rolls only; `n_rolls` is a legacy alias of `n_opens` (not “rolls per structure”).
- **mark_to_close** is **signed** (no floor at 0) so long-wing-dominated marks do not force false 100% TP.

### Assignment proxy

- Expiry short put ITM → long shares @ K (cash − K×100×n).
- Expiry short call ITM with stock → deliver shares @ K.
- Optional deep-ITM early assign stub (`deep_itm_assign_pct`).
- **Multi-leg:** after any short assignment, remaining shorts close at mid and **long wings are cash-settled at mid/intrinsic** (never wiped). Structure fully flattened — no half-books. Label remains `assignment_proxy` (not OCC/index fidelity).

### Stress VIX

`inject_crash_into_panels` multiplies VIX/VIX3M/VXST by `vix_spike_mult` (default 2.5) with `vix_floor` during/after the crash window. Equity path shock and vol-surface spike both apply.

### Book delta

After batch: `book_delta_report(results)` sums `approx_delta_end` / avg (stock + 100×n×BS delta per open leg). Not beta-weighted.

## Multi-window matrix

```powershell
python scripts/run_options_ta_matrix.py `
  --zoo paper_live/cloud/zoo_options_ta.json `
  --names-zoo paper_live/cloud/zoo_options_ta_names.json `
  --out reports/paper_options_ta_matrix
```

Windows:

| Name | Requested |
|------|-----------|
| `2022_bear` | 2022-01-03 → 2022-12-30 |
| `2023` | 2023-01-03 → 2023-12-29 |
| `2024` | 2024-01-02 → 2024-12-31 |
| `2025_study` | 2025-10-29 → last available |
| `stress_primary` | primary + synthetic −30% |

Clamped windows set `clamped=true` and document actual vs requested dates.

## Verification

```powershell
python -m pytest tests/test_paper_options_*.py -q --tb=short
```

## Non-goals

- Claiming live alpha from surface proxy.
- Full American early exercise tree / dividend modeling.
- Paid chain history.

## References

- `reports/OPTIONS_FOCUS_BLINDSPOT.md`
- `docs/design/2026-07-22_paper_options_strategies.md`
