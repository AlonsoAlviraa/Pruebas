# Design: Paper strategies with equity options (free / research)

**Status:** plan + scaffold v0  
**Date:** 2026-07-22  
**Capital:** VIRTUAL only — no live brokerage orders.

## Motivation

AUD-A/B showed long-only stock rules lag SPY while **QQQ/SPY hold** captures market. Options literature points to a second edge family: **volatility risk premium (VRP)** — implied vol tends to exceed subsequent realized vol, so *systematic premium selling* (with crash risk) has a documented risk premium, not a free lunch.

## Evidence from papers / industry (summary)

| Theme | Core idea | Sources (indicative) |
|-------|-----------|----------------------|
| **VRP / put-write** | Sell puts (or PUT index style); IV > RV compensation | Quantpedia VRP; CBOE PUT; arXiv put-writing + Kelly/VIX sizing (2025) |
| **Covered call** | Long stock + short OTM call; income, capped upside | CBOE BXM/BXY; QRG covered-call + VRP optimize |
| **Cash-secured put (CSP)** | Short OTM put fully collateralized in cash | Same VRP family; retail “wheel” |
| **Crash-aware short vol** | Sell ATM straddle / short put + buy far OTM put wing | Quantpedia (5% premium + 15% OTM hedge) |
| **OTM depth** | 5–10% OTM puts often better risk-adjusted than ATM in studies | arXiv hybrid Kelly–VIX put write |
| **Sizing** | Kelly / VIX-rank / hybrid sizing matters as much as strike | arXiv 2508.16598 |
| **PEAD / event straddles** | Long straddle into earnings IV crush is **opposite** trade; often *sell* post-event | Earnings IV crush literature (separate module) |

**Honest risks (non-negotiable in design):**

- Short vol has **left-tail** disasters (1987, 2008, 2020, 2022 vol spikes).
- Margin / collateral / gap risk must be modeled; naive “sell puts daily +1%” is fiction.
- Free **chain** history is scarce; v0 uses **Black–Scholes proxy** on OHLCV + HV/IV proxy, not NBBO options tapes.

## Goals

1. Add **options-aware paper strategies** to the free cloud study stack.
2. Benchmark every options book vs **SPY B&H**, **QQQ B&H**, and **equity baseline zoo**.
3. Gate: refuse silent “synthetic options” as “real fills” unless labeled `proxy_bs`.
4. Kill / DD rules stricter than equity (short vol).

## Non-goals (v0)

- Live multi-leg routing, assignment workflow, OCC reporting.
- Paid OPRA / ORATS / Polygon options history.
- Claiming live alpha from proxy backtests.

## Architecture (v0 → v1)

```
paper_live/options/
  bs.py              # Black–Scholes + greeks (pure)
  vol_proxy.py       # HV20/60, IV proxy = HV * mult or Parkinson
  strategies.py      # covered_call, cash_secured_put, collar, put_credit_spread
  replay_options.py  # daily mark-to-model of open legs
  zoo_options.json    # strategy definitions
scripts/run_paper_options_batch.py
reports/paper_options/
```

### Data ladder (free first)

| Tier | Source | Use |
|------|--------|-----|
| **v0 proxy** | Yahoo/seed OHLCV | HV; BS mark; no real quotes |
| **v1 chain snapshot** | Yahoo options chain (spotty) | Validate strikes near today only |
| **v2** | Optional paid history later | Research only if user adds key |

### Strategies to implement (priority)

| ID | Strategy | Legs (model) | Edge thesis |
|----|----------|--------------|-------------|
| **OPT01** | Covered call 30Δ~ / ~5% OTM, 30–45 DTE | Long 100 sh + short call | Income + mild VRP; underperforms strong bull |
| **OPT02** | Cash-secured put ~5–10% OTM, 30–45 DTE | Short put + cash collateral | Classic VRP; equity-like downside |
| **OPT03** | Put credit spread (bull put) | Short put + long lower put | Defined risk VRP |
| **OPT04** | Collar on stock | Long stock + long put + short call | Defensive equity |
| **OPT05** | Short strangle w/ wing | Short OTM put/call + far put | Higher income, tail hedge |
| **OPT06** | VRP gate | Only sell premium if HV60 &lt; IV_proxy * 0.85 | Avoid selling cheap vol |
| **OPT07** | Wheel | CSP → if assigned covered call | Process strategy |
| **OPT08** | No-trade control | 100% T-bills/cash | Floor |

### Risk (options paper)

- `max_portfolio_dd` lower (e.g. 12–15%) for short premium.
- Position size by **margin-at-risk** (put notional * OTMness factor), not full notional.
- Hard kill on 1-day portfolio drop &gt; X% (gap proxy).
- Never unlimited short naked call without stock (covered only in v0).

### Validation protocol

1. Same windows as equity cloud (2022, 2023, 2024, 2025, 2026 YTD).
2. Report: total return, max DD, Calmar, **CVaR**, worst month, vs SPY.
3. Stress: force 2020-like −30% month synthetic on marks.
4. Promote only if **after costs** and **defined-risk** variants beat cash *and* do not blow up in stress (even if lag SPY bull).

## PR plan

| PR | Title | Depends |
|----|-------|---------|
| **OPT-PR1** | BS + vol_proxy + unit tests | — |
| **OPT-PR2** | Covered call + CSP daily replay (proxy) | OPT-PR1 |
| **OPT-PR3** | Credit spread + collar + risk gates | OPT-PR2 |
| **OPT-PR4** | `run_paper_options_batch` + SUMMARY vs SPY + GH Actions optional job | OPT-PR2 |
| **OPT-PR5** | Multi-window + stress pack | OPT-PR4 |
| **OPT-PR6** | Optional real chain fetch (Yahoo) for “today” paper only | OPT-PR4 |

## Success criteria

- OPT-PR1/2 green tests on synthetic paths.
- At least one **defined-risk** OPT0x with documented proxy assumptions in SUMMARY.
- No unlabeled proxy as “real option fills”.
- Equity zoo v2 still runs on Actions (this design does not break equity path).

## Relationship to equity AUD

- Equity **no_extension / pullback** remain stock books.
- Options books are **separate capital sleeves** in paper multi-strategy dashboard later (not mixed fills in one ledger without explicit multi-asset OMS).

## References (starting set)

- Quantpedia: Volatility Risk Premium Effect  
- CBOE: PUT / BXM methodology notes; “Strategy Spotlight: Considerations in volatility trading”  
- arXiv: “Sizing the Risk: Kelly, VIX, and Hybrid Approaches in Put-Writing” (2025)  
- Christensen & Prabhala (1998) IV vs RV (classic VRP root)  
- Practitioner: covered-call VRP optimization (Envestnet QRG)  

---

_Not financial advice. Proxy options ≠ exchange fills._
