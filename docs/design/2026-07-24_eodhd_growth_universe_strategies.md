# Design — EODHD growth universe + distinct strategies

**Date:** 2026-07-24  
**Status:** APPROVED (implementation)  
**Modules:** DAT-05, UNI-01, STR-G1…G5, VAL/PROMO  
**Product:** Research only — does **not** auto-change paper freeze (`turbo_highvol_minalloc`)

---

## 1. Intent

Stop expanding highvol knobs on raw large universes. Build a **fundamentals-first L0** from **EODHD** deep history:

| Gate | Rule (PIT `available_at ≤ t`) |
|------|------------------------------|
| **G-Q** | Quarterly EPS YoY ≥ **+10%** (double-digit) |
| **G-A** | Annual growth: EPS TTM YoY ≥ **+15%** (fallback revenue TTM ≥ +15%) |
| **Rank** | Prefer highest growth among passers → top-N (default 80) |

Then run **structurally different** strategies on that L0 and judge by **financial OOS metrics** (CAGR, Sharpe, Sortino, MDD, excess vs SPY **and** vs L0 EW).

### Motivation

- Full-universe n=1121 without quality L0: minalloc CAGR ~19.5%, MDD −80% (vs highvol80 ~41% / −54%).
- Local `*_fundamentals.csv` only ~2024–25 (0 tickers with ≥8 quarters) → EODHD deep fund required.

---

## 2. Data contract (DAT-05)

**Source (preferred):** `GET https://eodhd.com/api/fundamentals/{SYMBOL}.US`  
**Label:** `source=eodhd`  
**Normalized CSV columns:**

```text
as_of, period, eps, revenue, net_income, available_at, source
```

- `as_of` = fiscal quarter end  
- `available_at` = `as_of + lag_days` (default **45** calendar days)  
- Prefer Earnings.History epsActual + Income_Statement quarterly revenue/netIncome  

**OHLCV:** existing EODHD EOD path (works on current plan).

**Fallback (2026-07-24 measured):** EODHD **Fundamentals returns HTTP 403** on the current token (EOD bars OK). Until Fundamentals add-on is enabled:

- Use `scripts/download_yahoo_fundamentals_deep.py` → same CSV schema, `source=yahoo`
- Growth gates / battery are **provider-agnostic** on the CSV schema
- Re-run `download_eodhd_fundamentals.py` when 403 clears; prefer `source=eodhd` rows

**Honesty:** not CRSP; lag is approximate; no look-ahead via `available_at`.

---

## 3. Universe contract (UNI-01)

```python
GrowthGateConfig(
  min_eps_q_yoy=0.10,
  min_eps_ttm_yoy=0.15,
  min_rev_ttm_yoy=0.15,
  min_price=5.0,
  min_adv=2e6,
  lag_days=45,
  top_n=80,
)
```

**Rank score (cross-sectional among passers):**

```text
0.50 * rank(eps_ttm_yoy) + 0.30 * rank(eps_q_yoy) + 0.20 * rank(rev_ttm_yoy)
```

**Rebuild:** each OOS year-start (1 Jan prior knowledge = as-of prior year-end).

**Out of scope v1:** consensus surprise G-S (optional later).

---

## 4. Strategies (STR-G*)

| ID | Name | Thesis |
|----|------|--------|
| S1 | `growth_ew` | Filter is the edge — EW L0 |
| S2 | `growth_trend_mom` | Growth + SMA50 + mom |
| S3 | `growth_turbo_minalloc` | ML turbo only inside L0 |
| S4 | `growth_cs_mom` | Cross-sectional mom rank top-K (no ML) residual-style |
| S5 | `growth_quality_strict` | Fund bar gate + strict dual golden |

**Primary falsifier:** residual of Si vs S1 (same L0).  
**Benches:** SPY, QQQ, L0 EW (=S1).

---

## 5. Eval + promotion

- Costs: commission 0.10% + slip 0.05%  
- OOS: 2018–2025 if fund depth allows; else first year with ≥200 growth members  
- Metrics: CAGR, Sharpe, Sortino, MDD, excess SPY, residual vs S1, year table, LOYO  
- Promotion funnel: 0 ADVANCE valid; freeze unchanged without human ADVANCE  

---

## 6. Kill criteria

- S3 residual vs S1 ≤ 0 → ML no value on growth L0  
- All MDD worse than −50% → no ADVANCE  
- Single-year concentration (LOYO fails) → HOLD only  
- Fund coverage too thin → shorten OOS, do not invent rows  

---

## 7. Verification

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest tests/test_eodhd_fundamentals_unit.py tests/test_growth_universe_unit.py -q
python scripts/probe_eodhd_fundamentals.py --tickers AAPL,NVDA,MSFT
python scripts/download_eodhd_fundamentals.py --ticker-file good_tickers_filtrados.txt --limit 50
python scripts/build_growth_universe.py --as-of 2023-12-31 --top-n 80
```

---

## 8. Disclaimer

Research software. Not financial advice. Past OOS ≠ future results.
