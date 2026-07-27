# Design — Metrics expansion, Monte Carlo robustness, promotion funnel

**Date:** 2026-07-23  
**Status:** APPROVED for implementation  
**Modules:** MET-01, MET-02 (VAL-MC-01), PROMO-01..03  

## Pre-registered promotion thresholds (frozen)

### Stage 0 — Eligibility
| Gate | Threshold |
|------|-----------|
| min_trades (full) | 50 |
| min_trades (smoke) | 20 |
| pathology | \|CAGR\| > 100% → kill |
| finite equity | required |

### Stage 1 — Edge quality
| Gate | Threshold |
|------|-----------|
| sortino_min | 0.50 |
| sharpe_min | 0.40 |
| residual_excess_cagr | > 0 (if style equity provided) |
| residual_sharpe | > 0 (if style equity provided) |
| mdd_min | ≥ −0.50 |
| profit_factor_min | 1.05 (if trades available; else skip) |
| expectancy_min | > 0 (if trades available; else skip) |

### Stage 2 — Monte Carlo (bootstrap primary)
| Gate | Threshold |
|------|-----------|
| n_sims | 2000 (smoke 200) |
| sortino_p5_min | 0.20 |
| mdd_p95_max | −0.60 (worst allowed p95 of max DD) |
| shuffle_sortino_p50_ratio | ≥ 0.50 × historical Sortino |
| min_trades_for_mc_advance | 50 (else MC diagnostic_only → cannot ADVANCE) |

### Stage 3 — Structural (product-dependent)
| Product | Requirement |
|---------|-------------|
| STYLE-US | Stage 0–2; geo optional stress only |
| ALPHA-PORTABLE | Stage 0–2 + residual R1; geo non-collapse preferred |

### Stage 4 — Multi-test
| Gate | Rule |
|------|------|
| max_advance | K=3 |
| DSR | append zoo trial; report deflated_sharpe_approx; if n_trials>20 require approx DSR>0 for ADVANCE |

## Labels
`KILL` | `HOLD` | `ADVANCE_STYLE` | `ADVANCE_ALPHA`

## Honesty
- MC does not replace residual vs style or geo FROZEN.
- Shuffle ends same total PnL; bootstrap varies total PnL.
- Sortino in gates uses MAR=0, annualized √252.

## CLI
`scripts/run_promotion_scorecard.py`
