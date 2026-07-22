#!/usr/bin/env python3
"""Build ~50 options strategy zoo from academic/CBOE/Twitter families (proxy_bs).

Sources (design notes, not endorsement):
- CBOE BXM/BXY buy-write, PUT put-write, CNDR iron condor (~15d short / ~5d wing)
- Quantpedia / literature VRP put-write OTM 5–10%
- Twitter/X: iron condor positioning, call-side skew for equity drift, one-roll discipline
- GitHub: vectorbt-style parametric grids (systematic parameter space)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_live" / "cloud" / "zoo_options_50.json"


def main() -> None:
    strats = []
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        sid = kw.pop("id", f"M{n:02d}")
        strats.append({"id": sid, **kw})

    # --- Cash control ---
    add(
        id="M00_cash",
        label="Cash control",
        kind="cash",
        underlying="SPY",
        notes="Floor benchmark",
        family="control",
        source="baseline",
    )

    # --- CBOE-style BuyWrite (BXM ~ ATM, BXY ~2% OTM) on SPY/QQQ ---
    for und in ("SPY", "QQQ"):
        for otm, dte, tag in [
            (0.00, 30, "bxm_atm"),
            (0.02, 30, "bxy_2otm"),
            (0.05, 30, "cc_5otm"),
            (0.07, 30, "cc_7otm"),
            (0.05, 45, "cc_5otm_45d"),
            (0.10, 45, "cc_10otm_45d"),
        ]:
            add(
                id=f"M_cc_{und}_{tag}",
                label=f"Covered call {und} otm={otm:.0%} dte={dte}",
                kind="covered_call",
                underlying=und,
                dte_days=dte,
                otm_pct=otm,
                max_portfolio_dd=0.18,
                max_margin_fraction=0.95,
                family="buywrite",
                source="CBOE BXM/BXY family",
            )

    # --- PUT-write / CSP grid (VRP literature) ---
    for und in ("SPY", "QQQ"):
        for otm, dte, tag in [
            (0.03, 30, "3otm_30"),
            (0.05, 30, "5otm_30"),
            (0.07, 30, "7otm_30"),
            (0.10, 30, "10otm_30"),
            (0.05, 45, "5otm_45"),
            (0.10, 45, "10otm_45"),
            (0.15, 45, "15otm_45"),
        ]:
            add(
                id=f"M_csp_{und}_{tag}",
                label=f"CSP {und} otm={otm:.0%} dte={dte}",
                kind="cash_secured_put",
                underlying=und,
                dte_days=dte,
                otm_pct=otm,
                max_portfolio_dd=0.15,
                max_margin_fraction=0.80,
                family="put_write",
                source="CBOE PUT / VRP OTM put-write",
            )

    # --- VRP gate CSPs ---
    for und in ("SPY", "QQQ"):
        add(
            id=f"M_csp_vrp_{und}",
            label=f"CSP VRP-gate {und} 5% 30d",
            kind="cash_secured_put",
            underlying=und,
            dte_days=30,
            otm_pct=0.05,
            premium_mult=1.20,
            meta={"require_hv_above_median": True},
            max_portfolio_dd=0.15,
            max_margin_fraction=0.80,
            family="put_write_vrp_gate",
            source="VRP only sell rich HV regime",
        )

    # --- Put credit spreads (defined risk bull put) ---
    for und in ("SPY", "QQQ"):
        for otm, wing, dte, tag in [
            (0.03, 0.08, 30, "3_8_30"),
            (0.05, 0.10, 30, "5_10_30"),
            (0.05, 0.15, 30, "5_15_30"),
            (0.07, 0.12, 45, "7_12_45"),
        ]:
            add(
                id=f"M_pcs_{und}_{tag}",
                label=f"PCS {und} short={otm:.0%} wing={wing:.0%} dte={dte}",
                kind="put_credit_spread",
                underlying=und,
                dte_days=dte,
                otm_pct=otm,
                wing_otm_pct=wing,
                max_portfolio_dd=0.12,
                max_single_day_drop=0.06,
                max_margin_fraction=0.50,
                family="vertical_put",
                source="defined-risk VRP / Twitter PCS preference vs IC",
            )

    # --- Call credit spreads (bear call) ---
    for und in ("SPY",):
        for otm, wing, tag in [
            (0.03, 0.08, "3_8"),
            (0.05, 0.10, "5_10"),
            (0.07, 0.12, "7_12"),
        ]:
            add(
                id=f"M_ccs_{und}_{tag}",
                label=f"CCS {und} short={otm:.0%} wing={wing:.0%}",
                kind="call_credit_spread",
                underlying=und,
                dte_days=30,
                otm_pct=otm,
                wing_otm_pct=wing,
                max_portfolio_dd=0.12,
                max_margin_fraction=0.50,
                family="vertical_call",
                source="defined-risk short call vertical",
            )

    # --- Iron condors (CNDR-like + Twitter skew call-side wider) ---
    for und in ("SPY", "QQQ"):
        for otm, wing, dte, tag, note in [
            (0.05, 0.10, 30, "sym_5_10", "symmetric IC"),
            (0.03, 0.08, 30, "sym_3_8", "tighter IC"),
            (0.07, 0.12, 30, "sym_7_12", "wider IC"),
            (0.05, 0.15, 45, "sym_5_15_45", "monthly longer"),
            (0.05, 0.12, 30, "cndrish", "CBOE CNDR-ish ~short 15d / wing ~5d proxy via %"),
        ]:
            add(
                id=f"M_ic_{und}_{tag}",
                label=f"Iron condor {und} {tag}",
                kind="iron_condor",
                underlying=und,
                dte_days=dte,
                otm_pct=otm,
                wing_otm_pct=wing,
                max_portfolio_dd=0.12,
                max_margin_fraction=0.45,
                family="iron_condor",
                source=f"CBOE CNDR / X IC discussion — {note}",
            )

    # --- Collars & protective puts ---
    for und in ("SPY", "QQQ"):
        add(
            id=f"M_collar_{und}",
            label=f"Collar {und} call5%/put8%",
            kind="collar",
            underlying=und,
            dte_days=30,
            otm_pct=0.05,
            wing_otm_pct=0.08,
            max_portfolio_dd=0.16,
            max_margin_fraction=0.95,
            family="collar",
            source="defensive equity sleeve",
        )
        add(
            id=f"M_pp_{und}",
            label=f"Protective put {und} 5% 30d",
            kind="protective_put",
            underlying=und,
            dte_days=30,
            otm_pct=0.05,
            max_portfolio_dd=0.20,
            max_margin_fraction=0.95,
            family="protective_put",
            source="long equity insurance",
        )

    # --- Premium mult stress (richer IV assumption) ---
    add(
        id="M_csp_spy_rich_iv",
        label="CSP SPY 5% rich IV mult=1.25",
        kind="cash_secured_put",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        premium_mult=1.25,
        max_portfolio_dd=0.15,
        max_margin_fraction=0.80,
        family="sensitivity",
        source="VRP mult sensitivity",
    )
    add(
        id="M_ic_spy_rich_iv",
        label="IC SPY 5/10 rich IV mult=1.25",
        kind="iron_condor",
        underlying="SPY",
        dte_days=30,
        otm_pct=0.05,
        wing_otm_pct=0.10,
        premium_mult=1.25,
        max_portfolio_dd=0.12,
        max_margin_fraction=0.45,
        family="sensitivity",
        source="IC VRP mult sensitivity",
    )

    # Target ~50–56 named families (no silent truncate of tail families)

    zoo = {
        "version": "paper-options-zoo-mega-50-v1",
        "capital0": 100000.0,
        "notes": (
            "Mega paper options study (~50 strategies). proxy_bs marks only. "
            "Families: CBOE buywrite/putwrite/CNDR, VRP grids, IC from X/literature, "
            "defined-risk verticals, collars. Not financial advice."
        ),
        "data_label": "proxy_bs",
        "risk": {
            "max_portfolio_dd": 0.15,
            "max_single_day_drop": 0.08,
            "max_margin_fraction": 0.75,
            "hard_kill_enabled": True,
            "cvar_alpha": 0.05,
        },
        "research_refs": [
            "CBOE BXM/BXY BuyWrite, PUT PutWrite, CNDR Iron Condor indexes",
            "Quantpedia Volatility Risk Premium / put-write",
            "X/Twitter: iron condor strike placement, equity-drift skew, CNDR regime notes",
            "GitHub parametric grids (vectorbt-options / systematic option backtest style)",
        ],
        "strategies": strats,
        "n_strategies": len(strats),
    }
    OUT.write_text(json.dumps(zoo, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with n={len(strats)}")
    fam: dict[str, int] = {}
    for s in strats:
        f = s.get("family") or "?"
        fam[f] = fam.get(f, 0) + 1
    print("families:", fam)


if __name__ == "__main__":
    main()
