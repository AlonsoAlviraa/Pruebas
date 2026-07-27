#!/usr/bin/env python3
"""Build large amplify options zoo (debit/PMCC/spreads) across indices + names."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_live" / "cloud" / "zoo_options_amplify.json"

INDICES = ["SPY", "QQQ", "IWM"]
NAMES = ["AAPL", "NVDA", "MSFT", "META", "AMZN", "GOOGL", "TSLA", "AMD"]


def main() -> None:
    strategies = []
    # cash control
    strategies.append(
        {
            "id": "AMP_CASH",
            "label": "Cash control",
            "kind": "cash",
            "underlying": "SPY",
            "hard_kill_enabled": False,
        }
    )

    def add(kind, und, **kw):
        meta = kw.pop("meta", {})
        sid = kw.pop("id", None) or f"AMP_{kind}_{und}_{len(strategies)}"
        strategies.append(
            {
                "id": sid,
                "label": kw.pop("label", f"{kind} {und}"),
                "kind": kind,
                "underlying": und,
                "dte_days": kw.pop("dte_days", 30),
                "otm_pct": kw.pop("otm_pct", 0.05),
                "wing_otm_pct": kw.pop("wing_otm_pct", 0.12),
                "contracts": kw.pop("contracts", 5),
                "max_portfolio_dd": kw.pop("max_portfolio_dd", 0.35),
                "max_margin_fraction": kw.pop("max_margin_fraction", 0.25),
                "max_single_day_drop": kw.pop("max_single_day_drop", 0.15),
                "meta": meta,
                "notes": kw.pop("notes", "Amplify research; proxy_bs|vix_surface"),
            }
        )

    # --- Index amplify ---
    for und in INDICES:
        add(
            "long_call",
            und,
            id=f"AMP_LC_{und}_5otm_30d",
            label=f"{und} long call 5% OTM 30D uptrend",
            meta={
                "require_uptrend": True,
                "max_premium_budget_frac": 0.12,
                "require_vrp_proxy_below": True,
                "max_vrp_proxy": 0.12,
            },
        )
        add(
            "long_call",
            und,
            id=f"AMP_LC_{und}_0otm_21d",
            label=f"{und} long ATM-ish call 21D",
            dte_days=21,
            otm_pct=0.01,
            meta={"require_sma200": True, "max_premium_budget_frac": 0.10},
        )
        add(
            "call_debit_spread",
            und,
            id=f"AMP_CDS_{und}_bull",
            label=f"{und} bull call debit spread uptrend",
            otm_pct=0.02,
            wing_otm_pct=0.10,
            meta={"require_uptrend": True, "max_premium_budget_frac": 0.12},
        )
        add(
            "long_put",
            und,
            id=f"AMP_LP_{und}_tail",
            label=f"{und} long put tail 8% OTM",
            otm_pct=0.08,
            dte_days=45,
            meta={
                "require_rsi_overbought": True,
                "min_rsi": 68,
                "max_premium_budget_frac": 0.05,
            },
        )
        add(
            "put_debit_spread",
            und,
            id=f"AMP_PDS_{und}_bear",
            label=f"{und} bear put debit spread",
            otm_pct=0.03,
            wing_otm_pct=0.12,
            meta={"require_rsi_overbought": True, "min_rsi": 65, "max_premium_budget_frac": 0.08},
        )
        add(
            "pmcc",
            und,
            id=f"AMP_PMCC_{und}",
            label=f"{und} PMCC LEAP+short call",
            dte_days=30,
            otm_pct=0.05,
            meta={
                "require_uptrend": True,
                "leap_dte_days": 180,
                "leap_otm_pct": 0.05,
                "max_premium_budget_frac": 0.20,
            },
        )
        add(
            "long_call",
            und,
            id=f"AMP_LEAP_{und}_180d",
            label=f"{und} LEAP-like long call 180D 10% OTM",
            dte_days=180,
            otm_pct=0.10,
            meta={"require_sma200": True, "max_premium_budget_frac": 0.15},
        )

    # --- Single-name amplify ---
    for und in NAMES:
        add(
            "long_call",
            und,
            id=f"AMP_LC_{und}_mom",
            label=f"{und} long call uptrend+vol",
            meta={
                "require_uptrend": True,
                "require_volume_confirm": True,
                "min_volume_ratio": 1.1,
                "max_premium_budget_frac": 0.08,
            },
            max_portfolio_dd=0.40,
        )
        add(
            "call_debit_spread",
            und,
            id=f"AMP_CDS_{und}",
            label=f"{und} bull call debit spread",
            meta={"require_uptrend": True, "max_premium_budget_frac": 0.10},
            max_portfolio_dd=0.35,
        )
        add(
            "pmcc",
            und,
            id=f"AMP_PMCC_{und}",
            label=f"{und} PMCC",
            meta={
                "require_uptrend": True,
                "leap_dte_days": 150,
                "max_premium_budget_frac": 0.18,
            },
            max_portfolio_dd=0.40,
        )
        add(
            "long_put",
            und,
            id=f"AMP_LP_{und}_hedge",
            label=f"{und} long put hedge RSI OB",
            otm_pct=0.07,
            meta={
                "require_rsi_overbought": True,
                "min_rsi": 70,
                "max_premium_budget_frac": 0.04,
            },
        )

    # Always-on long call QQQ (no gate) — aggressive amplify control
    add(
        "long_call",
        "QQQ",
        id="AMP_LC_QQQ_always",
        label="QQQ long call always (no TA gate)",
        meta={"max_premium_budget_frac": 0.15},
    )
    add(
        "call_debit_spread",
        "QQQ",
        id="AMP_CDS_QQQ_always",
        label="QQQ CDS always",
        meta={"max_premium_budget_frac": 0.15},
    )

    # Income controls (not amplify) for comparison
    for und in ("SPY", "QQQ"):
        strategies.append(
            {
                "id": f"AMP_CTRL_CC_{und}",
                "label": f"{und} covered call control (income)",
                "kind": "covered_call",
                "underlying": und,
                "dte_days": 30,
                "otm_pct": 0.05,
                "max_margin_fraction": 0.95,
                "meta": {"require_uptrend": True},
                "notes": "Income control — not amplify",
            }
        )
        strategies.append(
            {
                "id": f"AMP_CTRL_PCS_{und}",
                "label": f"{und} PCS control (income)",
                "kind": "put_credit_spread",
                "underlying": und,
                "dte_days": 30,
                "otm_pct": 0.05,
                "wing_otm_pct": 0.12,
                "max_margin_fraction": 0.40,
                "meta": {"require_range_regime": True, "max_atr_pctile": 0.45},
                "notes": "Income control — not amplify",
            }
        )

    zoo = {
        "version": "options-amplify-zoo-v1",
        "capital0": 100000.0,
        "data_label": "proxy_bs",
        "notes": (
            "Gain-amplify options research: long calls/puts, debit spreads, PMCC. "
            "Marks: proxy_bs|vix_surface. Chain history NOT used. VIRTUAL only."
        ),
        "risk": {
            "max_portfolio_dd": 0.40,
            "max_single_day_drop": 0.20,
            "max_margin_fraction": 0.25,
            "hard_kill_enabled": True,
            "max_contracts": 15,
            "notes": "Debit budget sized via max_premium_budget_frac meta",
        },
        "strategies": strategies,
        "n_strategies": len(strategies),
    }
    OUT.write_text(json.dumps(zoo, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} n={len(strategies)}")


if __name__ == "__main__":
    main()
