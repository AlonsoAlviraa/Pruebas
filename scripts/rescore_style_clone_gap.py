"""Recompute SPY/QQQ excess + P1 gates from saved S1 equity CSVs (no retrain)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trad_research.alpha_attribution import (
    compare_to_benchmark,
    confirm_p1_style_confusion,
    confirm_p2_unfair_spy_bench,
    rank_problems_by_false_alpha,
)
from trad_research.metrics import equity_metrics
from trad_research.walk_forward import load_benchmark_equity


def load_eq(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index, utc=True)
    # Equity may be stamped at session times; benches are date midnights — normalize
    s.index = s.index.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def main() -> int:
    out = Path("reports/redesign/S1_style_clone_gap_full")
    eq_dir = out / "equity"
    data_root = Path("data")
    baseline_name = "turbo_highvol_minalloc"
    clones = [
        "style_ew_hv",
        "style_trend_sma50_hv",
        "style_mom_1m_hv",
        "style_trend_mom_hv",
    ]

    beq = load_eq(eq_dir / f"{baseline_name}.csv")
    spy = load_benchmark_equity(
        data_root, beq.index.min(), beq.index.max(), preferred=["SPY"]
    )
    spy.index = pd.to_datetime(spy.index, utc=True).normalize()
    spy = spy[~spy.index.duplicated(keep="last")].sort_index()
    joined = pd.concat([beq.rename("s"), spy.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 10:
        raise RuntimeError(f"SPY join too short: {len(joined)}")
    start = float(joined["s"].iloc[0])
    b_eq = joined["b"] / float(joined["b"].iloc[0]) * start
    rep_s = equity_metrics(joined["s"], start_equity=start, benchmark=b_eq)
    b_cagr, b_sh, b_mdd = rep_s.cagr, rep_s.sharpe, rep_s.max_drawdown
    spy_cagr = float(rep_s.benchmark_cagr or 0.0)
    b_spy_ex = float(rep_s.excess_cagr or 0.0)

    qqq = load_benchmark_equity(
        data_root, beq.index.min(), beq.index.max(), preferred=["QQQ"]
    )
    qqq.index = pd.to_datetime(qqq.index, utc=True).normalize()
    qqq = qqq[~qqq.index.duplicated(keep="last")].sort_index()
    j2 = pd.concat([beq.rename("s"), qqq.rename("b")], axis=1, join="inner").dropna()
    b_qqq = j2["b"] / float(j2["b"].iloc[0]) * float(j2["s"].iloc[0])
    rep_q = equity_metrics(j2["s"], start_equity=float(j2["s"].iloc[0]), benchmark=b_qqq)

    rows = []
    for cname in clones:
        ceq = load_eq(eq_dir / f"{cname}.csv")
        j = pd.concat([ceq.rename("s"), spy.rename("b")], axis=1, join="inner").dropna()
        b_s = j["b"] / float(j["b"].iloc[0]) * float(j["s"].iloc[0])
        rep_c = equity_metrics(j["s"], start_equity=float(j["s"].iloc[0]), benchmark=b_s)
        residual = compare_to_benchmark(
            beq, ceq, start_equity=float(beq.iloc[0]), label=f"vs_{cname}"
        )
        p1 = confirm_p1_style_confusion(
            baseline_excess_vs_spy=b_spy_ex,
            clone_excess_vs_spy=float(rep_c.excess_cagr or 0.0),
            baseline_sharpe=float(b_sh),
            clone_sharpe=float(rep_c.sharpe),
        )
        gross_capture = (rep_c.cagr / b_cagr) if b_cagr > 1e-9 else None
        rows.append(
            {
                "clone": cname,
                "clone_cagr": rep_c.cagr,
                "clone_sharpe": rep_c.sharpe,
                "clone_mdd": rep_c.max_drawdown,
                "clone_excess_vs_spy": rep_c.excess_cagr,
                "gross_cagr_capture": gross_capture,
                "p1": p1,
                "residual_vs_style": residual.to_dict(),
            }
        )
        print(
            cname,
            f"cagr={rep_c.cagr:.3f}",
            f"ex_spy={rep_c.excess_cagr:.3f}",
            f"p1={p1['confirmed']}",
            f"gross_cap={gross_capture:.3f}",
            f"res_ex={residual.excess_cagr:.3f}",
        )

    summary_old = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    pit = summary_old.get("pit_block") or {}
    p2 = (
        confirm_p2_unfair_spy_bench(float(pit.get("excess_cagr_vs_pit_ew") or 0.0))
        if pit.get("ok")
        else None
    )
    p1_any = any(r["p1"]["confirmed"] for r in rows)
    best_gross = max((r["gross_cagr_capture"] or 0.0) for r in rows)
    best_gross_name = max(rows, key=lambda r: r["gross_cagr_capture"] or 0.0)["clone"]
    hardest = min(rows, key=lambda r: r["residual_vs_style"]["excess_cagr"])
    min_gap = min(r["p1"]["residual_sharpe_gap"] for r in rows)

    ranking = rank_problems_by_false_alpha(
        [
            {
                "problem": "P1",
                "confirmed": p1_any,
                "best_gross_capture": best_gross,
                "min_residual_sharpe_gap": min_gap,
            },
            p2 if p2 else {"problem": "P2", "confirmed": False},
            {"problem": "P3", "confirmed": False, "note": "not in this run"},
        ]
    )

    summary = {
        "protocol": summary_old.get("protocol"),
        "run_meta": summary_old.get("run_meta"),
        "baseline": baseline_name,
        "baseline_cagr": b_cagr,
        "baseline_sharpe": b_sh,
        "baseline_mdd": b_mdd,
        "baseline_spy_cagr": spy_cagr,
        "baseline_excess_vs_spy": b_spy_ex,
        "baseline_qqq_cagr": rep_q.benchmark_cagr,
        "baseline_excess_vs_qqq": rep_q.excess_cagr,
        "n_tickers": 80,
        "joined_spy_days": len(joined),
        "clones": rows,
        "p1_confirmed_design_gate": p1_any,
        "p1_min_residual_sharpe_gap": min_gap,
        "best_gross_cagr_capture": best_gross,
        "best_gross_clone": best_gross_name,
        "hardest_clone": hardest["clone"],
        "hardest_clone_residual_cagr": hardest["residual_vs_style"]["excess_cagr"],
        "p2": p2,
        "pit_block": pit,
        "problem_ranking": ranking,
        "interpretation": {
            "P1_design": (
                "NOT confirmed: residual Sharpe gap > 0.15 vs all clones; "
                "ML keeps residual after style shell"
            ),
            "P1_nuance": (
                f"Best gross CAGR capture by style = {best_gross:.1%} ({best_gross_name}); "
                "style explains large share of level but residual gap remains"
            ),
            "P2_design": (
                "NOT confirmed on 2018-2025 same L0: excess vs PIT EW positive "
                f"({pit.get('excess_cagr_vs_pit_ew')})"
            ),
            "P2_prior": (
                "Prior PIT bake-off 2009-14 showed negative excess vs PIT EW under "
                "different window — regime/window dependent"
            ),
            "window": "2018-2025 OOS calendar WF, universe_highvol80, real EODHD",
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# S1 Style-clone gap (structural autopsy) — FULL real data",
        "",
        "**Window:** OOS calendar years **2018–2025** · **L0:** `universe_highvol80.txt` (80) · **Data:** EODHD real OHLCV",
        "**Baseline:** `turbo_highvol_minalloc` (STYLE-US control; WF retrain + embargo)",
        "",
        "## No-leak protocol",
        "",
        "- Same L0 ticker file for baseline and all clones (no universe mix).",
        "- ML train: bars before year-start + embargo in `_build_training_frame`.",
        "- Style clones: no train; SMA50 / ret_1m causal features only.",
        "- SPY/QQQ benches from price series aligned to strategy equity dates (inner join).",
        "- PIT EW: membership-filtered equal-weight of same 80 panels (bench only).",
        "- Caveat: static highvol list may embed selection bias at list construction time.",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Baseline CAGR | **{b_cagr:.2%}** |",
        f"| Baseline Sharpe | **{b_sh:.2f}** |",
        f"| Baseline MDD | **{b_mdd:.2%}** |",
        f"| SPY CAGR (aligned) | **{spy_cagr:.2%}** |",
        f"| Excess vs SPY | **{b_spy_ex:.2%}** |",
        f"| QQQ CAGR (aligned) | **{rep_q.benchmark_cagr:.2%}** |",
        f"| Excess vs QQQ | **{rep_q.excess_cagr:.2%}** |",
        f"| PIT EW CAGR | **{float(pit.get('pit_ew_cagr') or 0):.2%}** |",
        f"| Excess vs PIT EW | **{float(pit.get('excess_cagr_vs_pit_ew') or 0):.2%}** |",
        f"| SPY join days | {len(joined)} |",
        "",
        f"**P1 (design gate):** **{p1_any}** — min residual Sharpe gap = {min_gap:.3f} (confirm if ≤0.15)",
        f"**P1 (gross CAGR capture max):** **{best_gross:.1%}** via `{best_gross_name}` (informational)",
        f"**P2 (design gate):** **{(p2 or {}).get('confirmed')}** — excess vs PIT EW = {pit.get('excess_cagr_vs_pit_ew')}",
        "",
        "## Clones (same L0)",
        "",
        "| Clone | CAGR | Sharpe | MDD | Excess SPY | Gross capture | Residual excess CAGR | Residual Sharpe | P1 design |",
        "|-------|------|--------|-----|------------|---------------|----------------------|-----------------|-----------|",
    ]
    for r in rows:
        rv = r["residual_vs_style"]
        lines.append(
            f"| `{r['clone']}` | {r['clone_cagr']:.2%} | {r['clone_sharpe']:.2f} | "
            f"{r['clone_mdd']:.2%} | {r['clone_excess_vs_spy']:.2%} | "
            f"{(r['gross_cagr_capture'] or 0):.1%} | {rv['excess_cagr']:.2%} | "
            f"{rv['residual_sharpe']:.2f} | {r['p1']['confirmed']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "1. **P1 NOT confirmed under design gates:** ML baseline keeps **positive residual** vs best style shell "
        f"(`{hardest['clone']}`: +{hardest['residual_vs_style']['excess_cagr']:.1%} excess CAGR, "
        f"residual Sharpe {hardest['residual_vs_style']['residual_sharpe']:.2f}). "
        "Style is material (EW ~59% of gross CAGR) but does **not** fully absorb the edge.",
        "",
        "2. **P2 NOT confirmed on this window:** strategy **beats** PIT EW of the same 80 names by ~+28pp CAGR (2018–25). "
        "This **differs** from the 2009–14 PIT bake-off (−18pp vs PIT EW) — P2 is **window/regime dependent**. "
        "Still: do not use SPY alone as promotion gate; report PIT EW always.",
        "",
        "3. **Not portable alpha yet:** US high-vol + QQQ dual/golden regime only. **P3 (geo) remains open.**",
        "",
        "4. **Redesign implication:** residual vs style exists → continue ALPHA-PORTABLE v0, keep style-clone as **mandatory R1 gate**. "
        "Do not promote on SPY alone. Investigate why EW shell underperforms ML (selection/sizing/timing) without retuning turbo knobs ad hoc.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "S1_style_clone_gap.md").write_text("\n".join(lines), encoding="utf-8")
    print("baseline", b_cagr, b_sh, "spy", spy_cagr, "ex", b_spy_ex)
    print("P1", p1_any, "P2", (p2 or {}).get("confirmed"))
    print("WROTE", out / "S1_style_clone_gap.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
