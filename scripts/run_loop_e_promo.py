"""Loop E: full OOS 2018–2025 stitch for Loop D top-3 + promotion scorecard.

Research only. Not financial advice. Does not change paper freeze.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load mega loop helpers without package install
_spec = importlib.util.spec_from_file_location(
    "vol_fund_mega", ROOT / "scripts" / "run_vol_fund_mega_loop.py"
)
_mega = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["vol_fund_mega"] = _mega
_spec.loader.exec_module(_mega)

from trad_research.promotion import (  # noqa: E402
    CandidateInput,
    apply_top_k,
    evaluate_candidate,
    scorecard_table,
)

# Top-3 from Loop D confirm + style control (same L0 spirit = growth_ew vol-only k100)
TOP3 = [
    "turbo_highvol_minalloc__volonly_k100_baseline",
    "turbo_highvol_minalloc__volonly_k100_vt60_only",
    "turbo_highvol_minalloc__volonly_k100_dd35_vt80_yr",
]
STYLE_ID = "growth_ew__volonly_k100_baseline"


def _cfg_for(config_id: str) -> Any:
    """Build GridConfig matching loop_d naming."""
    if config_id.startswith("growth_ew__"):
        # style EW control, vol-only k100
        return _mega.GridConfig(
            config_id=config_id,
            strategy="growth_ew",
            growth_hard=False,
            growth_top_k=100,
            lever_id="baseline",
            vol_only_top=100,
            label="loop_e_style",
        )
    # parse turbo_highvol_minalloc__volonly_k{N}_{lever}
    # e.g. turbo_highvol_minalloc__volonly_k100_dd35_vt80_yr
    rest = config_id.replace("turbo_highvol_minalloc__volonly_", "")
    # rest = k100_dd35_vt80_yr
    parts = rest.split("_", 1)
    kpart = parts[0]  # k100
    lever = parts[1] if len(parts) > 1 else "baseline"
    top = int(kpart.replace("k", ""))
    return _mega.GridConfig(
        config_id=config_id,
        strategy="turbo_highvol_minalloc",
        growth_hard=False,
        growth_top_k=top,
        lever_id=lever,
        vol_only_top=top,
        label="loop_e",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=ROOT / "universe_highvol200.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--first", type=int, default=2018)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "vol_fund_loop_e",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    configs_dir = out / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    l0_cache = out / "l0_cache"
    years = list(range(int(args.first), int(args.last) + 1))

    panel_file = Path(args.panel)
    static_pool = [
        ln.strip().upper()
        for ln in panel_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    print(f"Loop E full OOS {years[0]}–{years[-1]} panel n={len(static_pool)}", flush=True)

    all_ids = TOP3 + [STYLE_ID]
    rows: List[Dict[str, Any]] = []
    for cid in all_ids:
        cfg = _cfg_for(cid)
        print(f"[full] {cid} …", flush=True)
        r = _mega.run_config_years(
            cfg,
            years=years,
            data_root=Path(args.data_root),
            panel_file=panel_file,
            l0_cache=l0_cache,
            static_pool=static_pool,
            min_train_rows=2500,
            use_dynamic_vol=False,
        )
        safe = cid.replace("/", "_")
        cdir = configs_dir / safe
        cdir.mkdir(parents=True, exist_ok=True)
        if isinstance(r.get("equity"), pd.Series):
            r["equity"].to_csv(cdir / "equity.csv", header=["equity"])
        if isinstance(r.get("trades"), pd.DataFrame) and not r["trades"].empty:
            r["trades"].to_csv(cdir / "trades.csv", index=False)
        meta = {k: v for k, v in r.items() if k not in ("equity", "trades")}
        (cdir / "metrics.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  cagr={meta.get('cagr')} mdd={meta.get('max_drawdown')} "
            f"resid={meta.get('residual_cagr_vs_style')} n={meta.get('n_trades')}",
            flush=True,
        )
        rows.append(meta)

    # Promotion
    style_eq = configs_dir / STYLE_ID / "equity.csv"
    cards = []
    for cid in TOP3:
        eq_path = configs_dir / cid / "equity.csv"
        tr_path = configs_dir / cid / "trades.csv"
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
        eq.index = pd.to_datetime(eq.index, utc=True, errors="coerce")
        eq = eq[~eq.index.duplicated(keep="last")].dropna().sort_index()
        st = None
        if style_eq.is_file():
            st = pd.read_csv(style_eq, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
            st.index = pd.to_datetime(st.index, utc=True, errors="coerce")
            st = st[~st.index.duplicated(keep="last")].dropna().sort_index()
        pnls = None
        n_tr = None
        if tr_path.is_file():
            tdf = pd.read_csv(tr_path)
            if "net_profit" in tdf.columns:
                pnls = tdf["net_profit"].to_numpy(dtype=float)
                n_tr = int(len(pnls))
        print(f"[promo] {cid} n_bars={len(eq)} …", flush=True)
        card = evaluate_candidate(
            CandidateInput(
                name=cid,
                equity=eq,
                style_equity=st,
                trade_pnls=pnls,
                n_trades=n_tr,
                product="STYLE-US",
                smoke=False,
            ),
            n_sims=int(args.n_sims),
            seed=int(args.seed),
            n_trials_zoo=max(20, len(TOP3)),
        )
        print(f"  → {card.label} reasons={card.kill_reasons}", flush=True)
        cards.append(card)

    cards = apply_top_k(cards, k=3)
    table = scorecard_table(cards)
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "window": f"{args.first}-{args.last}",
        "panel": str(panel_file),
        "n_sims": int(args.n_sims),
        "full_oos_rows": rows,
        "promotion": [c.to_dict() for c in cards],
        "disclaimer": "Research only. Not financial advice. No paper freeze auto-change.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Loop E — full OOS + promotion",
        "",
        "> **Research only.** Not financial advice.",
        "",
        f"- Full OOS: **{args.first}–{args.last}** · panel highvol200 · n_sims={args.n_sims}",
        f"- Candidates: Loop D confirm top-3 (minalloc vol-only)",
        f"- Style residual bench: `{STYLE_ID}`",
        "",
        "## Full-window metrics",
        "",
        "| config | CAGR | Sortino | MDD | resid vs style | n_trades |",
        "|--------|------|---------|-----|----------------|----------|",
    ]
    for r in rows:
        if r.get("config_id") == STYLE_ID or r.get("strategy") == "growth_ew":
            continue
        lines.append(
            f"| `{r.get('config_id')}` | {100*float(r.get('cagr') or 0):.1f}% | "
            f"{float(r.get('sortino') or 0):.2f} | {100*float(r.get('max_drawdown') or 0):.1f}% | "
            f"{100*float(r.get('residual_cagr_vs_style') or 0):.1f}pp | {r.get('n_trades')} |"
        )
    lines += [
        "",
        "## Promotion scorecard",
        "",
        table,
        "",
        "## Decision",
        "",
        "- ADVANCE only if label starts with ADVANCE_*",
        "- Paper freeze unchanged unless human copies a candidate",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
