"""Loop F: A/B risk overlays on minalloc k100 full OOS 2018–2025.

Arms (research only; does not change paper freeze):
  A  baseline k100
  B  dd25_soft35 — new entries size×0.35 when book DD ≤ −25%
  C  dd25_soft35_yr — same with yearly peak reset
  D  soft-ban worst-8 losers (from k100 audit) on baseline
  E  dd25_soft35 + soft-ban8
  F  dd30_soft40 milder continuous soft scale
  G  dd35_vt80_yr control (already strong in Loop D/E)

Hypothesis (from AUDIT_k100_baseline): edge is asymmetry (WR~33%, payoff~3×);
hard skip on deep DD kills winners; soft size-scale + soft-ban may cut MDD
without residual ≤ 0.

Promo goal: path MDD ≥ −50% with residual > 0 (Stage1). Paper freeze stays
turbo_highvol_minalloc unless human ADVANCE.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from trad_research.risk_levers import LEVERS  # noqa: E402

# Audit k100 worst sum-PnL with n≥8 (research soft-ban; not live filter)
SOFT_BAN_8: Tuple[str, ...] = (
    "GSAT",
    "FCEL",
    "NAGE",
    "CDZI",
    "DBVT",
    "XNET",
    "CENX",
    "NVFY",
)

STYLE_ID = "growth_ew__volonly_k100_baseline"


def loop_f_arms() -> List[_mega.GridConfig]:
    """Minalloc vol-only k100 × Loop F risk / soft-ban arms."""
    base = "turbo_highvol_minalloc"
    k = 100
    arms: List[Tuple[str, str, Tuple[str, ...], str]] = [
        ("baseline", "baseline", (), "A_control"),
        ("dd25_soft35", "dd25_soft35", (), "B_soft_dd"),
        ("dd25_soft35_yr", "dd25_soft35_yr", (), "C_soft_dd_yr"),
        ("softban8", "baseline", SOFT_BAN_8, "D_softban"),
        ("dd25_soft35_ban8", "dd25_soft35", SOFT_BAN_8, "E_soft_dd_ban"),
        ("dd30_soft40", "dd30_soft40", (), "F_mild_soft"),
        ("dd35_vt80_yr", "dd35_vt80_yr", (), "G_loopd_control"),
    ]
    out: List[_mega.GridConfig] = []
    for suffix, lever, ban, label in arms:
        if lever not in LEVERS:
            raise KeyError(f"Missing lever {lever!r}")
        cid = f"{base}__volonly_k{k}_{suffix}"
        out.append(
            _mega.GridConfig(
                config_id=cid,
                strategy=base,
                growth_hard=False,
                growth_top_k=k,
                lever_id=lever,
                vol_only_top=k,
                vol_pool_n=200,
                label=label,
                exclude_tickers=tuple(ban),
            )
        )
    # Style EW residual bench (same L0 spirit)
    out.append(
        _mega.GridConfig(
            config_id=STYLE_ID,
            strategy="growth_ew",
            growth_hard=False,
            growth_top_k=k,
            lever_id="baseline",
            vol_only_top=k,
            label="style_ew",
        )
    )
    return out


def _promo_candidates(configs: Sequence[_mega.GridConfig]) -> List[str]:
    return [
        c.config_id
        for c in configs
        if c.strategy == "turbo_highvol_minalloc" and c.config_id != STYLE_ID
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Loop F risk A/B full OOS")
    ap.add_argument("--panel", type=Path, default=ROOT / "universe_highvol200.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--first", type=int, default=2018)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "vol_fund_loop_f",
    )
    ap.add_argument(
        "--arms",
        type=str,
        default="",
        help="Comma-separated config_id suffixes to run (empty=all). e.g. baseline,dd25_soft35",
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
    all_cfgs = loop_f_arms()
    if args.arms.strip():
        want = {s.strip() for s in args.arms.split(",") if s.strip()}
        filtered = []
        for c in all_cfgs:
            if c.config_id == STYLE_ID:
                filtered.append(c)
                continue
            # match suffix after volonly_k100_
            rest = c.config_id.split("__volonly_k100_", 1)[-1]
            if rest in want or c.label in want or c.config_id in want:
                filtered.append(c)
        all_cfgs = filtered

    print(
        f"Loop F full OOS {years[0]}–{years[-1]} panel n={len(static_pool)} "
        f"arms={len(all_cfgs)}",
        flush=True,
    )
    for c in all_cfgs:
        print(
            f"  - {c.config_id} lever={c.lever_id} ban={list(c.exclude_tickers) or '-'}",
            flush=True,
        )

    rows: List[Dict[str, Any]] = []
    for cfg in all_cfgs:
        print(f"[full] {cfg.config_id} …", flush=True)
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
        safe = cfg.config_id.replace("/", "_")
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

    promo_ids = _promo_candidates(all_cfgs)
    style_eq_path = configs_dir / STYLE_ID / "equity.csv"
    cards = []
    for cid in promo_ids:
        eq_path = configs_dir / cid / "equity.csv"
        tr_path = configs_dir / cid / "trades.csv"
        if not eq_path.is_file():
            print(f"[promo] skip {cid}: no equity", flush=True)
            continue
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
        eq.index = pd.to_datetime(eq.index, utc=True, errors="coerce")
        eq = eq[~eq.index.duplicated(keep="last")].dropna().sort_index()
        st = None
        if style_eq_path.is_file():
            st = (
                pd.read_csv(style_eq_path, index_col=0, parse_dates=True)
                .iloc[:, 0]
                .astype(float)
            )
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
            n_trials_zoo=max(20, len(promo_ids)),
        )
        print(f"  → {card.label} reasons={card.kill_reasons}", flush=True)
        cards.append(card)

    cards = apply_top_k(cards, k=3)
    table = scorecard_table(cards)

    baseline_row = next(
        (r for r in rows if str(r.get("config_id", "")).endswith("_baseline") and "growth_ew" not in str(r.get("config_id", ""))),
        None,
    )
    ab_lines: List[str] = []
    if baseline_row:
        b_cagr = float(baseline_row.get("cagr") or 0)
        b_mdd = float(baseline_row.get("max_drawdown") or 0)
        b_res = float(baseline_row.get("residual_cagr_vs_style") or 0)
        for r in rows:
            cid = str(r.get("config_id") or "")
            if "growth_ew" in cid or cid.endswith("_baseline"):
                continue
            ab_lines.append(
                f"| `{cid}` | {100*float(r.get('cagr') or 0):.1f}% "
                f"({100*(float(r.get('cagr') or 0)-b_cagr):+.1f}pp) | "
                f"{100*float(r.get('max_drawdown') or 0):.1f}% "
                f"({100*(float(r.get('max_drawdown') or 0)-b_mdd):+.1f}pp) | "
                f"{100*float(r.get('residual_cagr_vs_style') or 0):.1f}pp "
                f"({100*(float(r.get('residual_cagr_vs_style') or 0)-b_res):+.1f}pp) | "
                f"{r.get('n_trades')} |"
            )

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "loop": "F",
        "window": f"{args.first}-{args.last}",
        "panel": str(panel_file),
        "soft_ban_8": list(SOFT_BAN_8),
        "n_sims": int(args.n_sims),
        "full_oos_rows": rows,
        "promotion": [c.to_dict() for c in cards],
        "paper_freeze": "turbo_highvol_minalloc (unchanged unless human ADVANCE)",
        "disclaimer": "Research only. Not financial advice. No paper freeze auto-change.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Loop F — soft DD size-scale + soft-ban A/B (full OOS)",
        "",
        "> **Research only.** Not financial advice. Paper freeze unchanged.",
        "",
        f"- Full OOS: **{args.first}–{args.last}** · panel highvol200 · n_sims={args.n_sims}",
        f"- Soft-ban8 (audit losers n≥8): `{', '.join(SOFT_BAN_8)}`",
        "- Signal: **turbo_highvol_minalloc** vol-only k100 (same as Loop D/E winner family)",
        "",
        "## Full-window metrics",
        "",
        "| arm | config | CAGR | Sortino | MDD | resid vs style | n_trades |",
        "|-----|--------|------|---------|-----|----------------|----------|",
    ]
    for r in rows:
        cid = str(r.get("config_id") or "")
        if "growth_ew" in cid:
            continue
        label = next((c.label for c in all_cfgs if c.config_id == cid), "")
        lines.append(
            f"| {label} | `{cid}` | {100*float(r.get('cagr') or 0):.1f}% | "
            f"{float(r.get('sortino') or 0):.2f} | {100*float(r.get('max_drawdown') or 0):.1f}% | "
            f"{100*float(r.get('residual_cagr_vs_style') or 0):.1f}pp | {r.get('n_trades')} |"
        )

    lines += [
        "",
        "## Δ vs baseline k100",
        "",
        "| config | CAGR (Δ) | MDD (Δ) | residual (Δ) | n |",
        "|--------|----------|---------|--------------|---|",
    ]
    if baseline_row:
        lines.append(
            f"| `…_baseline` | {100*float(baseline_row.get('cagr') or 0):.1f}% | "
            f"{100*float(baseline_row.get('max_drawdown') or 0):.1f}% | "
            f"{100*float(baseline_row.get('residual_cagr_vs_style') or 0):.1f}pp | "
            f"{baseline_row.get('n_trades')} |"
        )
    lines.extend(ab_lines)

    lines += [
        "",
        "## Promotion scorecard",
        "",
        table,
        "",
        "## Decision rules",
        "",
        "- Goal: path **MDD ≥ −50%** with **residual > 0** (and promo not pure KILL on MC)",
        "- ADVANCE only if label starts with ADVANCE_*",
        "- Paper freeze stays **turbo_highvol_minalloc** unless human copies a candidate",
        "- Soft-ban is **research-only** (in-sample from full-window audit losers — leakage risk for claim)",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
