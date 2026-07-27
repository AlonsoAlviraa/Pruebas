"""Promotion scorecard: metrics + residual + Monte Carlo; only ADVANCE_* pass.

Usage::

    $env:PYTHONPATH = (Get-Location).Path
    python scripts/run_promotion_scorecard.py --smoke
    python scripts/run_promotion_scorecard.py --full --n-sims 2000

Reads equity CSVs from redesign packs (no retrain). Research only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.promotion import (
    CandidateInput,
    DEFAULT_THRESHOLDS,
    apply_top_k,
    evaluate_candidate,
    scorecard_table,
)
from trad_research.zoo import append_trial, default_registry_path, load_registry


def _load_eq(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    try:
        s.index = s.index.normalize()
    except Exception:
        pass
    return s[~s.index.duplicated(keep="last")].dropna().sort_index()


def _candidates_from_configs_dir(
    configs_dir: Path,
    *,
    style_name: Optional[str] = None,
    product: str = "STYLE-US",
) -> List[Dict[str, Any]]:
    """Load Phase A / mega-study ``configs/<id>/equity.csv`` trees for promotion.

    ``style_name``: config folder name used as residual benchmark (default:
    first ``*__baseline`` found, else minalloc baseline id).
    """
    configs_dir = Path(configs_dir)
    if not configs_dir.is_dir():
        return []
    equity_map: Dict[str, Path] = {}
    for sub in sorted(configs_dir.iterdir()):
        if not sub.is_dir():
            continue
        eq = sub / "equity.csv"
        if eq.is_file():
            equity_map[sub.name] = eq
    if not equity_map:
        return []

    style_key = style_name
    if style_key is None or style_key not in equity_map:
        for cand in (
            "turbo_highvol_minalloc__baseline",
            "turbo_highvol__baseline",
        ):
            if cand in equity_map:
                style_key = cand
                break
        if style_key is None or style_key not in equity_map:
            # any *baseline*
            for k in equity_map:
                if k.endswith("__baseline"):
                    style_key = k
                    break
        if style_key is None or style_key not in equity_map:
            style_key = next(iter(equity_map))

    style_path = equity_map[style_key]
    cands: List[Dict[str, Any]] = []
    for name, eq_path in equity_map.items():
        # Residual vs style for non-style rows only. Style/control row skips residual
        # (avoids arbitrary peer residual kills; STYLE-US residual is optional).
        if name == style_key:
            st: Optional[Path] = None
        else:
            st = style_path
        trades = eq_path.parent / "trades.csv"
        cands.append(
            {
                "name": name,
                "equity": eq_path,
                "style": st,
                "product": product,
                "geo_p3": None,
                "early_ok": None,
                "trades": trades if trades.is_file() else None,
                "style_key": style_key,
            }
        )
    return cands


def _default_candidates(smoke: bool) -> List[Dict[str, Any]]:
    """Map names to equity paths + style + product flags from existing packs."""
    base = ROOT / "reports" / "redesign"
    # Prefer full modern S1 pack for STYLE-US
    modern = base / "S1_style_clone_gap_full" / "equity"
    early = base / "S1b_early_window_full" / "equity"
    geo = base / "S1c_geo_frozen_full" / "summary.json"
    port = base / "S2_residual_train_smoke" / "equity"
    if smoke:
        # smaller sims only; same files
        pass

    geo_p3 = None
    if geo.is_file():
        try:
            g = json.loads(geo.read_text(encoding="utf-8"))
            geo_p3 = bool(g.get("p3_confirmed_any"))
        except Exception:
            geo_p3 = None

    early_ok = None
    early_sum = base / "S1b_early_window_full" / "summary.json"
    if early_sum.is_file():
        try:
            e = json.loads(early_sum.read_text(encoding="utf-8"))
            # residual positive vs hardest sane style → early residual ok for STYLE
            early_ok = not bool(e.get("p1_confirmed_any_clone"))
            # better: check residual exists positive
            for row in e.get("clones") or []:
                rvs = row.get("residual_vs_style") or {}
                if isinstance(rvs.get("excess_cagr"), (int, float)) and rvs["excess_cagr"] > 0:
                    early_ok = True
                    break
        except Exception:
            early_ok = None

    cands: List[Dict[str, Any]] = []
    if modern.is_dir():
        style = modern / "style_ew_hv.csv"
        for name, product in [
            ("turbo_highvol_minalloc", "STYLE-US"),
            ("style_ew_hv", "STYLE-US"),
            ("style_mom_1m_hv", "STYLE-US"),
        ]:
            p = modern / f"{name}.csv"
            if p.is_file():
                cands.append(
                    {
                        "name": f"modern::{name}",
                        "equity": p,
                        "style": style if name != "style_ew_hv" else modern / "style_mom_1m_hv.csv",
                        "product": product,
                        "geo_p3": None,
                        "early_ok": early_ok,
                    }
                )
    if port.is_dir() and (port / "alpha_portable_v0.csv").is_file():
        cands.append(
            {
                "name": "portable::residual_train",
                "equity": port / "alpha_portable_v0.csv",
                "style": port / "style_ew_hv.csv",
                "product": "ALPHA-PORTABLE",
                "geo_p3": geo_p3,
                "early_ok": early_ok,
            }
        )
    if early.is_dir() and (early / "turbo_highvol_minalloc.csv").is_file():
        cands.append(
            {
                "name": "early::turbo_highvol_minalloc",
                "equity": early / "turbo_highvol_minalloc.csv",
                "style": early / "style_mom_1m_hv.csv"
                if (early / "style_mom_1m_hv.csv").is_file()
                else early / "style_ew_hv.csv",
                "product": "STYLE-US",
                "geo_p3": None,
                "early_ok": True,
            }
        )
    return cands


def _trade_pnls_from_csv(path: Optional[Path]) -> Optional[Any]:
    if path is None or not Path(path).is_file():
        return None
    try:
        import numpy as np

        tdf = pd.read_csv(path)
        col = None
        for c in ("net_profit", "pnl", "profit", "ret"):
            if c in tdf.columns:
                col = c
                break
        if col is None:
            return None
        return np.asarray(tdf[col], dtype=float).ravel()
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Promotion scorecard (Sortino + MC + residual)")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "redesign" / "promotion_scorecard_v1")
    p.add_argument("--n-sims", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--max-advance", type=int, default=3)
    p.add_argument("--register-zoo", action="store_true", help="Append trials to strategy zoo registry")
    p.add_argument(
        "--from-configs-dir",
        type=Path,
        default=None,
        help="Load candidates from mega/week study configs/<id>/equity.csv tree",
    )
    p.add_argument(
        "--style-name",
        type=str,
        default=None,
        help="Config id used as style residual benchmark (default: minalloc baseline)",
    )
    p.add_argument(
        "--product",
        type=str,
        default="STYLE-US",
        help="Product mode for from-configs-dir candidates (default STYLE-US)",
    )
    args = p.parse_args(argv)

    if args.full:
        args.smoke = False
    n_sims = args.n_sims
    if n_sims is None:
        n_sims = 200 if args.smoke else 2000

    if args.from_configs_dir is not None:
        specs = _candidates_from_configs_dir(
            Path(args.from_configs_dir),
            style_name=args.style_name,
            product=str(args.product or "STYLE-US"),
        )
        source = f"configs_dir:{args.from_configs_dir}"
    else:
        specs = _default_candidates(smoke=bool(args.smoke))
        source = "default_redesign_packs"
    if not specs:
        print("No candidate equity files found.", flush=True)
        return 2

    reg_path = default_registry_path(ROOT)
    n_trials = 0
    if reg_path.is_file():
        n_trials = int(load_registry(reg_path).get("n_trials") or 0)

    cards = []
    for i, sp in enumerate(specs):
        eq = _load_eq(Path(sp["equity"]))
        st = _load_eq(Path(sp["style"])) if sp.get("style") and Path(sp["style"]).is_file() else None
        pnls = _trade_pnls_from_csv(sp.get("trades"))
        n_tr = int(len(pnls)) if pnls is not None else None
        print(f"[promo] evaluate {sp['name']} n={len(eq)} …", flush=True)
        card = evaluate_candidate(
            CandidateInput(
                name=sp["name"],
                equity=eq,
                style_equity=st,
                trade_pnls=pnls,
                n_trades=n_tr,
                product=sp.get("product") or "STYLE-US",
                geo_p3_confirmed=sp.get("geo_p3"),
                early_residual_ok=sp.get("early_ok"),
                smoke=bool(args.smoke),
            ),
            n_sims=n_sims,
            seed=args.seed + i,
            n_trials_zoo=max(n_trials + i + 1, 1),
        )
        print(f"  → {card.label} reasons={card.kill_reasons[:5]}", flush=True)
        cards.append(card)
        if args.register_zoo:
            append_trial(
                reg_path,
                strategy=sp["name"],
                tag="promotion_scorecard",
                mode="scorecard",
                market="US",
                metrics=card.metrics,
                passed=card.label.startswith("ADVANCE"),
                product_mode=card.label,
                notes=",".join(card.kill_reasons),
            )

    cards = apply_top_k(cards, k=int(args.max_advance))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_sims": n_sims,
        "smoke": bool(args.smoke),
        "source": source,
        "thresholds": DEFAULT_THRESHOLDS.__dict__,
        "candidates": [c.to_dict() for c in cards],
        "advance": [c.name for c in cards if c.label.startswith("ADVANCE")],
        "hold": [c.name for c in cards if c.label == "HOLD"],
        "kill": [c.name for c in cards if c.label == "KILL"],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = [
        "# Promotion scorecard",
        "",
        f"**n_sims:** {n_sims} · **smoke:** {args.smoke} · **source:** `{source}`",
        "",
        "Thresholds frozen in `docs/design/2026-07-23_metrics_montecarlo_promotion.md`.",
        "",
        "## Results",
        "",
        scorecard_table(cards),
        "",
        f"**ADVANCE:** {payload['advance'] or '*(none)*'}",
        f"**HOLD:** {payload['hold'] or '*(none)*'}",
        f"**KILL:** {payload['kill'] or '*(none)*'}",
        "",
        "## Notes",
        "",
        "- Monte Carlo: bootstrap (primary gates) + shuffle (path dependency).",
        "- Residual vs style required for ALPHA-PORTABLE.",
        "- 0 ADVANCE is a valid scientific outcome.",
        "- STYLE-US control remains `turbo_highvol_minalloc` unless ADVANCE.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[promo] wrote {out / 'SUMMARY.md'}", flush=True)
    print(f"[promo] ADVANCE={payload['advance']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
