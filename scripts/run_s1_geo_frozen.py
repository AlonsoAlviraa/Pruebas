"""S1c — Geo FROZEN falsification ES + DE (STR-01 / P3).

Protocol
--------
* **Train US only** (for ML baseline). Style clones: no train.
* **Eval foreign** ``data_es`` / ``data_de`` with FROZEN_US_TRANSFER.
* **No foreign retrain**, no threshold grid on ES/DE.
* Same foreign L0 ticker file for baseline path and style clones on that market.
* Date-normalize local index benches (IBEX / DAX).

P3 design gate (redesign doc): FROZEN ES/DE excess A1 < 0 and/or MDD > 1.5× US.

Usage::

    python scripts/run_s1_geo_frozen.py --smoke
    python scripts/run_s1_geo_frozen.py --full

Research only. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.alpha_attribution import compare_to_benchmark  # noqa: E402
from trad_research.backtest import BacktestConfig  # noqa: E402
from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.policies import DeploymentPolicy, get_policy  # noqa: E402
from trad_research.strategies import RESEARCH_BASELINE_US, get_strategy  # noqa: E402
from trad_research.strategy_runner import run_strategy_walk_forward  # noqa: E402
from trad_research.style_clone import STYLE_CLONE_NAMES  # noqa: E402
from trad_research.transfer import run_frozen_us_transfer  # noqa: E402
from trad_research.walk_forward import load_benchmark_equity  # noqa: E402


def _resolve(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()

logger = logging.getLogger("s1_geo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MARKETS = {
    "ES": {
        "eval_data_root": Path("data_es"),
        "eval_ticker_file": Path("spain_wf_universe.txt"),
        "preferred_index": ("IBEX",),
        "foreign_suffix_denylist": (".MC",),
        "regime_hint": "portable_not_deep_bear",
    },
    "DE": {
        "eval_data_root": Path("data_de"),
        "eval_ticker_file": Path("germany_wf_universe.txt"),
        "preferred_index": ("DAX",),
        "foreign_suffix_denylist": (".XETRA", ".DE"),
        "regime_hint": "portable_not_deep_bear",
    },
}


def _report_dict(rep: Any) -> Dict[str, Any]:
    if rep is None:
        return {}
    return rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)


def _align_spy_fields(equity: pd.Series, data_root: Path, preferred: tuple) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if equity is None or equity.empty:
        return out
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index, utc=True).normalize()
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    bench = load_benchmark_equity(
        data_root, eq.index.min(), eq.index.max(), preferred=list(preferred)
    )
    if bench is None or bench.empty:
        return out
    b = bench.copy()
    b.index = pd.to_datetime(b.index, utc=True).normalize()
    b = b[~b.index.duplicated(keep="last")].sort_index()
    joined = pd.concat([eq.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
    if len(joined) < 5:
        return out
    start_eq = float(joined["s"].iloc[0])
    b_eq = joined["b"] / float(joined["b"].iloc[0]) * start_eq
    rep = equity_metrics(joined["s"], start_equity=start_eq, benchmark=b_eq)
    out["cagr"] = rep.cagr
    out["sharpe"] = rep.sharpe
    out["max_drawdown"] = rep.max_drawdown
    out["benchmark_cagr"] = rep.benchmark_cagr
    out["excess_cagr_vs_index"] = rep.excess_cagr
    out["start_equity"] = start_eq
    out["index_name"] = preferred[0] if preferred else None
    return out


def run_style_on_foreign(
    name: str,
    *,
    eval_data_root: Path,
    eval_ticker_file: Path,
    universe_limit: int,
    first_oos: int,
    last_oos: int,
    preferred_index: tuple,
    regime_filter: Optional[str] = None,
    policy: Optional[DeploymentPolicy] = None,
) -> Dict[str, Any]:
    """Style clone / no-train strategy walk-forward on foreign panels only.

    When ``policy`` is set (e.g. portable_conservative), apply the same sizing
    scales as the FROZEN baseline for fair residual comparison.
    """
    strat = get_strategy(name)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(eval_ticker_file)
    if regime_filter and hasattr(strat, "regime_filter"):
        strat.regime_filter = regime_filter
    if policy is not None and policy.regime_filter and hasattr(strat, "regime_filter"):
        strat.regime_filter = policy.regime_filter
    base_bt = None
    if policy is not None:
        overrides = policy.to_backtest_overrides(strat.backtest_overrides())
        bt0 = BacktestConfig()
        fields = {**bt0.__dict__, **overrides}
        base_bt = BacktestConfig(
            **{k: v for k, v in fields.items() if k in BacktestConfig.__dataclass_fields__}
        )
    res = run_strategy_walk_forward(
        strat,
        data_root=eval_data_root,
        ticker_file=eval_ticker_file,
        universe_limit=universe_limit,
        first_oos_year=first_oos,
        last_oos_year=last_oos,
        preferred_index=preferred_index,
        base_bt=base_bt,
        use_pit_membership=False,
        pit_equal_weight_benchmark=False,
    )
    eq = res.get("equity")
    if eq is not None and not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    rep = _report_dict(res.get("report"))
    aligned = _align_spy_fields(eq, eval_data_root, preferred_index) if eq is not None else {}
    rep.update({k: v for k, v in aligned.items() if v is not None})
    return {
        "name": name,
        "mode": "FOREIGN_WF_NO_TRAIN",
        "report": rep,
        "equity": eq,
        "n_tickers": res.get("n_tickers"),
        "year_results": res.get("year_results"),
    }


def run_baseline_frozen(
    name: str,
    *,
    train_data_root: Path,
    train_ticker_file: Path,
    eval_data_root: Path,
    eval_ticker_file: Path,
    market_id: str,
    preferred_index: tuple,
    foreign_suffix_denylist: tuple,
    universe_limit_train: int,
    universe_limit_eval: int,
    first_oos: int,
    last_oos: int,
    policy_id: str = "portable_conservative",
) -> Dict[str, Any]:
    strat = get_strategy(name)
    policy = None
    try:
        policy = get_policy(policy_id)
    except Exception:
        policy = None
    # Prefer portable regime on foreign when available via policy
    res = run_frozen_us_transfer(
        strat,
        train_data_root=train_data_root,
        train_ticker_file=train_ticker_file,
        eval_data_root=eval_data_root,
        eval_ticker_file=eval_ticker_file,
        preferred_index=preferred_index,
        first_oos_year=first_oos,
        last_oos_year=last_oos,
        universe_limit_train=universe_limit_train,
        universe_limit_eval=universe_limit_eval,
        policy=policy,
        foreign_suffix_denylist=foreign_suffix_denylist,
        market_id=market_id,
        # Do not invent TRANSFER_CANDIDATE without a real US home PASS
        us_home_pass=False,
    )
    eq = res.get("equity")
    if eq is not None and not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    rep = _report_dict(res.get("report"))
    aligned = _align_spy_fields(eq, eval_data_root, preferred_index) if eq is not None else {}
    for k, v in aligned.items():
        if v is not None:
            rep[k] = v
    return {
        "name": name,
        "mode": "FROZEN_US_TRANSFER",
        "report": rep,
        "equity": eq,
        "gates": res.get("gates"),
        "transfer_passed": res.get("transfer_passed"),
        "product_mode": res.get("product_mode"),
        "n_train_tickers": res.get("n_train_tickers"),
        "n_eval_tickers": res.get("n_eval_tickers"),
        "regime_key": res.get("regime_key"),
        "year_results": res.get("year_results"),
        "market_id": market_id,
    }


def analyze_market(
    market_id: str,
    baseline: Dict[str, Any],
    clones: List[Dict[str, Any]],
    *,
    us_mdd_ref: Optional[float] = None,
) -> Dict[str, Any]:
    b_rep = baseline.get("report") or {}
    b_cagr = float(b_rep.get("cagr") or 0.0)
    b_sh = float(b_rep.get("sharpe") or 0.0)
    b_mdd = float(b_rep.get("max_drawdown") or 0.0)
    b_ex = float(b_rep.get("excess_cagr_vs_index") or b_rep.get("excess_cagr") or 0.0)

    rows = []
    hardest_residual = None
    hardest_name = None
    for c in clones:
        cr = c.get("report") or {}
        c_cagr = float(cr.get("cagr") or 0.0)
        residual_vs_style = None
        beq, ceq = baseline.get("equity"), c.get("equity")
        if beq is not None and ceq is not None:
            try:
                residual_vs_style = compare_to_benchmark(
                    beq, ceq, label=f"{baseline['name']}_vs_{c['name']}"
                ).to_dict()
            except Exception as exc:  # noqa: BLE001
                residual_vs_style = {"error": str(exc)}
        ex = None
        if isinstance(residual_vs_style, dict) and "excess_cagr" in residual_vs_style:
            ex = residual_vs_style["excess_cagr"]
            if hardest_residual is None or ex < hardest_residual:
                hardest_residual = ex
                hardest_name = c["name"]
        rows.append(
            {
                "clone": c["name"],
                "clone_cagr": c_cagr,
                "clone_sharpe": float(cr.get("sharpe") or 0.0),
                "clone_mdd": float(cr.get("max_drawdown") or 0.0),
                "residual_vs_style": residual_vs_style,
            }
        )

    # P3 design: excess A1 < 0 and/or MDD > 1.5× US
    mdd_ratio = None
    if us_mdd_ref is not None and us_mdd_ref < 0 and b_mdd < 0:
        mdd_ratio = abs(b_mdd) / max(abs(us_mdd_ref), 1e-9)
    p3_by_excess = b_ex < 0.0
    p3_by_mdd = mdd_ratio is not None and mdd_ratio > 1.5
    p3_confirmed = bool(p3_by_excess or p3_by_mdd)

    gates = baseline.get("gates")
    if gates is None and baseline.get("report"):
        # rebuild if missing
        try:
            from trad_research.metrics import PerformanceReport

            pass
        except Exception:
            pass

    return {
        "market_id": market_id,
        "baseline": baseline.get("name"),
        "mode": baseline.get("mode"),
        "baseline_cagr": b_cagr,
        "baseline_sharpe": b_sh,
        "baseline_mdd": b_mdd,
        "baseline_excess_vs_index": b_ex,
        "transfer_passed": baseline.get("transfer_passed"),
        "gates": baseline.get("gates"),
        "regime_key": baseline.get("regime_key"),
        "n_eval_tickers": baseline.get("n_eval_tickers") or baseline.get("n_tickers"),
        "clones": rows,
        "hardest_clone": hardest_name,
        "hardest_residual_cagr": hardest_residual,
        "p3": {
            "problem": "P3",
            "confirmed": p3_confirmed,
            "by_excess_vs_index": p3_by_excess,
            "by_mdd_vs_us": p3_by_mdd,
            "mdd_ratio_vs_us": mdd_ratio,
            "us_mdd_ref": us_mdd_ref,
            "thresholds": {"excess_lt_0": 0.0, "mdd_ratio_gt": 1.5},
        },
    }


def write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# S1c Geo FROZEN (ES + DE)",
        "",
        "**Mode:** FROZEN_US_TRANSFER for ML baseline; style clones foreign WF no-train.",
        "**No foreign retrain · no ES/DE threshold grid.**",
        "",
        f"**US MDD reference (for P3 ratio):** {summary.get('us_mdd_ref')}",
        f"**P3 confirmed any market:** **{summary.get('p3_confirmed_any')}**",
        "",
    ]
    for mid, block in (summary.get("markets") or {}).items():
        lines.extend(
            [
                f"## Market {mid}",
                "",
                f"- Baseline: `{block.get('baseline')}` mode={block.get('mode')}",
                f"- CAGR / Sharpe / MDD: {block.get('baseline_cagr', 0):.2%} / "
                f"{block.get('baseline_sharpe', 0):.2f} / {block.get('baseline_mdd', 0):.2%}",
                f"- Excess vs local index: {block.get('baseline_excess_vs_index')}",
                f"- Transfer primary passed: {block.get('transfer_passed')}",
                f"- Hardest residual vs style: {block.get('hardest_residual_cagr')} "
                f"(`{block.get('hardest_clone')}`)",
                f"- **P3 confirmed:** **{(block.get('p3') or {}).get('confirmed')}** "
                f"(detail: {json.dumps(block.get('p3'), default=str)})",
                "",
                "| Clone | CAGR | Sharpe | Residual excess |",
                "|-------|------|--------|-----------------|",
            ]
        )
        for r in block.get("clones") or []:
            rvs = r.get("residual_vs_style") or {}
            rex = rvs.get("excess_cagr") if isinstance(rvs, dict) else None
            rex_s = f"{rex:.2%}" if isinstance(rex, (int, float)) else "n/a"
            lines.append(
                f"| `{r['clone']}` | {r['clone_cagr']:.2%} | {r['clone_sharpe']:.2f} | {rex_s} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Protocol notes",
            "",
            "- Train panels: US `data/` only for ML.",
            "- Eval: `data_es` / `data_de` with local ticker files.",
            "- Isolation: train/eval roots differ; suffix denylist enforced.",
            "",
            "Research only. Not financial advice.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="S1c geo FROZEN ES/DE style residual")
    p.add_argument("--train-data-root", type=Path, default=ROOT / "data")
    p.add_argument("--train-ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    p.add_argument("--baseline", type=str, default=RESEARCH_BASELINE_US)
    p.add_argument(
        "--clones",
        type=str,
        default="style_ew_hv,style_trend_mom_hv",
        help="Style clones on foreign L0 (default subset for runtime)",
    )
    p.add_argument("--markets", type=str, default="ES,DE")
    p.add_argument("--first-oos", type=int, default=2018)
    p.add_argument("--last-oos", type=int, default=2025)
    p.add_argument("--universe-limit-train", type=int, default=40)
    p.add_argument("--universe-limit-eval", type=int, default=40)
    p.add_argument("--policy", type=str, default="portable_conservative")
    p.add_argument("--us-mdd-ref", type=float, default=-0.5128, help="S1 full baseline MDD")
    p.add_argument("--out", type=Path, default=ROOT / "reports/redesign/S1c_geo_frozen")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    args = p.parse_args(argv)

    args.train_data_root = _resolve(args.train_data_root)
    args.train_ticker_file = _resolve(args.train_ticker_file)
    args.out = _resolve(args.out)

    if args.full:
        args.universe_limit_train = 80
        args.universe_limit_eval = 80
        args.first_oos = 2018
        args.last_oos = 2025
        args.clones = ",".join(STYLE_CLONE_NAMES)
    if args.smoke:
        args.universe_limit_train = min(args.universe_limit_train, 20)
        args.universe_limit_eval = min(args.universe_limit_eval, 20)
        args.first_oos = 2022
        args.last_oos = 2024

    if not args.train_data_root.is_dir():
        print(f"ERROR: train data root missing: {args.train_data_root}", flush=True)
        return 2

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = None
    try:
        policy = get_policy(args.policy)
    except Exception:
        policy = None
    meta = {
        "track": "S1c_geo_frozen",
        "baseline": args.baseline,
        "clones": args.clones,
        "markets": args.markets,
        "first_oos": args.first_oos,
        "last_oos": args.last_oos,
        "universe_limit_train": args.universe_limit_train,
        "universe_limit_eval": args.universe_limit_eval,
        "policy": args.policy,
        "smoke": bool(args.smoke),
        "full": bool(args.full),
        "no_foreign_retrain": True,
        "no_threshold_grid": True,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[s1_geo] meta={meta}", flush=True)

    clone_names = [c.strip() for c in args.clones.split(",") if c.strip()]
    markets_out: Dict[str, Any] = {}
    p3_any = False

    for mid in [m.strip().upper() for m in args.markets.split(",") if m.strip()]:
        if mid not in MARKETS:
            print(f"[s1_geo] skip unknown market {mid}", flush=True)
            continue
        cfg = MARKETS[mid]
        eval_root = _resolve(cfg["eval_data_root"])
        eval_tf = _resolve(cfg["eval_ticker_file"])
        if not eval_root.is_dir() or not eval_tf.is_file():
            markets_out[mid] = {"error": f"missing {eval_root} or {eval_tf}"}
            print(f"[s1_geo] {mid}: missing data/universe", flush=True)
            continue

        print(f"[s1_geo] === {mid} baseline FROZEN {args.baseline} ===", flush=True)
        try:
            baseline = run_baseline_frozen(
                args.baseline,
                train_data_root=args.train_data_root,
                train_ticker_file=args.train_ticker_file,
                eval_data_root=eval_root,
                eval_ticker_file=eval_tf,
                market_id=mid,
                preferred_index=cfg["preferred_index"],
                foreign_suffix_denylist=cfg["foreign_suffix_denylist"],
                universe_limit_train=args.universe_limit_train,
                universe_limit_eval=args.universe_limit_eval,
                first_oos=args.first_oos,
                last_oos=args.last_oos,
                policy_id=args.policy,
            )
            print(
                f"  baseline CAGR={baseline['report'].get('cagr')} "
                f"pass={baseline.get('transfer_passed')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("baseline fail %s", mid)
            markets_out[mid] = {"error": str(exc)}
            print(f"  FAIL baseline: {exc}", flush=True)
            continue

        clones_res: List[Dict[str, Any]] = []
        for cn in clone_names:
            print(f"[s1_geo] {mid} style {cn} …", flush=True)
            try:
                # Prefer portable regime on foreign for style shell fairness
                cr = run_style_on_foreign(
                    cn,
                    eval_data_root=eval_root,
                    eval_ticker_file=eval_tf,
                    universe_limit=args.universe_limit_eval,
                    first_oos=args.first_oos,
                    last_oos=args.last_oos,
                    preferred_index=cfg["preferred_index"],
                    regime_filter=cfg.get("regime_hint"),
                    policy=policy,
                )
                clones_res.append(cr)
                print(f"  {cn} CAGR={cr['report'].get('cagr')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("clone fail %s %s", mid, cn)
                print(f"  FAIL {cn}: {exc}", flush=True)

        block = analyze_market(
            mid, baseline, clones_res, us_mdd_ref=float(args.us_mdd_ref)
        )
        if block.get("p3", {}).get("confirmed"):
            p3_any = True
        markets_out[mid] = block

        # Persist equities
        eq_dir = out_dir / "equity" / mid
        eq_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(baseline.get("equity"), pd.Series):
            baseline["equity"].rename("equity").to_csv(
                eq_dir / f"{args.baseline}.csv", header=True
            )
        for cr in clones_res:
            if isinstance(cr.get("equity"), pd.Series):
                cr["equity"].rename("equity").to_csv(
                    eq_dir / f"{cr['name']}.csv", header=True
                )

    summary = {
        "track": "S1c_geo_frozen",
        "run_meta": meta,
        "us_mdd_ref": args.us_mdd_ref,
        "p3_confirmed_any": p3_any,
        "markets": markets_out,
        "protocol": {
            "frozen": True,
            "no_foreign_retrain": True,
            "no_threshold_grid": True,
            "same_L0_per_market": "eval ticker file shared baseline+clones",
            "bench": "local index date-normalized",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    write_summary_md(out_dir / "summary.md", summary)
    print(f"[s1_geo] wrote {out_dir / 'summary.json'} p3_any={p3_any}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
