"""S1 structural autopsy: baseline STYLE-US vs style clones (same L0).

No-leak protocol
----------------
* **Same L0** for all strategies: CLI ``--ticker-file`` only (ignore per-strategy
  universe_source_file overrides that would mix highvol vs good_tickers).
* **Walk-forward OOS years**: train uses bars strictly before year-start with
  embargo (ML baseline only). Style clones need no training.
* **Features**: causal rolling only (SMA/ret_1m already lagged in engineer_m2).
* **Signals**: same-bar filters only; no future columns.
* **SPY bench**: always from ``load_benchmark_equity`` (not replaced by PIT).
* **PIT EW (P2)**: built separately from static panels + membership; does not
  change strategy signals. Static ticker list may still embed mild selection
  bias (documented); PIT membership filters listing for bench construction.

Does NOT retune turbo knobs.

Usage::

    python scripts/run_style_clone_gap.py --full
    python scripts/run_style_clone_gap.py --smoke

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

from trad_research.alpha_attribution import (
    compare_to_benchmark,
    confirm_p1_style_confusion,
    confirm_p2_unfair_spy_bench,
    rank_problems_by_false_alpha,
)
from trad_research.features import list_tickers
from trad_research.metrics import equity_metrics
from trad_research.strategies import RESEARCH_BASELINE_US, get_strategy
from trad_research.strategy_runner import run_strategy_walk_forward
from trad_research.style_clone import STYLE_CLONE_NAMES
from trad_research.walk_forward import _load_panels, load_benchmark_equity

logger = logging.getLogger("style_clone_gap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


NO_LEAK_PROTOCOL = {
    "same_L0": "CLI ticker-file only for all strategies",
    "wf": "OOS calendar years; ML train_end=year-start; embargo in _build_training_frame",
    "features": "engineer_m2 causal rolls; ret_1m = pct_change(21)",
    "style_signals": "no train; SMA/mom same-bar only",
    "spy_bench": "always SPY path via load_benchmark_equity",
    "pit_ew": "post-hoc membership-filtered EW of same ticker panels (bench only)",
    "not_claimed": "static universe list not fully PIT-selected at list build time",
}


def _report_dict(res: Dict[str, Any]) -> Dict[str, Any]:
    rep = res.get("report")
    if rep is None:
        return {}
    return rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)


def _equity_from_result(res: Dict[str, Any]) -> Optional[pd.Series]:
    eq = res.get("equity")
    if eq is None:
        return None
    if not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    return eq.dropna().astype(float)


def run_one(
    name: str,
    *,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    first_oos: int,
    last_oos: int,
) -> Dict[str, Any]:
    """Run one strategy with fixed L0. Never swap ticker file from strategy attrs."""
    strat = get_strategy(name)
    # Force same universe file for P1 fairness (critical no-mix rule)
    if hasattr(strat, "universe_source_file"):
        strat.universe_source_file = str(ticker_file)
    res = run_strategy_walk_forward(
        strat,
        data_root=data_root,
        ticker_file=ticker_file,
        universe_limit=universe_limit,
        first_oos_year=first_oos,
        last_oos_year=last_oos,
        use_pit_membership=False,  # keep static L0 identical; PIT only for P2 bench
        pit_equal_weight_benchmark=False,
    )
    equity = _equity_from_result(res)
    rep = _report_dict(res)
    # Force SPY excess with timezone-safe align + rebase to strategy start equity
    if equity is not None and len(equity) > 5:
        spy = load_benchmark_equity(
            data_root, equity.index.min(), equity.index.max(), preferred=["SPY"]
        )
        if spy is not None and not spy.empty:
            eq = equity.copy()
            eq.index = pd.to_datetime(eq.index, utc=True).normalize()
            eq = eq[~eq.index.duplicated(keep="last")].sort_index()
            spy = spy.copy()
            spy.index = pd.to_datetime(spy.index, utc=True).normalize()
            spy = spy[~spy.index.duplicated(keep="last")].sort_index()
            joined = pd.concat([eq.rename("s"), spy.rename("b")], axis=1, join="inner").dropna()
            if len(joined) > 5:
                start_eq = float(joined["s"].iloc[0])
                b_eq = joined["b"] / float(joined["b"].iloc[0]) * start_eq
                spy_rep = equity_metrics(
                    joined["s"], start_equity=start_eq, benchmark=b_eq
                )
                rep["spy_cagr"] = spy_rep.benchmark_cagr
                rep["spy_sharpe"] = spy_rep.benchmark_sharpe
                rep["excess_cagr_vs_spy"] = spy_rep.excess_cagr
                rep["excess_cagr"] = spy_rep.excess_cagr
                rep["start_equity"] = start_eq
                rep["cagr"] = spy_rep.cagr
                rep["sharpe"] = spy_rep.sharpe
                rep["max_drawdown"] = spy_rep.max_drawdown
    return {
        "name": name,
        "report": rep,
        "equity": equity,
        "year_results": res.get("year_results"),
        "n_tickers": res.get("n_tickers"),
        "use_pit_membership": res.get("use_pit_membership"),
    }


def build_pit_ew_excess(
    equity: pd.Series,
    *,
    data_root: Path,
    ticker_file: Path,
    universe_limit: int,
    start_equity: float,
) -> Dict[str, Any]:
    """P2: strategy excess CAGR vs PIT equal-weight of same ticker set."""
    out: Dict[str, Any] = {"ok": False}
    mem_path = data_root / "pit" / "membership_index.json"
    if not mem_path.is_file():
        out["error"] = f"missing {mem_path}"
        return out
    try:
        from trad_research.pit_universe import MembershipIndex, build_equal_weight_benchmark

        membership = MembershipIndex.load(mem_path)
        tickers = list_tickers(ticker_file, data_root, limit=universe_limit)
        panels = _load_panels(tickers, data_root)
        if not panels or equity is None or equity.empty:
            out["error"] = "empty panels or equity"
            return out
        t0, t1 = equity.index.min(), equity.index.max()
        pit_ew = build_equal_weight_benchmark(panels, membership, t0, t1)
        if pit_ew is None or pit_ew.empty:
            out["error"] = "empty pit_ew"
            return out
        pit_ew = pit_ew / float(pit_ew.iloc[0]) * start_equity
        residual = compare_to_benchmark(equity, pit_ew, start_equity=start_equity, label="vs_pit_ew")
        out.update(
            {
                "ok": True,
                "excess_cagr_vs_pit_ew": residual.excess_cagr,
                "pit_ew_cagr": residual.bench_cagr,
                "strategy_cagr": residual.strategy_cagr,
                "residual_sharpe": residual.residual_sharpe,
                "n_panels": len(panels),
            }
        )
        out["p2"] = confirm_p2_unfair_spy_bench(float(residual.excess_cagr))
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def analyze(
    baseline: Dict[str, Any],
    clones: List[Dict[str, Any]],
    *,
    pit_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    b_rep = baseline.get("report") or {}
    b_cagr = float(b_rep.get("cagr") or 0.0)
    b_sh = float(b_rep.get("sharpe") or 0.0)
    b_mdd = float(b_rep.get("max_drawdown") or 0.0)
    b_spy_ex = b_rep.get("excess_cagr_vs_spy")
    if b_spy_ex is None:
        b_spy_ex = b_rep.get("excess_cagr")
    b_spy_ex = float(b_spy_ex or 0.0)

    rows = []
    best_capture = None
    best_clone = None
    best_residual_cagr = None
    best_residual_clone = None

    for c in clones:
        cr = c.get("report") or {}
        c_cagr = float(cr.get("cagr") or 0.0)
        c_sh = float(cr.get("sharpe") or 0.0)
        c_spy_ex = cr.get("excess_cagr_vs_spy")
        if c_spy_ex is None:
            c_spy_ex = cr.get("excess_cagr")
        c_spy_ex = float(c_spy_ex or 0.0)
        p1 = confirm_p1_style_confusion(
            baseline_excess_vs_spy=b_spy_ex,
            clone_excess_vs_spy=c_spy_ex,
            baseline_sharpe=b_sh,
            clone_sharpe=c_sh,
            clone_cagr=c_cagr,
        )
        residual_cagr_simple = b_cagr - c_cagr
        residual_vs_style = None
        beq, ceq = baseline.get("equity"), c.get("equity")
        if beq is not None and ceq is not None:
            try:
                residual_vs_style = compare_to_benchmark(
                    beq, ceq, label=f"{baseline['name']}_vs_{c['name']}"
                ).to_dict()
            except Exception as exc:  # noqa: BLE001
                residual_vs_style = {"error": str(exc)}
        ex_style = None
        patho = bool(p1.get("pathology_suspect"))
        if isinstance(residual_vs_style, dict) and "excess_cagr" in residual_vs_style:
            ex_style = residual_vs_style["excess_cagr"]
            # Hardest residual among **non-pathological** clones only
            if not patho and (best_residual_cagr is None or ex_style < best_residual_cagr):
                best_residual_cagr = ex_style
                best_residual_clone = c["name"]
        row = {
            "clone": c["name"],
            "clone_cagr": c_cagr,
            "clone_sharpe": c_sh,
            "clone_mdd": float(cr.get("max_drawdown") or 0.0),
            "clone_excess_vs_spy": c_spy_ex,
            "baseline_minus_clone_cagr": residual_cagr_simple,
            "p1": p1,
            "pathology_suspect": patho,
            "residual_vs_style": residual_vs_style,
        }
        rows.append(row)
        cap = p1.get("style_capture")
        if (
            cap is not None
            and not patho
            and (best_capture is None or cap > best_capture)
        ):
            best_capture = cap
            best_clone = c["name"]

    p1_any = any(bool(r["p1"]["confirmed"]) and not r.get("pathology_suspect") for r in rows)
    # Also confirm P1 if residual excess vs hardest *sane* clone <= 0
    if best_residual_cagr is not None and best_residual_cagr <= 0.0:
        p1_any = True

    p2 = None
    if pit_block and pit_block.get("ok"):
        p2 = pit_block.get("p2")

    ranking = rank_problems_by_false_alpha(
        [{"problem": "P1", "confirmed": p1_any, "best_capture": best_capture}]
        + (
            [p2]
            if p2
            else [{"problem": "P2", "confirmed": False, "note": "pit excess n/a"}]
        )
        + [
            {"problem": "P3", "confirmed": False, "note": "transfer not in this run"},
            {"problem": "P4", "confirmed": False, "note": "signal corr pending"},
            {"problem": "P5", "confirmed": False, "note": "abs vs rel pending"},
        ]
    )

    return {
        "protocol": NO_LEAK_PROTOCOL,
        "baseline": baseline["name"],
        "baseline_cagr": b_cagr,
        "baseline_sharpe": b_sh,
        "baseline_mdd": b_mdd,
        "baseline_excess_vs_spy": b_spy_ex,
        "baseline_spy_cagr": b_rep.get("spy_cagr"),
        "n_tickers": baseline.get("n_tickers"),
        "clones": rows,
        "p1_confirmed_any_clone": p1_any,
        "best_style_capture": best_capture,
        "best_clone_by_capture": best_clone,
        "hardest_clone_residual_cagr": best_residual_cagr,
        "hardest_clone": best_residual_clone,
        "p2": p2,
        "pit_block": {
            k: v for k, v in (pit_block or {}).items() if k != "p2"
        },
        "problem_ranking": ranking,
    }


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# S1 Style-clone gap (structural autopsy)",
        "",
        "**Data:** real EODHD OHLCV under `data/` · **No synthetic prices**",
        "",
        "## No-leak protocol",
        "",
        "```json",
        json.dumps(summary.get("protocol"), indent=2),
        "```",
        "",
        f"**Baseline (STYLE-US control):** `{summary.get('baseline')}`",
        f"**N tickers (L0):** {summary.get('n_tickers')}",
        f"**Baseline CAGR / Sharpe / MDD:** "
        f"{summary.get('baseline_cagr', 0):.2%} / "
        f"{summary.get('baseline_sharpe', 0):.2f} / "
        f"{summary.get('baseline_mdd', 0):.2%}",
        f"**SPY CAGR (aligned):** {summary.get('baseline_spy_cagr')}",
        f"**Baseline excess vs SPY:** {summary.get('baseline_excess_vs_spy', 0):.2%}",
        "",
        f"**P1 confirmed (any clone):** **{summary.get('p1_confirmed_any_clone')}**",
        f"**Best style capture (clone excess / baseline excess vs SPY):** "
        f"{summary.get('best_style_capture')} (`{summary.get('best_clone_by_capture')}`)",
        f"**Hardest clone residual CAGR (baseline − clone path):** "
        f"{summary.get('hardest_clone_residual_cagr')} (`{summary.get('hardest_clone')}`)",
        "",
        "## Clones (same L0)",
        "",
        "| Clone | CAGR | Sharpe | MDD | Excess vs SPY | Base−clone CAGR | Residual excess | P1 |",
        "|-------|------|--------|-----|---------------|-----------------|-----------------|----|",
    ]
    for r in summary.get("clones") or []:
        rvs = r.get("residual_vs_style") or {}
        rex = rvs.get("excess_cagr") if isinstance(rvs, dict) else None
        rex_s = f"{rex:.2%}" if isinstance(rex, (int, float)) else "n/a"
        lines.append(
            f"| `{r['clone']}` | {r['clone_cagr']:.2%} | {r['clone_sharpe']:.2f} | "
            f"{r['clone_mdd']:.2%} | {r['clone_excess_vs_spy']:.2%} | "
            f"{r['baseline_minus_clone_cagr']:.2%} | {rex_s} | "
            f"{r['p1'].get('confirmed')} |"
        )
    pit = summary.get("pit_block") or {}
    lines.extend(
        [
            "",
            "## P2 — unfair SPY bench (vs PIT EW)",
            "",
            f"- PIT block ok: {pit.get('ok')}",
            f"- Strategy CAGR: {pit.get('strategy_cagr')}",
            f"- PIT EW CAGR: {pit.get('pit_ew_cagr')}",
            f"- Excess vs PIT EW: {pit.get('excess_cagr_vs_pit_ew')}",
            f"- P2 confirmed: **{(summary.get('p2') or {}).get('confirmed')}**",
            f"- Error: {pit.get('error')}",
            "",
            "## Problem ranking",
            "",
            "```json",
            json.dumps(summary.get("problem_ranking"), indent=2, default=str),
            "```",
            "",
            "Research only. Not financial advice.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Style-clone gap autopsy (S1 redesign)")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    p.add_argument("--universe-limit", type=int, default=80)
    p.add_argument("--first-oos", type=int, default=2018)
    p.add_argument("--last-oos", type=int, default=2025)
    p.add_argument("--baseline", type=str, default=RESEARCH_BASELINE_US)
    p.add_argument("--clones", type=str, default=",".join(STYLE_CLONE_NAMES))
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/redesign/S1_style_clone_gap",
    )
    p.add_argument("--smoke", action="store_true", help="Small real-data smoke")
    p.add_argument(
        "--full",
        action="store_true",
        help="Full real-data: universe_limit=80, OOS 2018-2025",
    )
    args = p.parse_args(argv)

    def _resolve(path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (ROOT / path).resolve()

    args.data_root = _resolve(args.data_root)
    args.ticker_file = _resolve(args.ticker_file)
    args.out = _resolve(args.out)

    if args.full:
        args.universe_limit = 80
        args.first_oos = 2018
        args.last_oos = 2025
        args.ticker_file = _resolve(Path("universe_highvol80.txt"))
    if args.smoke:
        args.universe_limit = min(args.universe_limit, 20)
        args.first_oos = 2022
        args.last_oos = 2024

    if not args.ticker_file.is_file():
        print(f"ERROR: ticker file missing: {args.ticker_file}", flush=True)
        return 2
    if not args.data_root.is_dir():
        print(f"ERROR: data root missing: {args.data_root}", flush=True)
        return 2

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "ticker_file": str(args.ticker_file),
        "universe_limit": args.universe_limit,
        "first_oos": args.first_oos,
        "last_oos": args.last_oos,
        "smoke": bool(args.smoke),
        "full": bool(args.full),
        "protocol": NO_LEAK_PROTOCOL,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[style_clone_gap] meta={meta}", flush=True)

    names = [args.baseline] + [c.strip() for c in args.clones.split(",") if c.strip()]
    results: Dict[str, Dict[str, Any]] = {}
    for name in names:
        print(f"[style_clone_gap] running {name} …", flush=True)
        try:
            results[name] = run_one(
                name,
                data_root=args.data_root,
                ticker_file=args.ticker_file,
                universe_limit=args.universe_limit,
                first_oos=args.first_oos,
                last_oos=args.last_oos,
            )
            rep = results[name].get("report") or {}
            print(
                f"  CAGR={rep.get('cagr')} Sharpe={rep.get('sharpe')} "
                f"excess_spy={rep.get('excess_cagr_vs_spy')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("FAIL %s", name)
            print(f"  FAIL {name}: {exc}", flush=True)
            results[name] = {"name": name, "error": str(exc), "report": {}}

    baseline = results.get(args.baseline) or {"name": args.baseline, "report": {}}
    clones = [
        results[n]
        for n in names
        if n != args.baseline and results.get(n) and "error" not in results[n]
    ]

    pit_block: Dict[str, Any] = {"ok": False}
    beq = baseline.get("equity")
    if beq is not None and "error" not in baseline:
        start_eq = float((baseline.get("report") or {}).get("start_equity") or 100_000.0)
        print("[style_clone_gap] building PIT EW residual (P2)…", flush=True)
        pit_block = build_pit_ew_excess(
            beq,
            data_root=args.data_root,
            ticker_file=args.ticker_file,
            universe_limit=args.universe_limit,
            start_equity=start_eq,
        )
        print(f"  pit_block={ {k: pit_block.get(k) for k in ('ok','excess_cagr_vs_pit_ew','error')} }", flush=True)

    summary = analyze(baseline, clones, pit_block=pit_block)
    summary["run_meta"] = meta

    # Persist equities lightly (dates + values) for audit
    eq_dir = out_dir / "equity"
    eq_dir.mkdir(exist_ok=True)
    for name, res in results.items():
        eq = res.get("equity")
        if isinstance(eq, pd.Series) and not eq.empty:
            eq.rename("equity").to_csv(eq_dir / f"{name}.csv", header=True)

    json_path = out_dir / "summary.json"
    slim = json.loads(json.dumps(summary, default=str))
    json_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    write_md(out_dir / "S1_style_clone_gap.md", summary)
    print(f"[style_clone_gap] wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
