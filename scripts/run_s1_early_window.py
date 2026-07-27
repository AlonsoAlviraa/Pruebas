"""S1b — Early-window style-clone gap falsification (STR-01 extension / P1).

Chosen window (data-constrained)
--------------------------------
Modern ``universe_highvol80`` EOD history mostly starts ~2014-06, so a true
2005–2014 OOS on the S1 modern L0 is **impossible** with this cache.

**Primary early window:** OOS calendar years **2010–2014** (smoke may use
2012–2014) on ``universe_early_longhist.txt``.

**Causal L0 construction (v2):**
* Membership **as-of first OOS** (``as_of={first_oos}-01-01``): history start
  on/before warm-up need; still listed on as_of.
* **No** requirement to survive past OOS end (no post-OOS survivorship filter).
* ADV$ window ends **strictly before** first OOS.
* Sidecar ``*.meta.json`` fingerprint forces rebuild when as_of/params change.

Same L0 ticker file for baseline + all style clones (no mix).

Caveats
-------
* Early L0 ≠ modern highvol80 L0 (documented; unavoidable with cache).
* List may still embed mild folder/selection bias (not full CRSP PIT).
* ML baseline trains only on bars before each OOS year (embargo in WF).
* SPY bench date-normalized + rebased (same as S1).

Usage::

    python scripts/run_s1_early_window.py --smoke
    python scripts/run_s1_early_window.py --full

Research only. Not financial advice.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402


def _load_style_clone_gap():
    path = ROOT / "scripts" / "run_style_clone_gap.py"
    spec = importlib.util.spec_from_file_location("run_style_clone_gap", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_scg = _load_style_clone_gap()
NO_LEAK_PROTOCOL = _scg.NO_LEAK_PROTOCOL
analyze = _scg.analyze
build_pit_ew_excess = _scg.build_pit_ew_excess
run_one = _scg.run_one
write_md = _scg.write_md

from trad_research.early_universe import ensure_early_universe_file  # noqa: E402
from trad_research.strategies import RESEARCH_BASELINE_US  # noqa: E402
from trad_research.style_clone import STYLE_CLONE_NAMES  # noqa: E402

logger = logging.getLogger("s1_early")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EARLY_PROTOCOL = {
    **NO_LEAK_PROTOCOL,
    "window_choice": (
        "OOS 2010-2014 (design: 2005-2014 or 2010-2014 if thinner); "
        "highvol80 lacks pre-2014 bars → early_longhist L0"
    ),
    "l0_note": "Same early ticker file for baseline+clones; not equal to modern highvol80",
}


def _resolve(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def apply_smoke_full_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply --full / --smoke presets (unit-testable; used by main)."""
    if getattr(args, "full", False):
        args.universe_limit = min(int(args.universe_limit), 40)
        args.first_oos = 2010
        args.last_oos = 2014
    if getattr(args, "smoke", False):
        args.universe_limit = min(int(args.universe_limit), 15)
        args.first_oos = 2012
        args.last_oos = 2014
    return args


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S1b early-window style-clone gap")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument(
        "--ticker-file",
        type=Path,
        default=ROOT / "universe_early_longhist.txt",
        help="Early L0 list (auto-built if missing)",
    )
    p.add_argument("--universe-limit", type=int, default=40)
    p.add_argument("--first-oos", type=int, default=2010)
    p.add_argument("--last-oos", type=int, default=2014)
    p.add_argument("--baseline", type=str, default=RESEARCH_BASELINE_US)
    p.add_argument("--clones", type=str, default=",".join(STYLE_CLONE_NAMES))
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/redesign/S1b_early_window",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--rebuild-universe", action="store_true")
    args = p.parse_args(argv)
    args.data_root = _resolve(args.data_root)
    args.ticker_file = _resolve(args.ticker_file)
    args.out = _resolve(args.out)
    return apply_smoke_full_defaults(args)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.data_root.is_dir():
        print(f"ERROR: data root missing: {args.data_root}", flush=True)
        return 2

    # Causal L0 as-of first OOS start; ADV ends before OOS
    as_of = f"{args.first_oos}-01-01"
    ensure_early_universe_file(
        args.ticker_file,
        data_root=args.data_root,
        max_names=max(args.universe_limit, 20),
        rebuild=bool(args.rebuild_universe) or not args.ticker_file.is_file(),
        as_of=as_of,
        history_start_need="2005-06-01",
    )
    if not args.ticker_file.is_file():
        print(f"ERROR: could not build ticker file {args.ticker_file}", flush=True)
        return 2

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    n_lines = len(
        [ln for ln in args.ticker_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    )
    meta = {
        "track": "S1b_early_window",
        "ticker_file": str(args.ticker_file),
        "n_tickers_file": n_lines,
        "universe_limit": args.universe_limit,
        "first_oos": args.first_oos,
        "last_oos": args.last_oos,
        "l0_as_of": as_of,
        "l0_causal": True,
        "adv_ends_before_oos": True,
        "no_post_oos_survivorship": True,
        "smoke": bool(args.smoke),
        "full": bool(args.full),
        "protocol": EARLY_PROTOCOL,
        "window_rationale": (
            "Chosen OOS 2010-2014 (thinner than 2005-2014) because most liquid "
            "names in data/ start 2014; L0 built as-of first OOS with ADV window "
            "strictly before OOS (no last_need survivorship past OOS end)."
        ),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[s1_early] meta={meta}", flush=True)

    names = [args.baseline] + [c.strip() for c in args.clones.split(",") if c.strip()]
    results: Dict[str, Dict[str, Any]] = {}
    for name in names:
        print(f"[s1_early] running {name} …", flush=True)
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

    pit_block: Dict[str, Any] = {"ok": False, "note": "PIT membership may lack early names"}
    beq = baseline.get("equity")
    if beq is not None and "error" not in baseline:
        start_eq = float((baseline.get("report") or {}).get("start_equity") or 100_000.0)
        print("[s1_early] PIT EW residual (best effort)…", flush=True)
        pit_block = build_pit_ew_excess(
            beq,
            data_root=args.data_root,
            ticker_file=args.ticker_file,
            universe_limit=args.universe_limit,
            start_equity=start_eq,
        )
        print(
            f"  pit_block={ {k: pit_block.get(k) for k in ('ok', 'excess_cagr_vs_pit_ew', 'error')} }",
            flush=True,
        )

    summary = analyze(baseline, clones, pit_block=pit_block)
    summary["protocol"] = EARLY_PROTOCOL
    summary["run_meta"] = meta
    summary["track"] = "S1b_early_window"

    eq_dir = out_dir / "equity"
    eq_dir.mkdir(exist_ok=True)
    import pandas as pd

    for name, res in results.items():
        eq = res.get("equity")
        if isinstance(eq, pd.Series) and not eq.empty:
            eq.rename("equity").to_csv(eq_dir / f"{name}.csv", header=True)

    json_path = out_dir / "summary.json"
    slim = json.loads(json.dumps(summary, default=str))
    json_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    # Human report (reuse S1 markdown + early header)
    md_path = out_dir / "summary.md"
    write_md(out_dir / "S1_style_clone_gap.md", summary)
    extra = [
        "# S1b Early-window style-clone gap",
        "",
        f"**Window:** OOS **{args.first_oos}–{args.last_oos}**",
        f"**L0:** `{args.ticker_file}` (limit={args.universe_limit})",
        f"**Rationale:** {meta['window_rationale']}",
        "",
        "See also `S1_style_clone_gap.md` / `summary.json`.",
        "",
    ]
    body = (out_dir / "S1_style_clone_gap.md").read_text(encoding="utf-8")
    md_path.write_text("\n".join(extra) + "\n" + body, encoding="utf-8")
    print(f"[s1_early] wrote {json_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
