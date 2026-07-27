#!/usr/bin/env python
"""Overnight smart mega-loop: risk/MDD lattice + survivor overlays + promotion.

Designed for ~7h wall time while you sleep. **Not** random CPU burn:
  - Only turbo_highvol_minalloc (STYLE-US)
  - No continuous+hard DD (cash-trap lesson)
  - Dense near HOLD (dd35/vt80); sparse tails
  - Batched checkpoints so partial results survive
  - Phase-2 overlays only on Phase-1 survivors
  - Promotion on top K at end if time remains

Research only. Not financial advice. Past backtests ≠ future results.

Usage (PowerShell, repo root):
  $env:PYTHONPATH = (Get-Location).Path
  python scripts/run_overnight_mega_loop.py --hours 7 --mode full --universe-limit 40

Resume after crash (skips fingerprints already in out/phase1*/configs):
  python scripts/run_overnight_mega_loop.py --hours 7 --mode full --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("overnight_mega")


def _chunk(xs: List[Any], n: int) -> List[List[Any]]:
    n = max(1, int(n))
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def _load_done_fps(out_dir: Path) -> Set[str]:
    done: Set[str] = set()
    for p in out_dir.rglob("metrics.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            fp = d.get("fingerprint")
            if fp:
                done.add(str(fp))
            # also label-based anchors
            lab = d.get("label") or d.get("id")
            if lab:
                done.add(str(lab))
        except Exception:
            continue
    # phase batch all_rows
    for p in out_dir.rglob("summary.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            for r in d.get("all_rows") or []:
                if r.get("fingerprint"):
                    done.add(str(r["fingerprint"]))
                if r.get("label"):
                    done.add(str(r["label"]))
                if r.get("id"):
                    done.add(str(r["id"]))
        except Exception:
            continue
    return done


def _strategy_overrides() -> Dict[str, Any]:
    from trad_research.strategies import HighVolMinAllocStrategy

    return dict(HighVolMinAllocStrategy().backtest_overrides())


def _rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]):
        mdd = float(r.get("max_drawdown") or -1.0)
        # Prefer shallower MDD (less negative) + higher excess + sharpe
        excess = float(r.get("excess_total_vs_spy") or -9.0)
        sh = float(r.get("sharpe") or -9.0)
        cagr = float(r.get("cagr") or -9.0)
        # Stage1-ish score: reward MDD > -0.50
        mdd_ok = 1.0 if mdd >= -0.50 else 0.0
        return (mdd_ok, mdd, excess, sh, cagr)

    return sorted(rows, key=key, reverse=True)


def _merge_rows(*row_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for rows in row_lists:
        for r in rows:
            cid = str(r.get("id") or "")
            if not cid or r.get("error"):
                continue
            by_id[cid] = r
    return list(by_id.values())


def _write_progress(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "PROGRESS.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_summary_md(
    out_dir: Path,
    *,
    phase1_rows: List[Dict[str, Any]],
    phase2_rows: List[Dict[str, Any]],
    promo: Optional[Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    ranked = _rank_rows(_merge_rows(phase1_rows, phase2_rows))
    lines = [
        "# Overnight mega-loop SUMMARY",
        "",
        "> **Research only.** Not financial advice. Past backtests ≠ future results.",
        "",
        f"- Mode: `{meta.get('mode')}`",
        f"- Hours budget: {meta.get('hours')}",
        f"- Elapsed: {meta.get('elapsed_sec')}s",
        f"- Universe limit: {meta.get('universe_limit')}",
        f"- OOS: {meta.get('first_oos')}–{meta.get('last_oos')}",
        f"- Phase1 configs completed: {len(phase1_rows)}",
        f"- Phase2 overlay configs completed: {len(phase2_rows)}",
        f"- Stop reason: {meta.get('stop_reason')}",
        "",
        "## Top 15 by research rank (MDD pass priority → MDD → excess → Sharpe)",
        "",
        "| rank | id | CAGR | WR | Sharpe | MDD | excess SPY | n | family |",
        "|------|-----|------|-----|--------|-----|------------|---|--------|",
    ]
    for i, r in enumerate(ranked[:15], 1):
        lines.append(
            f"| {i} | `{r.get('id')}` | {100*float(r.get('cagr') or 0):.1f}% | "
            f"{100*float(r.get('win_rate') or 0):.1f}% | {float(r.get('sharpe') or 0):.2f} | "
            f"{100*float(r.get('max_drawdown') or 0):.1f}% | "
            f"{100*float(r.get('excess_total_vs_spy') or 0):.0f}% | {r.get('n_trades')} | "
            f"{r.get('family') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Baseline vs best MDD-pass (if any)",
            "",
        ]
    )
    base = next((r for r in ranked if str(r.get("label")) == "baseline"), None)
    mdd_pass = [r for r in ranked if float(r.get("max_drawdown") or -1) >= -0.50]
    if base:
        lines.append(
            f"- Baseline: CAGR {100*float(base.get('cagr') or 0):.1f}% MDD "
            f"{100*float(base.get('max_drawdown') or 0):.1f}% Sharpe {float(base.get('sharpe') or 0):.2f}"
        )
    if mdd_pass:
        b = mdd_pass[0]
        lines.append(
            f"- Best Stage1-MDD-pass: `{b.get('id')}` CAGR {100*float(b.get('cagr') or 0):.1f}% "
            f"MDD {100*float(b.get('max_drawdown') or 0):.1f}% excess "
            f"{100*float(b.get('excess_total_vs_spy') or 0):.0f}%"
        )
    else:
        lines.append("- No config with MDD ≥ −50% in completed set.")

    if promo:
        lines.extend(["", "## Promotion (top-K)", ""])
        adv = promo.get("advance") or promo.get("n_advance") or []
        lines.append(f"- ADVANCE count / list: {adv}")
        lines.append(f"- See `phase3_promo/SUMMARY.md` for full funnel.")

    lines.extend(
        [
            "",
            "## Design notes",
            "",
            "- No continuous+hard DD circuits (cash trap).",
            "- Phase2 overlays only on Phase1 survivors.",
            "- Paper freeze not auto-updated.",
            "",
        ]
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Overnight smart mega risk/MDD loop")
    ap.add_argument("--hours", type=float, default=7.0, help="Wall-clock budget hours")
    ap.add_argument(
        "--mode",
        choices=["smoke", "medium", "full"],
        default="full",
        help="Grid density (full ~1.5k–2k risk arms)",
    )
    ap.add_argument("--universe-limit", type=int, default=40)
    ap.add_argument("--first-oos", type=int, default=2018)
    ap.add_argument("--last-oos", type=int, default=2025)
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--phase2-survivors", type=int, default=35)
    ap.add_argument("--promo-top", type=int, default=20)
    ap.add_argument("--n-sims", type=int, default=500, help="MC sims for promotion")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "overnight_2026-07-23",
    )
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--ticker-file",
        type=Path,
        default=ROOT / "universe_highvol80.txt",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip fingerprints already present under --out",
    )
    ap.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Only Phase1 risk lattice",
    )
    ap.add_argument(
        "--skip-promo",
        action="store_true",
        help="Skip promotion scorecard",
    )
    args = ap.parse_args()

    from trad_research.overnight_grid import (
        OvernightCell,
        build_phase1_risk_cells,
        build_phase2_overlay_cells,
        cells_to_mega_configs,
        estimate_grid_sizes,
        overlay_to_mega_configs,
    )
    from scripts.run_crash_entry_mega_study import run_mega

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tee log
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    budget_s = float(args.hours) * 3600.0
    t0 = time.time()
    stop_reason = "completed"

    sizes = estimate_grid_sizes()
    logger.info("Grid size estimates: %s", sizes)
    logger.info(
        "Start overnight mode=%s hours=%.1f univ=%s OOS=%s-%s out=%s",
        args.mode,
        args.hours,
        args.universe_limit,
        args.first_oos,
        args.last_oos,
        out_dir,
    )

    exclude: Set[str] = set()
    if args.resume:
        exclude = _load_done_fps(out_dir)
        logger.info("Resume: %d fingerprints/labels already done", len(exclude))

    cells = build_phase1_risk_cells(mode=args.mode, exclude_fps=exclude)
    # Also filter by label in exclude
    cells = [c for c in cells if c.label not in exclude and c.fingerprint() not in exclude]
    logger.info("Phase1 cells after exclude: %d", len(cells))

    overrides = _strategy_overrides()
    phase1_rows: List[Dict[str, Any]] = []
    phase2_rows: List[Dict[str, Any]] = []

    # Load prior batch rows if resume
    for p in sorted(out_dir.glob("phase1_batch_*/summary.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            phase1_rows.extend(d.get("all_rows") or [])
        except Exception:
            pass

    batches = _chunk(cells, args.batch_size)
    for bi, batch in enumerate(batches):
        elapsed = time.time() - t0
        if elapsed >= budget_s * 0.85:
            # leave ~15% for phase2 survivors + promo (overlays are few)
            stop_reason = f"time_budget_phase1_after_batch_{bi}"
            logger.warning("Stopping Phase1 for time budget (%.0fs used)", elapsed)
            break
        batch_dir = out_dir / f"phase1_batch_{bi:03d}"
        configs = cells_to_mega_configs(batch, strategy_overrides=overrides)
        # Attach fingerprint into metrics via label family stored in configs — run_mega
        # doesn't know fingerprint; we inject into id/label only. Store map.
        fp_map = {f"turbo_highvol_minalloc__{c.label}": c.fingerprint() for c in batch}
        fam_map = {f"turbo_highvol_minalloc__{c.label}": c.family for c in batch}
        logger.info(
            "Phase1 batch %d/%d n=%d elapsed=%.0fs",
            bi + 1,
            len(batches),
            len(configs),
            elapsed,
        )
        try:
            summary = run_mega(
                data_root=args.data_root,
                ticker_file=args.ticker_file,
                universe_limit=args.universe_limit,
                first_oos=args.first_oos,
                last_oos=args.last_oos,
                grid=f"overnight_phase1_b{bi}",
                out_dir=batch_dir,
                configs=configs,
            )
        except Exception:
            logger.error("Batch %d failed:\n%s", bi, traceback.format_exc())
            stop_reason = f"error_phase1_batch_{bi}"
            break

        rows = summary.get("all_rows") or []
        for r in rows:
            rid = str(r.get("id") or "")
            r["fingerprint"] = fp_map.get(rid)
            r["family"] = fam_map.get(rid)
            # Persist fingerprint into metrics.json
            mpath = batch_dir / "configs" / rid / "metrics.json"
            if mpath.exists():
                try:
                    md = json.loads(mpath.read_text(encoding="utf-8"))
                    md["fingerprint"] = r["fingerprint"]
                    md["family"] = r["family"]
                    mpath.write_text(json.dumps(md, indent=2, default=str), encoding="utf-8")
                except Exception:
                    pass
        # Rewrite summary with fingerprints
        summary["all_rows"] = rows
        (batch_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        phase1_rows = _merge_rows(phase1_rows, rows)
        _write_progress(
            out_dir,
            {
                "phase": "phase1",
                "batch": bi,
                "n_phase1": len(phase1_rows),
                "elapsed_sec": round(time.time() - t0, 1),
                "budget_sec": budget_s,
            },
        )
        # Partial global summary
        _write_summary_md(
            out_dir,
            phase1_rows=phase1_rows,
            phase2_rows=phase2_rows,
            promo=None,
            meta={
                "mode": args.mode,
                "hours": args.hours,
                "elapsed_sec": round(time.time() - t0, 1),
                "universe_limit": args.universe_limit,
                "first_oos": args.first_oos,
                "last_oos": args.last_oos,
                "stop_reason": "in_progress_phase1",
            },
        )

    # Phase 2: overlays on survivors
    if (not args.skip_phase2) and (
        stop_reason == "completed" or "time_budget_phase1" in stop_reason
    ):
        elapsed = time.time() - t0
        if elapsed < budget_s * 0.88:
            ranked = _rank_rows(phase1_rows)
            # Map id -> OvernightCell reconstruction from fingerprint is heavy;
            # instead re-match cells by label
            by_label = {c.label: c for c in build_phase1_risk_cells(mode=args.mode)}
            survivors: List[OvernightCell] = []
            for r in ranked:
                lab = str(r.get("label") or "")
                # strip overlay suffix if any
                if lab in by_label:
                    survivors.append(by_label[lab])
                if len(survivors) >= int(args.phase2_survivors):
                    break
            # Always include baseline + dd35 anchor if present
            for must in ("baseline", "dd35_vt80_yr"):
                if must in by_label and by_label[must] not in survivors:
                    survivors.insert(0, by_label[must])

            pairs = build_phase2_overlay_cells(
                survivors, max_survivors=int(args.phase2_survivors)
            )
            # Skip already done overlays
            pairs = [
                (c, t)
                for c, t in pairs
                if f"{c.label}__{t}" not in exclude
                and f"turbo_highvol_minalloc__{c.label}__{t}" not in exclude
            ]
            logger.info("Phase2 overlay pairs: %d", len(pairs))
            for bi, batch_pairs in enumerate(_chunk(pairs, max(20, args.batch_size // 2))):
                if time.time() - t0 >= budget_s * 0.90:
                    stop_reason = f"time_budget_phase2_after_batch_{bi}"
                    break
                batch_dir = out_dir / f"phase2_batch_{bi:03d}"
                configs = overlay_to_mega_configs(
                    batch_pairs, strategy_overrides=overrides
                )
                logger.info("Phase2 batch %d n=%d", bi, len(configs))
                try:
                    summary = run_mega(
                        data_root=args.data_root,
                        ticker_file=args.ticker_file,
                        universe_limit=args.universe_limit,
                        first_oos=args.first_oos,
                        last_oos=args.last_oos,
                        grid=f"overnight_phase2_b{bi}",
                        out_dir=batch_dir,
                        configs=configs,
                    )
                except Exception:
                    logger.error("Phase2 batch %d failed:\n%s", bi, traceback.format_exc())
                    stop_reason = f"error_phase2_batch_{bi}"
                    break
                rows = summary.get("all_rows") or []
                for r in rows:
                    r["family"] = "overlay"
                phase2_rows = _merge_rows(phase2_rows, rows)
                _write_progress(
                    out_dir,
                    {
                        "phase": "phase2",
                        "batch": bi,
                        "n_phase1": len(phase1_rows),
                        "n_phase2": len(phase2_rows),
                        "elapsed_sec": round(time.time() - t0, 1),
                    },
                )
        else:
            stop_reason = "time_budget_skip_phase2"
            logger.warning("Skip Phase2 — insufficient time")

    # Phase 3: promotion top-K
    promo_summary = None
    all_rows = _rank_rows(_merge_rows(phase1_rows, phase2_rows))
    if not args.skip_promo and time.time() - t0 < budget_s * 0.97 and all_rows:
        top = all_rows[: int(args.promo_top)]
        # Collect config dirs from batches for those ids
        promo_src = out_dir / "phase3_promo_src" / "configs"
        promo_src.mkdir(parents=True, exist_ok=True)
        n_linked = 0
        for r in top:
            rid = str(r.get("id") or "")
            # find equity
            found = list(out_dir.glob(f"**/configs/{rid}/equity.csv"))
            if not found:
                continue
            dest = promo_src / rid
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("equity.csv", "trades.csv", "metrics.json"):
                src = found[0].parent / name
                if src.exists():
                    data = src.read_bytes()
                    (dest / name).write_bytes(data)
            n_linked += 1
        logger.info("Promotion candidates linked: %d", n_linked)
        if n_linked >= 2:
            try:
                from scripts.run_promotion_scorecard import main as promo_main

                promo_out = out_dir / "phase3_promo"
                rc = promo_main(
                    [
                        "--from-configs-dir",
                        str(promo_src),
                        "--style-name",
                        "turbo_highvol_minalloc__baseline",
                        "--full",
                        "--n-sims",
                        str(args.n_sims),
                        "--out",
                        str(promo_out),
                    ]
                )
                logger.info("Promotion exit code=%s", rc)
                psum = promo_out / "summary.json"
                if psum.exists():
                    promo_summary = json.loads(psum.read_text(encoding="utf-8"))
            except Exception:
                logger.error("Promotion failed:\n%s", traceback.format_exc())
                if stop_reason == "completed":
                    stop_reason = "error_promo"
        else:
            logger.warning("Not enough configs for promotion")
    elif not args.skip_promo:
        if stop_reason == "completed":
            stop_reason = "time_budget_skip_promo"

    elapsed = round(time.time() - t0, 1)
    meta = {
        "mode": args.mode,
        "hours": args.hours,
        "elapsed_sec": elapsed,
        "universe_limit": args.universe_limit,
        "first_oos": args.first_oos,
        "last_oos": args.last_oos,
        "stop_reason": stop_reason,
        "n_phase1": len(phase1_rows),
        "n_phase2": len(phase2_rows),
        "disclaimer": "Research only. Not financial advice.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                **meta,
                "top15": _rank_rows(_merge_rows(phase1_rows, phase2_rows))[:15],
                "all_phase1": phase1_rows,
                "all_phase2": phase2_rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _write_summary_md(
        out_dir,
        phase1_rows=phase1_rows,
        phase2_rows=phase2_rows,
        promo=promo_summary,
        meta=meta,
    )
    _write_progress(out_dir, {**meta, "phase": "done"})
    logger.info("DONE elapsed=%.0fs stop=%s n1=%d n2=%d", elapsed, stop_reason, len(phase1_rows), len(phase2_rows))
    print(f"\nOvernight complete → {out_dir / 'SUMMARY.md'}")
    print(f"stop_reason={stop_reason} elapsed_sec={elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
