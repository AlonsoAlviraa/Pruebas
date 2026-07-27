#!/usr/bin/env python3
"""Week plan orchestrator: Phase A overlays → B promotion → C risk A/B → D freeze decision.

Research only. Equity long-only. No OPRA claims. No guaranteed edge.

Usage (PowerShell, repo root)::

  $env:PYTHONPATH = (Get-Location).Path
  python scripts/run_week_plan_study.py --smoke
  python scripts/run_week_plan_study.py --full --universe-limit 0 --first-oos 2018 --last-oos 2025

Design: docs/design/2026-07-23_week_overlay_risk_promotion.md
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.risk_levers import (
    WEEK_PRIMARY_LEVER_ID,
    decide_freeze_path,
    get_lever,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week_plan")

DEFAULT_OUT = ROOT / "reports" / "redesign" / "week_plan_2026-07-23"
CONTROL = "turbo_highvol_minalloc"


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return default


def _run_py(args: List[str]) -> int:
    cmd = [sys.executable, *args]
    logger.info("RUN %s", " ".join(cmd))
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT) + (
        (";" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return int(r.returncode)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_freeze_shadow(
    out_dir: Path,
    decision: Dict[str, Any],
) -> Optional[Path]:
    """Write Phase D decision + optional *report-side* candidate (never live freeze).

    - Always writes ``phase_d_freeze/DECISION.md``.
    - Writes ``strategy_freeze_candidate.json`` only when
      ``decision['write_shadow_candidate']`` / action ``register_shadow``.
    - Refuses any resolved path under ``paper_live/config/``.
    """
    freeze_dir = (Path(out_dir) / "phase_d_freeze").resolve()
    paper_cfg = (ROOT / "paper_live" / "config").resolve()
    try:
        freeze_dir.relative_to(paper_cfg)
        raise RuntimeError(
            f"Refusing freeze write under paper_live/config: {freeze_dir}"
        )
    except ValueError:
        pass  # not under paper_live/config — OK
    freeze_dir.mkdir(parents=True, exist_ok=True)

    note = freeze_dir / "DECISION.md"
    lines = [
        "# Phase D freeze decision",
        "",
        f"**action:** `{decision.get('action')}`",
        f"**control:** `{decision.get('strategy_id')}`",
        f"**shadow_enabled:** {decision.get('shadow_enabled')}",
        f"**shadow:** `{decision.get('shadow_strategy_id')}`",
        f"**write_shadow_candidate:** {bool(decision.get('write_shadow_candidate'))}",
        "",
        f"**reason:** {decision.get('reason')}",
        "",
        "Live paper freeze path remains `paper_live/config/strategy_freeze.json` "
        f"(`{CONTROL}`) unless a **human** copies a report candidate after ADVANCE.",
        "Candidates are report-only under `phase_d_freeze/` — never auto-deployed.",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    note.write_text("\n".join(lines), encoding="utf-8")

    write_cand = bool(
        decision.get("write_shadow_candidate")
        or decision.get("action") == "register_shadow"
    )
    if not write_cand:
        return None

    path = freeze_dir / "strategy_freeze_candidate.json"
    # Double-check isolation
    try:
        path.resolve().relative_to(paper_cfg)
        raise RuntimeError(f"Refusing candidate path under paper_live/config: {path}")
    except ValueError:
        pass
    payload = {
        "version": "strategy-freeze-candidate-v1",
        "mode": "paper",
        "strategy_id": decision.get("strategy_id") or CONTROL,
        "description": (
            "Week-plan candidate freeze (report-only). Virtual capital only. "
            "Human must copy after review — never auto-overwrites live freeze."
        ),
        "capital0": 100_000.0,
        "currency": "USD",
        "long_only": True,
        "max_leverage": 1.0,
        "knobs": {
            "note": "Inherits research baseline minalloc unless shadow replaces control",
            "base_strategy": CONTROL,
            "shadow_strategy_id": decision.get("shadow_strategy_id"),
            "shadow_enabled": bool(decision.get("shadow_enabled")),
        },
        "risk_paper": {
            "max_portfolio_dd": 0.18,
            "dd_soft_scale": 0.5,
            "source": "paper_live/config/strategy_freeze.json control risk (reference only)",
        },
        "shadow_strategy_id": decision.get("shadow_strategy_id"),
        "shadow_enabled": bool(decision.get("shadow_enabled")),
        "notes": list(decision.get("notes") or [])
        + [
            decision.get("reason") or "",
            "Generated by scripts/run_week_plan_study.py Phase D.",
            "Research only. Not financial advice.",
        ],
        "decision": decision,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_master_summary(
    out_dir: Path,
    *,
    smoke: bool,
    phase_a: Dict[str, Any],
    phase_b: Dict[str, Any],
    phase_c: Dict[str, Any],
    decision: Dict[str, Any],
    elapsed: float,
) -> None:
    rows_a = phase_a.get("all_rows") or phase_a.get("top_by_composite") or []
    adv = phase_b.get("advance") or []
    hold = phase_b.get("hold") or []
    kill = phase_b.get("kill") or []
    risk_rows = phase_c.get("all_rows") or []

    lines = [
        "# Week plan 2026-07-23 — SUMMARY",
        "",
        "> **Research only.** Not financial advice. Past backtests ≠ future results.",
        "> Equity long-only. No OPRA / short-vol claims.",
        "",
        f"- **smoke:** {smoke}",
        f"- **elapsed_sec:** {elapsed:.1f}",
        f"- **regime:** `{phase_a.get('regime', 'strict_dual_golden')}`",
        f"- **OOS:** {phase_a.get('first_oos')}–{phase_a.get('last_oos')}",
        f"- **universe_limit:** {phase_a.get('universe_limit')} "
        f"(loaded={phase_a.get('universe_n_loaded', '?')})",
        f"- **control:** `{CONTROL}`",
        "",
        "## Decision (Phase D)",
        "",
        f"**action:** `{decision.get('action')}`  ",
        f"**reason:** {decision.get('reason')}  ",
        f"**paper freeze:** remains `{CONTROL}` "
        + (
            f"with **shadow** `{decision.get('shadow_strategy_id')}` (candidate only)"
            if decision.get("shadow_enabled")
            else "(no shadow promotion)"
        ),
        "",
        "## Phase A — curated overlays (highvol80 spirit)",
        "",
        "| id | CAGR | WR | Sharpe | MDD | n_trades | excess vs SPY | composite |",
        "|----|------|-----|--------|-----|----------|---------------|-----------|",
    ]
    for r in rows_a:
        if r.get("error"):
            lines.append(f"| `{r.get('id')}` | ERR | | | | | | |")
            continue
        lines.append(
            f"| `{r.get('id')}` | {_safe_float(r.get('cagr'))*100:.1f}% | "
            f"{_safe_float(r.get('win_rate'))*100:.1f}% | {_safe_float(r.get('sharpe')):.2f} | "
            f"{_safe_float(r.get('max_drawdown'))*100:.1f}% | {r.get('n_trades')} | "
            f"{_safe_float(r.get('excess_total_vs_spy'))*100:.1f}% | "
            f"{_safe_float(r.get('composite')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Phase B — promotion funnel",
            "",
            f"- **ADVANCE:** {adv or '*(none)*'}",
            f"- **HOLD:** {hold or '*(none)*'}",
            f"- **KILL:** {kill or '*(none)*'}",
            "",
            "0 ADVANCE is a valid scientific outcome (MDD/MC honesty).",
            "",
            "## Phase C — risk A/B (primary lever: DD circuit 25%)",
            "",
            f"Primary lever id: `{WEEK_PRIMARY_LEVER_ID}` — "
            f"{get_lever(WEEK_PRIMARY_LEVER_ID).description}",
            "",
            "| id | CAGR | Sharpe | MDD | n_trades |",
            "|----|------|--------|-----|----------|",
        ]
    )
    for r in risk_rows:
        if r.get("error"):
            lines.append(f"| `{r.get('id')}` | ERR | | | |")
            continue
        lines.append(
            f"| `{r.get('id')}` | {_safe_float(r.get('cagr'))*100:.1f}% | "
            f"{_safe_float(r.get('sharpe')):.2f} | "
            f"{_safe_float(r.get('max_drawdown'))*100:.1f}% | {r.get('n_trades')} |"
        )
    # simple A/B narrative
    by_id = {r.get("id"): r for r in risk_rows if not r.get("error")}
    ctrl = by_id.get(f"{CONTROL}__baseline") or by_id.get(
        f"turbo_highvol_minalloc__baseline"
    )
    treat = by_id.get(f"{CONTROL}__{WEEK_PRIMARY_LEVER_ID}")
    if ctrl and treat:
        mdd_c = _safe_float(ctrl.get("max_drawdown"))
        mdd_t = _safe_float(treat.get("max_drawdown"))
        cagr_c = _safe_float(ctrl.get("cagr"))
        cagr_t = _safe_float(treat.get("cagr"))
        lines.extend(
            [
                "",
                "### Risk lever read",
                "",
                f"- Control MDD **{mdd_c*100:.1f}%** vs treatment **{mdd_t*100:.1f}%** "
                f"(Δ { (mdd_t-mdd_c)*100:.1f} pp).",
                f"- Control CAGR **{cagr_c*100:.1f}%** vs treatment **{cagr_t*100:.1f}%**.",
                "- Prefer treatment only if MDD improves **without** collapsing residual/Sharpe "
                "and promotion still fails honestly on residual gates when applicable.",
            ]
        )
    lines.extend(
        [
            "",
            "## Kill / hold criteria (honest)",
            "",
            "- No ADVANCE → **do not** change paper freeze knobs.",
            "- Deep MDD vs gate (−50%) or MC p5 fail → KILL/HOLD, not ADVANCE.",
            "- Geo retrain and OPRA out of scope.",
            "- n=smoke is wiring evidence only; full highvol80 OOS required for claims.",
            "",
            "## Commands",
            "",
            "```powershell",
            "$env:PYTHONPATH = (Get-Location).Path",
            "python scripts/run_week_plan_study.py --smoke",
            "python scripts/run_week_plan_study.py --full --universe-limit 0 --first-oos 2018 --last-oos 2025",
            "```",
            "",
            "Design: `docs/design/2026-07-23_week_overlay_risk_promotion.md`",
            "",
            "Research only. Not financial advice.",
            "",
        ]
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    master = {
        "version": "week_plan_2026-07-23",
        "disclaimer": "Research only. Not financial advice.",
        "smoke": smoke,
        "elapsed_sec": elapsed,
        "control": CONTROL,
        "phase_a": {
            "grid": phase_a.get("grid"),
            "first_oos": phase_a.get("first_oos"),
            "last_oos": phase_a.get("last_oos"),
            "universe_limit": phase_a.get("universe_limit"),
            "universe_n_loaded": phase_a.get("universe_n_loaded"),
            "n_configs": phase_a.get("n_configs"),
            "top_by_composite": (phase_a.get("top_by_composite") or [])[:5],
        },
        "phase_b": {
            "advance": adv,
            "hold": hold,
            "kill": kill,
            "n_sims": phase_b.get("n_sims"),
        },
        "phase_c": {
            "primary_lever": WEEK_PRIMARY_LEVER_ID,
            "rows": [
                {
                    k: r.get(k)
                    for k in (
                        "id",
                        "cagr",
                        "sharpe",
                        "max_drawdown",
                        "n_trades",
                        "win_rate",
                        "total_return",
                    )
                }
                for r in risk_rows
            ],
        },
        "phase_d": decision,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(master, indent=2, default=str), encoding="utf-8"
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Week overlay + risk + promotion plan")
    ap.add_argument("--smoke", action="store_true", help="Fast CI path (small univ / 1y)")
    ap.add_argument("--full", action="store_true", help="Full OOS path (heavy)")
    ap.add_argument("--universe-limit", type=int, default=None, help="0=full highvol80")
    ap.add_argument("--first-oos", type=int, default=None)
    ap.add_argument("--last-oos", type=int, default=None)
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "universe_highvol80.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-sims", type=int, default=None)
    ap.add_argument("--skip-a", action="store_true", help="Reuse existing phase_a/")
    ap.add_argument("--skip-c", action="store_true", help="Skip risk A/B")
    ap.add_argument(
        "--phases",
        type=str,
        default="A,B,C,D",
        help="Comma list of phases to run (default A,B,C,D)",
    )
    args = ap.parse_args(argv)

    if not args.smoke and not args.full:
        print(
            "error: require explicit --smoke or --full "
            "(no accidental full highvol80 OOS default)",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if args.smoke and args.full:
        print(
            "error: pass only one of --smoke or --full",
            file=sys.stderr,
            flush=True,
        )
        return 2

    smoke = bool(args.smoke) and not bool(args.full)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = {p.strip().upper() for p in (args.phases or "A,B,C,D").split(",") if p.strip()}
    t0 = time.time()

    phase_a_dir = out_dir / "phase_a"
    phase_b_dir = out_dir / "phase_b"
    phase_c_dir = out_dir / "phase_c"
    phase_a: Dict[str, Any] = {}
    phase_b: Dict[str, Any] = {}
    phase_c: Dict[str, Any] = {}

    # --- Phase A ---
    if "A" in phases and not args.skip_a:
        phase_a_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(ROOT / "scripts" / "run_crash_entry_mega_study.py"),
            "--grid",
            "week",
            "--out",
            str(phase_a_dir),
            "--ticker-file",
            str(args.ticker_file),
            "--data-root",
            str(args.data_root),
        ]
        if smoke:
            cmd.append("--smoke")
            # keep week grid; small default univ unless overridden
            univ = 12 if args.universe_limit is None else args.universe_limit
            cmd.extend(["--universe-limit", str(univ)])
            if args.first_oos is not None:
                cmd.extend(["--first-oos", str(args.first_oos)])
            if args.last_oos is not None:
                cmd.extend(["--last-oos", str(args.last_oos)])
        else:
            univ = 0 if args.universe_limit is None else args.universe_limit
            cmd.extend(["--universe-limit", str(univ)])
            cmd.extend(["--first-oos", str(args.first_oos or 2018)])
            cmd.extend(["--last-oos", str(args.last_oos or 2025)])
        rc = _run_py(cmd)
        if rc != 0:
            logger.error("Phase A failed rc=%s", rc)
            return rc
    phase_a = _load_json(phase_a_dir / "summary.json")

    # --- Phase B ---
    if "B" in phases:
        configs = phase_a_dir / "configs"
        if not configs.is_dir():
            logger.error("Phase B needs %s (run Phase A first)", configs)
            return 2
        n_sims = args.n_sims
        if n_sims is None:
            n_sims = 100 if smoke else 2000
        cmd = [
            str(ROOT / "scripts" / "run_promotion_scorecard.py"),
            "--from-configs-dir",
            str(configs),
            "--style-name",
            "turbo_highvol_minalloc__baseline",
            "--product",
            "STYLE-US",
            "--out",
            str(phase_b_dir),
            "--n-sims",
            str(n_sims),
        ]
        if smoke:
            cmd.append("--smoke")
        else:
            cmd.append("--full")
        rc = _run_py(cmd)
        if rc != 0:
            logger.error("Phase B failed rc=%s", rc)
            return rc
    phase_b = _load_json(phase_b_dir / "summary.json")

    # --- Phase C ---
    if "C" in phases and not args.skip_c:
        phase_c_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(ROOT / "scripts" / "run_crash_entry_mega_study.py"),
            "--grid",
            "week_risk",
            "--out",
            str(phase_c_dir),
            "--ticker-file",
            str(args.ticker_file),
            "--data-root",
            str(args.data_root),
        ]
        if smoke:
            cmd.append("--smoke")
            univ = 12 if args.universe_limit is None else args.universe_limit
            cmd.extend(["--universe-limit", str(univ)])
            if args.first_oos is not None:
                cmd.extend(["--first-oos", str(args.first_oos)])
            if args.last_oos is not None:
                cmd.extend(["--last-oos", str(args.last_oos)])
        else:
            univ = 0 if args.universe_limit is None else args.universe_limit
            cmd.extend(["--universe-limit", str(univ)])
            cmd.extend(["--first-oos", str(args.first_oos or 2018)])
            cmd.extend(["--last-oos", str(args.last_oos or 2025)])
        rc = _run_py(cmd)
        if rc != 0:
            logger.error("Phase C failed rc=%s", rc)
            return rc
    phase_c = _load_json(phase_c_dir / "summary.json")

    # --- Phase D ---
    decision = decide_freeze_path(
        advance_names=list(phase_b.get("advance") or []),
        control_strategy_id=CONTROL,
    )
    if "D" in phases:
        _write_freeze_shadow(out_dir, decision)

    elapsed = time.time() - t0
    _write_master_summary(
        out_dir,
        smoke=smoke,
        phase_a=phase_a,
        phase_b=phase_b,
        phase_c=phase_c,
        decision=decision,
        elapsed=elapsed,
    )
    logger.info(
        "DONE week plan action=%s ADVANCE=%s → %s",
        decision.get("action"),
        phase_b.get("advance"),
        out_dir / "SUMMARY.md",
    )
    print(
        f"WEEK_PLAN_DONE action={decision.get('action')} "
        f"advance={phase_b.get('advance') or []} out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
