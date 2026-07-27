#!/usr/bin/env python3
"""PR-OPT-N1: Score multi-window options matrix → promote / watch / kill.

Reads ``reports/paper_options_ta_matrix/latest/summary.json`` (or --in dir/file).
Writes SCORECARD.md + SCORECARD.json.

VIRTUAL research only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.options.scorecard import write_scorecard


def main() -> int:
    ap = argparse.ArgumentParser(description="Options matrix promote/watch/kill scorecard")
    ap.add_argument(
        "--in",
        dest="inp",
        default="reports/paper_options_ta_matrix/latest",
        help="Directory with summary.json or path to summary.json",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="SCORECARD.md path (default: <matrix root>/SCORECARD.md)",
    )
    ap.add_argument(
        "--config",
        default="paper_live/cloud/scorecard_options_config.json",
        help="Scorecard rules JSON",
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    if inp.is_dir():
        summary_path = inp / "summary.json"
        matrix_root = inp.parent if inp.name == "latest" else inp
    else:
        summary_path = inp
        matrix_root = inp.parent.parent if inp.parent.name == "latest" else inp.parent

    if not summary_path.is_file():
        print(f"FAIL: missing {summary_path}", file=sys.stderr)
        return 1

    out_md = Path(args.out) if args.out else matrix_root / "SCORECARD.md"
    # If user passes a .md under latest, still ok
    if out_md.suffix.lower() != ".md":
        out_md = out_md / "SCORECARD.md" if out_md.is_dir() or not out_md.suffix else out_md

    cfg = Path(args.config) if args.config else None
    payload = write_scorecard(
        summary_path,
        out_md=out_md,
        out_json=out_md.with_suffix(".json"),
        config_path=cfg if cfg and cfg.is_file() else None,
    )
    counts = payload.get("counts") or {}
    print(
        f"SCORECARD written: {out_md} | "
        f"PROMOTE={counts.get('PROMOTE_RESEARCH', 0)} "
        f"WATCH={counts.get('WATCH', 0)} "
        f"HOLD={counts.get('HOLD', 0)} "
        f"KILL={counts.get('KILL', 0)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
