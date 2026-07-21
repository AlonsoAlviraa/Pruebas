#!/usr/bin/env python3
"""Generate paper live daily/weekly digests + HTML from a ledger run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.ledger import PaperLedger
from paper_live.reports.pipeline import generate_reports_for_run


def _latest_run_id(db_path: Path) -> str:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        raise SystemExit(f"No runs in {db_path}")
    return str(row[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper daily/weekly digests (virtual capital)")
    ap.add_argument(
        "--ledger-root",
        default="paper_live/ledger_data",
        help="Directory containing paper_year.db",
    )
    ap.add_argument("--run-id", default=None, help="Run id (default: latest)")
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir (default: ledger-root/reports or reports/paper_year/<run>)",
    )
    ap.add_argument("--day", default=None, help="Single day YYYY-MM-DD (optional)")
    ap.add_argument("--week-of", default=None, help="Any day in the ISO week")
    args = ap.parse_args()

    root = Path(args.ledger_root)
    db = root / "paper_year.db"
    if not db.is_file():
        print(json.dumps({"error": f"missing {db}", "mode": "paper"}))
        return 2

    run_id = args.run_id or _latest_run_id(db)
    ledger = PaperLedger.open_run(root, run_id)
    out = Path(args.out) if args.out else (root / "reports")
    days = [args.day] if args.day else None
    bundle = generate_reports_for_run(
        ledger,
        out,
        days=days,
        week_of=args.week_of or args.day,
        write_html=True,
    )
    ledger.close()
    print(json.dumps(bundle.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
