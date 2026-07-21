#!/usr/bin/env python3
"""Initialize a paper live run (LIV-01 freeze + LIV-02 ledger). Virtual capital only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.freeze import load_freeze
from paper_live.ledger import PaperLedger


def main() -> int:
    ap = argparse.ArgumentParser(description="Init paper live year run (no real money)")
    ap.add_argument(
        "--ledger-root",
        default="paper_live/ledger_data",
        help="Directory for SQLite + JSONL audit + snapshots",
    )
    ap.add_argument(
        "--config-dir",
        default=None,
        help="Override freeze config dir (default paper_live/config)",
    )
    ap.add_argument("--run-id", default=None, help="Optional fixed run_id")
    args = ap.parse_args()

    freeze = load_freeze(args.config_dir)
    ledger = PaperLedger.create_run(
        Path(args.ledger_root),
        freeze,
        run_id=args.run_id,
        meta={"cli": "cli_init", "capital_label": "VIRTUAL"},
    )
    snap = ledger.write_snapshot("init")
    info = {
        "run_id": ledger.run_id,
        "config_hash": ledger.config_hash,
        "strategy_id": ledger.strategy_id,
        "mode": "paper",
        "capital0": freeze.strategy.capital0,
        "ledger_root": str(Path(args.ledger_root).resolve()),
        "db": str(ledger.db_path),
        "snapshot": str(snap),
        "commission_sample_100sh_50px": freeze.cost.estimate_commission(100, 50.0),
    }
    ledger.close()
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
