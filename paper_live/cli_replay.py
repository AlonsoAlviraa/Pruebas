#!/usr/bin/env python3
"""Replay paper session over historical daily bars (virtual capital only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.datafeed.replay import DailyReplayFeed
from paper_live.freeze import load_freeze
from paper_live.ledger import PaperLedger
from paper_live.replay_session import ReplaySession


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper daily replay (no real money)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--tickers", default="AAPL,MSFT,SPY,QQQ", help="Comma-separated")
    ap.add_argument("--ledger-root", default="paper_live/ledger_data_replay")
    ap.add_argument("--synthetic", action="store_true", help="Ignore CSVs; use synthetic panels")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()

    freeze = load_freeze()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.synthetic:
        feed = DailyReplayFeed.from_synthetic(tickers, n_days=400, start="2019-01-02", seed=7)
    else:
        feed = DailyReplayFeed.from_data_root(args.data_root, tickers)

    ledger = None
    if not args.no_ledger:
        ledger = PaperLedger.create_run(
            Path(args.ledger_root),
            freeze,
            meta={"cli": "cli_replay", "from": args.date_from, "to": args.date_to},
        )

    session = ReplaySession(feed, freeze, ledger=ledger)
    # Prefer requiring regime only if QQQ present
    if "QQQ" not in feed.tickers and "SPY" not in feed.tickers:
        session.pipeline.require_regime = False

    result = session.run(args.date_from, args.date_to)
    if ledger is not None:
        ledger.write_snapshot("replay_end")
        ledger.close()

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
