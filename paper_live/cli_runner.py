#!/usr/bin/env python3
"""Paper runner CLI: replay with kill switch, or live stub (TRAD_PAPER_ONLY=1)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.runner import build_runner


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper RTH runner (virtual capital only)")
    ap.add_argument("--mode", choices=("replay", "live-stub"), default="replay")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--day", default=None, help="Single day for live-stub")
    ap.add_argument("--tickers", default="AAA,BBB,QQQ")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--ledger-root", default="paper_live/ledger_data_runner")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    runner = build_runner(
        ledger_root=args.ledger_root,
        data_root=args.data_root,
        tickers=tickers,
        synthetic=args.synthetic or args.mode == "replay" and not Path(args.data_root).joinpath(
            f"{tickers[0]}_history.csv"
        ).is_file()
        if tickers
        else args.synthetic,
    )
    # Prefer synthetic when requested
    if args.synthetic:
        from paper_live.datafeed.replay import DailyReplayFeed

        runner.feed = DailyReplayFeed.from_synthetic(tickers, n_days=400, seed=42)
        runner.session = None

    if args.mode == "replay":
        if not args.date_from or not args.date_to:
            ap.error("replay requires --from and --to")
        # if QQQ missing regime off inside session after ensure
        out = runner.run_replay(args.date_from, args.date_to)
    else:
        if os.environ.get("TRAD_PAPER_ONLY") != "1":
            print(
                json.dumps(
                    {
                        "error": "Set TRAD_PAPER_ONLY=1 for live-stub mode",
                        "mode": "paper",
                    }
                )
            )
            return 2
        day = args.day or args.date_from
        if not day:
            ap.error("live-stub requires --day or --from")
        out = runner.run_live_day_stub(day, require_env=True)

    if runner.ledger is not None:
        runner.ledger.write_snapshot("runner_end")
        runner.ledger.close()
    print(json.dumps(out.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
