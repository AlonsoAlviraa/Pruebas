"""Download deep EODHD fundamentals → data/{T}_fundamentals.csv (+ raw JSON cache).

Writes coverage report under reports/eodhd_fundamentals_coverage.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.data.eodhd_client import fetch_fundamentals_many  # noqa: E402
from trad_research.features import list_tickers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "good_tickers_filtrados.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--limit", type=int, default=0, help="0 = all tickers with OHLCV")
    ap.add_argument("--lag-days", type=int, default=45)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--raw-cache", type=Path, default=ROOT / "data" / "eodhd_cache" / "fund_raw")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    limit = None if int(args.limit) <= 0 else int(args.limit)
    tickers = list_tickers(Path(args.ticker_file), data_root, limit=limit)
    print(f"Downloading fundamentals for n={len(tickers)} → {data_root}", flush=True)

    panels, sources = fetch_fundamentals_many(
        tickers,
        lag_days=int(args.lag_days),
        cache_dir=data_root,
        raw_cache_dir=Path(args.raw_cache),
        sleep_s=float(args.sleep),
        force=bool(args.force),
    )

    # Also write standard names (fetch already writes {T}_fundamentals.csv in cache_dir)
    depths = []
    for t, df in panels.items():
        # ensure as_of strings ok for csv re-read
        n = len(df)
        try:
            amin = str(pd.to_datetime(df["as_of"]).min())
            amax = str(pd.to_datetime(df["as_of"]).max())
        except Exception:
            amin = amax = None
        depths.append({"ticker": t, "n_quarters": n, "min_as_of": amin, "max_as_of": amax})

    n20 = sum(1 for d in depths if d["n_quarters"] >= 20)
    n8 = sum(1 for d in depths if d["n_quarters"] >= 8)
    coverage = {
        "n_requested": len(tickers),
        "n_ok": len(panels),
        "n_quarters_ge_8": n8,
        "n_quarters_ge_20": n20,
        "source_counts": {},
        "depths_sample": depths[:20],
        "sources_error_sample": {k: v for k, v in list(sources.items())[:30] if "error" in v or "http" in v or "empty" in v},
    }
    for v in sources.values():
        coverage["source_counts"][v] = coverage["source_counts"].get(v, 0) + 1

    out = ROOT / "reports" / "eodhd_fundamentals_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(
        f"OK={len(panels)}/{len(tickers)}  ≥8Q={n8}  ≥20Q={n20}  report={out}",
        flush=True,
    )
    # Phase1 gate soft warning
    if n20 < 50:
        print("WARNING: few tickers with ≥20 quarters — OOS start may need shortening.", flush=True)
    return 0 if panels else 1


if __name__ == "__main__":
    raise SystemExit(main())
