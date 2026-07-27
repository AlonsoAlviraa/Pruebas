"""Build causal high-vol panel (top-N by realized vol) — reuses universe.score_ticker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.universe import (  # noqa: E402
    build_scored_universe,
    select_high_vol,
    write_ticker_file,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--as-of", default="2017-12-31")
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "good_tickers_filtrados.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-dollar-vol", type=float, default=1_000_000.0)
    ap.add_argument("--limit-scan", type=int, default=0, help="0 = all in ticker file")
    ap.add_argument("--out", type=Path, default=ROOT / "universe_highvol200.txt")
    ap.add_argument(
        "--meta-out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "vol_fund_mega" / "highvol_panel_meta.json",
    )
    args = ap.parse_args()

    limit = None if int(args.limit_scan) <= 0 else int(args.limit_scan)
    print(f"Scoring universe as-of {args.as_of} …", flush=True)
    rows = build_scored_universe(
        Path(args.data_root),
        Path(args.ticker_file),
        as_of=args.as_of,
        limit_scan=limit,
        min_price=float(args.min_price),
        min_dollar_vol=float(args.min_dollar_vol),
    )
    top = select_high_vol(rows, n=int(args.n))
    write_ticker_file(Path(args.out), top)

    # vol stats
    by_t = {r.ticker: r for r in rows}
    vols = [float(by_t[t].vol) for t in top if t in by_t and by_t[t].vol == by_t[t].vol]
    meta = {
        "as_of": args.as_of,
        "n_requested": int(args.n),
        "n_scored": len(rows),
        "n_panel": len(top),
        "vol_min": min(vols) if vols else None,
        "vol_median": sorted(vols)[len(vols) // 2] if vols else None,
        "vol_max": max(vols) if vols else None,
        "tickers": top,
        "method": "realized_vol_252d_select_high_vol",
        "disclaimer": "Research only. Causal as-of panel; not growth-ranked.",
    }
    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Highvol panel n={len(top)} → {args.out}")
    print(f"vol median={meta['vol_median']:.3f} min={meta['vol_min']:.3f} max={meta['vol_max']:.3f}")
    print("Top 15:", top[:15])
    return 0 if top else 1


if __name__ == "__main__":
    raise SystemExit(main())
