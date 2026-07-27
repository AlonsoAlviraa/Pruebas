"""Build liquid US panel (50/80/100) as-of a cutoff for free SEC fund research."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.features import list_tickers, load_history  # noqa: E402
from trad_research.universe import write_ticker_file  # noqa: E402


def _score_liq(ticker: str, data_root: Path, as_of: pd.Timestamp) -> Optional[dict]:
    hist = load_history(ticker, data_root)
    if hist.empty or "date" not in hist.columns:
        return None
    h = hist[hist["date"] <= as_of]
    if len(h) < 400:
        return None
    close = h["close"].astype(float)
    last = float(close.iloc[-1])
    if last < 5.0:
        return None
    if "volume" not in h.columns:
        return None
    adv = float((h["close"].astype(float) * h["volume"].astype(float)).tail(60).mean())
    if not np.isfinite(adv) or adv < 2_000_000:
        return None
    return {"ticker": ticker, "last_close": last, "adv": adv, "n_bars": len(h)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--as-of", default="2017-12-31")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--sources",
        default="universe_highvol80.txt,good_tickers_filtrados.txt",
        help="Comma-separated ticker files (order = priority)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "universe_fund_panel_80.txt")
    ap.add_argument(
        "--meta-out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "growth_sec_panel" / "panel_meta.json",
    )
    args = ap.parse_args()

    as_of = pd.Timestamp(args.as_of, tz="UTC")
    data_root = Path(args.data_root)
    seen = set()
    candidates: List[str] = []
    for sf in str(args.sources).split(","):
        sf = sf.strip()
        if not sf:
            continue
        p = ROOT / sf if not Path(sf).is_file() else Path(sf)
        if not p.is_file():
            continue
        for t in list_tickers(p, data_root, limit=None):
            if t not in seen:
                seen.add(t)
                candidates.append(t)

    rows = []
    for i, t in enumerate(candidates):
        if (i + 1) % 100 == 0:
            print(f"  scoring {i+1}/{len(candidates)}", flush=True)
        r = _score_liq(t, data_root, as_of)
        if r:
            rows.append(r)

    rows.sort(key=lambda x: -float(x["adv"]))
    top = rows[: int(args.n)]
    tickers = [r["ticker"] for r in top]
    write_ticker_file(Path(args.out), tickers)

    meta = {
        "as_of": str(as_of),
        "n_requested": int(args.n),
        "n_scored_ok": len(rows),
        "n_panel": len(tickers),
        "min_adv": float(top[-1]["adv"]) if top else None,
        "median_adv": float(np.median([r["adv"] for r in top])) if top else None,
        "tickers": tickers,
        "disclaimer": "Research panel by liquidity as-of cutoff — not growth-ranked.",
    }
    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Panel n={len(tickers)} → {args.out}")
    print(f"Meta → {meta_path}")
    print("Sample:", tickers[:15])
    return 0 if tickers else 1


if __name__ == "__main__":
    raise SystemExit(main())
