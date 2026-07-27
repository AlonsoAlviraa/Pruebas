"""Download free SEC companyfacts fundamentals for a ticker panel."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.features import list_tickers  # noqa: E402
from trad_research.sec_fundamentals import (  # noqa: E402
    DEFAULT_UA,
    download_ticker_fundamentals,
    load_ticker_cik_map,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "universe_fund_panel_80.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "sec_cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--user-agent",
        default=None,
        help="SEC requires identifiable UA; default TRAD Research Bot",
    )
    ap.add_argument(
        "--coverage-out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "growth_sec_panel" / "sec_coverage.json",
    )
    args = ap.parse_args()

    ua = args.user_agent or DEFAULT_UA
    data_root = Path(args.data_root)
    limit = None if int(args.limit) <= 0 else int(args.limit)
    tickers = list_tickers(Path(args.ticker_file), data_root, limit=limit)
    print(f"SEC fund download n={len(tickers)} UA={ua!r}", flush=True)

    cik_map = load_ticker_cik_map(
        cache_path=Path(args.cache_dir) / "ticker_cik_map.json",
        user_agent=ua,
    )
    print(f"CIK map size={len(cik_map)}", flush=True)

    depths = []
    errors = {}
    ok = 0
    for i, t in enumerate(tickers, 1):
        success, msg, nq = download_ticker_fundamentals(
            t,
            cik_map=cik_map,
            data_root=data_root,
            cache_dir=Path(args.cache_dir),
            user_agent=ua,
            force=bool(args.force),
            sleep_s=float(args.sleep),
        )
        if success:
            ok += 1
            depths.append({"ticker": t, "n_quarters": nq, "status": msg})
        else:
            errors[t] = msg
            depths.append({"ticker": t, "n_quarters": 0, "status": msg})
        if i % 10 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)} ok={ok}", flush=True)

    n20 = sum(1 for d in depths if d["n_quarters"] >= 20)
    n8 = sum(1 for d in depths if d["n_quarters"] >= 8)
    cov = {
        "provider": "sec_companyfacts",
        "n_requested": len(tickers),
        "n_ok": ok,
        "n_quarters_ge_8": n8,
        "n_quarters_ge_20": n20,
        "pct_ge_20": round(100.0 * n20 / max(len(tickers), 1), 1),
        "gate_70pct_20q": n20 >= 0.7 * max(len(tickers), 1),
        "depths": depths,
        "errors": errors,
        "disclaimer": "Research only. filed date = available_at for PIT.",
    }
    out = Path(args.coverage_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cov, indent=2), encoding="utf-8")
    print(
        f"OK={ok}/{len(tickers)} ≥8Q={n8} ≥20Q={n20} gate70%={cov['gate_70pct_20q']} → {out}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
