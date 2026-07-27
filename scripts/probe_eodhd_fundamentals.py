"""Probe EODHD fundamentals for a few tickers (schema + depth)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.data.eodhd_client import fetch_fundamentals, get_token  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,NVDA,MSFT")
    ap.add_argument("--lag-days", type=int, default=45)
    args = ap.parse_args()
    get_token()  # fail fast
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    report = []
    for t in tickers:
        try:
            df = fetch_fundamentals(t, lag_days=int(args.lag_days))
            row = {
                "ticker": t,
                "ok": not df.empty,
                "n_quarters": int(len(df)),
                "min_as_of": str(df["as_of"].min()) if not df.empty else None,
                "max_as_of": str(df["as_of"].max()) if not df.empty else None,
                "eps_non_null": int(df["eps"].notna().sum()) if not df.empty else 0,
                "rev_non_null": int(df["revenue"].notna().sum()) if not df.empty else 0,
            }
            print(
                f"{t}: quarters={row['n_quarters']} "
                f"as_of=[{row['min_as_of']} .. {row['max_as_of']}] "
                f"eps={row['eps_non_null']} rev={row['rev_non_null']}"
            )
            report.append(row)
        except Exception as e:
            print(f"{t}: ERROR {type(e).__name__}: {e}")
            report.append({"ticker": t, "ok": False, "error": str(e)})
    out = ROOT / "reports" / "eodhd_fundamentals_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if any(r.get("ok") for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
