"""Build PIT growth top-N universe (G-Q ≥10% Q EPS YoY, G-A ≥15% annual)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trad_research.growth_universe import (  # noqa: E402
    GrowthGateConfig,
    build_growth_universe,
    build_growth_universe_yearly,
)
from trad_research.universe import write_ticker_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=str, default="2023-12-31")
    ap.add_argument("--first-oos", type=int, default=0, help="If set with --last-oos, build yearly")
    ap.add_argument("--last-oos", type=int, default=0)
    ap.add_argument("--ticker-file", type=Path, default=ROOT / "good_tickers_filtrados.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--top-n", type=int, default=80)
    ap.add_argument("--limit-scan", type=int, default=0)
    ap.add_argument("--min-eps-q-yoy", type=float, default=0.10)
    ap.add_argument("--min-eps-ttm-yoy", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=ROOT / "universe_growth_top80.txt")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "redesign" / "growth_universe",
    )
    args = ap.parse_args()

    cfg = GrowthGateConfig(
        min_eps_q_yoy=float(args.min_eps_q_yoy),
        min_eps_ttm_yoy=float(args.min_eps_ttm_yoy),
        top_n=int(args.top_n),
    )
    limit = None if int(args.limit_scan) <= 0 else int(args.limit_scan)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if int(args.first_oos) and int(args.last_oos):
        years = list(range(int(args.first_oos), int(args.last_oos) + 1))
        by = build_growth_universe_yearly(
            Path(args.data_root),
            Path(args.ticker_file),
            years,
            cfg=cfg,
            out_dir=out_dir,
            limit_scan=limit,
        )
        meta = {str(y): names for y, names in by.items()}
        (out_dir / "membership_growth.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # also write latest year as default out
        last_y = max(by)
        write_ticker_file(Path(args.out), by[last_y])
        print(f"Yearly L0 written under {out_dir}; latest {last_y} n={len(by[last_y])} → {args.out}")
        for y, names in by.items():
            print(f"  {y}: n={len(names)} sample={names[:5]}")
        return 0

    top, rows = build_growth_universe(
        Path(args.data_root),
        Path(args.ticker_file),
        args.as_of,
        cfg=cfg,
        limit_scan=limit,
    )
    write_ticker_file(Path(args.out), top)
    n_pass = sum(1 for r in rows if r.pass_all)
    passers = [r.__dict__ for r in rows if r.pass_all]
    if passers:
        import pandas as pd

        pd.DataFrame(passers).sort_values("growth_rank_score", ascending=False).to_csv(
            out_dir / f"growth_passers_{args.as_of}.csv", index=False
        )
    summary = {
        "as_of": args.as_of,
        "n_scored": len(rows),
        "n_pass": n_pass,
        "top_n": len(top),
        "top": top,
        "cfg": cfg.__dict__,
    }
    (out_dir / f"growth_universe_{args.as_of}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"as_of={args.as_of} scored={len(rows)} pass={n_pass} top={len(top)}")
    print(f"Wrote {args.out}")
    print("Top 15:", top[:15])
    return 0 if top else 1


if __name__ == "__main__":
    raise SystemExit(main())
