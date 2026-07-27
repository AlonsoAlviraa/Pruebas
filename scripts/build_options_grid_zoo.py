#!/usr/bin/env python3
"""Build combinatorial options grid zoo (thousands, no levered single-name)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.options.grid_zoo import write_grid_zoo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_live/cloud/zoo_options_grid.json")
    ap.add_argument("--max", type=int, default=2500)
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()
    zoo = write_grid_zoo(
        Path(args.out),
        max_strategies=int(args.max),
        include_names=not args.index_only,
    )
    print(f"Wrote {args.out} n={zoo['n_strategies']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
