"""Yearly L0: highvol pool ∩ growth gates (PIT). Used by vol_fund mega loop."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from trad_research.growth_universe import (
    GrowthGateConfig,
    rank_growth_passers,
    score_growth_ticker,
)
from trad_research.universe import (
    build_scored_universe,
    select_high_vol,
    write_ticker_file,
)


def highvol_pool_asof(
    data_root: Path,
    ticker_file: Path,
    as_of: str,
    *,
    n: int = 200,
    limit_scan: Optional[int] = None,
) -> List[str]:
    rows = build_scored_universe(
        data_root,
        ticker_file,
        as_of=as_of,
        limit_scan=limit_scan,
        min_price=5.0,
        min_dollar_vol=1_000_000.0,
    )
    return select_high_vol(rows, n=n)


def growth_l0_from_pool(
    pool: Sequence[str],
    data_root: Path,
    as_of: str,
    *,
    cfg: Optional[GrowthGateConfig] = None,
    top_k: int = 40,
    fund_root: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Apply G-Q/G-A on pool; return top_k by growth rank + diagnostics."""
    cfg = cfg or GrowthGateConfig(top_n=top_k)
    cfg = GrowthGateConfig(**{**cfg.__dict__, "top_n": int(top_k)})
    rows = []
    for t in pool:
        r = score_growth_ticker(
            t, Path(data_root), as_of, cfg=cfg, fund_root=fund_root
        )
        if r is not None:
            rows.append(r)
    ranked = rank_growth_passers(rows, cfg=cfg)
    top = [r.ticker for r in ranked[: int(top_k)]]
    diag = {
        "pool": len(pool),
        "scored": len(rows),
        "pass_all": sum(1 for r in rows if r.pass_all),
        "l0": len(top),
    }
    return top, diag


def write_year_l0(
    out_dir: Path,
    year: int,
    tickers: Sequence[str],
    *,
    tag: str = "l0",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{tag}_{year}.txt"
    write_ticker_file(p, tickers)
    return p
