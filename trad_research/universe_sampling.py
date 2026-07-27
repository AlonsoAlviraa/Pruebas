"""Universe Monte Carlo sampling for strategy generalization studies.

Research only. Seeds are pre-registered; draws never use future returns.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Market codes for seed namespacing (plan SSOT)
MARKET_SEED_CODE: Dict[str, int] = {
    "US": 0,
    "ES": 1,
    "DE": 2,
    "FR": 3,
    "UK": 4,
    "EU": 5,
}

# Relative paths from repo/dataset root (resolved via market_specs())
_MARKET_REL: Dict[str, Dict[str, Any]] = {
    "US": {
        "data_root": "data",
        "universe_file": "universe_longhist2010_pass.txt",
        "preferred_index": ("SPY", "QQQ"),
        "role": "screen",
    },
    "ES": {
        "data_root": "data_es",
        "universe_file": "spain_wf_universe.txt",
        "preferred_index": ("IBEX",),
        "role": "transfer",
    },
    "DE": {
        "data_root": "data_de",
        "universe_file": "germany_wf_universe.txt",
        "preferred_index": ("DAX",),
        "role": "transfer",
    },
    "FR": {
        "data_root": "data_fr",
        "universe_file": "france_wf_universe.txt",
        "preferred_index": ("CAC",),
        "role": "transfer",
    },
    "UK": {
        "data_root": "data_uk",
        "universe_file": "uk_wf_universe.txt",
        "preferred_index": ("FTSE",),
        "role": "transfer",
    },
}


def market_specs(repo_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Resolve market paths against repo/dataset root (Kaggle-friendly)."""
    root = Path(repo_root) if repo_root is not None else ROOT
    out: Dict[str, Dict[str, Any]] = {}
    for mid, rel in _MARKET_REL.items():
        out[mid] = {
            "data_root": root / str(rel["data_root"]),
            "universe_file": root / str(rel["universe_file"]),
            "preferred_index": tuple(rel["preferred_index"]),
            "role": rel["role"],
        }
    return out


# Back-compat default (local repo layout)
MARKET_SPECS: Dict[str, Dict[str, Any]] = market_specs(ROOT)


@dataclass(frozen=True)
class DrawSpec:
    """One pre-registered universe draw."""

    market: str
    series: str  # R50, R80, FULL, PREFIX, B50, ...
    seed: int
    draw_size: int
    strategy: str
    universe_limit: int
    ticker_file: Path
    data_root: Path
    preferred_index: Tuple[str, ...]
    screen_first: int
    screen_last: int
    confirm_first: int
    confirm_last: int
    run_screen: bool  # False for confirm-only markets (UK decade-poor)
    gate_trades: int
    arm_id: str


def read_tickers(path: Path) -> List[str]:
    if not path.is_file():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            out.append(t)
    return out


def write_tickers(path: Path, tickers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tickers) + ("\n" if tickers else ""), encoding="utf-8")


def draw_seed(base_seed: int, market: str, k: int) -> int:
    """Deterministic seed: BASE + 10000*market_code + k."""
    code = int(MARKET_SEED_CODE.get(market.upper(), 9))
    return int(base_seed) + 10000 * code + int(k)


def sample_without_replacement(
    pool: Sequence[str],
    m: int,
    seed: int,
) -> List[str]:
    """Random sample of size m; order is sample order (not re-sorted)."""
    pool_list = list(pool)
    if m < 0:
        raise ValueError("m must be >= 0")
    if m > len(pool_list):
        raise ValueError(f"m={m} > pool size {len(pool_list)}")
    rng = random.Random(int(seed))
    return rng.sample(pool_list, int(m))


def prefix_tickers(pool: Sequence[str], m: int) -> List[str]:
    """First m tickers in file order (Kaggle L50/L80 style)."""
    return list(pool)[: int(m)]


def ticker_history_start(data_root: Path, ticker: str) -> Optional[pd.Timestamp]:
    p = data_root / f"{ticker}_history.csv"
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p, usecols=lambda c: str(c).lower() in ("date", "datetime", "timestamp"))
        if df.empty:
            # fallback: read first column
            df = pd.read_csv(p, usecols=[0])
        col = df.columns[0]
        mn = pd.to_datetime(df[col], utc=True, errors="coerce").min()
        if pd.isna(mn):
            return None
        return pd.Timestamp(mn)
    except Exception:
        try:
            df = pd.read_csv(p)
            col = "date" if "date" in df.columns else df.columns[0]
            mn = pd.to_datetime(df[col], utc=True, errors="coerce").min()
            if pd.isna(mn):
                return None
            return pd.Timestamp(mn)
        except Exception:
            return None


def filter_pool_by_start(
    pool: Sequence[str],
    data_root: Path,
    *,
    max_start_year: int = 2010,
) -> List[str]:
    """Keep tickers with history start year <= max_start_year and file present."""
    out: List[str] = []
    for t in pool:
        mn = ticker_history_start(data_root, t)
        if mn is None:
            continue
        if int(mn.year) <= int(max_start_year):
            out.append(t)
    return out


def pool_coverage(
    market: str,
    *,
    data_root: Optional[Path] = None,
    universe_file: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    specs = market_specs(repo_root) if repo_root is not None else MARKET_SPECS
    spec = specs.get(market.upper(), {})
    root = Path(data_root or spec.get("data_root") or ROOT / "data")
    ufile = Path(universe_file or spec.get("universe_file") or ROOT / "universe_longhist2010_pass.txt")
    pool = read_tickers(ufile)
    n_files = 0
    n_2010 = 0
    n_2015 = 0
    starts: Dict[str, str] = {}
    for t in pool:
        mn = ticker_history_start(root, t)
        if mn is None:
            continue
        n_files += 1
        starts[t] = str(mn.date()) if hasattr(mn, "date") else str(mn)
        if int(mn.year) <= 2010:
            n_2010 += 1
        if int(mn.year) <= 2015:
            n_2015 += 1
    return {
        "market": market.upper(),
        "data_root": str(root),
        "universe_file": str(ufile),
        "n_universe": len(pool),
        "n_with_history": n_files,
        "n_start_le_2010": n_2010,
        "n_start_le_2015": n_2015,
        "preferred_index": list(spec.get("preferred_index") or ()),
        "generated": datetime.now(timezone.utc).isoformat(),
    }


def write_decade_pool(
    market: str,
    out_path: Path,
    *,
    max_start_year: int = 2010,
    data_root: Optional[Path] = None,
    universe_file: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[str]:
    specs = market_specs(repo_root) if repo_root is not None else MARKET_SPECS
    spec = specs.get(market.upper(), {})
    root = Path(data_root or spec.get("data_root") or ROOT / "data")
    ufile = Path(universe_file or spec.get("universe_file") or ROOT / "universe_longhist2010_pass.txt")
    pool = read_tickers(ufile)
    passed = filter_pool_by_start(pool, root, max_start_year=max_start_year)
    write_tickers(out_path, passed)
    return passed


def materialize_draw(
    pool: Sequence[str],
    *,
    series: str,
    m: int,
    seed: int,
    out_path: Path,
) -> List[str]:
    """Write draw file and return tickers."""
    series_u = series.upper()
    if series_u.startswith("PREFIX"):
        tickers = prefix_tickers(pool, m)
    elif series_u.startswith("FULL"):
        tickers = list(pool)
    else:
        tickers = sample_without_replacement(pool, m, seed)
    write_tickers(out_path, tickers)
    return tickers


def aggregate_numeric(
    values: Sequence[Optional[float]],
) -> Dict[str, Any]:
    xs = [float(v) for v in values if v is not None and v == v]  # drop NaN
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    s = pd.Series(xs, dtype=float)
    return {
        "n": int(len(xs)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(xs) > 1 else 0.0,
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def us_verdict(
    *,
    pass_rate: float,
    median_cagr: Optional[float],
    median_mdd: Optional[float],
    prefix_pass: bool,
    gate_cagr: float = 0.10,
    gate_mdd: float = -0.65,
) -> str:
    """Pre-registered US generalization verdict on S1·R50."""
    mc = median_cagr if median_cagr is not None else -1.0
    mm = median_mdd if median_mdd is not None else -1.0
    if pass_rate < 0.15 and prefix_pass:
        return "PREFIX-ONLY"
    if mc <= gate_cagr or mm < gate_mdd:
        return "FAIL"
    if pass_rate >= 0.40 and mc > gate_cagr and mm >= gate_mdd:
        return "GENERALIZES"
    if 0.15 <= pass_rate < 0.40:
        return "FRAGILE"
    return "FRAGILE"


def geo_verdict(
    market_median_pass: Dict[str, bool],
    *,
    uk_ok: Optional[bool] = None,
) -> str:
    """Pre-registered geo transfer verdict."""
    eu_core = [m for m in ("ES", "FR", "DE") if m in market_median_pass]
    n_ok = sum(1 for m in eu_core if market_median_pass.get(m))
    if not eu_core:
        return "FAIL_GEO"
    if n_ok >= 2 and (uk_ok is None or uk_ok):
        return "TRANSFERS"
    if n_ok == 0:
        return "FAIL_GEO" if not any(market_median_pass.values()) else "US_ONLY"
    if n_ok == 1:
        return "MIXED"
    return "MIXED"


def dumps_coverage(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"markets": list(rows), "generated": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
