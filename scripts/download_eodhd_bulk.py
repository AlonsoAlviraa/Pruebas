"""Bulk EODHD OHLCV download into data/{TICKER}_history.csv.

Anti-truncation: only overwrite existing CSV if the new series extends earlier
or later (or has more rows). Never replace a longer series with a shorter one.

Research only. Requires EODHD_API_TOKEN / EODHD_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_live.data.eodhd_client import fetch_eod, get_token  # noqa: E402

OHLCV = ["date", "open", "high", "low", "close", "volume"]


def _read_tickers(path: Path) -> List[str]:
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip().upper()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    out = df.copy()
    if "date" not in out.columns:
        return pd.DataFrame(columns=OHLCV)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = float("nan")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out[OHLCV].reset_index(drop=True)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=OHLCV)
    try:
        return _normalize(pd.read_csv(path))
    except Exception:
        return pd.DataFrame(columns=OHLCV)


def _merge_prefer_longer(old: pd.DataFrame, new: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Union on date; prefer new non-null closes when both present."""
    if new is None or new.empty:
        return old, "keep_old_empty_new"
    if old is None or old.empty:
        return new, "write_new"
    # If new is strictly shorter and starts later and ends earlier → reject
    o0, o1 = old["date"].min(), old["date"].max()
    n0, n1 = new["date"].min(), new["date"].max()
    if len(new) < len(old) and n0 >= o0 and n1 <= o1:
        return old, "reject_shorter"
    # Merge outer
    m = pd.concat([old.assign(_src=0), new.assign(_src=1)], ignore_index=True)
    m = m.sort_values(["date", "_src"]).drop_duplicates(subset=["date"], keep="last")
    m = m.drop(columns=["_src"])
    m = _normalize(m)
    if len(m) < len(old):
        return old, "reject_merge_shorter"
    tag = "merged"
    if n0 < o0:
        tag += "_extend_start"
    if n1 > o1:
        tag += "_extend_end"
    return m, tag


def download_one(
    ticker: str,
    *,
    data_root: Path,
    start: str,
    end: Optional[str],
    sleep_s: float,
) -> Dict[str, Any]:
    path = data_root / f"{ticker}_history.csv"
    old = _load_existing(path)
    old_n = int(len(old))
    old_min = str(old["date"].min()) if old_n else None
    try:
        raw = fetch_eod(ticker, start=start, end=end)
    except Exception as e:
        return {
            "ticker": ticker,
            "status": "error",
            "error": f"{type(e).__name__}:{e}",
            "old_rows": old_n,
            "old_min": old_min,
        }
    new = _normalize(raw)
    if new.empty:
        return {
            "ticker": ticker,
            "status": "empty",
            "old_rows": old_n,
            "old_min": old_min,
        }
    merged, action = _merge_prefer_longer(old, new)
    if action == "reject_shorter" or action == "reject_merge_shorter":
        return {
            "ticker": ticker,
            "status": "skip_keep_longer",
            "action": action,
            "old_rows": old_n,
            "new_rows": int(len(new)),
            "old_min": old_min,
            "new_min": str(new["date"].min()),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    if sleep_s > 0:
        time.sleep(sleep_s)
    return {
        "ticker": ticker,
        "status": "ok",
        "action": action,
        "rows": int(len(merged)),
        "old_rows": old_n,
        "min": str(merged["date"].min()),
        "max": str(merged["date"].max()),
    }


def coverage_report(
    tickers: Sequence[str],
    data_root: Path,
    *,
    year_2010: int = 2010,
    year_2012: int = 2012,
) -> Dict[str, Any]:
    rows = []
    for t in tickers:
        p = data_root / f"{t}_history.csv"
        if not p.is_file():
            rows.append({"ticker": t, "ok": False, "missing": True})
            continue
        df = _load_existing(p)
        if df.empty:
            rows.append({"ticker": t, "ok": False, "empty": True})
            continue
        y0 = int(df["date"].min().year)
        rows.append(
            {
                "ticker": t,
                "ok": True,
                "rows": int(len(df)),
                "min": str(df["date"].min()),
                "max": str(df["date"].max()),
                "start_year": y0,
                "le_2010": y0 <= year_2010,
                "le_2012": y0 <= year_2012,
            }
        )
    ok = [r for r in rows if r.get("ok")]
    n = len(tickers)
    n_ok = len(ok)
    n_2010 = sum(1 for r in ok if r.get("le_2010"))
    n_2012 = sum(1 for r in ok if r.get("le_2012"))
    pass_2010 = n_2010 >= max(1, int(0.80 * n))
    pass_2012 = n_2012 >= max(1, int(0.90 * n))
    passers = [r["ticker"] for r in ok if r.get("le_2010")]
    return {
        "n": n,
        "n_ok": n_ok,
        "n_le_2010": n_2010,
        "n_le_2012": n_2012,
        "frac_le_2010": n_2010 / n if n else 0.0,
        "frac_le_2012": n_2012 / n if n else 0.0,
        "pass_coverage_2010": pass_2010,
        "pass_coverage_2012": pass_2012,
        "pass_data": bool(pass_2010 and pass_2012),
        "passers_2010": passers,
        "rows": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EODHD bulk OHLCV → data/*_history.csv")
    ap.add_argument(
        "--tickers-file",
        type=Path,
        default=ROOT / "universe_longhist100.txt",
    )
    ap.add_argument("--extra", type=str, default="SPY,QQQ", help="Comma tickers extra")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--from", dest="from_date", type=str, default="2000-01-01")
    ap.add_argument("--to", dest="to_date", type=str, default="")
    ap.add_argument("--sleep", type=float, default=0.12)
    ap.add_argument("--max", type=int, default=0, help="0=all")
    ap.add_argument(
        "--coverage-out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "longpath_2010" / "data_coverage.json",
    )
    ap.add_argument(
        "--passers-out",
        type=Path,
        default=ROOT / "universe_longhist2010_pass.txt",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    # token early
    try:
        get_token()
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    tickers = _read_tickers(Path(args.tickers_file))
    for x in str(args.extra or "").split(","):
        x = x.strip().upper()
        if x and x not in tickers:
            tickers.append(x)
    if int(args.max) > 0:
        tickers = tickers[: int(args.max)]

    data_root = Path(args.data_root)
    end = args.to_date.strip() or None
    print(
        f"EODHD bulk n={len(tickers)} from={args.from_date} to={end or 'today'} "
        f"→ {data_root}",
        flush=True,
    )
    t0 = time.time()
    details: List[Dict[str, Any]] = []
    stats = {"ok": 0, "empty": 0, "error": 0, "skip": 0}
    for i, t in enumerate(tickers, 1):
        r = download_one(
            t,
            data_root=data_root,
            start=str(args.from_date),
            end=end,
            sleep_s=float(args.sleep),
        )
        details.append(r)
        st = str(r.get("status") or "")
        if st == "ok":
            stats["ok"] += 1
        elif st == "empty":
            stats["empty"] += 1
        elif st.startswith("skip"):
            stats["skip"] += 1
        else:
            stats["error"] += 1
        if i % 10 == 0 or i == len(tickers):
            print(
                f"  [{i}/{len(tickers)}] last={t} status={st} stats={stats}",
                flush=True,
            )

    # coverage only on universe file (exclude pure extras unless in file)
    uni = _read_tickers(Path(args.tickers_file))
    cov = coverage_report(uni, data_root)
    cov_out = Path(args.coverage_out)
    if not cov_out.is_absolute():
        cov_out = ROOT / cov_out
    cov_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "from": args.from_date,
        "to": end,
        "elapsed_sec": time.time() - t0,
        "stats": stats,
        "n_tickers": len(tickers),
        "coverage": {
            k: v
            for k, v in cov.items()
            if k != "rows"
        },
        "details_sample": details[:15],
        "details": details,
        "coverage_rows": cov["rows"],
    }
    cov_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    passers = cov.get("passers_2010") or []
    pout = Path(args.passers_out)
    if not pout.is_absolute():
        pout = ROOT / pout
    pout.write_text("\n".join(passers) + ("\n" if passers else ""), encoding="utf-8")

    print(
        f"Coverage le2010={cov['n_le_2010']}/{cov['n']} "
        f"le2012={cov['n_le_2012']}/{cov['n']} "
        f"pass_data={cov['pass_data']}",
        flush=True,
    )
    print(f"Wrote {cov_out}", flush=True)
    print(f"Passers 2010 n={len(passers)} → {pout}", flush=True)
    return 0 if stats["ok"] + stats["skip"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
