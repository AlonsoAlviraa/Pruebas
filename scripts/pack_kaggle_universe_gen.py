#!/usr/bin/env python
"""Pack US+EU OHLCV + code for Kaggle universe-generalization overnight.

Creates dist/kaggle_universe_gen_bundle/ ready for:
  kaggle datasets version -p <abs> -m "..." --dir-mode zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_tickers(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def _copy_histories(tickers: Set[str], data_root: Path, dest: Path) -> tuple[int, List[str]]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing: List[str] = []
    for t in sorted(tickers):
        src = data_root / f"{t}_history.csv"
        if src.is_file():
            shutil.copy2(src, dest / src.name)
            copied += 1
        else:
            missing.append(t)
    return copied, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "dist" / "kaggle_universe_gen_bundle",
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    universe_files = [
        "universe_longhist100.txt",
        "universe_longhist2010_pass.txt",
        "universe_highvol80.txt",
        "universe_highvol80_2010_pass.txt",
        "spain_wf_universe.txt",
        "germany_wf_universe.txt",
        "france_wf_universe.txt",
        "uk_wf_universe.txt",
    ]
    for uf in universe_files:
        p = ROOT / uf
        if p.is_file():
            shutil.copy2(p, out / uf)

    # US longhist + benchmarks
    us_tickers: Set[str] = {"SPY", "QQQ"}
    for uf in ("universe_longhist2010_pass.txt", "universe_longhist100.txt"):
        us_tickers.update(_read_tickers(ROOT / uf))
    n_us, miss_us = _copy_histories(us_tickers, ROOT / "data", out / "data")

    markets = {
        "ES": ("data_es", "spain_wf_universe.txt", {"IBEX"}),
        "DE": ("data_de", "germany_wf_universe.txt", {"DAX"}),
        "FR": ("data_fr", "france_wf_universe.txt", {"CAC"}),
        "UK": ("data_uk", "uk_wf_universe.txt", {"FTSE"}),
    }
    eu_stats = {}
    for mid, (dname, ufile, idx) in markets.items():
        ticks = set(idx)
        ticks.update(_read_tickers(ROOT / ufile))
        n, miss = _copy_histories(ticks, ROOT / dname, out / dname)
        eu_stats[mid] = {"copied": n, "missing": miss, "n_univ": len(ticks)}

    # code
    code_out = out / "code"
    code_out.mkdir(parents=True)
    for rel in ("trad_research",):
        src = ROOT / rel
        shutil.copytree(
            src,
            code_out / rel,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    scripts_out = code_out / "scripts"
    scripts_out.mkdir(parents=True)
    for name in (
        "run_universe_generalization_overnight.py",
        "pack_kaggle_universe_gen.py",
    ):
        src = ROOT / "scripts" / name
        if src.is_file():
            shutil.copy2(src, scripts_out / name)

    # notebook into dataset root for easy discovery
    nb = ROOT / "kaggle_redesign" / "notebook" / "KAGGLE_UNIV_GEN_T4X2.py"
    if nb.is_file():
        shutil.copy2(nb, out / "KAGGLE_UNIV_GEN_T4X2.py")

    # dataset metadata
    meta_src = ROOT / "kaggle_redesign" / "dataset-metadata.json"
    if meta_src.is_file():
        shutil.copy2(meta_src, out / "dataset-metadata.json")

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "purpose": "universe_generalization_overnight_us_eu",
        "us_history_copied": n_us,
        "us_missing": miss_us,
        "eu": eu_stats,
        "protocol": {
            "screen": "2010-2017",
            "confirm": "2018-2025",
            "gates": {"cagr_gt": 0.10, "mdd_ge": -0.65, "n_trades_ge": 80},
            "paper_freeze": "turbo_highvol_minalloc",
            "kaggle": "GPU T4 x2 · 8h · dense K",
        },
        "disclaimer": "Research only. Not financial advice.",
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out / "README_UPLOAD.md").write_text(
        "\n".join(
            [
                "# TRAD Universe Generalization (US + EU)",
                "",
                f"US history: **{n_us}**. EU: {json.dumps({k: v['copied'] for k, v in eu_stats.items()})}",
                "",
                "Kernel: attach this dataset, accelerator **GPU T4 x2**, run `KAGGLE_UNIV_GEN_T4X2.py`.",
                "",
                "Research only. Not financial advice.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Bundle → {out}")
    print(f"  US history={n_us} missing={len(miss_us)}")
    for mid, st in eu_stats.items():
        print(f"  {mid} history={st['copied']} missing={len(st['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
