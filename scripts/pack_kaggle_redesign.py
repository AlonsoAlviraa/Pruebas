"""Pack longhist OHLCV + code for Kaggle redesign mega dataset.

Creates a folder ready to zip/upload. Does not upload by itself.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_redesign.src.grids import theoretical_millions  # noqa: E402


def _read_tickers(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [
        ln.strip().upper()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "dist" / "kaggle_redesign_bundle",
    )
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    data_out = out / "data"
    code_out = out / "code"
    data_out.mkdir(parents=True)
    code_out.mkdir(parents=True)

    universe_files = [
        "universe_longhist100.txt",
        "universe_longhist2010_pass.txt",
        "universe_highvol80.txt",
        "universe_highvol80_2010_pass.txt",
        "universe_quality80.txt",
    ]
    tickers = set(["SPY", "QQQ"])
    for uf in universe_files:
        p = ROOT / uf
        if p.is_file():
            shutil.copy2(p, out / uf)
            tickers.update(_read_tickers(p))

    data_root = Path(args.data_root)
    copied = 0
    missing = []
    for t in sorted(tickers):
        src = data_root / f"{t}_history.csv"
        if src.is_file():
            shutil.copy2(src, data_out / src.name)
            copied += 1
        else:
            missing.append(t)

    # code slices
    for rel in [
        "trad_research",
        "kaggle_redesign",
        "docs/design/2026-07-25_kaggle_gpu_mega_redesign.md",
        "docs/design/2026-07-25_redesign_v2_features_graphs.md",
    ]:
        src = ROOT / rel
        if src.is_dir():
            shutil.copytree(
                src,
                code_out / src.name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
        elif src.is_file():
            dest = code_out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_tickers_requested": len(tickers),
        "n_history_copied": copied,
        "missing_history": missing,
        "grid_theory": theoretical_millions(),
        "protocol": {
            "screen": "2010-2017",
            "confirm": "2018-2025",
            "gates": {"cagr_gt": 0.10, "mdd_ge": -0.65, "n_trades_ge": 80},
            "paper_freeze": "turbo_highvol_minalloc — do not auto-change",
        },
        "disclaimer": "Research only. Not financial advice.",
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out / "README_UPLOAD.md").write_text(
        "\n".join(
            [
                "# Upload to Kaggle",
                "",
                "1. Zip this folder or use `kaggle datasets create`.",
                "2. Attach as input to GPU notebook.",
                "3. Follow `code/kaggle_redesign/README.md` and design doc.",
                "",
                f"History CSVs copied: **{copied}**. Missing: {len(missing)}.",
                f"Theoretical full grid: **{manifest['grid_theory']['full_grid_size']:,}** combos.",
                "",
                "Research only. Not financial advice.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Bundle → {out}")
    print(f"  history={copied} missing={len(missing)}")
    print(f"  full_grid={manifest['grid_theory']['full_grid_size']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
