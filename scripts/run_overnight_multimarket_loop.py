#!/usr/bin/env python
"""Overnight multi-market loop — anti-overfit protocol.

Why not US-only dense 2k grid?
  Fitting thousands of risk knobs on one market/window overfits.

Protocol
--------
1. **Screen US only** with *medium* smart grid (~450), OOS 2018–2025.
2. Take top-N US survivors (frozen knobs — no re-tune).
3. **Transfer** same configs to ES / DE / FR / UK (local data + local index regime).
4. **Global rank** = mean market score with min-market penalty.
5. Winner = best **global**, not best US-only.

Years: full ``--first-oos``…``--last-oos`` (default 2018–2025), never 2018 alone.
Train is **local per market/year** for risk overlays (honest local book); knobs
selected only from US screen ranking.

Research only. Not financial advice. Prior geo FROZEN often failed — transfer
may kill US winners. That is a valid scientific outcome.

Usage::

    $env:PYTHONPATH = (Get-Location).Path
    python scripts/run_overnight_multimarket_loop.py --hours 7
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("overnight_mm")


def _chunk(xs: List[Any], n: int) -> List[List[Any]]:
    n = max(1, int(n))
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def _strategy_overrides() -> Dict[str, Any]:
    from trad_research.strategies import HighVolMinAllocStrategy

    return dict(HighVolMinAllocStrategy().backtest_overrides())


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _rank_us(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from trad_research.multimarket import market_row_score

    return sorted(rows, key=lambda r: market_row_score(r), reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-market overnight anti-overfit loop")
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument(
        "--mode",
        choices=["smoke", "medium", "full"],
        default="medium",
        help="US screen density (default medium — full is overfit-prone alone)",
    )
    ap.add_argument("--us-universe-limit", type=int, default=40)
    ap.add_argument("--geo-universe-limit", type=int, default=40)
    ap.add_argument("--first-oos", type=int, default=2018)
    ap.add_argument("--last-oos", type=int, default=2025)
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--transfer-top", type=int, default=40)
    ap.add_argument(
        "--markets",
        type=str,
        default="US,ES,DE,FR,UK",
        help="Comma list; US must be first for screen",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "overnight_multimarket_2026-07-23",
    )
    ap.add_argument("--skip-transfer", action="store_true")
    args = ap.parse_args()

    from trad_research.multimarket import (
        available_markets,
        default_markets,
        global_rank_table,
    )
    from trad_research.overnight_grid import (
        build_phase1_risk_cells,
        cells_to_mega_configs,
        estimate_grid_sizes,
    )
    from scripts.run_crash_entry_mega_study import run_mega

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    budget_s = float(args.hours) * 3600.0
    t0 = time.time()
    stop_reason = "completed"

    want = {x.strip().upper() for x in args.markets.split(",") if x.strip()}
    specs = [
        m
        for m in available_markets(
            default_markets(
                us_univ_limit=args.us_universe_limit,
                geo_univ_limit=args.geo_universe_limit,
            )
        )
        if m.market_id in want
    ]
    if not any(m.market_id == "US" for m in specs):
        logger.error("US market required for screen")
        return 2

    us = next(m for m in specs if m.market_id == "US")
    transfer_specs = [m for m in specs if m.role == "transfer"]

    logger.info("Grid sizes %s", estimate_grid_sizes())
    logger.info(
        "Start multi-market hours=%.1f mode=%s OOS=%s-%s markets=%s",
        args.hours,
        args.mode,
        args.first_oos,
        args.last_oos,
        [m.market_id for m in specs],
    )
    logger.info(
        "Protocol: screen US → freeze top-%d knobs → transfer %s → global rank",
        args.transfer_top,
        [m.market_id for m in transfer_specs],
    )

    cells = build_phase1_risk_cells(mode=args.mode)
    overrides = _strategy_overrides()
    us_rows: List[Dict[str, Any]] = []

    # ---------- Phase US screen ----------
    batches = _chunk(cells, args.batch_size)
    for bi, batch in enumerate(batches):
        if time.time() - t0 >= budget_s * 0.55:
            stop_reason = f"time_budget_us_after_batch_{bi}"
            logger.warning("US screen stop for time (leave room for transfer)")
            break
        batch_dir = out_dir / "US" / f"batch_{bi:03d}"
        configs = cells_to_mega_configs(batch, strategy_overrides=overrides)
        fp_map = {f"turbo_highvol_minalloc__{c.label}": c.fingerprint() for c in batch}
        logger.info("US batch %d/%d n=%d", bi + 1, len(batches), len(configs))
        try:
            summary = run_mega(
                data_root=us.data_root,
                ticker_file=us.ticker_file,
                universe_limit=us.universe_limit,
                first_oos=args.first_oos,
                last_oos=args.last_oos,
                grid=f"mm_us_b{bi}",
                out_dir=batch_dir,
                configs=configs,
                regime_key=us.regime_key,
                preferred_index=list(us.preferred_index),
                market_id="US",
            )
        except Exception:
            logger.error("US batch fail:\n%s", traceback.format_exc())
            stop_reason = f"error_us_batch_{bi}"
            break
        rows = summary.get("all_rows") or []
        for r in rows:
            r["fingerprint"] = fp_map.get(str(r.get("id") or ""))
            r["market_id"] = "US"
        us_rows.extend(rows)
        _write_json(
            out_dir / "PROGRESS.json",
            {
                "phase": "us_screen",
                "batch": bi,
                "n_us": len(us_rows),
                "elapsed_sec": round(time.time() - t0, 1),
            },
        )

    us_ranked = _rank_us([r for r in us_rows if not r.get("error")])
    _write_json(out_dir / "US" / "all_rows.json", us_ranked)
    top_n = us_ranked[: int(args.transfer_top)]
    top_labels = [str(r.get("label")) for r in top_n]
    logger.info(
        "US screen done n=%d top1=%s score_fields cagr=%.2f mdd=%.2f",
        len(us_ranked),
        top_labels[0] if top_labels else None,
        float(top_n[0].get("cagr") or 0) if top_n else 0,
        float(top_n[0].get("max_drawdown") or 0) if top_n else 0,
    )

    # Rebuild cells for transfer (match labels)
    by_label = {c.label: c for c in cells}
    transfer_cells = [by_label[lab] for lab in top_labels if lab in by_label]
    # always include baseline + HOLD anchor if present
    for must in ("baseline", "dd35_vt80_yr"):
        if must in by_label and by_label[must] not in transfer_cells:
            transfer_cells.insert(0, by_label[must])

    per_market: Dict[str, List[Dict[str, Any]]] = {"US": us_ranked}

    # ---------- Transfer markets ----------
    if not args.skip_transfer and transfer_specs and transfer_cells:
        for mi, mkt in enumerate(transfer_specs):
            if time.time() - t0 >= budget_s * 0.95:
                stop_reason = f"time_budget_before_transfer_{mkt.market_id}"
                break
            logger.info(
                "=== TRANSFER %s (%d frozen configs) data=%s ===",
                mkt.market_id,
                len(transfer_cells),
                mkt.data_root,
            )
            mkt_rows: List[Dict[str, Any]] = []
            for bi, batch in enumerate(_chunk(transfer_cells, max(15, args.batch_size // 2))):
                if time.time() - t0 >= budget_s * 0.95:
                    stop_reason = f"time_budget_transfer_{mkt.market_id}_b{bi}"
                    break
                batch_dir = out_dir / mkt.market_id / f"batch_{bi:03d}"
                configs = cells_to_mega_configs(batch, strategy_overrides=overrides)
                try:
                    summary = run_mega(
                        data_root=mkt.data_root,
                        ticker_file=mkt.ticker_file,
                        universe_limit=mkt.universe_limit,
                        first_oos=args.first_oos,
                        last_oos=args.last_oos,
                        grid=f"mm_{mkt.market_id}_b{bi}",
                        out_dir=batch_dir,
                        configs=configs,
                        regime_key=mkt.regime_key,
                        preferred_index=list(mkt.preferred_index),
                        market_id=mkt.market_id,
                    )
                except Exception:
                    logger.error(
                        "Transfer %s batch %d fail:\n%s",
                        mkt.market_id,
                        bi,
                        traceback.format_exc(),
                    )
                    continue
                rows = summary.get("all_rows") or []
                for r in rows:
                    r["market_id"] = mkt.market_id
                mkt_rows.extend(rows)
            per_market[mkt.market_id] = mkt_rows
            _write_json(out_dir / mkt.market_id / "all_rows.json", mkt_rows)
            _write_json(
                out_dir / "PROGRESS.json",
                {
                    "phase": f"transfer_{mkt.market_id}",
                    "n_us": len(us_ranked),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "markets_done": list(per_market.keys()),
                },
            )
    elif args.skip_transfer:
        stop_reason = "skip_transfer"

    # ---------- Global rank ----------
    global_table = global_rank_table(per_market, id_key="label")
    _write_json(out_dir / "global_rank.json", global_table)

    # SUMMARY.md
    lines = [
        "# Overnight multi-market SUMMARY",
        "",
        "> **Research only.** Not financial advice. Past backtests ≠ future results.",
        "",
        "## Anti-overfit protocol",
        "",
        "1. Screen knobs on **US only** (full OOS years, not a single year).",
        "2. Freeze top-N configs — **no re-tune** on ES/DE/FR/UK.",
        "3. Transfer same knobs; rank by **global score** (mean − min penalty).",
        "",
        f"- Mode US screen: `{args.mode}`",
        f"- OOS: **{args.first_oos}–{args.last_oos}**",
        f"- Markets: {', '.join(per_market.keys())}",
        f"- US configs completed: {len(us_ranked)}",
        f"- Transfer top-N: {args.transfer_top}",
        f"- Elapsed: {round(time.time()-t0,1)}s",
        f"- Stop: `{stop_reason}`",
        "",
        "## Global top 15",
        "",
        "| rank | label | global | mean | min | n_ok |",
        "|------|-------|--------|------|-----|------|",
    ]
    for i, g in enumerate(global_table[:15], 1):
        lines.append(
            f"| {i} | `{g['label']}` | {g['global_score']:.2f} | "
            f"{g['mean_score']:.2f} | {g['min_score']:.2f} | "
            f"{g['n_markets_ok']}/{g['n_markets']} |"
        )

    if global_table:
        best = global_table[0]
        lines.extend(
            [
                "",
                "## Winner (global)",
                "",
                f"- **label:** `{best['label']}`",
                f"- **global_score:** {best['global_score']:.3f}",
                f"- **markets OK:** {best['n_markets_ok']}/{best['n_markets']}",
                "",
                "### Per-market snapshot",
                "",
            ]
        )
        for mid, d in (best.get("per_market") or {}).items():
            if d.get("missing"):
                lines.append(f"- **{mid}:** missing")
            else:
                lines.append(
                    f"- **{mid}:** CAGR {100*float(d.get('cagr') or 0):.1f}% "
                    f"MDD {100*float(d.get('max_drawdown') or 0):.1f}% "
                    f"Sharpe {float(d.get('sharpe') or 0):.2f} "
                    f"excess {100*float(d.get('excess_total_vs_spy') or 0):.0f}%"
                )

    lines.extend(
        [
            "",
            "## US-only top 5 (screen — may overfit if used alone)",
            "",
            "| rank | label | CAGR | MDD | Sharpe | excess |",
            "|------|-------|------|-----|--------|--------|",
        ]
    )
    for i, r in enumerate(us_ranked[:5], 1):
        lines.append(
            f"| {i} | `{r.get('label')}` | {100*float(r.get('cagr') or 0):.1f}% | "
            f"{100*float(r.get('max_drawdown') or 0):.1f}% | "
            f"{float(r.get('sharpe') or 0):.2f} | "
            f"{100*float(r.get('excess_total_vs_spy') or 0):.0f}% |"
        )

    lines.extend(
        [
            "",
            "## Honesty",
            "",
            "- Prior S1c geo FROZEN often failed ES/DE — US sleeves may **not** transfer.",
            "- Global winner is for **research ranking**, not paper freeze auto-promote.",
            "- Paper freeze remains `turbo_highvol_minalloc` until promotion ADVANCE.",
            "",
        ]
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "protocol": "us_screen_then_frozen_transfer_global_rank",
        "mode": args.mode,
        "first_oos": args.first_oos,
        "last_oos": args.last_oos,
        "markets": list(per_market.keys()),
        "n_us": len(us_ranked),
        "transfer_top": args.transfer_top,
        "elapsed_sec": round(time.time() - t0, 1),
        "stop_reason": stop_reason,
        "global_winner": global_table[0]["label"] if global_table else None,
        "disclaimer": "Research only. Not financial advice.",
    }
    _write_json(out_dir / "summary.json", {**meta, "global_top15": global_table[:15]})
    _write_json(out_dir / "PROGRESS.json", {**meta, "phase": "done"})

    logger.info(
        "DONE global_winner=%s elapsed=%.0fs stop=%s",
        meta["global_winner"],
        meta["elapsed_sec"],
        stop_reason,
    )
    print(f"\nMulti-market overnight → {out_dir / 'SUMMARY.md'}")
    print(f"winner={meta['global_winner']} stop={stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
