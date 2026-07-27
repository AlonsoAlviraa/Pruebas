#!/usr/bin/env python3
"""Offline multi-filter rescore of mega annual alpha pack.

Reads reports/mega_annual_alpha/latest/full_results.json (no network).
Emits realistic promote/watch/kill under vs-SPY and vs-best rules.

VIRTUAL research only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MARGIN = 0.03


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def _best(e: Mapping[str, Any]) -> Optional[float]:
    vals = [_f(e.get(k)) for k in ("spy_bh", "qqq_bh", "iwm_bh")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def passes(e: Mapping[str, Any], mode: str, margin: float = MARGIN) -> bool:
    r = _f(e.get("total_return"))
    if r is None or e.get("error"):
        return False
    if mode == "best":
        b = _best(e)
        return b is not None and r >= b + margin
    if mode == "spy":
        s = _f(e.get("spy_bh"))
        return s is not None and r >= s + margin
    if mode == "qqq":
        q = _f(e.get("qqq_bh"))
        return q is not None and r >= q + margin
    if mode == "qqq0":
        q = _f(e.get("qqq_bh"))
        return q is not None and r >= q  # beat or match QQQ
    return False


def aggregate(
    year_evals: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    margin: float = MARGIN,
) -> List[Dict[str, Any]]:
    by: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for e in year_evals:
        by[str(e.get("strategy_id") or "")].append(e)

    out: List[Dict[str, Any]] = []
    for sid, rows in by.items():
        if not sid:
            continue
        ok = [r for r in rows if not r.get("error")]
        flags = {str(r.get("year")): passes(r, mode, margin) for r in ok}
        n_pass = sum(1 for v in flags.values() if v)
        n_y = len(ok)
        rets = [_f(r.get("total_return")) for r in ok]
        rets = [x for x in rets if x is not None]
        dds = [_f(r.get("max_dd")) for r in ok]
        dds = [x for x in dds if x is not None]
        xs: List[float] = []
        xs_spy: List[float] = []
        for r in ok:
            ret = _f(r.get("total_return"))
            if ret is None:
                continue
            b = _best(r)
            if b is not None:
                xs.append(ret - b)
            sp = _f(r.get("spy_bh"))
            if sp is not None:
                xs_spy.append(ret - sp)
        ac = str(ok[0].get("asset_class") or "equity") if ok else "equity"
        kills = sum(1 for r in ok if r.get("hard_kill"))
        opens = sum(int(r.get("n_opens") or 0) for r in ok)
        out.append(
            {
                "strategy_id": sid,
                "asset_class": ac,
                "mode": mode,
                "years_evaluated": n_y,
                "years_passed": n_pass,
                "tier": f"{n_pass}/{n_y}" if n_y else "0/0",
                "beat_flags": flags,
                "mean_return": sum(rets) / len(rets) if rets else None,
                "mean_excess_vs_best": sum(xs) / len(xs) if xs else None,
                "mean_excess_vs_spy": sum(xs_spy) / len(xs_spy) if xs_spy else None,
                "worst_max_dd": min(dds) if dds else None,
                "hard_kill_years": kills,
                "total_opens": opens,
                "year_returns": {str(r.get("year")): r.get("total_return") for r in ok},
            }
        )
    out.sort(
        key=lambda x: (
            -int(x["years_passed"]),
            -(x["mean_excess_vs_spy"] or -999),
            -(x["mean_return"] or -999),
        )
    )
    return out


def decide_verdicts(
    spy_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Realistic scorecard from analysis plan.

    PROMOTE_RESEARCH: vs SPY+3 in ≥2/4, worst DD > -35%, no multi hard-kill
    WATCH: best+3 in exactly 1 year (2024-style) OR spy years==1 with solid mean
    KILL: mean ret < 0 and worst DD ≤ -25%, or hard_kill ≥ 2
    HOLD: else
    """
    best_by = {r["strategy_id"]: r for r in best_rows}
    verdicts: List[Dict[str, Any]] = []
    for r in spy_rows:
        sid = r["strategy_id"]
        b = best_by.get(sid) or {}
        reasons: List[str] = []
        worst_dd = r.get("worst_max_dd")
        mean_r = r.get("mean_return")
        spy_y = int(r.get("years_passed") or 0)
        best_y = int(b.get("years_passed") or 0)
        kills = int(r.get("hard_kill_years") or 0)

        if sid.endswith("_cash") or r.get("asset_class") == "options" and "cash" in sid.lower():
            verdict = "HOLD"
            reasons.append("cash_or_control")
        elif kills >= 2 or (
            mean_r is not None
            and mean_r < 0
            and worst_dd is not None
            and worst_dd <= -0.25
        ):
            verdict = "KILL"
            reasons.append("negative_mean_and_deep_dd_or_kills")
        elif (
            spy_y >= 2
            and (worst_dd is None or worst_dd > -0.35)
            and kills == 0
        ):
            verdict = "PROMOTE_RESEARCH"
            reasons.append(f"vs_SPY+3pp in {spy_y}/4 years; DD ok")
        elif best_y >= 1 or spy_y == 1:
            verdict = "WATCH"
            reasons.append(
                f"best+3 years={best_y}; spy+3 years={spy_y} (2024-style or partial)"
            )
        else:
            verdict = "HOLD"
            reasons.append("no_strong_multi_year_edge")

        verdicts.append(
            {
                **r,
                "best_years_passed": best_y,
                "best_tier": b.get("tier"),
                "verdict": verdict,
                "reasons": reasons,
            }
        )
    order = {"PROMOTE_RESEARCH": 0, "WATCH": 1, "HOLD": 2, "KILL": 3}
    verdicts.sort(key=lambda x: (order.get(str(x["verdict"]), 9), x["strategy_id"]))
    return verdicts


def _pct(x: Any) -> str:
    v = _f(x)
    return "n/a" if v is None else f"{v:.2%}"


def to_markdown(
    *,
    payload: Mapping[str, Any],
    verdicts: Sequence[Mapping[str, Any]],
    spy_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for v in verdicts:
        counts[str(v.get("verdict"))] += 1
    lines = [
        f"# Mega annual RESCORE — `{payload.get('generated_at')}`",
        "",
        "**Offline** rescore of existing pack (no re-backtest).",
        "",
        "## Why strict best+3pp every year failed",
        "",
        "Beating **max(SPY,QQQ,IWM)+3pp every year** requires beating **QQQ+3pp** in tech-bull years.",
        "Long-only QQQ hold itself cannot clear that bar (excess ≈ 0, not +3pp).",
        "Only **2024** showed stock-picking alpha vs QQQ in this pack.",
        "",
        "## Verdict counts (realistic rules)",
        "",
        "| Verdict | N |",
        "|---------|---|",
    ]
    for k in ("PROMOTE_RESEARCH", "WATCH", "HOLD", "KILL"):
        lines.append(f"| **{k}** | {counts.get(k, 0)} |")

    lines += [
        "",
        "### Rules",
        "",
        "- **PROMOTE_RESEARCH:** vs SPY +3pp in ≥2/4 years, maxDD > −35%, hard_kill_years=0",
        "- **WATCH:** best+3pp in ≥1 year OR spy+3pp in exactly 1 year",
        "- **KILL:** mean ret < 0 and deep DD, or multi hard-kill",
        "- **HOLD:** else (incl. cash / weak options)",
        "",
        "## Decision table",
        "",
        "| Verdict | ID | Class | SPY+3 yrs | Best+3 yrs | MeanRet | xsSPY | xsBest | WorstDD | Opens | Reasons |",
        "|---------|----|-------|-----------|------------|---------|-------|--------|---------|-------|---------|",
    ]
    for v in verdicts:
        lines.append(
            f"| {v.get('verdict')} | `{v.get('strategy_id')}` | {v.get('asset_class')} | "
            f"{v.get('years_passed')}/{v.get('years_evaluated')} | "
            f"{v.get('best_years_passed')}/{v.get('years_evaluated')} | "
            f"{_pct(v.get('mean_return'))} | {_pct(v.get('mean_excess_vs_spy'))} | "
            f"{_pct(v.get('mean_excess_vs_best'))} | {_pct(v.get('worst_max_dd'))} | "
            f"{v.get('total_opens')} | {'; '.join(v.get('reasons') or [])[:70]} |"
        )

    lines += [
        "",
        "## Tier: vs SPY +3pp",
        "",
        "| ID | Tier | Mean xs SPY | Mean ret | Flags |",
        "|----|------|-------------|----------|-------|",
    ]
    for r in spy_rows:
        flags = ",".join(
            f"{y}:{'Y' if f else 'n'}" for y, f in sorted((r.get("beat_flags") or {}).items())
        )
        lines.append(
            f"| `{r['strategy_id']}` | {r['tier']} | {_pct(r.get('mean_excess_vs_spy'))} | "
            f"{_pct(r.get('mean_return'))} | {flags} |"
        )

    lines += [
        "",
        "## Tier: vs best index +3pp",
        "",
        "| ID | Tier | Mean xs best | Mean ret | Flags |",
        "|----|------|--------------|----------|-------|",
    ]
    for r in best_rows:
        flags = ",".join(
            f"{y}:{'Y' if f else 'n'}" for y, f in sorted((r.get("beat_flags") or {}).items())
        )
        lines.append(
            f"| `{r['strategy_id']}` | {r['tier']} | {_pct(r.get('mean_excess_vs_best'))} | "
            f"{_pct(r.get('mean_return'))} | {flags} |"
        )

    prom = [v["strategy_id"] for v in verdicts if v.get("verdict") == "PROMOTE_RESEARCH"]
    watch = [v["strategy_id"] for v in verdicts if v.get("verdict") == "WATCH"]
    lines += [
        "",
        "## Promote (research)",
        "",
    ]
    if prom:
        for p in prom:
            lines.append(f"- `{p}`")
    else:
        lines.append("_None under realistic SPY+3 ≥2/4 rule_")
    lines += ["", "## Watch", ""]
    if watch:
        for p in watch:
            lines.append(f"- `{p}`")
    else:
        lines.append("_None_")

    lines += [
        "",
        "---",
        "Research rescore only. VIRTUAL capital. Not trade advice. Not OPRA fills.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline multi-filter rescore of mega annual pack")
    ap.add_argument(
        "--in",
        dest="inp",
        default="reports/mega_annual_alpha/latest",
        help="Dir with full_results.json or path to the file",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output markdown path (default: <pack>/RESCORE.md)",
    )
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    inp = Path(args.inp)
    if inp.is_dir():
        full_path = inp / "full_results.json"
        out_root = inp if inp.name != "latest" else inp.parent
        default_md = out_root / "RESCORE.md"
    else:
        full_path = inp
        out_root = inp.parent
        default_md = out_root / "RESCORE.md"

    if not full_path.is_file():
        print(f"FAIL: missing {full_path}", file=sys.stderr)
        return 1

    raw = json.loads(full_path.read_text(encoding="utf-8"))
    year_evals = raw.get("year_evals") or []
    if not year_evals:
        print("FAIL: no year_evals", file=sys.stderr)
        return 2

    margin = float(args.margin)
    spy_rows = aggregate(year_evals, mode="spy", margin=margin)
    best_rows = aggregate(year_evals, mode="best", margin=margin)
    qqq_rows = aggregate(year_evals, mode="qqq", margin=margin)
    verdicts = decide_verdicts(spy_rows, best_rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(full_path),
        "margin": margin,
        "n_strategies": len(spy_rows),
        "n_year_evals": len(year_evals),
        "windows": raw.get("windows"),
        "counts": {
            k: sum(1 for v in verdicts if v.get("verdict") == k)
            for k in ("PROMOTE_RESEARCH", "WATCH", "HOLD", "KILL")
        },
        "verdicts": verdicts,
        "tiers": {
            "vs_spy_plus_3pp": spy_rows,
            "vs_best_plus_3pp": best_rows,
            "vs_qqq_plus_3pp": qqq_rows,
        },
        "diagnosis": {
            "strict_best_every_year": 0,
            "note": (
                "Strict max(index)+3pp every year is near-impossible without leverage "
                "because QQQ is best index in most bull years; QQQ hold cannot beat QQQ+3pp."
            ),
        },
        "disclaimer": "Offline research rescore. VIRTUAL. Not financial advice.",
    }

    out_md = Path(args.out) if args.out else default_md
    out_json = out_md.with_suffix(".json")
    md = to_markdown(
        payload=payload,
        verdicts=verdicts,
        spy_rows=spy_rows,
        best_rows=best_rows,
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(
        f"Wrote {out_md} | PROMOTE={payload['counts']['PROMOTE_RESEARCH']} "
        f"WATCH={payload['counts']['WATCH']} HOLD={payload['counts']['HOLD']} "
        f"KILL={payload['counts']['KILL']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
