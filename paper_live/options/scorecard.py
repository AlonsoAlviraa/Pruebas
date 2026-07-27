"""Decision scorecard for options multi-window matrix (promote / watch / kill).

Reads matrix ``summary.json`` shape from ``run_options_ta_matrix``.
VIRTUAL research only — never claims exchange fills.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "options-scorecard-v1",
    "index_underlyings": ["SPY", "QQQ", "IWM", "DIA"],
    "bull_windows": ["2023", "2024", "2025_study"],
    "bear_windows": ["2022_bear"],
    "stress_window": "stress_primary",
    "cash_strategy_ids": ["OPT_TA12_cash", "OPT_NAME_cash", "cash"],
    "thresholds": {
        "max_dd_kill": 0.25,
        "stress_max_dd_promote": 0.20,
        "min_windows_vs_cash_promote": 3,
        "min_bull_worse_than_cash_for_kill": 2,
        "min_opens_for_verdict": 0,
    },
    "priority": ["KILL", "PROMOTE_RESEARCH", "WATCH", "HOLD"],
}


@dataclass
class WindowMetrics:
    name: str
    total_return: Optional[float] = None
    max_dd: Optional[float] = None
    cvar_5pct: Optional[float] = None
    vs_spy_bh: Optional[float] = None
    n_opens: int = 0
    n_tp: int = 0
    n_sl: int = 0
    n_time_exit: int = 0
    n_dte_rolls: int = 0
    hard_kill: bool = False
    defined_risk: bool = False


@dataclass
class StrategyScore:
    strategy_id: str
    label: str = ""
    kind: str = ""
    underlying: str = ""
    segment: str = "index"  # index | single_name
    defined_risk: bool = False
    windows: Dict[str, WindowMetrics] = field(default_factory=dict)
    mean_return: Optional[float] = None
    worst_return: Optional[float] = None
    worst_max_dd: Optional[float] = None
    mean_cvar: Optional[float] = None
    mean_vs_spy: Optional[float] = None
    hard_kill_count: int = 0
    total_opens: int = 0
    total_tp: int = 0
    total_sl: int = 0
    total_time_exit: int = 0
    stress_return: Optional[float] = None
    stress_max_dd: Optional[float] = None
    cash_return_by_window: Dict[str, Optional[float]] = field(default_factory=dict)
    verdict: str = "HOLD"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["windows"] = {k: asdict(v) for k, v in self.windows.items()}
        return d


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None and path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg.update({k: v for k, v in raw.items() if k != "thresholds"})
        if "thresholds" in raw:
            th = dict(cfg.get("thresholds") or {})
            th.update(raw["thresholds"] or {})
            cfg["thresholds"] = th
    return cfg


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def segment_for(underlying: str, index_underlyings: Sequence[str]) -> str:
    u = (underlying or "").upper()
    if u in {x.upper() for x in index_underlyings}:
        return "index"
    return "single_name"


def _cash_returns(
    windows: Sequence[Mapping[str, Any]],
    stress: Optional[Mapping[str, Any]],
    cash_ids: Sequence[str],
) -> Dict[str, Optional[float]]:
    """Map window name → cash strategy total_return (0.0 if no cash id found)."""
    out: Dict[str, Optional[float]] = {}
    cash_set = {str(x) for x in cash_ids}

    def from_strats(name: str, strats: Sequence[Mapping[str, Any]]) -> None:
        for s in strats:
            sid = str(s.get("strategy_id") or s.get("id") or "")
            if sid in cash_set or str(s.get("kind") or "") == "cash":
                out[name] = _f(s.get("total_return"))
                return
        # default cash ≈ 0 if missing
        if name not in out:
            out[name] = 0.0

    for w in windows:
        name = str(w.get("name") or "")
        from_strats(name, w.get("strategies") or [])
    if stress:
        from_strats(str(stress.get("name") or "stress_primary"), stress.get("strategies") or [])
    return out


def collect_strategy_rows(
    summary: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> List[StrategyScore]:
    """Flatten multi-window summary into per-strategy scores (no verdict yet)."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
        if "thresholds" in (config or {}):
            th = dict(DEFAULT_CONFIG["thresholds"])
            th.update(config["thresholds"] or {})  # type: ignore[index]
            cfg["thresholds"] = th

    windows: List[Mapping[str, Any]] = list(summary.get("windows") or [])
    stress = summary.get("stress")
    cash_map = _cash_returns(windows, stress if isinstance(stress, Mapping) else None, cfg["cash_strategy_ids"])
    index_u = list(cfg.get("index_underlyings") or DEFAULT_CONFIG["index_underlyings"])

    by_id: Dict[str, StrategyScore] = {}

    def ingest(name: str, strats: Sequence[Mapping[str, Any]], *, is_stress: bool = False) -> None:
        for s in strats:
            sid = str(s.get("strategy_id") or s.get("id") or "")
            if not sid:
                continue
            if sid not in by_id:
                und = str(s.get("underlying") or "")
                by_id[sid] = StrategyScore(
                    strategy_id=sid,
                    label=str(s.get("label") or sid),
                    kind=str(s.get("kind") or ""),
                    underlying=und,
                    segment=segment_for(und, index_u),
                    defined_risk=bool(s.get("defined_risk")),
                )
            sc = by_id[sid]
            sc.defined_risk = sc.defined_risk or bool(s.get("defined_risk"))
            wm = WindowMetrics(
                name=name,
                total_return=_f(s.get("total_return")),
                max_dd=_f(s.get("max_dd")),
                cvar_5pct=_f(s.get("cvar_5pct")),
                vs_spy_bh=_f(s.get("vs_spy_bh")),
                n_opens=int(s.get("n_opens") or s.get("n_rolls") or 0),
                n_tp=int(s.get("n_tp") or 0),
                n_sl=int(s.get("n_sl") or 0),
                n_time_exit=int(s.get("n_time_exit") or 0),
                n_dte_rolls=int(s.get("n_dte_rolls") or 0),
                hard_kill=bool(s.get("hard_kill")),
                defined_risk=bool(s.get("defined_risk")),
            )
            sc.windows[name] = wm
            if is_stress:
                sc.stress_return = wm.total_return
                sc.stress_max_dd = wm.max_dd
            if wm.hard_kill:
                sc.hard_kill_count += 1
            sc.total_opens += wm.n_opens
            sc.total_tp += wm.n_tp
            sc.total_sl += wm.n_sl
            sc.total_time_exit += wm.n_time_exit

    for w in windows:
        ingest(str(w.get("name") or ""), w.get("strategies") or [], is_stress=False)
    if isinstance(stress, Mapping):
        ingest(str(stress.get("name") or "stress_primary"), stress.get("strategies") or [], is_stress=True)

    for sc in by_id.values():
        sc.cash_return_by_window = dict(cash_map)
        rets = [wm.total_return for wm in sc.windows.values() if wm.total_return is not None and not wm.name.startswith("stress")]
        dds = [wm.max_dd for wm in sc.windows.values() if wm.max_dd is not None and not wm.name.startswith("stress")]
        cvars = [wm.cvar_5pct for wm in sc.windows.values() if wm.cvar_5pct is not None and not wm.name.startswith("stress")]
        vs = [wm.vs_spy_bh for wm in sc.windows.values() if wm.vs_spy_bh is not None and not wm.name.startswith("stress")]
        if rets:
            sc.mean_return = sum(rets) / len(rets)
            sc.worst_return = min(rets)
        if dds:
            sc.worst_max_dd = min(dds)  # more negative is worse
        if cvars:
            sc.mean_cvar = sum(cvars) / len(cvars)
        if vs:
            sc.mean_vs_spy = sum(vs) / len(vs)

    return sorted(by_id.values(), key=lambda x: x.strategy_id)


def decide_verdict(sc: StrategyScore, config: Optional[Mapping[str, Any]] = None) -> StrategyScore:
    """Apply promote / watch / kill rules; mutates and returns ``sc``."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
        if "thresholds" in (config or {}):
            th = dict(DEFAULT_CONFIG["thresholds"])
            th.update(config["thresholds"] or {})  # type: ignore[index]
            cfg["thresholds"] = th
    th = cfg["thresholds"]
    bull = list(cfg.get("bull_windows") or [])
    bear = list(cfg.get("bear_windows") or [])
    reasons: List[str] = []

    # Cash baseline: skip kill of cash control itself
    if sc.kind == "cash" or sc.strategy_id in set(cfg.get("cash_strategy_ids") or []):
        sc.verdict = "HOLD"
        sc.reasons = ["cash_control_benchmark"]
        return sc

    def ret_in(name: str) -> Optional[float]:
        wm = sc.windows.get(name)
        return wm.total_return if wm else None

    def cash_in(name: str) -> float:
        v = sc.cash_return_by_window.get(name)
        return float(v) if v is not None else 0.0

    # --- KILL rules ---
    kill = False
    max_dd_kill = float(th.get("max_dd_kill", 0.25))
    if sc.worst_max_dd is not None and sc.worst_max_dd <= -abs(max_dd_kill):
        kill = True
        reasons.append(f"worst_max_dd={sc.worst_max_dd:.2%} <= -{max_dd_kill:.0%}")

    min_bull_bad = int(th.get("min_bull_worse_than_cash_for_kill", 2))
    bull_worse = 0
    for b in bull:
        r = ret_in(b)
        if r is not None and r < cash_in(b):
            bull_worse += 1
    stress_ret = sc.stress_return
    stress_cash = cash_in(str(cfg.get("stress_window") or "stress_primary"))
    stress_worse = stress_ret is not None and stress_ret < stress_cash

    if stress_worse and bull_worse >= min_bull_bad:
        kill = True
        reasons.append(
            f"stress_ret={stress_ret:.2%} < cash and bull_worse_than_cash={bull_worse}>={min_bull_bad}"
        )

    if sc.hard_kill_count >= 2:
        kill = True
        reasons.append(f"hard_kill_count={sc.hard_kill_count}")

    if kill:
        sc.verdict = "KILL"
        sc.reasons = reasons or ["kill_rule"]
        return sc

    # --- PROMOTE_RESEARCH ---
    min_win = int(th.get("min_windows_vs_cash_promote", 3))
    stress_dd_cap = float(th.get("stress_max_dd_promote", 0.20))
    calendar_names = [n for n in sc.windows if not str(n).startswith("stress")]
    beat_cash = 0
    for n in calendar_names:
        r = ret_in(n)
        if r is not None and r > cash_in(n):
            beat_cash += 1
    stress_dd_ok = (
        sc.stress_max_dd is None
        or sc.stress_max_dd > -abs(stress_dd_cap)
    )
    if sc.defined_risk and beat_cash >= min_win and stress_dd_ok:
        reasons.append(
            f"defined_risk beat_cash={beat_cash}/{len(calendar_names)} stress_dd_ok"
        )
        sc.verdict = "PROMOTE_RESEARCH"
        sc.reasons = reasons
        return sc

    # --- WATCH: income / risk-profile (bear help, stress ok, loses vs SPY in bull) ---
    bear_ok = False
    for b in bear:
        r = ret_in(b)
        if r is not None and r >= cash_in(b):
            bear_ok = True
    stress_not_explode = stress_dd_ok and (stress_ret is None or stress_ret >= stress_cash - 0.05)
    loses_vs_spy_bull = False
    for b in bull:
        wm = sc.windows.get(b)
        if wm and wm.vs_spy_bh is not None and wm.vs_spy_bh < -0.02:
            loses_vs_spy_bull = True
            break
    if bear_ok and stress_not_explode and (loses_vs_spy_bull or (sc.mean_vs_spy is not None and sc.mean_vs_spy < 0)):
        reasons.append("bear_ok + stress_contained + lag_SPY_bull (income profile)")
        sc.verdict = "WATCH"
        sc.reasons = reasons
        return sc

    # Mild positive vs cash but not promote → WATCH
    if beat_cash >= 2 and stress_dd_ok:
        reasons.append(f"partial_edge beat_cash={beat_cash} not enough for promote")
        sc.verdict = "WATCH"
        sc.reasons = reasons
        return sc

    sc.verdict = "HOLD"
    sc.reasons = reasons or ["no_strong_signal"]
    return sc


def score_matrix(
    summary: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Full scorecard payload."""
    cfg = load_config(None)
    if config:
        cfg.update(config)
        if "thresholds" in (config or {}):
            th = dict(DEFAULT_CONFIG["thresholds"])
            th.update(config["thresholds"] or {})  # type: ignore[index]
            cfg["thresholds"] = th

    rows = collect_strategy_rows(summary, config=cfg)
    scored = [decide_verdict(r, cfg) for r in rows]

    by_verdict: Dict[str, List[str]] = {"KILL": [], "PROMOTE_RESEARCH": [], "WATCH": [], "HOLD": []}
    for s in scored:
        by_verdict.setdefault(s.verdict, []).append(s.strategy_id)

    segments = {
        "index": [s.strategy_id for s in scored if s.segment == "index"],
        "single_name": [s.strategy_id for s in scored if s.segment == "single_name"],
    }

    return {
        "version": cfg.get("version", "options-scorecard-v1"),
        "as_of": summary.get("as_of"),
        "zoo": summary.get("zoo"),
        "names_zoo": summary.get("names_zoo"),
        "config": {
            "bull_windows": cfg.get("bull_windows"),
            "bear_windows": cfg.get("bear_windows"),
            "thresholds": cfg.get("thresholds"),
        },
        "counts": {k: len(v) for k, v in by_verdict.items()},
        "by_verdict": by_verdict,
        "segments": segments,
        "strategies": [s.to_dict() for s in scored],
        "disclaimer": (
            "Research scorecard on model BS marks (proxy_bs|vix_surface). "
            "Not OPRA fills. VIRTUAL capital. Verdicts are research triage, not trade advice."
        ),
    }


def scorecard_to_markdown(payload: Mapping[str, Any]) -> str:
    """One-page human decision table."""
    lines = [
        f"# Options matrix SCORECARD — `{payload.get('as_of') or 'n/a'}`",
        "",
        f"**Zoo:** `{payload.get('zoo')}`"
        + (f" + names `{payload.get('names_zoo')}`" if payload.get("names_zoo") else ""),
        "",
        "## Verdict counts",
        "",
        "| Verdict | N |",
        "|---------|---|",
    ]
    counts = payload.get("counts") or {}
    for k in ("PROMOTE_RESEARCH", "WATCH", "HOLD", "KILL"):
        lines.append(f"| **{k}** | {counts.get(k, 0)} |")

    lines += [
        "",
        "## Decision table",
        "",
        "| Verdict | ID | Seg | Kind | Und | MeanRet | WorstRet | WorstDD | StressRet | Opens | TP/SL/TE | vsSPY | Reasons |",
        "|---------|----|-----|------|-----|---------|----------|---------|-----------|-------|----------|-------|---------|",
    ]
    order = {"PROMOTE_RESEARCH": 0, "WATCH": 1, "HOLD": 2, "KILL": 3}
    strats = sorted(
        payload.get("strategies") or [],
        key=lambda s: (order.get(str(s.get("verdict")), 9), s.get("strategy_id") or ""),
    )
    for s in strats:
        mean_r = s.get("mean_return")
        worst_r = s.get("worst_return")
        worst_dd = s.get("worst_max_dd")
        stress_r = s.get("stress_return")
        vs = s.get("mean_vs_spy")
        tpsl = f"{s.get('total_tp', 0)}/{s.get('total_sl', 0)}/{s.get('total_time_exit', 0)}"
        reasons = "; ".join(s.get("reasons") or [])[:80]
        lines.append(
            f"| {s.get('verdict')} | `{s.get('strategy_id')}` | {s.get('segment')} | "
            f"{s.get('kind')} | {s.get('underlying')} | "
            f"{_pct(mean_r)} | {_pct(worst_r)} | {_pct(worst_dd)} | {_pct(stress_r)} | "
            f"{s.get('total_opens', 0)} | {tpsl} | {_pct(vs)} | {reasons} |"
        )

    by_v = payload.get("by_verdict") or {}
    lines += ["", "## Kill list", ""]
    kills = by_v.get("KILL") or []
    if kills:
        for kid in kills:
            lines.append(f"- `{kid}`")
    else:
        lines.append("_None_")

    lines += ["", "## Promote (research)", ""]
    prom = by_v.get("PROMOTE_RESEARCH") or []
    if prom:
        for pid in prom:
            lines.append(f"- `{pid}`")
    else:
        lines.append("_None — tighten gates or accept income WATCH sleeves_")

    lines += [
        "",
        "---",
        str(payload.get("disclaimer") or ""),
        "",
    ]
    return "\n".join(lines)


def _pct(x: Any) -> str:
    v = _f(x)
    if v is None:
        return "n/a"
    return f"{v:.2%}"


def write_scorecard(
    summary_path: Path,
    *,
    out_md: Path,
    out_json: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load summary.json, score, write SCORECARD.md (+ optional json)."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    cfg = load_config(config_path)
    payload = score_matrix(summary, config=cfg)
    md = scorecard_to_markdown(payload)
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    jpath = out_json or out_md.with_suffix(".json")
    Path(jpath).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["_paths"] = {"md": str(out_md), "json": str(jpath)}
    return payload
