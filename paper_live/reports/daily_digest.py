"""Daily paper digest from ledger (post-close metrics)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from paper_live.ledger import EventType, PaperLedger
from paper_live.ledger.events import utc_now


@dataclass
class DailyDigest:
    run_id: str
    day: str
    strategy_id: str
    mode: str = "paper"
    capital_label: str = "VIRTUAL"
    equity: Optional[float] = None
    cash: Optional[float] = None
    gross_exposure: Optional[float] = None
    dd_from_peak: Optional[float] = None
    n_positions: int = 0
    n_fills: int = 0
    n_buys: int = 0
    n_sells: int = 0
    commission: float = 0.0
    fees: float = 0.0
    slippage_est: float = 0.0
    turnover: float = 0.0
    n_orders: int = 0
    n_rejects: int = 0
    n_kill_events: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Paper daily digest — `{self.day}`",
            "",
            f"**Run:** `{self.run_id}` · **strategy:** `{self.strategy_id}` · "
            f"**mode:** {self.mode} · **capital:** {self.capital_label}",
            "",
            "## Portfolio",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Equity | {_fmt_money(self.equity)} |",
            f"| Cash | {_fmt_money(self.cash)} |",
            f"| Gross exposure | {_fmt_money(self.gross_exposure)} |",
            f"| DD from peak | {_fmt_pct(self.dd_from_peak)} |",
            f"| Open positions | {self.n_positions} |",
            "",
            "## Activity",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Fills | {self.n_fills} (buys {self.n_buys} / sells {self.n_sells}) |",
            f"| Orders | {self.n_orders} |",
            f"| Rejects | {self.n_rejects} |",
            f"| Kill events | {self.n_kill_events} |",
            f"| Commission | {_fmt_money(self.commission)} |",
            f"| Fees | {_fmt_money(self.fees)} |",
            f"| Slippage est. | {_fmt_money(self.slippage_est)} |",
            f"| Turnover (notional proxy) | {_fmt_money(self.turnover)} |",
            "",
        ]
        if self.reject_reasons:
            lines += ["### Reject reasons", ""]
            for k, v in sorted(self.reject_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"- `{k}`: {v}")
            lines.append("")
        if self.fills:
            lines += [
                "### Fills",
                "",
                "| Time | Ticker | Side | Qty | Price | Comm | Fees |",
                "|------|--------|------|-----|-------|------|------|",
            ]
            for f in self.fills[:50]:
                lines.append(
                    f"| {str(f.get('ts', ''))[:19]} | {f.get('ticker')} | {f.get('side')} | "
                    f"{f.get('qty')} | {f.get('price'):.4f} | {f.get('commission', 0):.2f} | "
                    f"{f.get('fees', 0):.4f} |"
                )
            lines.append("")
        if self.positions:
            lines += [
                "### Open positions (EOD)",
                "",
                "| Ticker | Qty | Avg px | Stop |",
                "|--------|-----|--------|------|",
            ]
            for p in self.positions:
                lines.append(
                    f"| {p.get('ticker')} | {p.get('qty')} | {p.get('avg_px')} | {p.get('stop')} |"
                )
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""]
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        lines += [
            "---",
            f"_Generated {self.generated_at} · research paper only · not financial advice._",
            "",
        ]
        return "\n".join(lines)


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"${x:,.2f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.2%}"


def build_daily_digest(ledger: PaperLedger, day: Union[str, date]) -> DailyDigest:
    """Aggregate one calendar day from SQLite ledger tables + events."""
    d = day if isinstance(day, str) else day.isoformat()
    d = d[:10]
    run = ledger.get_run()
    nav_rows = ledger.list_nav(day=d)
    cost_rows = ledger.list_costs(day=d)
    fills = ledger.list_fills(day=d)
    orders = ledger.list_orders(day=d)
    decisions = ledger.list_decisions(day=d)

    nav = nav_rows[-1] if nav_rows else {}
    cost = cost_rows[-1] if cost_rows else {}

    n_buys = sum(1 for f in fills if str(f.get("side", "")).lower() == "buy")
    n_sells = sum(1 for f in fills if str(f.get("side", "")).lower() == "sell")
    turnover = sum(abs(float(f.get("qty") or 0) * float(f.get("price") or 0)) for f in fills)

    # costs: prefer costs_daily; else sum fills
    commission = float(cost.get("commission") or 0.0)
    fees = float(cost.get("fees") or 0.0)
    slip = float(cost.get("slippage_est") or 0.0)
    if not cost_rows and fills:
        commission = sum(float(f.get("commission") or 0) for f in fills)
        fees = sum(float(f.get("fees") or 0) for f in fills)
        slip = 0.0
        for f in fills:
            meta = f.get("meta_json")
            if isinstance(meta, str) and meta:
                try:
                    m = json.loads(meta)
                    slip += float(m.get("slippage_cost") or 0)
                except Exception:
                    pass

    reject_reasons: Dict[str, int] = {}
    n_rejects = 0
    for o in orders:
        st = str(o.get("status") or "").lower()
        if st in ("rejected", "cancelled"):
            n_rejects += 1
            reason = str(o.get("reason") or "unknown").split(":")[0]
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    for dec in decisions:
        if str(dec.get("action") or "").lower() in ("reject", "skip", "blocked"):
            n_rejects += 1
            filt = dec.get("filters") or {}
            reason = "decision_reject"
            if isinstance(filt, dict):
                conf = filt.get("confirm") or {}
                if isinstance(conf, dict) and conf.get("reason"):
                    reason = str(conf["reason"])
                elif filt.get("reason"):
                    reason = str(filt["reason"])
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    n_kill = ledger.count_events(EventType.KILL_SWITCH, day=d)
    positions = ledger.get_positions()

    notes: List[str] = []
    if n_kill:
        notes.append(f"Kill switch events today: {n_kill}")
    if nav and float(nav.get("dd_from_peak") or 0) <= -0.09:
        notes.append("Soft de-risk zone: DD from peak ≥ 9%")
    if nav and float(nav.get("dd_from_peak") or 0) <= -0.18:
        notes.append("HARD DD zone: peak DD ≥ 18% — entries should be blocked")
    if not fills and not nav_rows:
        notes.append("No NAV/fills for this day — empty or non-session date")

    return DailyDigest(
        run_id=ledger.run_id,
        day=d,
        strategy_id=str(run.get("strategy") or ledger.strategy_id),
        equity=float(nav["equity"]) if nav.get("equity") is not None else None,
        cash=float(nav["cash"]) if nav.get("cash") is not None else None,
        gross_exposure=float(nav["gross_exposure"]) if nav.get("gross_exposure") is not None else None,
        dd_from_peak=float(nav["dd_from_peak"]) if nav.get("dd_from_peak") is not None else None,
        n_positions=int(nav.get("n_positions") or len(positions)),
        n_fills=len(fills),
        n_buys=n_buys,
        n_sells=n_sells,
        commission=float(commission),
        fees=float(fees),
        slippage_est=float(slip),
        turnover=float(cost.get("turnover") or turnover),
        n_orders=len(orders),
        n_rejects=n_rejects,
        n_kill_events=n_kill,
        reject_reasons=reject_reasons,
        fills=[
            {
                "ts": f.get("ts"),
                "ticker": f.get("ticker"),
                "side": f.get("side"),
                "qty": f.get("qty"),
                "price": float(f.get("price") or 0),
                "commission": float(f.get("commission") or 0),
                "fees": float(f.get("fees") or 0),
            }
            for f in fills
        ],
        positions=positions,
        notes=notes,
        generated_at=utc_now().isoformat(),
    )


def write_daily_digest(
    digest: DailyDigest,
    out_dir: Union[str, Path],
    *,
    write_json: bool = True,
    write_md: bool = True,
) -> Dict[str, Path]:
    """Write digest JSON/MD under out_dir/daily/YYYY-MM-DD.*"""
    out = Path(out_dir)
    daily = out / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    if write_json:
        p = daily / f"{digest.day}.json"
        p.write_text(json.dumps(digest.to_dict(), indent=2, default=str), encoding="utf-8")
        paths["json"] = p
    if write_md:
        p = daily / f"{digest.day}.md"
        p.write_text(digest.to_markdown(), encoding="utf-8")
        paths["md"] = p
    return paths
