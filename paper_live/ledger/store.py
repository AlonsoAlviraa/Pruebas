"""PaperLedger: SQLite operational store + append-only JSONL audit log."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from paper_live.freeze import PaperFreeze, assert_paper_only
from paper_live.ledger.events import (
    EVENT_TYPES,
    EventType,
    new_decision_id,
    new_event_id,
    new_fill_id,
    new_order_id,
    new_run_id,
    utc_now,
)

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode = 'paper'),
    capital0 REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    status TEXT NOT NULL DEFAULT 'active',
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    strategy_id TEXT,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(run_id, event_type);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    order_type TEXT NOT NULL,
    limit_px REAL,
    status TEXT NOT NULL,
    reason TEXT,
    meta_json TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id, ts);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL DEFAULT 0,
    fees REAL NOT NULL DEFAULT 0,
    slippage_bps REAL NOT NULL DEFAULT 0,
    liquidity TEXT,
    meta_json TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_fills_run ON fills(run_id, ts);

CREATE TABLE IF NOT EXISTS positions (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    qty REAL NOT NULL,
    avg_px REAL NOT NULL,
    stop REAL,
    hard_stop REAL,
    opened_at TEXT,
    bars_held INTEGER DEFAULT 0,
    meta_json TEXT,
    PRIMARY KEY (run_id, ticker),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS nav_daily (
    run_id TEXT NOT NULL,
    date TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    gross_exposure REAL NOT NULL DEFAULT 0,
    dd_from_peak REAL NOT NULL DEFAULT 0,
    n_positions INTEGER NOT NULL DEFAULT 0,
    peak_equity REAL,
    PRIMARY KEY (run_id, date),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    p_buy REAL,
    score REAL,
    filters_json TEXT,
    config_hash TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id, ts);

CREATE TABLE IF NOT EXISTS costs_daily (
    run_id TEXT NOT NULL,
    date TEXT NOT NULL,
    commission REAL NOT NULL DEFAULT 0,
    fees REAL NOT NULL DEFAULT 0,
    slippage_est REAL NOT NULL DEFAULT 0,
    turnover REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, date),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


def _iso(ts: Optional[datetime] = None) -> str:
    t = ts or utc_now()
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    else:
        t = t.astimezone(timezone.utc)
    return t.isoformat()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, default=str, sort_keys=True)


@dataclass
class PaperLedger:
    """Dual-write ledger: SQLite (query) + JSONL (immutable audit).

    JSONL files are append-only under ``audit/YYYY-MM-DD.jsonl``.
    Closed-day files must not be rewritten; use event_type=correction.
    """

    root: Path
    run_id: str
    config_hash: str
    strategy_id: str
    db_path: Path
    audit_dir: Path
    snapshots_dir: Path
    _conn: sqlite3.Connection

    @classmethod
    def create_run(
        cls,
        root: Union[str, Path],
        freeze: PaperFreeze,
        *,
        run_id: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> "PaperLedger":
        """Initialize a new paper run under root (creates dirs + DB row)."""
        assert_paper_only(require_env=False)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        db_path = root / "paper_year.db"
        audit_dir = root / "audit"
        snapshots_dir = root / "snapshots"
        audit_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        rid = run_id or new_run_id()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        started = _iso()
        try:
            conn.execute(
                """
                INSERT INTO runs(run_id, started_at, strategy, config_hash, mode, capital0, currency, status, meta_json)
                VALUES (?, ?, ?, ?, 'paper', ?, ?, 'active', ?)
                """,
                (
                    rid,
                    started,
                    freeze.strategy.strategy_id,
                    freeze.config_hash,
                    float(freeze.strategy.capital0),
                    freeze.strategy.currency,
                    _json_dumps(dict(meta or {})),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.close()
            raise ValueError(f"run_id already exists: {rid}") from e

        ledger = cls(
            root=root,
            run_id=rid,
            config_hash=freeze.config_hash,
            strategy_id=freeze.strategy.strategy_id,
            db_path=db_path,
            audit_dir=audit_dir,
            snapshots_dir=snapshots_dir,
            _conn=conn,
        )
        # Persist freeze snapshot with the run (immutable reference)
        freeze_path = root / f"freeze_{rid}.json"
        freeze_path.write_text(
            _json_dumps(freeze.to_public_dict()),
            encoding="utf-8",
        )
        ledger.append_event(
            EventType.RUN_INIT,
            {
                "run_id": rid,
                "strategy_id": freeze.strategy.strategy_id,
                "config_hash": freeze.config_hash,
                "capital0": freeze.strategy.capital0,
                "mode": "paper",
                "freeze_path": str(freeze_path.name),
            },
            ts=datetime.fromisoformat(started),
        )
        return ledger

    @classmethod
    def open_run(cls, root: Union[str, Path], run_id: str) -> "PaperLedger":
        """Re-open an existing run (crash recovery)."""
        assert_paper_only(require_env=False)
        root = Path(root)
        db_path = root / "paper_year.db"
        if not db_path.is_file():
            raise FileNotFoundError(f"No ledger DB at {db_path}")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_SQL)
        row = conn.execute(
            "SELECT run_id, strategy, config_hash, mode FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            conn.close()
            raise KeyError(f"Unknown run_id: {run_id}")
        if row["mode"] != "paper":
            conn.close()
            raise RuntimeError(f"Refusing non-paper run mode={row['mode']!r}")
        return cls(
            root=root,
            run_id=row["run_id"],
            config_hash=row["config_hash"],
            strategy_id=row["strategy"],
            db_path=db_path,
            audit_dir=root / "audit",
            snapshots_dir=root / "snapshots",
            _conn=conn,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PaperLedger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # --- core event log ---

    def append_event(
        self,
        event_type: Union[EventType, str],
        payload: Optional[Mapping[str, Any]] = None,
        *,
        ts: Optional[datetime] = None,
        strategy_id: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> str:
        et = event_type.value if isinstance(event_type, EventType) else str(event_type)
        if et not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {et!r}. Known: {sorted(EVENT_TYPES)}")
        eid = event_id or new_event_id()
        ts_s = _iso(ts)
        payload_d = dict(payload or {})
        payload_d.setdefault("mode", "paper")
        row = {
            "event_id": eid,
            "run_id": self.run_id,
            "ts": ts_s,
            "event_type": et,
            "strategy_id": strategy_id or self.strategy_id,
            "payload": payload_d,
        }
        self._conn.execute(
            """
            INSERT INTO events(event_id, run_id, ts, event_type, strategy_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eid,
                self.run_id,
                ts_s,
                et,
                row["strategy_id"],
                _json_dumps(payload_d),
            ),
        )
        self._conn.commit()
        self._append_jsonl(row)
        return eid

    def _append_jsonl(self, row: Mapping[str, Any]) -> None:
        """Append one audit line. Never truncates the file."""
        day = str(row["ts"])[:10]
        path = self.audit_dir / f"{day}.jsonl"
        line = _json_dumps(row) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    def list_events(
        self,
        *,
        event_type: Optional[Union[EventType, str]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        et = None
        if event_type is not None:
            et = event_type.value if isinstance(event_type, EventType) else str(event_type)
        if et:
            rows = self._conn.execute(
                """
                SELECT event_id, run_id, ts, event_type, strategy_id, payload_json
                FROM events WHERE run_id = ? AND event_type = ?
                ORDER BY ts ASC LIMIT ?
                """,
                (self.run_id, et, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT event_id, run_id, ts, event_type, strategy_id, payload_json
                FROM events WHERE run_id = ?
                ORDER BY ts ASC LIMIT ?
                """,
                (self.run_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "event_id": r["event_id"],
                    "run_id": r["run_id"],
                    "ts": r["ts"],
                    "event_type": r["event_type"],
                    "strategy_id": r["strategy_id"],
                    "payload": json.loads(r["payload_json"]),
                }
            )
        return out

    # --- domain helpers ---

    def record_order(
        self,
        *,
        ticker: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_px: Optional[float] = None,
        status: str = "submitted",
        reason: Optional[str] = None,
        order_id: Optional[str] = None,
        ts: Optional[datetime] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> str:
        oid = order_id or new_order_id()
        ts_s = _iso(ts)
        self._conn.execute(
            """
            INSERT INTO orders(order_id, run_id, ts, ticker, side, qty, order_type, limit_px, status, reason, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                oid,
                self.run_id,
                ts_s,
                ticker.upper(),
                side.lower(),
                float(qty),
                order_type,
                limit_px,
                status,
                reason,
                _json_dumps(dict(meta or {})),
            ),
        )
        self._conn.commit()
        self.append_event(
            EventType.ORDER_SUBMITTED,
            {
                "order_id": oid,
                "ticker": ticker.upper(),
                "side": side.lower(),
                "qty": float(qty),
                "order_type": order_type,
                "limit_px": limit_px,
                "status": status,
                "reason": reason,
            },
            ts=ts,
        )
        return oid

    def update_order_status(
        self,
        order_id: str,
        status: str,
        *,
        reason: Optional[str] = None,
        event: Optional[Union[EventType, str]] = None,
        ts: Optional[datetime] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Update order row status and optionally emit ack/reject/cancel event."""
        self._conn.execute(
            "UPDATE orders SET status = ?, reason = COALESCE(?, reason) WHERE order_id = ? AND run_id = ?",
            (status, reason, order_id, self.run_id),
        )
        self._conn.commit()
        if event is not None:
            self.append_event(
                event,
                {
                    "order_id": order_id,
                    "status": status,
                    "reason": reason,
                    **dict(meta or {}),
                },
                ts=ts,
            )

    def record_fill(
        self,
        *,
        order_id: str,
        ticker: str,
        side: str,
        qty: float,
        price: float,
        commission: float = 0.0,
        fees: float = 0.0,
        slippage_bps: float = 0.0,
        liquidity: Optional[str] = None,
        fill_id: Optional[str] = None,
        ts: Optional[datetime] = None,
        meta: Optional[Mapping[str, Any]] = None,
        order_status: str = "filled",
    ) -> str:
        fid = fill_id or new_fill_id()
        ts_s = _iso(ts)
        self._conn.execute(
            """
            INSERT INTO fills(fill_id, order_id, run_id, ts, ticker, side, qty, price,
                              commission, fees, slippage_bps, liquidity, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fid,
                order_id,
                self.run_id,
                ts_s,
                ticker.upper(),
                side.lower(),
                float(qty),
                float(price),
                float(commission),
                float(fees),
                float(slippage_bps),
                liquidity,
                _json_dumps(dict(meta or {})),
            ),
        )
        # update order status lightly (partial fills use order_status="partial")
        self._conn.execute(
            "UPDATE orders SET status = ? WHERE order_id = ? AND run_id = ?",
            (order_status, order_id, self.run_id),
        )
        self._conn.commit()
        gross = float(qty) * float(price)
        self.append_event(
            EventType.FILL,
            {
                "fill_id": fid,
                "order_id": order_id,
                "ticker": ticker.upper(),
                "side": side.lower(),
                "qty": float(qty),
                "price": float(price),
                "commission": float(commission),
                "fees": float(fees),
                "slippage_bps": float(slippage_bps),
                "liquidity": liquidity,
                "gross_notional": gross,
                "net_cost": float(commission) + float(fees),
                "order_status": order_status,
                **dict(meta or {}),
            },
            ts=ts,
        )
        return fid

    def upsert_position(
        self,
        *,
        ticker: str,
        qty: float,
        avg_px: float,
        stop: Optional[float] = None,
        hard_stop: Optional[float] = None,
        opened_at: Optional[datetime] = None,
        bars_held: int = 0,
        meta: Optional[Mapping[str, Any]] = None,
        event: Optional[EventType] = None,
    ) -> None:
        t = ticker.upper()
        opened_s = _iso(opened_at) if opened_at else None
        self._conn.execute(
            """
            INSERT INTO positions(run_id, ticker, qty, avg_px, stop, hard_stop, opened_at, bars_held, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, ticker) DO UPDATE SET
                qty=excluded.qty,
                avg_px=excluded.avg_px,
                stop=excluded.stop,
                hard_stop=excluded.hard_stop,
                opened_at=COALESCE(excluded.opened_at, positions.opened_at),
                bars_held=excluded.bars_held,
                meta_json=excluded.meta_json
            """,
            (
                self.run_id,
                t,
                float(qty),
                float(avg_px),
                stop,
                hard_stop,
                opened_s,
                int(bars_held),
                _json_dumps(dict(meta or {})),
            ),
        )
        self._conn.commit()
        et = event or (
            EventType.POSITION_OPENED if qty > 0 else EventType.POSITION_CLOSED
        )
        self.append_event(
            et,
            {
                "ticker": t,
                "qty": float(qty),
                "avg_px": float(avg_px),
                "stop": stop,
                "hard_stop": hard_stop,
                "bars_held": int(bars_held),
            },
        )

    def close_position(self, ticker: str, *, meta: Optional[Mapping[str, Any]] = None) -> None:
        t = ticker.upper()
        self._conn.execute(
            "DELETE FROM positions WHERE run_id = ? AND ticker = ?",
            (self.run_id, t),
        )
        self._conn.commit()
        self.append_event(
            EventType.POSITION_CLOSED,
            {"ticker": t, "qty": 0.0, **dict(meta or {})},
        )

    def get_positions(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE run_id = ? AND qty != 0",
            (self.run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_decision(
        self,
        *,
        ticker: str,
        action: str,
        p_buy: Optional[float] = None,
        score: Optional[float] = None,
        filters: Optional[Mapping[str, Any]] = None,
        decision_id: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> str:
        did = decision_id or new_decision_id()
        ts_s = _iso(ts)
        self._conn.execute(
            """
            INSERT INTO decisions(decision_id, run_id, ts, ticker, action, p_buy, score, filters_json, config_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                did,
                self.run_id,
                ts_s,
                ticker.upper(),
                action,
                p_buy,
                score,
                _json_dumps(dict(filters or {})),
                self.config_hash,
            ),
        )
        self._conn.commit()
        et = (
            EventType.ENTRY_CANDIDATE
            if action in ("enter", "candidate", "buy")
            else EventType.ENTRY_REJECTED
            if action in ("reject", "skip", "blocked")
            else EventType.SIGNAL_COMPUTED
        )
        self.append_event(
            et,
            {
                "decision_id": did,
                "ticker": ticker.upper(),
                "action": action,
                "p_buy": p_buy,
                "score": score,
                "filters": dict(filters or {}),
            },
            ts=ts,
        )
        return did

    def record_nav_daily(
        self,
        day: Union[str, date],
        *,
        equity: float,
        cash: float,
        gross_exposure: float = 0.0,
        dd_from_peak: float = 0.0,
        n_positions: int = 0,
        peak_equity: Optional[float] = None,
    ) -> None:
        d = day if isinstance(day, str) else day.isoformat()
        self._conn.execute(
            """
            INSERT INTO nav_daily(run_id, date, equity, cash, gross_exposure, dd_from_peak, n_positions, peak_equity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, date) DO UPDATE SET
                equity=excluded.equity,
                cash=excluded.cash,
                gross_exposure=excluded.gross_exposure,
                dd_from_peak=excluded.dd_from_peak,
                n_positions=excluded.n_positions,
                peak_equity=excluded.peak_equity
            """,
            (
                self.run_id,
                d,
                float(equity),
                float(cash),
                float(gross_exposure),
                float(dd_from_peak),
                int(n_positions),
                peak_equity,
            ),
        )
        self._conn.commit()
        self.append_event(
            EventType.DAILY_NAV,
            {
                "date": d,
                "equity": float(equity),
                "cash": float(cash),
                "gross_exposure": float(gross_exposure),
                "dd_from_peak": float(dd_from_peak),
                "n_positions": int(n_positions),
                "peak_equity": peak_equity,
                "capital_label": "VIRTUAL",
            },
        )

    def record_costs_daily(
        self,
        day: Union[str, date],
        *,
        commission: float = 0.0,
        fees: float = 0.0,
        slippage_est: float = 0.0,
        turnover: float = 0.0,
    ) -> None:
        d = day if isinstance(day, str) else day.isoformat()
        self._conn.execute(
            """
            INSERT INTO costs_daily(run_id, date, commission, fees, slippage_est, turnover)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, date) DO UPDATE SET
                commission=excluded.commission,
                fees=excluded.fees,
                slippage_est=excluded.slippage_est,
                turnover=excluded.turnover
            """,
            (
                self.run_id,
                d,
                float(commission),
                float(fees),
                float(slippage_est),
                float(turnover),
            ),
        )
        self._conn.commit()

    def write_snapshot(self, label: Optional[str] = None) -> Path:
        """Write end-of-day style portfolio snapshot JSON (recovery aid)."""
        ts = utc_now()
        name = label or ts.strftime("%Y%m%dT%H%M%SZ")
        path = self.snapshots_dir / f"snapshot_{self.run_id}_{name}.json"
        payload = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "strategy_id": self.strategy_id,
            "mode": "paper",
            "capital_label": "VIRTUAL",
            "ts": _iso(ts),
            "positions": self.get_positions(),
            "n_events": self._conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE run_id = ?", (self.run_id,)
            ).fetchone()["c"],
        }
        path.write_text(_json_dumps(payload), encoding="utf-8")
        self.append_event(EventType.SNAPSHOT, {"path": path.name, "n_positions": len(payload["positions"])})
        return path

    def get_run(self) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (self.run_id,)
        ).fetchone()
        return dict(row) if row else {}

    def sum_commissions(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(commission), 0) AS s FROM fills WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        return float(row["s"])

    def sum_fees(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(fees), 0) AS s FROM fills WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        return float(row["s"])

    def list_fills(
        self,
        *,
        day: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10_000,
    ) -> List[Dict[str, Any]]:
        """Fills for this run, optional day (YYYY-MM-DD) or [start, end] inclusive."""
        q = "SELECT * FROM fills WHERE run_id = ?"
        params: List[Any] = [self.run_id]
        if day:
            q += " AND substr(ts, 1, 10) = ?"
            params.append(str(day)[:10])
        else:
            if start:
                q += " AND substr(ts, 1, 10) >= ?"
                params.append(str(start)[:10])
            if end:
                q += " AND substr(ts, 1, 10) <= ?"
                params.append(str(end)[:10])
        q += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def list_orders(
        self,
        *,
        day: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10_000,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM orders WHERE run_id = ?"
        params: List[Any] = [self.run_id]
        if day:
            q += " AND substr(ts, 1, 10) = ?"
            params.append(str(day)[:10])
        else:
            if start:
                q += " AND substr(ts, 1, 10) >= ?"
                params.append(str(start)[:10])
            if end:
                q += " AND substr(ts, 1, 10) <= ?"
                params.append(str(end)[:10])
        q += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def list_nav(
        self,
        *,
        day: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM nav_daily WHERE run_id = ?"
        params: List[Any] = [self.run_id]
        if day:
            q += " AND date = ?"
            params.append(str(day)[:10])
        else:
            if start:
                q += " AND date >= ?"
                params.append(str(start)[:10])
            if end:
                q += " AND date <= ?"
                params.append(str(end)[:10])
        q += " ORDER BY date ASC"
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def list_costs(
        self,
        *,
        day: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM costs_daily WHERE run_id = ?"
        params: List[Any] = [self.run_id]
        if day:
            q += " AND date = ?"
            params.append(str(day)[:10])
        else:
            if start:
                q += " AND date >= ?"
                params.append(str(start)[:10])
            if end:
                q += " AND date <= ?"
                params.append(str(end)[:10])
        q += " ORDER BY date ASC"
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def list_decisions(
        self,
        *,
        day: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10_000,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM decisions WHERE run_id = ?"
        params: List[Any] = [self.run_id]
        if day:
            q += " AND substr(ts, 1, 10) = ?"
            params.append(str(day)[:10])
        else:
            if start:
                q += " AND substr(ts, 1, 10) >= ?"
                params.append(str(start)[:10])
            if end:
                q += " AND substr(ts, 1, 10) <= ?"
                params.append(str(end)[:10])
        q += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        rows = []
        for r in self._conn.execute(q, params).fetchall():
            d = dict(r)
            try:
                d["filters"] = json.loads(d.get("filters_json") or "{}")
            except Exception:
                d["filters"] = {}
            rows.append(d)
        return rows

    def count_events(
        self,
        event_type: Union[EventType, str],
        *,
        day: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> int:
        et = event_type.value if isinstance(event_type, EventType) else str(event_type)
        q = "SELECT COUNT(*) AS c FROM events WHERE run_id = ? AND event_type = ?"
        params: List[Any] = [self.run_id, et]
        if day:
            q += " AND substr(ts, 1, 10) = ?"
            params.append(str(day)[:10])
        else:
            if start:
                q += " AND substr(ts, 1, 10) >= ?"
                params.append(str(start)[:10])
            if end:
                q += " AND substr(ts, 1, 10) <= ?"
                params.append(str(end)[:10])
        return int(self._conn.execute(q, params).fetchone()["c"])
