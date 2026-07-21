"""LIV-01: frozen paper config bundle + hash + paper-only guard."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "config"


class PaperModeError(RuntimeError):
    """Raised when real-money paths are requested or paper guard fails."""


def assert_paper_only(*, require_env: bool = False) -> None:
    """Hard guard: paper engine never enables real-money mode.

    If require_env=True (live runner, later PRs), TRAD_PAPER_ONLY must be '1'.
    """
    mode = os.environ.get("TRAD_TRADING_MODE", "paper").strip().lower()
    if mode in ("live", "real", "production_money"):
        raise PaperModeError(
            f"TRAD_TRADING_MODE={mode!r} is forbidden. Paper engine is virtual capital only."
        )
    if require_env and os.environ.get("TRAD_PAPER_ONLY", "") != "1":
        raise PaperModeError(
            "TRAD_PAPER_ONLY=1 is required to start the paper live runner. "
            "No real-money trading path is implemented in LIV-01/02."
        )


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing freeze config: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be object: {path}")
    return data


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_config_hash(bundle: Mapping[str, Any]) -> str:
    """Stable SHA-256 over canonical JSON of the freeze bundle (no run metadata)."""
    raw = _canonical_json(bundle).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class StrategyFreeze:
    version: str
    mode: str
    strategy_id: str
    description: str
    capital0: float
    currency: str
    long_only: bool
    max_leverage: float
    knobs: Dict[str, Any]
    risk_paper: Dict[str, Any]
    shadow_strategy_id: Optional[str] = None
    shadow_enabled: bool = False
    notes: tuple = ()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StrategyFreeze":
        mode = str(d.get("mode") or "paper").lower()
        if mode != "paper":
            raise PaperModeError(f"strategy freeze mode must be 'paper', got {mode!r}")
        notes = d.get("notes") or []
        return cls(
            version=str(d.get("version") or "strategy-freeze-v1"),
            mode="paper",
            strategy_id=str(d["strategy_id"]),
            description=str(d.get("description") or ""),
            capital0=float(d.get("capital0") or 100_000.0),
            currency=str(d.get("currency") or "USD"),
            long_only=bool(d.get("long_only", True)),
            max_leverage=float(d.get("max_leverage") or 1.0),
            knobs=dict(d.get("knobs") or {}),
            risk_paper=dict(d.get("risk_paper") or {}),
            shadow_strategy_id=(
                str(d["shadow_strategy_id"]) if d.get("shadow_strategy_id") else None
            ),
            shadow_enabled=bool(d.get("shadow_enabled", False)),
            notes=tuple(notes) if isinstance(notes, list) else (str(notes),),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "strategy_id": self.strategy_id,
            "description": self.description,
            "capital0": self.capital0,
            "currency": self.currency,
            "long_only": self.long_only,
            "max_leverage": self.max_leverage,
            "knobs": self.knobs,
            "risk_paper": self.risk_paper,
            "shadow_strategy_id": self.shadow_strategy_id,
            "shadow_enabled": self.shadow_enabled,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CostModel:
    version: str
    broker_profile: str
    commission: Dict[str, float]
    slippage: Dict[str, float]
    spread: Dict[str, Any]
    sec_fee_sell_only: bool
    sec_fee_per_million: float
    finra_taf: bool
    finra_taf_per_share: float
    finra_taf_max_per_trade: float
    min_price: float
    max_participation_rate: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CostModel":
        return cls(
            version=str(d.get("version") or "cost-v1"),
            broker_profile=str(d.get("broker_profile") or "default"),
            commission={k: float(v) for k, v in dict(d.get("commission") or {}).items()},
            slippage={k: float(v) for k, v in dict(d.get("slippage") or {}).items()},
            spread=dict(d.get("spread") or {}),
            sec_fee_sell_only=bool(d.get("sec_fee_sell_only", True)),
            sec_fee_per_million=float(d.get("sec_fee_per_million") or 8.0),
            finra_taf=bool(d.get("finra_taf", True)),
            finra_taf_per_share=float(d.get("finra_taf_per_share") or 0.000166),
            finra_taf_max_per_trade=float(d.get("finra_taf_max_per_trade") or 8.3),
            min_price=float(d.get("min_price") or 2.0),
            max_participation_rate=float(d.get("max_participation_rate") or 0.02),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def estimate_commission(self, qty: int, price: float) -> float:
        """Per-order commission estimate (buy or sell)."""
        q = abs(int(qty))
        px = float(price)
        notional = q * px
        if q <= 0 or notional <= 0:
            return 0.0
        per_share = float(self.commission.get("per_share", 0.005))
        min_order = float(self.commission.get("min_per_order", 1.0))
        max_pct = float(self.commission.get("max_pct_of_notional", 0.005))
        raw = q * per_share
        capped = min(raw, notional * max_pct) if max_pct > 0 else raw
        return float(max(min_order, capped))

    def estimate_sell_fees(self, qty: int, price: float) -> float:
        """SEC + FINRA TAF approximation on sells only."""
        q = abs(int(qty))
        notional = q * float(price)
        if q <= 0 or notional <= 0:
            return 0.0
        fees = 0.0
        if self.sec_fee_sell_only and self.sec_fee_per_million > 0:
            fees += notional * (self.sec_fee_per_million / 1_000_000.0)
        if self.finra_taf:
            taf = q * self.finra_taf_per_share
            fees += min(taf, self.finra_taf_max_per_trade)
        return float(fees)

    def slip_price(
        self,
        side: str,
        mid: float,
        *,
        is_stop: bool = False,
        participation_pct: float = 0.0,
    ) -> float:
        """Apply adverse slippage in bps to mid price."""
        mid = float(mid)
        if mid <= 0:
            return mid
        side_l = side.lower()
        bps = float(
            self.slippage.get("entry_bps" if side_l == "buy" else "exit_bps", 5.0)
        )
        if is_stop:
            bps += float(self.slippage.get("stop_extra_bps", 10.0))
        impact = float(self.slippage.get("impact_bps_per_adv_pct", 0.0)) * max(
            0.0, float(participation_pct)
        )
        bps += impact
        # fallback half-spread if no quote
        bps += float(self.spread.get("fallback_bps", 0.0)) / 2.0
        mult = bps / 10_000.0
        if side_l == "buy":
            return mid * (1.0 + mult)
        return mid * (1.0 - mult)


@dataclass(frozen=True)
class ScheduleConfig:
    version: str
    timezone: str
    pre_open_hhmm: str
    rth_open_hhmm: str
    entry_window_start_hhmm: str
    entry_window_end_hhmm: str
    midday_rescan_hhmm: str
    exit_check_start_hhmm: str
    exit_check_end_hhmm: str
    force_flatten_hhmm: str
    post_close_hhmm: str
    night_job_hhmm: str
    bar_size_phase1: str
    bar_size_stops: str
    skip_first_minutes_after_open: int
    heartbeat_seconds: int
    stale_quote_seconds: int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScheduleConfig":
        return cls(
            version=str(d.get("version") or "schedule-v1"),
            timezone=str(d.get("timezone") or "America/New_York"),
            pre_open_hhmm=str(d.get("pre_open_hhmm") or "09:00"),
            rth_open_hhmm=str(d.get("rth_open_hhmm") or "09:30"),
            entry_window_start_hhmm=str(d.get("entry_window_start_hhmm") or "09:45"),
            entry_window_end_hhmm=str(d.get("entry_window_end_hhmm") or "10:30"),
            midday_rescan_hhmm=str(d.get("midday_rescan_hhmm") or "12:00"),
            exit_check_start_hhmm=str(d.get("exit_check_start_hhmm") or "15:30"),
            exit_check_end_hhmm=str(d.get("exit_check_end_hhmm") or "15:50"),
            force_flatten_hhmm=str(d.get("force_flatten_hhmm") or "15:55"),
            post_close_hhmm=str(d.get("post_close_hhmm") or "16:15"),
            night_job_hhmm=str(d.get("night_job_hhmm") or "18:00"),
            bar_size_phase1=str(d.get("bar_size_phase1") or "5m"),
            bar_size_stops=str(d.get("bar_size_stops") or "1m"),
            skip_first_minutes_after_open=int(d.get("skip_first_minutes_after_open") or 15),
            heartbeat_seconds=int(d.get("heartbeat_seconds") or 60),
            stale_quote_seconds=int(d.get("stale_quote_seconds") or 300),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UniverseConfig:
    version: str
    ticker_file: str
    limit: int
    min_price: float
    min_adv20_usd: float
    regime_symbols: tuple
    trade_regime_symbols: bool
    monthly_rescore: bool
    rescore_method: str
    exclude_pure_max_vol_dynamic: bool
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "UniverseConfig":
        reg = d.get("regime_symbols") or ["QQQ", "SPY"]
        return cls(
            version=str(d.get("version") or "universe-v1"),
            ticker_file=str(d.get("ticker_file") or "universe_longhist100.txt"),
            limit=int(d.get("limit") or 80),
            min_price=float(d.get("min_price") or 5.0),
            min_adv20_usd=float(d.get("min_adv20_usd") or 5_000_000.0),
            regime_symbols=tuple(str(x) for x in reg),
            trade_regime_symbols=bool(d.get("trade_regime_symbols", False)),
            monthly_rescore=bool(d.get("monthly_rescore", True)),
            rescore_method=str(d.get("rescore_method") or "causal_highvol_quality_hybrid"),
            exclude_pure_max_vol_dynamic=bool(d.get("exclude_pure_max_vol_dynamic", True)),
            notes=str(d.get("notes") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["regime_symbols"] = list(self.regime_symbols)
        return d


@dataclass(frozen=True)
class PaperFreeze:
    """Full LIV-01 freeze bundle with stable config_hash."""

    strategy: StrategyFreeze
    cost: CostModel
    schedule: ScheduleConfig
    universe: UniverseConfig
    config_hash: str
    source_dir: str

    def to_bundle_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "cost": self.cost.to_dict(),
            "schedule": self.schedule.to_dict(),
            "universe": self.universe.to_dict(),
        }

    def to_public_dict(self) -> Dict[str, Any]:
        out = self.to_bundle_dict()
        out["config_hash"] = self.config_hash
        out["source_dir"] = self.source_dir
        return out


def load_freeze(
    config_dir: Optional[Union[str, Path]] = None,
    *,
    enforce_paper: bool = True,
) -> PaperFreeze:
    """Load strategy/cost/schedule/universe freeze files and compute config_hash."""
    if enforce_paper:
        assert_paper_only(require_env=False)
    cdir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    strategy = StrategyFreeze.from_dict(_read_json(cdir / "strategy_freeze.json"))
    cost = CostModel.from_dict(_read_json(cdir / "cost_model.json"))
    schedule = ScheduleConfig.from_dict(_read_json(cdir / "schedule.json"))
    universe = UniverseConfig.from_dict(_read_json(cdir / "universe.json"))
    bundle = {
        "strategy": strategy.to_dict(),
        "cost": cost.to_dict(),
        "schedule": schedule.to_dict(),
        "universe": universe.to_dict(),
    }
    h = compute_config_hash(bundle)
    return PaperFreeze(
        strategy=strategy,
        cost=cost,
        schedule=schedule,
        universe=universe,
        config_hash=h,
        source_dir=str(cdir.resolve()),
    )
