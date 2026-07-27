"""P3 research honesty: strategy zoo registry + Deflated Sharpe note.

Append-only ledger of bake-offs / transfer runs. DSR note is documentation-grade
(Bailey & López de Prado style haircut intuition), not a full production DSR library.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_registry_path(root: Path | None = None) -> Path:
    base = root or Path(".")
    return base / "reports" / "strategy_zoo_registry.json"


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"version": "zoo-v1", "trials": [], "n_trials": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def append_trial(
    path: Path,
    *,
    strategy: str,
    tag: str,
    mode: str,
    market: str,
    metrics: Dict[str, Any],
    passed: bool,
    product_mode: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = load_registry(path)
    trial = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "tag": tag,
        "mode": mode,
        "market": market,
        "passed": passed,
        "product_mode": product_mode,
        "metrics": metrics,
        "notes": notes,
    }
    reg.setdefault("trials", []).append(trial)
    reg["n_trials"] = len(reg["trials"])
    path.write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")
    return trial


def deflated_sharpe_note(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int = 8 * 252,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Dict[str, Any]:
    """Bailey–López de Prado DSR note (delegates to ``trad_research.falsify``).

    ``n_trials`` is required for honest multi-test accounting. Prefer importing
    ``deflated_sharpe_ratio`` from ``trad_research.falsify`` for full fields.
    """
    try:
        from trad_research.falsify.deflated_sharpe import (
            deflated_sharpe_note as _dsr_note,
        )

        return _dsr_note(
            observed_sharpe,
            n_trials=n_trials,
            n_obs=n_obs,
            skew=skew,
            kurtosis=kurtosis,
        )
    except Exception:
        # Fallback: rough haircut if falsify import fails
        n_trials = max(int(n_trials), 1)
        e_max_z = math.sqrt(2.0 * math.log(n_trials)) if n_trials > 1 else 0.0
        t_years = max(n_obs / 252.0, 1.0)
        haircut = e_max_z / math.sqrt(t_years)
        deflated = observed_sharpe - haircut
        return {
            "observed_sharpe": observed_sharpe,
            "n_trials": n_trials,
            "n_obs_approx": n_obs,
            "rough_haircut": haircut,
            "deflated_sharpe_approx": deflated,
            "note": (
                "Fallback DSR haircut (falsify unavailable). "
                f"N={n_trials} trials."
            ),
            "skew_assumed": skew,
            "kurtosis_assumed": kurtosis,
        }


def format_dsr_markdown(dsr: Dict[str, Any]) -> str:
    try:
        from trad_research.falsify.deflated_sharpe import (
            format_dsr_markdown as _fmt,
        )

        return _fmt(dsr)
    except Exception:
        return (
            f"- Observed Sharpe: **{dsr['observed_sharpe']:.2f}**\n"
            f"- N trials in zoo: **{dsr['n_trials']}**\n"
            f"- Rough multi-test haircut: **{dsr.get('rough_haircut', float('nan')):.2f}**\n"
            f"- Deflated Sharpe (approx): **{dsr.get('deflated_sharpe_approx', float('nan')):.2f}**\n"
            f"- {dsr.get('note', '')}\n"
        )
