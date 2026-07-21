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
    """Approximate DSR haircut intuition (not a full Bailey–LdP implementation).

    Returns a note dict for MD reports: expected max Sharpe under N independent
    trials ~ rough order sqrt(2 log N) * sigma, plus haircut suggestion.
    """
    n_trials = max(int(n_trials), 1)
    # Very rough multiple-testing haircut: expected max of N N(0,1) ≈ sqrt(2 ln N)
    e_max_z = math.sqrt(2.0 * math.log(n_trials)) if n_trials > 1 else 0.0
    # Annualized Sharpe noise scale ~ 1/sqrt(T) for independent daily returns
    sigma_sh = 1.0 / math.sqrt(max(n_obs, 1))
    expected_max_noise_sharpe = e_max_z * sigma_sh * math.sqrt(252) / math.sqrt(252)
    # Keep simple: haircut = observed - c * sqrt(2 ln N) / sqrt(T_years)
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
            "Approximate Deflated Sharpe haircut for multiple testing "
            f"(N={n_trials} trials). Not a full Bailey & López de Prado DSR "
            "implementation — use as research honesty bound, not a p-value."
        ),
        "skew_assumed": skew,
        "kurtosis_assumed": kurtosis,
    }


def format_dsr_markdown(dsr: Dict[str, Any]) -> str:
    return (
        f"- Observed Sharpe: **{dsr['observed_sharpe']:.2f}**\n"
        f"- N trials in zoo: **{dsr['n_trials']}**\n"
        f"- Rough multi-test haircut: **{dsr['rough_haircut']:.2f}**\n"
        f"- Deflated Sharpe (approx): **{dsr['deflated_sharpe_approx']:.2f}**\n"
        f"- {dsr['note']}\n"
    )
