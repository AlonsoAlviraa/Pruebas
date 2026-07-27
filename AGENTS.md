# AGENTS.md — TRAD / Equity ML Research Platform

Rules for any agent (Grok, Claude, Cursor, etc.) working in this repository.
**Read this file before any non-trivial change.** Combine with the `trad-local` skill and `/loop-engineering`.

---

## Mission

Build and improve a **systematic equity trading research stack** that tries to extract edge from US stocks using:

1. **Data pipeline** — clean OHLCV (and optional fundamentals/sentiment)
2. **Feature engineering** — stationary, causal features only
3. **Labels** — Triple Barrier / event-based (López de Prado style)
4. **Models** — XGBoost primary + meta-labeling; optional DRL portfolio agents
5. **Validation** — purged / embargoed CV, walk-forward, stress tests
6. **Execution layer** — Lean/QuantConnect strategy with risk management

Goal is **robust out-of-sample edge**, not in-sample Sharpe maximization.

---

## Repo map

| Path | Role |
|------|------|
| `data/` | Cached per-ticker CSV (large; usually not committed) |
| `download_data.py` | Parallel download + quality filters |
| `triple_barrier_labeling.py` | Event labels (TP/SL/time) |
| `train_signal_model_v2.py` | Signal / meta model training |
| `run_backtest_signal_v2.py` | Research backtest |
| `lean_strategy/` | Production-style Lean strategy (M1 confirm + M2 signal) |
| `drl_platform/` | RLlib portfolio DRL research |
| `models/` | Joblib / RLlib checkpoints (large) |
| `.grok/skills/` | Installed agent skills (equity / ML / data / quant) |
| `docs/` | Design docs, modular plan, skill catalog |
| `tests/` | Synthetic unit/integration tests |

---

## Non-negotiable research rules

### Causality & leakage

- Features at time `t` may use only information available at or before `t` (and known publication lags).
- Labels that look into the future are **training targets only** — never as features.
- No peeking at future bars, corporate actions not yet announced, or same-day close when simulating open-entry.
- Prefer **purged K-fold + embargo** over random K-fold for time series.

### Labels & meta-labeling

- Prefer **Triple Barrier** (or similar path-dependent labels) over fixed-horizon “return > X%”.
- Primary model = side / direction; secondary (meta) model = size / take-or-skip.
- Document feature names and counts in code and in `lean_strategy/modules/config.py` — they must match training.

### Validation & “beat the market”

- Report **OOS** metrics: Sharpe, Sortino, max DD, Calmar, hit rate, profit factor, turnover, capacity notes.
- Always compare vs **buy-and-hold benchmark** (SPY or equal-weight universe) on the same period.
- When the portfolio is partially in cash, also report **w·SPY + (1−w)·cash** with the same invested weight `w` (fair cash-aware benchmark). Keep full SPY/QQQ BH as secondary.
- Stress at least one crisis window (e.g. 2020, 2022) separately.
- Walk-forward or expanding window required before claiming an edge.
- Reject strategies that only work with razor-thin parameter optima.

### Options marks honesty (permanent)

- **Real marks always** for short-vol research claims. Codified as `marks_mode` / `data_label`:
  - `real_chain` (or `yahoo_chain` / `eodhd_options_eod` / OPRA-class) = exchange or marketplace option quotes/fills
  - `proxy_bs` / `vix_surface` / `proxy_bs|vix_surface` = model Black–Scholes on proxy IV (**not** fills)
- **Never claim OPRA edge from `proxy_bs`.** If real option marks are unavailable, do **not** evaluate short-premium as if real.
- When `marks_mode` is proxy: **exclude** pure short-vol kinds from portfolio meta-study evaluation and research claims: `put_credit_spread`, `call_credit_spread`, `iron_condor`, `cash_secured_put` (grid zoo ban + meta gate). `covered_call` may remain as equity-linked control; still not an OPRA claim.
- When real marks are available: short-vol kinds may re-enter evaluation; keep explicit `marks_mode=real_chain` in reports.
- Implementation SSOT: `paper_live/options/marks_policy.py`.

### Risk & execution realism

- Include commissions + conservative slippage in research backtests.
- Position sizing: volatility targeting / Kelly fraction caps; never unbounded leverage in default configs.
- Hard stop, time stop, and trailing logic must be explicit and testable.
- Live/paper trading is out of scope until OOS + stress gates pass.

### Code quality

- Prefer pure functions for features/labels; inject config via dataclasses.
- Keep modules small; no god-scripts for new work (migrate legacy scripts gradually).
- Spanish OK for user-facing docs; code/identifiers in English preferred for new modules.
- Type hints on public APIs; logging over print for library code.

### Data & privacy / size

- Do not commit huge `data/` or `models/` blobs unless explicitly requested.
- Do not invent tickers or fabricate prices in “real” datasets; synthetic data only in tests.
- Never delete user-downloaded CSVs or trained checkpoints without confirmation.

---

## Engineering loop (mandatory for non-trivial work)

1. **Design** — `/design` or design skill; produce design doc + PR plan under `docs/design/`.
2. **Plan alignment** — update `docs/11_plan_implementacion_modular.md` module state.
3. **Implement** — `/execute-plan` or `/implement --effort 2|3` with project rules injected.
4. **Review** — zero open issues; fix or explicit wontfix.
5. **Verify** — `/check-work`: pytest, import smoke, and any domain scripts listed in the module.
6. **Quality** — optional `code-review` for structural cleanup.

Always invoke **`trad-local`** so domain skills (backtest-expert, feature-engineering, walk-forward-validation, etc.) are applied when relevant.

---

## Domain skills (installed under `.grok/skills/`)

See `docs/skills/INSTALLED_SKILLS.txt` and `docs/skills/SOURCES.md`.

**Core for this repo (use often):**

| Skill | When |
|-------|------|
| `trad-local` | Any work in this repo |
| `backtest-expert` | Strategy validation, robustness |
| `feature-engineering` | ML features |
| `walk-forward-validation` | OOS protocol |
| `vectorbt` / `backtrader` | Fast research backtests |
| `signal-classification` | Primary model labels |
| `regime-detection` | Bull/bear/vol regimes |
| `risk-management` / `position-sizing` / `kelly-criterion` | Risk |
| `market-data` / `ohlcv-processing` / `data-quality` | Data layer |
| `llmquant-equities` / `llmquant-data` / `llmquant-strategies` | Research workflows |
| `edge-pipeline-orchestrator` | Edge discovery loops |

---

## Verification commands (PowerShell, repo root)

```powershell
# Unit tests (synthetic)
python -m pytest tests/ -q --tb=short

# Environment / model layout for Lean
python setup_env.py

# Signal research train/backtest (heavy; only when needed)
# python train_signal_model_v2.py ...
# python run_backtest_signal_v2.py ...

# DRL CLI smoke (requires deps)
# python main.py --help
```

---

## What “done” means

- Spec matches the modular plan module.
- No look-ahead leakage introduced.
- Features/labels/config documented and consistent across train / backtest / Lean.
- Tests pass; OOS or unit evidence attached to the plan history entry.
- Large artifacts not force-committed; docs updated.

---

## Disclaimers

This is **research software**. Nothing here is financial advice. Past backtests do not guarantee future results. Agents must not claim guaranteed alpha.
