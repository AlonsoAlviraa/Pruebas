# Plan de implementación modular — TRAD Equity ML

**Última actualización:** 2026-07-21 (DAT-04 PIT / survivorship-free)  
**Modo de trabajo:** loop-engineering (`design` → `execute-plan`/`implement` → `check-work` → review)  
**Skill de proyecto:** `trad-local` + skills en `.grok/skills/`

---

## Visión por capas

```
DATA → FEATURES → LABELS → MODELS → VALIDATION → BACKTEST → LEAN / DRL
```

| Capa | Módulos | Estado actual (baseline) |
|------|---------|--------------------------|
| DATA | DAT-01..04 | **EODHD primary** + **PIT membership / delisted / SF benches** |
| FEATURES | FEA-01..02 | Parcial (Lean feature_engine + train scripts) |
| LABELS | LAB-01 | Parcial (`triple_barrier_labeling.py`) |
| MODELS | MOD-01..03 | Parcial (XGBoost M1/M2, DRL experimental) |
| VALIDATION | VAL-01..02 | DONE-ish (walk-forward multi-año en `trad_research`) |
| BACKTEST | BKT-01..02 | PARTIAL+ (portfolio event-driven research; Lean pending parity) |
| RISK/EXEC | RSK-01 | Parcial (risk_manager Lean) |
| PLATFORM | PLT-01..02 | Nuevo (AGENTS, skills, tests, CI-local) |

---

## Módulos

### PLT-01 — Scaffold loop-engineering + skills

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (2026-07-17) |
| **Objetivo** | AGENTS.md, plan modular, skill `trad-local`, 128 skills equity/ML/data, tests skeleton, gitignore |
| **Entregables** | `AGENTS.md`, `docs/11_*`, `.grok/skills/*`, `tests/`, `requirements.txt`, `README.md` |
| **Verificación** | Skills listadas; `pytest` smoke; docs presentes |
| **Historia** | 2026-07-17 — bootstrap inicial del loop |

### PLT-02 — Estructura de paquetes Python limpia

| Campo | Valor |
|-------|--------|
| **Estado** | PENDING |
| **Objetivo** | Extraer lógica de scripts raíz a paquetes (`trad_data`, `trad_features`, `trad_labels`, `trad_models`, `trad_backtest`) sin romper Lean/DRL |
| **Dependencias** | PLT-01 |
| **Verificación** | imports + pytest; scripts legacy como thin CLI |

### VAL-03 — Geo / domain transfer validation (FROZEN_US_TRANSFER)

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (2026-07-17) |
| **Objetivo** | Train US only; eval foreign; preferred BH; transfer gates; product modes |
| **Código** | `trad_research/transfer.py`, `scripts/run_transfer_validation.py` |
| **Constraint** | No Spain bars in train; LOCAL_WF ES = diagnostic only |
| **Verificación** | `tests/test_transfer_validation.py` |

### RSK-02 — DeploymentPolicy + portable risk overlays

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (2026-07-17) |
| **Objetivo** | Pre-registered policies (0.6× portable_conservative); portable_* regimes |
| **Código** | `trad_research/policies.py`, aliases in `regime.py` |
| **Verificación** | policy unit tests in `test_transfer_validation.py` |

### FEA-03 — Portable features via Strategy.feature_names

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (2026-07-17) |
| **Objetivo** | ML strategies use `feature_names`; M2_REL variants `turbo_rel_*`, `champion_ml_rel` |
| **Código** | `trad_research/strategies.py` |
| **Verificación** | feature_names unit test |

### MOD-04 — US OOS threshold freeze

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (code 2026-07-17; heavy grid optional) |
| **Objetivo** | Discrete conf grid US 2018–21 only; forbid foreign roots |
| **Código** | `trad_research/calibration.py`, `scripts/run_mod04_calibration.py` |

### DAT-01 — Contrato de datos OHLCV

| Campo | Valor |
|-------|--------|
| **Estado** | PARTIAL |
| **Objetivo** | Schema único por ticker (`date,open,high,low,close,volume` + opcionales), validación de gaps/splits, índice ordenado |
| **Código actual** | `data/*.csv`, `download_data.py`, `drl_platform/data_pipeline.py` |
| **Verificación** | tests sintéticos de schema + quality report |

### DAT-02 — Downloader moderno

| Campo | Valor |
|-------|--------|
| **Estado** | PARTIAL |
| **Objetivo** | Desacoplar de `ANTIGUOPROGRAMA`; filtros precio/volumen/SPAC documentados; reanudable |
| **Verificación** | dry-run con mocks; no borrar cache sin flag |

### DAT-03 — Universo y quality gates

| Campo | Valor |
|-------|--------|
| **Estado** | PARTIAL |
| **Objetivo** | `good_tickers_filtrados.txt` regenerable con criterios versionados |
| **Verificación** | snapshot de criterios en config YAML/JSON |

### DAT-04 — Point-in-time membership + delisted + SF benchmark

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (2026-07-21, research-grade EODHD — not CRSP) |
| **Objetivo** | Membership as-of por día OOS; incluir delisted (residual→cash); ISIN roll M&A best-effort; benchmarks EW/DVW del mismo universo PIT |
| **Código** | `trad_research/pit_universe.py`, `backtest.py` delist/roll, `strategy_runner.py` flags, `scripts/download_pit_universe.py`, `scripts/run_pit_survivorship_bakeoff.py` |
| **Datos** | `data/pit/membership_index.json`, catalogs active/delisted, `pit_trade_universe.txt` |
| **Skills** | `market-data`, `data-quality`, `backtest-expert`, `walk-forward-validation` |
| **Verificación** | `tests/test_pit_universe.py` (9); bake-off `reports/dashboard_pit_sf_v4/` + `reports/PIT_SURVIVORSHIP_FREE.md` — survivor CAGR 14.8% vs PIT 11.7% (2009–14), delist_exits≥1 |
| **Límites** | first/last EOD ≠ listing calendar CRSP; cobertura delisted incompleta; ISIN chains imperfectos |

### FEA-01 — Feature engine unificado

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (research SSOT 2026-07-17) |
| **Objetivo** | Una sola fuente de features para train, research backtest y Lean; nombres y orden fijos |
| **Código** | `trad_research/features.py`, `trad_research/config.py` FeatureConfig; Lean runtime still QC-side |
| **Skills** | `feature-engineering`, `ohlcv-processing` |
| **Verificación** | `tests/test_fea01_lab01.py` Lean parity 17 names |

### FEA-02 — Features avanzadas (frac diff / microstructure)

| Campo | Valor |
|-------|--------|
| **Estado** | PENDING |
| **Objetivo** | Frac-diff, vol z-score, regime flags — solo si mejoran OOS |
| **Dependencias** | FEA-01, VAL-01 |

### LAB-01 — Triple Barrier production-ready

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (2026-07-17) |
| **Objetivo** | API clara, parámetros k_tp/k_sl/horizon versionados, dataset export reproducible |
| **Código** | `trad_research/labels.py` + `LabelConfig`; legacy `triple_barrier_labeling.py` wraps SSOT |
| **Skills** | `signal-classification` |
| **Verificación** | unit tests barriers + LdP mapping |

### MOD-01 — Primary signal model (M2)

| Campo | Valor |
|-------|--------|
| **Estado** | PARTIAL |
| **Objetivo** | XGBoost multi-class BUY/HOLD/SELL con 17 features alineadas a Lean config |
| **Código actual** | `train_signal_model_v2.py`, `lean_strategy/modules/config.py` |
| **Verificación** | feature count assert; metadata JSON junto al joblib |

### MOD-02 — Meta-label / confirmation (M1)

| Campo | Valor |
|-------|--------|
| **Estado** | PARTIAL |
| **Objetivo** | Filtro de confirmación o meta-labeling según plan de 7 días; reducir trades basura |
| **Docs** | `PLAN_METALABELING_7_DIAS.md`, `ANALISIS_MEJORAS_PROPUESTAS.md` |
| **Verificación** | OOS win-rate vs primary alone |

### MOD-03 — DRL portfolio (research)

| Campo | Valor |
|-------|--------|
| **Estado** | EXPERIMENTAL |
| **Objetivo** | RLlib multi-asset env con rewards PnL/Sharpe/Sortino/Calmar; purged validation |
| **Código actual** | `drl_platform/`, `main.py` |
| **Skills** | `rl-execution`, `portfolio-analytics` |
| **Verificación** | smoke train 1 iter synthetic; no NaN rewards |

### VAL-01 — Walk-forward + purged CV harness

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (2026-07-17) |
| **Objetivo** | Harness único reutilizable por XGB y DRL |
| **Código** | `trad_research/walk_forward.py`, `scripts/run_multi_year_validation.py` |
| **Skills** | `walk-forward-validation`, `backtest-expert` |
| **Verificación** | Expanding WF 2018–2025; embargo; reports en `reports/` |

### VAL-02 — Benchmark & stress suite

| Campo | Valor |
|-------|--------|
| **Estado** | PARTIAL→DONE-ish |
| **Objetivo** | SPY BH, 2022 stress, costos pesimistas, report JSON/MD |
| **Código** | SPY+QQQ via EODHD; `scripts/run_stress_year.py`; champion year slices |
| **Skills** | `backtest-expert`, `performance-metrics` |

### BKT-01 — Research backtest (vectorizado)

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (research path) |
| **Objetivo** | Backtest research con costos; export de trades y equity curve |
| **Código** | `trad_research/backtest.py` (event-driven multi-ticker) |
| **Skills** | `vectorbt`, `backtrader` |

### BKT-02 — Lean strategy parity

| Campo | Valor |
|-------|--------|
| **Estado** | DONE (export 2026-07-17) |
| **Objetivo** | Mismas features/señales/risk que research en `lean_strategy/` |
| **Código** | `trad_research/export_lean.py`, `scripts/export_lean_models.py`; storage `xgb_m1` (6f) + `xgb_m2` (17f) |
| **Verificación** | setup_env EXPECTED 6/17; metadata JSON; unit tests BKT-02 |

### RSK-01 — Risk manager unificado

| Campo | Valor |
|-------|--------|
| **Estado** | PARTIAL |
| **Objetivo** | Chandelier/hard/time stops + vol targeting compartidos |
| **Skills** | `risk-management`, `position-sizing`, `kelly-criterion`, `drawdown-circuit-breaker` |

### LIV-01…08 — Paper Live Year (sin dinero real)

| Campo | Valor |
|-------|--------|
| **Estado** | DONE core stack PR1–5 (LIV-01…08) 2026-07-21; OPS-01/BKT-03 optional |
| **Objetivo** | Correr `turbo_highvol_minalloc` **12 meses paper**: reloj de mercado (no solo EOD), comisiones/slippage por fill, ledger de cada decisión/orden/fill, digests, kill switches; **cero capital real** |
| **Design** | `docs/design/2026-07-21_paper_live_year_mega_plan.md` |
| **Código** | `paper_live/` — freeze + ledger (PR1); datafeed/OMS/runner planned |
| **Strategy freeze** | `turbo_highvol_minalloc` (+ shadow opcional sector_rot) |
| **Verificación PR1–5** | all `tests/test_paper_*.py`; digests + HTML via `scripts/run_paper_daily_digest.py` |
| **Fuera de alcance year-1** | Órdenes con dinero real; multi-país live |

| Submódulo | Rol | Estado |
|-----------|-----|--------|
| LIV-01 | Config freeze + cost model (JSON) | **DONE** PR1 |
| LIV-02 | Ledger SQLite + JSONL append-only | **DONE** PR1 |
| LIV-03 | Market data adapter (daily replay; live delayed later) | **DONE** PR3 |
| LIV-04 | Signal daily→entry (confirm open + session) | **DONE** PR3 |
| LIV-05 | Paper OMS / fill model + commissions | **DONE** PR2 |
| LIV-06 | Kill switch + portfolio risk paper | **DONE** PR4 |
| LIV-07 | Runner RTH + schedule | **DONE** PR4 |
| LIV-08 | Digests diarios/semanales + HTML dashboard | **DONE** PR5 |
| BKT-03 | Gap audit research EOD vs paper | PLANNED |
| OPS-01 | Watchdog, heartbeat, backups | PLANNED |

---

## Orden recomendado (DAG)

```
PLT-01 ✓
  └─► PLT-02
  └─► DAT-01 → DAT-02 → DAT-03
  └─► FEA-01 → LAB-01 → MOD-01 → MOD-02
                    └─► VAL-01 → VAL-02 → BKT-01 → BKT-02
  └─► RSK-01 (paralelo con BKT)
  └─► MOD-03 (paralelo research, no bloquea Lean path)
```

---

## Próximo incremento sugerido

1. **Run / re-run** `transfer_es_v1` full matrix after any policy change (FROZEN, no ES retrain).  
2. **FEA-03 US regression** — compare turbo_rel vs turbo absolute on US WF before adopting.  
3. **XETRA download full** + second FROZEN transfer for `MULTI_REGION_PORTABLE` path.  
4. **BKT-02 cloud** — Lean ObjectStore (US_ONLY until transfer labels allow).

---

## Historial de entregas

| Fecha | Módulo | Resultado | Archivos clave |
|-------|--------|-----------|----------------|
| 2026-07-17 | PLT-01 | Bootstrap skills + loop scaffold | AGENTS.md, docs/11, .grok/skills/*, tests/, trad-local |
| 2026-07-17 | VAL-01 / BKT-01 / MOD-02 | Multi-year WF OOS **PASS** research gates; champion CHAMPION_v7: CAGR 10.8%, Sharpe 0.61, MDD -29%, 1677 trades, 8y, 75% years green. Stretch Sharpe 0.80 not met (QQQ ~0.81). Meta-label + regime + chandelier. MCP guide added. | `trad_research/*`, `scripts/run_multi_year_validation.py`, `reports/multi_year_CHAMPION_v7.*`, `docs/MCP_TRADING.md`, `docs/design/2026-07-17_multi_year_oos_mcp_validation.md` |
| 2026-07-17 | DAT + VAL | EODHD MCP + bulk downloader; **CHAMPION_EODHD_v2 PASS**: CAGR **15.8%**, Sharpe **0.58**, MDD **-28.7%**, 1715 trades, 8y OOS, 75% years+. Fixed calendar years metric. | `scripts/download_eodhd_bulk.py`, `reports/multi_year_CHAMPION_EODHD_v2.*`, EODHD MCP, `docs/design/2026-07-17_eodhd_full_stack_loop.md` |
| 2026-07-17 | FEA-01 + LAB-01 + VAL-02 | FeatureConfig/LabelConfig SSOT; legacy triple_barrier wraps labels; SPY/QQQ EODHD; stress_2022 script; tests green | `trad_research/config.py`, `labels.py`, `tests/test_fea01_lab01.py`, `scripts/run_stress_year.py` |
| 2026-07-17 | **BKT-02** | Export M2=17f/3c + M1=6f/2c a `lean_strategy/storage` + metadata; setup_env alineado; train_end 2025-01-01 | `trad_research/export_lean.py`, `scripts/export_lean_models.py`, `reports/lean_export_bkt02.json` |
| 2026-07-17 | Strategies | Bake-off: champion_ml **PASS** (CAGR 15.8% S 0.58); rules (trend/rsi/breakout/defensive) + hybrid no superan gates | `trad_research/strategies.py`, `strategy_runner.py`, `reports/strategy_comparison_*.md` |
| 2026-07-17 | Aggressive | **aggressive_turbo** user-max-return: CAGR **20.8%**, Sharpe 0.58, MDD **-48.9%**, 2022 still ~-24%; beats champion CAGR | `AggressiveTurboStrategy`, `reports/CHAMPION_AGGRESSIVE_TURBO.*`, `strategy_comparison_aggressive_v2.md` |
| 2026-07-17 | Regime filters | Industry index gates on turbo: **turbo_strict PASS** CAGR **29.2%** S **0.77** MDD **-34%**, 2022 **-9%** | `trad_research/regime.py`, `reports/strategy_comparison_turbo_regime_v1.md` |
| 2026-07-17 | Spain OOS overfit | US champs on IBEX/`data_es`: **0/5 PASS**. turbo_strict **−2.3%** vs US **+29%**; best ES aggressive_turbo **10.8%** MDD **−65%**. Strong US-regime overfit signal | `reports/SPAIN_OOS_OVERFIT_CHECK.md`, `strategy_comparison_spain_oos_v1.md` |
| 2026-07-17 | IBEX regime redesign | Design 2010–17: add `ibex_*` filters + stateful hyst. OOS ES: **ibex_not_bear** CAGR **8.3%** S **0.47** MDD **−48%** (vs legacy −65%, strict −2% CAGR). Still gates FAIL | `trad_research/regime.py`, `reports/SPAIN_IBEX_REGIME_REDESIGN.md` |
| 2026-07-17 | Cross-market gen. design | **No IBEX retrain.** VAL-03 FROZEN_US_TRANSFER + product modes US_ONLY/TRANSFER_CANDIDATE/MULTI_REGION. Industry: home train, geo gate, invariant feats, portable risk. Design approved rev.2 | `docs/design/2026-07-17_cross_market_generalization.md` |
| 2026-07-17 | PR1–9 execute | VAL-03/RSK-02/FEA-03/MOD-04/zoo/LOMO: transfer dual-panel, policies 0.6×, M2_REL names, calibration, zoo+DSR, XETRA skeleton | `trad_research/transfer.py`, `policies.py`, `zoo.py`, `calibration.py`, `scripts/run_transfer_validation.py` |
| 2026-07-17 | Highvol/quality variants | **turbo_highvol × highvol80 PASS** CAGR **50.5%** S **0.83** MDD **−33.5%** (beats strict 29%). Quality price proxy weak. Fund history only ~2024+ | `universe.py`, `universe_highvol80.txt`, `CHAMPION_TURBO_HIGHVOL.md` |
| 2026-07-17 | Multi-market dashboard | 15 scenarios US+ES × windows; HTML equity/DD/trades day-by-day | `scripts/run_multi_market_dashboard.py`, `reports/dashboard_multi_v1/` |
| 2026-07-18 | Adaptive exits v3 | **turbo_strict_adaptive PASS** CAGR **31.2%** S **0.87** MDD **−28.6%** (beats strict). Highvol adaptive 34.7% no bate 50.5% bruto. Extensiones 67–80 | `backtest.py` adaptive_*, `ADAPTIVE_EXIT_RESULTS.md`, `MEGA_TRADE_AUDIT_adaptive_v3.md` |
| 2026-07-18 | Robust pack v4 (5 pts) | Adaptive auto/trail-only + DD kill 18% + ticker caps + **dynamic highvol yearly** (causal). Dynamic top-vol **fails** CAGR −0.9%. ES FROZEN stress only. | `ROBUST_V4_FULL_AUDIT.md`, `turbo_highvol_robust`, `strategy_runner` dynamic |
| 2026-07-18 | Bottleneck close-out A/B/C | **US research baseline** = `turbo_highvol_minalloc` (CAGR 40.8% S 0.70 micro 0%). Composite `minalloc_softreg` (20.5% S 0.45, 2022=163 trades, xfer DE/EU/UK). ES non-highvol **still FAIL**. Phase freeze: `PHASE_CLOSEOUT_BOTTLENECK_V1.md` | `RESEARCH_BASELINE.md`, `turbo_highvol_minalloc_softreg`, `scripts/run_phase_closeout_audit.py` |
| 2026-07-18 | sector_rot + early OOS | **`turbo_highvol_minalloc_sector_rot`** (sector SMA50 + rotation). Early 2012–17: sector_rot CAGR **24.6%** S **0.73** (best); modern 2018–25 **23.3%** S 0.48 vs minalloc **40.8%**. Data from 2010 only (no 2000). | `EARLY_OOS_SECTOR_ROT.md`, `strategy_runner` sector/rot wiring, `run_early_oos_stress.py` |
| 2026-07-18 | Long hist + live pick | EODHD force from **2000**; `universe_longhist100`. Bake-off 8 strats OOS **2005–2025**. **PAPER pick: `turbo_highvol_minalloc`** score 91.5 CAGR 30% S 1.01 MDD −47% micro 0%. Veto micro: highvol/strict/aggressive. | `LIVE_CANDIDATE_DECISION.md`, `run_live_candidate_bakeoff.py`, `dashboard_live_candidate_v1/` |
| 2026-07-21 | **DAT-04 PIT / SF** | Point-in-time membership (EODHD active+delisted), delist residual→cash, ISIN roll, EW+DVW benches. Survivor vs PIT bake-off: **14.8% vs 11.7% CAGR** (2009–14 minalloc); delist_exits≥1; excess vs PIT EW −18.4%. | `pit_universe.py`, `download_pit_universe.py`, `run_pit_survivorship_bakeoff.py`, `reports/PIT_SURVIVORSHIP_FREE.md`, `tests/test_pit_universe.py` |
| 2026-07-21 | **LIV mega-plan** | Paper live year sin dinero real: reloj RTH, fees por fill, ledger total, digests, kill switches, 8 PRs (LIV-01…08 + BKT-03 + OPS-01). Strategy freeze minalloc. | `docs/design/2026-07-21_paper_live_year_mega_plan.md`, docs/11 LIV section |
| 2026-07-21 | **LIV-01+02 PR1** | Config freeze (`strategy/cost/schedule/universe` JSON + `config_hash`) + `PaperLedger` SQLite/JSONL/snapshots; paper-only guard; cli_init; 8 unit tests. | `paper_live/`, `tests/test_paper_live_ledger.py` |
| 2026-07-21 | **LIV-05 PR2** | Paper OMS: `FillModel` (slip/commission/SEC/ADV/TWAP) + `PaperBroker` submit/execute/cancel; cash/positions; ledger fills with costs; 12 unit tests. | `paper_live/oms/`, `tests/test_paper_fill_costs.py` |
| 2026-07-21 | **LIV-03+04 PR3** | Daily replay feed (CSV/synthetic, causal features); rule signal post-close; open confirm; `ReplaySession` entries/exits via OMS+ledger; cli_replay; 6 tests. | `paper_live/datafeed/`, `signals/`, `replay_session.py`, `tests/test_paper_replay_signals.py` |
| 2026-07-21 | **LIV-06+07 PR4** | PortfolioRisk + sticky KillSwitch (DD 18%/−15%, Sharpe 20d); soft size scale; ScheduleClock; PaperRunner replay/live-stub; wired into ReplaySession; 8 risk tests. | `paper_live/risk/`, `runner.py`, `schedule_clock.py`, `tests/test_paper_risk_runner.py` |
| 2026-07-21 | **LIV-08 PR5** | Daily digest + weekly scorecard (costs, micro%, kill flags) + self-contained HTML dashboard; ledger query helpers; `run_paper_daily_digest.py`. | `paper_live/reports/`, `scripts/run_paper_daily_digest.py`, `tests/test_paper_digests.py` |
| 2026-07-21 | **Paper cloud free** | 10-strategy zoo + Stooq free data + GitHub Actions daily cron; results in `reports/paper_cloud/` for study without local PC. | `paper_live/cloud/`, `.github/workflows/paper_live_daily.yml`, `scripts/run_paper_cloud_batch.py` |
