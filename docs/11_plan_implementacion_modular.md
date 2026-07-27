# Plan de implementación modular — TRAD Equity ML

**Última actualización:** 2026-07-24 (EODHD growth universe + STR-G strategies)  
**Modo de trabajo:** loop-engineering (`design` → `execute-plan`/`implement` → `check-work` → review)  
**Skill de proyecto:** `trad-local` + skills en `.grok/skills/`

**Product modes (STR-06 draft):**

| Mode | Role |
|------|------|
| **STYLE-US** | `turbo_highvol_minalloc` control book — no portable/alpha claim |
| **ALPHA-PORTABLE** | L0/L1/L2 residual path — style clone + PIT EW gates |

---

## Visión por capas

```
DATA → FEATURES → LABELS → MODELS → VALIDATION → BACKTEST → LEAN / DRL
         └─ ALPHA-PORTABLE: L0 membership → L1 CS residual → L2 portfolio
```

| Capa | Módulos | Estado actual (baseline) |
|------|---------|--------------------------|
| DATA | DAT-01..04 | **EODHD primary** + **PIT membership / delisted / SF benches** |
| FEATURES | FEA-01..02, **FEA-04** | Invariant ranks on ALPHA path; ban abs OHLC/ma_* |
| LABELS | LAB-01, **STR-03** | Triple barrier + residual/beat_style API |
| MODELS | MOD-01..03 | Parcial (XGBoost M1/M2, DRL experimental) |
| VALIDATION | VAL-01..03, **STR-01/04**, **VAL-MC/PROMO** | WF + transfer + **Monte Carlo + multi-stage promotion** |
| BACKTEST | BKT-01..02 | PARTIAL+ (portfolio event-driven research; Lean pending parity) |
| RISK/EXEC | RSK-01 | Parcial (risk_manager Lean) |
| PLATFORM | PLT-01..02 | Nuevo (AGENTS, skills, tests, CI-local) |
| **REDESIGN** | **STR-01..06** | **S1 + S1b/S1c FULL + residual_train L1 + STR-02/03/05** |

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

### EQ-01 — Equity mega grid + signal broker leverage

| Campo | Valor |
|-------|--------|
| **Estado** | DONE study v1 (2026-07-23) |
| **Objetivo** | Miles de estrategias long-only US equity; apalancamiento ≤2× solo con señal fuerte; financiación + comisiones/slippage IBKR-like |
| **Código** | `paper_live/equity/*`, `scripts/run_equity_mega_lever_study.py`, `scripts/build_equity_grid_zoo.py` |
| **Datos** | EODHD EOD |
| **Resultado v1** | 1500 sleeves 2015–2025; ver `reports/equity_mega_lever/latest/SUMMARY.md` |
| **Verificación** | `tests/test_equity_mega_unit.py` |

### OPT-01 — Options portfolio + meta-label long history

| Campo | Valor |
|-------|--------|
| **Estado** | DONE study v1 (2026-07-22); **honesty patch v2** (2026-07-23) |
| **Objetivo** | Miles de sleeves de opciones (sin apalancamiento single-name), meta-label take/skip, asignación con caps, WF multi-año desde 2010 |
| **Design** | `docs/design/2026-07-22_options_portfolio_metalabel_longhist.md` |
| **Código** | `paper_live/options/marks_policy.py`, `grid_zoo.py`, `paper_live/portfolio/`, `scripts/run_options_portfolio_meta_study.py` |
| **Resultado v1** | 1000 sleeves · 2010–2025 · port mean 0.9% vs SPY 14.6% (conservador por cash residual); marks `proxy_bs\|vix_surface` |
| **v2 (2026-07-23)** | Norma **marks reales SIEMPRE**; proxy excluye IC/CCS/PCS/CSP; meta label default `beat_spy`; 1 sleeve/und antes de caps; bench `w·SPY+(1−w)·cash`; CLI `--rescore-only` / `--label-mode` / `--marks-mode` |
| **Verificación** | `tests/test_portfolio_meta_unit.py`; report `reports/options_portfolio_meta/latest/SUMMARY.md` |

### VAL-MC-01 / MET-01/02 — Extended metrics + Monte Carlo

| Campo | Valor |
|-------|--------|
| **Estado** | DONE scaffold (2026-07-23) |
| **Objetivo** | Sortino in gates; ulcer/tail/CVaR; trade shuffle/bootstrap MC percentiles |
| **Design** | `docs/design/2026-07-23_metrics_montecarlo_promotion.md` |
| **Código** | `trad_research/risk_metrics.py`, `trad_research/monte_carlo.py` |
| **Verificación** | `tests/test_risk_metrics_unit.py`, `tests/test_monte_carlo_unit.py` |

### PROMO-01/02 — Promotion funnel (only best ADVANCE)

| Campo | Valor |
|-------|--------|
| **Estado** | DONE scaffold (2026-07-23) |
| **Objetivo** | Multi-stage KILL/HOLD/ADVANCE_STYLE/ADVANCE_ALPHA; top-K; residual + Sortino + MC p5 |
| **Código** | `trad_research/promotion.py`, `scripts/run_promotion_scorecard.py` |
| **Verificación** | `tests/test_promotion_unit.py`; report `reports/redesign/promotion_scorecard_v1/` |

### FALSIFY-01 — Falsification Framework v1 (evaluation OS)

| Campo | Valor |
|-------|--------|
| **Estado** | DONE scaffold (2026-07-27) |
| **Objetivo** | Pre-registered KILL/HOLD gates for research candidates: purged/CPCV + embargo, Bailey DSR with real n_trials, leakage scan, book corr, costs/capacity, ResearchMemory JSONL; no ADVANCE in v1 |
| **Design** | `docs/design/2026-07-27_falsification_framework.md` |
| **Código** | `trad_research/falsify/*` (config, purged_cv, deflated_sharpe, leakage, book_corr, feature_store, regime_features, costs_capacity, research_memory, scorecard, pipeline); zoo DSR delegates to falsify |
| **Verificación** | `tests/test_falsify_framework_unit.py`; smoke `python -m trad_research.falsify` |
| **Non-goals** | Residual turbo campaign; paper freeze change; YouTube; heavy LOB/Nautilus |
| **Next research line** | Residual improvement on turbo_strict / minalloc (spec later) |

### STR-01 — Structural problem dossier (P1–P10)

| Campo | Valor |
|-------|--------|
| **Estado** | S1 + S1b + S1c MEASURED (2026-07-23) |
| **Objetivo** | Rank root-causes limiting residual alpha + generalization; freeze before L1 model |
| **Design** | `docs/design/2026-07-23_structural_redesign_alpha.md` |
| **Código** | `trad_research/alpha_attribution.py`, `reports/redesign/STRUCTURAL_PROBLEMS.md`, `scripts/run_s1_early_window.py`, `scripts/run_s1_geo_frozen.py` |
| **Evidencia S1** | P1 design **NOT confirmed** modern (residual vs EW +16.9% CAGR); P2 **NOT confirmed** 2018–25 |
| **Evidencia S1b FULL** | OOS 2010–14 n≈39: base CAGR **18.8%** S0.79 excess SPY +3.8pp; residual vs hardest sane style **+68.8pp**; `style_ew` pathology excluded; `p1_confirmed_any_clone=False` |
| **Evidencia S1c FULL** | P3 **CONFIRMED** via DE: excess DAX **−5.0pp**, transfer FAIL; ES full +11.1% excess IBEX **+4.2pp** (smoke was worse); no foreign retrain |
| **Verificación** | `tests/test_alpha_attribution_unit.py`; `reports/redesign/S1*_*/` |

### STR-02 — L0/L1/L2 portable architecture

| Campo | Valor |
|-------|--------|
| **Estado** | IN PROGRESS v0 + residual_train (2026-07-23) |
| **Objetivo** | Decouple universe (L0), CS signal (L1), portfolio (L2) |
| **Dependencias** | STR-01 |
| **Código** | `trad_research/portable/membership_l0.py`, `score_l1.py`, `portfolio_l2.py`, `cs_features.py`, `residual_labels.py`, `scripts/run_redesign_eval.py` |
| **v0 L1** | `rule_rank` ablation + **`residual_train`** yearly WF logistic on beat_style (invariant CS ranks only; horizon embargo) |
| **Verificación** | `tests/test_portable_cs_unit.py`, `tests/test_portable_l0_l2_unit.py`, `tests/test_residual_train_unit.py`; smoke `S2_portable_v0/`, `S2_residual_train_smoke/` |

### STR-03 — Residual / beat-style labels

| Campo | Valor |
|-------|--------|
| **Estado** | API solid + **L1 train path** (2026-07-23) |
| **Objetivo** | Train targets = beat style / residual excess H (not only barrier meta) |
| **Código** | `trad_research/portable/residual_labels.py`; `score_l1.walk_forward_residual_scores` uses `panel_beat_style_vs_ew` |
| **Verificación** | `tests/test_portable_cs_unit.py`, `tests/test_residual_train_unit.py` |

### STR-04 — Style-clone bench as SSOT gate

| Campo | Valor |
|-------|--------|
| **Estado** | S1 RUN DONE (2026-07-23) |
| **Objetivo** | Same L0 shell, dumb L1 (EW / SMA50 / mom); primary residual gate |
| **Código** | `trad_research/style_clone.py`, `scripts/run_style_clone_gap.py`, `scripts/rescore_style_clone_gap.py` |
| **Evidencia** | Hardest clone `style_ew_hv`; residual Sharpe ~0.40; see `S1_style_clone_gap_full` |
| **Verificación** | `tests/test_style_clone_unit.py` |

### STR-05 — ALPHA-PORTABLE v0 + falsification

| Campo | Valor |
|-------|--------|
| **Estado** | STARTED (v0 rule_rank + residual_train smoke 2026-07-23) |
| **Objetivo** | One redesign candidate; gates R1–R6; no turbo retune |
| **Dependencias** | STR-02, STR-03, STR-04, FEA-04 |
| **Código** | `scripts/run_redesign_eval.py --l1-mode rule_rank\|residual_train`; evidence `S2_portable_v0/`, `S2_residual_train_smoke/` |
| **Honesty** | `engine_mismatch` / `diagnostic_only` / `R2=not_evaluated` / `pass_core=False` until cost-matched |

### STR-06 — Product bifurcation STYLE-US vs ALPHA-PORTABLE

| Campo | Valor |
|-------|--------|
| **Estado** | DRAFT (docs) |
| **Objetivo** | Two product modes; ban single “PASS CAGR vs SPY” as sole gate |
| **Dependencias** | STR-01, STR-05 |

### FEA-04 — Invariant feature contract (ALPHA path)

| Campo | Valor |
|-------|--------|
| **Estado** | ACTIVE on ALPHA path (2026-07-23) |
| **Objetivo** | Ban absolute OHLC on ALPHA-PORTABLE; ranks + scale-free only |
| **Código** | `trad_research/portable/cs_features.py` (`assert_no_absolute_prices` in L1 pipeline) |
| **Verificación** | `tests/test_portable_cs_unit.py` |

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

### DAT-05 — EODHD deep fundamentals (growth path)

| Campo | Valor |
|-------|--------|
| **Estado** | IN PROGRESS (2026-07-24) — client + parse + downloader + unit tests |
| **Objetivo** | Quarterly EPS/revenue history from EODHD with PIT `available_at` lag; replace thin Yahoo fund CSVs (~2024-only) |
| **Código** | `paper_live/data/eodhd_client.py` (`fetch_fundamentals*`, `parse_fundamentals_payload`), `scripts/download_eodhd_fundamentals.py`, `scripts/probe_eodhd_fundamentals.py` |
| **Schema** | `as_of, period, eps, revenue, net_income, available_at, source=eodhd` |
| **Verificación** | `tests/test_eodhd_fundamentals_unit.py`; probe live token; coverage JSON |

### UNI-01 — Double-digit growth universe (G-Q / G-A)

| Campo | Valor |
|-------|--------|
| **Estado** | IN PROGRESS (2026-07-24) |
| **Objetivo** | Hard gates: Q EPS YoY ≥10%, annual EPS TTM YoY ≥15% (rev fallback); rank top-N highest growth; yearly PIT L0 |
| **Código** | `trad_research/growth_universe.py`, `scripts/build_growth_universe.py` |
| **Strategies** | `growth_ew`, `growth_trend_mom`, `growth_cs_mom`, `growth_turbo_minalloc`, `growth_quality_strict` |
| **Battery** | `scripts/run_growth_strategy_battery.py` |
| **Verificación** | `tests/test_growth_universe_unit.py`; no paper freeze change |

---

## Próximo incremento sugerido

1. **DAT-05 download** — run `download_eodhd_fundamentals.py` on `good_tickers_filtrados.txt` (need ≥400 names with ≥20Q).  
2. **UNI-01 + battery** — build yearly growth L0; run S1–S5 OOS; residual vs `growth_ew`.  
3. ~~**S1b/S1c full**~~ — **DONE** early 2010–2014 + geo 2018–2025 FULL packs.  
4. ~~**residual_train L1**~~ — **DONE** WF logistic path; next: cost-matched engine + dual-window R1.  
5. **S3 ALPHA** — costs + crisis blocks on residual gates R1–R6.  
6. No turbo knob patches; no foreign retrain for STYLE-US transfer claims.  
7. Paper freeze stays `turbo_highvol_minalloc` until promotion ADVANCE.

---

## Historial de entregas

| Fecha | Módulo | Resultado | Archivos clave |
|-------|--------|-----------|----------------|
| 2026-07-27 | **Sistema A ORB+HTF falsification** | Spec + EOD proxy (`orb_htf_daily_proxy`); G1 dual-MA + prior-day high break; costs 10+5 bps; risk 0.75% + TP 2R; longhist50 2010–25 **KILL** (CAGR ~−35%, MDD ~−100%, excess SPY ≪0); SPY/QQQ-only also neg CAGR; ADVANCE forbidden; freeze untouched; `data_label=eod_proxy` | `docs/design/2026-07-27_orb_htf_falsification.md`, `trad_research/orb_htf.py`, `scripts/run_orb_htf_falsification.py`, `reports/redesign/orb_htf_falsification_v1/`, `tests/test_orb_htf_unit.py` |
| 2026-07-27 | **Social strategy intel v0** | Bounded X+YouTube pipeline (not full scrape); 20 in-window videos (2026-04-27→07-27) + transcripts + G1–G6 TRAD scorecard; seed `Dbof8VUxP9E`; Weibo firehose gap logged; 13 MARKETING / 5 WEAK / 2 STRONG-structure (not live alpha); freeze untouched | `trad_research/social_intel/`, `reports/social_intel/BATCH_20260727c/`, `tests/test_social_intel_unit.py`, `docs/design/2026-07-27_social_strategy_intel.md` |
| 2026-07-25 | **Overnight definitive search LAUNCH** | Screen 10–17 / confirm 18–25 / full stitch; zoo r2_*+controls+mr notches; seed redesign_v2 PROGRESS; gates CAGR>10% MDD≥−65% n≥80; research PASS=confirm∩full only; freeze untouched | `scripts/run_overnight_definitive_search.py`, `reports/redesign/overnight_definitive/`, `tests/test_overnight_definitive_unit.py` |
| 2026-07-25 | **Kaggle GPU mega plan** | Hierarchical 1e6+ Stage1 GPU sample → Stage3 full WF; math spectral/IC/DSR; pack script + grids; no auto-freeze | `docs/design/2026-07-25_kaggle_gpu_mega_redesign.md`, `kaggle_redesign/`, `pack_kaggle_redesign.py` |
| 2026-07-25 | **Redesign v2 mega LAUNCH** | FEA-02 features (resid/mom_sharpe/trend_stack/vov/graph); 5 `r2_*` strategies; 16h screen/confirm mega + HTML co-occurrence graphs; freeze untouched | `docs/design/2026-07-25_redesign_v2_features_graphs.md`, `trad_research/redesign_v2/`, `run_redesign_v2_mega.py`, `reports/redesign/redesign_v2/` |
| 2026-07-25 | **Universe limit screen/confirm** | Pre-reg grid {40,50,60,80} minalloc; screen 10–17 PASS (40/50/60) but **confirm 18–25 all FAIL** (best 50: CAGR 7.3%/MDD −60%); full stitch limit50 PASS 12.5% but not research PASS; **overfit to 2010–17**; freeze unchanged | `run_universe_limit_screen_confirm.py`, `universe_limit_sc/DECISION.md`, unit tests |
| 2026-07-27 | **FALSIFY-01 framework DONE** | Evaluation OS scaffold: combinatorial purged CV + embargo, Bailey DSR (`n_trials` required + ResearchMemory), leakage detectors, book corr, cost/capacity, scorecard KILL\|HOLD only (no ADVANCE); design + unit tests + CLI smoke; paper freeze unchanged | `docs/design/2026-07-27_falsification_framework.md`, `trad_research/falsify/*`, `tests/test_falsify_framework_unit.py`, zoo DSR wire |
| 2026-07-25 | **Longpath 2010 data+gate** | EODHD restore longhist100 **100/100 ≤2010**; primary minalloc limit**80** 2010–25 **FAIL** CAGR 4.4% MDD −60%; exploratory limit**54** **PASS** 12.8%/−54% (not pre-reg); highvol2010-pass FAIL MDD −86%; freeze unchanged | `download_eodhd_bulk.py`, `run_longpath_2010_gate.py`, `longpath_2010/DECISION.md` |
| 2026-07-25 | **Loop G purged soft-ban COMPLETE** | Ban frozen 2018–21 → confirm 22–25: purged softban **FAIL** (CAGR 22% vs base 48%, resid +6 vs +32, MDD worse); full path KILL MDD; Loop F softban **illusory**; baseline confirm ADVANCE_STYLE window-local; **0 freeze change** | `vol_fund_loop_g/DECISION.md`, `run_loop_g_purged_softban.py` |
| 2026-07-25 | **Loop F risk A/B COMPLETE** | k100 full 2018–25: soft DD size×0.35 **FAIL** (MDD≈−60%); **softban8** CAGR 51.9% MDD **−42.8%** resid +38pp → **HOLD** (mc_mdd_tail); 0 ADVANCE; soft-ban **in-sample** from audit (Loop G purged); freeze minalloc | `vol_fund_loop_f/SUMMARY.md`, `run_loop_f_risk_ab.py`, levers `dd25_soft35*` |
| 2026-07-25 | **Loop E promo COMPLETE** | Full 2018–25 top-3: CAGR 36–46% resid +28–38pp vs style EW; **all KILL mdd_too_deep** (−52.5% to −66%); 0 ADVANCE; freeze minalloc; best risk research = k100_dd35_yr | `vol_fund_loop_e/DECISION.md`, `run_loop_e_promo.py` |
| 2026-07-25 | **Loop D minalloc+risk COMPLETE** | 20 configs vol-only minalloc × levers; confirm **winner `volonly_k100_baseline` CAGR 48% resid +32.5pp MDD −43.5%**; yearly dd35/dd25 also residual+; hard dd25 continuous fails; Success B research HOLD; no paper auto | `vol_fund_loop_d/DECISION.md`, `--grid loop_d` |
| 2026-07-25 | **Vol∩Fund mega COMPLETE** | highvol200 + SEC 159/200≥20Q; 53 configs; screen 18–21: vol-only CAGR 35–40% resid +31–36pp; **confirm 22–25: minalloc vol-only CAGR 24.2% resid +1.7pp (best)**; growth hard confirm **−9.7% resid −8.5pp KILL**; 0 ADVANCE; freeze minalloc | `vol_fund_mega/DECISION.md`, `SUMMARY.md`, `run_vol_fund_mega_loop.py` |
| 2026-07-24 | **DAT-05b SEC free + growth battery 2018–25** | Mega-research → SEC companyfacts 0€ (not EODHD Fund). Panel 80 liq; **71/80 ≥20Q**; battery: growth_ew CAGR **1.8%** MDD −41%, residual timing/ML ≤0; **DECISION: no pay APIs, no ADVANCE**, freeze minalloc. | `sec_fundamentals.py`, `download_sec_fundamentals.py`, `build_fund_panel.py`, `growth_sec_battery_2018_2025/DECISION.md`, tests sec unit |
| 2026-07-24 | **DAT-05 + UNI-01 + STR-G growth** | Design + EODHD fund parse (live **403** Fundamentals); Yahoo deep fallback; gates G-Q≥10% G-A≥15%; smoke 2024–25 L0-snapshot caveat (inflated). | `docs/design/2026-07-24_eodhd_growth_universe_strategies.md`, `growth_universe.py` |
| 2026-07-24 | **Full-universe top5** | n=1121 filtrados 2018–25: minalloc CAGR 19.5% MDD −80% (vs highvol80 ~41%/−54%); sector_rot ranks 1st but MDD −72%; 0 ADVANCE | `reports/redesign/full_universe_top5_2018_2025/` |
| 2026-07-24 | **Mega-audit multi-market overnight** | Winner US CAGR 32.6% MDD −41.9% real vs baseline; **2/5 beat index** (not 5/5); 2020-heavy (LOO ~21%); twin vt corr≈1; score +50 MDD artifact; no paper change | `reports/.../AUDIT.md`, `AUDIT_AUTO.json`, `scripts/audit_multimarket_results.py`, `tests/test_audit_multimarket_unit.py` |
| 2026-07-23 | **Alt MDD loop1–2** | Loop1: continuous hard DD → permanent cash FAIL. Loop2: yearly peak / soft breach; **`dd35_vt80_yr` success B** MDD −45.6% (+25pp) CAGR 30.4% excess+ promo **HOLD** (MC tail blocks ADVANCE); freeze stays minalloc; n=40 2018–25 | `risk_levers.py`, `breadth_gate.py`, `backtest.dd_breach_size_scale`, mega `--grid alt_mdd|alt_mdd_v2`, `docs/design/2026-07-23_alt_strategy_loop.md`, `reports/redesign/alt_loop_2026-07-23/SUMMARY.md`, unit tests |
| 2026-07-23 | **Week plan A–D** | Curated highvol80 overlays (`--grid week`, `--universe-limit 0`); promotion from configs dir; single MDD lever `dd_circuit_25` A/B; freeze decision path keeps `turbo_highvol_minalloc` unless ADVANCE; orchestrator + unit tests + smoke | `docs/design/2026-07-23_week_overlay_risk_promotion.md`, `risk_levers.py`, `run_week_plan_study.py`, mega `--grid week|week_risk`, `run_promotion_scorecard.py --from-configs-dir`, `tests/test_risk_levers_unit.py`, `tests/test_week_grid_unit.py`, `reports/redesign/week_plan_2026-07-23/` |
| 2026-07-23 | **Crash entry + WR** | Causal SPY/QQQ RSI/DD crash overlays on turbo_highvol; hard_stop cooldown + ATR tight; mega study harness (smoke/medium/full); variants `turbo_highvol_crash_rsi*` | `crash_entry.py`, `backtest.py`, `strategy_runner.py`, `run_crash_entry_mega_study.py`, `test_crash_entry_unit.py`, `docs/design/2026-07-23_crash_entry_wr_overlays.md`, `reports/redesign/crash_entry_mega_study/` |
| 2026-07-23 | **VAL-MC + PROMO** | Sortino in gates; Monte Carlo shuffle/bootstrap; multi-stage promotion funnel — scorecard smoke **0 ADVANCE** (MDD/residual/MC honesty) | `risk_metrics.py`, `monte_carlo.py`, `promotion.py`, `run_promotion_scorecard.py`, `docs/design/2026-07-23_metrics_montecarlo_promotion.md`, `reports/redesign/promotion_scorecard_v1/` |
| 2026-07-23 | **STR-01 FULL S1b/S1c + STR-02/03/05 residual_train** | Early FULL base 18.8% residual+68.8pp sane; P1 False (pathology gate); P3 DE full −5pp DAX; residual_train L1 WF; ES full +11% | `S1b_early_window_full/`, `S1c_geo_frozen_full/`, `score_l1.py`, `run_redesign_eval.py --l1-mode`, `test_residual_train_unit.py`, `STRUCTURAL_PROBLEMS.md` |
| 2026-07-23 | **STR-01 S1b/S1c + STR-02 v0** | Early-window smoke residual +36.8% vs style; P3 FROZEN ES/DE **confirmed**; ALPHA-PORTABLE L0/L1/L2 + `run_redesign_eval` | `scripts/run_s1_early_window.py`, `run_s1_geo_frozen.py`, `run_redesign_eval.py`, `trad_research/portable/*`, `reports/redesign/S1b_*`, `S1c_*`, `S2_*` |
| 2026-07-23 | **STR-01 S1 FULL** | Real-data style-clone gap 2018–25 highvol80: baseline CAGR 41.7% S0.97; hardest style EW residual +16.9%; P1/P2 design gates **not confirmed** this window; dossier filled; SPY date-normalize fix | `reports/redesign/S1_style_clone_gap_full/*`, `STRUCTURAL_PROBLEMS.md`, `scripts/rescore_style_clone_gap.py` |
| 2026-07-23 | **STR-01/03/04 + FEA-04** | Design structural redesign; style clones; residual attribution; portable CS/label scaffold; S1 harness CLI | `docs/design/2026-07-23_structural_redesign_alpha.md`, `trad_research/style_clone.py`, `alpha_attribution.py`, `portable/*`, `scripts/run_style_clone_gap.py`, tests unit |
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
| 2026-07-22 | **OPT portfolio meta** | Grid zoo 3000 (no ×2 lottery); meta-label GBT + sleeve caps; WF 2010–2025; 1000 sleeves EODHD + proxy_bs. Port mean **0.9%** vs SPY **14.6%** (cash residual + caps); naive top5 **16.9%**; 2022 port −0.2% vs SPY −20%. Feed history/cache speedups. | `docs/design/2026-07-22_options_portfolio_metalabel_longhist.md`, `paper_live/options/grid_zoo.py`, `paper_live/portfolio/*`, `scripts/run_options_portfolio_meta_study.py`, `reports/options_portfolio_meta/latest/SUMMARY.md`, `tests/test_portfolio_meta_unit.py` |
| 2026-07-23 | **EQ mega lever** | 2500 equity grid zoo; 1500 run 2015–2025 EODHD; signal-scaled L≤2× + financing + IBKR-like costs; rank vs SPY/QQQ. | `paper_live/equity/*`, `scripts/run_equity_mega_lever_study.py`, `reports/equity_mega_lever/latest/SUMMARY.md`, `tests/test_equity_mega_unit.py` |
| 2026-07-23 | **OPT marks honesty v2** | Permanent norm: real marks for short-vol claims. `marks_policy` gates proxy→exclude PCS/CCS/IC/CSP; meta label `beat_spy`/`utility_excess`/`positive_ret`; one sleeve/und before caps; `spy_cash_blend` bench; zoo ban_rules+filter; `--rescore-only` without replaying sleeves. | `paper_live/options/marks_policy.py`, `grid_zoo.py`, `meta_label_selector.py`, `sleeve_portfolio.py`, `scripts/run_options_portfolio_meta_study.py`, `AGENTS.md`, `tests/test_portfolio_meta_unit.py` |
