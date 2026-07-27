# Plan — Kaggle GPU mega redesign (millones de combos, math nueva, multi-prueba)

**Fecha:** 2026-07-25  
**Objetivo:** Empaquetar y lanzar en **Kaggle** un rediseño masivo de estrategia/features/matemáticas, con presupuesto de cómputo de **horas GPU** (P100/T4), sin auto-cambiar paper freeze.  
**Producto:** research only · gates confirm 2018–25 · screen 2010–17 solo para freeze de candidatos  

---

## 0. Realidad de hardware (leer antes de soñar)

| Trabajo | ¿GPU ayuda? | Notas |
|---------|-------------|--------|
| XGBoost / LightGBM train | **Sí** (`device=cuda` / `gpu_hist`) | Mejor ROI en Kaggle |
| Feature matrix densa (polinomial, CS ranks) | **Sí** (CuPy / PyTorch) | Precompute 1× |
| Backtest bar-a-bar (actual `trad_research.backtest`) | **Casi no** | Python loop CPU; no reescribir en 1 día a CUDA kernels |
| Vectorbt / numba portfolio sim | **Parcial** | Acelera sim si vectorizamos señales |
| Graph spectral (eigen, Laplacian) | **Sí** (torch/cupy) | Features de red / clustering |
| DRL (RLlib) | **Sí** | Rama opcional fase 3 |

**Conclusión de arquitectura:**  
**Millones de combos ≠ millones de walk-forwards completos 2010–25.**  
Eso no cabe ni en 100 h GPU si cada path tarda minutos.

Usamos **búsqueda jerárquica**:

```
Stage 0  Precompute features + labels  (GPU/CPU, 1×)
Stage 1  Micro-score  ~1e6–1e7 configs  (GPU: train tiny / score vectorized, NO full BT)
Stage 2  Fast path     ~1e4–5e4 survivors (vectorbt o short window 2018–21 only)
Stage 3  Full WF       ~200–500         (screen 10–17 + confirm 18–25 + full stitch)
Stage 4  Stress        top 20           (LOYO, crisis, MC, graph audit)
```

Solo Stage 3+ puede reclamar **research PASS**. Stage 1–2 son filtros baratos.

---

## 1. Lecciones locales (priors para el grid)

Del run `redesign_v2` (parcial, 12/24) y estudios previos:

| Prior | Implicación para Kaggle |
|-------|-------------------------|
| minalloc L50: screen OK, confirm FAIL | No confiar en 2010–17 |
| **turbo_strict L80 confirm PASS** (parcial) | Incluir familia “strict / defensive” en grid grande |
| r2_trend_stack L50 confirm ~−45% | Stack duro sin sizing → matar |
| Soft-ban / limit=54 | **Prohibidos** como free params post-hoc |
| highvol decade MDD −75/−86% | Highvol solo como sub-universo acotado, no default |

**Freeze paper:** `turbo_highvol_minalloc` hasta human ADVANCE.

---

## 2. Espacio de búsqueda (cómo llegar a “millones”)

### 2.1 Factores pre-registrados (cartesiano teórico)

| Eje | Cardinalidad | Ejemplos |
|-----|--------------|----------|
| L0 universe | 6 | longhist50/80, highvol2010_50, quality80, mixed, random_ctrl |
| Signal family | 12 | residual_mom, mom_sharpe, trend_stack, rsi_reclaim, ml_m2, ml_rel, cs_rank, low_beta, vol_break, meanrev, hybrid, strict_shell |
| Feature pack | 8 | M2, M2_REL, R2_ext, residual+graph, spectral_top, poly2_rel, csrank_only, ablate |
| Label | 5 | TB(k_tp,k_sl,h) grid coarse, side3, meta binary |
| Model | 6 | XGB_gpu, LGBM_gpu, logistic, ridge, ensemble2, rules_only |
| Entry filter | 10 | regime×trend×dd_floor×vov×beta_cap |
| Sizing | 8 | vt{0.015…0.04}, min_alloc, max_pos{6…16}, pos_cap |
| Exit | 8 | hard_stop, atr_stop, horizon, trail packs |
| Risk overlay | 6 | none, vt_tight, soft_dd, yearly_peak, sector_gate, corr_cap |

**Producto bruto:** ~6×12×8×5×6×10×8×8×6 ≈ **6.6×10⁷** (decenas de millones).  

**No se evalúan todos en full BT.** Se muestrean:

| Stage | n configs | Método |
|-------|-----------|--------|
| 1a Random / Sobol sample | **2e6** | Uniform over discrete grids |
| 1b Local mutate top-1% | **+5e5** | ±1 notch neighbors |
| 2 Fast BT | **2e4** | top by Stage1 score |
| 3 Full WF | **300** | top Stage2 + forced controls |
| 4 Stress | **20** | Stage3 confirm passers |

---

## 3. Matemáticas nuevas (bloques a implementar en Kaggle package)

### 3.1 Ya en repo (llevar)
- `redesign_v2.features_ext`: resid, mom_sharpe, trend_stack, vov, dd_peak, rsi_reclaim  
- `redesign_v2.graph_math`: corr graph, co-occurrence, hubs  

### 3.2 Nuevos (Kaggle `kaggle_redesign/math/`)

| Módulo | Fórmulas / idea | Uso Stage |
|--------|-----------------|-----------|
| `spectral.py` | Laplacian de corr rolling; fiedler / participation ratio por ticker-cluster | Feature: distancia a hub de corr |
| `orthogonal_mom.py` | Residualizar ret vs SPY+QQQ+sector ETF (multi-β) | Score residual multi-factor |
| `information_coef.py` | IC rank(feature, forward_ret) causal expanding | Stage1 feature selection |
| `drawdown_geometry.py` | time-under-water, Ulcer, Calmar rolling | Stage2 filter |
| `capacity_math.py` | turnover, Herfindahl de PnL, effective N | Stage4 kill concentration |
| `purged_score.py` | Embargoed score: train ≤ T−h, score on [T,T+w] | Stage1 integrity |
| `sobol_grid.py` | Quasi-MC sampling of discrete grids | Stage1 sampling |
| `deflated_sharpe.py` | DSR / haircut by n_trials | Stage3–4 honesty |

**Prohibido en Stage3 claim:** optimizar thresholds en confirm 2018–25.

---

## 4. Micro-score Stage 1 (el “millón” real)

Para cada config sampleada \(c\):

1. Features precomputed (GPU once per ticker/year pack).  
2. Train model **solo 2010–2015** (o expanding yearly mini) en GPU.  
3. Score OOS **2016–2017 only** (short):  
   - IC mean, hit rate top-decile, simple long-flat CAGR proxy, MDD proxy.  
4. `stage1_score = 1.5·IC + 1.0·proxy_cagr − 1.0·proxy_mdd_depth − 0.2·turnover`  

**Tiempo target:** ≤ 5–20 ms/config en batch → 2e6 configs en **pocas horas GPU**.

Implementación recomendada:
- Parquet features `/kaggle/input/.../features_longhist.parquet`  
- PyTorch o CuPy batch matmul for linear/logistic; XGB-GPU for nonlinear subset  
- Joblib memmap + multiprocessing CPU for rule-only families (GPU free for ML families)

---

## 5. Stage 2–4 (validación seria)

### Stage 2 — Fast path (~2e4)
- Window **2018–2021 only** (4y) vectorized or existing WF with `min_train_rows` lower  
- Kill if CAGR≤0 or MDD < −70%  
- Keep top 300 by confirm-style score **without** using 2022–25  

### Stage 3 — Full protocol (~300)  **CLAIM WINDOW**
```
screen  2010–2017  → rank freeze top-K within family (≤3 per family)
confirm 2018–2025  → gates:
  CAGR > 10%
  MDD  ≥ −65%
  n_trades ≥ 80
  excess SPY total > 0   (soft prefer; hard optional flag)
full stitch 2010–2025 → report; research PASS = confirm∩full gates
```

### Stage 4 — Stress top 20
- LOYO drop 2020 / 2022  
- Green-year frac ≥ 0.55  
- MC block bootstrap MDD p5 ≥ −70%  
- Graph hub concentration HHI of positive PnL  
- Capacity: average $ notional vs ADV proxy  

---

## 6. Paquete Kaggle (estructura a crear)

```
kaggle_redesign/
  README.md                 # cómo lanzar
  requirements_kaggle.txt   # xgboost, lightgbm, torch, polars, pyarrow, vectorbt?
  dataset_manifest.json     # tickers + hashes + from=2000
  notebook/
    00_setup_and_data.ipynb
    01_precompute_features.ipynb    # GPU/CPU
    02_stage1_million_scan.ipynb    # GPU
    03_stage2_fast_path.ipynb
    04_stage3_full_wf.ipynb         # CPU-heavy
    05_stage4_stress_graphs.ipynb
    06_export_results.ipynb
  src/
    grids.py                # discrete axes + sobol sample
    stage1_scorer.py
    stage2_fast_bt.py
    stage3_wf_adapter.py    # wraps trad_research WF
    math/                   # spectral, IC, orthogonal_mom, ...
    report.py
  tests/
    test_grids_unit.py
    test_stage1_no_leak.py
    test_deflated_sharpe.py
```

### Dataset Kaggle (input)
| Contenido | Fuente local |
|-----------|--------------|
| `data/{T}_history.csv` longhist100 + highvol2010 pass + SPY/QQQ | Ya EODHD from 2000 |
| `universe_*.txt` | repo |
| Código `trad_research/` mínimo | zip sin `data/` full 1121 si no hace falta |
| **No** models/ pesados ni paper secrets |

**Tamaño estimado:** longhist100+hv ~ 100–200 MB comprimidos (OK dataset privado).

### Dataset Kaggle (output working)
- `stage1_top.parquet`  
- `stage2_results.json`  
- `stage3_arms/**`  
- `SUMMARY.md` / `DECISION.md`  
- graphs HTML  

---

## 7. Sesiones Kaggle (calendario práctico)

Kaggle GPU session ~**9–12 h** (límites cambian; planear 2–3 sesiones).

| Sesión | Horas | Trabajo |
|--------|-------|---------|
| **S0** | 1–2 | Upload dataset + smoke import + 100 configs Stage1 |
| **S1** | 6–9 GPU | Precompute + Stage1 2e6 sample + export top |
| **S2** | 6–9 GPU/CPU | Stage2 2e4 + Stage3 start 150 arms |
| **S3** | 6–9 CPU | Stage3 rest + Stage4 stress + DECISION |

**CPU notebook** puede hacer Stage3 si GPU se gasta en Stage1.

Resume: todos los stages leen/escriben Parquet en `/kaggle/working` y se **versionan** como dataset output o se descargan.

---

## 8. Anti-overfit (no negociable)

1. **Pre-register** ejes de grid en este doc (no añadir 54 post-hoc).  
2. Confirm 2018–25 **nunca** en Stage1 score.  
3. Report **n_trials** + deflated Sharpe / top-K.  
4. Al menos **1 control aleatorio** (scores shuffled) debe fallar gates.  
5. Family cap: máx 2 research PASS por signal family.  
6. Paper freeze **no** se escribe desde Kaggle.  
7. Si Stage1 top no corre en Stage3 (code path bug) → kill batch, no inventar métricas.

---

## 9. Criterio de éxito del mega Kaggle

| Nivel | Criterio |
|-------|----------|
| **Infra PASS** | Stage1 2e6 completes; Stage3 ≥200 arms; reports downloadable |
| **Research HOLD** | ≥1 arm confirm gates (CAGR>10%, MDD≥−65%) + full not pathological |
| **Research ADVANCE candidate** | HOLD + Stage4 MC/LOYO OK + residual vs SPY>0 + human review |
| **FAIL honesto** | 0 confirm passers → documentar; freeze stays |

---

## 10. Checklist de implementación (orden de código local → Kaggle)

### Fase A — Empaque (local, ~2–4 h eng)
- [ ] `scripts/pack_kaggle_redesign.py` — zip `trad_research` + universes + longhist CSVs  
- [ ] `kaggle_redesign/src/grids.py` — axes + sobol/random sample to 2e6  
- [ ] `kaggle_redesign/src/math/*` — spectral, multi-β residual, IC, DSR  
- [ ] Unit tests no-leak Stage1 (train end < score start)  
- [ ] `dataset_manifest.json` con tickers y min dates  

### Fase B — Notebooks
- [ ] 00 setup paths `/kaggle/input/...`  
- [ ] 01 precompute features → parquet  
- [ ] 02 Stage1 GPU scan  
- [ ] 03 Stage2 fast  
- [ ] 04 Stage3 adapter a `run_strategy_walk_forward` / lightweight clone  
- [ ] 05 graphs + stress  
- [ ] 06 SUMMARY/DECISION export  

### Fase C — Launch
- [ ] Crear dataset Kaggle privado `trad-longhist-2010`  
- [ ] Notebook GPU P100/T4, Internet off si posible  
- [ ] Commit working outputs cada 2 h  
- [ ] Descargar DECISION + top equities al repo `reports/redesign/kaggle_mega_*/`  

### Fase D — Post
- [ ] Actualizar `docs/11` con evidencia  
- [ ] **No** touch `paper_live` freeze sin human  

---

## 11. Estimación de coste / tiempo

| Item | Estimación |
|------|------------|
| Empaque + código Stage1 | 1 día eng |
| Upload data | 30–90 min |
| Stage1 2e6 GPU | 4–10 h |
| Stage2 2e4 | 3–8 h |
| Stage3 300 full WF | 8–20 h **CPU** (varios sessions) |
| Stage4 | 1–3 h |
| **Total wall** | **~3–5 días calendario** con 2–3 sesiones Kaggle/día |

---

## 12. Relación con run local interrumpido

`reports/redesign/redesign_v2/PROGRESS.json` quedó en **12/24** arms.  
Acción recomendada **antes o en paralelo** al pack Kaggle:

1. Relanzar local `run_redesign_v2_mega.py` (resume) para cerrar baseline 24 arms (~2–4 h).  
2. Meter `turbo_strict__longhist_L80` (confirm PASS parcial) como **seed control** en Stage3 Kaggle.  
3. No usar resultados Stage1 Kaggle para “mejorar” strict post-hoc en el mismo confirm window.

---

## 13. Riesgos

| Riesgo | Mitigación |
|--------|------------|
| “Millones” solo Stage1 basura | Publicar deflated metrics + random control |
| GPU infra mala para backtest | Stage3 en CPU notebook |
| Dataset too big | Solo longhist100 + hv50, no 1121 tickers |
| Session kill mid-run | PROGRESS + parquet shards cada N configs |
| Leakage feature | Unit test: feature at t independent of close[t+1] |

---

## 14. Definition of done (este plan)

- [x] Doc aprobado (este archivo)  
- [x] Pack script + dataset manifest  
- [x] Stage1 sample API + unit tests (17 green with redesign_v2)  
- [x] Notebook `KAGGLE_GPU_RUN.py` (Stage0–1)  
- [x] Dataset Kaggle privado **subido**: `alonsoalviraaaa/trad-longhist-2010`  
- [x] Protocolo Stage3 documentado (gates CAGR>10%, MDD≥−65%)  
- [x] DECISION / freeze: no auto-freeze  
- [ ] Usuario: Run All en notebook GPU + descargar stage1_top  
- [ ] Stage2–3 full en sesión(es) siguientes  

### PR plan (ejecutado local 2026-07-25)

| PR | Estado |
|----|--------|
| PR1 grids + math + stage1_scorer + tests | **done** |
| PR2 precompute_features + KAGGLE_GPU_RUN notebook | **done** |
| PR3 pack_kaggle_redesign + dataset upload CLI | **done** (dataset created) |
| PR4 Stage2/3 adapters on Kaggle | pending next session |

**Launch:** `kaggle_redesign/LAUNCH.md`

---

## 15. Disclaimer

Research software. Kaggle GPU results are not live trading signals. Past OOS does not guarantee future results. Paper freeze remains human-gated.

**Research only. Not financial advice.**
