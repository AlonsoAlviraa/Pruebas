# MEGA-PLAN: Paper Live Year — simulación de 12 meses sin dinero real

**Date:** 2026-07-21  
**Module family:** LIV-01 … LIV-08 (+ RSK-02, BKT-03, OPS-01)  
**Goal:** Correr el candidato research **todo un año** en modo paper, con:

1. **Entradas/salidas en tiempos reales de mercado** (no solo open/close del día del backtest research).  
2. **Comisiones, slippage y spreads** contados en cada fill.  
3. **Ledger completo** de cada decisión, orden, fill, skip y estado de cartera.  
4. **Cero capital real** (paper / sim / shadow).  
5. **Análisis continuo** (diario / semanal / mensual) para ver cómo “funcionaría” en vivo.

**Strategy freeze (paper primary):** `turbo_highvol_minalloc`  
**Shadow optional:** `turbo_highvol_minalloc_sector_rot`  
**Explicit non-goal year-1:** multi-mercado, sizing real, broker live orders.

---

## 0. Principios no negociables

| # | Principio | Detalle |
|---|-----------|---------|
| P0 | **No real money** | `mode=paper` hard-coded; no API keys de trading live en default config |
| P1 | **Causality** | Señales solo con info disponible en el timestamp de decisión |
| P2 | **Parity intent** | Misma lógica de señal/risk que research *hasta donde el tiempo real lo permita*; diferencias documentadas |
| P3 | **Every event is logged** | Si no está en el ledger, no existió |
| P4 | **Costs always on** | Commission + slippage + (opcional) borrow/short fee = 0 si long-only |
| P5 | **Kill switches** | Paper se detiene (freeze entries) si se rompen gates de riesgo |
| P6 | **Research ≠ Live clock** | Research EOD es upper-bound; paper year mide el gap |

**Disclaimer:** Research + paper. No es consejo financiero. OOS pasado ≠ futuro.

---

## 1. Por qué el backtest actual no basta

| Research hoy (`trad_research/backtest.py`) | Live real / paper realista |
|-------------------------------------------|----------------------------|
| Barras **diarias** | Intradía (1m–5m) o al menos **decision times** fijos |
| Fill implícito en **close** del día (o open simulado) | Fill al **next bar open** o marketable limit con slip |
| Universo estático en CSV | Universe + liquidez **as-of** |
| Sin estado de orden (submit/partial/reject) | Order lifecycle completo |
| Equity curve post-hoc | Mark-to-market **intraday** |
| Comisiones agregadas en fill sintético | Commission **por fill** + fee schedule broker |

El paper-year existe para medir el **gap research → ejecución**, no para repetir el bake-off.

---

## 2. Arquitectura objetivo

```
┌─────────────────────────────────────────────────────────────────┐
│                     PAPER LIVE YEAR PLATFORM                      │
├──────────────┬──────────────────┬────────────────┬──────────────┤
│ DATA FEED    │ SIGNAL ENGINE    │ EXEC SIM       │ LEDGER/OPS   │
│ (real-time   │ (research parity │ (paper broker  │ (SQLite +    │
│  delayed OK) │  + schedule)     │  no real $)    │  parquet)    │
└──────┬───────┴────────┬─────────┴───────┬────────┴──────┬───────┘
       │                │                 │               │
       ▼                ▼                 ▼               ▼
  bars / quotes    decisions.jsonl    orders/fills     dashboards
  universe PIT     signals.parquet    positions        weekly MD
```

### 2.1 Dos motores (elegir uno primario + uno shadow)

| Motor | Rol | Pros | Contras |
|-------|-----|------|---------|
| **A. Lean / QuantConnect Paper** | Producción-style; ya hay `lean_strategy/` con schedule 30m after open / 15m before close | Broker model, fees, fill models nativos; paper cloud | Paridad con `turbo_highvol_minalloc` incompleta (hoy es M1/M2 hierarchical, no minalloc research) |
| **B. Local Paper Engine (recomendado year-1)** | `paper_live/` en este repo: reusa `trad_research` features/labels/strategies + loop intradía | Control total del ledger; misma config minalloc; offline replay | Hay que construir feed + fill sim |
| **C. Broker paper (Alpaca/IBKR paper)** | Órdenes paper en broker real | Slippage más real | Acoplamiento API; keys; más riesgo de “accidental live” |

**Recomendación del plan:**

1. **Primario year-1:** **Motor B** (local paper) + replay y live delayed.  
2. **Secundario:** portar reglas a **Lean paper** cuando parity BKT-02/LIV esté hecha.  
3. **No** Motor C con live keys hasta 6 meses de paper limpio.

---

## 3. Modelo de reloj y entradas “reales” (no solo inicio/fin de día)

### 3.1 Sesión US equities (ET)

| Evento | Hora (default) | Acción |
|--------|----------------|--------|
| Pre-open data | 09:00 | Refresh quotes, corporate actions, halts |
| **RTH open** | 09:30 | No entries primeros N minutos (auction noise) |
| **Entry window** | 09:45 – 10:30 (configurable) | Scan señales en **barra 1m o 5m**; submit orders |
| Intraday risk | cada 1–5 min | Hard/trail stop, DD kill, cancel stale |
| Midday optional | 12:00 | Re-scan rotation (si enable_rotation) |
| **Exit check** | 15:30 – 15:50 | Time-stop / eod policy |
| **Force flatten optional** | 15:55 | Solo si `eod_force` policy ON |
| Post-close | 16:15 | Snapshot equity, write daily report |
| Night | 18:00 | Retrain job (opcional semanal, no diario al inicio) |

### 3.2 Política de fill (realista, documentada)

| Tipo | Regla paper |
|------|-------------|
| Entry | Señal en barra `t` → orden **marketable** al open de `t+1` (1m/5m) |
| Slippage | `max(fixed_bps, half_spread_est, k * ATR_1m)` |
| Partial fills | Si notional > ADV% cap → fill fraccionado en N barras |
| Reject | Halt, SSR short (N/A long), no quote, price &lt; min_price |
| Exit stop | Stop **market** en breach; slip extra `stop_slip_bps` |

### 3.3 Resoluciones

| Fase | Bar size | Motivo |
|------|----------|--------|
| Mes 0–1 (shadow dry-run) | Daily + decision times fijos (10:00, 15:45) | Smoke barato |
| Mes 1–3 | **5-minute** | Balance CPU/realismo |
| Mes 3–12 | 5m entries + **1m** stop monitor | Stops más honestos |
| Opcional | 1m full | Solo si infra aguanta |

**No** se pretende HFT ni level-2 market making.

---

## 4. Costos: comisiones y fricciones (siempre ON)

### 4.1 Schedule de fees (configurable, versionado)

Archivo: `paper_live/config/cost_model.yaml`

```yaml
version: cost-v1
broker_profile: ibkr_pro_tiered_like   # o flat_alpaca_like
commission:
  per_share: 0.005          # $
  min_per_order: 1.00
  max_pct_of_notional: 0.005
slippage:
  entry_bps: 5              # 0.05%
  exit_bps: 5
  stop_extra_bps: 10        # stops peores
  impact_bps_per_adv_pct: 2 # si size/ADV crece
spread:
  use_quote_if_available: true
  fallback_bps: 2
sec_fee_sell_only: true     # approx US
finra_taf: true
min_price: 2.0
max_participation_rate: 0.02  # 2% ADV day
```

### 4.2 Contabilidad por fill

Cada fill escribe:

- `gross_notional`, `commission`, `sec_fee`, `slippage_cost`, `net_cash_delta`
- `cost_bps_vs_mid` (si hay mid)

Equity paper:

```
cash -= (buy_notional + commission + fees)
cash += (sell_notional - commission - fees)
equity = cash + Σ shares * mark_price
```

### 4.3 Comparación de fricción

Dashboard mensual:

| Metric | Research EOD | Paper live |
|--------|--------------|------------|
| Turnover | … | … |
| Total commissions $ | … | … |
| Slippage $ | … | … |
| Cost drag (annualized) | … | … |
| Net CAGR vs gross | … | … |

---

## 5. Ledger: guardar **cada** cosa (sin dinero real)

### 5.1 Store

| Store | Path | Contenido |
|-------|------|-----------|
| SQLite (operational) | `paper_live/ledger/paper_year.db` | orders, fills, positions, events, daily_nav |
| Parquet (analytics) | `paper_live/ledger/parquet/` | signals, decisions, equity ticks |
| JSONL (append-only audit) | `paper_live/ledger/audit/YYYY-MM-DD.jsonl` | immutable event log |
| Snapshots | `paper_live/ledger/snapshots/` | end-of-day portfolio JSON |

**Regla:** el JSONL es append-only; nunca se reescribe un día cerrado (correcciones = eventos `correction`).

### 5.2 Esquema de eventos (mínimo)

```text
event_id, ts_utc, event_type, strategy_id, run_id, payload_json
```

**event_type enum:**

| event_type | Cuándo |
|------------|--------|
| `session_open` / `session_close` | Cada día de mercado |
| `bar` | (opcional sample) |
| `signal_computed` | Tras features + model |
| `entry_candidate` | Pasa filtros soft |
| `entry_rejected` | Motivo: regime / min_alloc / sector / slots / kill |
| `order_submitted` | Paper order |
| `order_ack` / `order_reject` | Sim broker |
| `fill` | Partial o full |
| `position_opened` / `position_updated` / `position_closed` | Estado |
| `stop_updated` | Trail/hard |
| `risk_block` | DD kill, ticker cap |
| `retrain_start` / `retrain_end` | Si hay retrain |
| `daily_nav` | EOD equity |
| `kill_switch` | Freeze entries |
| `heartbeat` | Watchdog proceso vivo |

### 5.3 Tablas SQL

```sql
runs(run_id, started_at, strategy, config_hash, mode='paper', capital0)
orders(order_id, run_id, ts, ticker, side, qty, order_type, limit_px, status, reason)
fills(fill_id, order_id, ts, qty, price, commission, fees, slippage_bps, liquidity)
positions(run_id, ticker, qty, avg_px, stop, hard_stop, opened_at, bars_held, meta_json)
nav_daily(run_id, date, equity, cash, gross_exposure, dd_from_peak, n_positions)
decisions(decision_id, ts, ticker, action, p_buy, score, filters_json, config_hash)
costs_daily(run_id, date, commission, fees, slippage_est, turnover)
```

### 5.4 Idempotencia y re-arranque

- `run_id` + `config_hash` fijos al arrancar el año.  
- Al crash: reload positions desde último snapshot; **no** re-simular fills del pasado.  
- Heartbeat cada 60s → alert si &gt; 5 min sin tick en RTH.

---

## 6. Motor de señal y parity con research

### 6.1 Source of truth (year-1)

| Componente | Source |
|------------|--------|
| Strategy knobs | `get_strategy("turbo_highvol_minalloc")` + frozen YAML snapshot |
| Features daily | `trad_research.features` (mismas columnas M2) |
| Features intradía | Subset + ATR/stops en 5m; **no** re-inventar 17 features en 1m el día 1 |
| Regime | QQQ dual MA / strict_dual_golden as research |
| Sizing | vol target + min_alloc 1.5% + max_positions |
| Stops | hard + chandelier + time_stop (+ adaptive if enabled) |

### 6.2 Gap de parity conocido (documentar en `parity_matrix.md`)

| Item | Research | Paper live | Gap risk |
|------|----------|------------|----------|
| Signal bar | Daily close | Decision at 09:45–10:30 on prior daily signal OR rolling | Alto si se usa solo close de ayer |
| Train cadence | 1×/año OOS | **Weekly or monthly** retrain expanding window | Medio |
| Universe | Fixed file | Monthly re-score highvol **causal** (no dynamic pure-vol fail) | Medio |
| Sector rot | Optional shadow | Shadow book B | Bajo |

### 6.3 Señal daily → ejecución intradía (default recomendado mes 1–3)

1. **Post-close D-1 (16:30):** compute features, regime, candidate list for day D.  
2. **09:45 D:** confirm price still above hard filters (gap risk, halt, min_price).  
3. **Submit entries** staggered (TWAP-lite: 3 clips en 15 min) si size grande.  
4. **Intraday:** manage stops only.  
5. **15:45:** time-stop / trail check.

Esto da “entradas reales en el reloj del mercado” sin requerir ML intradía maduro el día 1.

### 6.4 Evolución (mes 4+)

- Recalcular score con barras 5m de la mañana (momentum confirm).  
- Entry only if 5m trend filter OK (optional A/B book).

---

## 7. Risk paper (obligatorio)

| Control | Valor default paper year |
|---------|---------------------------|
| Initial virtual capital | $100,000 |
| max_positions | como strategy (p.ej. 8–12) |
| min_alloc_pct | **1.5%** |
| max_position_pct | strategy default |
| Portfolio DD kill (block entries) | **18%** from peak |
| Soft de-risk | 50% size at 9% DD |
| Kill paper if | DD from start &lt; **−15%** OR 20d Sharpe &lt; **−1** |
| Max daily new entries | 5 |
| Max participation | 2% ADV |
| Long only | true |
| No leverage &gt; 1.0 | true year-1 |

Kill switch escribe `kill_switch` event y deja de abrir; puede seguir cerrando por stops.

---

## 8. Datos en vivo (sin dinero real)

### 8.1 Feeds aceptables

| Feed | Uso | Notas |
|------|-----|-------|
| EODHD delayed / live (si plan) | Quotes + EOD | Ya usado en research |
| Polygon / Alpaca **data only** | 1m/5m | Solo market data keys, no trading keys |
| Yahoo/yfinance | Backup only | No confiar para paper serio |
| Local replay | Nights/weekends | Validate pipeline |

### 8.2 Corporate actions

- Adjusted series para señales.  
- Corporate action log diario: split, div, symbol change → adjust position qty/avg.  
- Delist: force close residual cash (misma regla DAT-04).

### 8.3 Universo paper

- Start: `universe_longhist100` ∩ liquid (ADV20 &gt; $5M, price ≥ $5).  
- Monthly: re-score **causal** highvol-quality hybrid (no pure max-vol).  
- Always include SPY/QQQ for regime (no trade unless configured).

---

## 9. Operación del año (calendario)

### Fase 0 — Foundation (semanas 1–2) — **LIV-01, LIV-02**

- [ ] Repo package `paper_live/`  
- [ ] Config freeze YAML de minalloc + cost_model  
- [ ] SQLite + JSONL ledger  
- [ ] Sim broker paper (submit/fill/cancel)  
- [ ] Unit tests sintéticos (no market)  
- [ ] Dry-run 5 días replay histórico 5m  

**Done when:** replay produce fills + commissions + NAV sin crash; pytest green.

### Fase 1 — Shadow paper (semanas 3–6) — **LIV-03**

- [ ] Live delayed feed RTH  
- [ ] Daily signal post-close + entry 09:45  
- [ ] Stops intradía 5m  
- [ ] Heartbeat + alertas (email/Telegram/log file)  
- [ ] Dashboard HTML diario  
- [ ] **No** retrain aún (modelo fixed o último export Lean)  

**Done when:** 20 días de sesión con ledger completo y 0 data gaps &gt; 10 min sin handle.

### Fase 2 — Full paper year start (mes 2–3) — **LIV-04, RSK-02**

- [ ] Kill switches armed  
- [ ] Weekly metrics vs SPY + research shadow EOD  
- [ ] Cost drag report  
- [ ] Optional shadow book sector_rot  
- [ ] Monthly universe refresh causal  

**Done when:** 60 trading days paper protocol from LIVE_CANDIDATE satisfied for “stability review”.

### Fase 3 — Hardening (mes 4–6) — **LIV-05, BKT-03**

- [ ] 1m stop monitor  
- [ ] Partial fills + ADV caps  
- [ ] Weekly retrain expanding (embargo 5d) **or** monthly (safer)  
- [ ] Parity report research-EOD vs paper fills gap  
- [ ] PIT membership filter on live universe (DAT-04 lite)  

**Done when:** gap analysis document + no unexplained PnL.

### Fase 4 — Year completion & decision (mes 7–12) — **LIV-06, LIV-07**

- [ ] Quarterly deep audit (like mega trade audit)  
- [ ] Crisis protocol if DD kill trips  
- [ ] End-of-year report: net vs gross, vs SPY, vs research expectations  
- [ ] Go / No-Go real capital (default **No-Go** unless gates pass)  

**Done when:** `reports/PAPER_YEAR_FINAL.md` + archived ledger.

### Fase 5 (opcional paralelo) — Lean paper parity — **LIV-08**

- [ ] Port minalloc knobs into `lean_strategy`  
- [ ] QC paper node or local lean live paper  
- [ ] Compare local engine vs Lean fills for 30 days  

---

## 10. Módulos técnicos a construir

| ID | Módulo | Path propuesto | Dependencias |
|----|--------|----------------|--------------|
| **LIV-01** | Paper config + cost model | `paper_live/config/` | — |
| **LIV-02** | Ledger + event bus | `paper_live/ledger/` | LIV-01 |
| **LIV-03** | Market data adapter | `paper_live/datafeed/` | LIV-01 |
| **LIV-04** | Signal service (daily→intraday) | `paper_live/signals/` | trad_research, LIV-03 |
| **LIV-05** | Paper broker / OMS | `paper_live/oms/` | LIV-02, cost model |
| **LIV-06** | Risk & kill switch | `paper_live/risk/` | LIV-05, RSK |
| **LIV-07** | Scheduler / runner RTH | `paper_live/runner.py` | all |
| **LIV-08** | Reporting & dashboards | `paper_live/reports/` + `scripts/run_paper_daily_digest.py` | ledger |
| **BKT-03** | Research vs paper gap audit | `scripts/audit_paper_vs_research.py` | LIV-08 |
| **OPS-01** | Process watchdog + backups | `paper_live/ops/` | LIV-02 |
| **RSK-02** | Unified risk constants paper/research | `trad_research` + paper | RSK-01 |

### Layout de carpetas

```text
paper_live/
  README.md
  config/
    strategy_freeze.yaml      # hash-pinned knobs
    cost_model.yaml
    schedule.yaml
    universe.yaml
  datafeed/
    base.py
    eodhd_live.py
    polygon_bars.py
    replay.py
  signals/
    daily_pipeline.py
    entry_confirm.py
  oms/
    paper_broker.py
    order_types.py
    fill_model.py
  risk/
    kill_switch.py
    portfolio_risk.py
  ledger/
    db.py
    events.py
    snapshots.py
  reports/
    daily_digest.py
    weekly_scorecard.py
  runner.py                   # main RTH loop
  shadow_book.py              # second strategy virtual
tests/
  test_paper_live_ledger.py
  test_paper_fill_costs.py
  test_paper_replay_smoke.py
```

---

## 11. PR Plan (DAG ejecutable)

```text
PR1 LIV-01+02  config freeze + ledger + events + unit tests
    │
    ├─► PR2 LIV-05     paper broker + cost model fills
    │       │
    │       └─► PR3 LIV-03+04  replay datafeed + daily signal→entry
    │               │
    │               └─► PR4 LIV-06+07  risk + runner + 5-day replay integration
    │                       │
    │                       ├─► PR5 LIV-08  daily/weekly digests + HTML
    │                       │
    │                       └─► PR6 OPS-01  watchdog, backup, heartbeat alerts
    │
    └─► PR7 BKT-03     gap audit research vs paper (can start after PR4)
            │
            └─► PR8 LIV-08 Lean parity spike (optional, after 60 paper days)
```

Cada PR: tests + no real-money paths + update `docs/11`.

---

## 12. Análisis continuo (qué mirar todo el año)

### Diario (auto, post-close)

- NAV, DD, n positions, trades del día  
- Commissions del día  
- Rejects top reasons  
- Data gaps  
- Kill switch status  

### Semanal

| Metric | Gate paper (soft) |
|--------|-------------------|
| Excess vs SPY (week) | Informational |
| Rolling 20d Sharpe | kill if &lt; −1 |
| Win rate / PF | vs research band |
| Cost drag bps | flag if &gt; 2× research assumption |
| Turnover | flag if &gt; 3× research |
| Micro trades | must stay ~0% |
| Slippage realized vs model | recalibrate |

### Mensual

- Full trade audit (exit reasons pie)  
- Concentration (top 5 PnL names)  
- Regime days risk-off fraction  
- Research shadow EOD backtest same month vs paper PnL **gap**  
- Update `reports/paper_year/YYYY-MM.md` |

### Trimestral

- Go/No-Go checklist  
- Optional config change → **new run_id** (no silent mutación)  

### Anual

| Pregunta | Criterio de “funcionaría” |
|----------|---------------------------|
| ¿Net paper &gt; 0 tras costs? | Sí/No |
| ¿Excess vs SPY &gt; 0? | Sí/No (honest) |
| ¿DD kill tripped? | Count + recovery |
| ¿Gap research-paper &lt; X%? | X=30% relative opcional |
| ¿Ops uptime RTH &gt; 99%? | Sí/No |
| ¿Promover capital real? | Default **NO** si falta cualquiera de: excess, DD control, ops, 12m complete |

---

## 13. Seguridad anti “dinero real”

| Control | Implementación |
|---------|----------------|
| `TRAD_PAPER_ONLY=1` env required | runner exits if not set |
| No live order endpoints in default adapters | trading client not imported |
| Separate credentials file for **data only** | `secrets/data_keys.env` gitignored |
| Code review gate: ban `submit_order` to real broker | CI grep |
| Capital label always `VIRTUAL` in reports | UI watermark |

---

## 14. Métricas de éxito del mega-plan (no de la strategy)

El plan se considera **infra DONE** cuando:

1. 20 días RTH paper con ledger 100% eventos.  
2. Cada fill tiene commission &gt; 0 (salvo fee schedule free).  
3. Restart recovery sin duplicar posiciones.  
4. Daily digest auto.  
5. Tests: ledger, costs, replay smoke.  
6. Documento de gap research vs paper mes 1.

El plan se considera **year SUCCESS (análisis)** cuando:

1. 12 meses de NAV + trades archivados.  
2. Informe final con costs breakdown.  
3. Decisión documentada Go/No-Go real capital.

---

## 15. Estimación de esfuerzo

| Fase | Esfuerzo (1 dev half-time) |
|------|----------------------------|
| PR1–4 foundation + replay | 2–3 semanas |
| PR5–6 digests + ops | 1 semana |
| Live delayed feed hardening | 1–2 semanas |
| 60-day stability | calendar time |
| Rest of year | maintenance ~2–4 h/semana + monthly audit |

---

## 16. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Feed cae en RTH | Heartbeat + freeze new entries; keep stops if last quote fresh |
| Overfit retrain semanal | Empezar monthly; embargo; log model hash |
| Paper fill demasiado optimista | Calibrate slip up after month 1; use worse-of mid/last |
| Operador “sube a real” por euforia | Written No-Go until 12m; AGENTS.md rule |
| CPU 1m universe 80 | Tier universe: trade top 40, watch 80 |
| Diferencia Lean vs research | LIV-08 only after local stable |

---

## 17. Relación con AGENTS.md

AGENTS dice: *Live/paper trading out of scope until OOS + stress gates pass.*

**Interpretación de este plan:**

- Paper year es **investigación operacional**, no “live trading”.  
- Capital real sigue **out of scope**.  
- Gates ya tienen paper pick + stress + PIT honesty; paper year **añade** evidencia de ejecución, no la salta.  
- Si paper year falla, se **cancela** cualquier path a real money.

---

## 18. Primeros comandos (target API — a implementar)

```powershell
# Freeze config + init ledger
python -m paper_live.runner init --strategy turbo_highvol_minalloc --capital 100000

# Replay 5 trading days 5m (no live)
python -m paper_live.runner replay --from 2024-06-03 --to 2024-06-07 --bar 5m

# Start paper RTH loop (requires TRAD_PAPER_ONLY=1)
$env:TRAD_PAPER_ONLY=1
python -m paper_live.runner live --bar 5m

# Daily digest
python scripts/run_paper_daily_digest.py --run-id latest

# Gap audit vs research EOD shadow
python scripts/audit_paper_vs_research.py --month 2026-08
```

---

## 19. Checklist de arranque (semana 1)

1. Congelar strategy YAML + cost_model + hash en git.  
2. Implementar LIV-01/02 (ledger).  
3. Implementar paper broker + 1 test de commission.  
4. Replay 5 días.  
5. Revisar que **cada** order/fill/reject esté en JSONL.  
6. Solo entonces conectar feed live delayed.

---

## 20. Resumen en una frase

**Construir un broker paper local con reloj de mercado real, fees por fill, ledger append-only y digests, corriendo `turbo_highvol_minalloc` un año sin dinero real, para medir el gap research→ejecución antes de cualquier capital.**

---

## PR Plan (copy for execute-plan)

| PR | Title | Depends | Modules |
|----|-------|---------|---------|
| 1 | Paper config + append-only ledger | — | LIV-01, LIV-02 |
| 2 | Paper OMS + cost model fills | 1 | LIV-05 |
| 3 | Replay feed + daily signal→intraday entry | 1–2 | LIV-03, LIV-04 |
| 4 | Risk/kill + RTH runner integration | 2–3 | LIV-06, LIV-07 |
| 5 | Daily/weekly digests + HTML | 4 | LIV-08 |
| 6 | Watchdog, backups, alerts | 4 | OPS-01 |
| 7 | Research vs paper gap audit | 4–5 | BKT-03 |
| 8 | Optional Lean paper parity spike | 5+60d | LIV-08 |

**Verify each PR:** `python -m pytest tests/test_paper_live*.py -q` + manual replay smoke.
