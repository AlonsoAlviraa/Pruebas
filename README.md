# TRAD — Equity ML Research Platform

Plataforma de investigación para **trading sistemático de acciones** con machine learning, meta-labeling, validación walk-forward/purged y ejecución vía Lean/QuantConnect, más un path experimental de DRL (RLlib).

## Inicio rápido (agentes / loop-engineering)

1. Abre este repo en Grok Build (cwd = raíz del repo).
2. Las skills de trading están en `.grok/skills/` (128 skills equity/ML/datos/quant).
3. Lee **`AGENTS.md`** y **`docs/11_plan_implementacion_modular.md`**.
4. Trabaja con el loop:

```text
/loop-engineering Unificar features y triple barrier (FEA-01 + LAB-01)
```

O en cualquier prompt:

```text
Use trad-local + loop-engineering. Read AGENTS.md and docs/11 first.
```

### Skills clave

| Skill | Uso |
|-------|-----|
| `trad-local` | Reglas del proyecto (siempre) |
| `loop-engineering` | design → implement → review → check-work |
| `backtest-expert` | Robustez de backtests |
| `feature-engineering` | Features ML |
| `walk-forward-validation` | OOS |
| `vectorbt` / `backtrader` | Motores de backtest |
| `llmquant-equities` / `llmquant-data` | Workflows de research |

Catálogo: `docs/skills/INSTALLED_SKILLS.txt` · Fuentes: `docs/skills/SOURCES.md`  
Clones completos (incl. crypto no filtrado): `_vendor_skills/`

## Componentes de código

| Componente | Descripción |
|------------|-------------|
| `data/` | Cache OHLCV por ticker |
| `download_data.py` | Descarga y filtros de calidad |
| `triple_barrier_labeling.py` | Etiquetado Triple Barrier |
| `train_signal_model_v2.py` | Entrenamiento señales / meta |
| `run_backtest_signal_v2.py` | Backtest research |
| `lean_strategy/` | Estrategia Lean (M1+M2) |
| `drl_platform/` + `main.py` | DRL multi-activo (RLlib) |
| `models/` | Checkpoints |
| `tests/` | Tests sintéticos |

## Setup Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/ -q
python setup_env.py
```

Dependencias pesadas (Ray/RLlib, Lean) son opcionales según el path que uses.

## Principios de research

- Sin look-ahead; CV purgada/embargo o walk-forward.
- Costos realistas; comparar vs benchmark (SPY).
- Paridad de features entre train, backtest research y Lean.
- No es consejo financiero.

## Plan modular

Ver `docs/11_plan_implementacion_modular.md` para estados FEA/LAB/MOD/VAL/BKT y el orden DAG.
