# PROMPT MAESTRO — TRAD Equity ML

Copia/pega al iniciar una sesión seria:

```text
Use trad-local + loop-engineering best practices.
Read AGENTS.md and docs/11_plan_implementacion_modular.md first.
Work only on module <MODULE-ID>: <one-sentence goal>.
No look-ahead leakage. Prefer purged/walk-forward validation.
Keep Lean feature parity (lean_strategy/modules/config.py) unless the design explicitly migrates counts.
Update docs/11 history when done. Run pytest.
```

## Ejemplos

**Unificar features**

```text
Use trad-local + loop-engineering. Module FEA-01: single feature engine shared by train, research backtest, and Lean. Design first, then implement with tests for vector parity.
```

**Triple barrier + dataset**

```text
Use trad-local + loop-engineering. Module LAB-01: production Triple Barrier API with versioned k_tp/k_sl/horizon, synthetic tests, export dataset for MOD-01.
```

**Validación OOS**

```text
Use trad-local + loop-engineering. Module VAL-01: reusable purged CV + walk-forward harness; integrate backtest-expert red flags.
```

**Meta-labeling**

```text
Use trad-local + loop-engineering. Module MOD-02 following PLAN_METALABELING_7_DIAS.md and docs/11; measure OOS win-rate vs primary alone with realistic costs.
```

## Slash shortcuts

| Command | Purpose |
|---------|---------|
| `/loop-engineering <task>` | Full quality loop |
| `/design <spec>` | Spec + PR plan |
| `/execute-plan docs/design/...` | Multi-PR isolated implement |
| `/check-work` | Verification gate |
| `/trad-local` | Force project skill |
| `/backtest-expert` | Robustness methodology |
| `/feature-engineering` | ML features |
| `/walk-forward-validation` | OOS protocol |
