# Design: Mega alpha post-mortem + realistic next cycle

**Status:** implementing  
**Date:** 2026-07-22  
**Capital:** VIRTUAL  

## Finding

Strict `max(SPY,QQQ,IWM)+3pp` **every year** yielded **0** strategies. Root cause: QQQ was the best index in 2023/2024/2025_study; long-only cannot systematically clear QQQ+3pp (even QQQ hold fails).

## Artifacts

| Path | Role |
|------|------|
| `scripts/rescore_mega_annual.py` | Offline multi-filter rescore |
| `reports/mega_annual_alpha/RESCORE.md` | Realistic promote/watch/kill |
| `zoo_mega_alpha.json` | Audit mislabeled SPY proxy; regime/NVDA/meta sleeves |

## MCP tasks

Weekly reminder only (`tasks` server) — does not run local Yahoo backtests.

## Options

Out of scope for index-beating alpha; keep income scorecard separately.
