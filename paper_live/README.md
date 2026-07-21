# paper_live — Paper Live Year (virtual capital only)

**No real money.** Mode is hard-coded to `paper`. Live broker order paths are out of scope for LIV-01/02.

Design: `docs/design/2026-07-21_paper_live_year_mega_plan.md`

## LIV-01 — Config freeze

| File | Role |
|------|------|
| `config/strategy_freeze.json` | Strategy id + knobs + risk paper (minalloc baseline) |
| `config/cost_model.json` | Commission, slippage bps, SEC/TAF, participation |
| `config/schedule.json` | RTH clock (entry 09:45–10:30 ET, etc.) |
| `config/universe.json` | Ticker file, liquidity floors |

```python
from paper_live import load_freeze, compute_config_hash

freeze = load_freeze()
print(freeze.config_hash)
print(freeze.strategy.strategy_id)  # turbo_highvol_minalloc
print(freeze.cost.estimate_commission(100, 50.0))
```

Any change to freeze files → new `config_hash` → start a **new** `run_id`.

## LIV-02 — Ledger

Dual write:

- **SQLite** `ledger_data/paper_year.db` — queryable orders/fills/positions/nav
- **JSONL** `ledger_data/audit/YYYY-MM-DD.jsonl` — append-only audit (never rewrite closed days)
- **Snapshots** `ledger_data/snapshots/` — crash recovery

```python
from paper_live import load_freeze, PaperLedger

freeze = load_freeze()
with PaperLedger.create_run("paper_live/ledger_data", freeze) as led:
    oid = led.record_order(ticker="AAPL", side="buy", qty=10)
    led.record_fill(
        order_id=oid,
        ticker="AAPL",
        side="buy",
        qty=10,
        price=190.0,
        commission=freeze.cost.estimate_commission(10, 190.0),
    )
    led.upsert_position(ticker="AAPL", qty=10, avg_px=190.0)
    led.record_nav_daily("2026-07-21", equity=100_000, cash=98_100, n_positions=1)
```

### Init CLI

```powershell
python -m paper_live.cli_init --ledger-root paper_live/ledger_data
```

### Paper-only guard

```python
from paper_live import assert_paper_only
assert_paper_only(require_env=False)           # blocks TRAD_TRADING_MODE=live
# assert_paper_only(require_env=True)          # runner: needs TRAD_PAPER_ONLY=1
```

## Tests

```powershell
python -m pytest tests/test_paper_live_ledger.py -q
```

## LIV-05 — Paper OMS + fill model

```python
from paper_live import load_freeze, PaperLedger, PaperBroker, FillQuote

freeze = load_freeze()
led = PaperLedger.create_run("paper_live/ledger_data", freeze)
broker = PaperBroker(freeze.cost, capital0=freeze.strategy.capital0, ledger=led)

order, fills = broker.submit_and_execute(
    "AAPL", "buy", 100, FillQuote(mid=190.0, adv_shares=50_000_000)
)
# cash -= notional + commission (+ fees on sells)
# ledger records order_submitted, order_ack, fill (commission, slippage_bps, net_cash_delta)

broker.submit_and_execute("AAPL", "sell", 100, FillQuote(mid=192.0))
print(broker.state.to_dict())  # equity, total_commission, total_fees, VIRTUAL label
```

| Piece | Role |
|-------|------|
| `FillModel` | Slip price, commission, SEC/TAF, ADV cap, TWAP clips |
| `PaperBroker` | submit / ack / execute / cancel; cash + positions |
| `FillQuote` | mid/bid/ask/adv/halt |

Rejects: halt, min_price, kill_switch, short (long_only), insufficient cash/shares, limit not marketable.

## LIV-03 / LIV-04 — Daily replay + signal → entry

```python
from paper_live import (
    load_freeze, PaperLedger, DailyReplayFeed, ReplaySession
)

freeze = load_freeze()
feed = DailyReplayFeed.from_synthetic(["AAA", "BBB", "QQQ"], n_days=400, seed=1)
# or: DailyReplayFeed.from_data_root("data", ["AAPL", "MSFT", "QQQ", "SPY"])

led = PaperLedger.create_run("paper_live/ledger_data_replay", freeze)
session = ReplaySession(feed, freeze, ledger=led)
result = session.run("2020-06-01", "2020-08-31")
print(result.to_dict())
```

**Flow (per session day D):**
1. **Open** — confirm candidates from close D−1 (gap/min price) → `PaperBroker` buy at open  
2. **Day** — stop vs low; trail; time-stop at horizon  
3. **Close** — NAV + generate candidates for D+1 (causal features only)

```powershell
python -m paper_live.cli_replay --synthetic --from 2020-06-01 --to 2020-08-31
python -m paper_live.cli_replay --from 2024-01-02 --to 2024-03-29 --tickers AAPL,MSFT,QQQ,SPY
```

## LIV-06 — Risk + kill switch

| Control | Default (freeze) |
|---------|------------------|
| Portfolio DD kill (block entries) | **18%** from peak |
| Soft de-risk size scale | 50% size from **9%** DD |
| Hard kill from start | **−15%** |
| Rolling 20d Sharpe kill | **&lt; −1** |
| Sticky | hard kill stays on for the run |

```python
from paper_live import PaperRunner, build_runner

runner = build_runner(ledger_root="paper_live/ledger_data", tickers=["AAA","QQQ"], synthetic=True)
out = runner.run_replay("2020-06-01", "2020-08-01")
print(out.kill_state, out.risk_last)
```

## LIV-07 — RTH runner

- `ScheduleClock` — maps wall time → phase (entry_window, exit_check, …)
- `PaperRunner.run_replay` — multi-day session + risk
- `tick_live` / `run_live_day_stub` — require `TRAD_PAPER_ONLY=1`

```powershell
python -m paper_live.cli_runner --mode replay --synthetic --from 2020-06-01 --to 2020-07-15
$env:TRAD_PAPER_ONLY=1
python -m paper_live.cli_runner --mode live-stub --synthetic --day 2020-06-15
```

## LIV-08 — Daily / weekly digests + HTML

```powershell
# After a paper run with ledger:
python scripts/run_paper_daily_digest.py --ledger-root paper_live/ledger_data_runner_demo

# Or in code:
from paper_live import PaperLedger, generate_reports_for_run
led = PaperLedger.open_run("paper_live/ledger_data", "paper_...")
bundle = generate_reports_for_run(led, "reports/paper_year/demo")
# → daily/*.md|json, weekly/*.md|json, dashboard.html, INDEX.md
```

| Output | Content |
|--------|---------|
| `daily/YYYY-MM-DD.md` | NAV, DD, fills, commissions, rejects, kill events |
| `weekly/WS_WE.md` | Week return, Sharpe approx, cost drag bps, micro%, flags |
| `dashboard.html` | Dark self-contained HTML (equity sparkline + tables) |

## Status

| Module | Status |
|--------|--------|
| LIV-01 config freeze | DONE (PR1) |
| LIV-02 ledger | DONE (PR1) |
| LIV-05 paper OMS + fills | DONE (PR2) |
| LIV-03 replay datafeed | DONE (PR3) |
| LIV-04 daily signal → entry | DONE (PR3) |
| LIV-06 risk / kill | DONE (PR4) |
| LIV-07 RTH runner | DONE (PR4) |
| LIV-08 digests + HTML | DONE (PR5) |
| **Cloud free multi-strat** | **GitHub Actions daily** |

## Free cloud (no local PC)

10 strategies run **on GitHub Actions** and save under `reports/paper_cloud/`:

```text
reports/paper_cloud/latest/SUMMARY.md      ← ranking diario
reports/paper_cloud/latest/dashboard.html
reports/paper_cloud/history/YYYY-MM-DD/strategies/<id>/
```

- Workflow: `.github/workflows/paper_live_daily.yml` (cron weekdays 21:30 UTC)
- Zoo: `paper_live/cloud/strategy_zoo.json`
- Docs: `reports/paper_cloud/README.md`

Enable: GitHub → Actions → **Read and write** permissions → Run workflow once.

```powershell
python -m pytest tests/test_paper_*.py tests/test_paper_cloud_batch.py -q
```

Research only. Not financial advice.
