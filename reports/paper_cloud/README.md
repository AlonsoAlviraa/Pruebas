# Paper cloud — multi-strategy study (FREE)

**No local PC required.** Runs on **GitHub Actions** (free for public repos; limited minutes on private).

| | |
|--|--|
| Capital | **VIRTUAL** $100k · mode=`paper` |
| Strategies | **10** rule variants (`paper_live/cloud/strategy_zoo.json`) |
| Data | **Stooq free** daily OHLCV; synthetic fallback if download fails |
| Schedule | Weekdays **21:30 UTC** (+ manual “Run workflow”) |
| Output | `reports/paper_cloud/latest/` + `history/YYYY-MM-DD/` |

## Where to study each day

After each run:

| File | What |
|------|------|
| [`latest/SUMMARY.md`](latest/SUMMARY.md) | Ranking of all 10 strategies |
| [`latest/dashboard.html`](latest/dashboard.html) | Comparison table |
| [`latest/summary.json`](latest/summary.json) | Machine-readable |
| `history/YYYY-MM-DD/strategies/<id>/dashboard.html` | Per-strategy digests |
| [`INDEX.md`](INDEX.md) | History of daily runs |

## Enable on GitHub (one-time)

1. Push this repo to GitHub (already: `AlonsoAlviraa/Pruebas`).
2. **Settings → Actions → General → Workflow permissions → Read and write**.
3. Open **Actions → paper-live-cloud-daily → Run workflow**.
4. Wait ~5–15 min; refresh `reports/paper_cloud/latest/`.

Optional: keep repo **public** for more free Actions minutes.

## Run locally (optional smoke)

```powershell
python scripts/run_paper_cloud_batch.py --out reports/paper_cloud --synthetic
```

## What this is / is not

- **Is:** free cloud paper lab to compare ~10 configs daily and keep history in git.
- **Is not:** broker live trading, real money, or full XGB minalloc parity (rule-based cloud signals).
- **Not financial advice.**

## Strategies (zoo)

See `paper_live/cloud/strategy_zoo.json` — baseline minalloc, no-regime, tight/wide stops, concentrated, diversified, high/low vol, aggressive, defensive.
