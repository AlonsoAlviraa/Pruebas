"""Loop G: purged soft-ban — freeze losers on screen 2018–21, confirm 2022–25.

Protocol (no look-ahead for banlist construction):
  1. Screen trades from minalloc k100 baseline years 2018–21 only.
  2. Freeze top-K worst sum-PnL tickers with n >= min_n.
  3. Confirm 2022–25: baseline vs softban(frozen) vs style EW.
  4. Full path (honest stitch): baseline equity 2018–21 + softban 2022–25.
  5. Promotion on confirm + full purged path.

Research only. Does not change paper freeze.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "vol_fund_mega", ROOT / "scripts" / "run_vol_fund_mega_loop.py"
)
_mega = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["vol_fund_mega"] = _mega
_spec.loader.exec_module(_mega)

from trad_research.metrics import equity_metrics  # noqa: E402
from trad_research.promotion import (  # noqa: E402
    CandidateInput,
    apply_top_k,
    evaluate_candidate,
    scorecard_table,
)
from trad_research.risk_metrics import extended_risk_from_equity  # noqa: E402

STRATEGY = "turbo_highvol_minalloc"
K = 100
STYLE_ID = "growth_ew__volonly_k100_baseline"
BASE_ID = f"{STRATEGY}__volonly_k{K}_baseline"
BAN_ID = f"{STRATEGY}__volonly_k{K}_softban_purged"
IN_SAMPLE_BAN_ID = f"{STRATEGY}__volonly_k{K}_softban_insample_loopf"
PATH_ID = f"{STRATEGY}__volonly_k{K}_path_purged_softban"

# Loop F in-sample ban (for side-by-side honesty only)
LOOP_F_BAN: Tuple[str, ...] = (
    "GSAT",
    "FCEL",
    "NAGE",
    "CDZI",
    "DBVT",
    "XNET",
    "CENX",
    "NVFY",
)


def _cfg(
    config_id: str,
    *,
    strategy: str = STRATEGY,
    ban: Sequence[str] = (),
    lever: str = "baseline",
    label: str = "",
) -> Any:
    return _mega.GridConfig(
        config_id=config_id,
        strategy=strategy,
        growth_hard=False,
        growth_top_k=K,
        lever_id=lever,
        vol_only_top=K,
        vol_pool_n=200,
        label=label,
        exclude_tickers=tuple(str(t).upper() for t in ban),
    )


def build_softban_from_trades(
    trades: pd.DataFrame,
    *,
    screen_years: Sequence[int],
    top_k: int = 8,
    min_n: int = 5,
    pnl_col: str = "net_profit",
    year_col: str = "oos_year",
    ticker_col: str = "ticker",
) -> Tuple[List[str], pd.DataFrame]:
    """Freeze worst sum-PnL tickers from screen years only (pure)."""
    if trades is None or trades.empty:
        return [], pd.DataFrame()
    df = trades.copy()
    if year_col not in df.columns:
        # try entry year
        if "entry_date" in df.columns:
            df[year_col] = pd.to_datetime(df["entry_date"], utc=True, errors="coerce").dt.year
        else:
            raise ValueError("trades need oos_year or entry_date")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df[ticker_col] = df[ticker_col].astype(str).str.upper()
    sub = df[df[year_col].isin(list(screen_years))].copy()
    if sub.empty or pnl_col not in sub.columns:
        return [], pd.DataFrame()
    aggs: Dict[str, Any] = {pnl_col: ["count", "sum"]}
    if "trade_return" in sub.columns:
        aggs["trade_return"] = "mean"
    g = sub.groupby(ticker_col).agg(aggs)
    # flatten multiindex columns
    if isinstance(g.columns, pd.MultiIndex):
        g.columns = [
            "n" if c == (pnl_col, "count") else
            "sum_pnl" if c == (pnl_col, "sum") else
            "mean_ret" if c[0] == "trade_return" else "_".join(str(x) for x in c)
            for c in g.columns
        ]
    else:
        g = g.rename(columns={pnl_col: "sum_pnl"})
    g = g.reset_index()
    if "n" not in g.columns:
        g["n"] = sub.groupby(ticker_col)[pnl_col].count().reindex(g[ticker_col]).to_numpy()
    g = g[g["n"] >= int(min_n)].sort_values(["sum_pnl", "n"], ascending=[True, False])
    ban = [str(t).upper() for t in g.head(int(top_k))[ticker_col].tolist()]
    return ban, g


def _load_trades_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _save_run(cdir: Path, r: Dict[str, Any]) -> Dict[str, Any]:
    cdir.mkdir(parents=True, exist_ok=True)
    if isinstance(r.get("equity"), pd.Series):
        r["equity"].to_csv(cdir / "equity.csv", header=["equity"])
    if isinstance(r.get("trades"), pd.DataFrame) and not r["trades"].empty:
        r["trades"].to_csv(cdir / "trades.csv", index=False)
    meta = {k: v for k, v in r.items() if k not in ("equity", "trades")}
    (cdir / "metrics.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return meta


def _run(
    cfg: Any,
    *,
    years: Sequence[int],
    data_root: Path,
    panel_file: Path,
    l0_cache: Path,
    static_pool: List[str],
    min_train_rows: int,
) -> Dict[str, Any]:
    return _mega.run_config_years(
        cfg,
        years=list(years),
        data_root=data_root,
        panel_file=panel_file,
        l0_cache=l0_cache,
        static_pool=static_pool,
        min_train_rows=min_train_rows,
        use_dynamic_vol=False,
    )


def _eq_from_csv(path: Path) -> Optional[pd.Series]:
    if not path.is_file():
        return None
    eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    eq.index = pd.to_datetime(eq.index, utc=True, errors="coerce")
    eq = eq[~eq.index.duplicated(keep="last")].dropna().sort_index()
    return eq


def _slice_years(eq: pd.Series, first: int, last: int) -> pd.Series:
    if eq is None or eq.empty:
        return pd.Series(dtype=float)
    years = eq.index.year
    return eq[(years >= first) & (years <= last)].copy()


def _stitch_path(seg_a: pd.Series, seg_b: pd.Series) -> pd.Series:
    return _mega._stitch([seg_a, seg_b])


def _metrics_row(
    config_id: str,
    eq: pd.Series,
    trades: Optional[pd.DataFrame],
    *,
    residual: Optional[float] = None,
    note: str = "",
) -> Dict[str, Any]:
    if eq is None or eq.empty:
        return {"config_id": config_id, "error": "empty_equity", "note": note}
    start = float(eq.iloc[0])
    tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    rep = equity_metrics(eq, start_equity=start, trades=tdf if not tdf.empty else None)
    risk = extended_risk_from_equity(
        eq.to_numpy(),
        trade_pnls=tdf["net_profit"].to_numpy()
        if not tdf.empty and "net_profit" in tdf.columns
        else None,
    )
    return {
        "config_id": config_id,
        "cagr": rep.cagr,
        "sharpe": rep.sharpe,
        "sortino": risk.sortino,
        "max_drawdown": rep.max_drawdown,
        "n_trades": rep.n_trades,
        "win_rate": rep.win_rate,
        "total_return": float(eq.iloc[-1] / start - 1.0),
        "residual_cagr_vs_style": residual,
        "note": note,
    }


def _promo(
    name: str,
    eq: pd.Series,
    style_eq: Optional[pd.Series],
    trades: Optional[pd.DataFrame],
    *,
    n_sims: int,
    seed: int,
    n_trials: int,
):
    pnls = None
    n_tr = None
    if isinstance(trades, pd.DataFrame) and not trades.empty and "net_profit" in trades.columns:
        pnls = trades["net_profit"].to_numpy(dtype=float)
        n_tr = int(len(pnls))
    return evaluate_candidate(
        CandidateInput(
            name=name,
            equity=eq,
            style_equity=style_eq,
            trade_pnls=pnls,
            n_trades=n_tr,
            product="STYLE-US",
            smoke=False,
        ),
        n_sims=n_sims,
        seed=seed,
        n_trials_zoo=n_trials,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Loop G purged soft-ban")
    ap.add_argument("--panel", type=Path, default=ROOT / "universe_highvol200.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--screen-first", type=int, default=2018)
    ap.add_argument("--screen-last", type=int, default=2021)
    ap.add_argument("--confirm-first", type=int, default=2022)
    ap.add_argument("--confirm-last", type=int, default=2025)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-train-rows", type=int, default=2500)
    ap.add_argument(
        "--screen-trades",
        type=Path,
        default=ROOT
        / "reports/redesign/vol_fund_loop_f/configs"
        / "turbo_highvol_minalloc__volonly_k100_baseline"
        / "trades.csv",
        help="Baseline trades with oos_year (prefer Loop F/E). Empty → re-run screen.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "redesign" / "vol_fund_loop_g",
    )
    ap.add_argument(
        "--rerun-screen",
        action="store_true",
        help="Ignore --screen-trades and re-run baseline screen years",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    configs_dir = out / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    l0_cache = out / "l0_cache"
    data_root = Path(args.data_root)
    panel_file = Path(args.panel)
    static_pool = [
        ln.strip().upper()
        for ln in panel_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    screen_years = list(range(int(args.screen_first), int(args.screen_last) + 1))
    confirm_years = list(range(int(args.confirm_first), int(args.confirm_last) + 1))
    full_years = list(range(int(args.screen_first), int(args.confirm_last) + 1))

    print(
        f"Loop G purged soft-ban | screen {screen_years[0]}–{screen_years[-1]} "
        f"→ confirm {confirm_years[0]}–{confirm_years[-1]} | panel n={len(static_pool)}",
        flush=True,
    )

    # --- 1) Screen trades for banlist ---
    screen_trades_path = Path(args.screen_trades)
    trades_screen_src: Optional[pd.DataFrame] = None
    screen_eq: Optional[pd.Series] = None

    if not args.rerun_screen and screen_trades_path.is_file():
        print(f"[screen] load trades {screen_trades_path}", flush=True)
        all_tr = _load_trades_csv(screen_trades_path)
        assert all_tr is not None
        trades_screen_src = all_tr[
            pd.to_numeric(all_tr.get("oos_year"), errors="coerce").isin(screen_years)
        ].copy()
        # optional equity for path stitch from same run
        eq_cand = screen_trades_path.parent / "equity.csv"
        if eq_cand.is_file():
            full_eq = _eq_from_csv(eq_cand)
            if full_eq is not None:
                screen_eq = _slice_years(full_eq, screen_years[0], screen_years[-1])
    else:
        print("[screen] re-run baseline screen years …", flush=True)
        r_sc = _run(
            _cfg(BASE_ID + "_screen", ban=(), label="screen_baseline"),
            years=screen_years,
            data_root=data_root,
            panel_file=panel_file,
            l0_cache=l0_cache,
            static_pool=static_pool,
            min_train_rows=int(args.min_train_rows),
        )
        _save_run(configs_dir / (BASE_ID + "_screen"), r_sc)
        trades_screen_src = r_sc.get("trades")
        screen_eq = r_sc.get("equity") if isinstance(r_sc.get("equity"), pd.Series) else None

    ban, ban_table = build_softban_from_trades(
        trades_screen_src if trades_screen_src is not None else pd.DataFrame(),
        screen_years=screen_years,
        top_k=int(args.top_k),
        min_n=int(args.min_n),
    )
    ban_meta = {
        "screen_years": screen_years,
        "top_k": int(args.top_k),
        "min_n": int(args.min_n),
        "soft_ban": ban,
        "n_screen_trades": int(len(trades_screen_src) if trades_screen_src is not None else 0),
        "ticker_table": ban_table.to_dict(orient="records") if not ban_table.empty else [],
        "source_trades": str(screen_trades_path) if not args.rerun_screen else "rerun_screen",
        "loop_f_insample_ban": list(LOOP_F_BAN),
        "overlap_with_loop_f": sorted(set(ban) & set(LOOP_F_BAN)),
    }
    (out / "banlist_screen.json").write_text(
        json.dumps(ban_meta, indent=2, default=str), encoding="utf-8"
    )
    if not ban_table.empty:
        ban_table.to_csv(out / "banlist_screen_tickers.csv", index=False)
    print(f"[banlist] frozen n={len(ban)}: {ban}", flush=True)
    print(f"[banlist] overlap Loop F in-sample: {ban_meta['overlap_with_loop_f']}", flush=True)
    if not ban:
        print("ERROR: empty banlist — check screen trades / min_n", flush=True)
        return 2

    # --- 2) Confirm window runs ---
    confirm_cfgs = [
        _cfg(BASE_ID + "_confirm", ban=(), label="confirm_baseline"),
        _cfg(BAN_ID + "_confirm", ban=ban, label="confirm_softban_purged"),
        _cfg(
            IN_SAMPLE_BAN_ID + "_confirm",
            ban=LOOP_F_BAN,
            label="confirm_softban_insample_loopf",
        ),
        _cfg(STYLE_ID + "_confirm", strategy="growth_ew", ban=(), label="confirm_style"),
    ]
    confirm_rows: List[Dict[str, Any]] = []
    confirm_payload: Dict[str, Dict[str, Any]] = {}
    for cfg in confirm_cfgs:
        print(f"[confirm] {cfg.config_id} ban={list(cfg.exclude_tickers) or '-'} …", flush=True)
        r = _run(
            cfg,
            years=confirm_years,
            data_root=data_root,
            panel_file=panel_file,
            l0_cache=l0_cache,
            static_pool=static_pool,
            min_train_rows=int(args.min_train_rows),
        )
        meta = _save_run(configs_dir / cfg.config_id, r)
        print(
            f"  cagr={meta.get('cagr')} mdd={meta.get('max_drawdown')} "
            f"resid={meta.get('residual_cagr_vs_style')} n={meta.get('n_trades')}",
            flush=True,
        )
        confirm_rows.append(meta)
        confirm_payload[cfg.config_id] = r

    # residual vs style confirm (recompute if style ran same years)
    style_key = STYLE_ID + "_confirm"
    style_eq_c = confirm_payload.get(style_key, {}).get("equity")
    if isinstance(style_eq_c, pd.Series) and not style_eq_c.empty:
        s_start = float(style_eq_c.iloc[0])
        s_cagr = float(equity_metrics(style_eq_c, start_equity=s_start).cagr)
        for meta in confirm_rows:
            if meta.get("config_id", "").startswith("growth_ew"):
                continue
            if meta.get("cagr") is not None:
                meta["residual_cagr_vs_style"] = float(meta["cagr"]) - s_cagr

    # --- 3) Full path purged: baseline screen equity + softban confirm equity ---
    base_c = confirm_payload.get(BASE_ID + "_confirm", {})
    ban_c = confirm_payload.get(BAN_ID + "_confirm", {})
    if screen_eq is None or (hasattr(screen_eq, "empty") and screen_eq.empty):
        # re-run screen baseline for path
        print("[path] re-run screen baseline equity …", flush=True)
        r_sc2 = _run(
            _cfg(BASE_ID + "_screen", ban=(), label="screen_baseline"),
            years=screen_years,
            data_root=data_root,
            panel_file=panel_file,
            l0_cache=l0_cache,
            static_pool=static_pool,
            min_train_rows=int(args.min_train_rows),
        )
        _save_run(configs_dir / (BASE_ID + "_screen"), r_sc2)
        screen_eq = r_sc2.get("equity")

    ban_eq_c = ban_c.get("equity")
    path_eq = None
    path_trades = None
    if (
        isinstance(screen_eq, pd.Series)
        and not screen_eq.empty
        and isinstance(ban_eq_c, pd.Series)
        and not ban_eq_c.empty
    ):
        path_eq = _stitch_path(screen_eq, ban_eq_c)
        # trades: screen baseline trades + confirm ban trades
        tr_parts = []
        if trades_screen_src is not None and not trades_screen_src.empty:
            t0 = trades_screen_src.copy()
            t0["segment"] = "screen_baseline"
            tr_parts.append(t0)
        tr_ban = ban_c.get("trades")
        if isinstance(tr_ban, pd.DataFrame) and not tr_ban.empty:
            t1 = tr_ban.copy()
            t1["segment"] = "confirm_softban"
            tr_parts.append(t1)
        path_trades = pd.concat(tr_parts, ignore_index=True) if tr_parts else pd.DataFrame()
        cdir = configs_dir / PATH_ID
        cdir.mkdir(parents=True, exist_ok=True)
        path_eq.to_csv(cdir / "equity.csv", header=["equity"])
        if path_trades is not None and not path_trades.empty:
            path_trades.to_csv(cdir / "trades.csv", index=False)

    # Full baseline path (stitch screen + confirm baseline) for fair Δ
    base_eq_c = base_c.get("equity")
    full_base_eq = None
    if (
        isinstance(screen_eq, pd.Series)
        and not screen_eq.empty
        and isinstance(base_eq_c, pd.Series)
        and not base_eq_c.empty
    ):
        full_base_eq = _stitch_path(screen_eq, base_eq_c)
        (configs_dir / (BASE_ID + "_path")).mkdir(parents=True, exist_ok=True)
        full_base_eq.to_csv(
            configs_dir / (BASE_ID + "_path") / "equity.csv", header=["equity"]
        )

    # Style full path for residual
    style_full = None
    if isinstance(style_eq_c, pd.Series) and not style_eq_c.empty:
        # run style screen or slice from loop F if available
        style_f = (
            ROOT
            / "reports/redesign/vol_fund_loop_f/configs"
            / STYLE_ID
            / "equity.csv"
        )
        st_full = _eq_from_csv(style_f)
        if st_full is not None:
            st_sc = _slice_years(st_full, screen_years[0], screen_years[-1])
            style_full = _stitch_path(st_sc, style_eq_c)
        else:
            print("[path] run style screen for residual …", flush=True)
            r_st = _run(
                _cfg(STYLE_ID + "_screen", strategy="growth_ew", label="style_screen"),
                years=screen_years,
                data_root=data_root,
                panel_file=panel_file,
                l0_cache=l0_cache,
                static_pool=static_pool,
                min_train_rows=int(args.min_train_rows),
            )
            st_sc = r_st.get("equity")
            if isinstance(st_sc, pd.Series):
                style_full = _stitch_path(st_sc, style_eq_c)

    path_rows: List[Dict[str, Any]] = []
    if full_base_eq is not None:
        b_resid = None
        if style_full is not None and not style_full.empty:
            b_resid = float(equity_metrics(full_base_eq, start_equity=float(full_base_eq.iloc[0])).cagr) - float(
                equity_metrics(style_full, start_equity=float(style_full.iloc[0])).cagr
            )
        path_rows.append(
            _metrics_row(
                BASE_ID + "_path",
                full_base_eq,
                None,
                residual=b_resid,
                note="stitch baseline screen+confirm",
            )
        )
    if path_eq is not None:
        p_resid = None
        if style_full is not None and not style_full.empty:
            p_resid = float(equity_metrics(path_eq, start_equity=float(path_eq.iloc[0])).cagr) - float(
                equity_metrics(style_full, start_equity=float(style_full.iloc[0])).cagr
            )
        path_rows.append(
            _metrics_row(
                PATH_ID,
                path_eq,
                path_trades,
                residual=p_resid,
                note="stitch baseline screen + softban confirm (purged protocol)",
            )
        )
        (configs_dir / PATH_ID / "metrics.json").write_text(
            json.dumps(path_rows[-1], indent=2, default=str), encoding="utf-8"
        )

    # --- 4) Promotion ---
    cards = []
    promo_specs = [
        (
            BASE_ID + "_confirm",
            base_c.get("equity"),
            style_eq_c if isinstance(style_eq_c, pd.Series) else None,
            base_c.get("trades"),
        ),
        (
            BAN_ID + "_confirm",
            ban_c.get("equity"),
            style_eq_c if isinstance(style_eq_c, pd.Series) else None,
            ban_c.get("trades"),
        ),
        (
            IN_SAMPLE_BAN_ID + "_confirm",
            confirm_payload.get(IN_SAMPLE_BAN_ID + "_confirm", {}).get("equity"),
            style_eq_c if isinstance(style_eq_c, pd.Series) else None,
            confirm_payload.get(IN_SAMPLE_BAN_ID + "_confirm", {}).get("trades"),
        ),
    ]
    if path_eq is not None:
        promo_specs.append(
            (
                PATH_ID,
                path_eq,
                style_full if isinstance(style_full, pd.Series) else None,
                path_trades,
            )
        )

    for name, eq, st, tr in promo_specs:
        if not isinstance(eq, pd.Series) or eq.empty:
            print(f"[promo] skip {name}", flush=True)
            continue
        print(f"[promo] {name} n_bars={len(eq)} …", flush=True)
        card = _promo(
            name,
            eq,
            st if isinstance(st, pd.Series) else None,
            tr if isinstance(tr, pd.DataFrame) else None,
            n_sims=int(args.n_sims),
            seed=int(args.seed),
            n_trials=max(20, len(promo_specs)),
        )
        print(f"  → {card.label} reasons={card.kill_reasons}", flush=True)
        cards.append(card)

    cards = apply_top_k(cards, k=3)
    table = scorecard_table(cards)

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "loop": "G",
        "screen_years": f"{args.screen_first}-{args.screen_last}",
        "confirm_years": f"{args.confirm_first}-{args.confirm_last}",
        "banlist": ban_meta,
        "confirm_rows": confirm_rows,
        "path_rows": path_rows,
        "promotion": [c.to_dict() for c in cards],
        "paper_freeze": "turbo_highvol_minalloc (unchanged unless human ADVANCE)",
        "disclaimer": "Research only. Not financial advice. Purged banlist = screen-only.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # --- Markdown ---
    def _pct(x: Any) -> str:
        try:
            return f"{100 * float(x):.1f}%"
        except Exception:
            return "n/a"

    def _pp(x: Any) -> str:
        try:
            return f"{100 * float(x):.1f}pp"
        except Exception:
            return "n/a"

    lines = [
        "# Loop G — purged soft-ban (screen → confirm)",
        "",
        "> **Research only.** Not financial advice. Paper freeze unchanged.",
        "",
        f"- Screen (banlist freeze): **{args.screen_first}–{args.screen_last}**",
        f"- Confirm (apply frozen ban): **{args.confirm_first}–{args.confirm_last}**",
        f"- Panel highvol200 · minalloc vol-only k{K} · top_k={args.top_k} · min_n={args.min_n}",
        f"- Frozen soft-ban: `{', '.join(ban)}`",
        f"- Overlap with Loop F in-sample ban: `{', '.join(ban_meta['overlap_with_loop_f']) or 'none'}`",
        "",
        "## Screen banlist (worst sum PnL, n≥min_n)",
        "",
        "| ticker | n | sum_pnl |",
        "|--------|---|---------|",
    ]
    for t in ban:
        row = ban_table[ban_table["ticker"] == t] if not ban_table.empty else pd.DataFrame()
        if not row.empty:
            r0 = row.iloc[0]
            lines.append(f"| {t} | {int(r0['n'])} | {float(r0['sum_pnl']):.0f} |")
        else:
            lines.append(f"| {t} | ? | ? |")

    lines += [
        "",
        "## Confirm metrics (2022–25)",
        "",
        "| arm | CAGR | Sortino | MDD | resid vs style | n_trades |",
        "|-----|------|---------|-----|----------------|----------|",
    ]
    for meta in confirm_rows:
        cid = str(meta.get("config_id") or "")
        if "growth_ew" in cid:
            continue
        label = cid.replace(f"{STRATEGY}__volonly_k{K}_", "")
        lines.append(
            f"| `{label}` | {_pct(meta.get('cagr'))} | {float(meta.get('sortino') or 0):.2f} | "
            f"{_pct(meta.get('max_drawdown'))} | {_pp(meta.get('residual_cagr_vs_style'))} | "
            f"{meta.get('n_trades')} |"
        )

    lines += [
        "",
        "## Full path (honest stitch)",
        "",
        "Screen years = baseline (no ban). Confirm years = softban purged. "
        "Banlist frozen *before* confirm.",
        "",
        "| path | CAGR | Sortino | MDD | resid | note |",
        "|------|------|---------|-----|-------|------|",
    ]
    for meta in path_rows:
        lines.append(
            f"| `{meta.get('config_id')}` | {_pct(meta.get('cagr'))} | "
            f"{float(meta.get('sortino') or 0):.2f} | {_pct(meta.get('max_drawdown'))} | "
            f"{_pp(meta.get('residual_cagr_vs_style'))} | {meta.get('note')} |"
        )

    lines += [
        "",
        "## Promotion scorecard",
        "",
        table,
        "",
        "## Decision rules",
        "",
        "- ADVANCE only if label starts with ADVANCE_* on **confirm** or **purged path**",
        "- Paper freeze stays **turbo_highvol_minalloc** unless human copies candidate",
        "- Loop F softban8 was in-sample; this loop is the honest OOS check",
        "",
        "Research only. Not financial advice.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
