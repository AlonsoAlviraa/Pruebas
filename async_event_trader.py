#!/usr/bin/env python3
"""Asynchronous, event-driven trading loop based on per-ticker evaluations."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from drl_platform.data_pipeline import DataPipeline, PipelineConfig

logger = logging.getLogger("async_event_trader")


@dataclass
class FeatureVector:
    """Container with the latest feature snapshot for a ticker."""

    async def run(self) -> List[ModelDecision]:
        tasks = [asyncio.create_task(self._guarded_task(ticker)) for ticker in self._tickers]
        raw_results = await asyncio.gather(*tasks)
        return [result for result in raw_results if result is not None]


def _discover_tickers(data_root: Path) -> List[str]:
    history_files = data_root.glob("*_history.csv")
    tickers = sorted({path.name.replace("_history.csv", "") for path in history_files})
    if not tickers:
        raise FileNotFoundError(f"No *_history.csv files were found inside {data_root}")
    return tickers


def _load_model(model_path: Path) -> object:
    joblib_module = None
    spec = importlib.util.find_spec("joblib")
    if spec is not None:
        joblib_module = importlib.import_module("joblib")
    if joblib_module is not None:
        try:
            return joblib_module.load(model_path)
        except Exception as exc:  # pragma: no cover - fallback to pickle
            logger.warning("joblib failed to load %s (%s), falling back to pickle", model_path, exc)
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def _parse_tickers(args: argparse.Namespace, data_root: Path) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(part.strip().upper() for part in args.tickers.split(",") if part.strip())
    if args.ticker_file:
        content = Path(args.ticker_file).read_text(encoding="utf-8").splitlines()
        tickers.extend(item.strip().upper() for item in content if item.strip())
    if not tickers:
        tickers = _discover_tickers(data_root)
    unique = []
    seen: Set[str] = set()
    for ticker in tickers:
        if ticker not in seen:
            unique.append(ticker)
            seen.add(ticker)
    return unique


def _write_summary(results: Sequence[ModelDecision], output_path: Path) -> None:
    if not results:
        logger.warning("No decisions to persist in %s", output_path)
        return
    rows = [
        {
            "ticker": decision.ticker,
            "timestamp": decision.timestamp.isoformat(),
            "action": decision.action,
            "confidence": decision.confidence,
            "should_execute": decision.should_execute,
            "scores": json.dumps(decision.scores),
        }
        for decision in results
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    logger.info("Saved %d decisions into %s", len(rows), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asynchronous event-driven trading loop")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a pickled sklearn estimator")
    parser.add_argument("--tickers", help="Comma-separated ticker list", default="")
    parser.add_argument("--ticker-file", type=Path, help="Optional file with one ticker per line")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Directory with cached *_history.csv files")
    parser.add_argument("--min-confidence", type=float, default=0.65, help="Confidence threshold required to trigger trades")
    parser.add_argument("--max-concurrency", type=int, default=32, help="Maximum number of concurrent ticker evaluations")
    parser.add_argument("--class-order", default="HOLD,BUY,SHORT", help="Expected order of the model outputs")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Threshold applied when the estimator lacks predict_proba()",
    )
    parser.add_argument("--include-summary", action="store_true", help="Append cached fundamentals to the feature vector")
    parser.add_argument("--no-indicators", action="store_true", help="Disable indicator engineering for faster scans")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV path with the evaluation summary")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    data_root = args.data_root
    tickers = _parse_tickers(args, data_root)
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    loader = FeatureLoader(
        pipeline,
        indicators=not args.no_indicators,
        include_summary=args.include_summary,
    )
    estimator = _load_model(Path(args.model_path))
    class_order = [label.strip().upper() for label in args.class_order.split(",") if label.strip()]
    signal_model = ProbabilisticSignalModel(
        estimator,
        class_order=class_order,
        score_threshold=args.score_threshold,
    )
    trader = AsyncEventDrivenTrader(
        tickers,
        loader,
        signal_model,
        min_confidence=args.min_confidence,
        max_concurrency=args.max_concurrency,
    )
    results = asyncio.run(trader.run())
    logger.info(
        "Evaluated %d tickers -> %d actionable trades",
        len(results),
        sum(1 for decision in results if decision.should_execute),
    )
    if args.output_csv:
        _write_summary(results, Path(args.output_csv))


if __name__ == "__main__":
    main()