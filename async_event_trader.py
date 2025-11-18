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

    ticker: str
    timestamp: pd.Timestamp
    values: np.ndarray
    feature_names: List[str]


@dataclass
class ModelDecision:
    """Model output for a single ticker evaluation."""

    ticker: str
    timestamp: pd.Timestamp
    action: str
    confidence: float
    feature_names: Sequence[str]
    scores: Dict[str, float]
    should_execute: bool = False


class FeatureLoader:
    """Loads the latest feature vector for a ticker using ``DataPipeline``."""

    def __init__(
        self,
        pipeline: DataPipeline,
        *,
        indicators: bool = True,
        include_summary: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._indicators = indicators
        self._include_summary = include_summary

    def load_latest(self, ticker: str) -> FeatureVector:
        frame = self._pipeline.load_feature_view(
            ticker, indicators=self._indicators, include_summary=self._include_summary
        )
        if frame.empty:
            raise ValueError(f"No feature rows available for ticker {ticker}")
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError(f"Ticker {ticker} does not expose numeric features")
        vector = numeric.tail(1)
        values = vector.to_numpy(dtype=np.float32).ravel()
        timestamp = pd.to_datetime(frame.iloc[-1]["date"])
        return FeatureVector(
            ticker=ticker,
            timestamp=timestamp,
            values=values,
            feature_names=list(vector.columns),
        )


class ProbabilisticSignalModel:
    """Adapter that turns ``scikit-learn``-style estimators into trade signals."""

    def __init__(
        self,
        estimator: object,
        *,
        class_order: Optional[Sequence[str]] = None,
        score_threshold: float = 0.0,
    ) -> None:
        self._estimator = estimator
        self._explicit_classes = [label.upper() for label in class_order] if class_order else None
        inferred = getattr(estimator, "classes_", None)
        self._inferred_classes = [str(label).upper() for label in inferred] if inferred is not None else None
        self._score_threshold = float(score_threshold)

    def _resolve_labels(self, count: int) -> List[str]:
        if self._explicit_classes:
            labels = list(self._explicit_classes)
        elif self._inferred_classes:
            labels = list(self._inferred_classes)
        else:
            labels = []

        if len(labels) < count:
            labels = labels + [f"CLASS_{idx}" for idx in range(len(labels), count)]
        return labels[:count]

    def _score_to_probs(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 1:
            return np.array([1.0], dtype=float)
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        denom = np.sum(exp_scores)
        return exp_scores / denom if denom != 0 else np.full_like(exp_scores, 1.0 / len(exp_scores))

    def decide(self, vector: FeatureVector) -> ModelDecision:
        estimator = self._estimator
        input_vector = vector.values.reshape(1, -1)

        if hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(input_vector)[0], dtype=float)
            labels = self._resolve_labels(len(probabilities))
            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            scores = dict(zip(labels, probabilities))
        elif hasattr(estimator, "decision_function"):
            raw_scores = np.asarray(estimator.decision_function(input_vector))
            raw_scores = raw_scores.ravel()
            labels = self._resolve_labels(len(raw_scores))
            probabilities = self._score_to_probs(raw_scores)
            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            scores = dict(zip(labels, probabilities))
        else:
            prediction = estimator.predict(input_vector)
            predicted_label = str(prediction[0]).upper()
            labels = [predicted_label]
            confidence = 1.0
            scores = {predicted_label: 1.0}
            best_idx = 0

        action = labels[best_idx]
        if self._score_threshold and confidence < self._score_threshold:
            hold_label = "HOLD" if "HOLD" in labels else action
            action = hold_label
            confidence = 0.0

        return ModelDecision(
            ticker=vector.ticker,
            timestamp=vector.timestamp,
            action=action,
            confidence=float(confidence),
            feature_names=vector.feature_names,
            scores=scores,
        )


def _default_execute(decision: ModelDecision) -> None:
    logger.info(
        "Executing %s for %s (confidence %.2f)",
        decision.action,
        decision.ticker,
        decision.confidence,
    )


class AsyncEventDrivenTrader:
    """Coordinates the asynchronous evaluation of a dynamic ticker universe."""

    def __init__(
        self,
        tickers: Sequence[str],
        feature_loader: FeatureLoader,
        signal_model: ProbabilisticSignalModel,
        *,
        min_confidence: float = 0.6,
        max_concurrency: int = 32,
        tradeable_actions: Optional[Iterable[str]] = None,
        execute_trade: Optional[Callable[[ModelDecision], None]] = None,
    ) -> None:
        self._tickers = list(tickers)
        self._loader = feature_loader
        self._model = signal_model
        self._min_confidence = float(min_confidence)
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        actions = tradeable_actions or ("BUY", "SELL", "SHORT")
        self._tradeable_actions: Set[str] = {action.upper() for action in actions}
        self._execute_trade = execute_trade or _default_execute

    async def _evaluate_ticker(self, ticker: str) -> Optional[ModelDecision]:
        try:
            vector = await asyncio.to_thread(self._loader.load_latest, ticker)
        except FileNotFoundError:
            logger.warning("Missing cached data for ticker %s", ticker)
            return None
        except ValueError as exc:
            logger.debug("Skipping %s: %s", ticker, exc)
            return None

        decision = self._model.decide(vector)
        should_trade = (
            decision.action.upper() in self._tradeable_actions
            and decision.confidence >= self._min_confidence
        )
        decision.should_execute = should_trade
        if should_trade:
            await asyncio.to_thread(self._execute_trade, decision)
        return decision

    async def _guarded_task(self, ticker: str) -> Optional[ModelDecision]:
        async with self._semaphore:
            return await self._evaluate_ticker(ticker)

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