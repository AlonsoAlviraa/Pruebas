#!/usr/bin/env python3
"""Async event-driven trading loop with volatility-aware bet sizing."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from drl_platform.data_pipeline import DataPipeline, PipelineConfig
from model_factory import BUY_CLASS


logger = logging.getLogger("async_event_trader")


@dataclass
class FeatureVector:
    ticker: str
    timestamp: pd.Timestamp
    features: pd.DataFrame
    price: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> List[str]:
        return list(self.features.columns)


@dataclass
class ModelDecision:
    ticker: str
    timestamp: datetime
    action: str
    confidence: float
    should_execute: bool
    scores: Dict[str, float]
    position_size: float


class FeatureLoader:
    def __init__(self, pipeline: DataPipeline, *, indicators: bool, include_summary: bool) -> None:
        self._pipeline = pipeline
        self._indicators = indicators
        self._include_summary = include_summary

    def load_vector(self, ticker: str) -> FeatureVector:
        frame = self._pipeline.load_feature_view(ticker, indicators=self._indicators, include_summary=self._include_summary)
        if frame.empty:
            raise ValueError(f"No data available for {ticker}")
        latest = frame.iloc[-1]
        drop_cols = ["date", "label", "ticker", "index", "summary", "tp_pct", "sl_pct", "time_exit_return"]
        feature_frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns])
        feature_frame = feature_frame.select_dtypes(include=[np.number])
        
        # Reconstruct as strict DataFrame row to satisfy sklearn validation
        row = feature_frame.tail(1).copy()
        row.index = [0]
        
        timestamp = pd.to_datetime(latest["date"], utc=True) if "date" in latest else pd.Timestamp.utcnow()
        price = float(latest.get("close", np.nan))
        return FeatureVector(
            ticker=ticker,
            timestamp=timestamp,
            features=row.fillna(0.0),
            price=price,
        )


class ProbabilisticSignalModel:
    def __init__(self, estimator: Any, *, class_order: Sequence[str], score_threshold: float) -> None:
        self._estimator = estimator
        self._score_threshold = score_threshold
        self._buy_label = BUY_CLASS
        self._class_names = [label.strip().upper() for label in class_order] or ["HOLD", "BUY", "SHORT"]

    def evaluate(self, vector: FeatureVector) -> ModelDecision:
        frame = vector.features
        
        # Critical: Align features with training data schema
        if hasattr(self._estimator, "feature_names_in_"):
            frame = frame.reindex(columns=list(self._estimator.feature_names_in_), fill_value=0.0)
        elif hasattr(self._estimator, "n_features_in_"):
            frame = frame.iloc[:, : self._estimator.n_features_in_]

        confidence = 0.0
        action = "HOLD"
        scores: Dict[str, float] = {}
        timestamp = vector.timestamp.to_pydatetime()

        if hasattr(self._estimator, "predict_proba"):
            probabilities = self._estimator.predict_proba(frame)[0]
            classes = list(getattr(self._estimator, "classes_", range(len(probabilities))))
            scores = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
            if BUY_CLASS in classes:
                confidence = float(probabilities[classes.index(BUY_CLASS)])
            else:
                confidence = float(max(probabilities))
        else:
            prediction = self._estimator.predict(frame)[0]
            scores = {str(prediction): 1.0}
            
            # Fallback for models without proba (e.g., SVM without probability=True)
            raw_score = 0.0
            if hasattr(self._estimator, "decision_function"):
                raw_score = float(self._estimator.decision_function(frame)[0])
            else:
                raw_score = 1.0 if prediction == self._buy_label else 0.0
            
            if raw_score < self._score_threshold:
                confidence = 0.0
            else:
                # Sigmoid approximation for confidence
                confidence = float(1 / (1 + np.exp(-raw_score)))
                if prediction != self._buy_label:
                    confidence = 1.0 - confidence

        if confidence >= 0.5:
            action = "BUY"

        return ModelDecision(
            ticker=vector.ticker,
            timestamp=timestamp,
            action=action,
            confidence=confidence,
            should_execute=False,
            scores=scores,
            position_size=0.0,
        )


class AsyncEventDrivenTrader:
    def __init__(
        self,
        tickers: Sequence[str],
        loader: FeatureLoader,
        signal_model: ProbabilisticSignalModel,
        *,
        min_confidence: float,
        base_position_size: float,
        max_concurrency: int,
    ) -> None:
        self._tickers = list(tickers)
        self._loader = loader
        self._signal_model = signal_model
        self._min_confidence = min_confidence
        self._base_position = base_position_size
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_one(self, ticker: str) -> Optional[ModelDecision]:
        try:
            # Offload blocking I/O to thread pool
            vector = await asyncio.to_thread(self._loader.load_vector, ticker)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", ticker, exc)
            return None
        
        # Offload CPU-bound inference
        decision = await asyncio.to_thread(self._signal_model.evaluate, vector)
        
        # Bet Sizing Formula: Linear scaling from 50% to 100% confidence
        size_factor = max(0.0, (decision.confidence - 0.5) * 2.0)
        decision.position_size = self._base_position * size_factor
        
        decision.should_execute = decision.confidence >= self._min_confidence and decision.position_size > 0
        return decision

    async def _guarded_task(self, ticker: str) -> Optional[ModelDecision]:
        async with self._semaphore:
            return await self._score_one(ticker)

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
        except Exception as exc:  # pragma: no cover
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
            "position_size": decision.position_size,
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
    parser.add_argument("--base-position", type=float, default=1.0, help="Reference position size for bet sizing")
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
        base_position_size=args.base_position,
        max_concurrency=args.max_concurrency,
    )
    results = asyncio.run(trader.run())
    actionable = sum(1 for decision in results if decision.should_execute)
    logger.info(
        "Evaluated %d tickers -> %d actionable trades", len(results), actionable
    )
    if args.output_csv:
        _write_summary(results, Path(args.output_csv))


if __name__ == "__main__":
    main()