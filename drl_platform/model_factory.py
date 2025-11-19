"""Factory helpers to build ML estimators for the signal model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # Optional dependency
    from xgboost import XGBClassifier  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]

from sklearn.ensemble import RandomForestClassifier

STOP_CLASS = 0
HOLD_CLASS = 1
BUY_CLASS = 2
CLASS_MAPPING = {-1: STOP_CLASS, 0: HOLD_CLASS, 1: BUY_CLASS}
CLASS_NAMES = {STOP_CLASS: "STOP", HOLD_CLASS: "HOLD", BUY_CLASS: "BUY"}


@dataclass
class ModelParams:
    model_type: str = "rf"
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    learning_rate: float = 0.05
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_jobs: int = -1
    class_weighted: bool = False
    random_state: Optional[int] = 42

    def asdict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "n_jobs": self.n_jobs,
            "class_weighted": self.class_weighted,
            "random_state": self.random_state,
        }


def build_model(
    *,
    model_type: str,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    learning_rate: float = 0.05,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    n_jobs: int = -1,
    class_weighted: bool = False,
    random_state: Optional[int] = None,
) -> Any:
    """Return a configured estimator based on ``model_type``."""

    model_type = model_type.lower()
    if model_type == "rf":
        class_weight = "balanced_subsample" if class_weighted else None
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
        )

    if model_type == "xgb":
        if XGBClassifier is None:  # pragma: no cover - optional dependency
            raise RuntimeError("xgboost is required for model_type='xgb'")
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 6,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_samples_split,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


__all__ = [
    "build_model",
    "ModelParams",
    "STOP_CLASS",
    "HOLD_CLASS",
    "BUY_CLASS",
    "CLASS_MAPPING",
    "CLASS_NAMES",
]