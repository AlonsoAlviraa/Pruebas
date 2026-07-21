"""BKT-02: Train Lean-compatible M1/M2 and export to lean_strategy/storage."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from trad_research.config import DEFAULT_LABEL_CONFIG, FeatureConfig, LabelConfig
from trad_research.features import (
    M1_FEATURE_NAMES,
    M2_FEATURE_NAMES,
    feature_matrix,
    list_tickers,
    load_featured,
)
from trad_research.labels import attach_labels
from trad_research.walk_forward import _build_training_frame

logger = logging.getLogger(__name__)


@dataclass
class LeanExportConfig:
    data_root: Path = Path("data")
    ticker_file: Path = Path("good_tickers_wf80.txt")
    universe_limit: int = 80
    train_end: str = "2025-01-01"  # exclusive
    embargo_days: int = 5
    label: LabelConfig = None  # type: ignore
    lean_storage: Path = Path("lean_strategy/storage")
    models_dir: Path = Path("models")
    random_state: int = 42
    m2_buy_threshold: float = 0.45  # for M1 training sample filter

    def __post_init__(self) -> None:
        if self.label is None:
            self.label = DEFAULT_LABEL_CONFIG


def _load_panels(tickers: Sequence[str], data_root: Path) -> Dict[str, pd.DataFrame]:
    panels: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = load_featured(t, data_root)
        if not df.empty:
            # Ensure M1 aliases
            if "ma_50" not in df.columns and "sma_50" in df.columns:
                df["ma_50"] = df["sma_50"]
            if "ma_200" not in df.columns and "sma_200" in df.columns:
                df["ma_200"] = df["sma_200"]
            panels[t] = df
    return panels


def train_m2_multiclass(
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> XGBClassifier:
    """Primary model: 17 features, classes 0/1/2."""
    FeatureConfig().assert_lean_parity()
    X = feature_matrix(train_df, M2_FEATURE_NAMES)
    y = train_df["y_side"].astype(int)
    counts = y.value_counts().to_dict()
    n = len(y)
    weight_map = {c: n / (len(counts) * max(counts[c], 1)) for c in counts}
    sw = y.map(weight_map).astype(float)
    model = XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=40,
        reg_lambda=3.0,
        objective="multi:softprob",
        num_class=3,
        n_jobs=4,
        random_state=random_state,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    model.fit(X, y, sample_weight=sw)
    # Ensure sklearn feature names for Lean debugging
    try:
        model.feature_names_in_ = np.array(list(M2_FEATURE_NAMES), dtype=object)
    except Exception:
        pass
    return model


def train_m1_confirm(
    train_df: pd.DataFrame,
    m2: XGBClassifier,
    buy_threshold: float = 0.45,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Confirmation filter: 6 M1 features, binary CONFIRM if triple-barrier TP (y_meta).
    Trained on rows where M2 assigns material BUY probability.
    """
    X_m2 = feature_matrix(train_df, M2_FEATURE_NAMES)
    proba = m2.predict_proba(X_m2)
    classes = list(m2.classes_)
    buy_i = classes.index(2) if 2 in classes else int(np.argmax(classes))
    p_buy = proba[:, buy_i]
    # also include true BUY labels for denser training
    mask_arr = (p_buy >= buy_threshold) | (train_df["y_side"].to_numpy() == 2)
    if int(mask_arr.sum()) < 300:
        mask_arr = np.ones(len(train_df), dtype=bool)
    mask = pd.Series(mask_arr, index=train_df.index)

    # M1 frame with ma_* names expected by Lean
    base = train_df.copy()
    if "ma_50" not in base.columns and "sma_50" in base.columns:
        base["ma_50"] = base["sma_50"]
    if "ma_200" not in base.columns and "sma_200" in base.columns:
        base["ma_200"] = base["sma_200"]
    X1 = feature_matrix(base, M1_FEATURE_NAMES)
    Xm = X1.loc[mask]
    ym = train_df.loc[mask, "y_meta"].astype(int)
    pos = max(int(ym.sum()), 1)
    neg = max(int((ym == 0).sum()), 1)
    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=25,
        reg_lambda=3.0,
        objective="binary:logistic",
        scale_pos_weight=min(neg / pos, 6.0),
        n_jobs=4,
        random_state=random_state,
        tree_method="hist",
    )
    model.fit(Xm, ym)
    try:
        model.feature_names_in_ = np.array(list(M1_FEATURE_NAMES), dtype=object)
    except Exception:
        pass
    return model


def _write_model(
    model: Any,
    path: Path,
    metadata: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta_path = Path(str(path) + ".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %s", path, meta_path)


def export_lean_models(cfg: LeanExportConfig) -> Dict[str, Any]:
    """Train M1/M2 and export to lean storage + models/."""
    FeatureConfig().assert_lean_parity()
    tickers = list_tickers(cfg.ticker_file, cfg.data_root, limit=cfg.universe_limit)
    if not tickers:
        # fallback
        tickers = list_tickers(Path("good_tickers_filtrados.txt"), cfg.data_root, limit=cfg.universe_limit)
    logger.info("Loading %d tickers for Lean export", len(tickers))
    panels = _load_panels(tickers, cfg.data_root)
    train_end = pd.Timestamp(cfg.train_end, tz="UTC")
    train_df = _build_training_frame(
        panels,
        train_end=train_end,
        embargo_days=cfg.embargo_days,
        k_tp=cfg.label.k_tp,
        k_sl=cfg.label.k_sl,
        label_horizon=cfg.label.max_horizon,
    )
    if len(train_df) < 5000:
        raise RuntimeError(f"Insufficient training rows: {len(train_df)}")

    logger.info("Training M2 multiclass on %d rows...", len(train_df))
    m2 = train_m2_multiclass(train_df, random_state=cfg.random_state)
    logger.info("Training M1 confirm filter...")
    m1 = train_m1_confirm(
        train_df, m2, buy_threshold=cfg.m2_buy_threshold, random_state=cfg.random_state
    )

    common_meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "train_end_exclusive": cfg.train_end,
        "n_train_rows": len(train_df),
        "n_tickers": len(panels),
        "tickers_sample": tickers[:20],
        "label_config": cfg.label.to_dict(),
        "bkt02_version": "1.0",
    }
    m2_meta = {
        **common_meta,
        "model": "M2",
        "role": "primary_signal",
        "n_features": 17,
        "feature_names": list(M2_FEATURE_NAMES),
        "n_classes": 3,
        "class_map": {"0": "SELL", "1": "HOLD", "2": "BUY"},
        "filename": "xgb_m2.joblib",
    }
    m1_meta = {
        **common_meta,
        "model": "M1",
        "role": "confirmation_filter",
        "n_features": 6,
        "feature_names": list(M1_FEATURE_NAMES),
        "n_classes": 2,
        "class_map": {"0": "REJECT", "1": "CONFIRM"},
        "filename": "xgb_m1.joblib",
    }

    lean_m2 = cfg.lean_storage / "xgb_m2.joblib"
    lean_m1 = cfg.lean_storage / "xgb_m1.joblib"
    _write_model(m2, lean_m2, m2_meta)
    _write_model(m1, lean_m1, m1_meta)

    # Also models/ for setup_env / archives
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    _write_model(m2, cfg.models_dir / "xgb_m2.joblib", m2_meta)
    _write_model(m1, cfg.models_dir / "xgb_m1.joblib", m1_meta)

    # Sanity predict
    X2 = feature_matrix(train_df.head(5), M2_FEATURE_NAMES)
    p2 = m2.predict_proba(X2)
    assert p2.shape[1] == 3
    assert m2.n_features_in_ == 17
    assert m1.n_features_in_ == 6

    return {
        "m1_path": str(lean_m1),
        "m2_path": str(lean_m2),
        "m1_features": int(m1.n_features_in_),
        "m2_features": int(m2.n_features_in_),
        "m1_classes": list(map(int, m1.classes_)),
        "m2_classes": list(map(int, m2.classes_)),
        "n_train_rows": len(train_df),
        "n_tickers": len(panels),
        "metadata": {"m1": m1_meta, "m2": m2_meta},
    }
