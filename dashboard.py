#!/usr/bin/env python3
"""Streamlit dashboard for visualizing experiment results from MLflow."""
from __future__ import annotations

import os
from typing import List

import pandas as pd

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None

try:  # pragma: no cover - optional dependency
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("Streamlit must be installed to run the dashboard") from exc


def load_experiments(tracking_uri: str) -> pd.DataFrame:
    """Carga experimentos desde MLflow."""
    if mlflow is None:
        raise ImportError("MLflow is required to load experiments")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()  # Usar search_experiments en lugar de list_experiments
    frames: List[pd.DataFrame] = []
    
    experiment_ids = [exp.experiment_id for exp in experiments]
    if not experiment_ids:
        return pd.DataFrame()

    # Buscar runs en todos los experimentos
    runs = client.search_runs(experiment_ids=experiment_ids)
    
    experiment_map = {exp.experiment_id: exp.name for exp in experiments}

    for run in runs:
        metrics = run.data.metrics
        metrics["run_id"] = run.info.run_id
        metrics["experiment_name"] = experiment_map.get(run.info.experiment_id, "default")
        frames.append(pd.DataFrame([metrics]))
        
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).fillna(value=pd.NA)


def main() -> None:  # pragma: no cover - UI only
    """Punto de entrada del dashboard de Streamlit."""
    st.set_page_config(layout="wide")
    st.title("DRL Research Platform Dashboard")
    
    tracking_uri_default = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    tracking_uri = st.text_input("MLflow Tracking URI", value=tracking_uri_default)
    
    if not tracking_uri:
        st.info("Please provide an MLflow tracking URI to load experiments.")
        return

    try:
        experiments = load_experiments(tracking_uri)
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Failed to load experiments: {exc}")
        return

    if experiments.empty:
        st.warning("No experiments found.")
        return

    st.subheader("All Experiment Runs")
    st.dataframe(experiments)

    st.subheader("Metric Analysis")
    metric = st.selectbox(
        "Metric",
        options=[col for col in experiments.columns if col not in {"run_id", "experiment_name"}],
    )
    if metric:
        st.line_chart(experiments, x="run_id", y=metric)
    else:
        st.info("No metrics found to plot.")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()