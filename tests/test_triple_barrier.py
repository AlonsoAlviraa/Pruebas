import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from drl_platform.data_pipeline import DataPipeline, PipelineConfig


def _build_frame(close: np.ndarray, high: np.ndarray | None = None, low: np.ndarray | None = None) -> pd.DataFrame:
    high = high if high is not None else close
    low = low if low is not None else close
    dates = pd.date_range("2023-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(len(close), 1_000_000),
        }
    )


def test_tp_barrier_is_hit_before_stop():
    close = np.array([100, 101, 107, 109, 110], dtype=float)
    high = np.array([101, 105, 110, 111, 112], dtype=float)
    low = np.array([99, 100, 103, 107, 108], dtype=float)
    frame = _build_frame(close, high, low)
    
    # CAMBIO CLAVE: Configuramos ATR window a 1 para que no devuelva None con pocos datos
    config = PipelineConfig(atr_window=1)
    pipeline = DataPipeline(config)

    # Horizon 3, TP 5% (Target 105), SL -3% (Target 97)
    labelled = pipeline.create_triple_barrier_labels(frame, horizon=3, take_profit=0.05, stop_loss=-0.03)

    assert labelled.loc[0, "label"] == 1
    assert labelled.loc[0, "tp_pct"] == pytest.approx(0.05)
    assert labelled.loc[0, "sl_pct"] == pytest.approx(-0.03)


def test_stop_loss_gap_event():
    close = np.array([50, 49, 40, 39, 41], dtype=float)
    high = np.array([51, 50, 41, 40, 42], dtype=float)
    low = np.array([49, 30, 38, 37, 39], dtype=float)
    frame = _build_frame(close, high, low)
    
    # CAMBIO CLAVE
    config = PipelineConfig(atr_window=1)
    pipeline = DataPipeline(config)

    # Horizon 2, TP 5%, SL -2% (Target 49)
    labelled = pipeline.create_triple_barrier_labels(frame, horizon=2, take_profit=0.05, stop_loss=-0.02)

    assert labelled.loc[0, "label"] == -1
    assert labelled.loc[0, "sl_pct"] == pytest.approx(-0.02)


def test_tail_rows_are_nan():
    close = np.array([100, 101, 102, 103, 104], dtype=float)
    frame = _build_frame(close)
    
    # CAMBIO CLAVE
    config = PipelineConfig(atr_window=1)
    pipeline = DataPipeline(config)

    labelled = pipeline.create_triple_barrier_labels(frame, horizon=2, take_profit=0.05, stop_loss=-0.02)

    tail = labelled.tail(2)
    assert tail["label"].isna().all()
    assert tail["tp_pct"].isna().all()
    assert tail["sl_pct"].isna().all()