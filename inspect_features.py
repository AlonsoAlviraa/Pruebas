
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path.cwd()))

try:
    from drl_platform.data_pipeline import DataPipeline, PipelineConfig
except ImportError:
    print("Could not import DataPipeline")
    sys.exit(1)

def inspect_features():
    data_root = Path("data")
    pipeline = DataPipeline(PipelineConfig(data_root=data_root))
    
    # Get a ticker
    tickers = [f.name.replace("_history.csv", "") for f in data_root.glob("*_history.csv")]
    if not tickers:
        print("No tickers found")
        return

    ticker = tickers[0]
    print(f"Inspecting ticker: {ticker}")
    
    df = pipeline.load_feature_view(ticker, indicators=True)
    print("Columns:", df.columns.tolist())
    
    # Check what train_signal_model_v2 would keep
    drop_cols = ["date", "label", "ticker", "index", "tp_pct", "sl_pct", "time_exit_return", "summary"]
    features = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    numeric_features = features.select_dtypes(include=['number'])
    
    print("\nFeatures used in training (currently):")
    print(numeric_features.columns.tolist())

if __name__ == "__main__":
    inspect_features()
