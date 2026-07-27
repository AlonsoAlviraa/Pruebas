"""
SHAP Blocker Analysis - Why isn't M2 firing?
=============================================
This script analyzes the M2 XGBoost model to identify which features
are systematically pushing the BUY probability DOWN.

Output:
- SHAP summary plot (beeswarm)
- Top 3 "Trade Killer" features
- Analysis of "undecided" days (P_Buy between 0.20-0.40)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Install shap if not present
try:
    import shap
except ImportError:
    print("Installing shap...")
    os.system("pip install shap")
    import shap

# =============================================================================
# CONFIGURATION - M2 FEATURES (EXACT MATCH)
# =============================================================================

M2_FEATURE_NAMES = [
    "open", "high", "low", "close", "atr", "atr_norm",
    "rsi_7", "rsi_14", "rsi_21", "sma_50", "dist_sma_50",
    "sma_200", "dist_sma_200", "volatility_20", "volume_sma",
    "volume_ratio", "volume_zscore"
]

MODEL_PATH = "lean_strategy/storage/xgb_m2.joblib"
DATA_DIR = "data"
OUTPUT_DIR = "shap_analysis"

# =============================================================================
# STEP 1: LOAD DATA FROM LOCAL CSVs
# =============================================================================

def load_local_data(symbol: str = "AAPL") -> pd.DataFrame:
    """Load price data from local CSV (Yahoo Finance format)."""
    
    # Try different file patterns
    patterns = [
        f"{DATA_DIR}/{symbol}_history.csv",
        f"{DATA_DIR}/{symbol}.csv",
        f"{DATA_DIR}/{symbol.lower()}.csv",
    ]
    
    for path in patterns:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            # Read first line to check headers
            with open(path, 'r') as f:
                header = f.readline().strip().lower()
            
            # Determine date column name
            date_col = 'date' if 'date' in header.split(',') else 'Date'
            
            try:
                df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
                # Ensure index is DatetimeIndex
                df.index = pd.to_datetime(df.index, utc=True)
                return df
            except Exception as e:
                print(f"  Error reading {path}: {e}")
                continue
    
    # Fallback: try to download with yfinance
    try:
        import yfinance as yf
        print(f"Downloading {symbol} from yfinance...")
        df = yf.download(symbol, start="2018-01-01", end="2024-12-31", progress=False)
        return df
    except:
        pass
    
    raise FileNotFoundError(f"No data found for {symbol}")


def find_available_symbols() -> list:
    """Find available symbols in data directory."""
    symbols = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith("_history.csv"):
                sym = f.replace("_history.csv", "")
                symbols.append(sym)
    return symbols[:5]  # Limit to first 5


# =============================================================================
# STEP 2: FEATURE ENGINEERING (EXACT M2 REPLICATION)
# =============================================================================

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculate RSI manually."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def engineer_m2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the EXACT 17 features for M2 model.
    
    ORDER MUST MATCH:
    [open, high, low, close, atr, atr_norm, rsi_7, rsi_14, rsi_21,
     sma_50, dist_sma_50, sma_200, dist_sma_200, volatility_20,
     volume_sma, volume_ratio, volume_zscore]
    """
    result = pd.DataFrame(index=df.index)
    
    # Normalize column names
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    close = df['close'] if 'close' in df.columns else df['adj close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # Basic OHLC
    result['open'] = open_price
    result['high'] = high
    result['low'] = low
    result['close'] = close
    
    # ATR (14)
    atr = calculate_atr(high, low, close, 14)
    result['atr'] = atr
    result['atr_norm'] = atr / close  # Normalized ATR
    
    # RSI (7, 14, 21)
    result['rsi_7'] = calculate_rsi(close, 7)
    result['rsi_14'] = calculate_rsi(close, 14)
    result['rsi_21'] = calculate_rsi(close, 21)
    
    # SMA (50, 200) and distances
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    
    result['sma_50'] = sma_50
    result['dist_sma_50'] = (close / sma_50) - 1.0  # Relative return
    result['sma_200'] = sma_200
    result['dist_sma_200'] = (close / sma_200) - 1.0
    
    # Volatility (20-day annualized)
    returns = close.pct_change()
    result['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
    
    # Volume features
    volume_sma = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    
    result['volume_sma'] = volume_sma
    result['volume_ratio'] = volume / volume_sma
    result['volume_zscore'] = (volume - volume_sma) / volume_std.replace(0, np.nan)
    
    # Clean up NaN
    result = result.dropna()
    
    # Validate column order
    result = result[M2_FEATURE_NAMES]
    
    print(f"Engineered {len(result)} samples with {len(result.columns)} features")
    return result


# =============================================================================
# STEP 3: SHAP ANALYSIS
# =============================================================================

def analyze_shap(model, X: pd.DataFrame, class_idx: int = 2) -> tuple:
    """
    Perform SHAP analysis on the model.
    
    Args:
        model: XGBoost classifier
        X: Feature DataFrame
        class_idx: Class to analyze (2 = BUY)
    
    Returns:
        explainer, shap_values
    """
    print(f"\nInitializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    print(f"Calculating SHAP values for {len(X)} samples...")
    shap_values = explainer.shap_values(X)
    
    print(f"SHAP values type: {type(shap_values)}")
    if hasattr(shap_values, 'shape'):
        print(f"SHAP values shape: {shap_values.shape}")
    
    # Handle differnet return types
    if isinstance(shap_values, list):
        print(f"Multi-class model with {len(shap_values)} classes (List)")
        shap_values_buy = shap_values[class_idx]
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        # (samples, features, classes)
        print(f"Multi-class model with 3D array: {shap_values.shape}")
        shap_values_buy = shap_values[:, :, class_idx]
    else:
        # Binary or regression (samples, features)
        print("Binary/Regression or single-class output")
        shap_values_buy = shap_values
        
    print(f"SHAP values (class {class_idx}) shape: {shap_values_buy.shape}")
    return explainer, shap_values_buy


def find_trade_killers(model, X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """
    Identify features that systematically push BUY probability DOWN.
    Focus on "undecided" days where P(BUY) is between 0.20 and 0.40.
    """
    print("\n" + "="*60)
    print("TRADE KILLER ANALYSIS")
    print("="*60)
    
    # Get probabilities
    proba = model.predict_proba(X)
    prob_buy = proba[:, 2]  # Class 2 = BUY
    
    # Filter "undecided" days (P_Buy between 0.20 and 0.40)
    undecided_mask = (prob_buy >= 0.20) & (prob_buy <= 0.40)
    n_undecided = undecided_mask.sum()
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Undecided days (P_Buy 0.20-0.40): {n_undecided} ({n_undecided/len(X)*100:.1f}%)")
    
    if n_undecided == 0:
        print("No undecided days found - adjusting range to 0.15-0.50")
        undecided_mask = (prob_buy >= 0.15) & (prob_buy <= 0.50)
        n_undecided = undecided_mask.sum()
        print(f"Undecided days (adjusted): {n_undecided}")
    
    # Get SHAP values for undecided days
    shap_undecided = shap_values[undecided_mask]
    
    # Calculate mean SHAP contribution per feature
    mean_shap = pd.DataFrame({
        'feature': M2_FEATURE_NAMES,
        'mean_shap': shap_undecided.mean(axis=0),
        'abs_mean_shap': np.abs(shap_undecided).mean(axis=0)
    })
    
    # Identify "killers" - features with consistently NEGATIVE SHAP
    mean_shap = mean_shap.sort_values('mean_shap')
    
    print("\n" + "-"*60)
    print("TOP 5 TRADE KILLERS (Features pushing probability DOWN)")
    print("-"*60)
    
    killers = mean_shap.head(5)
    for i, row in killers.iterrows():
        impact_pct = abs(row['mean_shap']) * 100
        print(f"  • {row['feature']}: reduces P(BUY) by ~{impact_pct:.1f}% on average")
    
    print("\n" + "-"*60)
    print("TOP 5 TRADE BOOSTERS (Features pushing probability UP)")
    print("-"*60)
    
    boosters = mean_shap.tail(5).iloc[::-1]
    for i, row in boosters.iterrows():
        impact_pct = row['mean_shap'] * 100
        print(f"  • {row['feature']}: increases P(BUY) by ~{impact_pct:.1f}% on average")
    
    return mean_shap


def generate_plots(X: pd.DataFrame, shap_values: np.ndarray, output_dir: str):
    """Generate SHAP visualization plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary Plot (Beeswarm)
    print("\nGenerating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=17)
    plt.title("SHAP Summary - Impact on P(BUY)\nRed = High value, Blue = Low value", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/shap_summary_beeswarm.png")
    
    # 2. Bar Plot (Mean absolute impact)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=17)
    plt.title("Feature Importance (Mean |SHAP|)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance_bar.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/shap_importance_bar.png")
    
    plt.close('all')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("SHAP BLOCKER ANALYSIS - Why isn't M2 firing?")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Make sure you run this from the project root directory")
        sys.exit(1)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"  Model type: {type(model).__name__}")
    print(f"  Features expected: {model.n_features_in_}")
    print(f"  Classes: {model.classes_}")
    
    # Find available symbols
    symbols = find_available_symbols()
    if not symbols:
        symbols = ["AAPL"]  # Default
    print(f"\nAvailable symbols: {symbols}")
    
    # Load and process data for multiple symbols
    all_features = []
    
    for symbol in symbols[:3]:  # Limit to 3 symbols
        try:
            print(f"\n--- Processing {symbol} ---")
            df = load_local_data(symbol)
            features = engineer_m2_features(df)
            
            # Take last 500 days
            features = features.tail(500)
            all_features.append(features)
            
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    if not all_features:
        print("\nNo data could be processed!")
        sys.exit(1)
    
    # Combine all data
    X = pd.concat(all_features, axis=0)
    print(f"\nCombined dataset: {len(X)} samples, {len(X.columns)} features")
    
    # Validate features match
    if len(X.columns) != model.n_features_in_:
        print(f"\nWARNING: Feature count mismatch!")
        print(f"  Model expects: {model.n_features_in_}")
        print(f"  We have: {len(X.columns)}")
    
    # Get predictions first
    print("\nGetting model predictions...")
    proba = model.predict_proba(X)
    print(f"  P(SELL) range: [{proba[:,0].min():.3f}, {proba[:,0].max():.3f}]")
    print(f"  P(HOLD) range: [{proba[:,1].min():.3f}, {proba[:,1].max():.3f}]")
    print(f"  P(BUY) range:  [{proba[:,2].min():.3f}, {proba[:,2].max():.3f}]")
    print(f"  P(BUY) mean:   {proba[:,2].mean():.3f}")
    
    # SHAP Analysis
    explainer, shap_values = analyze_shap(model, X, class_idx=2)
    
    # Find trade killers
    killer_analysis = find_trade_killers(model, X, shap_values)
    
    # Generate plots
    generate_plots(X, shap_values, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nPlots saved to: {OUTPUT_DIR}/")
    print("\nKey insights:")
    print("  • Check shap_summary_beeswarm.png for direction of impact")
    print("  • Red points on LEFT = high values reduce P(BUY)")
    print("  • Blue points on LEFT = low values reduce P(BUY)")
    
    return killer_analysis


if __name__ == "__main__":
    main()
