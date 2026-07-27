#!/usr/bin/env python3
"""
================================================================================
FEATURE NAME EXTRACTOR - Data Forensic Script
================================================================================
Extracts feature names from trained XGBoost/Sklearn models to diagnose
shape mismatch issues.

Author: Data Forensic Engineer
Date: 2024-12-30
================================================================================
"""
import sys
from pathlib import Path
from typing import Any, List, Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATHS = {
    "M1": Path("lean_strategy/storage/xgb_m1.joblib"),
    "M2": Path("lean_strategy/storage/xgb_m2.joblib"),
}

# Fallback paths (original names)
FALLBACK_PATHS = {
    "M1": Path("models/trend_model_triple_barrier.joblib"),
    "M2": Path("models/signal_model.joblib"),
}


# ==============================================================================
# EXTRACTION METHODS
# ==============================================================================

def extract_feature_names(model: Any) -> Optional[List[str]]:
    """
    Try all known methods to extract feature names from a model.
    Returns None if no method succeeds.
    """
    methods_tried = []
    
    # Method 1: Sklearn standard (feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        methods_tried.append("feature_names_in_")
        names = model.feature_names_in_
        if names is not None and len(names) > 0:
            return list(names)
    
    # Method 2: XGBoost native (get_booster().feature_names)
    if hasattr(model, "get_booster"):
        methods_tried.append("get_booster().feature_names")
        try:
            booster = model.get_booster()
            if hasattr(booster, "feature_names"):
                names = booster.feature_names
                if names is not None and len(names) > 0:
                    return list(names)
        except Exception:
            pass
    
    # Method 3: Direct feature_names attribute
    if hasattr(model, "feature_names"):
        methods_tried.append("feature_names")
        names = model.feature_names
        if names is not None and len(names) > 0:
            return list(names)
    
    # Method 4: feature_name_ (some sklearn versions)
    if hasattr(model, "feature_name_"):
        methods_tried.append("feature_name_")
        names = model.feature_name_
        if names is not None and len(names) > 0:
            return list(names)
    
    # Method 5: XGBoost Booster feature_names via attr
    if hasattr(model, "get_booster"):
        methods_tried.append("get_booster().attr('feature_names')")
        try:
            booster = model.get_booster()
            attr = booster.attr("feature_names")
            if attr:
                # Often stored as string, need to parse
                return attr.split(",")
        except Exception:
            pass
    
    # Method 6: Check for internal _features attribute
    if hasattr(model, "_features"):
        methods_tried.append("_features")
        if model._features is not None:
            return list(model._features)
    
    # Method 7: For pipelines, check steps
    if hasattr(model, "steps"):
        methods_tried.append("pipeline.steps")
        for name, step in model.steps:
            names = extract_feature_names(step)
            if names:
                return names
    
    print(f"    Methods tried: {methods_tried}")
    return None


def extract_model_info(model: Any) -> dict:
    """Extract comprehensive model information."""
    info = {
        "type": type(model).__name__,
        "n_features": None,
        "n_classes": None,
        "feature_names": None,
        "class_names": None,
    }
    
    # Feature count
    if hasattr(model, "n_features_in_"):
        info["n_features"] = model.n_features_in_
    elif hasattr(model, "n_features_"):
        info["n_features"] = model.n_features_
    
    # Class count
    if hasattr(model, "n_classes_"):
        info["n_classes"] = model.n_classes_
    elif hasattr(model, "classes_"):
        info["n_classes"] = len(model.classes_)
        info["class_names"] = list(model.classes_)
    
    # Feature names
    info["feature_names"] = extract_feature_names(model)
    
    return info


def print_banner(text: str, char: str = "=") -> None:
    """Print a visual banner."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_model_report(model_id: str, info: dict) -> None:
    """Print detailed model report."""
    print(f"\n  Model Type: {info['type']}")
    print(f"  Number of Features: {info['n_features']}")
    print(f"  Number of Classes: {info['n_classes']}")
    
    if info.get("class_names"):
        print(f"  Class Labels: {info['class_names']}")
    
    print()
    
    if info["feature_names"]:
        print(f"  ┌{'─' * 50}┐")
        print(f"  │ {'FEATURE NAMES':^48} │")
        print(f"  ├{'─' * 50}┤")
        for i, name in enumerate(info["feature_names"]):
            print(f"  │ {i:>3}. {name:<44} │")
        print(f"  └{'─' * 50}┘")
    else:
        print("  ╔════════════════════════════════════════════════╗")
        print("  ║  ⚠️  FEATURE NAMES NOT FOUND IN MODEL          ║")
        print("  ║                                                ║")
        print("  ║  The model was saved without feature names.   ║")
        print("  ║  You must inspect the TRAINING CODE to find   ║")
        print("  ║  the exact column order used during training. ║")
        print("  ╚════════════════════════════════════════════════╝")


def generate_python_list(names: Optional[List[str]], model_id: str) -> None:
    """Generate copy-pastable Python code."""
    if not names:
        return
    
    print(f"\n  # Copy-paste ready Python list for {model_id}:")
    print(f"  {model_id}_FEATURES = [")
    for name in names:
        print(f'      "{name}",')
    print("  ]")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    """Main entry point."""
    print_banner("FEATURE NAME EXTRACTOR - FORENSIC ANALYSIS")
    
    # Import joblib
    try:
        import joblib
        print("  ✓ joblib imported successfully")
    except ImportError:
        print("  ✗ ERROR: joblib not installed. Run: pip install joblib")
        return 1
    
    # Suppress XGBoost warnings about format
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    results = {}
    
    for model_id, path in MODEL_PATHS.items():
        print_banner(f"ANALYZING {model_id} MODEL", "-")
        
        # Try primary path, then fallback
        if not path.exists():
            path = FALLBACK_PATHS.get(model_id)
            if path and not path.exists():
                print(f"  ✗ Model file not found: {path}")
                continue
        
        print(f"  Loading: {path}")
        
        try:
            model = joblib.load(path)
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            continue
        
        # Extract info
        info = extract_model_info(model)
        results[model_id] = info
        
        # Print report
        print_model_report(model_id, info)
        
        # Generate Python code
        generate_python_list(info["feature_names"], model_id)
    
    # Summary
    print_banner("SUMMARY")
    
    for model_id, info in results.items():
        n_features = info.get("n_features", "?")
        has_names = "✓" if info.get("feature_names") else "✗"
        print(f"  {model_id}: {n_features} features | Names: {has_names}")
    
    print_banner("NEXT STEPS")
    
    if not all(info.get("feature_names") for info in results.values()):
        print("""
  Since feature names were not found in the models, you need to:
  
  1. Open your training script: train_signal_model_v2.py
  
  2. Find where features are selected, look for:
     - df[feature_columns]
     - X = df.drop(columns=[...])
     - FEATURE_COLS = [...]
     
  3. Copy the exact list of columns used for training
  
  4. Update lean_strategy/modules/feature_engine.py with the
     correct features in the EXACT SAME ORDER
""")
    else:
        print("""
  Feature names were found! Use the Python lists above to update:
  
  lean_strategy/modules/feature_engine.py
  lean_strategy/modules/config.py
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
