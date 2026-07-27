#!/usr/bin/env python3
"""
================================================================================
META-LABELING STRATEGY - Environment Setup & Model Validation Script
================================================================================
This script:
1. Copies and renames trained models to standardized names
2. Validates model input dimensions (M1=9 features, M2=10 features)
3. Generates/updates configuration files

Run from project root: python setup_env.py
================================================================================
"""
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Any

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Source model files (relative to project root)
# Prefer BKT-02 exports (models/xgb_m*.joblib); fall back to legacy names.
SOURCE_MODELS = {
    "M1": "models/xgb_m1.joblib",
    "M2": "models/xgb_m2.joblib",
}
LEGACY_SOURCE_MODELS = {
    "M1": "models/trend_model_triple_barrier.joblib",
    "M2": "models/signal_model.joblib",
}

# Target names (standardized) — must match lean_strategy.modules.config
TARGET_NAMES = {
    "M1": "xgb_m1.joblib",
    "M2": "xgb_m2.joblib",
}

# Expected feature counts (Lean Hierarchical Voting ground truth)
EXPECTED_FEATURES = {
    "M1": 6,   # close, atr, ma_50, ma_200, rsi_14, volume_sma
    "M2": 17,  # full M2_FEATURE_NAMES
}
EXPECTED_CLASSES = {
    "M1": 2,
    "M2": 3,
}

# Target directory for Lean
LEAN_STRATEGY_DIR = Path("lean_strategy")
STORAGE_DIR = LEAN_STRATEGY_DIR / "storage"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_banner(text: str, char: str = "=") -> None:
    """Print a banner for visual separation."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_status(message: str, status: str = "INFO") -> None:
    """Print a status message with prefix."""
    icons = {
        "INFO": "ℹ️ ",
        "OK": "✅",
        "WARN": "⚠️ ",
        "ERROR": "❌",
        "CHECK": "🔍",
    }
    icon = icons.get(status, "  ")
    print(f"{icon} {message}")


def load_model(path: Path) -> Optional[Any]:
    """Load a joblib model file."""
    try:
        import joblib
        return joblib.load(path)
    except ImportError:
        print_status("joblib not installed. Run: pip install joblib", "ERROR")
        return None
    except Exception as e:
        print_status(f"Failed to load model: {e}", "ERROR")
        return None


def get_model_feature_count(model: Any) -> Optional[int]:
    """Extract expected feature count from model."""
    # Method 1: sklearn/xgboost n_features_in_
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_
    
    # Method 2: feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return len(model.feature_names_in_)
    
    # Method 3: XGBoost Booster
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if hasattr(booster, "num_features"):
            return booster.num_features()
    
    # Method 4: n_features_ (deprecated but still used)
    if hasattr(model, "n_features_"):
        return model.n_features_
    
    return None


def get_model_classes(model: Any) -> Optional[int]:
    """Get number of output classes."""
    if hasattr(model, "n_classes_"):
        return model.n_classes_
    if hasattr(model, "classes_"):
        return len(model.classes_)
    return None


# ==============================================================================
# MAIN SETUP LOGIC
# ==============================================================================

def setup_storage_directory() -> Path:
    """Create storage directory if it doesn't exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    print_status(f"Storage directory: {STORAGE_DIR.absolute()}", "OK")
    return STORAGE_DIR


def copy_and_rename_models(project_root: Path) -> dict:
    """Copy models to storage with standardized names."""
    print_banner("STEP 1: COPYING & RENAMING MODELS")
    
    results = {}
    
    for model_id, source_rel in SOURCE_MODELS.items():
        source_path = project_root / source_rel
        if not source_path.exists():
            legacy = LEGACY_SOURCE_MODELS.get(model_id)
            if legacy:
                alt = project_root / legacy
                if alt.exists():
                    source_path = alt
        # Already in storage from export_lean_models
        target_path = STORAGE_DIR / TARGET_NAMES[model_id]
        if not source_path.exists() and target_path.exists():
            print_status(f"{model_id} already in storage: {target_path}", "OK")
            results[model_id] = {"success": True, "path": target_path, "error": None}
            continue

        print(f"\n  {model_id} Model:")
        print(f"    Source: {source_path}")
        print(f"    Target: {target_path}")
        
        if not source_path.exists():
            print_status(
                f"Source not found. Run: python scripts/export_lean_models.py",
                "ERROR",
            )
            results[model_id] = {"success": False, "path": None, "error": "File not found"}
            continue
        
        try:
            shutil.copy2(source_path, target_path)
            # copy metadata if present
            meta_src = Path(str(source_path) + ".metadata.json")
            if meta_src.exists():
                shutil.copy2(meta_src, Path(str(target_path) + ".metadata.json"))
            print_status(f"{model_id} copied successfully", "OK")
            results[model_id] = {"success": True, "path": target_path, "error": None}
        except Exception as e:
            print_status(f"Failed to copy {model_id}: {e}", "ERROR")
            results[model_id] = {"success": False, "path": None, "error": str(e)}
    
    return results


def validate_model_dimensions(copy_results: dict) -> dict:
    """Validate that models have expected feature counts."""
    print_banner("STEP 2: VALIDATING MODEL DIMENSIONS")
    
    validation = {}
    
    for model_id in ["M1", "M2"]:
        if not copy_results.get(model_id, {}).get("success"):
            print_status(f"Skipping {model_id} validation (copy failed)", "WARN")
            validation[model_id] = {"valid": False, "features": None, "classes": None}
            continue
        
        model_path = copy_results[model_id]["path"]
        expected_features = EXPECTED_FEATURES[model_id]
        
        print(f"\n  Validating {model_id}:")
        print(f"    Path: {model_path}")
        print(f"    Expected features: {expected_features}")
        
        model = load_model(model_path)
        if model is None:
            validation[model_id] = {"valid": False, "features": None, "classes": None}
            continue
        
        actual_features = get_model_feature_count(model)
        num_classes = get_model_classes(model)
        
        print(f"    Actual features: {actual_features}")
        print(f"    Output classes: {num_classes}")
        
        exp_cls = EXPECTED_CLASSES.get(model_id)
        # Validate feature count (+ classes if known)
        feat_ok = actual_features == expected_features
        cls_ok = exp_cls is None or num_classes == exp_cls
        if feat_ok and cls_ok:
            print_status(
                f"{model_id} PASSED ({actual_features} features, {num_classes} classes)",
                "OK",
            )
            validation[model_id] = {
                "valid": True,
                "features": actual_features,
                "classes": num_classes,
            }
        else:
            print_status(
                f"{model_id} MISMATCH: features {actual_features}/{expected_features}, "
                f"classes {num_classes}/{exp_cls}",
                "WARN",
            )
            validation[model_id] = {
                "valid": False,
                "features": actual_features,
                "classes": num_classes,
            }
    
    # Critical check: hierarchical voting contract
    m2_features = validation.get("M2", {}).get("features")
    m1_features = validation.get("M1", {}).get("features")
    if m2_features is not None and m2_features != 17:
        print_banner("⚠️  CRITICAL: M2 must have 17 features (Lean config)", "!")
        print("    Run: python scripts/export_lean_models.py")
    if m1_features is not None and m1_features != 6:
        print_banner("⚠️  CRITICAL: M1 must have 6 features (Lean config)", "!")
        print("    Run: python scripts/export_lean_models.py")
    
    return validation


def generate_lean_config() -> None:
    """Generate/update lean config.json."""
    print_banner("STEP 3: GENERATING LEAN CONFIG")
    
    config = {
        "algorithm-type-name": "MetaLabelingStrategy",
        "algorithm-language": "Python",
        "algorithm-location": ".",
        "parameters": {
            "model_m1": TARGET_NAMES["M1"],
            "model_m2": TARGET_NAMES["M2"],
        },
        "description": "Meta-Labeling Strategy with M1+M2 Cascade (López de Prado)",
        "cloud-id": 0,
        "local-id": 0,
        "organization-id": "",
        "python-venv": None,
        "version": "1.0.0"
    }
    
    config_path = LEAN_STRATEGY_DIR / "config.json"
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    print_status(f"Config written to: {config_path}", "OK")


def print_summary(copy_results: dict, validation: dict) -> None:
    """Print final summary."""
    print_banner("SETUP SUMMARY")
    
    print("\n  Model Status:")
    for model_id in ["M1", "M2"]:
        copied = copy_results.get(model_id, {}).get("success", False)
        valid = validation.get(model_id, {}).get("valid", False)
        features = validation.get(model_id, {}).get("features", "?")
        
        status = "✅ READY" if (copied and valid) else "⚠️  CHECK REQUIRED"
        print(f"    {model_id}: {status} ({features} features)")
    
    print("\n  Files created:")
    print(f"    - {STORAGE_DIR / TARGET_NAMES['M1']}")
    print(f"    - {STORAGE_DIR / TARGET_NAMES['M2']}")
    print(f"    - {LEAN_STRATEGY_DIR / 'config.json'}")


def print_next_steps() -> None:
    """Print next steps for the user."""
    print_banner("NEXT STEPS")
    
    print("""
  1. If models are READY:
     
     cd lean_strategy
     lean backtest main.py
     
  2. If dimension mismatch:
     
     - Check your training data columns
     - Verify feature order matches feature_engine.py
     - Re-train if necessary
     
  3. For QuantConnect Cloud:
     
     - Upload xgb_m1.joblib and xgb_m2.joblib to ObjectStore
     - Upload all files from lean_strategy/
     - Run backtest in cloud
""")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> int:
    """Main entry point."""
    print_banner("META-LABELING STRATEGY - SETUP SCRIPT", "=")
    
    # Determine project root
    project_root = Path(__file__).parent.absolute()
    print_status(f"Project root: {project_root}", "INFO")
    
    # Check required packages
    try:
        import joblib
        import numpy
        print_status("Required packages (joblib, numpy) available", "OK")
    except ImportError as e:
        print_status(f"Missing package: {e}", "ERROR")
        print_status("Run: pip install joblib numpy", "INFO")
        return 1
    
    # Step 1: Setup storage directory
    setup_storage_directory()
    
    # Step 2: Copy and rename models
    copy_results = copy_and_rename_models(project_root)
    
    # Step 3: Validate dimensions
    validation = validate_model_dimensions(copy_results)
    
    # Step 4: Generate config
    generate_lean_config()
    
    # Summary
    print_summary(copy_results, validation)
    print_next_steps()
    
    # Return code
    all_valid = all(v.get("valid", False) for v in validation.values())
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
