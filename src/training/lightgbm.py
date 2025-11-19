# LightGBM Training Script

import os
import json
import joblib
from lightgbm import LGBMClassifier

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import MODELS_DIR, TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import MODELS_DIR, TUNING_DIR


def load_best_params(model_name="lightgbm"):
    """Load tuned hyperparameters if they exist."""
    file_path = os.path.join(TUNING_DIR, f"{model_name}_best.json")

    if os.path.exists(file_path):
        print(f"Using tuned parameters from: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    print("No tuned parameters found — using default LightGBM settings.")
    return None


def lightgbm():
    """Train LightGBM with tuned or default parameters."""

    # Load + preprocess
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Load tuned parameters
    best_params = load_best_params("lightgbm")

    if best_params:
        model = LGBMClassifier(**best_params, random_state=42)
    else:
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

    print("Training LightGBM model...")
    model.fit(X_train_res, y_train_res)

    # Save trained model
    save_path = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
    joblib.dump(model, save_path)
    print(f"LightGBM model saved at: {save_path}")

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")

    return model


if __name__ == "__main__":
    lightgbm()
