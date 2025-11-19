# Random Forest Training Script

import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import MODELS_DIR, TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import MODELS_DIR, TUNING_DIR


def load_best_params(model_name="random_forest"):
    """Load tuned hyperparameters for Random Forest if they exist."""
    file_path = os.path.join(TUNING_DIR, f"{model_name}_best.json")

    if os.path.exists(file_path):
        print(f"Loaded tuned parameters from: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    print("No tuned parameters found — using default Random Forest settings.")
    return None


def train_random_forest():
    """Train Random Forest using tuned or default parameters."""

    # Load and preprocess
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Load tuned params if exist
    best_params = load_best_params("random_forest")

    if best_params:
        model = RandomForestClassifier(**best_params, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    print("Training Random Forest model...")
    model.fit(X_train_res, y_train_res)

    # Save trained model
    save_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    joblib.dump(model, save_path)
    print(f"Random Forest model saved at: {save_path}")

    # Save scaler (same for all models)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")

    return model


if __name__ == "__main__":
    train_random_forest()
