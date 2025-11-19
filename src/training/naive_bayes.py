# Naive Bayes Training Script

import os
import json
import joblib
from sklearn.naive_bayes import GaussianNB

# Flexible imports depending on execution path
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import MODELS_DIR, TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import MODELS_DIR, TUNING_DIR


def load_best_params(model_name="naive_bayes"):
    """Load tuned hyperparameters if available."""
    file_path = os.path.join(TUNING_DIR, f"{model_name}_best.json")

    if os.path.exists(file_path):
        print(f"Loaded tuned parameters from: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    print("No tuned parameters found — using default parameters.")
    return None


def train_naive_bayes():
    """Train Naive Bayes using tuned or default parameters."""

    # Load + preprocess dataset
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Load tuned hyperparameters if exist
    best_params = load_best_params("naive_bayes")

    if best_params:
        model = GaussianNB(**best_params)
    else:
        model = GaussianNB()

    print("Training Naive Bayes model...")
    model.fit(X_train_res, y_train_res)

    # Save model
    save_path = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")
    joblib.dump(model, save_path)
    print(f"Model saved at: {save_path}")

    # Save scaler (same for all models)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")

    return model


if __name__ == "__main__":
    train_naive_bayes()
