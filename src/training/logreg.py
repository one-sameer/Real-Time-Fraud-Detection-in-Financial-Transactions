# Logistic Regression Training Script

import os
import json
import joblib
from sklearn.linear_model import LogisticRegression

# Flexible imports depending on execution path
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import MODELS_DIR, TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import MODELS_DIR, TUNING_DIR


def load_best_params(model_name="logreg"):
    """Load tuned hyperparameters if available."""
    file_path = os.path.join(TUNING_DIR, f"{model_name}_best.json")

    if os.path.exists(file_path):
        print(f"Loaded tuned parameters from: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    print("No tuned parameters found — using default parameters.")
    return None


def train_logreg():
    """Train Logistic Regression using tuned or default hyperparameters."""

    # Load dataset + preprocess
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Load tuned hyperparameters (if they exist)
    best_params = load_best_params("logreg")

    if best_params:
        model = LogisticRegression(**best_params)
    else:
        # Safe fallback
        model = LogisticRegression(max_iter=1000)

    print("Training Logistic Regression model...")
    model.fit(X_train_res, y_train_res)

    # Save trained model
    model_path = os.path.join(MODELS_DIR, "logreg_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    # Save the scaler (used during inference)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")

    return model


if __name__ == "__main__":
    train_logreg()
