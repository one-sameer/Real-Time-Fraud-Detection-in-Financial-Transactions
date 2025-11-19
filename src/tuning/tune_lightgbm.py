# Hyperparameter tuning for LightGBM

import os
import json
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import TUNING_DIR


def tune_lightgbm():
    """Tune LightGBM and save the best hyperparameters."""

    # Load + preprocess
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Parameter space
    param_grid = {
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300, 500],
        "min_child_samples": [5, 10, 20, 30],
        "boosting_type": ["gbdt"],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    model = LGBMClassifier(random_state=42)

    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=12,
        scoring="roc_auc",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("Tuning LightGBM...")
    tuner.fit(X_train_res, y_train_res)

    best_params = tuner.best_params_
    print("Best LightGBM Params:", best_params)

    # Save results
    save_path = os.path.join(TUNING_DIR, "lightgbm_best.json")
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Tuned hyperparameters saved at: {save_path}")


if __name__ == "__main__":
    tune_lightgbm()
