# Hyperparameter tuning for Naive Bayes (GaussianNB)

import os
import json
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import TUNING_DIR


def tune_naive_bayes():
    """Tune Naive Bayes and save best hyperparameters."""

    # Load and preprocess data
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Tuning space: only var_smoothing
    param_grid = {
        "var_smoothing": np.logspace(-12, -6, 10)
    }

    model = GaussianNB()

    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("Tuning Naive Bayes...")
    tuner.fit(X_train_res, y_train_res)

    best_params = tuner.best_params_
    print("Best Params:", best_params)

    # Save tuned parameters
    save_path = os.path.join(TUNING_DIR, "naive_bayes_best.json")
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Tuned hyperparameters saved at: {save_path}")


if __name__ == "__main__":
    tune_naive_bayes()
