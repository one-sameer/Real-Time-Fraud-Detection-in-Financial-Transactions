# Hyperparameter tuning for Logistic Regression

import os
import json
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import TUNING_DIR


def tune_logreg():
    """Tune Logistic Regression and save best hyperparameters."""

    # Load dataset + preprocess
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Hyperparameter search space
    param_grid = {
        "C": np.logspace(-3, 3, 10),
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [200, 500, 1000]
    }

    model = LogisticRegression()

    # Randomized Search (faster than grid)
    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("Tuning Logistic Regression...")
    tuner.fit(X_train_res, y_train_res)

    best_params = tuner.best_params_
    print("Best Params:", best_params)

    # Save to tuning results folder
    save_path = os.path.join(TUNING_DIR, "logreg_best.json")
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Tuned hyperparameters saved at: {save_path}")


if __name__ == "__main__":
    tune_logreg()
