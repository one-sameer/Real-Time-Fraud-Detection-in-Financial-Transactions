# Hyperparameter tuning for Random Forest

import os
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import TUNING_DIR


def tune_random_forest():
    """Tune Random Forest and save the best hyperparameters."""

    # Load + preprocess
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Hyperparameter search space
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    model = RandomForestClassifier(random_state=42)

    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,            # try 10 random combos
        scoring="roc_auc",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("Tuning Random Forest...")
    tuner.fit(X_train_res, y_train_res)

    best_params = tuner.best_params_
    print("Best Parameters:", best_params)

    # Save best params to tuning_results
    save_path = os.path.join(TUNING_DIR, "random_forest_best.json")
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Tuned parameters saved at: {save_path}")


if __name__ == "__main__":
    tune_random_forest()
