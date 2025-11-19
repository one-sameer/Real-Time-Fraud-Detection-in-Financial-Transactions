# Hyperparameter tuning for SVM (Support Vector Machine)

import os
import json
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.config import TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from config import TUNING_DIR


def tune_svm():
    """Tune SVM hyperparameters and save the best ones."""

    # Load + preprocess
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    # Hyperparameter search space (balanced for speed + performance)
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.001, 0.01, 0.1],
        "kernel": ["rbf"],
        "probability": [True],   # Needed for ROC-AUC + ensemble
    }

    model = SVC()

    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("Tuning SVM...")
    tuner.fit(X_train_res, y_train_res)

    best_params = tuner.best_params_
    print("Best Params:", best_params)

    # Save tuned parameters
    save_path = os.path.join(TUNING_DIR, "svm_best.json")
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Tuned hyperparameters saved at: {save_path}")


if __name__ == "__main__":
    tune_svm()