"""
Tune thresholds for the ensemble (no calibration).
Saves results to tuning_results/thresholds.json.
"""

import os
import json
import numpy as np
from sklearn.metrics import f1_score, recall_score

# Flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.ensemble.ensemble import FraudEnsemble
    from src.config import TUNING_DIR
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from ensemble.ensemble import FraudEnsemble
    from config import TUNING_DIR


# -----------------------------------------------------
def find_best_binary_threshold(y_true, y_prob):
    thresholds = np.linspace(0, 1, 101)
    best = {"thr": 0.5, "score": -1}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best["score"]:
            best = {"thr": float(t), "score": float(score)}

    return best


# -----------------------------------------------------
def evaluate_three_class(y_true, y_prob, safe_thr, fraud_thr):
    y_pred = []
    for p in np.ravel(y_prob):
        if p < safe_thr:
            y_pred.append(0)    # safe
        elif p > fraud_thr:
            y_pred.append(2)    # fraud
        else:
            y_pred.append(1)    # review

    y_pred = np.array(y_pred)

    # fraud recall
    y_true_bin = (y_true == 1).astype(int)
    fraud_selected = (y_pred == 2).astype(int)
    fraud_recall = recall_score(y_true_bin, fraud_selected)

    # safe specificity
    safe_selected = (y_pred == 0).astype(int)
    y_true_safe = (y_true == 0).astype(int)

    tn = np.sum((y_true_safe == 1) & (safe_selected == 1))
    fp = np.sum((y_true_safe == 0) & (safe_selected == 1))

    safe_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    review_rate = np.mean(y_pred == 1)

    return {
        "fraud_recall": float(fraud_recall),
        "safe_specificity": float(safe_specificity),
        "review_rate": float(review_rate)
    }


# -----------------------------------------------------
def tune_pair_thresholds(y_true, y_prob, alpha=0.7, beta=0.3):
    safe_grid = np.linspace(0.0, 0.49, 50)
    fraud_grid = np.linspace(0.51, 1.0, 50)

    best = {"score": -1}

    for s in safe_grid:
        for f in fraud_grid:
            if s >= f:
                continue

            m = evaluate_three_class(y_true, y_prob, s, f)
            score = alpha * m["fraud_recall"] + beta * m["safe_specificity"]

            if score > best["score"]:
                best = {
                    "score": float(score),
                    "safe_thr": float(s),
                    "fraud_thr": float(f),
                    "metrics": m
                }

    return best


# -----------------------------------------------------
def main():
    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    ensemble = FraudEnsemble()

    # -------------------------------------------------
    # Generate ensemble probabilities (no calibration)
    # -------------------------------------------------
    print("Generating ensemble probabilities on validation/test set...")
    y_prob = ensemble.predict_proba(X_test)
    y_prob = np.array(y_prob).ravel()

    # 1) Binary threshold tuning (F1)
    best_bin = find_best_binary_threshold(y_test, y_prob)
    print("Best binary threshold (F1):", best_bin)

    # 2) Pair thresholds tuning (weighted objective)
    pair = tune_pair_thresholds(y_test, y_prob, alpha=0.7, beta=0.3)
    print("Best pair thresholds:", pair)

    out = {
        "best_binary": best_bin,
        "best_pair": pair
    }

    os.makedirs(TUNING_DIR, exist_ok=True)
    save_path = os.path.join(TUNING_DIR, "thresholds.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=4)

    print("Saved thresholds to:", save_path)


if __name__ == "__main__":
    main()