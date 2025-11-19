# Enhanced Ensemble Evaluation Script
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)

# flexible imports
try:
    from src.load_data import load_dataset
    from src.preprocess import preprocess_data
    from src.ensemble.ensemble import FraudEnsemble
    from src.config import TEST_SIZE, RANDOM_STATE, PROJECT_ROOT
except ModuleNotFoundError:
    from load_data import load_dataset
    from preprocess import preprocess_data
    from ensemble.ensemble import FraudEnsemble
    from config import TEST_SIZE, RANDOM_STATE, PROJECT_ROOT


SAVE_DIR = os.path.join(PROJECT_ROOT, "evaluation")
os.makedirs(SAVE_DIR, exist_ok=True)


def evaluate_ensemble():
    print("Loading dataset...")
    df = load_dataset()

    print("Preprocessing dataset...")
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    print("Loading Ensemble model...")
    ensemble = FraudEnsemble()

    print("Generating predictions...")
    y_prob = ensemble.predict_proba(X_test)
    y_pred = (y_prob >= ensemble.binary_threshold).astype(int)

    # ============================================
    # Compute Scores
    # ============================================
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=4)

    # Save metrics JSON
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report
    }

    with open(os.path.join(SAVE_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved textual results to evaluation_results.json")

    # ============================================
    # ---------- PLOTS ----------
    # ============================================

    # 1. Confusion Matrix heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
    plt.close()

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
    plt.close()

    # 3. Precision–Recall Curve
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "pr_curve.png"))
    plt.close()

    # 4. Probability distribution (Fraud vs Not Fraud)
    plt.figure(figsize=(6, 5))
    sns.histplot(y_prob[y_test == 0], bins=50, color="green", label="Non-Fraud", stat="density")
    sns.histplot(y_prob[y_test == 1], bins=50, color="red", label="Fraud", stat="density")
    plt.title("Fraud vs Non-Fraud Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "probability_distribution.png"))
    plt.close()

    # 5. F1 vs Threshold Curve
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores)
    plt.title("F1 Score vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "f1_vs_threshold.png"))
    plt.close()

    print("All evaluation plots saved to: evaluation/")



if __name__ == "__main__":
    evaluate_ensemble()