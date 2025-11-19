# This module handles all the preprocessing required for the loaded dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
try:
    from src.config import RANDOM_STATE, TEST_SIZE
except ModuleNotFoundError:
    from config import RANDOM_STATE, TEST_SIZE


def normalize(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def get_class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts()
    perc = y.value_counts(normalize=True) * 100
    return {"counts": counts, "percentages": perc}


def preprocess_data(df: pd.DataFrame):
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column as the target variable.")

    # Split features and labels
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Normalize features
    X_scaled, scaler = normalize(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    dist_before = get_class_distribution(y_train)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    dist_after = get_class_distribution(y_train_res)

    info = {
        "train_shape_before": X_train.shape,
        "train_shape_after": X_train_res.shape,
        "dist_before": dist_before,
        "dist_after": dist_after
    }

    return X_train_res, X_test, y_train_res, y_test, scaler, info


if __name__ == "__main__":
    from src.load_data import load_dataset

    df = load_dataset()
    X_train_res, X_test, y_train_res, y_test, scaler, info = preprocess_data(df)

    print("Preprocessing Summary:")
    print(f"Original training size: {info['train_shape_before']}")
    print(f"After SMOTE: {info['train_shape_after']}")
    print("\nClass distribution BEFORE SMOTE:")
    print(info["dist_before"]["counts"])
    print(info["dist_before"]["percentages"])
    print("\nClass distribution AFTER SMOTE:")
    print(info["dist_after"]["counts"])
    print(info["dist_after"]["percentages"])
