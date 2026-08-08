"""
train_models.py
================
Trains five classification models on the UCI Wine Quality dataset (red + white
combined) and evaluates each on a held-out test set with six metrics:
Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient.

The task: predict whether a wine is "good" (quality >= 7) from its 12 physico-
chemical features (11 chemical measurements + a wine_type indicator).

Running this script:
    * downloads the two UCI CSVs (red, white),
    * builds a combined, balanced-feature dataset (12 features, 6497 rows),
    * does a stratified 80/20 train/test split,
    * standardises features (fit on train only),
    * trains LogisticRegression, DecisionTree, kNN, GaussianNB, RandomForest,
    * prints a metric comparison table,
    * saves each fitted model + the scaler to model/*.joblib,
    * writes test_data.csv (raw test rows, for the Streamlit app to consume).
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RANDOM_STATE = 42


def load_data():
    """Load red + white wine, add a wine_type feature, build a binary 'good' target."""
    red = pd.read_csv(RED_URL, sep=";")
    white = pd.read_csv(WHITE_URL, sep=";")
    red["wine_type"] = 0      # 0 = red
    white["wine_type"] = 1    # 1 = white
    df = pd.concat([red, white], ignore_index=True)

    # Binary target: a wine is "good" if its quality rating is 7 or higher.
    df["good"] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns=["quality"])       # drop the raw score; keep only the label
    return df


def get_models():
    """Return the five classification models required by the assignment."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def evaluate(model, X_test, y_test):
    """Compute the six required metrics for a fitted model."""
    y_pred = model.predict(X_test)
    # AUC needs a probability / score for the positive class.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    df = load_data()
    feature_cols = [c for c in df.columns if c != "good"]
    X = df[feature_cols].values
    y = df["good"].values
    print(f"Dataset: {df.shape[0]} instances, {len(feature_cols)} features")
    print(f"Positive rate (good wines): {y.mean():.3f}\n")

    # Stratified 80/20 split preserves the class imbalance in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Standardise features. Fit on train only, then apply to test (no leakage).
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # The models train in well under a second, so the Streamlit app re-trains them
    # on startup rather than shipping large pickled files. We therefore do NOT save
    # *.joblib artifacts here — the repo keeps only the training code (.py / .ipynb).
    results = {}
    for name, model in get_models().items():
        model.fit(X_train_s, y_train)
        results[name] = evaluate(model, X_test_s, y_test)

    # Print the comparison table (README uses these exact numbers).
    table = pd.DataFrame(results).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    pd.set_option("display.width", 120)
    print("Metric comparison (test set):\n")
    print(table.round(4).to_string())
    print("\nOverall winner (by MCC): ", table["MCC"].idxmax())

    # Save the raw (un-scaled) test rows for the Streamlit app + submission.
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["good"] = y_test
    test_df.to_csv(os.path.join(REPO, "test_data.csv"), index=False)
    print(f"\nSaved test_data.csv with {len(test_df)} rows to repo root.")


if __name__ == "__main__":
    main()
