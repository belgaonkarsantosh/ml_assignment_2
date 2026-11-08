"""
Train five classifiers on the UCI Dry Bean dataset and save them for the Streamlit app.

Outputs
-------
model/<name>.joblib        fitted sklearn pipelines (scaler + classifier)
model/label_encoder.joblib maps class names <-> integer labels
model/metrics.csv          hold-out metrics for every model
model/feature_columns.json input feature order expected by the models
test_data.csv              stratified hold-out split (features + Class), used by the app
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dry_bean.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Class"
SEED = 7

# SVC(probability=True) is deprecated from scikit-learn 1.9; requirements.txt pins 1.9.0.
warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluate(model, X, y_true):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def build_searches():
    """Each model is a scaler + classifier pipeline tuned with 5-fold stratified CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def search(estimator, grid):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        return GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)

    return {
        "Logistic Regression": search(
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.1, 1, 10, 100]},
        ),
        "Decision Tree": search(
            DecisionTreeClassifier(random_state=SEED),
            {
                "clf__max_depth": [6, 8, 10, 12, None],
                "clf__min_samples_leaf": [1, 5, 10],
            },
        ),
        "KNN": search(
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 9, 15, 21], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": search(
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        "Support Vector Machine": search(
            SVC(kernel="rbf", probability=True, random_state=SEED),
            {"clf__C": [1, 10], "clf__gamma": ["scale", 0.05]},
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_raw} rows, {n_raw - len(df)} duplicates removed -> {len(df)} rows")

    feature_cols = [c for c in df.columns if c != TARGET]
    encoder = LabelEncoder().fit(df[TARGET])
    X = df[feature_cols]
    y = encoder.transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Classes: {list(encoder.classes_)}")

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name, search in build_searches().items():
        start = time.time()
        search.fit(X_train, y_train)
        best = search.best_estimator_
        scores = evaluate(best, X_test, y_test)
        joblib.dump(best, MODEL_DIR / MODEL_FILES[name])
        rows.append({"ML Model Name": name, **scores, "Best params": json.dumps(search.best_params_)})
        print(
            f"{name:24s} acc={scores['Accuracy']:.4f} auc={scores['AUC']:.4f} "
            f"f1={scores['F1']:.4f} mcc={scores['MCC']:.4f}  "
            f"{search.best_params_}  ({time.time() - start:.1f}s)"
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    test_df = X_test.copy()
    test_df[TARGET] = encoder.inverse_transform(y_test)
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    print("\nHold-out comparison:")
    print(metrics.drop(columns="Best params").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
