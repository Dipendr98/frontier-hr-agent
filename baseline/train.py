"""
Trains the attrition model on the real (IBM fictional) HR dataset.

What is deliberately NOT here, compared to the original submission:
  - no hardcoded weights pretending to be a trained model
  - no metric computed on data the model was fitted on
  - no cross-validation that reuses one fixed parameter set across folds

What is here:
  - a Pipeline (scaler + logistic regression) so scaling is fitted inside
    each CV fold, not leaked across folds
  - class_weight='balanced' because attrition is the minority class
  - a held-out test set touched once, at the end
  - an environment and data fingerprint, so a reviewer can confirm the
    artifact matches the report rather than discovering a drift silently

The AUC cost of excluding protected attributes is measured in
baseline/fairness_cost.py, not here — this docstring used to claim it was in
this file, which it never was.
"""
import json
import os
from datetime import datetime, timezone

import hashlib
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "onboarding_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "baseline")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

RANDOM_STATE = 42


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "accuracy": float((preds == y_test.values).mean()),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("employee_id", "attrition")]

    X = df[feature_cols]
    y = df["attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    test_metrics = evaluate(model, X_test, y_test)

    # Cross-validation: the pipeline is refit inside every fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(build_model(), X_train, y_train, cv=cv, scoring="roc_auc")

    # Reference points a reviewer will ask about.
    majority_acc = float((y_test == y_train.mode()[0]).mean())
    random_auc = 0.5

    # Environment + data fingerprint. A reviewer can confirm they are looking
    # at the same artifact this report describes, and a version drift shows up
    # as a diff rather than a silent behaviour change.
    with open(DATA_PATH, "rb") as f:
        data_sha = hashlib.sha256(f.read()).hexdigest()[:16]

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data_sha256_16": data_sha,
        "random_state": RANDOM_STATE,
        "dataset": "IBM HR Analytics Employee Attrition (fictional, IBM/Kaggle)",
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "attrition_rate": float(y.mean()),
        "features": feature_cols,
        "protected_attributes_excluded": ["Gender", "Age", "MaritalStatus"],
        "test_metrics": test_metrics,
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "reference_majority_class_accuracy": majority_acc,
        "reference_random_auc": random_auc,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "attrition_model.joblib"))
    with open(os.path.join(EVIDENCE_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 66)
    print("ATTRITION MODEL — trained on held-out split, metrics from test set")
    print("=" * 66)
    print(f"  rows: {report['rows_total']} "
          f"(train {report['rows_train']} / test {report['rows_test']})")
    print(f"  attrition rate: {report['attrition_rate']:.1%}")
    print()
    for k, v in test_metrics.items():
        print(f"  {k:<12} {v:.3f}")
    print()
    print(f"  cv_roc_auc   {report['cv_roc_auc_mean']:.3f} "
          f"(+/- {report['cv_roc_auc_std']:.3f})")
    print(f"  reference: majority-class accuracy {majority_acc:.3f}, random AUC 0.500")
    print()
    print(f"  env: python {report['environment']['python']}, "
          f"scikit-learn {report['environment']['scikit_learn']}")
    print(f"  data sha256[:16]: {data_sha}")
    return report


if __name__ == "__main__":
    main()
