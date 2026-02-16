import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def train_logreg(X_train, y_train, X_dev, y_dev):
    """
    Logistic Regression with scaling (recommended).
    """
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_dev)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_dev, probs)
    acc = accuracy_score(y_dev, preds)
    return clf, {"auc": auc, "acc": acc}


def train_xgboost(X_train, y_train, X_dev, y_dev):
    """
    XGBoost classifier (CPU).
    """
    if XGBClassifier is None:
        raise ImportError("xgboost not installed. pip install xgboost")

    # scale_pos_weight helps imbalance; use ratio negatives/positives
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = float(neg) / float(max(pos, 1))

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",      # CPU fast
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_dev)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_dev, probs)
    acc = accuracy_score(y_dev, preds)
    return clf, {"auc": auc, "acc": acc}
