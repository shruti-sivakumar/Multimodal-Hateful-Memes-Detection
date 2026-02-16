from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _pos_weight_from_labels(y: np.ndarray) -> float:
    neg = float((y == 0).sum())
    pos = float((y == 1).sum())
    pos = max(pos, 1.0)
    return neg / pos


def tune_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    from sklearn.metrics import f1_score, accuracy_score

    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds)
        elif metric == "acc":
            score = accuracy_score(y_true, preds)
        else:
            raise ValueError("metric must be 'f1' or 'acc'")
        if score > best_score:
            best_score, best_t = float(score), float(t)
    return best_t, best_score


@dataclass
class SklearnTrainConfig:
    threshold_metric: str = "f1"  # "f1" or "acc"

    # CV (optional)
    kfold_cv: bool = False
    n_splits: int = 5
    random_state: int = 42


ReturnDict = Dict[str, Union[str, float]]


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# -----------------------------
# Logistic Regression
# -----------------------------


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    save_path: str,
    cfg: Optional[SklearnTrainConfig] = None,
) -> ReturnDict:
    """Train Logistic Regression (CPU) on embeddings.

    - Uses StandardScaler + LogisticRegression(class_weight='balanced').
    - Optional Stratified K-fold CV on train for reporting.
    - Fits final model on full train.
    - Tunes best_threshold on dev.
    """

    if cfg is None:
        cfg = SklearnTrainConfig()

    pw = _pos_weight_from_labels(y_train)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=None,
                ),
            ),
        ]
    )

    cv_report: Optional[Dict[str, Any]] = None
    if cfg.kfold_cv:
        from sklearn.metrics import roc_auc_score

        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            pipe.fit(X_tr, y_tr)
            va_probs = pipe.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, va_probs) if len(np.unique(y_va)) > 1 else float("nan")
            aucs.append(float(auc))

        cv_report = {
            "n_splits": cfg.n_splits,
            "random_state": cfg.random_state,
            "fold_auc": aucs,
            "mean_auc": float(np.nanmean(aucs)),
            "std_auc": float(np.nanstd(aucs)),
        }

    # Fit on full train
    pipe.fit(X_train, y_train)

    # Tune threshold on dev
    dev_probs = pipe.predict_proba(X_dev)[:, 1]
    best_thr, _ = tune_threshold(y_dev.astype(int), dev_probs, metric=cfg.threshold_metric)

    _ensure_dir(save_path)
    import joblib

    joblib.dump(
        {
            "pipeline": pipe,
            "best_threshold": float(best_thr),
            "pos_weight": float(pw),
            "cv_report": cv_report,
        },
        save_path,
    )
    print(f"Saved: {save_path}")

    # Optional: also save a sidecar json for quick report viewing
    if cfg.kfold_cv:
        sidecar = os.path.splitext(save_path)[0] + "_cv.json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(cv_report, f, indent=2)
        print(f"Saved CV report: {sidecar}")

    return {"model_path": save_path, "best_threshold": float(best_thr), "pos_weight": float(pw)}


# -----------------------------
# XGBoost
# -----------------------------


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    save_path: str,
    cfg: Optional[SklearnTrainConfig] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> ReturnDict:
    """Train XGBoost (CPU) on embeddings.

    - Uses scale_pos_weight = neg/pos.
    - Optional Stratified K-fold CV on train for reporting.
    - Fits final model on full train.
    - Tunes best_threshold on dev.
    """

    if cfg is None:
        cfg = SklearnTrainConfig()

    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost not installed. pip install xgboost") from e

    pw = _pos_weight_from_labels(y_train)

    params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # good CPU default
        "random_state": 42,
        "scale_pos_weight": pw,
    }
    if xgb_params:
        params.update(xgb_params)

    cv_report: Optional[Dict[str, Any]] = None
    if cfg.kfold_cv:
        from sklearn.metrics import roc_auc_score

        skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            va_probs = clf.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, va_probs) if len(np.unique(y_va)) > 1 else float("nan")
            aucs.append(float(auc))

        cv_report = {
            "n_splits": cfg.n_splits,
            "random_state": cfg.random_state,
            "fold_auc": aucs,
            "mean_auc": float(np.nanmean(aucs)),
            "std_auc": float(np.nanstd(aucs)),
        }

    # Fit final model
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)

    dev_probs = clf.predict_proba(X_dev)[:, 1]
    best_thr, _ = tune_threshold(y_dev.astype(int), dev_probs, metric=cfg.threshold_metric)

    _ensure_dir(save_path)
    import joblib

    joblib.dump(
        {
            "model": clf,
            "best_threshold": float(best_thr),
            "pos_weight": float(pw),
            "params": params,
            "cv_report": cv_report,
        },
        save_path,
    )
    print(f"Saved: {save_path}")

    if cfg.kfold_cv:
        sidecar = os.path.splitext(save_path)[0] + "_cv.json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(cv_report, f, indent=2)
        print(f"Saved CV report: {sidecar}")

    return {"model_path": save_path, "best_threshold": float(best_thr), "pos_weight": float(pw)}
