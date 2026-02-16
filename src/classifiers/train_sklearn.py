import os
import json
import numpy as np

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# Threshold tuning + metrics
def tune_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
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


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auc": float(auc),
        "acc": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
    }


def _summarize_folds(rows: list[Dict[str, float]]) -> Dict[str, Any]:
    # rows contain keys: auc, acc, f1, best_threshold
    out: Dict[str, Any] = {"folds": rows}
    for k in ["auc", "acc", "f1", "best_threshold"]:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals))
    return out


# Config
@dataclass
class SklearnConfig:
    threshold_metric: str = "f1"      # "f1" or "acc" (used for dev threshold tuning + CV fold threshold tuning)
    save_dir: str = "models/sklearn"
    cv_folds: int = 0                # 0/1 disables CV; >=2 runs Stratified K-Fold on TRAIN only
    cv_seed: int = 42


# Cross-validation helpers (TRAIN ONLY)
def cross_validate_logreg(
    X: np.ndarray,
    y: np.ndarray,
    cfg: SklearnConfig,
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.cv_seed)
    fold_rows: list[Dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=-1,
                solver="lbfgs"
            ))
        ])

        model.fit(X[tr_idx], y[tr_idx])
        p_val = model.predict_proba(X[va_idx])[:, 1]

        thr, _ = tune_threshold(y[va_idx], p_val, metric=cfg.threshold_metric)
        m = compute_metrics(y[va_idx], p_val, thr)
        fold_rows.append({
            "fold": float(fold),
            "auc": m["auc"],
            "acc": m["acc"],
            "f1": m["f1"],
            "best_threshold": float(thr),
        })

    return _summarize_folds(fold_rows)


def cross_validate_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cfg: SklearnConfig,
    params: Optional[Dict] = None,
) -> Dict[str, Any]:
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. pip install xgboost")

    params = params or {}
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.cv_seed)
    fold_rows: list[Dict[str, float]] = []

    base_defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        y_tr = y[tr_idx]
        neg = float((y_tr == 0).sum())
        pos = float((y_tr == 1).sum())
        scale_pos_weight = neg / max(pos, 1.0)

        fold_params = dict(base_defaults)
        fold_params.update(params)
        fold_params["scale_pos_weight"] = scale_pos_weight

        model = xgb.XGBClassifier(**fold_params)
        model.fit(X[tr_idx], y_tr)

        p_val = model.predict_proba(X[va_idx])[:, 1]
        thr, _ = tune_threshold(y[va_idx], p_val, metric=cfg.threshold_metric)
        m = compute_metrics(y[va_idx], p_val, thr)

        fold_rows.append({
            "fold": float(fold),
            "auc": m["auc"],
            "acc": m["acc"],
            "f1": m["f1"],
            "best_threshold": float(thr),
        })

    return _summarize_folds(fold_rows)


# Trainers
def train_logreg(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev: np.ndarray, y_dev: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: Optional[SklearnConfig] = None,
    run_name: str = "logreg"
) -> Dict[str, Any]:
    """
    Logistic Regression (CPU) with class_weight='balanced'.

    - Optional: Stratified K-Fold CV on TRAIN only (cfg.cv_folds >= 2)
      Reports mean±std metrics across folds (threshold tuned inside each fold val).
    - Final model: fit on full TRAIN, tune threshold on DEV, report TRAIN/DEV/TEST.
    """
    cfg = cfg or SklearnConfig()

    out: Dict[str, Any] = {}

    # ---- Optional CV on TRAIN only
    if cfg.cv_folds and cfg.cv_folds >= 2:
        out["cv"] = cross_validate_logreg(X_train, y_train, cfg)

    # ---- Final fit on full TRAIN
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs"
        ))
    ])

    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_dev = model.predict_proba(X_dev)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # tune threshold on DEV only
    thr, _ = tune_threshold(y_dev, p_dev, metric=cfg.threshold_metric)

    m_train = compute_metrics(y_train, p_train, thr)
    m_dev = compute_metrics(y_dev, p_dev, thr)
    m_test = compute_metrics(y_test, p_test, thr)

    out.update({
        "best_threshold": float(thr),
        "train_auc": m_train["auc"], "train_acc": m_train["acc"], "train_f1": m_train["f1"],
        "dev_auc": m_dev["auc"], "dev_acc": m_dev["acc"], "dev_f1": m_dev["f1"],
        "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
    })

    os.makedirs(cfg.save_dir, exist_ok=True)
    import joblib
    joblib.dump(model, os.path.join(cfg.save_dir, f"{run_name}.joblib"))
    with open(os.path.join(cfg.save_dir, f"{run_name}_meta.json"), "w") as f:
        json.dump(out, f, indent=2)

    return out


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev: np.ndarray, y_dev: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: Optional[SklearnConfig] = None,
    run_name: str = "xgb",
    params: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    XGBoost (CPU).

    - Optional: Stratified K-Fold CV on TRAIN only (cfg.cv_folds >= 2)
      Reports mean±std metrics across folds (threshold tuned inside each fold val).
    - Final model: fit on full TRAIN, tune threshold on DEV, report TRAIN/DEV/TEST.
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. pip install xgboost")

    cfg = cfg or SklearnConfig()
    params = params or {}

    out: Dict[str, Any] = {}

    # ---- Optional CV on TRAIN only
    if cfg.cv_folds and cfg.cv_folds >= 2:
        out["cv"] = cross_validate_xgboost(X_train, y_train, cfg, params=params)

    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1.0)

    default_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_dev = model.predict_proba(X_dev)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    thr, _ = tune_threshold(y_dev, p_dev, metric=cfg.threshold_metric)

    m_train = compute_metrics(y_train, p_train, thr)
    m_dev = compute_metrics(y_dev, p_dev, thr)
    m_test = compute_metrics(y_test, p_test, thr)

    out.update({
        "best_threshold": float(thr),
        "train_auc": m_train["auc"], "train_acc": m_train["acc"], "train_f1": m_train["f1"],
        "dev_auc": m_dev["auc"], "dev_acc": m_dev["acc"], "dev_f1": m_dev["f1"],
        "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
        "scale_pos_weight": float(scale_pos_weight)
    })

    os.makedirs(cfg.save_dir, exist_ok=True)
    model.save_model(os.path.join(cfg.save_dir, f"{run_name}.json"))
    with open(os.path.join(cfg.save_dir, f"{run_name}_meta.json"), "w") as f:
        json.dump(out, f, indent=2)

    return out