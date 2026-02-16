from __future__ import annotations

import os
import csv
from typing import Dict, Any, Optional, Tuple

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_core_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Returns: accuracy, precision, recall, f1, roc_auc.

    If y_proba is None, roc_auc will be NaN.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is None:
        metrics["roc_auc"] = float("nan")
    else:
        y_proba = np.asarray(y_proba).astype(float)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan")

    return metrics


def threshold_predictions(y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities into hard predictions."""
    y_proba = np.asarray(y_proba).astype(float)
    return (y_proba >= float(threshold)).astype(int)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Confusion matrix heatmap (matplotlib)."""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    _ensure_dir(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    # annotate
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def compare_modalities(
    results_dict: Dict[str, Dict[str, float]],
    save_path: str,
    metric: str = "f1",
    title: str = "Modality Comparison",
) -> None:
    """Compare text-only vs image-only vs fused.

    results_dict example:
        {
          "text_only": {"acc":..., "f1":..., ...},
          "image_only": {...},
          "fused": {...}
        }
    """
    import matplotlib.pyplot as plt

    keys = list(results_dict.keys())
    vals = [results_dict[k].get(metric, float("nan")) for k in keys]

    _ensure_dir(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(keys, vals)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def compare_fusion_methods(
    results_dict: Dict[str, Dict[str, float]],
    save_path: str,
    metric: str = "f1",
    title: str = "Fusion Method Comparison",
) -> None:
    """Compare concat vs mean vs gated (or any fusion keys you pass)."""
    import matplotlib.pyplot as plt

    keys = list(results_dict.keys())
    vals = [results_dict[k].get(metric, float("nan")) for k in keys]

    _ensure_dir(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(keys, vals)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_metrics_csv(
    rows: Dict[str, Dict[str, float]],
    save_path: str,
    extra_cols: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Save a dict-of-metrics into CSV.

    rows: {run_name: {metric_name: value}}
    extra_cols: {run_name: {col: value}}  (optional)
    """
    _ensure_dir(save_path)

    # union of all metric keys
    metric_keys = sorted({k for r in rows.values() for k in r.keys()})
    extra_keys = sorted({k for r in (extra_cols or {}).values() for k in r.keys()})

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + extra_keys + metric_keys)
        for run, metrics in rows.items():
            extras = (extra_cols or {}).get(run, {})
            writer.writerow(
                [run]
                + [extras.get(k, "") for k in extra_keys]
                + [metrics.get(k, "") for k in metric_keys]
            )


def generate_ablation_summary(
    all_results: Dict[str, Dict[str, float]],
    save_dir: str,
    title: str = "Ablation Summary",
    primary_metric: str = "f1",
) -> Dict[str, str]:
    """Create a comprehensive report directory.

    Produces:
    - metrics.csv
    - bar plot for primary metric

    Returns paths dict.
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "metrics.csv")
    save_metrics_csv(all_results, csv_path)

    # bar plot
    keys = list(all_results.keys())
    vals = [all_results[k].get(primary_metric, float("nan")) for k in keys]

    fig = plt.figure(figsize=(max(6, len(keys) * 0.9), 4))
    ax = fig.add_subplot(111)
    ax.bar(keys, vals)
    ax.set_title(title)
    ax.set_ylabel(primary_metric)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    plot_path = os.path.join(save_dir, f"{primary_metric}_bar.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return {"metrics_csv": csv_path, "primary_plot": plot_path}
