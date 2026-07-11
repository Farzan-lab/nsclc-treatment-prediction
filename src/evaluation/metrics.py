"""
Evaluation metrics for treatment prediction.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Compute all classification metrics.
    Returns dict with accuracy, macro_f1, weighted_f1, auc_ovr.
    """
    metrics = {
        'accuracy':    accuracy_score(y_true, y_pred),
        'macro_f1':    f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
    }

    if y_prob is not None:
        try:
            metrics['auc_ovr'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            )
        except Exception:
            metrics['auc_ovr'] = float('nan')

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "") -> None:
    """Pretty print metrics."""
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")
