from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def tnr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = tpr >= target_tpr
    if not np.any(mask):
        return np.nan
    return float(np.max(1.0 - fpr[mask]))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1.0 - 1e-12)
    return {
        "AUROC": float(roc_auc_score(y_true, clipped)),
        "AUPRC": float(average_precision_score(y_true, clipped)),
        "Brier": float(brier_score_loss(y_true, clipped)),
        "LogLoss": float(log_loss(y_true, clipped)),
        "TNR@TPR95": tnr_at_tpr(y_true, clipped, target_tpr=0.95),
    }
