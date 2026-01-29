import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Classification metrics:
    accuracy, macro precision, recall, F1, and FPR.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.size == 0 or y_true.shape != y_pred.shape:
        return {
            "acc": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "fpr_macro": 0.0,
        }

    n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    precisions, recalls, f1s, fprs = [], [], [], []

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        fprs.append(fpr)

    return {
        "acc": acc,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "fpr_macro": float(np.mean(fprs)),
    }


def compute_latency_median_ms(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Median inference latency per sample (ms).
    """
    if X.size == 0:
        return 0.0

    model = model.to(device)
    model.eval()

    times = []
    n = X.shape[0]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            t0 = time.time()
            _ = model(batch)
            t1 = time.time()
            times.append((t1 - t0) * 1000.0 / max(1, len(batch)))

    return float(np.median(times)) if times else 0.0