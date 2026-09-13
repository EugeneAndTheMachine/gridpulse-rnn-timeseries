"""Metrics for anomaly detection."""
import numpy as np
from gridpulse.anomaly.event_grouping import group_anomalies


def point_wise_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Standard classification metrics per point.

    y_true, y_pred: boolean arrays. True = anomaly.
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def event_wise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    max_gap: int = 2,
    min_duration: int = 1,
) -> dict[str, float]:
    """
    Event-level precision / recall / F1 based on overlap.

    A true event counts as detected (TP) if any predicted event overlaps it.
    A predicted event with no overlap on any true event is a FP.
    A true event with no overlapping prediction is a FN.

    Args:
        y_true, y_pred: 0/1 or boolean arrays at point level.
        scores: optional score array used when grouping (falls back to y_pred/y_true as float).
        max_gap, min_duration: forwarded to group_anomalies.
    """
    y_true_b = y_true.astype(bool)
    y_pred_b = y_pred.astype(bool)

    true_scores = scores if scores is not None else y_true_b.astype(float)
    pred_scores = scores if scores is not None else y_pred_b.astype(float)

    true_events = group_anomalies(y_true_b, true_scores, max_gap=max_gap, min_duration=min_duration)
    pred_events = group_anomalies(y_pred_b, pred_scores, max_gap=max_gap, min_duration=min_duration)

    def overlaps(a, b) -> bool:
        return not (a.end_idx < b.start_idx or b.end_idx < a.start_idx)

    tp = sum(1 for te in true_events if any(overlaps(te, pe) for pe in pred_events))
    fn = len(true_events) - tp
    fp = sum(1 for pe in pred_events if not any(overlaps(pe, te) for te in true_events))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_events": len(true_events),
        "pred_events": len(pred_events),
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
    }