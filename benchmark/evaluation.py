"""Classification metrics shared by the suite evaluator and comparison report."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, y_score):
    if not len(y_true):
        return {key: None for key in (
            "accuracy", "precision", "recall", "f1", "fpr", "fnr",
            "balanced_accuracy", "auroc", "majority_accuracy", "positive_rate",
        )} | {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    )
    positives, negatives = tp + fn, tn + fp
    recall = recall_score(y_true, y_pred, zero_division=0) if positives else None
    fpr = fp / negatives if negatives else None
    return {
        "n": len(y_true), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall) if recall is not None else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": fpr, "fnr": fn / positives if positives else None,
        "balanced_accuracy": (recall + 1 - fpr) / 2 if positives and negatives else None,
        "auroc": float(roc_auc_score(y_true, y_score)) if positives and negatives else None,
        "majority_accuracy": max(positives, negatives) / len(y_true),
        "positive_rate": (tp + fp) / len(y_true),
    }
