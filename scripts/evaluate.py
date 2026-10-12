from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
OUTPUT_DIR = Path("outputs")
def evaluate(dataset: str):
    path = (
        OUTPUT_DIR /
        f"{dataset}_predictions.csv"
    )
    df = pd.read_csv(path)
    print("\n" + "=" * 70)
    print(dataset.upper())
    print("=" * 70)
    print("Total rows:", len(df))
    # -----------------------------------------------
    # Remove failed requests
    # -----------------------------------------------
    valid = df[
        df["error"].fillna("") == ""
    ].copy()
    print("Valid:", len(valid))
    print(
        "Errors:",
        len(df) - len(valid)
    )
    y_true = valid["label"].astype(int)
    y_pred = valid["prediction"].astype(int)
    y_score = valid["score"].astype(float)
    # -----------------------------------------------
    # Metrics
    # -----------------------------------------------
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    auroc = roc_auc_score(
        y_true,
        y_score,
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    # -----------------------------------------------
    # Latency
    # -----------------------------------------------
    latency = valid[
        "latency_ms"
    ].dropna()
    print("\nMetrics")
    print("-------------------------")
    print(
        f"Accuracy : {accuracy:.4f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall   : {recall:.4f}"
    )
    print(
        f"F1       : {f1:.4f}"
    )
    print(
        f"AUROC    : {auroc:.4f}"
    )
    print("\nConfusion Matrix")
    print("-------------------------")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print("\nLatency")
    print("-------------------------")
    print(
        f"Mean: {latency.mean():.2f} ms"
    )
    print(
        f"P50 : {np.percentile(latency, 50):.2f} ms"
    )
    print(
        f"P95 : {np.percentile(latency, 95):.2f} ms"
    )
    print(
        f"P99 : {np.percentile(latency, 99):.2f} ms"
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
    )
    args = parser.parse_args()
    evaluate(args.dataset)
if __name__ == "__main__":
    main()

