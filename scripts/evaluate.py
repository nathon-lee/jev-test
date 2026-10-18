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
from benchmark.pii_metrics import compute_pii_metrics

OUTPUT_DIR = Path("outputs")


def _normalize_dataset_name(dataset: str) -> str:
    return dataset.strip().lower().replace("-", "_")


def _iter_candidate_paths(dataset: str, provider: str | None = None, prompt_version: str | None = None, threshold: float | None = None):
    dataset_stem = _normalize_dataset_name(dataset)
    candidates = []

    if provider:
        provider_dir = OUTPUT_DIR / provider
        if provider_dir.exists():
            candidates.append(provider_dir / f"{dataset}.csv")
            candidates.append(provider_dir / f"{dataset}_predictions.csv")
            if prompt_version and threshold is not None:
                candidates.append(provider_dir / f"{dataset}_{prompt_version}_t{threshold}.csv")
                candidates.append(provider_dir / f"{dataset}_{prompt_version}_t{float(threshold):.1f}.csv")

    candidates.extend([
        OUTPUT_DIR / f"{dataset}.csv",
        OUTPUT_DIR / f"{dataset}_predictions.csv",
        OUTPUT_DIR / f"{dataset_stem}_predictions.csv",
    ])

    if prompt_version and threshold is not None:
        candidates.extend([
            OUTPUT_DIR / f"{dataset}_{prompt_version}_t{threshold}.csv",
            OUTPUT_DIR / f"{dataset}_{prompt_version}_t{float(threshold):.1f}.csv",
            OUTPUT_DIR / f"{dataset_stem}_{prompt_version}_t{threshold}.csv",
            OUTPUT_DIR / f"{dataset_stem}_{prompt_version}_t{float(threshold):.1f}.csv",
        ])

    seen = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            yield path


def resolve_prediction_path(dataset: str, provider: str | None = None, prompt_version: str | None = None, threshold: float | None = None) -> Path:
    dataset_name = dataset.strip()
    if prompt_version and threshold is not None:
        for path in _iter_candidate_paths(dataset_name, provider=provider, prompt_version=prompt_version, threshold=threshold):
            if path.exists():
                return path

    for path in _iter_candidate_paths(dataset_name, provider=provider, prompt_version=prompt_version, threshold=threshold):
        if path.exists():
            return path

    if prompt_version and threshold is not None:
        searched = []
        for root in OUTPUT_DIR.glob("**/*.csv"):
            name = root.name
            if dataset_name in name and f"_{prompt_version}_t{threshold}" in name:
                return root
            if _normalize_dataset_name(dataset_name) in _normalize_dataset_name(name) and f"_{prompt_version}_t{threshold}" in name:
                searched.append(root)
        if searched:
            return searched[0]

    raise FileNotFoundError(
        f"No prediction file found for dataset '{dataset}'"
        f"{f' provider={provider}' if provider else ''}"
        f"{f' prompt_version={prompt_version}' if prompt_version else ''}"
        f"{f' threshold={threshold}' if threshold is not None else ''}."
    )


def evaluate(dataset: str, provider: str | None = None, prompt_version: str | None = None, threshold: float | None = None):
    path = resolve_prediction_path(dataset, provider=provider, prompt_version=prompt_version, threshold=threshold)
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
def evaluate_pii(dataset: str, provider: str | None = None, prompt_version: str | None = None, threshold: float | None = None):
    path = resolve_prediction_path(dataset, provider=provider, prompt_version=prompt_version, threshold=threshold)
    df = pd.read_csv(path)
    print("\n" + "=" * 70)
    print(dataset.upper())
    print("=" * 70)
    print("Total rows:", len(df))

    rows = []
    for _, row in df.iterrows():
        source_text = str(row.get("source_text") or row.get("text") or "")
        predicted_text = str(row.get("prediction") or row.get("text") or "")
        target_text = str(row.get("target_text") or row.get("text") or "")
        privacy_mask = row.get("privacy_mask")
        if isinstance(privacy_mask, str):
            try:
                import ast
                privacy_mask = ast.literal_eval(privacy_mask)
            except Exception:
                privacy_mask = []
        metrics = compute_pii_metrics(
            source_text=source_text,
            predicted_text=predicted_text,
            target_text=target_text,
            privacy_mask=privacy_mask or [],
        )
        rows.append({
            "sample_id": row.get("sample_id"),
            **metrics,
        })

    metrics_df = pd.DataFrame(rows)
    summary = metrics_df.mean(numeric_only=True)
    for name, value in summary.items():
        print(f"{name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    if "pii" in args.dataset.lower():
        evaluate_pii(args.dataset, provider=args.provider, prompt_version=args.prompt_version, threshold=args.threshold)
    else:
        evaluate(args.dataset, provider=args.provider, prompt_version=args.prompt_version, threshold=args.threshold)


if __name__ == "__main__":
    main()

