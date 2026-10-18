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


def _provider_dirs(provider: str | None = None) -> list[Path]:
    dirs: list[Path] = []

    if provider:
        dirs.append(OUTPUT_DIR / provider)

    if OUTPUT_DIR.exists():
        children = sorted(
            [p for p in OUTPUT_DIR.iterdir() if p.is_dir()],
            key=lambda p: (p.name != "typesafe.ai", p.name.lower()),
        )
        if children:
            dirs.extend(children)

    dirs.append(OUTPUT_DIR)

    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _iter_candidate_paths(dataset: str, provider: str | None = None, prompt_version: str | None = None, threshold: float | None = None):
    dataset_stem = _normalize_dataset_name(dataset)
    candidates = []

    for base_dir in _provider_dirs(provider):
        if prompt_version:
            candidates.extend([
                base_dir / f"{dataset}_{prompt_version}.csv",
                base_dir / f"{dataset}_{prompt_version}_predictions.csv",
                base_dir / f"{dataset_stem}_{prompt_version}.csv",
                base_dir / f"{dataset_stem}_{prompt_version}_predictions.csv",
            ])

        if prompt_version and threshold is not None:
            threshold_variant = f"{prompt_version}_t{float(threshold)}"
            threshold_variant_1 = f"{prompt_version}_t{float(threshold):.1f}"
            candidates.extend([
                base_dir / f"{dataset}_{threshold_variant}.csv",
                base_dir / f"{dataset}_{threshold_variant_1}.csv",
                base_dir / f"{dataset_stem}_{threshold_variant}.csv",
                base_dir / f"{dataset_stem}_{threshold_variant_1}.csv",
            ])

        if threshold is not None:
            candidates.extend([
                base_dir / f"{dataset}_t{float(threshold)}.csv",
                base_dir / f"{dataset}_t{float(threshold):.1f}.csv",
                base_dir / f"{dataset_stem}_t{float(threshold)}.csv",
                base_dir / f"{dataset_stem}_t{float(threshold):.1f}.csv",
            ])

        candidates.extend([
            base_dir / f"{dataset}.csv",
            base_dir / f"{dataset}_predictions.csv",
            base_dir / f"{dataset_stem}_predictions.csv",
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

        raise FileNotFoundError(
            f"No threshold-specific prediction file found for dataset '{dataset}' "
            f"with prompt_version={prompt_version} and threshold={threshold}. "
            f"Expected files like '{dataset}_{prompt_version}_t{float(threshold)}.csv'."
        )

    for path in _iter_candidate_paths(dataset_name, provider=provider, prompt_version=prompt_version, threshold=threshold):
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No prediction file found for dataset '{dataset}'"
        f"{f' provider={provider}' if provider else ''}"
        f"{f' prompt_version={prompt_version}' if prompt_version else ''}"
        f"{f' threshold={threshold}' if threshold is not None else ''}."
    )


def evaluate(dataset: str, provider: str | None = None, prompt_version: str | None = None, threshold: float | None = None, debug: bool = False):
    candidate_paths = list(_iter_candidate_paths(dataset, provider=provider, prompt_version=prompt_version, threshold=threshold))
    if debug:
        print(f"DEBUG dataset={dataset} provider={provider} prompt_version={prompt_version} threshold={threshold}")
        print("DEBUG candidate paths:")
        for p in candidate_paths:
            print(f"  - {p} exists={p.exists()}")
    path = resolve_prediction_path(dataset, provider=provider, prompt_version=prompt_version, threshold=threshold)
    if debug:
        print(f"DEBUG selected path: {path}")
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
    latency = valid["latency_ms"].dropna() if "latency_ms" in valid.columns else pd.Series(dtype=float)
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
    if latency.empty:
        print("Latency unavailable: 'latency_ms' column not present in this CSV.")
    else:
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
def _classification_summary_from_frame(df: pd.DataFrame) -> dict[str, float]:
    if "error" in df.columns:
        valid = df[df["error"].fillna("") == ""].copy()
    else:
        valid = df.copy()
    if valid.empty:
        return {}

    y_true = valid["label"].astype(int) if "label" in valid.columns else pd.Series([1] * len(valid), index=valid.index)
    y_pred = valid["prediction"].astype(int) if "prediction" in valid.columns else y_true.copy()
    y_score = valid["score"].astype(float) if "score" in valid.columns else y_pred.astype(float)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }


def evaluate_pii(dataset: str, provider: str | None = None, prompt_version: str | None = None, threshold: float | None = None, debug: bool = False):
    candidate_paths = list(_iter_candidate_paths(dataset, provider=provider, prompt_version=prompt_version, threshold=threshold))
    if debug:
        print(f"DEBUG dataset={dataset} provider={provider} prompt_version={prompt_version} threshold={threshold}")
        print("DEBUG candidate paths:")
        for p in candidate_paths:
            print(f"  - {p} exists={p.exists()}")
    path = resolve_prediction_path(dataset, provider=provider, prompt_version=prompt_version, threshold=threshold)
    if debug:
        print(f"DEBUG selected path: {path}")
    df = pd.read_csv(path)
    print("\n" + "=" * 70)
    print(dataset.upper())
    print("=" * 70)
    print("Total rows:", len(df))

    standard_metrics = _classification_summary_from_frame(df)
    if standard_metrics:
        print("\nClassification metrics")
        print("-------------------------")
        for name, value in standard_metrics.items():
            print(f"{name}: {value:.4f}")

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
    summary = metrics_df.drop(columns=["sample_id"], errors="ignore").mean(numeric_only=True)
    primary_order = [
        "pii_leakage_rate",
        "masking_success_rate",
        "span_precision",
        "span_recall",
        "span_f1",
        "over_masking_rate",
        "text_preservation",
    ]
    print("\nPII privacy metrics")
    print("-------------------------")
    for name in primary_order:
        if name in summary:
            print(f"{name}: {summary[name]:.4f}")
    for name, value in summary.items():
        if name not in primary_order:
            print(f"{name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--debug-path", action="store_true", help="Print the exact CSV candidate list and selected file before evaluation.")
    args = parser.parse_args()

    if "pii" in args.dataset.lower():
        evaluate_pii(args.dataset, provider=args.provider, prompt_version=args.prompt_version, threshold=args.threshold, debug=args.debug_path)
    else:
        evaluate(args.dataset, provider=args.provider, prompt_version=args.prompt_version, threshold=args.threshold, debug=args.debug_path)


if __name__ == "__main__":
    main()

