import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark.evaluation import evaluate_predictions

OUTPUT_DIR = Path("outputs")


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    valid = df[df["error"].fillna("") == ""].copy()
    y_true = valid["label"].astype(int)
    y_pred = valid["prediction"].astype(int)
    y_score = valid["score"].astype(float)
    latency = valid["latency_ms"].dropna() if "latency_ms" in valid.columns else pd.Series(dtype=float)

    metrics = {
        "file": str(path),
        "rows": len(df),
        "valid": len(valid),
        "error": len(df) - len(valid),
        **evaluate_predictions(y_true, y_pred, y_score),
        "mean_latency_ms": latency.mean() if not latency.empty else None,
        "p95_latency_ms": np.percentile(latency, 95) if not latency.empty else None,
    }
    return metrics


def iter_csv_files(provider: str | None = None) -> list[Path]:
    base = OUTPUT_DIR
    files: list[Path] = []
    if provider is None:
        for child in base.iterdir():
            if child.is_dir() and not (child / "comparison.json").exists():
                files.extend(sorted(child.glob("*.csv")))
    else:
        provider_dir = base / provider
        if provider_dir.exists() and not (provider_dir / "comparison.json").exists():
            files.extend(sorted(provider_dir.glob("*.csv")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all provider benchmark results across datasets, prompt versions, and thresholds.")
    parser.add_argument("--provider", default=None, help="Optional provider name, e.g. typesafe.ai")
    args = parser.parse_args()

    csv_files = iter_csv_files(args.provider)
    results = [summarize_file(path) for path in csv_files]

    summary = pd.DataFrame(results)
    if summary.empty:
        print("No CSV results found in outputs.")
        return

    cols = [
        "file",
        "rows",
        "valid",
        "error",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "mean_latency_ms",
        "p95_latency_ms",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
