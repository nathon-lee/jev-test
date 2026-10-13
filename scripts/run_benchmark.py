from pathlib import Path
import argparse
import time
import pandas as pd
from benchmark.jev_client import run_jev
import httpx
from typesafe_sdk import TypeSafeClient
import time

DATA_DIR = Path("data/prepared")
OUTPUT_DIR = Path("outputs")
def load_existing(output_path: Path):
    if not output_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(output_path)
    if df["sample_id"].duplicated().any():
        raise ValueError(
            f"Duplicate sample_id detected in {output_path}. "
            "Please inspect the existing results before resuming. "
        )
    print(
        f"Found existing results: {len(df)}"
    )
    return df

def run_with_retry(
    client,
    dataset,
    text,
    prompt_version="v1",
    max_attempts=3,
):
    last_error = None
    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            return run_jev(
                client=client,
                dataset=dataset,
                text=text,
                prompt_version=prompt_version,
            )
        except Exception as e:
            last_error = e
            print(
                f"Attempt {attempt}/"
                f"{max_attempts} failed: "
                f"{repr(e)}"
            )
            if attempt < max_attempts:
                time.sleep(1)
    raise last_error


def run_dataset(
    client,
    dataset: str,
    limit: int | None = None,
    prompt_version: str = "v1",
):
    if dataset in {"benign-behaviors", "harmful-behaviors", "judge-comparison"}:
        input_path = DATA_DIR / f"{dataset}.csv"
    elif dataset == "jailbreak":
        input_path = DATA_DIR / "jailbreak_all.csv"
    else:
        input_path = DATA_DIR / f"{dataset}.csv"
    output_path = (
        OUTPUT_DIR /
        f"{dataset}_{prompt_version}_predictions.csv"
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    df = pd.read_csv(input_path)
    if limit is not None:
        df = df.head(limit)
    print("\n" + "=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Samples: {len(df)}")
    print("=" * 80)
    # --------------------------------------------------
    # Resume support
    # --------------------------------------------------
    existing = load_existing(output_path)
    completed_ids = set()
    if not existing.empty:
        successful = existing[
                existing["error"]
                .fillna("")
                .eq("")
            ]
        completed_ids = set(
            successful["sample_id"].tolist()
        )
    rows = []
    start_all = time.perf_counter()
    for index, row in df.iterrows():
        sample_id = row["sample_id"]
        if sample_id in completed_ids:
            continue
        try:
            result = run_with_retry(
                client=client,
                dataset=dataset,
                text=row["text"],
                prompt_version=prompt_version,
            )
            record = {
                "dataset": dataset,
                "sample_id": sample_id,
                "text": row["text"],
                "label": int(row["label"]),
                "prompt_version": prompt_version,
                "score": result["score"],
                "prediction": result["prediction"],
                "latency_ms": result["latency_ms"],
                "model": result["model"],
                "error": "",
            }
            completed_ids.add(sample_id)
        except Exception as e:
            record = {
                "dataset": dataset,
                "sample_id": sample_id,
                "text": row["text"],
                "label": int(row["label"]),
                "prompt_version": prompt_version,
                "score": None,
                "prediction": None,
                "latency_ms": None,
                "model": None,
                "error": repr(e),
            }
        rows.append(record)
        # --------------------------------------------------
        # Progress
        # --------------------------------------------------
        processed = len(completed_ids)
        if (
            processed % 50 == 0
            or processed == len(df)
        ):
            elapsed = (
                time.perf_counter()
                - start_all
            )
            print(
                f"{dataset}: "
                f"{processed}/{len(df)} "
                f"({processed / len(df):.1%}) "
                f"elapsed={elapsed:.1f}s"
            )
        # --------------------------------------------------
        # Checkpoint every 100 requests
        # --------------------------------------------------
        if len(rows) >= 100:
            append_results(
                rows,
                output_path,
            )
            rows = []
    # final checkpoint
    if rows:
        append_results(
            rows,
            output_path,
        )
    print(
        f"\nFinished {dataset}"
    )
    print(
        f"Output: {output_path}"
    )
def append_results(
    rows,
    output_path,
):
    batch = pd.DataFrame(rows)
    header = not output_path.exists()
    batch.to_csv(
        output_path,
        mode="a",
        header=header,
        index=False,
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "toxigen",
            "hatecheck",
            "aegis2",
            "benign-behaviors",
            "harmful-behaviors",
            "judge-comparison",
            "jailbreak",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v1",
        help="Jev question version",
    )
    args = parser.parse_args()
    # ========================================================
    # KEEP YOUR WORKING JEV INITIALIZATION HERE
    # ========================================================
    http_client = httpx.Client(
        verify=False
    )

    client = TypeSafeClient(
        http_client=http_client,
        timeout=30.0
    )
    # ========================================================
    if args.dataset == "all":
        datasets = [
            "toxigen",
            "hatecheck",
            "aegis2",
            "benign-behaviors",
            "harmful-behaviors",
            "judge-comparison",
            "jailbreak",
        ]
    else:
        datasets = [
            args.dataset
        ]
    for dataset in datasets:
        run_dataset(
            client=client,
            dataset=dataset,
            limit=args.limit,
            prompt_version=args.prompt_version,
        )
if __name__ == "__main__":
    main()
