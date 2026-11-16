"""Run with python -m scripts.benchmark_pooling_pii."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from benchmark.pooling_pii import load_samples, run_benchmark


DATASETS = {
    "nano": ("pii-masking-nano-1k", "ca0a26dffb2c1f35947fa6d8d71b0d61e3373985"),
    "mini": ("pii-masking-mini-10k", "7b686c2e7475b02e10e38add1eb54a9f604f4361"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict PII span F1 and pooling HTTP latency/throughput.")
    parser.add_argument("--input", type=Path, help="Original AI4Privacy JSONL or prepared CSV with gold spans")
    parser.add_argument("--dataset", choices=DATASETS, default="nano")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/pooling_pii"))
    parser.add_argument("--dataset-base-url", default="https://huggingface.co",
                        help="Explicit public dataset host override; no automatic mirror fallback")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/pooling")
    parser.add_argument("--model", default="/models/current")
    parser.add_argument("--runtime", default="classic")
    parser.add_argument("--label-map", type=Path, help='JSON object, e.g. {"GIVENNAME":"person","SURNAME":"person"}')
    parser.add_argument("--schema", type=Path, help="JSON entity-name to description object; must cover all gold types")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are not overwritten")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.concurrency < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("--concurrency/--repeats must be >= 1; --warmup must be >= 0")
    if not 0 <= args.threshold <= 1 or not 0 < args.timeout < float("inf"):
        parser.error("--threshold must be in [0, 1]; --timeout must be positive and finite")
    output = args.output_dir or Path("outputs/pooling_pii") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists():
        parser.error("--output-dir already exists; choose a new directory")

    path = args.input
    if path is None:
        name, revision = DATASETS[args.dataset]
        path = args.cache_dir / name / revision / f"{args.split}.jsonl"
        if not path.exists():
            url = f"{args.dataset_base_url.rstrip('/')}/datasets/ai4privacy/{name}/resolve/{revision}/data/{args.split}.jsonl"
            print(f"Downloading pinned public dataset: {name} ({args.split})", flush=True)
            with httpx.Client(follow_redirects=True, timeout=180) as client:
                response = client.get(url)
                response.raise_for_status()
            # Validate before caching, so an invalid download cannot poison later runs.
            records = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            if not records or any(not isinstance(r, dict) or "privacy_mask" not in r for r in records):
                raise ValueError("Downloaded dataset does not contain privacy_mask annotations.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)

    mapping = json.loads(args.label_map.read_text(encoding="utf-8")) if args.label_map else {}
    if not isinstance(mapping, dict):
        parser.error("--label-map must contain a JSON object")
    samples, mapping = load_samples(path, mapping)
    # Derive a single schema from the entire dataset, never each sample's gold labels.
    schema = {label: "" for label in sorted(set(mapping.values()))}
    if args.schema:
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        if not isinstance(schema, dict):
            parser.error("--schema must contain a JSON object")
        if not set(mapping.values()) <= set(schema):
            parser.error("--schema must cover every mapped dataset category before applying --limit")
    if args.limit:
        samples = samples[:args.limit]
    print(f"Samples={len(samples)}, categories={len(schema)}, concurrency={args.concurrency}", flush=True)
    rows, summary = run_benchmark(
        samples, schema, endpoint=args.endpoint, model=args.model, runtime=args.runtime,
        threshold=args.threshold, concurrency=args.concurrency, warmup=args.warmup,
        repeats=args.repeats, timeout=args.timeout,
    )
    summary["dataset"] = {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "label_map": mapping,
    }
    if args.input is None:
        name, revision = DATASETS[args.dataset]
        summary["dataset"].update(
            repository=f"ai4privacy/{name}", revision=revision, split=args.split,
            dataset_base_url=args.dataset_base_url,
        )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved: {output}")
    if summary["failed_requests"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
