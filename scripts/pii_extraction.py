"""Prepare, run, and evaluate PII extraction independently of classification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from benchmark.pii_extraction import (
    DEFAULT_EXTRACTION_MODEL, PII_LABELS, call_fastino_extraction,
    entity_dicts, load_samples, score_extraction, validate_sample, write_jsonl,
)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return value


def prepare_records(records, *, dataset, split, revision):
    return [validate_sample({**record, "dataset": dataset, "split": split,
                             "revision": revision}) for record in records]


def run_samples(samples, *, model, threshold, api_key, base_url):
    """Retain references for offline evaluation, but send only text and fixed labels."""
    rows = []
    for sample in samples:
        started = perf_counter()
        row = {**sample, "task": "pii_extraction", "provider": "fastino.ai",
               "request_model": model, "threshold": threshold,
               "predicted_entities": None, "error": ""}
        try:
            entities, resolved_model = call_fastino_extraction(
                sample["source_text"], model=model, threshold=threshold,
                api_key=api_key, base_url=base_url,
            )
            row.update(predicted_entities=entity_dicts(entities), resolved_model=resolved_model)
        except Exception as exc:
            # Do not persist response bodies, URLs, or credentials in error logs.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            row["error"] = type(exc).__name__ + (f" HTTP {status}" if status else "")
        row["elapsed_ms"] = (perf_counter() - started) * 1000
        rows.append(row)
    return rows


def save_summary(path, rows, **metadata):
    summary = {"task": "pii_extraction", **metadata, **score_extraction(rows)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Download one explicit partition with intact gold spans")
    prepare.add_argument("--dataset", choices=["ai4privacy/pii-masking-nano-1k", "ai4privacy/pii-masking-mini-10k"], required=True)
    prepare.add_argument("--split", default="validation", choices=["train", "validation"])
    prepare.add_argument("--revision", default="main", help="HF revision; resolved to a commit SHA before download")
    prepare.add_argument("--limit", type=positive_int)
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="Extract spans with Fastino GLiNER (not GLiDE)")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="JSONL predictions; use an extraction-only directory")
    run.add_argument("--summary", type=Path, required=True)
    run.add_argument("--model", default=DEFAULT_EXTRACTION_MODEL)
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--base-url", default=os.getenv("FASTINO_BASE_URL", "https://api.fastino.ai/v1"))
    run.add_argument("--limit", type=positive_int)

    evaluate = subparsers.add_parser("evaluate", help="Score canonical span predictions offline")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command != "prepare":
        paths = [args.input.resolve(), args.summary.resolve()]
        if args.command == "run":
            paths.append(args.output.resolve())
        if len(set(paths)) != len(paths):
            parser.error("Input, predictions, and summary paths must be different")

    if args.command == "prepare":
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
        records = load_dataset(args.dataset, split=args.split, revision=revision)
        if args.limit:
            records = records.select(range(min(args.limit, len(records))))
        rows = prepare_records(records, dataset=args.dataset, split=args.split, revision=revision)
        if not rows:
            parser.error("Dataset partition is empty")
        write_jsonl(args.output, rows)
        print(f"Saved {len(rows)} samples from {args.dataset}@{revision} split={args.split} to {args.output}")
        return

    samples = load_samples(args.input)
    metadata = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "partitions": sorted({(str(row.get("dataset", "unknown")), str(row.get("split", "unknown")),
                               str(row.get("revision", "unknown"))) for row in samples}),
    }
    if args.command == "run":
        if not 0 <= args.threshold <= 1:
            parser.error("--threshold must be in [0, 1]")
        if args.model.lower() == "fastino/glide":
            parser.error("GLiDE does not return extraction spans; choose a GLiNER model")
        api_key = os.getenv("FASTINO_API_KEY")
        if not api_key:
            parser.error("Set FASTINO_API_KEY; no hosted requests were sent")
        if args.limit:
            samples = samples[:args.limit]
        metadata["partitions"] = sorted({
            (str(row.get("dataset", "unknown")), str(row.get("split", "unknown")),
             str(row.get("revision", "unknown"))) for row in samples
        })
        rows = run_samples(samples, model=args.model, threshold=args.threshold,
                           api_key=api_key, base_url=args.base_url)
        write_jsonl(args.output, rows)
        metadata.update(provider="fastino.ai", request_model=args.model,
                        resolved_models=sorted({row["resolved_model"] for row in rows if not row["error"]}),
                        threshold=args.threshold, entity_schema=PII_LABELS,
                        limit=args.limit)
    else:
        rows = samples
        metadata["model_configurations"] = [json.loads(value) for value in sorted({
            json.dumps({key: row.get(key) for key in
                        ("provider", "request_model", "resolved_model", "threshold")}, sort_keys=True)
            for row in rows
        })]
    summary = save_summary(args.summary, rows, **metadata)
    if summary["failed_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
