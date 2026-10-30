"""Checkpointed held-out GLiNER extraction; never call this a GLiDE result."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from benchmark.pii_extraction import DEFAULT_EXTRACTION_MODEL, PII_LABELS, load_samples
from scripts.pii_extraction import run_samples, save_summary
from scripts.rerun_classification import digest, utc_now


def successful_checkpoint(path, expected):
    cached = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["sample_id"] not in expected:
                raise ValueError("Unknown checkpoint sample ID")
            if not row["error"]:
                cached[row["sample_id"]] = row
    return cached


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_EXTRACTION_MODEL)
    args = parser.parse_args(argv)
    if args.env_file:
        load_dotenv(args.env_file)
    key = os.getenv("FASTINO_API_KEY")
    if not key or args.workers < 1:
        parser.error("FASTINO_API_KEY and positive workers are required")
    samples = load_samples(args.input)
    if not samples or any(row.get("partition_role") != "held_out" or row.get("split") not in ("test", "validation", "valid", "dev") for row in samples):
        parser.error("Only explicit nonempty held-out samples are permitted")
    ids = {row["sample_id"] for row in samples}
    if len(ids) != len(samples):
        parser.error("Duplicate sample IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    checkpoint = args.output.with_suffix(".checkpoint.jsonl")
    base_url = os.getenv("FASTINO_BASE_URL", "https://api.fastino.ai/v1")
    identity = {"input_sha256": digest(args.input), "model": args.model, "threshold": .5,
                "base_url": base_url, "entity_schema": PII_LABELS, "expected_rows": len(samples),
                "partitions": sorted({(r["dataset"], r["split"], r["revision"]) for r in samples})}
    # JSON normalization keeps tuple/list identity comparisons stable on resume.
    identity = json.loads(json.dumps(identity))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("Extraction checkpoint identity differs")
    else:
        manifest = {**identity, "started_at": utc_now(), "workers": args.workers}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cached = successful_checkpoint(checkpoint, ids)
    pending = [r for r in samples if r["sample_id"] not in cached]
    print(f"START GLiNER expected={len(samples)} resumed={len(cached)}", flush=True)
    def extract(sample):
        return run_samples([sample], model=args.model, threshold=.5, api_key=key, base_url=base_url)[0]
    completed = failed = 0
    with checkpoint.open("a") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Only a bounded batch is submitted, so access denial cannot schedule an
        # entire dataset of doomed calls before the runner can stop.
        for start in range(0, len(pending), args.workers):
            for future in as_completed([pool.submit(extract, r) for r in pending[start:start + args.workers]]):
                row = future.result()
                cached[row["sample_id"]] = row
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                completed += 1
                failed += bool(row["error"])
                if any(f"HTTP {status}" in row["error"] for status in (401, 402, 403, 404)):
                    raise RuntimeError(f"Extraction access blocked: {row['error']}")
                if completed >= 16 and failed == completed:
                    raise RuntimeError("All initial extraction calls failed; check provider support")
            print(f"PROGRESS GLiNER rows={len(cached)}/{len(samples)} errors={failed}", flush=True)
    rows = [cached[r["sample_id"]] for r in samples]
    args.output.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    manifest.update(finished_at=utc_now(), output_sha256=digest(args.output))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    save_summary(args.output.with_suffix(".summary.json"), rows, **identity, provider="fastino.ai")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
