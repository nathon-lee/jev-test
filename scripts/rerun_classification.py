"""Checkpointed Fastino/Laya Noul reruns; derive threshold metrics offline.

One live inference per source row and prompt. Threshold variants are not repeat
inferences. Existing result directories are left untouched.
"""

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from benchmark.evaluation import evaluate_predictions
from benchmark.provider_suite import (
    DATASET_SPECS, PROVIDER_SPECS, _extract_systemone_noul,
    normalize_provider_name, prompt_for_dataset, resolve_base_url,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_input(source, require_heldout=False):
    frame = pd.read_csv(source, dtype={"sample_id": str})
    provenance = {}
    if require_heldout:
        from benchmark.dataset_partitions import PROVENANCE_COLUMNS
        manifest_path = source.with_suffix(".manifest.json")
        if not manifest_path.exists() or not set(PROVENANCE_COLUMNS).issubset(frame.columns):
            raise ValueError("Held-out evaluation requires source provenance and a sidecar manifest")
        provenance = json.loads(manifest_path.read_text())
        if provenance["csv_sha256"] != digest(source) or provenance["prepared_rows"] != len(frame):
            raise ValueError("Held-out input digest/count does not match its manifest")
        if provenance["partition_role"] != "held_out" or provenance["source_split"] not in ("test", "validation", "valid", "dev"):
            raise ValueError("Training-reference input cannot be used for held-out evaluation")
        for name in PROVENANCE_COLUMNS:
            if name != "source_row" and not frame[name].eq(provenance[name]).all():
                raise ValueError("Input provenance disagrees with its manifest")
    if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Input requires nonempty unique sample IDs")
    if not frame["label"].isin([0, 1]).all() or frame["text"].isna().any():
        raise ValueError("Input requires text and binary labels")
    extra = [name for name in ("source_repository", "source_config", "source_revision", "source_split", "partition_role", "source_row") if name in frame]
    samples = [{"sample_id": str(row["sample_id"]), "text": str(row["text"]), "label": int(row["label"]),
                **{name: row[name] for name in extra}} for _, row in frame.iterrows()]
    return samples, provenance


def read_checkpoint(path, expected):
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sample = str(row["sample_id"])
        if sample not in expected:
            raise ValueError("Checkpoint contains an unknown sample ID")
        if not row["error"]:
            _extract_systemone_noul({"answers": {"score": {"noul": row["score"]}}}, "checkpoint")
            rows[sample] = row
    return rows


def infer(client, url, provider, model, prompt, sample, retries):
    started = time.perf_counter()
    result = {"sample_id": str(sample["sample_id"]), "score": None,
              "raw_confidence": None, "resolved_model": None, "error": "",
              "http_status": None, "attempts": 0, "observed_at": utc_now()}
    for attempt in range(retries + 1):
        result["attempts"] += 1
        try:
            response = client.post(url, json={
                "model": model, "state": {"user_input": sample["text"]},
                "questions": {"score": {"type": "noul", "instructions": prompt}},
            })
            result["http_status"] = response.status_code
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                delay = response.headers.get("retry-after", "")
                time.sleep(min(float(delay), 60) if delay.isdigit() else min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            payload = response.json()
            result["score"] = _extract_systemone_noul(payload, provider)
            result["resolved_model"] = payload.get("model")
            confidence = payload["answers"]["score"].get("confidence")
            if type(confidence) in (int, float) and math.isfinite(confidence):
                result["raw_confidence"] = confidence
            usage = payload.get("usage")
            result["usage"] = usage if isinstance(usage, dict) else None
            result["error"] = ""
            break
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            result["error"] = type(exc).__name__
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
                continue
        except Exception as exc:
            # Never store response bodies, request headers, or API keys.
            result["error"] = type(exc).__name__
        break
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result


def export_results(samples, cached, path, provider, dataset, version, model):
    rows = []
    for sample in samples:
        response = cached.get(str(sample["sample_id"]))
        if response is None:
            continue
        score = response["score"]
        rows.append({"dataset": dataset, **sample, **response,
                     "provider": provider, "request_model": model,
                     "prompt_version": version, "threshold": 0.5,
                     "prediction": int(score >= 0.5) if score is not None else None,
                     "score_field": "answers.score.noul"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    frame = frame.drop(columns=["usage"], errors="ignore")
    path = Path(path)
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)
    valid = frame[frame["error"].eq("") & frame["score"].notna()]
    threshold_metrics = {}
    for threshold in (0.5, 0.6, 0.7, 0.8):
        threshold_metrics[str(threshold)] = evaluate_predictions(
            valid["label"].astype(int), (valid["score"] >= threshold).astype(int), valid["score"].astype(float))
    return {"provider": provider, "dataset": dataset, "prompt_version": version,
            "expected_rows": len(samples), "rows": len(frame), "valid": len(valid),
            "errors": len(frame) - len(valid), "complete": len(frame) == len(samples),
            "resolved_models": sorted(set(valid["resolved_model"].dropna())),
            "threshold_metrics": threshold_metrics,
            "latency_mean_ms": float(valid["latency_ms"].mean()) if len(valid) else None,
            "latency_p50_ms": float(valid["latency_ms"].median()) if len(valid) else None,
            "latency_p95_ms": float(valid["latency_ms"].quantile(0.95)) if len(valid) else None,
            "pii_detection_only": dataset.startswith("pii_"),
            "file_sha256": digest(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=["fastino.ai", "laya-ai.com", "typesafe.ai"])
    parser.add_argument("--datasets", default="jailbreak-classification,hatecheck,aegis2,toxigen,pii_masking_nano_1k,pii_masking_mini_10k,detect-jailbreak,jailbreakhub")
    parser.add_argument("--prompts", default="v1,v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--require-heldout", action="store_true", help="Reject unversioned/training-reference inputs before sending requests")
    args = parser.parse_args(argv)
    if args.env_file:
        load_dotenv(args.env_file)
    if args.workers < 1 or args.retries < 0:
        parser.error("Workers must be positive and retries nonnegative")
    provider = normalize_provider_name(args.provider)
    spec = PROVIDER_SPECS[provider]
    key = os.getenv(spec["api_key_env"])
    if not key:
        parser.error(f"Missing {spec['api_key_env']}")
    url = resolve_base_url(provider).rstrip("/")
    if provider == "typesafe.ai" and not url.endswith(("/v1", "/v1/systemone")):
        url += "/v1"
    if not url.endswith("/systemone"):
        url += "/systemone"
    output_dir = args.output_dir / provider
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [(dataset, version) for dataset in args.datasets.split(",") for version in args.prompts.split(",")]
    # Validate every selected partition before making any billable requests.
    inputs = {dataset: load_input(args.data_dir / DATASET_SPECS[dataset]["path"], args.require_heldout)
              for dataset, _ in configs}
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=90,
                      limits=httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)) as client:
        for dataset, version in configs:
            source = args.data_dir / DATASET_SPECS[dataset]["path"]
            samples, provenance = inputs[dataset]
            prompt = prompt_for_dataset(dataset, version)
            prefix = output_dir / f"{dataset}_{version}"
            manifest_path = prefix.with_suffix(".manifest.json")
            checkpoint = prefix.with_suffix(".checkpoint.jsonl")
            identity = {"provider": provider, "model": spec["model"], "dataset": dataset,
                        "prompt_version": version, "prompt": prompt, "input_sha256": digest(source),
                        "submitted_input_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
                        "pandas_version": pd.__version__,
                        "expected_rows": len(samples), "score_field": "answers.score.noul",
                        "thresholds_derived_offline": True, "repeat_inferences": False,
                        "workers": args.workers, "endpoint": url}
            if args.require_heldout:
                identity.update(evaluation_partition=provenance, require_heldout=True)
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text())
                if any(previous.get(k) != value for k, value in identity.items() if k != "workers"):
                    raise ValueError("Checkpoint identity differs; use a new output directory")
                manifest = previous
            else:
                manifest = {**identity, "started_at": utc_now()}
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            expected = {row["sample_id"] for row in samples}
            cached = read_checkpoint(checkpoint, expected)
            pending = iter(row for row in samples if row["sample_id"] not in cached)
            print(f"START {provider} {dataset} {version} rows={len(samples)} resumed={len(cached)}", flush=True)
            last_update = time.monotonic()
            session_start = time.monotonic()
            completed = failed = 0
            with checkpoint.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
                active = {}
                def fill():
                    while len(active) < args.workers * 2:
                        sample = next(pending, None)
                        if sample is None:
                            break
                        future = pool.submit(infer, client, url, provider, spec["model"], prompt, sample, args.retries)
                        active[future] = sample
                fill()
                while active:
                    done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                    for future in done:
                        active.pop(future)
                        row = future.result()
                        cached[row["sample_id"]] = row
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        completed += 1
                        failed += bool(row["error"])
                        if row["http_status"] in (401, 402, 403):
                            stream.flush()
                            raise RuntimeError(f"Access blocked: {provider} HTTP {row['http_status']}")
                        if completed >= 25 and failed == completed:
                            stream.flush()
                            raise RuntimeError(f"All initial {completed} requests failed for {provider}; check the endpoint before resuming")
                    fill()
                    if time.monotonic() - last_update >= 10:
                        stream.flush()
                        print(f"PROGRESS {provider} {dataset} {version} rows={len(cached)}/{len(samples)} errors={failed} rate={completed/max(time.monotonic()-session_start,1):.1f}/s", flush=True)
                        last_update = time.monotonic()
            summary = export_results(samples, cached, prefix.with_suffix(".csv"), provider, dataset, version, spec["model"])
            manifest.update(finished_at=utc_now(), result=summary)
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print(f"SAVED {provider} {dataset} {version} valid={summary['valid']} errors={summary['errors']}", flush=True)


if __name__ == "__main__":
    main()
