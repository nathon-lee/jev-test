#!/usr/bin/env python3
"""Benchmark the original Galileo PI/PII presets through Agent Control.

Run from this folder: python benchmark.py --env-file .env
Use --limit 3 for a connectivity test before the full benchmark.
No local project imports, custom metrics, model uploads, or prompt changes.
"""

import argparse
import base64
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import ssl
import threading
import time
import uuid

from dotenv import load_dotenv
import httpx
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
import truststore


PRESETS = {"pi": "prompt_injection_luna", "pii": "input_pii"}
PREPARED_TASKS = {
    "toxigen": "toxicity", "hatecheck": "hate", "aegis2": "safety",
    "detect_jailbreak": "pi", "jailbreak_classification": "pi", "jailbreakhub": "pi",
}
PI_REVISION = "5f68a89245ce067ba4258c92198633dbebd02dac"
PII_REVISION = "7b686c2e7475b02e10e38add1eb54a9f604f4361"
PII_NANO_REVISION = "ca0a26dffb2c1f35947fa6d8d71b0d61e3373985"
PII_MAP = {"CREDITCARDNUMBER": "credit_card_info", "EMAIL": "email",
           "TELEPHONENUM": "phone_number", "SOCIALNUM": "ssn"}
PII_CLASSES = sorted(PII_MAP.values())
PII_EXPOSED = {"account_info", "address", "credit_card_info", "date_of_birth",
               "email", "name", "network_info", "password", "phone_number",
               "ssn", "username"}
RETRYABLE = {408, 429, 500, 502, 503, 504}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


class API:
    def __init__(self, url, key):
        self.url = url.rstrip("/")
        self.key = key
        self.client = httpx.Client(
            headers={"Galileo-API-Key": key}, timeout=90,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        )

    def close(self):
        self.client.close()

    def request(self, method, path, body=None, headers=None):
        for attempt in range(4):
            try:
                r = self.client.request(method, self.url + path, json=body, headers=headers)
                if r.status_code in RETRYABLE and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                if r.is_error:
                    # Avoid echoing headers, tokens, or potentially sensitive response bodies.
                    raise RuntimeError(f"{method} {path}: HTTP {r.status_code}")
                return r.json()
            except httpx.TransportError:
                if attempt == 3:
                    raise RuntimeError(f"{method} {path}: transport failure") from None
                time.sleep(2 ** attempt)


def download(client, cache, repo, revision, filename):
    path = cache / repo.replace("/", "__") / revision / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        r = client.get(url)
        r.raise_for_status()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(r.content)
        temporary.replace(path)
    return path


def load_sources(cache, pi_size, seed, metric, pii_dataset="mini", pii_split="validation"):
    """Download pinned public data; never send the Galileo key to Hugging Face."""
    sources, provenance = {}, {}
    with httpx.Client(follow_redirects=True, timeout=180,
                      verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)) as client:
        if metric in ("both", "pi"):
            path = download(client, cache, "GuardrailsAI/detect-jailbreak",
                            PI_REVISION, "dataset.parquet")
            data = pq.read_table(io.BytesIO(path.read_bytes())).to_pylist()
            if not {"prompt", "is_jailbreak"} <= set(data[0]):
                raise ValueError("PI dataset schema changed")
            if not 1 <= pi_size <= len(data):
                raise ValueError(f"--pi-size must be between 1 and {len(data)}")
            positive = [i for i, row in enumerate(data) if row["is_jailbreak"]]
            negative = [i for i, row in enumerate(data) if not row["is_jailbreak"]]
            n_positive = round(pi_size * len(positive) / len(data))
            rng = random.Random(seed)
            indices = sorted(rng.sample(positive, n_positive) +
                             rng.sample(negative, pi_size - n_positive))
            sources["pi"] = [{"index": j, "source_index": i, "input": data[i]["prompt"],
                              "gold": int(data[i]["is_jailbreak"])}
                             for j, i in enumerate(indices)]
            provenance["pi"] = {"dataset": "GuardrailsAI/detect-jailbreak",
                                "revision": PI_REVISION, "split": "train", "seed": seed,
                                "sample_size": pi_size, "source_indices": indices,
                                "sampling": "proportional stratified, without replacement",
                                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if metric in ("both", "pii"):
            if pii_dataset not in ("mini", "nano") or pii_split not in ("train", "validation", "all"):
                raise ValueError("Unsupported PII dataset or split")
            repo = f"ai4privacy/pii-masking-{'nano-1k' if pii_dataset == 'nano' else 'mini-10k'}"
            revision = PII_NANO_REVISION if pii_dataset == "nano" else PII_REVISION
            splits = ("train", "validation") if pii_split == "all" else (pii_split,)
            data, hashes = [], {}
            for split in splits:
                path = download(client, cache, repo, revision, f"data/{split}.jsonl")
                with path.open(encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle if line.strip()]
                if not records or any(
                        not {"source_text", "privacy_mask", "language", "uid"} <= set(row)
                        for row in records):
                    raise ValueError(f"PII dataset schema changed: {repo}/{split}")
                data.extend(records)
                hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            sources["pii"] = []
            for i, row in enumerate(data):
                labels = {span["label"] for span in row["privacy_mask"]}
                sources["pii"].append({
                    "index": i, "source_index": row["uid"], "input": row["source_text"],
                    "gold": sorted({PII_MAP[x] for x in labels if x in PII_MAP}),
                    "language": row["language"], "dataset_labels": sorted(labels),
                })
            provenance["pii"] = {"dataset": repo, "revision": revision, "split": pii_split,
                                 "mapping": PII_MAP, "file_sha256":
                                 hashes[pii_split] if pii_split != "all" else hashes}
    return sources, provenance


def load_prepared(directory, datasets, presets, limit=None):
    selected = list(PREPARED_TASKS) if datasets == "all" else [
        name.strip().replace("-", "_") for name in datasets.split(",")]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("--datasets must contain unique dataset names")
    sources, provenance, mapping = {}, {}, {}
    for dataset in selected:
        if dataset not in PREPARED_TASKS:
            raise ValueError(f"Unsupported prepared dataset: {dataset}. "
                             "Local PII CSVs are excluded; original PII validation is retained.")
        task = PREPARED_TASKS[dataset]
        name = presets.get(task)
        if not name or name.startswith("YOUR_"):
            raise ValueError(f"{dataset} requires --{task}-preset or "
                             f"GALILEO_{task.upper()}_PRESET. No substitute preset is used.")
        path = directory / f"{dataset}.csv"
        rows, ids = [], set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"sample_id", "text", "label"} <= set(reader.fieldnames or []):
                raise ValueError(f"{path}: required columns are sample_id,text,label")
            for index, row in enumerate(reader):
                if row["label"] not in ("0", "1") or not row["sample_id"] or row["text"] is None:
                    raise ValueError(f"{path}: invalid sample at CSV record {index + 2}")
                if row["sample_id"] in ids:
                    raise ValueError(f"{path}: duplicate sample_id {row['sample_id']}")
                ids.add(row["sample_id"])
                rows.append({"index": index, "source_index": row["sample_id"],
                             "sample_id": row["sample_id"], "dataset": dataset,
                             "input": row["text"], "gold": int(row["label"])})
        if not rows:
            raise ValueError(f"{path}: dataset is empty")
        sources[dataset] = rows[:limit] if limit is not None else rows
        provenance[dataset] = {"file": str(path.resolve()), "task": task,
                               "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "total_rows": len(rows), "selected_rows": len(sources[dataset])}
        mapping[dataset] = name
    return sources, provenance, mapping


def setup(api, project_id, sources, provenance, output, presets=None, dataset_workers=1):
    presets = PRESETS if presets is None else presets
    project = api.request("GET", f"/projects/{project_id}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    state = {"api_url": api.url, "project_id": project_id,
             "project_name": project["name"], "run_id": run_id,
             "source_fingerprint": digest(sources), "datasets": provenance, "metrics": {},
             "dataset_subdirectories": dataset_workers > 1}
    for metric in sources:
        name = presets[metric]
        catalog = api.request("POST", "/v2/scorers/list", {
            "filters": [{"name": "name", "operator": "eq", "value": name}]})
        found = [x for x in catalog["scorers"] if x["name"] == name and x["scorer_type"] == "preset"]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one original preset named {name}")
        scorer = found[0]
        if scorer["model_type"] != "slm" or not scorer.get("default_version_id"):
            raise ValueError(f"{name} is not a versioned SLM preset")
        agent = f"public-benchmark-{metric}-{run_id}"
        stream = api.request("POST", f"/projects/{project_id}/log_streams", {"name": agent})
        config = {"scorer_id": scorer["id"], "scorer_version_id": scorer["default_version_id"],
                  "scorer_label": name, "operator": "any" if metric == "pii" else "gte",
                  "timeout_ms": 60000}
        if metric != "pii":
            config["threshold"] = 0.5
        control = api.request("PUT", "/agent-control/api/v1/controls", {
            "name": agent, "data": {"enabled": True, "execution": "server",
                "description": "Public dataset benchmark using the unchanged original preset.",
                "scope": {"step_types": ["trace"], "step_names": [name], "stages": ["post"]},
                "condition": {"selector": {"path": "*"},
                              "evaluator": {"name": "galileo.luna", "config": config}},
                "action": {"decision": "observe"}}})
        cid = control.get("control_id") or control["id"]
        api.request("PUT", "/agent-control/api/v1/control-bindings/by-key", {
            "target_type": "log_stream", "target_id": stream["id"], "control_id": cid, "enabled": True})
        registered = api.request("POST", "/agent-control/api/v1/agents/initAgent", {
            "agent": {"agent_name": agent}, "steps": [{"type": "trace", "name": name}],
            "conflict_mode": "overwrite", "target_type": "log_stream", "target_id": stream["id"]})
        if not any((c.get("control_id") or c.get("id")) == cid for c in registered["controls"]):
            raise RuntimeError("Agent initialization did not attach the requested control")
        state["metrics"][metric] = {
            "scorer_name": name, "scorer_id": scorer["id"],
            "scorer_version_id": scorer["default_version_id"], "stream_id": stream["id"],
            "agent_name": agent, "control_id": cid,
        }
        print(f"Configured {name}, version {scorer['default_version_id']}", flush=True)
    save(output / "manifest.json", state)
    return state


class Evaluator:
    def __init__(self, api, info, metric):
        self.api, self.info, self.metric = api, info, metric
        self.lock = threading.Lock()
        self.token, self.expires = None, 0

    def bearer(self, refresh=False):
        with self.lock:
            if refresh or self.token is None or time.time() + 30 >= self.expires:
                result = self.api.request("POST", "/agent-control/api/v1/auth/runtime-token-exchange", {
                    "target_type": "log_stream", "target_id": self.info["stream_id"]})
                self.token = result["token"]
                try:
                    payload = self.token.split(".")[1]
                    self.expires = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]
                except (ValueError, KeyError, IndexError):
                    self.expires = time.time() + 300
            return self.token

    def evaluate(self, row):
        info = self.info
        trace_id = uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response = None
            for attempt in range(2):
                try:
                    response = self.api.request("POST", "/agent-control/api/v1/evaluation", {
                        "agent_name": info["agent_name"], "stage": "post",
                        "target_type": "log_stream", "target_id": info["stream_id"],
                        "step": {"type": "trace", "name": info["scorer_name"],
                                 "input": row["input"], "output": ""}}, headers={
                        "Authorization": "Bearer " + self.bearer(refresh=bool(attempt)),
                        "X-Trace-Id": trace_id, "X-Span-Id": uuid.uuid4().hex[:16]})
                    break
                except RuntimeError as exc:
                    if "HTTP 401" not in str(exc) or attempt:
                        raise
            executions = [x for bucket in ("matches", "non_matches", "errors")
                          for x in response.get(bucket) or []
                          if x.get("result", {}).get("metadata", {}).get("scorer_id") == info["scorer_id"]]
            if len(executions) != 1:
                raise ValueError("Expected exactly one requested scorer execution")
            execution = executions[0]
            result = execution["result"]
            meta = result["metadata"]
            if str(execution["control_id"]) != str(info["control_id"]) or meta.get(
                    "requested_scorer_version_id") != info["scorer_version_id"]:
                raise ValueError("Control or scorer version mismatch")
            if result.get("error") or meta.get("error_message") or meta.get("status") != "success":
                raise ValueError("Scorer execution failed: " + str(meta.get("error_message") or result.get("error") or meta.get("status")))
            score = meta.get("score")
            if self.metric != "pii":
                if isinstance(score, list) and len(score) == 1:
                    score = score[0]
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Expected a positive-class probability in [0, 1]")
                prediction = int(score >= 0.5)
            else:
                if not isinstance(score, list) or any(not isinstance(x, str) or x not in PII_EXPOSED for x in score):
                    raise ValueError("Expected the original preset's exposed PII class list")
                prediction = sorted(set(score) & set(PII_CLASSES))
            return {**row, "score": score, "prediction": prediction, "error": None,
                    "scorer_version_id": info["scorer_version_id"], "trace_id": trace_id,
                    "latency_seconds": time.perf_counter() - start, "execution": execution}
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {**row, "score": None, "prediction": None,
                    "error": str(exc).replace(self.api.key, "[REDACTED]"),
                    "scorer_version_id": info["scorer_version_id"], "trace_id": trace_id}


def binary(rows, label=None):
    pairs = [(r["gold"], r["prediction"]) if label is None else
             (int(label in r["gold"]), int(label in r["prediction"])) for r in rows]
    counts = Counter(pairs)
    tn, fp, fn, tp = counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1]
    result = prf(tp, fp, fn)
    return {**result, "tn": tn, "support": tp + fn,
            "accuracy": (tp + tn) / len(rows) if rows else None}


def prf(tp, fp, fn):
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def multilabel(rows):
    classes = {label: binary(rows, label) for label in PII_CLASSES}
    totals = [sum(r[key] for r in classes.values()) for key in ("tp", "fp", "fn")]
    return {"rows": len(rows), "per_class": classes, "micro": prf(*totals),
            "macro_f1": sum(r["f1"] for r in classes.values()) / len(classes),
            "exact_match_accuracy": sum(r["gold"] == r["prediction"] for r in rows) / len(rows) if rows else None}


def summarize(metric, records, requested):
    valid = [r for r in records if r["error"] is None]
    summary = {"requested_rows": requested, "successful_rows": len(valid),
               "failed_rows": len(records) - len(valid), "coverage": len(valid) / requested,
               "error_counts": dict(Counter(r["error"] for r in records if r["error"]))}
    if not valid:
        return summary  # Never present infrastructure failure as zero model F1.
    if metric != "pii":
        summary.update(binary(valid))
        summary["threshold"] = 0.5
        if len({r["gold"] for r in valid}) == 2:
            truth, scores = [r["gold"] for r in valid], [r["score"] for r in valid]
            summary.update(auroc=float(roc_auc_score(truth, scores)),
                           average_precision=float(average_precision_score(truth, scores)))
    else:
        summary.update(multilabel(valid))
        summary["by_language"] = {
            lang: {"requested_rows": sum(r.get("language") == lang for r in records),
                   **multilabel([r for r in valid if r["language"] == lang])}
            for lang in sorted({r["language"] for r in records})}
        summary["predicted_document_counts_all_exposed_classes"] = dict(
            Counter(label for r in valid for label in set(r["score"])))
    return summary


def run(api, state, sources, output, workers, retry_errors, dataset_workers=1):
    report = {"pii_mapping": PII_MAP, "pii_primary_classes": PII_CLASSES, "metrics": {}}
    output.mkdir(parents=True, exist_ok=True)
    if dataset_workers > 1:
        failures = {}
        with ThreadPoolExecutor(max_workers=dataset_workers) as pool:
            futures = {
                pool.submit(run, api, state, {metric: rows}, output / metric,
                            1, retry_errors): metric
                for metric, rows in sources.items()
            }
            for future in as_completed(futures):
                metric = futures[future]
                try:
                    result = future.result()
                    report["metrics"].update(result["metrics"])
                except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
                    failures[metric] = str(exc).replace(api.key, "[REDACTED]")
                    print(f"{metric}: failed: {failures[metric]}", flush=True)
                report["dataset_errors"] = failures
                save(output / "summary.json", report)
        if failures:
            raise RuntimeError("Dataset failures: " + ", ".join(failures))
        return report
    for metric, rows in sources.items():
        path = output / f"{metric}-predictions.jsonl"
        latest = {}
        if path.exists():
            # Iterate physical lines: multilingual text can contain Unicode line
            # separators that str.splitlines() would incorrectly split inside JSON.
            with path.open(encoding="utf-8") as handle:
                latest = {r["index"]: r for r in (json.loads(line) for line in handle)}
        for index, old in latest.items():
            if not isinstance(index, int) or index < 0 or index >= len(rows) or any(old[k] != rows[index][k] for k in ("input", "gold", "source_index")) or old[
                    "scorer_version_id"] != state["metrics"][metric]["scorer_version_id"]:
                raise ValueError("Checkpoint does not match this dataset/version. Use a new --output folder.")
        evaluator = Evaluator(api, state["metrics"][metric], metric)
        pending = [r for r in rows if r["index"] not in latest or (retry_errors and latest[r["index"]]["error"])]
        # Stop on infrastructure/configuration failure before issuing a large batch.
        if pending:
            print(f"{metric}: requesting preflight sample {pending[0]['index']}", flush=True)
            first = evaluator.evaluate(pending[0])
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(first, ensure_ascii=False) + "\n")
            latest[first["index"]] = first
            if first["error"]:
                raise RuntimeError(f"{metric} preflight failed: {first['error']}")
            pending = pending[1:]
            print(f"{metric}: preflight passed; processing {len(rows)} rows", flush=True)
        with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
            if workers == 1:
                records = map(evaluator.evaluate, pending)
            else:
                futures = [pool.submit(evaluator.evaluate, row) for row in pending]
                records = (future.result() for future in as_completed(futures))
            for record in records:
                latest[record["index"]] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if len(latest) % 50 == 0 or len(latest) == len(rows):
                    print(f"{metric}: {len(latest)}/{len(rows)}, errors={sum(bool(r['error']) for r in latest.values())}", flush=True)
        report["metrics"][metric] = summarize(metric, list(latest.values()), len(rows))
        save(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--metric", choices=["both", "pi", "pii"], default="both")
    parser.add_argument("--pii-dataset", choices=["mini", "nano"], default="mini")
    parser.add_argument("--pii-split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--pi-size", type=int, default=1000, help="Stratified train sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Connectivity test: first N selected rows per dataset")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--source", choices=["public", "prepared"], default="public")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "prepared")
    parser.add_argument("--datasets", default="all",
                        help="Prepared non-PII datasets: all or comma-separated names")
    parser.add_argument("--dataset-workers", type=int, default=1,
                        help="Concurrent datasets; values above 1 use one sequential worker per dataset")
    for task in ("toxicity", "hate", "safety"):
        parser.add_argument(f"--{task}-preset")
    parser.add_argument("--list-presets", action="store_true",
                        help="List visible preset names without creating benchmark resources")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    required = ["GALILEO_API_URL", "GALILEO_API_KEY", "GALILEO_PROJECT_ID"]
    if any(not os.getenv(k) or os.getenv(k).startswith("YOUR_") for k in required):
        parser.error("Set GALILEO_API_URL, GALILEO_API_KEY and GALILEO_PROJECT_ID in .env or your environment")
    if args.workers < 1 or args.dataset_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers, --dataset-workers and --limit must be positive")
    if args.list_presets:
        api = API(os.environ["GALILEO_API_URL"], os.environ["GALILEO_API_KEY"])
        try:
            catalog = api.request("POST", "/v2/scorers/list", {"filters": []})
            for scorer in catalog["scorers"]:
                if scorer.get("scorer_type") == "preset":
                    print(scorer["name"], scorer.get("model_type"), flush=True)
        finally:
            api.close()
        return 0
    presets = dict(PRESETS)
    if args.source == "prepared" and args.metric != "pii":
        try:
            sources, provenance, presets = load_prepared(
                args.data_dir, args.datasets,
                {"pi": PRESETS["pi"], **{
                    task: getattr(args, f"{task}_preset") or os.getenv(f"GALILEO_{task.upper()}_PRESET")
                    for task in ("toxicity", "hate", "safety")}}, args.limit)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        if args.metric == "both":
            pii_sources, pii_provenance = load_sources(
                args.cache, args.pi_size, args.seed, "pii", args.pii_dataset, args.pii_split)
            sources.update(pii_sources)
            provenance.update(pii_provenance)
            presets["pii"] = PRESETS["pii"]
    else:
        sources, provenance = load_sources(
            args.cache, args.pi_size, args.seed, args.metric, args.pii_dataset, args.pii_split)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.limit:
        sources = {key: value[:args.limit] for key, value in sources.items()}
    for metric, rows in sources.items():
        print(f"Loaded {metric}: {len(rows)} rows; input/gold schema validated", flush=True)
    api = API(os.environ["GALILEO_API_URL"], os.environ["GALILEO_API_KEY"])
    try:
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            state = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (state["api_url"] != api.url or state["project_id"] != os.environ["GALILEO_PROJECT_ID"]
                    or state["source_fingerprint"] != digest(sources)):
                raise ValueError("Existing results have a different configuration. Use a new --output folder.")
            if any(state["metrics"][metric]["scorer_name"] != presets[metric] for metric in sources):
                raise ValueError("Preset mapping changed. Use a new --output folder.")
            if "pii" in sources and any(
                    state["datasets"]["pii"][key] != provenance["pii"][key]
                    for key in ("dataset", "revision", "split")):
                raise ValueError("PII dataset configuration changed. Use a new --output folder.")
            if state.get("dataset_subdirectories", False) != (args.dataset_workers > 1):
                raise ValueError("Dataset checkpoint layout changed. Use a new --output folder.")
        else:
            state = setup(api, os.environ["GALILEO_PROJECT_ID"], sources, provenance, args.output,
                          presets, args.dataset_workers)
        if args.dataset_workers > 1:
            print(f"Dataset concurrency: {args.dataset_workers}; one sequential worker per dataset",
                  flush=True)
        result = run(api, state, sources, args.output, args.workers, args.retry_errors,
                     args.dataset_workers)
        print("\nMetric / class                         Precision   Recall       F1")
        for metric, summary in result["metrics"].items():
            statistics = {metric: summary} if metric != "pii" else {
                **{f"PII: {label}": value for label, value in summary.get("per_class", {}).items()},
                "PII: micro average (4 classes)": summary.get("micro", {})}
            for label, value in statistics.items():
                if "f1" in value:
                    print(f"{label:38} {value['precision']:9.2%} {value['recall']:9.2%} {value['f1']:9.2%}")
        print(f"\nSaved predictions, pinned scorer IDs and metrics to {args.output.resolve()}")
        return int(any(r["failed_rows"] for r in result["metrics"].values()))
    finally:
        api.close()


if __name__ == "__main__":
    raise SystemExit(main())
