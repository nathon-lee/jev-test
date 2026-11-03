#!/usr/bin/env python3
"""Benchmark the original Galileo PI/PII presets through Agent Control.

Run from this folder: python benchmark.py --env-file .env
Use --limit 3 for a connectivity test before the full benchmark.
No local project imports, custom metrics, model uploads, or prompt changes.
"""

import argparse
import base64
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
PI_REVISION = "5f68a89245ce067ba4258c92198633dbebd02dac"
PII_REVISION = "7b686c2e7475b02e10e38add1eb54a9f604f4361"
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


def load_sources(cache, pi_size, seed, metric):
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
            path = download(client, cache, "ai4privacy/pii-masking-mini-10k",
                            PII_REVISION, "data/validation.jsonl")
            with path.open(encoding="utf-8") as handle:
                data = [json.loads(line) for line in handle if line.strip()]
            if not {"source_text", "privacy_mask", "language", "uid"} <= set(data[0]):
                raise ValueError("PII dataset schema changed")
            sources["pii"] = []
            for i, row in enumerate(data):
                labels = {span["label"] for span in row["privacy_mask"]}
                sources["pii"].append({
                    "index": i, "source_index": row["uid"], "input": row["source_text"],
                    "gold": sorted({PII_MAP[x] for x in labels if x in PII_MAP}),
                    "language": row["language"], "dataset_labels": sorted(labels),
                })
            provenance["pii"] = {"dataset": "ai4privacy/pii-masking-mini-10k",
                                 "revision": PII_REVISION, "split": "validation",
                                 "mapping": PII_MAP, "file_sha256":
                                 hashlib.sha256(path.read_bytes()).hexdigest()}
    return sources, provenance


def setup(api, project_id, sources, provenance, output):
    project = api.request("GET", f"/projects/{project_id}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    state = {"api_url": api.url, "project_id": project_id,
             "project_name": project["name"], "run_id": run_id,
             "source_fingerprint": digest(sources), "datasets": provenance, "metrics": {}}
    for metric in sources:
        name = PRESETS[metric]
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
        if metric == "pi":
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
            if self.metric == "pi":
                if isinstance(score, list) and len(score) == 1:
                    score = score[0]
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Expected a PI probability in [0, 1]")
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
    if metric == "pi":
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


def run(api, state, sources, output, workers, retry_errors):
    report = {"pii_mapping": PII_MAP, "pii_primary_classes": PII_CLASSES, "metrics": {}}
    for metric, rows in sources.items():
        path = output / f"{metric}-predictions.jsonl"
        latest = {}
        if path.exists():
            # Iterate physical lines: multilingual text can contain Unicode line
            # separators that str.splitlines() would incorrectly split inside JSON.
            with path.open(encoding="utf-8") as handle:
                latest = {r["index"]: r for r in (json.loads(line) for line in handle)}
        for index, old in latest.items():
            if index >= len(rows) or any(old[k] != rows[index][k] for k in ("input", "gold", "source_index")) or old[
                    "scorer_version_id"] != state["metrics"][metric]["scorer_version_id"]:
                raise ValueError("Checkpoint does not match this dataset/version. Use a new --output folder.")
        evaluator = Evaluator(api, state["metrics"][metric], metric)
        pending = [r for r in rows if r["index"] not in latest or (retry_errors and latest[r["index"]]["error"])]
        # Stop on infrastructure/configuration failure before issuing a large batch.
        if pending:
            first = evaluator.evaluate(pending[0])
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(first, ensure_ascii=False) + "\n")
            latest[first["index"]] = first
            if first["error"]:
                raise RuntimeError(f"{metric} preflight failed: {first['error']}")
            pending = pending[1:]
            print(f"{metric}: preflight passed; processing {len(rows)} rows", flush=True)
        with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(evaluator.evaluate, row) for row in pending]
            for future in as_completed(futures):
                record = future.result()
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
    parser.add_argument("--pi-size", type=int, default=1000, help="Stratified train sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Connectivity test: first N selected rows per dataset")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    required = ["GALILEO_API_URL", "GALILEO_API_KEY", "GALILEO_PROJECT_ID"]
    if any(not os.getenv(k) or os.getenv(k).startswith("YOUR_") for k in required):
        parser.error("Set GALILEO_API_URL, GALILEO_API_KEY and GALILEO_PROJECT_ID in .env or your environment")
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sources, provenance = load_sources(args.cache, args.pi_size, args.seed, args.metric)
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
        else:
            state = setup(api, os.environ["GALILEO_PROJECT_ID"], sources, provenance, args.output)
        result = run(api, state, sources, args.output, args.workers, args.retry_errors)
        print("\nMetric / class                         Precision   Recall       F1")
        for metric, summary in result["metrics"].items():
            statistics = {"Prompt Injection": summary} if metric == "pi" else {
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
