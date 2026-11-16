"""Strict entity-span evaluation and client-side pooling performance."""

import csv
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


Span = tuple[str, int, int]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    gold: frozenset[Span]


def validate_span(text: str, item: dict[str, Any], value_key: str) -> tuple[int, int]:
    start, end = item.get("start"), item.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Invalid span offsets; expected Python character offsets [start, end).")
    if not isinstance(item.get(value_key), str) or text[start:end] != item[value_key]:
        raise ValueError("Span text does not match the source at [start, end).")
    return start, end


def load_samples(
    path: Path, label_map: dict[str, str] | None = None,
) -> tuple[list[Sample], dict[str, str]]:
    """Keep all annotated categories; never infer gold spans from masked text."""
    mapping = dict(label_map or {})
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip()
           for k, v in mapping.items()):
        raise ValueError("Label map must map nonempty dataset labels to nonempty entity names.")
    samples = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = csv.DictReader(handle) if path.suffix.lower() == ".csv" else (
            json.loads(line) for line in handle if line.strip()
        )
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                raise ValueError(f"Row {index + 1}: expected a JSON object.")
            text = row.get("source_text", row.get("text"))
            masks = row.get("privacy_mask")
            if isinstance(masks, str):
                masks = json.loads(masks)
            if not isinstance(text, str) or not text or not isinstance(masks, list):
                raise ValueError(f"Row {index + 1}: source_text/text and privacy_mask list required.")
            sample_id = str(row.get("sample_id", row.get("uid", index)))
            if sample_id in seen:
                raise ValueError(f"Row {index + 1}: duplicate sample ID.")
            seen.add(sample_id)
            gold = set()
            for item in masks:
                if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not item["label"]:
                    raise ValueError(f"Row {index + 1}: each annotation needs a nonempty label.")
                label = item["label"]
                mapping.setdefault(label, label.lower())
                start, end = validate_span(text, item, "value")
                gold.add((mapping[label], start, end))
            samples.append(Sample(sample_id, text, frozenset(gold)))
    if not samples or not any(sample.gold for sample in samples):
        raise ValueError(
            "Dataset is empty or has no gold spans. Older prepared PII CSVs lost privacy_mask "
            "annotations; use original AI4Privacy JSONL or regenerate the CSV."
        )
    return samples, mapping


def parse_entities(payload: Any, text: str, schema: dict[str, str], threshold: float) -> set[Span]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise ValueError("Pooling response requires data.entities.")
    entities = payload["data"].get("entities")
    if not isinstance(entities, dict) or set(entities) != set(schema):
        raise ValueError("Response entity categories must match the requested schema (use [] for no hits).")
    spans = set()
    for label, items in entities.items():
        if not isinstance(items, list):
            raise ValueError("Each entity category must contain a list.")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Entity must be an object.")
            start, end = validate_span(text, item, "text")
            confidence = item.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Entity confidence must be a finite number in [0, 1].")
            if confidence >= threshold:
                spans.add((label, start, end))
    return spans


def span_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_benchmark(
    samples: list[Sample], schema: dict[str, str], *, endpoint: str,
    model: str, runtime: str = "classic", threshold: float = 0.0,
    concurrency: int = 1, warmup: int = 3, repeats: int = 1,
    timeout: float = 60.0, transport: httpx.BaseTransport | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not samples or not schema:
        raise ValueError("At least one sample and one entity category are required.")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in schema.items()):
        raise ValueError("Schema must map nonempty entity names to string descriptions.")
    if concurrency < 1 or warmup < 0 or repeats < 1:
        raise ValueError("concurrency/repeats must be >= 1 and warmup must be >= 0.")
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("timeout must be positive and finite; threshold must be in [0, 1].")
    if any(label not in schema for sample in samples for label, _, _ in sample.gold):
        raise ValueError("Schema must cover every mapped gold category; categories cannot be silently excluded.")

    def request(client: httpx.Client, sample: Sample) -> tuple[set[Span], Any]:
        response = client.post(endpoint, json={
            "model": model,
            "data": {
                "text": sample.text, "schema": {"entities": schema}, "runtime": runtime,
                "include_spans": True, "include_confidence": True,
            },
        })
        response.raise_for_status()
        payload = response.json()
        return parse_entities(payload, sample.text, schema, threshold), payload.get("usage")

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    with httpx.Client(timeout=timeout, limits=limits, trust_env=False, transport=transport) as client:
        # A failed warmup aborts rather than measuring an unavailable/misconfigured service.
        for index in range(warmup):
            request(client, samples[index % len(samples)])

        def measure(job: tuple[int, Sample]) -> dict[str, Any]:
            repeat, sample = job
            start = time.perf_counter()
            row: dict[str, Any] = {
                "sample_id": sample.sample_id, "repeat": repeat, "gold": sorted(sample.gold),
            }
            try:
                predicted, usage = request(client, sample)
                row.update(
                    predicted=sorted(predicted), gold=sorted(sample.gold),
                    usage=usage, error=None,
                )
            except (httpx.HTTPError, ValueError) as exc:
                # Do not log server bodies or source text in failure messages.
                if isinstance(exc, httpx.HTTPStatusError):
                    error = f"HTTP {exc.response.status_code}"
                elif isinstance(exc, ValueError):
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    error = type(exc).__name__
                row.update(error=error)
            row["latency_seconds"] = time.perf_counter() - start
            return row

        jobs = [(repeat, sample) for repeat in range(repeats) for sample in samples]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            rows = list(executor.map(measure, jobs))
        wall_seconds = time.perf_counter() - start

    good = [row for row in rows if row["error"] is None]
    counts: dict[str, Counter[str]] = {label: Counter() for label in schema}
    for row in good:
        predicted, gold = set(map(tuple, row["predicted"])), set(map(tuple, row["gold"]))
        for key, spans in (("tp", predicted & gold), ("fp", predicted - gold), ("fn", gold - predicted)):
            for label, _, _ in spans:
                counts[label][key] += 1
    totals: Counter[str] = Counter()
    for count in counts.values():
        totals.update(count)
    latency = [row["latency_seconds"] for row in good]
    summary = {
        "requests": len(rows), "successful_requests": len(good),
        "failed_requests": len(rows) - len(good),
        "error_rate": (len(rows) - len(good)) / len(rows),
        "wall_seconds": wall_seconds, "successful_requests_per_second": len(good) / wall_seconds,
        "latency_seconds_successful": {
            "mean": sum(latency) / len(latency) if latency else None,
            "p50": percentile(latency, .50), "p95": percentile(latency, .95),
            "p99": percentile(latency, .99), "max": max(latency) if latency else None,
        },
        "quality_successful_requests_only": {
            "micro": span_metrics(totals["tp"], totals["fp"], totals["fn"]) if good else None,
            "per_category": {
                label: span_metrics(count["tp"], count["fp"], count["fn"])
                for label, count in counts.items()
            } if good else None,
        },
        "errors": dict(Counter(row["error"] for row in rows if row["error"])),
        "config": {
            "endpoint": endpoint, "model": model, "runtime": runtime, "threshold": threshold,
            "concurrency": concurrency, "warmup": warmup, "repeats": repeats, "schema": schema,
            "samples": len(samples),
            "input_sha256": hashlib.sha256(json.dumps(
                [(s.sample_id, s.text, sorted(s.gold)) for s in samples], ensure_ascii=False,
            ).encode("utf-8")).hexdigest(),
        },
    }
    return rows, summary
