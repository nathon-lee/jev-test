"""Strict PII extraction contracts, hosted Fastino adapter, and offline scoring.

Offsets are Python Unicode character indices, start inclusive and end exclusive.
This module deliberately does not consume classification scores or redacted text.
"""

import ast
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


# Fixed benchmark taxonomy, never inferred from a test row's gold annotations.
PII_LABELS = {
    "TITLE": "personal honorific or title",
    "GIVENNAME": "person's given name",
    "SURNAME": "person's family name",
    "DATE": "calendar date",
    "STREET": "street address or street name",
    "BUILDINGNUM": "building or house number",
    "CITY": "city name",
    "EMAIL": "email address",
    "TELEPHONENUM": "telephone number",
    "ZIPCODE": "postal or zip code",
    "AGE": "person's age",
    "SEX": "person's sex",
    "GENDER": "person's gender",
    "TAXNUM": "tax identification number",
    "CREDITCARDNUMBER": "credit card number",
    "SOCIALNUM": "social security or social insurance number",
    "IDCARDNUM": "identity card number",
    "PASSPORTNUM": "passport number",
    "DRIVERLICENSENUM": "driver license number",
}
DEFAULT_EXTRACTION_MODEL = "fastino/gliner2.5-multi-v1"


@dataclass(frozen=True, order=True)
class Entity:
    start: int
    end: int
    label: str


def parse_annotation_list(value):
    """Read native arrays, JSON, or legacy Python-literal CSV annotations strictly."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Invalid annotations; regenerate the prepared dataset") from exc
    if not isinstance(value, list):
        raise ValueError("Annotations must be an explicit list (use [] for no PII)")
    return value


def normalize_entities(text, annotations):
    """Validate typed spans; deduplicate exact entities, retaining repeated mentions."""
    entities = set()
    for item in parse_annotation_list(annotations):
        if not isinstance(item, dict):
            raise ValueError("Each entity must be an object")
        start, end, label = item.get("start"), item.get("end"), item.get("label")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)):
            raise ValueError("Entity offsets must be half-open integer character indices")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Each entity requires a nonempty label")
        for key in ("text", "value"):
            if key in item and item[key] != text[start:end]:
                raise ValueError("Entity text/value does not match its source offsets")
        entities.add(Entity(start, end, label.strip().upper()))
    return sorted(entities)


def entity_dicts(entities):
    return [asdict(entity) for entity in entities]


def validate_sample(row):
    text = row.get("source_text", row.get("text"))
    if not isinstance(text, str) or not text:
        raise ValueError("Each sample requires nonempty source_text or text")
    raw = row.get("gold_entities", row.get("privacy_mask"))
    gold = normalize_entities(text, raw)
    target = row.get("target_text", row.get("masked_text"))
    if not gold and target is not None and target != text:
        raise ValueError("Empty annotations with a changed masked target; regenerate the dataset")
    sample_id = row.get("sample_id", row.get("uid"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError("Each sample requires a sample_id or uid")
    return {**row, "sample_id": str(sample_id), "source_text": text,
            "gold_entities": entity_dicts(gold)}


def load_samples(path):
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(handle))
        else:
            rows = [json.loads(line) for line in handle if line.strip()]
    samples = [validate_sample(row) for row in rows]
    ids = [row["sample_id"] for row in samples]
    if not samples or len(set(ids)) != len(ids):
        raise ValueError("Dataset must be nonempty with unique sample IDs")
    return samples


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def parse_fastino_entities(payload, text):
    """Parse the documented GLiNER chat envelope, without searching for numbers."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a chat completion response object")
    metadata = payload.get("x_fastino") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Invalid extraction metadata")
    if metadata.get("truncated"):
        raise ValueError("Truncated extraction cannot be scored as a complete response")
    try:
        choice = payload["choices"][0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise ValueError("Incomplete extraction response")
        content = json.loads(choice["message"]["content"])
        groups = content["entities"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Expected GLiNER JSON entities in choices[0].message.content") from exc
    if isinstance(groups, dict):
        groups = [groups]
    if not isinstance(groups, list):
        raise ValueError("Expected GLiNER entity groups")
    annotations = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Expected an entity label-to-spans object")
        for label, mentions in group.items():
            if not isinstance(mentions, list):
                raise ValueError("Expected an entity mention list")
            for mention in mentions:
                if not isinstance(mention, dict):
                    raise ValueError("Extraction requires character spans, not value-only strings")
                annotations.append({**mention, "label": label})
    return normalize_entities(text, annotations)


def call_fastino_extraction(text, *, api_key, model=DEFAULT_EXTRACTION_MODEL,
                            base_url="https://api.fastino.ai/v1", threshold=0.5,
                            labels=None, timeout=60.0):
    if not api_key:
        raise ValueError("Set FASTINO_API_KEY to run hosted extraction")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Extraction threshold must be in [0, 1]")
    if model.lower() == "fastino/glide":
        raise ValueError("GLiDE is a decision model; select a span extraction model")
    url = base_url.rstrip("/")
    if url.endswith("/systemone"):
        raise ValueError("Extraction requires a GLiNER chat/completions base URL")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions" if url.endswith("/v1") else "/v1/chat/completions"
    response = httpx.post(
        url, headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": text}],
              "schema": {"entities": [{"name": name, "description": description}
                                      for name, description in (labels or PII_LABELS).items()]},
              "threshold": threshold, "include_spans": True,
              "include_confidence": True, "store": False,
              "truncate_overflow_text": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    entities = parse_fastino_entities(payload, text)
    return entities, payload.get("model", model)


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def score_extraction(rows):
    """Corpus micro counts, per-type and macro F1; failures remain visible.

    End-to-end metrics treat failed responses as no extracted entities, counting
    their gold as FN and never counting them as exact document matches. Quality
    conditional on successful responses is reported separately with coverage.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot score an empty extraction run")
    typed = Counter()
    boundary = Counter()
    by_type = {}
    exact = success = gold_total = 0
    successful_rows = []
    ids = set()
    for row in rows:
        sample = validate_sample(row)
        if sample["sample_id"] in ids:
            raise ValueError("Duplicate sample_id in extraction run")
        ids.add(sample["sample_id"])
        text = sample["source_text"]
        gold = set(normalize_entities(text, sample["gold_entities"]))
        gold_total += len(gold)
        failed = bool(row.get("error"))
        if failed:
            predicted = set()
        else:
            predicted = set(normalize_entities(text, row.get("predicted_entities")))
            successful_rows.append(row)
            success += 1
            exact += predicted == gold
        for counts, expected, actual in (
            (typed, gold, predicted),
            (boundary, {(e.start, e.end) for e in gold}, {(e.start, e.end) for e in predicted}),
        ):
            counts.update(tp=len(expected & actual), fp=len(actual - expected), fn=len(expected - actual))
        for label in {e.label for e in gold | predicted}:
            counts = by_type.setdefault(label, Counter())
            expected = {e for e in gold if e.label == label}
            actual = {e for e in predicted if e.label == label}
            counts.update(tp=len(expected & actual), fp=len(actual - expected), fn=len(expected - actual))
    per_type = {label: {**_prf(**counts), "support": counts["tp"] + counts["fn"]}
                for label, counts in sorted(by_type.items())}
    summary = {
        "rows": len(rows), "successful_rows": success, "failed_rows": len(rows) - success,
        "request_success_rate": success / len(rows), "gold_entities": gold_total,
        "entity_micro": _prf(**typed), "boundary_micro": _prf(**boundary),
        "per_type": per_type,
        "entity_macro_f1": sum(v["f1"] for v in per_type.values()) / len(per_type) if per_type else 0.0,
        "exact_document_match": exact / len(rows),
    }
    if success and success != len(rows):
        summary["successful_only"] = score_extraction(successful_rows)
    return summary
