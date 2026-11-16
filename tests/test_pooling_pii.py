import csv
import json
import subprocess
import sys
import threading

import httpx
import pytest

from benchmark.pooling_pii import Sample, load_samples, parse_entities, run_benchmark


SCHEMA = {"person": "", "location": ""}
TEXT = "Alice in Paris"
SAMPLE = Sample("a", TEXT, frozenset({("person", 0, 5), ("location", 9, 14)}))


def payload():
    return {"data": {"entities": {
        "person": [{"text": "Alice", "start": 0, "end": 5, "confidence": .9}],
        "location": [{"text": "Paris", "start": 9, "end": 14, "confidence": .7}],
    }}}


def test_strict_metrics_errors_warmup_and_payload():
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 3:
            return httpx.Response(503, text="do not log this body")
        result = payload()
        # A boundary error is an FP and FN, not a fuzzy match.
        result["data"]["entities"]["location"][0].update(text="Pari", end=13)
        return httpx.Response(200, json=result)

    rows, summary = run_benchmark(
        [SAMPLE], SCHEMA, endpoint="http://test/pooling", model="/models/current",
        warmup=1, repeats=2, transport=httpx.MockTransport(handle),
    )
    assert len(calls) == 3
    assert calls[0] == {
        "model": "/models/current",
        "data": {"text": TEXT, "schema": {"entities": SCHEMA}, "runtime": "classic",
                 "include_spans": True, "include_confidence": True},
    }
    assert len(rows) == 2
    assert rows[1]["error"] == "HTTP 503"
    assert summary["successful_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["error_rate"] == .5
    assert summary["quality_successful_requests_only"]["micro"] == {
        "tp": 1, "fp": 1, "fn": 1, "precision": .5, "recall": .5, "f1": .5,
    }
    assert summary["successful_requests_per_second"] == 1 / summary["wall_seconds"]
    assert summary["latency_seconds_successful"]["p95"] == rows[0]["latency_seconds"]


def test_confidence_duplicates_unicode_and_category_matching():
    result = payload()
    result["data"]["entities"]["person"] *= 2
    assert parse_entities(result, TEXT, SCHEMA, .8) == {("person", 0, 5)}
    unicode_payload = {"data": {"entities": {"person": [
        {"text": "李明", "start": 1, "end": 3, "confidence": 1},
    ]}}}
    assert parse_entities(unicode_payload, "🙂李明", {"person": ""}, 0) == {("person", 1, 3)}
    wrong_type = Sample("a", TEXT, frozenset({("location", 0, 5), ("person", 9, 14)}))
    _, summary = run_benchmark(
        [wrong_type], SCHEMA, endpoint="http://test", model="m", warmup=0,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload())),
    )
    assert summary["quality_successful_requests_only"]["micro"]["f1"] == 0


@pytest.mark.parametrize("change", ["missing", "unknown", "offset", "text", "confidence", "bool_offset"])
def test_malformed_predictions_are_errors(change):
    result = payload()
    item = result["data"]["entities"]["person"][0]
    if change == "missing":
        del result["data"]["entities"]["location"]
    elif change == "unknown":
        result["data"]["entities"]["unexpected"] = []
    elif change == "offset":
        item["end"] = 100
    elif change == "bool_offset":
        item["start"] = False
    elif change == "text":
        item["text"] = "Bob"
    else:
        item["confidence"] = float("nan")
    with pytest.raises(ValueError):
        parse_entities(result, TEXT, SCHEMA, 0)


def test_dataset_mapping_empty_rows_and_missing_annotations(tmp_path):
    path = tmp_path / "source.jsonl"
    records = [
        {"uid": "a", "source_text": TEXT, "privacy_mask": [
            {"label": "GIVENNAME", "start": 0, "end": 5, "value": "Alice"},
        ]},
        {"uid": "b", "source_text": "Nothing.", "privacy_mask": []},
    ]
    path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    samples, mapping = load_samples(path, {"GIVENNAME": "person"})
    assert mapping == {"GIVENNAME": "person"}
    assert samples[0].gold == {("person", 0, 5)}
    assert not samples[1].gold
    records[0]["privacy_mask"] = []
    path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    with pytest.raises(ValueError, match="no gold spans"):
        load_samples(path)


def test_schema_from_whole_dataset_and_csv_validation(tmp_path):
    path = tmp_path / "source.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "privacy_mask"])
        writer.writeheader()
        for label in ("PERSON", "NAME"):
            writer.writerow({"text": "Alice", "privacy_mask": json.dumps([
                {"label": label, "start": 0, "end": 5, "value": "Alice"},
            ])})
    samples, mapping = load_samples(path)
    assert mapping == {"PERSON": "person", "NAME": "name"}
    with pytest.raises(ValueError, match="every mapped gold category"):
        run_benchmark(samples, {"person": ""}, endpoint="http://test", model="m")


def test_concurrency_and_all_failed_quality():
    barrier = threading.Barrier(2)

    def handle(request):
        barrier.wait(timeout=5)
        raise httpx.ReadTimeout("do not log this arbitrary transport message")

    rows, summary = run_benchmark(
        [SAMPLE], SCHEMA, endpoint="http://test", model="m",
        concurrency=2, repeats=2, warmup=0, transport=httpx.MockTransport(handle),
    )
    assert len(rows) == 2
    assert summary["failed_requests"] == 2
    assert summary["quality_successful_requests_only"]["micro"] is None
    assert summary["latency_seconds_successful"]["p50"] is None
    assert summary["errors"] == {"ReadTimeout": 2}


def test_failed_warmup_aborts():
    with pytest.raises(httpx.HTTPStatusError):
        run_benchmark(
            [SAMPLE], SCHEMA, endpoint="http://test", model="m",
            transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        )


def test_cli_rejects_lost_annotations_without_writing_success(tmp_path):
    source = tmp_path / "bad.jsonl"
    output = tmp_path / "results"
    source.write_text(json.dumps({"text": TEXT, "privacy_mask": []}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_pooling_pii", "--input", str(source),
         "--output-dir", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "no gold spans" in result.stderr
    assert not output.exists()


def test_preparation_preserves_array_masks_as_json_csv(tmp_path):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("datasets")
    from benchmark.datasets import _parse_privacy_mask, save_dataset

    annotations = np.array([{"label": "PERSON", "start": 0, "end": 5, "value": "Alice"}])
    parsed = _parse_privacy_mask(annotations)
    assert parsed == [{"label": "PERSON", "start": 0, "end": 5, "value": "Alice"}]
    frame = pd.DataFrame([{"text": "Alice", "label": 1, "privacy_mask": parsed}])
    target = tmp_path / "prepared.csv"
    save_dataset(frame, target)
    samples, _ = load_samples(target)
    assert samples[0].gold == {("person", 0, 5)}
    assert isinstance(frame.iloc[0]["privacy_mask"], list)
