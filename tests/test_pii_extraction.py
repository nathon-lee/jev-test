import json

import httpx
import numpy as np
import pytest

from benchmark.datasets import _parse_privacy_mask
from benchmark.pii_extraction import (
    Entity, call_fastino_extraction, load_samples, normalize_entities,
    parse_fastino_entities, score_extraction, validate_sample, write_jsonl,
)
from scripts.pii_extraction import main, prepare_records, run_samples


def entity(start=0, end=5, label="GIVENNAME", **extra):
    return {"start": start, "end": end, "label": label, **extra}


def sample(gold=None, predicted=None, **extra):
    return {"sample_id": "0", "source_text": "Alice met Alice.",
            "gold_entities": [entity()] if gold is None else gold,
            "predicted_entities": [entity()] if predicted is None else predicted, **extra}


def envelope(entities, **extra):
    return {"model": "resolved-gliner", "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"entities": entities})}}], **extra}


def test_exact_match_and_empty_negative():
    result = score_extraction([sample(), sample([], [], sample_id="negative")])
    assert result["entity_micro"] == {"tp": 1, "fp": 0, "fn": 0,
                                      "precision": 1, "recall": 1, "f1": 1}
    assert result["exact_document_match"] == 1


def test_counts_are_corpus_micro_not_mean_document_f1():
    result = score_extraction([
        sample([entity(), entity(10, 15)], [entity()]),
        sample([], [entity()], sample_id="negative"),
    ])
    assert result["entity_micro"]["tp"] == 1
    assert result["entity_micro"]["fp"] == 1
    assert result["entity_micro"]["fn"] == 1
    assert result["entity_micro"]["f1"] == 0.5
    assert result["per_type"]["GIVENNAME"]["support"] == 2


def test_wrong_type_counts_fp_and_fn_but_boundary_matches():
    result = score_extraction([sample(predicted=[entity(label="CITY")])])
    assert result["entity_micro"]["f1"] == 0
    assert result["entity_micro"]["fp"] == result["entity_micro"]["fn"] == 1
    assert result["boundary_micro"]["f1"] == 1
    assert result["entity_macro_f1"] == 0


def test_partial_overlap_is_not_an_exact_entity():
    result = score_extraction([sample(predicted=[entity(end=4)])])
    assert result["boundary_micro"]["tp"] == 0
    assert result["entity_micro"]["tp"] == 0
    assert result["exact_document_match"] == 0


def test_per_type_macro_differs_from_micro_with_unequal_support():
    result = score_extraction([sample(
        [entity(), entity(10, 15), entity(6, 9, "OTHER")],
        [entity(), entity(10, 15)],
    )])
    assert result["entity_micro"]["f1"] == pytest.approx(0.8)
    assert result["entity_macro_f1"] == 0.5
    assert result["per_type"]["OTHER"]["fn"] == 1


def test_deduplicate_exact_entities_but_keep_repeated_mentions():
    result = score_extraction([sample([entity(), entity(10, 15)], [entity(), entity()])])
    assert result["entity_micro"]["tp"] == 1
    assert result["entity_micro"]["fn"] == 1
    assert result["entity_micro"]["fp"] == 0


def test_unicode_offsets_and_case_normalization():
    assert normalize_entities("😀 李明", [entity(2, 4, " person ", value="李明")]) == [Entity(2, 4, "PERSON")]


@pytest.mark.parametrize("annotation", [
    entity(-1, 5), entity(0, 99), entity(5, 5), entity(True, 5),
    entity(0.0, 5), entity(0, "5"), entity(label=""), entity(text="Bob"),
    entity(value="alice"), {"value": "Alice", "label": "GIVENNAME"}, 1,
])
def test_malformed_entities_fail_instead_of_getting_credit(annotation):
    with pytest.raises(ValueError):
        normalize_entities("Alice", [annotation])


@pytest.mark.parametrize("annotations", [None, "", "not JSON", "{}", 1])
def test_missing_and_invalid_annotations_are_not_empty_gold(annotations):
    with pytest.raises(ValueError):
        normalize_entities("Alice", annotations)


def test_arrow_array_annotations_and_legacy_csv_literals_survive():
    raw = np.array([entity(value="Alice")], dtype=object)
    assert _parse_privacy_mask(raw) == [entity(value="Alice")]
    assert normalize_entities("Alice", str([entity()])) == [Entity(0, 5, "GIVENNAME")]


def test_erased_checked_in_annotations_are_rejected():
    with pytest.raises(ValueError, match="regenerate"):
        validate_sample(sample([], [], target_text="[GIVENNAME] met Alice."))


def test_failed_requests_lower_coverage_and_recall_and_never_match_negative():
    rows = [sample(), sample(sample_id="failed", predicted=None, error="HTTP 403"),
            sample([], [], sample_id="failed-negative", error="HTTP 403")]
    result = score_extraction(rows)
    assert result["failed_rows"] == 2
    assert result["request_success_rate"] == pytest.approx(1 / 3)
    assert result["entity_micro"]["recall"] == 0.5
    assert result["exact_document_match"] == pytest.approx(1 / 3)
    assert result["successful_only"]["entity_micro"]["f1"] == 1


def test_all_failed_run_has_zero_f1_and_no_successful_only_score():
    result = score_extraction([sample(error="HTTP 401")])
    assert result["request_success_rate"] == result["entity_micro"]["f1"] == 0
    assert "successful_only" not in result


def test_empty_corpus_duplicate_ids_and_missing_predictions_rejected():
    for rows in ([], [sample(), sample()], [{"sample_id": 0, "text": "Alice", "privacy_mask": []}]):
        with pytest.raises(ValueError):
            score_extraction(rows)


def test_fastino_parser_uses_spans_and_rejects_classification():
    assert parse_fastino_entities(envelope([{"GIVENNAME": [entity(text="Alice")]}]), "Alice") == [Entity(0, 5, "GIVENNAME")]
    assert parse_fastino_entities(envelope({}), "Alice") == []
    assert parse_fastino_entities(envelope({}, x_fastino=None), "Alice") == []
    for payload in ({"answers": {"score": {"noul": 1, "confidence": 0.9}}},
                    envelope({"GIVENNAME": ["Alice"]}),
                    envelope({}, x_fastino={"truncated": True})):
        with pytest.raises(ValueError):
            parse_fastino_entities(payload, "Alice")


def test_fastino_adapter_contract(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(200, json=envelope({"GIVENNAME": [entity(text="Alice")]}),
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    entities, resolved = call_fastino_extraction("Alice", api_key="test-secret", base_url="https://example.test")
    assert entities == [Entity(0, 5, "GIVENNAME")]
    assert resolved == "resolved-gliner"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert "verify" not in captured  # TLS verification stays enabled.
    payload = captured["json"]
    assert payload["model"] != "fastino/GLiDE"
    assert payload["messages"] == [{"role": "user", "content": "Alice"}]
    assert payload["include_spans"] is True
    assert payload["store"] is False
    assert payload["truncate_overflow_text"] is False
    assert len(payload["schema"]["entities"]) == 19
    assert "gold_entities" not in payload


@pytest.mark.parametrize("kwargs", [
    {"api_key": None}, {"threshold": float("nan")}, {"threshold": 2},
    {"model": "fastino/GLiDE"}, {"base_url": "https://example.test/v1/systemone"},
])
def test_adapter_rejects_unsupported_configuration_before_network(monkeypatch, kwargs):
    def unexpected(*args, **kwargs):
        pytest.fail("Should not make a request")
    monkeypatch.setattr(httpx, "post", unexpected)
    with pytest.raises(ValueError):
        call_fastino_extraction("Alice", **{"api_key": "test", **kwargs})


def test_preparation_preserves_partition_identity_and_gold():
    records = [{"uid": 42, "source_text": "Alice", "masked_text": "[GIVENNAME_1]",
                "privacy_mask": [entity(value="Alice")]}]
    prepared = prepare_records(records, dataset="test", split="validation", revision="abc123")
    assert prepared[0]["sample_id"] == "42"
    assert prepared[0]["split"] == "validation"
    assert prepared[0]["revision"] == "abc123"
    assert prepared[0]["gold_entities"] == [entity()]


def test_offline_cli_round_trip_and_failed_exit(tmp_path):
    predictions, summary = tmp_path / "predictions.jsonl", tmp_path / "metrics.json"
    write_jsonl(predictions, [sample()])
    main(["evaluate", "--input", str(predictions), "--summary", str(summary)])
    result = json.loads(summary.read_text())
    assert result["entity_micro"]["f1"] == 1
    assert result["input_sha256"]
    assert load_samples(predictions)[0]["source_text"] == "Alice met Alice."
    write_jsonl(predictions, [sample(error="HTTP 403")])
    with pytest.raises(SystemExit, match="1"):
        main(["evaluate", "--input", str(predictions), "--summary", str(summary)])
    assert json.loads(summary.read_text())["failed_rows"] == 1


def test_run_preserves_references_and_reports_http_restrictions(monkeypatch):
    def denied(*args, **kwargs):
        response = httpx.Response(403, request=httpx.Request("POST", "https://example.test"))
        raise httpx.HTTPStatusError("sensitive response body", request=response.request, response=response)
    monkeypatch.setattr("scripts.pii_extraction.call_fastino_extraction", denied)
    rows = run_samples([sample()], model="test-model", threshold=0.5,
                       api_key="test-secret", base_url="https://example.test")
    assert rows[0]["error"] == "HTTPStatusError HTTP 403"
    assert rows[0]["predicted_entities"] is None
    assert rows[0]["gold_entities"] == [entity()]
    assert rows[0]["elapsed_ms"] >= 0
    assert "test-secret" not in json.dumps(rows)


def test_run_cli_with_mocked_hosted_response(tmp_path, monkeypatch):
    source, predictions, summary = (tmp_path / name for name in ("data.jsonl", "predictions.jsonl", "metrics.json"))
    write_jsonl(source, [sample()])
    monkeypatch.setenv("FASTINO_API_KEY", "test-key")
    monkeypatch.setattr("scripts.pii_extraction.call_fastino_extraction",
                        lambda *args, **kwargs: ([Entity(0, 5, "GIVENNAME")], "resolved-model"))
    main(["run", "--input", str(source), "--output", str(predictions), "--summary", str(summary)])
    result = json.loads(summary.read_text())
    assert result["entity_micro"]["f1"] == 1
    assert result["resolved_models"] == ["resolved-model"]
    assert load_samples(predictions)[0]["task"] == "pii_extraction"


def test_cli_prevents_overwriting_input_with_summary(tmp_path):
    source = tmp_path / "data.jsonl"
    write_jsonl(source, [sample()])
    before = source.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main(["evaluate", "--input", str(source), "--summary", str(source)])
    assert source.read_bytes() == before
