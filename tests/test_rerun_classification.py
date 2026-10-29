import json

import httpx
import pandas as pd
import pytest

from scripts.rerun_classification import export_results, infer, read_checkpoint, load_input, digest
from scripts.report_classification_rerun import build_report, paired_metrics


def test_live_request_and_saved_probability_use_noul_not_confidence():
    def handle(request):
        body = json.loads(request.content)
        assert body["state"] == {"user_input": "Example"}
        assert body["questions"]["score"]["type"] == "noul"
        return httpx.Response(200, json={"model": "glide_v2", "answers": {
            "score": {"noul": 0.06, "confidence": 0.94}}})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = infer(client, "https://example.test/systemone", "fastino.ai", "fastino/GLiDE",
                       "Is it unsafe?", {"sample_id": "0", "text": "Example"}, 0)
    assert result["score"] == 0.06
    assert result["raw_confidence"] == 0.94
    assert result["resolved_model"] == "glide_v2"


def test_heldout_loader_rejects_training_and_tampered_partitions(tmp_path):
    path = tmp_path / "input.csv"
    provenance = {"source_repository": "test/data", "source_config": "default", "source_revision": "pinned",
                  "source_split": "test", "partition_role": "held_out"}
    pd.DataFrame([{**provenance, "source_row": 17, "sample_id": "007", "text": "text", "label": 1}]).to_csv(path, index=False)
    manifest = {**provenance, "csv_sha256": digest(path), "prepared_rows": 1}
    sidecar = path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest))
    samples, metadata = load_input(path, True)
    assert samples[0]["sample_id"] == "007"
    assert samples[0]["source_row"] == 17
    assert metadata["source_revision"] == "pinned"
    manifest["partition_role"] = "training_reference"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Training-reference"):
        load_input(path, True)
    manifest["partition_role"] = "held_out"
    manifest["csv_sha256"] = "changed"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest/count"):
        load_input(path, True)


def test_typesafe_checkpointed_request_also_requires_noul():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"answers": {"score": {"noul": .2, "confidence": .99}}}))) as client:
        result = infer(client, "https://example.test/v1/systemone", "typesafe.ai", "jev-latest",
                       "unsafe?", {"sample_id": "0", "text": "example"}, 0)
    assert result["score"] == .2


def test_checkpoint_retries_errors_and_rejects_unknown_samples(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(json.dumps({"sample_id": "0", "score": 0.2, "error": ""}) + "\n" +
                    json.dumps({"sample_id": "1", "score": None, "error": "HTTPStatusError"}) + "\n")
    assert list(read_checkpoint(path, {"0", "1"})) == ["0"]
    with pytest.raises(ValueError, match="unknown sample"):
        read_checkpoint(path, {"0"})


def test_invalid_noul_is_an_error_without_confidence_fallback():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"answers": {"score": {"confidence": 0.99}}}))) as client:
        result = infer(client, "https://example.test/systemone", "laya-ai.com", "typed-decisions",
                       "Is it unsafe?", {"sample_id": "0", "text": "Example"}, 0)
    assert result["score"] is None
    assert result["error"] == "ValueError"


def test_http_errors_do_not_include_sensitive_response_bodies():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        403, text="private response body"))) as client:
        result = infer(client, "https://example.test/systemone", "laya-ai.com", "typed-decisions",
                       "Is it unsafe?", {"sample_id": "0", "text": "Example"}, 0)
    assert result["http_status"] == 403
    assert "private response body" not in json.dumps(result)


def test_threshold_metrics_are_derived_from_one_probability_per_sample(tmp_path):
    samples = [{"sample_id": "0", "text": "Positive", "label": 1},
               {"sample_id": "1", "text": "Negative", "label": 0}]
    cached = {str(i): {"sample_id": str(i), "score": score, "error": "",
                      "resolved_model": "glide_v2", "latency_ms": 100} for i, score in enumerate((0.7, 0.2))}
    result = export_results(samples, cached, tmp_path / "scores.csv", "fastino.ai", "hatecheck", "v1", "fastino/GLiDE")
    assert result["valid"] == 2
    assert result["threshold_metrics"]["0.5"]["f1"] == 1
    assert result["threshold_metrics"]["0.8"]["f1"] == 0
    assert result["latency_p50_ms"] == 100


def test_comparison_uses_only_shared_successes_and_checks_labels_and_text():
    fresh = pd.DataFrame([
        {"sample_id": "0", "text": "Same", "label": 1, "score": 0.7, "prediction": 1, "error": ""},
        {"sample_id": "1", "text": "Failed", "label": 0, "score": None, "prediction": None, "error": "Timeout"},
    ])
    reference = pd.DataFrame([
        {"sample_id": "0", "text": "Same", "label": 1, "score": 0.2, "prediction": 0, "error": ""},
        {"sample_id": "1", "text": "Failed", "label": 0, "score": 0.2, "prediction": 0, "error": ""},
    ])
    result = paired_metrics(fresh, reference)
    assert result["common_successful_rows"] == 1
    assert result["rerun"]["f1"] == 1
    assert result["historical_jev"]["f1"] == 0
    reference.loc[0, "text"] = "Different"
    with pytest.raises(ValueError, match="mismatch"):
        paired_metrics(fresh, reference)


def test_known_legacy_nul_truncation_is_explicitly_excluded_from_pairing():
    fresh = pd.DataFrame([
        {"sample_id": "0", "text": "Same", "label": 1, "score": 0.7, "prediction": 1, "error": ""},
        {"sample_id": "1", "text": "prefix\x00suffix", "label": 0, "score": 0.2, "prediction": 0, "error": ""},
    ])
    old = fresh.copy()
    old.loc[1, "text"] = "prefix"
    result = paired_metrics(fresh, old)
    assert result["common_successful_rows"] == 1
    assert result["excluded_legacy_nul_rows"] == ["1"]
    assert result["rerun"]["n"] == 1
    old.loc[1, "label"] = 1
    with pytest.raises(ValueError, match="mismatch"):
        paired_metrics(fresh, old)


def test_report_excludes_blocked_partial_provider_and_records_access_restriction(tmp_path):
    provider_dir = tmp_path / "fastino.ai"
    provider_dir.mkdir()
    result = export_results(
        [{"sample_id": "0", "text": "example", "label": 1}],
        {"0": {"sample_id": "0", "score": 0.9, "error": "", "resolved_model": "glide_v2", "latency_ms": 10}},
        provider_dir / "hatecheck_v1.csv", "fastino.ai", "hatecheck", "v1", "fastino/GLiDE")
    (provider_dir / "hatecheck_v1.manifest.json").write_text(json.dumps({
        "provider": "fastino.ai", "dataset": "hatecheck", "prompt_version": "v1", "workers": 1, "result": result}))
    blocked_dir = tmp_path / "laya-ai.com"
    blocked_dir.mkdir()
    (blocked_dir / "hatecheck_v1.manifest.json").write_text(json.dumps({"provider": "laya-ai.com"}))
    (tmp_path / "access_restrictions.json").write_text(json.dumps([
        {"provider": "laya-ai.com", "http_status": 402, "status": "blocked"}]))
    report = build_report(tmp_path, historical_dir=tmp_path / "unavailable-reference")
    assert report["completed_providers"] == ["fastino.ai"]
    assert len(report["runs"]) == 1
    assert report["access_restrictions"][0]["http_status"] == 402
    assert "partial Laya scores are not published" in (tmp_path / "README.md").read_text()
