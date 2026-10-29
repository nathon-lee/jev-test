import json

import httpx
import pandas as pd
import pytest

from scripts.rerun_classification import export_results, infer, read_checkpoint
from scripts.report_classification_rerun import paired_metrics


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
