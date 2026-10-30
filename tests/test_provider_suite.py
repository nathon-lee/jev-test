import csv
import json

import httpx
import pytest

from benchmark.provider_suite import (
    DEFAULT_THRESHOLDS,
    available_datasets,
    call_provider,
    prompt_for_dataset,
    resolve_provider,
    normalize_provider_name,
    run_single_benchmark,
    save_rows,
)


def test_provider_aliases_normalize():
    assert normalize_provider_name("typesafe") == "typesafe.ai"
    assert normalize_provider_name("fastino") == "fastino.ai"
    assert normalize_provider_name("laya") == "laya-ai.com"
    assert normalize_provider_name("galileo") == "galileo.ai"
    assert normalize_provider_name("microsoft") == "microsoft-decision-1"


def test_default_thresholds_exist():
    assert DEFAULT_THRESHOLDS == [0.5, 0.6, 0.7, 0.8]


def test_prompt_versions_are_defined():
    assert "Does this text express toxicity" in prompt_for_dataset("toxigen", "v2")
    assert "Is this text hateful?" in prompt_for_dataset("hatecheck", "v1")
    assert "Is this text unsafe?" in prompt_for_dataset("aegis2", "v1")


def test_dataset_names_are_available():
    datasets = available_datasets()
    assert set(datasets) >= {
        "toxigen",
        "hatecheck",
        "aegis2",
        "detect-jailbreak",
        "jailbreak-classification",
        "jailbreakhub",
        "pii_masking_nano_1k",
        "pii_masking_mini_10k",
    }


def test_pii_prompt_versions_are_defined():
    assert "pii" in prompt_for_dataset("pii_masking_nano_1k", "v1").lower()
    assert "pii" in prompt_for_dataset("pii_masking_mini_10k", "v2").lower()


def test_provider_config_is_resolved():
    provider = resolve_provider("typesafe")
    assert provider["kind"] == "api_key"
    assert provider["api_key_env"] == "TYPESAFE_API_KEY"


def test_fastino_uses_systemone_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, verify=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json

        class DummyResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "model": "glide_v2",
                    "answers": {
                        "score": {
                            "type": "noul",
                            "noul": 0.06008665017400762,
                            "confidence": 0.8798266996519848,
                        }
                    },
                    "usage": {"input_tokens": 385, "output_tokens": 0},
                    "token_usage": 422,
                }

        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    score = call_provider(
        "fastino.ai",
        "Does this text contain PII?",
        "John Doe <john@example.com>",
        api_key="fast_test_key",
        base_url="https://api.fastino.ai/v1",
    )

    assert score == 0.06008665017400762
    assert captured["url"].endswith("/v1/systemone")
    assert "Authorization" in captured["headers"]
    assert "state" in captured["json"]
    assert "questions" in captured["json"]


def test_laya_uses_systemone_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, verify=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json

        class DummyResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "model": "laya-standard",
                    "answers": {
                        "score": {
                            "type": "noul",
                            "noul": 0.9261,
                            "confidence": 0.91,
                        }
                    },
                }

        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    score = call_provider(
        "laya-ai.com",
        "Is this text a jailbreak attempt?",
        "This is clean text.",
        api_key="laya_test_key",
        base_url="https://api.laya-ai.com/v1",
    )

    assert score == 0.9261
    assert captured["url"].endswith("/v1/systemone")
    assert "Authorization" in captured["headers"]
    assert "state" in captured["json"]
    assert "questions" in captured["json"]
    assert "messages" not in captured["json"]


@pytest.mark.parametrize("unsafe_score", [0.0, 0.1, 0.5, 1.0])
def test_finetuned_fastino_uses_exact_model_and_direct_unsafe_confidence(monkeypatch, unsafe_score):
    def fake_post(url, *, headers, json: dict, timeout):
        assert url == "https://api.fastino.ai/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "c94233af-98e6-4942-8e26-98ad204851bf"
        assert json["messages"] == [{"role": "user", "content": "Example"}]
        assert json["schema"]["classifications"] == [{
            "task": "Is this text hateful?", "labels": ["safe", "unsafe"],
            "multi_label": True, "cls_threshold": 0.0, "top_k": 2,
        }]
        assert json["threshold"] == 0.0 and json["store"] is False
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "model": json["model"], "x_fastino": None,
            "choices": [{"message": {"content": (
                '{"Is this text hateful?": [{"label": "safe", "confidence": 0.8},'
                f'{{"label": "unsafe", "confidence": {unsafe_score}}}]}}'
            )}}],
        })

    monkeypatch.setattr(httpx, "post", fake_post)
    score = call_provider(
        "fastino-ft-gg6-r1-ml2048-s7", "Is this text hateful?", "Example",
        api_key="test-key", base_url="https://api.fastino.ai/v1",
    )
    assert score == unsafe_score


@pytest.mark.parametrize("answer", [
    [{"label": "safe", "confidence": 0.9}],
    [{"label": "safe", "confidence": 0.9}, {"label": "safe", "confidence": 0.2}],
    [{"label": "safe", "confidence": 0.9}, {"label": "unsafe", "confidence": True}],
    [{"label": "safe", "confidence": 0.9}, {"label": "unsafe", "confidence": "0.5"}],
    [{"label": "safe", "confidence": 0.9}, {"label": "unsafe", "confidence": float("nan")}],
    [{"label": "safe", "confidence": 0.9}, {"label": "unsafe", "confidence": 1.1}],
])
def test_finetuned_fastino_rejects_missing_or_invalid_scores(monkeypatch, answer):
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("POST", url), json={
            "model": "c94233af-98e6-4942-8e26-98ad204851bf",
            "choices": [{"message": {"content": json.dumps({"question": answer})}}],
        },
    ))
    with pytest.raises(ValueError):
        call_provider("fastino-ft-gg6-r1-ml2048-s7", "question", "Example", api_key="test-key")


@pytest.mark.parametrize("response", [
    {"model": "fastino/GLiDE"},
    {"model": "c94233af-98e6-4942-8e26-98ad204851bf", "x_fastino": {"truncated": True}},
])
def test_finetuned_fastino_rejects_wrong_model_or_truncated_inputs(monkeypatch, response):
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(
        200, request=httpx.Request("POST", url), json=response,
    ))
    with pytest.raises(ValueError):
        call_provider("fastino-ft-gg6-r1-ml2048-s7", "question", "Example", api_key="test-key")


def test_save_rows_custom_output_root_preserves_existing_results(monkeypatch, tmp_path):
    monkeypatch.setattr("benchmark.provider_suite.OUTPUT_DIR", tmp_path / "old")
    old = save_rows([{"score": 0.1}], "fastino.ai", "toxigen", "v1", 0.5)
    new = save_rows(
        [{"score": 0.9}], "fastino-ft-gg6-r1-ml2048-s7", "toxigen", "v1", 0.5,
        output_dir=tmp_path / "new",
    )
    assert new.parent == tmp_path / "new" / "fastino-ft-gg6-r1-ml2048-s7"
    assert "0.1" in old.read_text() and "0.9" in new.read_text()


@pytest.mark.parametrize("provider", ["fastino.ai", "laya-ai.com"])
@pytest.mark.parametrize(
    "noul,confidence", [(0.06, 0.94), (0.94, 0.06), (0.5, 0.1), (0, 1), (1, 0)]
)
def test_systemone_predictions_use_noul(monkeypatch, tmp_path, provider, noul, confidence):
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", "https://example.test/systemone"),
            json={
                "score": confidence,
                "confidence": confidence,
                "answers": {"score": {"noul": noul, "confidence": confidence}},
                "usage": {"input_tokens": 385},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("benchmark.provider_suite.DATA_DIR", tmp_path)
    (tmp_path / "hatecheck.csv").write_text(
        "sample_id,text,label\n0,Example text,0\n", encoding="utf-8"
    )

    rows = run_single_benchmark(provider, "hatecheck", "v1", 0.5, api_key="test-key")

    assert rows[0]["error"] == ""
    assert rows[0]["score"] == noul
    assert rows[0]["prediction"] == int(noul >= 0.5)


@pytest.mark.parametrize("provider", ["fastino.ai", "laya-ai.com"])
@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"confidence": 0.99},
        {"noul": None, "confidence": 0.99},
        {"noul": True},
        {"noul": "0.2"},
        {"noul": "invalid"},
        {"noul": -0.1},
        {"noul": 1.1},
        {"noul": float("nan")},
        {"noul": float("inf")},
        {"noul": float("-inf")},
    ],
)
def test_systemone_rejects_invalid_noul(monkeypatch, provider, answer):
    class DummyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"answers": {"score": answer}, "confidence": 0.99, "token_usage": 422}

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: DummyResponse())

    with pytest.raises(ValueError, match=r"answers\.score\.noul"):
        call_provider(provider, "Is this unsafe?", "Example text", api_key="test-key")


@pytest.mark.parametrize("provider", ["fastino.ai", "laya-ai.com"])
def test_systemone_missing_noul_is_saved_as_error(monkeypatch, tmp_path, provider):
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", "https://example.test/systemone"),
            json={"answers": {"score": {"confidence": 0.99}}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("benchmark.provider_suite.DATA_DIR", tmp_path)
    monkeypatch.setattr("benchmark.provider_suite.OUTPUT_DIR", tmp_path / "outputs")
    (tmp_path / "hatecheck.csv").write_text(
        "sample_id,text,label\n0,Example text,0\n", encoding="utf-8"
    )

    rows = run_single_benchmark(provider, "hatecheck", "v1", 0.5, api_key="test-key")
    assert rows[0]["score"] is None
    assert rows[0]["prediction"] is None
    assert "answers.score.noul" in rows[0]["error"]

    path = save_rows(rows, provider, "hatecheck", "v1", 0.5)
    with path.open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["score"] == ""
    assert saved[0]["prediction"] == ""
    assert "answers.score.noul" in saved[0]["error"]


@pytest.mark.parametrize("provider", ["fastino.ai", "laya-ai.com"])
@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"answers": None},
        {"answers": []},
        {"answers": {}},
        {"answers": {"score": 0.2}},
        {"answers": {"other": {"noul": 0.2}}},
    ],
)
def test_systemone_rejects_missing_answer(monkeypatch, provider, payload):
    class DummyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: DummyResponse())

    with pytest.raises(ValueError, match=r"answers\.score\.noul"):
        call_provider(provider, "Is this unsafe?", "Example text", api_key="test-key")
