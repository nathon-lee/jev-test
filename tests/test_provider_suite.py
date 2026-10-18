import httpx

from benchmark.provider_suite import (
    DEFAULT_THRESHOLDS,
    available_datasets,
    call_provider,
    prompt_for_dataset,
    resolve_provider,
    normalize_provider_name,
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
                    "answers": {
                        "score": {"score": 0.91},
                    }
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

    assert score == 0.91
    assert captured["url"].endswith("/v1/systemone")
    assert "Authorization" in captured["headers"]
    assert "state" in captured["json"]
    assert "questions" in captured["json"]
