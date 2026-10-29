from benchmark.provider_suite import call_provider


def test_typesafe_provider_uses_sdk(monkeypatch):
    calls = {}

    class FakeNoul:
        def __init__(self, instructions):
            self.instructions = instructions

    class FakeTypeSafeClient:
        def __init__(self, **kwargs):
            calls["client_init"] = kwargs

        def system_one(self, **kwargs):
            calls["system_one"] = kwargs
            return {"prediction": 1, "score": 0.91}

    monkeypatch.setattr("benchmark.provider_suite.TypeSafeClient", FakeTypeSafeClient, raising=False)
    monkeypatch.setattr("benchmark.provider_suite.Noul", FakeNoul, raising=False)

    def fail_http_post(*args, **kwargs):
        raise AssertionError("typesafe.ai should not hit /chat/completions")

    monkeypatch.setattr("benchmark.provider_suite.httpx.post", fail_http_post)

    score = call_provider(
        provider_name="typesafe.ai",
        prompt="Does this contain harmful text?",
        text="This is a test",
        api_key="demo-key",
        base_url="https://api.typesafe.ai",
    )

    assert score == 0.91
    assert calls["client_init"]["model"] == "jev-latest"
    assert "system_one" in calls
    assert calls["system_one"]["state"] == {"user_input": "This is a test"}
    assert "score" in calls["system_one"]["questions"]
