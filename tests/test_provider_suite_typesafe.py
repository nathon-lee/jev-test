import httpx

from benchmark.provider_suite import call_provider


def test_typesafe_provider_uses_http_systemone(monkeypatch):
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
                    "model": "jev",
                    "answers": {
                        "score": {
                            "type": "noul",
                            "noul": 0.91,
                        }
                    },
                }

        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    score = call_provider(
        provider_name="typesafe.ai",
        prompt="Does this contain harmful text?",
        text="This is a test",
        api_key="demo-key",
        base_url="https://api.typesafe.ai",
    )

    assert score == 0.91
    assert captured["url"].endswith("/v1/systemone")
    assert captured["json"]["state"] == {"user_input": "This is a test"}
    assert "score" in captured["json"]["questions"]
    assert "Authorization" in captured["headers"]
