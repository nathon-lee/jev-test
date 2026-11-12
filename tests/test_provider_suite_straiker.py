import csv
import json
from datetime import datetime, timezone
from unittest.mock import Mock

import httpx
import pytest

from benchmark.provider_suite import call_provider, run_single_benchmark, save_rows
from benchmark.straiker_client import _retry_delay, detect_straiker
from scripts.evaluate_model_suite import summarize_file
from scripts.run_model_suite import provider_list
from scripts.run_model_suite import main as run_suite


KEY = "sk_agt_test_key"
BASE = "https://straiker.example.test"


def response(payload, status=200, headers=None):
    return httpx.Response(
        status, json=payload, headers=headers,
        request=httpx.Request("POST", BASE + "/api/v3/detect"),
    )


@pytest.mark.parametrize("base", [BASE, BASE + "/", BASE + "/api/v3/detect"])
@pytest.mark.parametrize("action,score", [("allow", 0.0), ("block", 1.0)])
def test_straiker_sends_original_content_and_maps_only_action(monkeypatch, capsys, base, action, score):
    post = Mock(return_value=response({"action": action, "score": 1 - score}))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("STRAIKER_INTEGRATION_KEY", KEY)
    monkeypatch.setenv("STRAIKER_BASE", base)
    text = 'Original text\nwith "quotes" and \u4e2d\u6587'

    assert call_provider("straiker", "This prompt must not be sent", text, debug=True) == score

    post.assert_called_once_with(
        BASE + "/api/v3/detect",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        json={"content": text, "role": "user"},
        timeout=60.0,
    )
    log = capsys.readouterr().out
    assert KEY not in log
    assert "Original text" not in log
    assert "straiker.ai" in provider_list("all")


@pytest.mark.parametrize("key", [None, "", "s6r_pat_test", "eyJ.jwt.test", KEY + "\n"])
def test_straiker_rejects_non_integration_credentials_without_request(monkeypatch, key):
    post = Mock()
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ValueError, match="integration key"):
        detect_straiker("Example", key, BASE)
    post.assert_not_called()


def test_straiker_requires_explicit_tenant_region(monkeypatch):
    post = Mock()
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.delenv("STRAIKER_BASE", raising=False)
    with pytest.raises(ValueError, match="STRAIKER_BASE"):
        call_provider("straiker", "question", "Example", api_key=KEY)
    post.assert_not_called()


@pytest.mark.parametrize("payload", [
    {}, {"action": "unknown"}, {"action": None}, {"action": True},
    {"score": 0.9}, [{"action": "block"}],
])
def test_straiker_does_not_infer_verdict_from_malformed_response(monkeypatch, payload):
    post = Mock(return_value=response(payload))
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ValueError, match="action"):
        detect_straiker("Example", KEY, BASE)
    assert post.call_count == 1


@pytest.mark.parametrize("first", [
    response({}, 429, {"Retry-After": "2.5"}),
    response({}, 503),
    httpx.ConnectError("Temporary failure"),
])
def test_straiker_retries_temporary_failures(monkeypatch, first):
    post = Mock(side_effect=[first, response({"action": "block"})])
    sleep = Mock()
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("benchmark.straiker_client.time.sleep", sleep)

    assert detect_straiker("Example", KEY, BASE)["action"] == "block"
    assert post.call_count == 2
    expected = 2.5 if isinstance(first, httpx.Response) and first.status_code == 429 else 1.0
    sleep.assert_called_once_with(expected)


@pytest.mark.parametrize("status,attempts", [(401, 1), (404, 1), (429, 4), (503, 4)])
def test_straiker_surfaces_http_failures_and_bounds_retries(monkeypatch, status, attempts):
    post = Mock(return_value=response({}, status))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("benchmark.straiker_client.time.sleep", Mock())
    with pytest.raises(httpx.HTTPStatusError):
        detect_straiker("Example", KEY, BASE)
    assert post.call_count == attempts


def test_straiker_surfaces_exhausted_network_failures(monkeypatch):
    post = Mock(side_effect=httpx.ReadTimeout("Temporary failure"))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("benchmark.straiker_client.time.sleep", Mock())
    with pytest.raises(httpx.ReadTimeout):
        detect_straiker("Example", KEY, BASE)
    assert post.call_count == 4


def test_retry_after_accepts_http_date_and_invalid_header_uses_backoff(monkeypatch):
    clock = Mock()
    clock.now.return_value = datetime(2026, 11, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("benchmark.straiker_client.datetime", clock)
    assert _retry_delay("Thu, 12 Nov 2026 00:00:10 GMT", 0) == 10.0
    assert _retry_delay("invalid", 2) == 4.0
    assert _retry_delay("nan", 1) == 2.0


def test_straiker_local_dataset_output_and_existing_evaluator(monkeypatch, tmp_path):
    monkeypatch.setattr("benchmark.provider_suite.DATA_DIR", tmp_path)
    with (tmp_path / "hatecheck.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "text", "label"])
        writer.writerows([["safe-id", "Hello\nworld", 0], ["unsafe-id", "Test input", 1],
                          ["error-id", "Failed request", 1]])
    verdicts = [
        {"action": "allow", "controls": [], "blocked_by": [], "config_hash": "config-1"},
        {"action": "block", "controls": ["hate"], "blocked_by": ["hate"],
         "config_hash": "config-1", "turn_id": "turn-1", "trace_id": "trace-1"},
    ]
    post = Mock(side_effect=[response(item) for item in verdicts] + [response({}, 401)])
    monkeypatch.setattr(httpx, "post", post)

    rows = run_single_benchmark("straiker.ai", "hatecheck", "v1", 0.5,
                                api_key=KEY, base_url=BASE)
    assert [row["sample_id"] for row in rows] == ["safe-id", "unsafe-id", "error-id"]
    assert [row["prediction"] for row in rows] == [0, 1, None]
    assert [row["score"] for row in rows] == [0.0, 1.0, None]
    assert rows[0]["text"] == "Hello\nworld"
    assert json.loads(rows[1]["straiker_response"]) == verdicts[1]
    assert rows[1]["score_kind"] == "binary_action"
    assert "401" in rows[2]["error"]

    path = save_rows(rows, "straiker.ai", "hatecheck", "v1", 0.5, output_dir=tmp_path)
    summary = summarize_file(path)
    assert (summary["rows"], summary["valid"], summary["error"]) == (3, 2, 1)
    assert summary["accuracy"] == summary["precision"] == summary["recall"] == summary["f1"] == 1.0


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("key,kind", [
    ("", "missing"), ("s6r_pat_test_private", "PAT"),
    ("eyJ.private.jwt", "JWT"), (KEY + "\n", "whitespace"),
])
def test_suite_straiker_preflight_fails_without_reading_data_or_exposing_key(
    monkeypatch, tmp_path, capsys, workers, key, kind,
):
    post = Mock()
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("STRAIKER_INTEGRATION_KEY", key)
    monkeypatch.setenv("STRAIKER_BASE", BASE)
    monkeypatch.setattr("benchmark.provider_suite.DATA_DIR", tmp_path / "missing-data")
    monkeypatch.setattr("sys.argv", [
        "run_model_suite", "--providers", "straiker", "--datasets", "hatecheck",
        "--prompt-versions", "v1", "--thresholds", "0.5",
        "--max-workers", str(workers), "--output-dir", str(tmp_path / "output"),
    ])

    with pytest.raises(SystemExit) as exc:
        run_suite()
    assert exc.value.code == 1
    log = capsys.readouterr().out
    assert f"credential_type={kind}" in log
    assert "All benchmark runs are complete." not in log
    if key:
        assert key not in log
    assert not (tmp_path / "output").exists()
    post.assert_not_called()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("status", [200, 401, 403, 404])
def test_suite_reports_sample_counts_and_fails_for_api_errors(
    monkeypatch, tmp_path, capsys, workers, status,
):
    private_text = "private-sample-content"
    monkeypatch.setattr("benchmark.provider_suite.DATA_DIR", tmp_path)
    (tmp_path / "hatecheck.csv").write_text(
        f"sample_id,text,label\n0,first-private-sample,0\n1,{private_text},0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STRAIKER_INTEGRATION_KEY", KEY)
    monkeypatch.setenv("STRAIKER_BASE", BASE)
    monkeypatch.setattr(httpx, "post", Mock(side_effect=[
        response({"action": "allow"}),
        response({"action": "allow", "private": KEY}, status),
    ]))
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", [
        "run_model_suite", "--providers", "straiker", "--datasets", "hatecheck",
        "--prompt-versions", "v1", "--thresholds", "0.5", "--debug",
        "--max-workers", str(workers), "--output-dir", str(output),
    ])
    if status == 200:
        run_suite()
    else:
        with pytest.raises(SystemExit) as exc:
            run_suite()
        assert exc.value.code == 1
    log = capsys.readouterr().out
    assert "credential_type=integration_key" in log
    assert f"HTTP {status}" in log
    assert KEY not in log
    assert private_text not in log
    assert "first-private-sample" not in log
    assert ("All benchmark runs are complete." in log) == (status == 200)
    assert f"valid={1 + int(status == 200)}" in log
    assert f"error={int(status != 200)}" in log
    with (output / "straiker.ai" / "hatecheck_v1_t0.5.csv").open(newline="") as file:
        first_row, row = list(csv.DictReader(file))
    assert first_row["error"] == ""
    assert float(first_row["prediction"]) == 0.0
    assert bool(row["error"]) == (status != 200)
    assert KEY not in row["error"]


def test_straiker_transport_error_details_cannot_leak_into_logs_or_error_column(
    monkeypatch, tmp_path, capsys,
):
    monkeypatch.setattr("benchmark.provider_suite.DATA_DIR", tmp_path)
    (tmp_path / "hatecheck.csv").write_text(
        "sample_id,text,label\n0,private-sample-content,0\n", encoding="utf-8",
    )
    monkeypatch.setattr(httpx, "post", Mock(side_effect=httpx.ConnectError(KEY)))
    monkeypatch.setattr("benchmark.straiker_client.time.sleep", Mock())
    rows = run_single_benchmark(
        "straiker", "hatecheck", "v1", 0.5, api_key=KEY, base_url=BASE, debug=True,
    )
    assert "ConnectError" in rows[0]["error"]
    assert KEY not in rows[0]["error"]
    log = capsys.readouterr().out
    assert KEY not in log
    assert "private-sample-content" not in log


@pytest.mark.parametrize("base", [
    f"https://user:{KEY}@straiker.example.test",
    f"{BASE}?api_key={KEY}",
    f"{BASE}#{KEY}",
])
def test_straiker_rejects_secret_bearing_urls_without_logging_them(monkeypatch, capsys, base):
    post = Mock()
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ValueError, match="STRAIKER_BASE") as exc:
        detect_straiker("Example", KEY, base, debug=True)
    assert KEY not in str(exc.value)
    assert KEY not in capsys.readouterr().out
    post.assert_not_called()
