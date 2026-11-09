import importlib.util
import csv
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

for dependency in ("dotenv", "httpx", "pyarrow", "sklearn", "truststore"):
    pytest.importorskip(dependency, reason="Install demo-preset-benchmark/requirements-test.txt")

MODULE_PATH = Path(__file__).parents[1] / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("demo_preset_benchmark", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_binary_metrics_include_confusion_counts():
    rows = [
        {"gold": 1, "prediction": 1},
        {"gold": 1, "prediction": 0},
        {"gold": 0, "prediction": 1},
        {"gold": 0, "prediction": 0},
    ]

    result = benchmark.binary(rows)

    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0.5


def test_multilabel_metrics_cover_primary_pii_categories():
    rows = [
        {"gold": ["email"], "prediction": ["email", "ssn"]},
        {"gold": ["phone_number"], "prediction": ["phone_number"]},
    ]

    result = benchmark.multilabel(rows)

    assert set(result["per_class"]) == set(benchmark.PII_CLASSES)
    assert result["micro"]["tp"] == 2
    assert result["micro"]["fp"] == 1
    assert result["micro"]["fn"] == 0
    assert result["exact_match_accuracy"] == 0.5


def test_failed_calls_are_not_counted_as_zero_quality():
    result = benchmark.summarize(
        "pi",
        [{"error": "service unavailable", "gold": 1, "prediction": None}],
        requested=1,
    )

    assert result["successful_rows"] == 0
    assert result["failed_rows"] == 1
    assert result["coverage"] == 0
    assert "f1" not in result
    assert result["mean_latency_ms"] is None


@pytest.mark.parametrize("dataset, split, expected_splits", [
    ("mini", "validation", ["validation"]),
    ("nano", "validation", ["validation"]),
    ("nano", "train", ["train"]),
    ("nano", "all", ["train", "validation"]),
])
def test_pii_dataset_selection(tmp_path, monkeypatch, dataset, split, expected_splits):
    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    calls = []

    def download(client, cache, repo, revision, filename):
        calls.append((repo, revision, filename))
        path = tmp_path / Path(filename).name
        path.write_text(json.dumps({
            "uid": filename, "source_text": "unchanged text",
            "privacy_mask": [{"label": "EMAIL"}, {"label": "SURNAME"}],
            "language": "en",
        }) + "\n", encoding="utf-8")
        return path

    monkeypatch.setattr(benchmark.httpx, "Client", Client)
    monkeypatch.setattr(benchmark.truststore, "SSLContext", lambda *args: None)
    monkeypatch.setattr(benchmark, "download", download)
    sources, provenance = benchmark.load_sources(tmp_path, 1000, 42, "pii", dataset, split)
    repo = f"ai4privacy/pii-masking-{'nano-1k' if dataset == 'nano' else 'mini-10k'}"
    revision = benchmark.PII_NANO_REVISION if dataset == "nano" else benchmark.PII_REVISION
    assert calls == [(repo, revision, f"data/{value}.jsonl") for value in expected_splits]
    assert len(sources["pii"]) == len(expected_splits)
    assert all(row["gold"] == ["email"] for row in sources["pii"])
    assert provenance["pii"]["dataset"] == repo
    assert provenance["pii"]["split"] == split


def write_prepared(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "text", "label"])
        writer.writerows(rows)


def test_prepared_preserves_samples_and_applies_limit(tmp_path):
    write_prepared(tmp_path / "detect_jailbreak.csv",
                   [["42", "original\ntext\u2028unchanged", "1"], ["43", "safe", "0"]])
    sources, provenance, presets = benchmark.load_prepared(
        tmp_path, "detect-jailbreak", {"pi": "prompt_injection_luna"}, limit=1)
    row = sources["detect_jailbreak"][0]
    assert row["input"] == "original\ntext\u2028unchanged"
    assert row["sample_id"] == row["source_index"] == "42"
    assert row["gold"] == 1
    assert provenance["detect_jailbreak"]["total_rows"] == 2
    assert presets == {"detect_jailbreak": "prompt_injection_luna"}


@pytest.mark.parametrize("dataset", ["toxigen", "hatecheck", "aegis2"])
def test_prepared_requires_matching_preset(tmp_path, dataset):
    with pytest.raises(ValueError, match="requires"):
        benchmark.load_prepared(tmp_path, dataset, {"pi": "prompt_injection_luna"})


@pytest.mark.parametrize("rows, message", [
    ([["1", "text", "2"]], "invalid sample"),
    ([["1", "text", "0"], ["1", "other", "1"]], "duplicate sample_id"),
    ([], "dataset is empty"),
])
def test_prepared_rejects_invalid_data(tmp_path, rows, message):
    write_prepared(tmp_path / "detect_jailbreak.csv", rows)
    with pytest.raises(ValueError, match=message):
        benchmark.load_prepared(tmp_path, "detect_jailbreak", {"pi": "prompt_injection_luna"})


def test_local_pii_is_not_substituted(tmp_path):
    with pytest.raises(ValueError, match="original PII validation"):
        benchmark.load_prepared(tmp_path, "pii_masking_mini_10k", {})


def test_dataset_threads_are_parallel_but_rows_are_sequential(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    calls = {}

    class FakeEvaluator:
        def __init__(self, api, info, metric):
            self.metric = metric

        def evaluate(self, row):
            calls.setdefault(self.metric, []).append(threading.get_ident())
            if row["index"] == 0:
                barrier.wait(timeout=5)
            return {**row, "score": 0.9, "prediction": 1, "error": None,
                    "scorer_version_id": "v1"}

    monkeypatch.setattr(benchmark, "Evaluator", FakeEvaluator)
    sources = {
        metric: [{"index": i, "source_index": str(i), "input": f"text {i}\u2028",
                  "gold": 1} for i in range(3)]
        for metric in ("toxigen", "hatecheck")
    }
    state = {"metrics": {metric: {"scorer_version_id": "v1"} for metric in sources}}
    api = SimpleNamespace(key="test-key")
    result = benchmark.run(api, state, sources, tmp_path, 4, False, dataset_workers=2)
    assert set(result["metrics"]) == set(sources)
    assert all(len(ids) == 3 and len(set(ids)) == 1 for ids in calls.values())
    assert calls["toxigen"][0] != calls["hatecheck"][0]
    for metric in sources:
        path = tmp_path / metric / f"{metric}-predictions.jsonl"
        with path.open(encoding="utf-8") as handle:
            assert len([json.loads(line) for line in handle]) == 3

    def unexpected_call(*args):
        raise AssertionError("Completed rows must not be requested again")

    monkeypatch.setattr(FakeEvaluator, "evaluate", unexpected_call)
    resumed = benchmark.run(api, state, sources, tmp_path, 4, False, dataset_workers=2)
    assert resumed["metrics"] == result["metrics"]


def test_concurrent_failure_keeps_other_results(tmp_path, monkeypatch):
    class FakeEvaluator:
        def __init__(self, api, info, metric):
            self.metric = metric

        def evaluate(self, row):
            error = "unavailable" if self.metric == "toxigen" else None
            return {**row, "score": 0.9, "prediction": 1, "error": error,
                    "scorer_version_id": "v1"}

    monkeypatch.setattr(benchmark, "Evaluator", FakeEvaluator)
    sources = {metric: [{"index": 0, "source_index": "0", "input": "text", "gold": 1}]
               for metric in ("toxigen", "hatecheck")}
    state = {"metrics": {metric: {"scorer_version_id": "v1"} for metric in sources}}
    with pytest.raises(RuntimeError, match="Dataset failures: toxigen"):
        benchmark.run(SimpleNamespace(key="test-key"), state, sources, tmp_path,
                      4, False, dataset_workers=2)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["metrics"]["hatecheck"]["successful_rows"] == 1
    assert "toxigen" in summary["dataset_errors"]


@pytest.mark.parametrize("metric, score, prediction", [
    ("toxigen", 0.75, 1),
    ("hatecheck", 0.25, 0),
    ("pii", ["email", "name"], ["email"]),
])
def test_evaluator_binary_tasks_and_original_pii(metric, score, prediction):
    info = {"agent_name": "agent", "stream_id": "stream", "scorer_name": "preset",
            "scorer_id": "scorer", "scorer_version_id": "v1", "control_id": "control"}
    execution = {"control_id": "control", "result": {"metadata": {
        "scorer_id": "scorer", "requested_scorer_version_id": "v1",
        "status": "success", "score": score}}}
    api = SimpleNamespace(key="test-key", request=lambda *args, **kwargs: {"matches": [execution]})
    evaluator = benchmark.Evaluator(api, info, metric)
    evaluator.token, evaluator.expires = "test-token", float("inf")
    result = evaluator.evaluate({"index": 0, "input": "text", "gold": 1})
    assert result["error"] is None
    assert result["score"] == score
    assert result["prediction"] == prediction
    assert result["latency_seconds"] >= 0


def test_summary_averages_successful_call_latency_only():
    result = benchmark.summarize("toxigen", [
        {"error": None, "gold": 1, "prediction": 1, "score": 0.9,
         "latency_seconds": 0.1},
        {"error": None, "gold": 0, "prediction": 0, "score": 0.1,
         "latency_seconds": 0.3},
        {"error": "failed", "gold": 1, "prediction": None, "score": None,
         "latency_seconds": 10.0},
        {"error": None, "gold": 1, "prediction": 1, "score": 0.9},
    ], requested=4)

    assert result["latency_sample_count"] == 2
    assert result["mean_latency_seconds"] == pytest.approx(0.2)
    assert result["mean_latency_ms"] == pytest.approx(200)


def test_existing_checkpoint_rebuilds_latency_summary_without_api_call(tmp_path, monkeypatch):
    row = {"index": 0, "source_index": "0", "input": "text", "gold": 1}
    record = {**row, "score": 0.9, "prediction": 1, "error": None,
              "scorer_version_id": "v1", "latency_seconds": 0.125}
    (tmp_path / "toxigen-predictions.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8")

    class UnexpectedEvaluator:
        def __init__(self, *args):
            pass

        def evaluate(self, *args):
            raise AssertionError("A completed sample should be resumed from its checkpoint")

    monkeypatch.setattr(benchmark, "Evaluator", UnexpectedEvaluator)
    result = benchmark.run(
        SimpleNamespace(key="test-key"),
        {"metrics": {"toxigen": {"scorer_version_id": "v1"}}},
        {"toxigen": [row]}, tmp_path, 1, False,
    )
    assert result["metrics"]["toxigen"]["mean_latency_seconds"] == pytest.approx(0.125)
    assert result["metrics"]["toxigen"]["mean_latency_ms"] == pytest.approx(125)


def test_retry_errors_replaces_latest_checkpoint(tmp_path, monkeypatch):
    row = {"index": 0, "source_index": "0", "input": "text", "gold": 1}
    previous = {**row, "score": None, "prediction": None,
                "error": "temporary failure", "scorer_version_id": "v1"}
    (tmp_path / "toxigen-predictions.jsonl").write_text(
        json.dumps(previous) + "\n", encoding="utf-8")

    class FakeEvaluator:
        def __init__(self, *args):
            pass

        def evaluate(self, sample):
            return {**sample, "score": 0.9, "prediction": 1, "error": None,
                    "scorer_version_id": "v1"}

    monkeypatch.setattr(benchmark, "Evaluator", FakeEvaluator)
    result = benchmark.run(SimpleNamespace(key="test-key"),
                           {"metrics": {"toxigen": {"scorer_version_id": "v1"}}},
                           {"toxigen": [row]}, tmp_path, 1, True)
    assert result["metrics"]["toxigen"]["successful_rows"] == 1
    assert result["metrics"]["toxigen"]["failed_rows"] == 0
