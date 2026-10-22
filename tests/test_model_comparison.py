import csv
import json

import pytest

from benchmark.comparison import (
    DATASETS,
    Prediction,
    classification_metrics,
    collect_report,
    fingerprint,
    parse_logs,
    read_predictions,
    read_reference,
    write_report,
)
from scripts.evaluate_model_suite import summarize_file
from scripts import evaluate_model_suite


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(provider):
    return [
        {
            "dataset": "toxigen", "sample_id": i, "text": f"sample {i}",
            "label": label, "prompt_version": "v1", "threshold": 0.5,
            "score": score, "prediction": int(score >= 0.5), "provider": provider,
            "error": "",
        }
        for i, (label, score) in enumerate(((0, 0.1), (1, 0.9), (1, 0.8)))
    ]


def prepare(tmp_path):
    data, outputs = tmp_path / "data", tmp_path / "outputs"
    reference = [
        {"sample_id": row["sample_id"], "text": row["text"], "label": row["label"]}
        for row in prediction_rows("fastino.ai")
    ]
    for filename in DATASETS.values():
        save_csv(data / filename, reference)
    for provider in ("fastino.ai", "laya-ai.com"):
        save_csv(outputs / provider / "toxigen_v1_t0.5.csv", prediction_rows(provider))
    return data, outputs


def test_shared_metrics_ties_and_single_class():
    rows = [Prediction(0, 1, 0.5), Prediction(1, 1, 0.5)]
    result = classification_metrics(rows)
    assert result["auroc"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["fpr"] == 1
    assert result["fnr"] == 0
    single = classification_metrics([Prediction(1, 1, 0.9)])
    assert single["auroc"] is None
    assert single["fpr"] is None
    assert single["balanced_accuracy"] is None
    assert classification_metrics([])["f1"] is None


def test_existing_evaluator_reuses_metrics_and_handles_errors(tmp_path):
    path = tmp_path / "run.csv"
    rows = prediction_rows("fastino.ai")
    save_csv(path, rows)
    result = summarize_file(path)
    shared = classification_metrics(Prediction(r["label"], r["prediction"], r["score"]) for r in rows)
    for key in ("accuracy", "precision", "recall", "f1", "auroc"):
        assert result[key] == shared[key]
    for row in rows:
        row.update(error="TimeoutError('private details')", prediction="", score="")
    save_csv(path, rows)
    empty = summarize_file(path)
    assert empty["valid"] == 0
    assert empty["f1"] is None
    assert empty["p95_latency_ms"] is None
    save_csv(path, [prediction_rows("fastino.ai")[1]])
    assert summarize_file(path)["auroc"] is None


def test_log_threshold_alignment_partial_progress_and_timestamps(tmp_path):
    path = tmp_path / "run.log"
    path.write_text(
        "[2026-10-19] Running benchmark for fastino.ai | toxigen | v1 | threshold=0.5 | total_rows=100\n"
        "Progress: 50/100 (50%) | provider=fastino.ai | dataset=toxigen | prompt=v1 | elapsed=5.0s\n"
        "Progress: 100/100 (100%) | provider=fastino.ai | dataset=toxigen | prompt=v1 | elapsed=8.0s\n"
        "Running benchmark for fastino.ai | toxigen | v1 | threshold=0.6 | total_rows=100\n"
        "Progress: 50/100 (50%) | provider=fastino.ai | dataset=toxigen | prompt=v1 | elapsed=7.0s\n",
        encoding="utf-8",
    )
    result = parse_logs([path])
    assert result[("fastino.ai", "toxigen", "v1", 0.5)]["elapsed_s"] == 8
    assert result[("fastino.ai", "toxigen", "v1", 0.6)]["processed"] == 50
    path.write_text(
        "Running benchmark for fastino.ai | toxigen | v1 | threshold=0.5 | total_rows=100\n"
        "Progress: 50/100 (50%) | provider=fastino.ai | dataset=hatecheck | prompt=v1 | elapsed=5s\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="interleaved"):
        parse_logs([path])


def test_fair_intersection_missing_progress_and_output(tmp_path):
    data, outputs = prepare(tmp_path)
    path = outputs / "laya-ai.com" / "toxigen_v1_t0.5.csv"
    rows = prediction_rows("laya-ai.com")
    rows[2].update(error="TimeoutError('do not export me')", prediction="", score="")
    save_csv(path, rows)
    (outputs / "laya-ai.com" / "run.log").write_text(
        "Running benchmark for laya-ai.com | toxigen | v1 | threshold=0.5 | total_rows=3\n"
        "Progress: 3/3 (100%) | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed=1.2s\n"
        "Running benchmark for laya-ai.com | toxigen | v1 | threshold=0.6 | total_rows=3\n"
        "Progress: 1/3 (33%) | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed=0.5s\n",
        encoding="utf-8",
    )
    hidden = outputs / "laya-ai.com" / ".ipynb_checkpoints"
    hidden.mkdir()
    (hidden / "invalid.csv").write_text("ignored")
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    assert report["baseline_datasets"] == ["toxigen"]
    assert len(report["paired"]) == 2
    assert all(row["n"] == 2 for row in report["paired"])
    assert len(report["pairwise"]) == 2
    assert all(row["n"] == 2 for row in report["pairwise"])
    own = next(row for row in report["runs"] if row["file"] == str(path))
    assert own["valid"] == 2
    assert own["mean_elapsed_ms"] == 400
    assert own["error_rate"] == pytest.approx(1 / 3)
    pending = next(row for row in report["runs"] if row["provider"] == "laya-ai.com"
                   and row["dataset"] == "toxigen" and row["prompt"] == "v1" and row["threshold"] == 0.6)
    assert pending["status"] == "in_progress"
    assert pending["mean_elapsed_ms"] is None
    destination = tmp_path / "report"
    write_report(report, destination)
    exported = json.loads((destination / "comparison.json").read_text())
    assert exported["conclusions"]
    assert exported["model_identities"]["fastino.ai"]["resolved_model"] == "glide_v2"
    assert exported["model_identities"]["laya-ai.com"]["serving_model"] == "laya-rl-agent"
    assert exported["model_identities"]["laya-ai.com"]["routed_model"] == "typed-decisions"
    assert "do not export me" not in (destination / "comparison.json").read_text()
    document = (destination / "comparison.html").read_text()
    assert "fastino/GLiDE" in document
    assert "typed-decisions" in document
    assert "resolved model=glide_v2" in document
    assert "serving model=laya-rl-agent" in document
    assert "details-table" in document
    assert "https://" not in document


def test_partial_files_not_ranked_and_corrupt_rows_rejected(tmp_path):
    data, outputs = prepare(tmp_path)
    path = outputs / "laya-ai.com" / "toxigen_v1_t0.5.csv"
    rows = prediction_rows("laya-ai.com")[:2]
    save_csv(path, rows)
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    assert report["paired"] == []
    assert all(row["macro_f1"] is None for row in report["overview"])
    reference = read_reference(data / "toxigen.csv")
    rows[0]["text"] = "modified"
    save_csv(path, rows)
    with pytest.raises(ValueError, match="mismatch"):
        read_predictions(path, reference, "laya-ai.com", "toxigen", "v1", 0.5)
    rows = prediction_rows("laya-ai.com")
    rows[0]["score"] = float("nan")
    save_csv(path, rows)
    with pytest.raises(ValueError, match="invalid score"):
        read_predictions(path, reference, "laya-ai.com", "toxigen", "v1", 0.5)
    save_csv(path, [rows[1], rows[1]])
    with pytest.raises(ValueError, match="duplicate"):
        read_predictions(path, reference, "laya-ai.com", "toxigen", "v1", 0.5)


def test_reference_matches_original_runner_parser_and_platform_newlines(tmp_path):
    path = tmp_path / "reference.csv"
    save_csv(path, [
        {"sample_id": 0, "text": "prefix\x00truncated", "label": 1},
        {"sample_id": 1, "text": "", "label": 0},
    ])
    reference = read_reference(path)
    assert reference["0"][1] == fingerprint("prefix")
    assert reference["1"][1] == fingerprint("nan")
    assert fingerprint("one\r\ntwo\rthree") == fingerprint("one\ntwo\nthree")


def test_original_suite_discovery_ignores_generated_report_csvs(tmp_path, monkeypatch):
    data, outputs = prepare(tmp_path)
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    write_report(report, outputs / "comparison")
    monkeypatch.setattr(evaluate_model_suite, "OUTPUT_DIR", outputs)
    discovered = evaluate_model_suite.iter_csv_files()
    assert len(discovered) == 2
    assert all(path.name == "toxigen_v1_t0.5.csv" for path in discovered)


def test_pairwise_comparison_uses_more_completed_datasets(tmp_path):
    data, outputs = prepare(tmp_path)
    for provider in ("fastino.ai", "typesafe.ai"):
        save_csv(outputs / provider / "toxigen_v1_t0.5.csv", prediction_rows(provider))
        rows = prediction_rows(provider)
        for row in rows:
            row["dataset"] = "hatecheck"
        save_csv(outputs / provider / "hatecheck_v1_t0.5.csv", rows)
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com", "typesafe.ai"])
    assert report["baseline_datasets"] == ["toxigen"]
    extended = [row for row in report["pairwise"] if row["pair"] == "fastino.ai vs typesafe.ai"]
    assert {row["dataset"] for row in extended} == {"toxigen", "hatecheck"}
    assert len(extended) == 4
    write_report(report, tmp_path / "report")
    exported = json.loads((tmp_path / "report" / "comparison.json").read_text())
    assert exported["model_identities"]["typesafe.ai"] == {
        "request_model": "jev-latest", "resolved_model": None,
    }
    document = (tmp_path / "report" / "comparison.html").read_text()
    assert "request model=jev-latest；resolved model=待接口返回确认" in document
    assert "typesafe-standard" not in document
