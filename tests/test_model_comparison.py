import csv
import json
import os
from hashlib import sha256

import pytest

from benchmark.comparison import (
    DATASETS,
    THRESHOLDS,
    Prediction,
    classification_metrics,
    collect_report,
    fingerprint,
    parse_logs,
    parse_repeat_log_timings,
    read_predictions,
    read_reference,
    stability_metrics,
    write_report,
)
from scripts.evaluate_model_suite import summarize_file
from scripts import evaluate_model_suite
from benchmark import provider_suite
from benchmark.external_summary import (
    FOCUSED_DATASETS,
    METRICS,
    attach_openjev_summary,
    read_openjev_summary,
    read_internal_summary,
    reference_averages,
)


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


def save_repeat_runs(outputs, provider, dataset, prompt, scores, failed=()):
    for repeat, values in enumerate(scores):
        rows = prediction_rows(provider)[:len(values)]
        for sample, (row, score) in enumerate(zip(rows, values)):
            row.update(
                dataset=dataset, prompt_version=prompt, threshold=THRESHOLDS[repeat],
                score=score, prediction=int(score >= THRESHOLDS[repeat]),
            )
            if (repeat, sample) in failed:
                row.update(error="TimeoutError('do not export repeat details')", score="", prediction="")
        save_csv(outputs / provider / f"{dataset}_{prompt}_t{THRESHOLDS[repeat]}.csv", rows)


def test_incremental_preserves_history_and_fills_full_sample_comparisons(tmp_path):
    data, outputs = prepare(tmp_path)
    providers = ["fastino.ai", "laya-ai.com"]
    rows = prediction_rows("fastino.ai")
    for row in rows:
        row["dataset"] = "hatecheck"
    save_csv(outputs / "fastino.ai/hatecheck_v1_t0.5.csv", rows)
    previous = collect_report(outputs, data, providers)
    for path in outputs.glob("*/*.csv"):
        path.unlink()
    for row in rows:
        row["provider"] = "laya-ai.com"
    save_csv(outputs / "laya-ai.com/hatecheck_v1_t0.5.csv", rows)
    updated = collect_report(outputs, data, providers, previous_report=previous)
    assert updated["previous_generated_at"] == previous["generated_at"]
    assert updated["baseline_datasets"] == ["hatecheck", "toxigen"]
    assert len(updated["runs"]) == len(previous["runs"])
    old = next(r for r in previous["runs"] if r["file"] and r["dataset"] == "toxigen")
    retained = next(r for r in updated["runs"] if r["file"] == old["file"])
    assert retained == {**old, "source_kind": "historical_summary"}
    assert [r for r in updated["paired"] if r["dataset"] == "toxigen"] == previous["paired"]
    added = [r for r in updated["paired"] if r["dataset"] == "hatecheck"]
    assert len(added) == 2 and all(r["n"] == 3 and r["f1"] == 1 for r in added)
    assert all(r["complete"] == 2 for r in updated["overview"])
    assert any("增量更新" in warning for warning in updated["warnings"])
    write_report(updated, tmp_path / "report")
    assert "增量更新" in (tmp_path / "report/comparison.md").read_text()


def test_incremental_replacement_invalidates_stale_intersections_and_timing(tmp_path):
    data, outputs = prepare(tmp_path)
    providers = ["fastino.ai", "laya-ai.com"]
    for provider in providers:
        save_repeat_runs(outputs, provider, "toxigen", "v1", [[0.2, 0.8, 0.8]] * 4)
    previous = collect_report(outputs, data, providers)
    for row in previous["runs"]:
        if row["file"]:
            row.update(mean_elapsed_ms=123, elapsed_s=0.369)
    for path in outputs.glob("*/*.csv"):
        path.unlink()
    rows = prediction_rows("laya-ai.com")
    rows[0].update(error="TimeoutError('private details')", score="", prediction="")
    save_csv(outputs / "laya-ai.com/toxigen_v1_t0.5.csv", rows)
    updated = collect_report(outputs, data, providers, previous_report=previous)
    assert updated["baseline_datasets"] == []
    assert not any(r["threshold"] == 0.5 for r in updated["paired"])
    assert updated["pairwise"] == []
    fresh = next(r for r in updated["runs"] if r.get("source_kind") == "local_predictions")
    assert fresh["valid"] == 2 and fresh["errors"] == 1
    assert fresh["mean_elapsed_ms"] is fresh["elapsed_s"] is None
    assert updated["stability_paired"] == []
    stability = next(r for r in updated["stability"] if r["provider"] == "laya-ai.com"
                     and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert stability["complete_repeats"] == 4
    assert stability["status"] == "unavailable_raw"
    assert stability["mean_score_std"] is stability["coverage"] is None
    assert stability["errors"] == 1
    assert any("无法重新对齐" in warning for warning in updated["warnings"])
    assert "private details" not in json.dumps(updated)


def test_incremental_uses_raw_intersections_when_all_predictions_are_available(tmp_path):
    data, outputs = prepare(tmp_path)
    providers = ["fastino.ai", "laya-ai.com"]
    previous = collect_report(outputs, data, providers)
    rows = prediction_rows("laya-ai.com")
    rows[0].update(error="TimeoutError", score="", prediction="")
    save_csv(outputs / "laya-ai.com/toxigen_v1_t0.5.csv", rows)
    updated = collect_report(outputs, data, providers, previous_report=previous)
    assert updated["baseline_datasets"] == ["toxigen"]
    assert all(r["n"] == 2 for r in updated["paired"])
    assert all(r["common_coverage"] == pytest.approx(2 / 3) for r in updated["paired"])
    previous["providers"] = list(reversed(providers))
    with pytest.raises(ValueError, match="providers/model identities"):
        collect_report(outputs, data, providers, previous_report=previous)


def test_stability_compares_scores_and_uses_one_inclusive_decision_threshold():
    frames = []
    for threshold, score in zip(THRESHOLDS, (0.49, 0.5, 0.51, 0.5)):
        frames.append({
            "steady": Prediction(1, int(0.65 >= threshold), 0.65),
            "boundary": Prediction(1, int(0.5 >= threshold), 0.5),
            "flips": Prediction(1, int(score >= threshold), score),
        })
    result = stability_metrics(frames, frames[0])
    assert result["n"] == 3
    assert result["flipped"] == 1
    assert result["decision_flip_rate"] == pytest.approx(1 / 3)
    assert result["mean_score_std"] == pytest.approx((0.00005 ** 0.5) / 3)
    assert result["mean_score_range"] == pytest.approx(0.02 / 3)
    steady = stability_metrics(frames, ["steady", "boundary"])
    assert steady["mean_score_std"] == steady["mean_score_range"] == steady["decision_flip_rate"] == 0
    empty = stability_metrics(frames, [])
    assert empty["n"] == 0
    assert all(empty[key] is None for key in ("mean_score_std", "mean_score_range", "decision_flip_rate"))
    with pytest.raises(ValueError, match="four repeat"):
        stability_metrics(frames[:3], ["steady"])


def test_stability_aligns_all_successes_separates_prompts_and_exports(tmp_path):
    data, outputs = prepare(tmp_path)
    for provider in ("fastino.ai", "laya-ai.com"):
        scores = [[0.4, 0.7, 0.1], [0.6, 0.7, 0.9], [0.4, 0.7, 0.1], [0.6, 0.7, 0.9]]
        save_repeat_runs(
            outputs, provider, "toxigen", "v1", scores,
            failed={(1, 2)} if provider == "laya-ai.com" else (),
        )
        save_repeat_runs(outputs, provider, "toxigen", "v2", [[0.2, 0.8, 0.8]] * 4)
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    fast = next(r for r in report["stability"] if r["provider"] == "fastino.ai"
                and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    laya = next(r for r in report["stability"] if r["provider"] == "laya-ai.com"
                and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert fast["n"] == 3
    assert fast["decision_flip_rate"] == pytest.approx(2 / 3)
    assert laya["n"] == 2
    assert laya["coverage"] == pytest.approx(2 / 3)
    assert laya["rows"] == 12
    assert laya["errors"] == 1
    assert laya["error_rate"] == pytest.approx(1 / 12)
    paired = [r for r in report["stability_paired"] if r["prompt"] == "v1"]
    assert len(paired) == 2
    assert all(r["n"] == 2 for r in paired)
    assert all(r["decision_flip_rate"] == 0.5 for r in paired)
    assert all(r["mean_score_std"] == pytest.approx(0.05) for r in paired)
    assert all(r["mean_score_range"] == pytest.approx(0.1) for r in paired)
    v2 = [r for r in report["stability_paired"] if r["prompt"] == "v2"]
    assert all(r["n"] == 3 and r["decision_flip_rate"] == 0 for r in v2)
    write_report(report, tmp_path / "report")
    saved = json.loads((tmp_path / "report/comparison.json").read_text())
    assert saved["stability_settings"]["std_ddof"] == 0
    assert "do not export repeat details" not in (tmp_path / "report/comparison.json").read_text()
    document = (tmp_path / "report/comparison.html").read_text()
    assert "重复调用一致性" in document
    assert "恒定错误输出" in document
    for name in ("stability", "stability_paired", "stability_overview"):
        with (tmp_path / f"report/{name}.csv").open(encoding="utf-8-sig") as stream:
            assert len(list(csv.DictReader(stream))) == len(report[name])


def test_stability_requires_four_complete_runs_and_distinguishes_failures(tmp_path):
    data, outputs = prepare(tmp_path)
    providers = ["fastino.ai", "laya-ai.com"]
    for provider in providers:
        save_repeat_runs(outputs, provider, "toxigen", "v1", [[0.2, 0.8, 0.8]] * 4)
    path = outputs / "laya-ai.com/toxigen_v1_t0.8.csv"
    path.unlink()
    missing = collect_report(outputs, data, providers)
    row = next(r for r in missing["stability"] if r["provider"] == "laya-ai.com"
               and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert row["status"] == "incomplete_repeats"
    assert row["complete_repeats"] == row["available_repeats"] == 3
    assert row["coverage"] is None
    assert row["decision_flip_rate"] is None
    assert row["error_rate"] == 0
    assert missing["stability_paired"] == []
    assert all(r["mean_score_std"] is None for r in missing["stability_overview"])
    # A partial fourth CSV must not be treated as four completed repeats.
    save_repeat_runs(outputs, "laya-ai.com", "toxigen", "v1",
                     [[0.2, 0.8, 0.8]] * 3 + [[0.2, 0.8]])
    partial = collect_report(outputs, data, providers)
    row = next(r for r in partial["stability"] if r["provider"] == "laya-ai.com"
               and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert row["available_repeats"] == 4 and row["complete_repeats"] == 3
    assert row["mean_score_std"] is None
    # A complete CSV with all failed requests has zero coverage, not zero variation.
    save_repeat_runs(outputs, "laya-ai.com", "toxigen", "v1", [[0.2, 0.8, 0.8]] * 4,
                     failed={(3, sample) for sample in range(3)})
    failed = collect_report(outputs, data, providers)
    row = next(r for r in failed["stability"] if r["provider"] == "laya-ai.com"
               and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert row["status"] == "no_common_success"
    assert row["coverage"] == 0
    assert row["error_rate"] == 0.25
    assert row["decision_flip_rate"] is None
    assert all(r["n"] == 0 for r in failed["stability_paired"])


def test_stability_macro_is_dataset_equal_and_excludes_pii(tmp_path):
    data, outputs = prepare(tmp_path)
    reference = [
        {key: row[key] for key in ("sample_id", "text", "label")}
        for row in prediction_rows("fastino.ai")[:2]
    ]
    save_csv(data / DATASETS["hatecheck"], reference)
    for provider in ("fastino.ai", "laya-ai.com"):
        save_repeat_runs(outputs, provider, "toxigen", "v1", [[0.2, 0.8, 0.8]] * 4)
        save_repeat_runs(outputs, provider, "hatecheck", "v1",
                         [[0.4, 0.4], [0.6, 0.6], [0.4, 0.4], [0.6, 0.6]])
        save_repeat_runs(outputs, provider, "pii_masking_nano_1k", "v1",
                         [[0.4, 0.4, 0.4], [0.6, 0.6, 0.6]] * 2)
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    for row in report["stability_overview"]:
        if row["prompt"] == "v1":
            assert row["datasets"] == ["hatecheck", "toxigen"]
            assert row["n"] == row["expected"] == 5
            assert row["decision_flip_rate"] == 0.5  # Not the pooled 2/5, nor including PII.
            assert row["mean_score_std"] == pytest.approx(0.05)
            assert row["mean_score_range"] == pytest.approx(0.1)
        else:
            assert row["datasets"] == []
            assert row["decision_flip_rate"] is None


def external_row(dataset="toxigen", prompt="v1", **metrics):
    return {
        "dataset": dataset, "prompt": prompt, "threshold": 0.5, "rows": 3,
        "accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.6462,
        "auroc": 0.9, "mean_elapsed_ms": 1000, **metrics,
    }


def internal_row(provider="openjev-32b", **values):
    return {
        **external_row(), "provider": provider, "valid": 2, "errors": 1, **values,
    }


def test_internal_summaries_preserve_missing_metrics_counts_and_original_v2(tmp_path):
    data, outputs = prepare(tmp_path)
    providers = ["fastino.ai", "laya-ai.com"]
    for provider in providers:
        save_repeat_runs(outputs, provider, "toxigen", "v2", [[0.2, 0.8, 0.8]] * 4)
    report = collect_report(outputs, data, providers)
    summary = tmp_path / "summary.csv"
    save_csv(summary, [external_row(), external_row(prompt="v2")])
    attach_openjev_summary(report, summary)
    original = json.dumps({k: v for k, v in report.items() if k != "external_comparison"}, sort_keys=True)
    original_v2 = [row for row in report["external_comparison"]["overview"] if row["prompt"] == "v2"]
    source = tmp_path / "internal_model_summary.csv"
    save_csv(source, [
        internal_row(accuracy=""),
        internal_row("microsoft-decision-1", mean_elapsed_ms=1200),
    ])
    attach_openjev_summary(report, summary)
    assert json.dumps({k: v for k, v in report.items() if k != "external_comparison"}, sort_keys=True) == original
    comparison = report["external_comparison"]
    assert [row for row in comparison["overview"] if row["prompt"] == "v2"] == original_v2
    assert comparison["providers"] == [*providers, "openjev", "openjev-32b", "microsoft-decision-1"]
    row = next(row for row in comparison["runs"] if row["provider"] == "openjev-32b")
    assert row["valid"] == 2 and row["errors"] == 1
    assert row["accuracy"] is None
    assert row["f1"] == 0.6462 and row["auroc"] == 0.9
    assert row["mean_elapsed_ms"] == 1000
    assert row["latency_source_kind"] == "single_complete_run"
    average = next(row for row in comparison["overview"] if row["provider"] == "openjev-32b")
    assert average["datasets"] == ["toxigen"] and average["accuracy"] is None
    assert average["f1"] == 0.6462 and average["auroc"] == 0.9
    assert average["mean_elapsed_ms"] == 1000
    write_report(report, tmp_path / "report")
    assert (tmp_path / "report/internal_model_summary.csv").exists()
    assert "Microsoft Decision-1" in (tmp_path / "report/comparison.md").read_text()


@pytest.mark.parametrize("values", [
    {"valid": 3, "errors": 1}, {"valid": -1, "errors": 4},
    {"provider": "unknown"}, {"precision": 1.1},
])
def test_internal_summary_rejects_inconsistent_counts_or_metrics(tmp_path, values):
    source = tmp_path / "internal.csv"
    save_csv(source, [internal_row(**values)])
    with pytest.raises(ValueError):
        read_internal_summary(source)


@pytest.mark.parametrize("case", [
    "complete", "partial_log", "partial_csv", "wrong_threshold", "v2_only", "unverified_counts",
])
def test_v1_reference_single_run_timing_requires_complete_matched_half_threshold_log(tmp_path, case):
    data, outputs = prepare(tmp_path)
    prompt = "v2" if case == "v2_only" else "v1"
    threshold = 0.6 if case == "wrong_threshold" else 0.5
    processed = 2 if case in ("partial_log", "partial_csv") else 3
    if case == "v2_only":
        save_csv(outputs / "laya-ai.com/toxigen_v2_t0.5.csv", [
            {**row, "prompt_version": "v2"} for row in prediction_rows("laya-ai.com")
        ])
    elif case == "partial_csv":
        save_csv(outputs / "laya-ai.com/toxigen_v1_t0.5.csv", prediction_rows("laya-ai.com")[:2])
    log = outputs / "laya-ai.com/single.log"
    log.write_text(
        f"Running benchmark for laya-ai.com | toxigen | {prompt} | threshold={threshold} | total_rows=3\n"
        f"Progress: {processed}/3 | provider=laya-ai.com | dataset=toxigen | prompt={prompt} | elapsed=2.4s\n",
    )
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    if case == "unverified_counts":
        next(row for row in report["runs"] if (
            row["provider"] == "laya-ai.com" and row["dataset"] == "toxigen"
            and row["prompt"] == "v1" and row["threshold"] == 0.5
        )).pop("log_total")
    original = json.dumps(report, sort_keys=True)
    summary = tmp_path / "summary.csv"
    save_csv(summary, [external_row(), external_row(prompt="v2")])
    attach_openjev_summary(report, summary)
    row = next(row for row in report["external_comparison"]["runs"] if (
        row["provider"] == "laya-ai.com" and row["dataset"] == "toxigen" and row["prompt"] == "v1"
    ))
    if case == "complete":
        assert row["mean_elapsed_ms"] == 800
        assert row["latency_source_kind"] == "single_complete_run"
        assert row["latency_log_file"] == str(log)
    else:
        assert row["mean_elapsed_ms"] is None
        assert "latency_source_kind" not in row
    if case == "v2_only":
        assert next(row for row in report["external_comparison"]["runs"] if (
            row["provider"] == "laya-ai.com" and row["dataset"] == "toxigen" and row["prompt"] == "v2"
        ))["mean_elapsed_ms"] is None
    assert json.dumps({key: value for key, value in report.items() if key != "external_comparison"}, sort_keys=True) == original


def test_external_summary_keeps_raw_baselines_separate_and_matches_timing_method(tmp_path):
    data, outputs = prepare(tmp_path)
    save_csv(data / DATASETS["hatecheck"], [
        {key: row[key] for key in ("sample_id", "text", "label")}
        for row in prediction_rows("fastino.ai")[:2]
    ])
    for provider in ("fastino.ai", "laya-ai.com"):
        for prompt in ("v1", "v2"):
            save_repeat_runs(outputs, provider, "toxigen", prompt, [[0.2, 0.8, 0.8]] * 4)
        save_repeat_runs(outputs, provider, "hatecheck", "v1", [[0.2, 0.8]] * 4)
        save_repeat_runs(outputs, provider, "pii_masking_nano_1k", "v1", [[0.2, 0.8, 0.8]] * 4)
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    for row in report["runs"]:
        if row["status"] == "complete":
            row["mean_elapsed_ms"] = 100 * (THRESHOLDS.index(row["threshold"]) + 1)
            if row["dataset"] == "hatecheck":
                row["mean_elapsed_ms"] += 200
            if row["provider"] == "laya-ai.com" and row["prompt"] == "v2" and row["threshold"] == 0.8:
                row["mean_elapsed_ms"] = None
    original = json.dumps(report, sort_keys=True)
    path = tmp_path / "summary.csv"
    save_csv(path, [
        external_row(), external_row(prompt="v2", accuracy=0.9),
        external_row("hatecheck", rows=2, accuracy=0.6, mean_elapsed_ms=1200),
        external_row("pii_masking_nano_1k", accuracy=1.0, auroc=""),
    ])
    attach_openjev_summary(report, path)
    comparison = report.pop("external_comparison")
    assert json.dumps(report, sort_keys=True) == original
    report["external_comparison"] = comparison
    source = comparison["source_rows"]
    assert source[0]["accuracy"] == 0.8
    assert source[0]["valid"] is source[0]["errors"] is None
    assert source[-1]["auroc"] is None
    assert "openjev" not in report["providers"]
    own = next(row for row in comparison["runs"] if row["provider"] == "fastino.ai"
               and row["dataset"] == "toxigen" and row["prompt"] == "v1")
    assert own["accuracy"] == 1.0  # Quality remains t0.5, not an average across thresholds.
    assert own["mean_elapsed_ms"] == 250  # Timing is the four-run mean, not t0.5's 100.
    averages = [row for row in comparison["overview"] if row["prompt"] == "v1"]
    assert all(row["datasets"] == ["toxigen", "hatecheck"] for row in averages)
    assert averages[0]["mean_elapsed_ms"] == 350
    assert averages[-1]["accuracy"] == pytest.approx(0.7)
    assert averages[-1]["mean_elapsed_ms"] == 1100
    assert next(row for row in comparison["overview"] if row["provider"] == "laya-ai.com"
                and row["prompt"] == "v2")["mean_elapsed_ms"] is None
    destination = tmp_path / "report"
    write_report(report, destination)
    document = (destination / "comparison.html").read_text()
    assert "OpenJev 内部汇总结果" in document
    assert "不参与上述共同成功样本基线" in document
    assert "有效样本数" in document and "未知" in document
    assert "PII 普通二分类（是否含 PII，不是脱敏质量）" in document
    assert "当前数据只有正样本" in document
    for name in ("openjev_summary", "external_comparison_runs", "external_comparison_overview"):
        assert (destination / f"{name}.csv").is_file()
    saved = json.loads((destination / "comparison.json").read_text())
    assert saved["external_comparison"]["source_rows"][0]["accuracy"] == 0.8


@pytest.mark.parametrize(("updates", "message"), [
    ({"accuracy": "nan"}, "invalid accuracy"),
    ({"precision": 1.1}, "invalid precision"),
    ({"mean_elapsed_ms": -1}, "invalid mean_elapsed_ms"),
    ({"threshold": 0.6}, "threshold 0.5"),
    ({"rows": 0}, "positive rows"),
    ({"dataset": "unknown"}, "unknown dataset/prompt"),
])
def test_external_summary_rejects_invalid_metrics(tmp_path, updates, message):
    path = tmp_path / "summary.csv"
    save_csv(path, [external_row(**updates)])
    with pytest.raises(ValueError, match=message):
        read_openjev_summary(path)


def test_external_summary_rejects_duplicates_and_missing_columns(tmp_path):
    path = tmp_path / "summary.csv"
    save_csv(path, [external_row(), external_row()])
    with pytest.raises(ValueError, match="duplicate"):
        read_openjev_summary(path)
    path.write_text("dataset,prompt\ntoxigen,v1\n")
    with pytest.raises(ValueError, match="missing summary columns"):
        read_openjev_summary(path)


def test_three_dataset_means_exclude_other_datasets_and_require_all_three():
    providers = ["fastino.ai", "laya-ai.com", "typesafe.ai", "openjev"]
    records = [
        {
            "provider": provider, "prompt": prompt, "dataset": dataset,
            "status": "complete", "valid": 10,
            **{metric: value if metric != "mean_elapsed_ms" else value * 1000 for metric in METRICS},
        }
        for provider in providers for prompt in ("v1", "v2")
        for dataset, value in zip((*FOCUSED_DATASETS, "detect-jailbreak"), (0.1, 0.3, 0.5, 0.99))
    ]
    original = reference_averages(records, providers)
    assert original[0]["accuracy"] == pytest.approx((0.1 + 0.3 + 0.5 + 0.99) / 4)
    selected = reference_averages(records, providers, FOCUSED_DATASETS)
    assert len(selected) == 8
    for row in selected:
        assert row["datasets"] == list(FOCUSED_DATASETS)
        for metric in METRICS:
            assert row[metric] == pytest.approx(300 if metric == "mean_elapsed_ms" else 0.3)
    records = [
        row for row in records if not (
            row["provider"] == "laya-ai.com" and row["prompt"] == "v1" and row["dataset"] == "aegis2"
        )
    ]
    missing = reference_averages(records, providers, FOCUSED_DATASETS)
    for row in missing:
        assert row["datasets"] == list(FOCUSED_DATASETS)
        if row["prompt"] == "v1":
            assert all(row[metric] is None for metric in METRICS)  # Never substitute a two-dataset mean.
        else:
            assert row["accuracy"] == pytest.approx(0.3)


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


def test_ambiguous_log_is_excluded_without_losing_valid_logs(tmp_path):
    path = tmp_path / "parallel.log"
    start = "Running benchmark for laya-ai.com | toxigen | v1 | threshold=0.5 | total_rows=3\n"
    progress = "Progress: 3/3 | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed=1.2s\n"
    path.write_text(start + progress + start.replace("0.5", "0.6") +
                    progress.replace("dataset=toxigen", "dataset=hatecheck"))
    valid = tmp_path / "valid.log"
    valid.write_text(start + progress)
    warnings = []
    result = parse_logs([path, valid], warnings=warnings)
    assert set(result) == {("laya-ai.com", "toxigen", "v1", 0.5)}
    assert result[("laya-ai.com", "toxigen", "v1", 0.5)]["log_file"] == str(valid)
    assert len(warnings) == 1 and "整份日志" in warnings[0]
    path.write_text(start.strip() + start.replace("0.5", "0.6") + progress)
    with pytest.raises(ValueError, match="interleaved task starts"):
        parse_logs([path])


def test_incremental_update_attaches_new_logs_without_changing_quality(tmp_path):
    data, outputs = prepare(tmp_path)
    providers = ["fastino.ai", "laya-ai.com"]
    previous = collect_report(outputs, data, providers)
    (outputs / "laya-ai.com/run.log").write_text(
        "Running benchmark for laya-ai.com | toxigen | v1 | threshold=0.5 | total_rows=3\n"
        "Progress: 3/3 | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed=1.2s\n"
    )
    updated = collect_report(outputs, data, providers, previous_report=previous)
    row = next(r for r in updated["runs"] if r["provider"] == "laya-ai.com" and r["file"])
    assert row["mean_elapsed_ms"] == 400
    assert row["elapsed_s"] == 1.2
    assert updated["paired"] == previous["paired"]
    assert updated["pairwise"] == previous["pairwise"]
    assert row["valid"] == 3 and row["f1"] == 1


def parallel_timing_log(outputs):
    starts = [
        f"Running benchmark for laya-ai.com | toxigen | v1 | threshold={threshold} | total_rows=3"
        for threshold in THRESHOLDS
    ]
    completions = [
        f"Progress: 3/3 | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed={elapsed}s"
        for elapsed in (1, 4, 2, 3)
    ]
    saved = [
        f"Saved: {outputs}/laya-ai.com/toxigen_v1_t{threshold}.csv (rows=3)"
        for threshold in THRESHOLDS
    ]
    return "".join(starts) + "\n" + "\n".join(completions + saved) + "\n"


def test_parallel_complete_repeats_supply_only_reference_average_timing(tmp_path):
    data, outputs = prepare(tmp_path)
    for provider in ("fastino.ai", "laya-ai.com"):
        save_repeat_runs(outputs, provider, "toxigen", "v1", [[0.2, 0.8, 0.8]] * 4)
    (outputs / "laya-ai.com/parallel.log").write_text(parallel_timing_log(outputs))
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    timing = report["repeat_log_timings"][0]
    assert timing["mean_elapsed_ms"] == pytest.approx(2500 / 3)
    assert timing["rows_per_run"] == 3
    assert all(r["mean_elapsed_ms"] is None for r in report["runs"])
    path = tmp_path / "summary.csv"
    save_csv(path, [external_row()])
    attach_openjev_summary(report, path)
    row = next(r for r in report["external_comparison"]["runs"]
               if r["provider"] == "laya-ai.com" and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert row["mean_elapsed_ms"] == pytest.approx(2500 / 3)
    assert row["latency_source_kind"] == "four_run_log_aggregate"
    assert row["f1"] == 1
    report["runs"] = [
        {**r, "status": "missing"} if r["provider"] == "laya-ai.com" and r["threshold"] == 0.8 else r
        for r in report["runs"]
    ]
    attach_openjev_summary(report, path)
    row = next(r for r in report["external_comparison"]["runs"]
               if r["provider"] == "laya-ai.com" and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert row["mean_elapsed_ms"] is None


@pytest.mark.parametrize("invalid", ["missing_completion", "duplicate_completion", "duplicate_start", "missing_save"])
def test_parallel_average_requires_exactly_four_verifiable_runs(tmp_path, invalid):
    path = tmp_path / "parallel.log"
    text = parallel_timing_log(tmp_path)
    if invalid == "missing_completion":
        text = text.replace("Progress: 3/3", "Progress: 2/3", 1)
    elif invalid == "duplicate_completion":
        text += "Progress: 3/3 | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed=5s\n"
    elif invalid == "duplicate_start":
        text += "Running benchmark for laya-ai.com | toxigen | v1 | threshold=0.5 | total_rows=3\n"
    else:
        text = text.replace("t0.5.csv", "t0.6.csv")
    path.write_text(text)
    assert parse_repeat_log_timings([path]) == []


def test_later_rerun_invalidates_older_parallel_average(tmp_path):
    old, new = tmp_path / "old.log", tmp_path / "new.log"
    old.write_text(parallel_timing_log(tmp_path))
    new.write_text("Running benchmark for laya-ai.com | toxigen | v1 | threshold=0.5 | total_rows=3\n")
    timestamp = old.stat().st_mtime_ns + 1
    os.utime(new, ns=(timestamp, timestamp))
    assert parse_repeat_log_timings([new, old]) == []


def test_replaced_csv_cannot_reuse_historical_log_or_repeat_timing(tmp_path):
    data, outputs = prepare(tmp_path)
    for provider in ("fastino.ai", "laya-ai.com"):
        save_repeat_runs(outputs, provider, "toxigen", "v1", [[0.2, 0.8, 0.8]] * 4)
    csv_path = outputs / "laya-ai.com/toxigen_v1_t0.5.csv"
    (outputs / "log_input_hashes.json").write_text(json.dumps({
        "laya-ai.com/history.log": {csv_path.name: sha256(csv_path.read_bytes()).hexdigest()},
    }))
    lines = []
    for threshold in THRESHOLDS:
        lines.extend([
            f"Running benchmark for laya-ai.com | toxigen | v1 | threshold={threshold} | total_rows=3",
            "Progress: 3/3 | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed=1.2s",
            f"Saved: {outputs}/laya-ai.com/toxigen_v1_t{threshold}.csv (rows=3)",
        ])
    (outputs / "laya-ai.com/history.log").write_text("\n".join(lines))
    matched = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    assert matched["repeat_log_timings"][0]["mean_elapsed_ms"] == 400
    rows = prediction_rows("laya-ai.com")
    rows[0].update(score=0.9, prediction=1)
    save_csv(csv_path, rows)
    updated = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    row = next(r for r in updated["runs"] if r["file"] == str(csv_path))
    assert row["rows"] == row["expected"] == 3
    assert row["mean_elapsed_ms"] is row["log_file"] is None
    assert updated["repeat_log_timings"] == []
    assert any("已被新结果替换" in warning for warning in updated["warnings"])
    assert updated["baseline_datasets"] == ["toxigen"]
    source = tmp_path / "summary.csv"
    save_csv(source, [external_row()])
    attach_openjev_summary(updated, source)
    row = next(r for r in updated["external_comparison"]["runs"]
               if r["provider"] == "laya-ai.com" and r["dataset"] == "toxigen" and r["prompt"] == "v1")
    assert row["mean_elapsed_ms"] is None


def test_provider_own_log_takes_priority_over_newer_cross_provider_log(tmp_path):
    _, outputs = prepare(tmp_path)
    own = outputs / "laya-ai.com/own.log"
    cross = outputs / "fastino.ai/combined.log"
    lines = []
    for threshold in THRESHOLDS:
        lines.extend([
            f"Running benchmark for laya-ai.com | toxigen | v1 | threshold={threshold} | total_rows=3",
            "Progress: 3/3 | provider=laya-ai.com | dataset=toxigen | prompt=v1 | elapsed=1.2s",
            f"Saved: {outputs}/laya-ai.com/toxigen_v1_t{threshold}.csv (rows=3)",
        ])
    own.write_text("\n".join(lines))
    cross.write_text("\n".join(lines).replace("elapsed=1.2s", "elapsed=0.001s"))
    timestamp = own.stat().st_mtime_ns + 100
    os.utime(cross, ns=(timestamp, timestamp))
    records = parse_logs([own, cross])
    assert all(row["log_file"] == str(own) and row["elapsed_s"] == 1.2 for row in records.values())
    aggregate = parse_repeat_log_timings([own, cross])
    assert len(aggregate) == 1 and aggregate[0]["mean_elapsed_ms"] == 400
    own.write_text(lines[0])
    os.utime(own, ns=(timestamp - 1, timestamp - 1))
    assert parse_repeat_log_timings([own, cross]) == []


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
    assert all(row["macro_accuracy"] is None for row in report["overview"])
    assert all(row["macro_elapsed_ms"] is None for row in report["overview"])
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


def test_model_averages_use_common_samples_and_equal_dataset_weights(tmp_path):
    data, outputs = prepare(tmp_path)
    hatecheck = prediction_rows("fastino.ai")
    hatecheck.append({**hatecheck[0], "sample_id": 3, "text": "sample 3"})
    save_csv(data / DATASETS["hatecheck"], [
        {key: row[key] for key in ("sample_id", "text", "label")} for row in hatecheck
    ])
    for provider, toxigen_seconds, hatecheck_seconds in (
        ("fastino.ai", 3, 8), ("laya-ai.com", 1.2, 12),
    ):
        toxigen = prediction_rows(provider)
        if provider == "fastino.ai":
            toxigen[2].update(score=0.1, prediction=0)  # Excluded by the common-success intersection.
        else:
            toxigen[2].update(error="TimeoutError", score="", prediction="")
        save_csv(outputs / provider / "toxigen_v1_t0.5.csv", toxigen)
        rows = [{**row, "dataset": "hatecheck", "provider": provider} for row in hatecheck]
        rows[1].update(score=0.1, prediction=0)
        save_csv(outputs / provider / "hatecheck_v1_t0.5.csv", rows)
        (outputs / provider / "run.log").write_text(
            f"Running benchmark for {provider} | toxigen | v1 | threshold=0.5 | total_rows=3\n"
            f"Progress: 3/3 | provider={provider} | dataset=toxigen | prompt=v1 | elapsed={toxigen_seconds}s\n"
            f"Running benchmark for {provider} | hatecheck | v1 | threshold=0.5 | total_rows=4\n"
            f"Progress: 4/4 | provider={provider} | dataset=hatecheck | prompt=v1 | elapsed={hatecheck_seconds}s\n",
            encoding="utf-8",
        )
    report = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    for row, expected_latency in zip(report["overview"], (1500, 1700)):
        assert row["macro_accuracy"] == pytest.approx((1 + 0.75) / 2)
        assert row["macro_precision"] == 1
        assert row["macro_recall"] == pytest.approx((1 + 0.5) / 2)
        assert row["macro_auroc"] == pytest.approx((1 + 0.75) / 2)
        assert row["macro_elapsed_ms"] == pytest.approx(expected_latency)
        assert row["macro_elapsed_ms"] != pytest.approx(row["baseline_elapsed_ms"])
    # Incomplete timing must not silently average only the timed subset.
    log = outputs / "fastino.ai" / "run.log"
    log.write_text(log.read_text().split("Running benchmark for fastino.ai | hatecheck")[0])
    incomplete = collect_report(outputs, data, ["fastino.ai", "laya-ai.com"])
    assert incomplete["overview"][0]["macro_elapsed_ms"] is None


def test_reference_matches_original_runner_parser_and_platform_newlines(tmp_path, monkeypatch):
    path = tmp_path / "hatecheck.csv"
    save_csv(path, [
        {"sample_id": 0, "text": "prefix\x00truncated", "label": 1},
        {"sample_id": 1, "text": "", "label": 0},
        {"sample_id": 2, "text": "one\r\ntwo\rthree", "label": 0},
        {"sample_id": 3, "text": "李明 😀", "label": 1},
    ])
    submitted = []

    def capture_provider_input(**kwargs):
        submitted.append(kwargs["text"])
        return 0.5

    monkeypatch.setattr(provider_suite, "DATA_DIR", tmp_path)
    monkeypatch.setattr(provider_suite, "call_provider", capture_provider_input)
    predictions = provider_suite.run_single_benchmark("fastino.ai", "hatecheck", "v1", 0.5)
    reference = read_reference(path)
    assert len(submitted) == len(reference) == 4
    assert all(row["error"] == "" for row in predictions)
    # Match the actual model input rather than hard-coding pandas-version-specific
    # NUL truncation or the string representation of missing text.
    for row, text in zip(predictions, submitted):
        assert reference[str(row["sample_id"])] == (row["label"], fingerprint(text))
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
