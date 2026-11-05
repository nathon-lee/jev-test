import base64
import csv
import copy
import hashlib
import html
import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pytest

from benchmark.comparison import DATASETS, MODEL_IDENTITIES, MODELS
from benchmark.external_summary import (
    DATASET_CATEGORIES, GLIGUARD_DISPLAY_NAME, GLIGUARD_PROVIDER,
    GLINER25_DISPLAY_NAME, GLINER25_MODEL, GLINER25_PROVIDER, LUNA_PROVIDER, METRICS,
    POC_DISPLAY_NAME, POC_MODEL, POC_PROVIDER, QWEN_PROVIDER,
    attach_finetuned_run, attach_gliguard_run, attach_gliner25_run,
    attach_luna_summary, attach_openjev_summary, attach_poc_run,
    attach_qwen_summary, category_reference_averages,
    reference_averages, three_dataset_averages, v1_reference_scope_averages,
)
from benchmark.paper_figures import (
    coverage_figure,
    dataset_metrics_figure,
    english_warning,
    external_averages_figure,
    external_category_figure,
    external_dataset_figure,
    external_reference_providers,
    highlight_reference_bars,
    model_averages_figure,
    quality_figure,
    stability_averages_figure,
    stability_figure,
    threshold_figure,
    three_dataset_averages_figure,
    tradeoff_figure,
    write_paper_figures,
    write_fastino_comparison,
    write_summary_model_details,
    luna_summary_section,
    layout_reference_figure,
)


@pytest.fixture
def report():
    providers = list(MODELS)
    paired, runs, overview, stability, stability_overview = [], [], [], [], []
    stability_paired = []
    for index, provider in enumerate(providers):
        for threshold, value in ((0.5, 0.6 + index * 0.1), (0.7, 0.5 + index * 0.1)):
            paired.append({
                "provider": provider, "dataset": "toxigen", "prompt": "v1",
                "threshold": threshold, "n": 10, "f1": value, "auroc": 0.8,
            })
        overview.append({
            "provider": provider, "macro_f1": 0.6 + index * 0.1, "macro_auroc": 0.8,
            "macro_accuracy": 0.51 + index * 0.1, "macro_precision": 0.4,
            "macro_recall": 0.7, "macro_elapsed_ms": None if index == 1 else 300 + index * 100,
            "macro_fpr": 0.2, "macro_fnr": 0.3,
            "baseline_elapsed_ms": None if index == 1 else 200 + index * 100,
        })
        runs.append({
            "provider": provider, "dataset": "toxigen", "prompt": "v1", "threshold": 0.5,
            "rows": 10, "expected": 10, "valid": 10, "coverage": 1.0, "status": "complete",
            "accuracy": 0.2, "precision": 0.3, "recall": 0.4, "auroc": 0.5,
            "mean_elapsed_ms": 250 + index * 100,
            # Intentionally different: fair quality plots must use paired, not per-file metrics.
            "f1": 0.01,
        })
        for prompt in ("v1", "v2"):
            stability.append({
                "provider": provider, "dataset": "toxigen", "prompt": prompt,
                "mean_score_std": 0.01 if prompt == "v1" else 0.02,
                "mean_score_range": 0.04, "decision_flip_rate": 0.2,
                "coverage": 0.9, "error_rate": 0.01,
            })
            stability_overview.append({
                "provider": provider, "prompt": prompt, "datasets": ["toxigen"] if prompt == "v1" else [],
                "mean_score_std": 0.005 if prompt == "v1" else None,
                "mean_score_range": 0.02 if prompt == "v1" else None,
                "decision_flip_rate": 0.1 if prompt == "v1" else None,
            })
            stability_paired.append({
                "provider": provider, "dataset": "toxigen", "prompt": prompt,
                "n": 8, "expected": 10, "coverage": 0.8,
                "mean_score_std": 0.005 if prompt == "v1" else 0.01,
                "mean_score_range": 0.02, "decision_flip_rate": 0.1,
            })
    runs.extend([
        {
            "provider": providers[0], "dataset": "hatecheck", "prompt": "v1", "threshold": 0.5,
            "rows": 10, "expected": 10, "valid": 0, "coverage": 0.0, "status": "complete",
            "mean_elapsed_ms": 900,
        },
        {
            "provider": providers[1], "dataset": "hatecheck", "prompt": "v1", "threshold": 0.5,
            "rows": 0, "expected": 10, "valid": 0, "coverage": 0.0, "status": "missing",
        },
        {
            "provider": providers[0], "dataset": "pii_masking_nano_1k",
            "prompt": "v1", "threshold": 0.5,
            "rows": 10, "expected": 10, "valid": 10, "coverage": 1.0, "status": "complete",
            "accuracy": 0.9, "precision": 1.0, "recall": 0.9, "f1": 0.95, "auroc": None,
            "mean_elapsed_ms": 500,
        },
    ])
    return {
        "generated_at": "2026-10-21T00:00:00+00:00", "providers": providers,
        "model_identities": MODEL_IDENTITIES, "baseline_datasets": ["toxigen"],
        "overview": overview, "paired": paired, "runs": runs,
        "stability": stability, "stability_overview": stability_overview,
        "stability_paired": stability_paired,
        "stability_settings": {
            "repeat_thresholds": [0.5, 0.6, 0.7, 0.8],
            "decision_threshold": 0.5, "std_ddof": 0, "required_repeats": 4,
        },
    }


@pytest.mark.parametrize("datasets", [
    ["toxigen"],
    ["aegis2", "detect-jailbreak", "hatecheck", "jailbreak-classification", "jailbreakhub", "toxigen"],
])
def test_quality_chart_uses_paired_values_and_names(report, datasets):
    report["baseline_datasets"] = datasets
    report["paired"] = [{**row, "dataset": dataset} for dataset in datasets for row in report["paired"]]
    figure = quality_figure(report)
    try:
        heights = [bar.get_height() for bar in figure.axes[0].patches]
        assert heights == pytest.approx([value for value in (60, 70, 80) for _ in datasets])
        legend = "\n".join(text.get_text() for text in figure.legends[0].texts)
        assert "request: jev-latest" in legend
        assert "resolved: unconfirmed (API response required)" in legend
        assert "resolved: glide_v2" in legend
        assert "serving: laya-rl-agent" in legend
        assert "routed: typed-decisions" in legend
        assert figure.axes[0].get_position().y0 > figure.axes[1].get_position().y1
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        title, subtitle, _ = figure.texts
        padding = figure.dpi * 6 / 72
        assert title.get_window_extent(renderer).y0 >= subtitle.get_window_extent(renderer).y1 + padding
        assert subtitle.get_window_extent(renderer).y0 >= figure.legends[0].get_window_extent(renderer).y1 + padding
        assert figure.legends[0].get_window_extent(renderer).y0 >= figure.axes[0].title.get_window_extent(renderer).y1 + padding
        for ax in figure.axes:
            assert ax.get_position().width > 0.85
            labels = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
            assert all(left.x1 < right.x0 for left, right in zip(labels, labels[1:]))
            assert all(label.x0 >= 0 and label.x1 <= figure.bbox.x1 for label in labels)
        assert figure.legends[0].get_window_extent(renderer).y0 > figure.axes[0].get_window_extent(renderer).y1
    finally:
        plt.close(figure)


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("values, quality_winners, timing_winners", [
    ([0.8, 0.8, None], [0, 1], [0, 1]),
    ([0.8, 0.800000001, 0.7], [1], [2]),
    ([None, np.nan, None], [], []),
])
def test_reference_highlights_respect_ties_missing_values_and_metric_direction(
    horizontal, values, quality_winners, timing_winners,
):
    providers = list(MODELS)
    metrics = ("accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms")
    record = {
        provider: dict.fromkeys(metrics, value)
        for provider, value in zip(providers, values)
    }
    figure, axes = plt.subplots(3, 2, figsize=(16, 20))
    try:
        for ax in axes.flat:
            plotted = [value if value is not None else np.nan for value in values]
            if horizontal:
                ax.barh(range(3), plotted, linewidth=0.4)
                ax.set_yticks(range(3), providers)
            else:
                ax.bar(range(3), plotted, linewidth=0.4)
            for bar, value in zip(ax.patches, plotted):
                if math.isfinite(value):
                    point = (value, bar.get_y() + bar.get_height() / 2) if horizontal else (
                        bar.get_x() + bar.get_width() / 2, value
                    )
                    ax.annotate(str(value), point, xytext=(4, 0) if horizontal else (0, 4),
                                textcoords="offset points")
        highlight_reference_bars(figure, providers, [record], horizontal=horizontal)
        figure.canvas.draw()
        for index, ax in enumerate(axes.flat):
            winners = timing_winners if index == 5 else quality_winners
            assert [
                i for i, bar in enumerate(ax.patches) if bar.get_linewidth() == 2.2
            ] == winners
            arrows = [text for text in ax.texts if text.arrow_patch is not None]
            assert len(arrows) == len(winners)
            numbers = [text for text in ax.texts if text.arrow_patch is None
                       and text.get_gid() != "reference-highlight-label"]
            assert all(text.get_zorder() > 5 and text.get_bbox_patch() is not None for text in numbers)
            assert all(abs(text.get_position()[0]) <= 14 and abs(text.get_position()[1]) <= 14
                       for text in arrows)
            if winners:
                callout = next(text for text in ax.texts if text.get_gid() == "reference-highlight-label")
                label = callout.get_text().replace("\n", " ")
                box = callout.get_bbox_patch().get_window_extent(figure.canvas.get_renderer())
                assert all(not box.overlaps(text.get_window_extent(figure.canvas.get_renderer()))
                           for text in numbers)
                assert ("Lowest time:" if index == 5 else "Highest:") in label
                assert all({"fastino.ai": "Fastino", "laya-ai.com": "Laya", "typesafe.ai": "TypeSafe"}[providers[i]]
                           in label for i in winners)
                if horizontal:
                    assert box.y0 > ax.get_window_extent().y1
    finally:
        plt.close(figure)


def test_threshold_curve_does_not_interpolate_or_zero_fill_missing_runs(report):
    figure = threshold_figure(report)
    try:
        values = figure.axes[0].lines[0].get_ydata()
        assert values[0] == pytest.approx(60)
        assert np.isnan(values[1])
        assert values[2] == pytest.approx(50)
        assert np.isnan(values[3])
        assert np.isnan(figure.axes[0].lines[1].get_ydata()).all()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("empty_paired", [False, True])
def test_threshold_curve_preserves_own_results_when_other_models_are_missing(report, empty_paired):
    report["paired"] = [row for row in report["paired"] if row["threshold"] == 0.5]
    fastino, laya, typesafe = report["providers"]
    for provider, threshold, valid, f1 in (
        (fastino, 0.6, 0, 0.99),
        (fastino, 0.7, 3, 0.64),
        (fastino, 0.8, 10, 0.67),
        (typesafe, 0.6, 10, 0.94),
        (typesafe, 0.7, 10, 0.93),
        (typesafe, 0.8, 10, 0.92),
        (laya, 0.6, 0, None),
    ):
        row = {
            "provider": provider, "dataset": "toxigen", "prompt": "v1",
            "threshold": threshold, "valid": valid, "f1": f1,
        }
        report["runs"].append(row)
        if empty_paired:
            report["paired"].append({**row, "n": 0, "f1": None})
    figure = threshold_figure(report)
    try:
        ax = figure.axes[0]
        assert ax.lines[0].get_ydata() == pytest.approx([60, np.nan, 64, 67], nan_ok=True)
        assert ax.lines[2].get_ydata() == pytest.approx([70, np.nan, np.nan, np.nan], nan_ok=True)
        assert ax.lines[4].get_ydata() == pytest.approx([80, 94, 93, 92])
        assert ax.lines[0].get_markevery() == [0]
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[0.7, 64], [0.8, 67]])
        np.testing.assert_allclose(ax.collections[1].get_offsets(), [
            [0.6, 94], [0.7, 93], [0.8, 92],
        ])
        assert ax.collections[0].get_facecolors().size == 0
        assert "Own successful samples" in [text.get_text() for text in ax.get_legend().texts]
    finally:
        plt.close(figure)


def test_coverage_distinguishes_all_failed_from_missing(report):
    figure = coverage_figure(report)
    try:
        values = figure.axes[0].images[0].get_array()
        assert values[1, 0] == 0  # HateCheck, Fastino: all requests failed.
        assert np.ma.is_masked(values[1, 1])  # HateCheck, Laya: no predictions.
        labels = [text.get_text() for text in figure.axes[0].texts]
        assert any("ALL FAILED" in label for label in labels)
        assert "NOT AVAILABLE" in labels
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        ax = figure.axes[0]
        identities = figure.texts[-len(report["providers"]):]
        boxes = [text.get_window_extent(renderer) for text in identities]
        assert not figure.legends
        for index, (identity, box) in enumerate(zip(identities, boxes)):
            left = ax.transData.transform((index - 0.5, 0))[0]
            right = ax.transData.transform((index + 0.5, 0))[0]
            assert identity.get_va() == "top"
            assert left <= box.x0 < box.x1 <= right
            assert (box.x0 + box.x1) / 2 == pytest.approx((left + right) / 2)
            assert box.y1 == pytest.approx(boxes[0].y1)
        assert max(box.y1 for box in boxes) < min(
            label.get_window_extent(renderer).y0 for label in ax.get_xticklabels()
        )
        note = figure.texts[2].get_window_extent(renderer)
        assert note.x0 == pytest.approx(ax.get_window_extent(renderer).x0)
        assert 0 < note.y0 < note.y1 < min(box.y0 for box in boxes)
        assert note.x1 <= figure.bbox.x1
        assert "suite is incomplete" not in figure.texts[2].get_text()
        assert figure.texts[0].get_text() == "Evaluation Coverage and Request Failures"
    finally:
        plt.close(figure)


def test_dataset_metrics_use_own_samples_and_preserve_unavailable_values(report):
    figure = dataset_metrics_figure(report)
    try:
        assert len(figure.axes) == 6
        # First eight bars are Fastino's datasets, in DATASETS order.
        assert figure.axes[0].patches[0].get_height() == pytest.approx(20)
        assert figure.axes[3].patches[0].get_height() == pytest.approx(1)
        assert np.isnan(figure.axes[0].patches[1].get_height())  # All requests failed.
        assert np.isnan(figure.axes[0].patches[9].get_height())  # Missing result.
        assert figure.axes[0].patches[6].get_height() == pytest.approx(90)  # PII detection.
        assert np.isnan(figure.axes[4].patches[6].get_height())  # Single-class AUROC.
        assert figure.axes[5].patches[1].get_height() == pytest.approx(900)  # Failed run timing.
        assert any(text.get_text() == "N/A" for text in figure.axes[4].texts)
    finally:
        plt.close(figure)


def test_model_averages_use_paired_macros_and_dataset_equal_latency(report):
    figure = model_averages_figure(report)
    try:
        assert [bar.get_height() for bar in figure.axes[0].patches] == pytest.approx([51, 61, 71])
        assert figure.axes[1].patches[0].get_height() == pytest.approx(40)
        assert figure.axes[2].patches[0].get_height() == pytest.approx(70)
        assert figure.axes[3].patches[0].get_height() == pytest.approx(80)
        times = [bar.get_height() for bar in figure.axes[4].patches]
        assert times == pytest.approx([250, 350, 450])
        assert report["overview"][1]["macro_elapsed_ms"] is None
    finally:
        plt.close(figure)


def test_model_averages_latency_uses_same_complete_subset_with_equal_dataset_weights(report):
    report["baseline_datasets"] = ["toxigen", "aegis2", "hatecheck"]
    report["runs"] = [row for row in report["runs"] if row["dataset"] != "hatecheck"]
    for index, provider in enumerate(report["providers"]):
        report["overview"][index]["macro_elapsed_ms"] = None
        for dataset, rows, elapsed in (
            ("aegis2", 30, 600 + index * 100),
            ("hatecheck", 10, None if index == 1 else 9000),
        ):
            report["runs"].append({
                "provider": provider, "dataset": dataset, "prompt": "v1", "threshold": 0.5,
                "status": "complete", "rows": rows, "mean_elapsed_ms": elapsed,
            })
        for prompt, threshold in (("v2", 0.5), ("v1", 0.7)):
            report["runs"].append({
                "provider": provider, "dataset": "toxigen", "prompt": prompt, "threshold": threshold,
                "status": "complete", "rows": 10, "mean_elapsed_ms": 9999,
            })
    figure = model_averages_figure(report)
    try:
        times = [bar.get_height() for bar in figure.axes[4].patches]
        assert times == pytest.approx([425, 525, 625])
        assert figure.axes[0].patches[0].get_height() == pytest.approx(51)
        text = "\n".join(item.get_text() for item in figure.texts)
        assert "quality: 3 shared non-PII datasets" in text
        assert "latency: 2 shared fully timed datasets" in text
        assert "ToxiGen, AEGIS2" in text
        assert "Quality and latency dataset scopes can differ" in text
        assert all(row["macro_elapsed_ms"] is None for row in report["overview"])
    finally:
        plt.close(figure)


def test_model_averages_latency_does_not_use_different_subsets_or_old_macro_values(report):
    report["runs"][0]["mean_elapsed_ms"] = None
    figure = model_averages_figure(report)
    try:
        assert all(np.isnan(bar.get_height()) for bar in figure.axes[4].patches)
        assert [text.get_text() for text in figure.axes[4].texts] == ["N/A"] * 3
        text = "\n".join(item.get_text() for item in figure.texts)
        assert "latency: 0 shared fully timed datasets" in text
        assert "N/A timing means no shared fully timed baseline datasets" in text
    finally:
        plt.close(figure)


def test_tradeoff_recomputes_both_axes_on_same_all_model_timed_subset(report):
    report["baseline_datasets"] = ["toxigen", "aegis2", "hatecheck"]
    for index, provider in enumerate(report["providers"]):
        report["overview"][index]["baseline_elapsed_ms"] = None
        report["runs"].append({
            "provider": provider, "dataset": "aegis2", "prompt": "v1", "threshold": 0.5,
            "status": "complete", "rows": 30, "mean_elapsed_ms": 600 + index * 100,
        })
        report["paired"].extend([
            {
                "provider": provider, "dataset": dataset, "prompt": "v1", "threshold": 0.5,
                "n": 10, "auroc": auroc,
            }
            for dataset, auroc in [("aegis2", 0.9), ("hatecheck", 0.99)]
        ])
    figure = tradeoff_figure(report)
    try:
        points = [collection.get_offsets()[0] for collection in figure.axes[1].collections]
        assert np.asarray(points) == pytest.approx(np.array([
            [(250 * 10 + 600 * 30) / 40, 85],
            [(350 * 10 + 700 * 30) / 40, 85],
            [(450 * 10 + 800 * 30) / 40, 85],
        ]))
        text = "\n".join(item.get_text() for item in figure.texts)
        assert "(a) 3 shared datasets" in text
        assert "(b) 2 shared fully timed datasets" in text
        assert "ToxiGen, AEGIS2" in text
    finally:
        plt.close(figure)


def test_tradeoff_explains_when_no_shared_complete_timing_exists(report):
    report["runs"][0]["mean_elapsed_ms"] = None
    figure = tradeoff_figure(report)
    try:
        assert len(figure.axes[1].collections) == 0
        assert any("N/A" in text.get_text() for text in figure.axes[1].texts)
    finally:
        plt.close(figure)


def test_stability_figures_use_only_all_model_common_datasets_and_samples(report):
    # Two-model-only results and empty all-model intersections must not be plotted.
    template = report["stability_paired"][0]
    report["stability_paired"].extend([
        {**template, "provider": provider, "dataset": "hatecheck"}
        for provider in report["providers"][:2]
    ] + [
        {**template, "provider": provider, "dataset": "aegis2", "n": 0}
        for provider in report["providers"]
    ])
    for prompt, expected_std in (("v1", 0.5), ("v2", 1)):
        figure = stability_figure(report, prompt)
        try:
            assert len(figure.axes[0].patches) == 3
            assert figure.axes[0].patches[0].get_height() == pytest.approx(expected_std)
            assert [text.get_text() for text in figure.axes[0].get_xticklabels()] == ["ToxiGen\n(n = 8)"]
            assert figure.axes[2].patches[0].get_height() == pytest.approx(10)
            assert figure.axes[2].get_ylim()[1] == pytest.approx(13)
            assert figure.axes[3].patches[0].get_height() == pytest.approx(80)
            assert figure.axes[4].patches[0].get_height() == pytest.approx(1)
            assert "lower is better" in figure.axes[0].get_ylabel()
            assert "lower is better" in figure.axes[2].get_ylabel()
            assert not any(text.get_text() == "N/A" for text in figure.axes[0].texts)
        finally:
            plt.close(figure)
    figure = stability_averages_figure(report)
    try:
        assert figure.axes[0].patches[0].get_height() == pytest.approx(0.5)  # Paired, not own.
        assert figure.axes[2].patches[0].get_height() == pytest.approx(10)
        assert figure.axes[2].get_ylim()[1] == pytest.approx(13)
        assert all(np.isnan(bar.get_height()) for ax in figure.axes[3:] for bar in ax.patches)
    finally:
        plt.close(figure)


def test_stability_marks_measured_zeros_and_labels_empty_shared_data(report):
    for rows in (report["stability_paired"], report["stability_overview"]):
        for row in rows:
            if row["provider"] in report["providers"][:2] and row["prompt"] == "v1":
                row.update(mean_score_std=0.0, mean_score_range=0.0, decision_flip_rate=0.0)
    for figure in (stability_figure(report, "v1"), stability_averages_figure(report)):
        try:
            assert [bar.get_height() for bar in figure.axes[0].patches] == pytest.approx([0, 0, 0.5])
            assert [line.get_marker() for line in figure.axes[0].lines] == ["o", "s"]
            assert all(line.get_ydata()[0] == 0 for line in figure.axes[0].lines)
            assert [text.get_text() for text in figure.axes[0].texts].count("0.00") == 2
        finally:
            plt.close(figure)
    report["stability_paired"] = []
    figure = stability_figure(report, "v1")
    try:
        assert not figure.axes[0].patches
        assert any(text.get_text() == "NO SHARED FOUR-CALL SAMPLES" for text in figure.axes[0].texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("warning, expected", [
    ("incomplete timing log", "incomplete timing log"),
    (
        "laya.log: interleaved task starts cannot be assigned to progress"
        "；整份日志不用于逐配置耗时比较，不猜测并发进度的阈值归属。可核验的完整四轮均值仅用于四模型参考图。",
        "laya.log: interleaved task starts cannot be assigned to progress; "
        "the entire log is excluded from per-configuration timing comparisons; "
        "threshold assignments are not inferred from interleaved progress. "
        "Verifiable complete four-run averages are used only in the four-model reference figures.",
    ),
    (
        "fastino.ai/jailbreak-classification_v1_t0.5.csv"
        "：历史日志对应的预测文件已被新结果替换，不沿用该旧日志的耗时；质量指标仍使用新 CSV。",
        "fastino.ai/jailbreak-classification_v1_t0.5.csv: "
        "the prediction file associated with the historical log has been replaced by new results. "
        "Timing from that old log is not reused; quality metrics still use the new CSV.",
    ),
])
def test_legacy_warning_translation_preserves_identifiers_and_timing_caveats(warning, expected):
    assert english_warning(warning) == expected


def test_export_all_figure_formats_and_reject_empty_comparison(report, tmp_path):
    with pytest.raises(ValueError, match="at least 72"):
        write_paper_figures(report, tmp_path, dpi=10)
    with pytest.raises(ValueError, match="No shared valid baseline"):
        write_paper_figures({**report, "baseline_datasets": []}, tmp_path)
    with pytest.raises(ValueError, match="Regenerate"):
        write_paper_figures({key: value for key, value in report.items() if key != "stability"}, tmp_path)
    report["warnings"] = [
        "incomplete timing log", "historical log mismatch",
        "fastino.ai/jailbreak-classification_v1_t0.5.csv"
        "：历史日志对应的预测文件已被新结果替换，不沿用该旧日志的耗时；质量指标仍使用新 CSV。",
    ]
    original_report = json.dumps(report)
    manifest = write_paper_figures(report, tmp_path, dpi=72)
    assert len(manifest["figures"]) == 9
    for figure in manifest["figures"]:
        for extension, filename in figure["files"].items():
            content = (tmp_path / filename).read_bytes()
            assert len(content) > 1000
            if extension == "pdf":
                assert content.startswith(b"%PDF")
            elif extension == "png":
                assert content.startswith(b"\x89PNG")
            else:
                assert b"<svg" in content
    gallery = (tmp_path / "index.html").read_text()
    assert '<html lang="en">' in gallery
    assert not re.search(r"[\u3400-\u9fff]", gallery)
    overview = gallery.split("<section>", 1)[0]
    assert "<h1>Safety Model Evaluation</h1>" in overview
    assert '<time datetime="' + report["generated_at"] + '">2026-10-21 00:00 UTC</time>' in overview
    actions = re.search(r'<nav class="report-actions".*?>(.*?)</nav>', overview, re.DOTALL)
    assert actions is not None
    assert re.findall(r'href="([^"]+)"', actions.group(1)) == ["paper_figures.pdf", "../comparison.html"]
    data = re.search(r"<details><summary>Data &amp; Downloads</summary>(.*?)</details>", overview, re.DOTALL)
    methods = re.search(r"<details><summary>Methodology &amp; Limitations</summary>(.*?)</details>", overview, re.DOTALL)
    assert data is not None and methods is not None
    assert 'href="table01_baseline.tex"' in data.group(1)
    assert 'href="../stability.csv"' in data.group(1)
    assert "PNG defaults to 300 DPI" in data.group(1)
    assert "TypeSafe's resolved model must be confirmed by an API response" in methods.group(1)
    visible_overview = overview.split('<div class="report-resources">', 1)[0]
    assert "they are not fair rankings" in visible_overview
    assert "TypeSafe's resolved model" not in visible_overview
    assert "Chart labels and captions are in English" not in overview
    assert gallery.count("<img ") == 6
    title = "Sample-Aligned Classification Comparison (Fastino, Laya, TypeSafe)"
    paired = re.search(
        r"<section><details><summary>" + re.escape(title) + r"</summary>(.*?)</details></section>",
        gallery, re.DOTALL,
    )
    assert paired is not None
    assert "exactly the same successful sample IDs" in paired.group(1)
    assert "unlike the multi-model reference pages" in paired.group(1)
    embedded = re.search(
        r'<img src="data:image/png;base64,([^"]+)" alt="' + re.escape(title) + r'">', paired.group(1),
    )
    assert embedded is not None
    assert base64.b64decode(embedded.group(1), validate=True) == (tmp_path / "fig01_paired_quality.png").read_bytes()
    assert '<img src="fig01_paired_quality.svg"' not in gallery
    assert 'href="fig01_paired_quality.svg"' in gallery
    assert 'href="fig01_paired_quality.pdf"' in gallery
    assert all(f'href="fig01_paired_quality.{extension}"' in paired.group(1) for extension in ("png", "svg", "pdf"))
    tradeoff_title = "Error Rates and Latency Trade-offs (Fastino, Laya, TypeSafe)"
    tradeoff = re.search(
        r"<section><details><summary>" + re.escape(tradeoff_title) + r"</summary>(.*?)</details></section>",
        gallery, re.DOTALL,
    )
    assert tradeoff is not None
    assert manifest["figures"][1]["title"] == "Error Rates and Latency Trade-offs"
    assert '<img src="fig02_quality_cost_tradeoff.svg"' in tradeoff.group(1)
    assert all(
        f'href="fig02_quality_cost_tradeoff.{extension}"' in tradeoff.group(1)
        for extension in ("png", "svg", "pdf")
    )
    for figure in manifest["figures"][-3:]:
        assert f'<img src="{figure["files"]["svg"]}"' not in gallery
        assert f'<a href="{figure["files"]["svg"]}">{figure["title"]}</a>' in gallery
    details = re.search(
        r"<details><summary>Per-Dataset Performance Metrics \(Fastino, Laya, TypeSafe\)</summary>(.*?)</details>",
        gallery, re.DOTALL,
    )
    assert details is not None
    assert '<img src="fig05_dataset_metrics.svg"' in details.group(1)
    assert 'href="fig05_dataset_metrics.pdf"' in details.group(1)
    details = re.search(
        r"<details><summary>Sample-Aligned Model Averages \(Fastino, Laya, TypeSafe\)</summary>(.*?)</details>",
        gallery, re.DOTALL,
    )
    assert details is not None
    assert '<img src="fig06_model_averages.svg"' in details.group(1)
    assert 'href="fig06_model_averages.pdf"' in details.group(1)
    assert "<summary>Prompt and Threshold Sensitivity (Fastino, Laya, TypeSafe)</summary>" in gallery
    for title in ("Repeat-Call Consistency: v1", "Repeat-Call Consistency: v2", "Repeat-Call Consistency Averages"):
        assert title + " (Fastino, Laya, TypeSafe)" in gallery
    assert "<summary>Repeat-Call Consistency (Fastino, Laya, TypeSafe)</summary>" in gallery
    following = gallery.split("<summary>" + tradeoff_title + "</summary>", 1)[1]
    assert all(
        title.isascii()
        for title in re.findall(r"<(?:h2|summary)>([^<]+)</(?:h2|summary)>", following)
    )
    following_sections = re.findall(r"<section>(.*?)</section>", gallery.split(tradeoff.group(0), 1)[1], re.DOTALL)
    assert len(following_sections) == 6
    assert all(section.startswith("<details><summary>") and section.endswith("</details>") for section in following_sections)
    assert 'href="../stability.csv"' in gallery
    assert "https://" not in gallery
    assert gallery.rfind("<section>") == gallery.index("<section><details><summary>Data Warnings</summary>")
    for warning in manifest["warnings"]:
        assert gallery.count(warning) == 1
    assert len(manifest["warnings"]) == len(report["warnings"])
    assert "Timing from that old log is not reused; quality metrics still use the new CSV." in gallery
    assert json.dumps(report) == original_report
    table = (tmp_path / "table01_baseline.tex").read_text()
    assert r"\toprule" in table
    assert r"glide\_v2" in table
    assert "unconfirmed (API response required)" in table
    assert (tmp_path / "paper_figures.pdf").read_bytes().startswith(b"%PDF")
    saved = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert saved["source_generated_at"] == report["generated_at"]
    assert saved["model_identities"] == report["model_identities"]
    assert saved["stability_settings"] == report["stability_settings"]
    assert not plt.get_fignums()


def test_openjev_reference_exports_preserve_original_baselines_and_stability_links(report, tmp_path):
    path = tmp_path / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "dataset", "prompt", "threshold", "rows", "accuracy", "precision", "recall",
            "f1", "auroc", "mean_elapsed_ms",
        ])
        writer.writeheader()
        for prompt in ("v1", "v2"):
            writer.writerow({
                "dataset": "toxigen", "prompt": prompt, "threshold": 0.5, "rows": 10,
                "accuracy": 0.81, "precision": 0.9, "recall": 0.7,
                "f1": 0.7875, "auroc": 0.85, "mean_elapsed_ms": 900,
            })
    laya = next(row for row in report["runs"] if (
        row["provider"] == "laya-ai.com" and row["dataset"] == "toxigen"
        and row["prompt"] == "v1" and row["threshold"] == 0.5
    ))
    laya.update(log_file="laya-single.log", log_processed=10, log_total=10)
    attach_openjev_summary(report, path)
    figure = external_dataset_figure(report, "v1")
    try:
        assert figure.axes[0].patches[24].get_height() == pytest.approx(81)
        assert figure.axes[5].patches[24].get_height() == 900
        assert np.isnan(figure.axes[5].patches[0].get_height())  # No complete four-run timing.
        assert figure.axes[5].patches[8].get_height() == 350
        assert figure.axes[5].texts[8].get_text() == "350.0*"
        legend = "\n".join(text.get_text() for text in figure.legends[0].texts)
        assert "OpenJev 4B | supplied internal summary" in legend
        assert "not sample-paired" in legend
    finally:
        plt.close(figure)
    baseline = quality_figure(report)
    try:
        assert len(baseline.axes[0].patches) == 3
        assert [bar.get_height() for bar in baseline.axes[0].patches] == pytest.approx([60, 70, 80])
    finally:
        plt.close(baseline)
    destination = tmp_path / "figures"
    manifest = write_paper_figures(report, destination, dpi=72)
    assert len(manifest["figures"]) == 14
    assert manifest["external_comparison"]["source_kind"] == "user_supplied_aggregate"
    assert manifest["model_identities"] == report["model_identities"]
    gallery = (destination / "index.html").read_text()
    assert gallery.count("<img ") == 31
    assert "not a sample-aligned baseline" in gallery
    assert '<html lang="en">' in gallery
    assert not re.search(r"[\u3400-\u9fff]", gallery)
    assert 'href="../openjev_summary.csv"' in gallery
    for figure in manifest["figures"][6:9]:
        assert f'<img src="{figure["files"]["svg"]}"' not in gallery
        assert f'<a href="{figure["files"]["svg"]}">{figure["title"]}</a>' in gallery
    for figure in manifest["figures"][9:13]:
        if figure.get("panels"):
            assert f'<img src="{figure["files"]["svg"]}"' not in gallery
        else:
            assert f'<img src="{figure["files"]["svg"]}"' in gallery
        for filename in figure["files"].values():
            assert (destination / filename).stat().st_size > 1000
    collapsed = next(figure for figure in manifest["figures"] if figure["name"] == "fig11_openjev_reference_v2")
    assert collapsed["title"] == (
        "Per-Dataset Reference Comparison: v2 / Threshold 0.5 (Fastino, Laya, TypeSafe, OpenJev 4B)"
    )
    details = re.search(
        r"<details><summary>" + re.escape(collapsed["title"]) + r"</summary>(.*?)</details>",
        gallery, re.DOTALL,
    )
    assert details is not None
    assert f'<img src="{collapsed["files"]["svg"]}"' in details.group(1)
    for filename in collapsed["files"].values():
        assert f'href="{filename}"' in details.group(1)
    focused = manifest["figures"][-1]
    assert focused["name"] == "fig13_three_dataset_averages"
    assert gallery.index("<section>") == gallery.index(
        "<section><h2>" + focused["title"] + "</h2>"
    )
    v1_reference = next(figure for figure in manifest["figures"] if figure["name"] == "fig10_openjev_reference_v1")
    reference_averages_title = next(
        figure["title"] for figure in manifest["figures"] if figure["name"] == "fig12_openjev_reference_averages"
    )
    coverage_title = "Evaluation Coverage and Request Failures"
    assert manifest["figures"][3]["title"] == coverage_title
    category = next(figure for figure in manifest["figures"] if figure["name"] == "fig14_category_reference_averages")
    assert re.findall(r"<section><(?:h2>|details><summary>)([^<]+)", gallery)[:5] == [
        focused["title"], category["title"], v1_reference["title"], reference_averages_title, coverage_title,
    ]
    assert len(category["panels"]) == 6
    panel_sizes = []
    for panel in category["panels"]:
        svg = (destination / panel["files"]["svg"]).read_text()
        view_box = re.search(r'viewBox="([^"]+)"', svg).group(1)
        panel_sizes.append([float(value) for value in view_box.split()])
    for size in panel_sizes[1:]:
        assert size == pytest.approx(panel_sizes[0])
    assert len({
        plt.imread(destination / panel["files"]["png"]).shape
        for panel in category["panels"]
    }) == 1
    category_page = (destination / "category_reference_comparison.html").read_text()
    assert category_page.count("<img ") == 6
    assert 'src="data:image/png;base64,' in category_page
    assert 'href="category_reference_averages.csv"' in category_page
    assert "equal weight" in category_page
    assert 'href="category_reference_comparison.html"' in gallery
    category_card = gallery.split("<section><h2>" + category["title"] + "</h2>", 1)[1].split("</section>", 1)[0]
    primary, details = category_card.split("<details>", 1)
    assert '<aside class="pii-task-notice"' in primary
    assert primary.index('class="pii-task-notice"') < primary.index("<img ")
    assert re.findall(r'data-metric="([^"]+)"', primary) == ["accuracy", "f1", "mean_elapsed_ms"]
    assert re.findall(r'data-metric="([^"]+)"', details) == ["precision", "recall", "auroc"]
    with (destination / "category_reference_averages.csv").open(encoding="utf-8-sig") as stream:
        category_rows = list(csv.DictReader(stream))
    assert len(category_rows) == 12
    assert {row["category"] for row in category_rows} == set(DATASET_CATEGORIES)
    assert all(row["datasets"] == ",".join(DATASET_CATEGORIES[row["category"]]) for row in category_rows)
    assert all(row["prompt"] == "v1" and row["threshold"] == "0.5" for row in category_rows)
    assert all(row["auroc"] == "" for row in category_rows if row["category"] == "pii")
    assert len(v1_reference["panels"]) == 6
    v1_card = gallery.split("<section><h2>" + v1_reference["title"] + "</h2>", 1)[1].split("</section>", 1)[0]
    primary, detailed = v1_card.split("<details>", 1)
    assert '<aside class="pii-task-notice"' in primary
    assert "<strong>PII = ordinary binary classification" in primary
    assert primary.index('class="pii-task-notice"') < primary.index("<img ")
    detailed, methodology = detailed.split("<details>", 1)
    assert re.findall(r'data-metric="([^"]+)"', primary) == ["accuracy", "f1", "mean_elapsed_ms"]
    assert re.findall(r'data-metric="([^"]+)"', detailed) == ["precision", "recall", "auroc"]
    assert primary.count('class="model-legend"') == 1
    assert "* Single-run timing (v1 / threshold 0.5), not a four-run mean" in primary
    assert "Methodology and data scope" in methodology
    assert html.escape(v1_reference["caption"]) in methodology
    assert html.escape(v1_reference["caption"]) not in primary
    assert v1_card.endswith("</p></details>")
    for panel in v1_reference["panels"]:
        assert f'<img src="{panel["files"]["svg"]}"' in v1_card
        for filename in panel["files"].values():
            assert (destination / filename).stat().st_size > 1000
    assert all(f'href="{filename}"' in v1_card for filename in v1_reference["files"].values())
    assert f'<img src="{focused["files"]["svg"]}"' not in gallery
    focused_card = gallery.split("<section><h2>" + focused["title"] + "</h2>", 1)[1].split("</section>", 1)[0]
    visible = focused_card.split("<details>", 1)[0]
    assert 'class="reference-grid"' in visible
    assert re.findall(r'data-metric="([^"]+)"', visible) == [
        "accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms",
    ]
    assert "not a fair ranking" in visible
    assert "N/A = not supplied" in visible
    assert focused_card.count("<details>") == 1
    assert "Methodology and data scope" in focused_card
    scope = html.unescape(focused_card.split("<details>", 1)[1])
    assert "no internal dataset was available when the six models above were evaluated" in scope
    assert "Other model evaluations may also have included an additional internal dataset." in scope
    assert "Whether that dataset contributed to their supplied averages has not been confirmed" in scope
    assert "scope confirmed" not in scope and "was a typo" not in scope
    assert len(focused["panels"]) == 6
    for panel in focused["panels"]:
        assert f'<img src="{panel["files"]["svg"]}"' in focused_card
        for filename in panel["files"].values():
            assert (destination / filename).stat().st_size > 1000
    assert 'href="three_dataset_comparison.html"' in gallery
    page = (destination / "three_dataset_comparison.html").read_text()
    assert page.count("<img ") == 6
    encoded = re.search(r'<img src="data:image/png;base64,([^"]+)"', page)
    assert encoded is not None
    assert base64.b64decode(encoded.group(1), validate=True) == (
        destination / focused["panels"][0]["files"]["png"]
    ).read_bytes()
    assert re.findall(r'data-metric="([^"]+)"', page.split("<details>", 1)[0]) == [
        "accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms",
    ]
    assert page.count("<details>") == 1
    assert 'href="index.html"' in page
    assert 'href="three_dataset_averages.csv"' in page
    assert "equal weights of one third per dataset" in page
    assert not re.search(r"[\u4e00-\u9fff]", focused["caption"])
    for filename in focused["files"].values():
        assert (destination / filename).stat().st_size > 1000
    with (destination / "three_dataset_averages.csv").open(encoding="utf-8-sig") as stream:
        averages = list(csv.DictReader(stream))
    assert len(averages) == 9
    assert all(row["datasets"] == "toxigen,hatecheck,aegis2" for row in averages)
    assert all(row["accuracy"] == "" for row in averages[:4])  # Fixture lacks two required datasets.
    assert all(row["prompt"] == "v1" for row in averages[:4])
    supplied = {row["provider"]: row for row in averages[4:]}
    assert set(supplied) == {"Dynamo CPU", "Dynamo GPU (A100)", "Dynamo GPU (L4)", "Mistral", "Warden"}
    assert all(row["prompt"] == "not_supplied" for row in supplied.values())
    for row in supplied.values():
        precision, recall = float(row["precision"]), float(row["recall"])
        assert float(row["f1"]) == pytest.approx(2 * precision * recall / (precision + recall))
        assert row["f1_source"] == "harmonic_mean_of_supplied_precision_recall"
    assert float(supplied["Dynamo CPU"]["mean_elapsed_ms"]) == 20000
    assert float(supplied["Dynamo GPU (A100)"]["mean_elapsed_ms"]) == 55
    assert float(supplied["Dynamo GPU (L4)"]["mean_elapsed_ms"]) == 100
    assert supplied["Mistral"]["auroc"] == supplied["Warden"]["auroc"] == ""
    assert float(supplied["Mistral"]["recall"]) == 0.834
    assert "Three-dataset Multi-model Reference Averages" in page
    assert "ToxiGen, HateCheck, AEGIS2: Multi-model Reference Averages" in page
    assert "timing protocols were not supplied" in page
    assert "Whether that dataset contributed to their supplied averages has not been confirmed" in html.unescape(page)
    averages_figure = next(figure for figure in manifest["figures"] if figure["name"] == "fig12_openjev_reference_averages")
    reference_card = gallery.split(
        "<section><h2>" + averages_figure["title"] + "</h2>", 1,
    )[1].split("</section>", 1)[0]
    assert "PII binary classification" in reference_card
    assert "Ordinary binary classification: PII present vs absent" in reference_card
    assert "All eight datasets" in reference_card
    assert 'href="v1_reference_scope_averages.csv"' in reference_card
    assert re.findall(r'data-metric="([^"]+)"', reference_card.split("<details>", 1)[0]) == [
        "accuracy", "f1", "precision", "recall", "auroc", "mean_elapsed_ms",
    ]
    assert 'class="reference-grid"' in reference_card.split("<details>", 1)[0]
    assert reference_card.index('class="model-legend"') < reference_card.index("<img ")
    assert reference_card.count('class="model-legend"') == 1
    assert len(averages_figure["panels"]) == 6
    for panel in averages_figure["panels"]:
        assert f'<img src="{panel["files"]["svg"]}"' in reference_card
        for filename in panel["files"].values():
            assert (destination / filename).stat().st_size > 1000
    before_methodology, methodology = reference_card.rsplit("<details>", 1)
    assert methodology.startswith("<summary>Methodology and data scope</summary>")
    assert html.escape(averages_figure["caption"]) in methodology
    assert html.escape(averages_figure["caption"]) not in before_methodology
    assert reference_card.endswith("</p></details>")
    with (destination / "v1_reference_scope_averages.csv").open(encoding="utf-8-sig") as stream:
        scopes = list(csv.DictReader(stream))
    assert len(scopes) == 12
    assert {row["scope"] for row in scopes} == {"non_pii", "pii", "all"}
    assert {row["prompt"] for row in scopes} == {"v1"}
    assert {row["threshold"] for row in scopes} == {"0.5"}
    assert all(row["auroc"] == "" for row in scopes if row["scope"] != "non_pii")
    figure = three_dataset_averages_figure(report)
    try:
        figure_notes = "\n".join(text.get_text() for text in figure.texts)
        assert "no internal dataset was available" in figure_notes
        assert "Its inclusion in their averages is unconfirmed" in figure_notes
        assert "scope confirmed" not in figure_notes and "was a typo" not in figure_notes
        assert len(figure.axes[0].patches) == 9
        assert [text.get_text() for text in figure.axes[0].get_yticklabels()][:4] == [
            "Fastino", "Laya", "TypeSafe", "OpenJev 4B",
        ]
        assert [bar.get_width() for bar in figure.axes[0].patches[4:]] == pytest.approx(
            [89.65, 89.41, 89.39, 83.33, 77.73],
        )
        assert [bar.get_width() for bar in figure.axes[3].patches[4:]] == pytest.approx(
            [float(row["f1"]) * 100 for row in supplied.values()],
        )
        f1_labels = [text.get_text() for text in figure.axes[3].texts if text.arrow_patch is None
                     and text.get_gid() != "reference-highlight-label"]
        assert all(label.endswith("*") for label in f1_labels[4:])
        assert all(not label.endswith("*") for label in f1_labels[:4])
        assert figure.axes[3].get_ylim() == (8.5, -0.5)
        assert [bar.get_width() for bar in figure.axes[4].patches[4:7]] == pytest.approx(
            [95.27, 95.24, 95.24],
        )
        assert all(np.isnan(bar.get_width()) for bar in figure.axes[4].patches[7:])
        assert figure.axes[5].get_xscale() == "symlog"
        assert [bar.get_width() for bar in figure.axes[5].patches[4:]] == pytest.approx(
            [20000, 55, 100, 1671.08, 328.53],
        )
        labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
        for ax in figure.axes:
            assert [text.get_text() for text in ax.get_yticklabels()] == labels
            assert all(text.get_fontsize() >= 11 for text in ax.texts)
            assert [line.get_ydata()[0] for line in ax.lines] == [2.5, 3.5, 6.5]
    finally:
        plt.close(figure)
    assert not plt.get_fignums()


@pytest.mark.parametrize("no_shared_timing", [False, True])
def test_external_averages_use_v1_half_threshold_shared_timing_without_changing_source(
    report, no_shared_timing,
):
    providers = [*report["providers"], "openjev", "openjev-32b", "microsoft-decision-1"]
    records = []
    report["runs"] = []
    for index, provider in enumerate(providers):
        for prompt in ("v1", "v2"):
            if index >= 4 and prompt == "v2":
                continue
            for dataset_index, dataset in enumerate(("toxigen", "hatecheck", "aegis2")):
                elapsed = (index + 1 + dataset_index) * 100 + (1000 if prompt == "v2" else 0)
                if (prompt, provider, dataset) in {
                    ("v1", providers[0], "aegis2"),
                    ("v2", providers[1], "hatecheck"),
                } or (no_shared_timing and prompt == "v1" and provider == providers[0]):
                    elapsed = None
                records.append({
                    "provider": provider, "dataset": dataset, "prompt": prompt,
                    "status": "complete", "rows": 10 ** (dataset_index + 1), "valid": 10,
                    "source_kind": "user_supplied_aggregate" if index >= 3 else "local_predictions",
                    **{metric: 0.8 for metric in ("accuracy", "precision", "recall", "f1", "auroc")},
                    "mean_elapsed_ms": elapsed,
                })
                if index < 3:
                    report["runs"].append({**records[-1], "threshold": 0.5})
                    records[-1]["mean_elapsed_ms"] = 9999
                    report["runs"].append({
                        **records[-1], "prompt": "v1", "threshold": 0.8, "mean_elapsed_ms": 99999,
                    })
    overview = reference_averages(records, providers)
    report["external_comparison"] = {
        "providers": providers, "model_identities": report["model_identities"],
        "runs": records, "overview": overview,
    }
    saved = json.dumps(report)
    figure = external_averages_figure(report)
    try:
        expected_times = [np.nan if no_shared_timing else 150 + index * 100 for index in range(6)]
        times = [bar.get_height() for bar in figure.axes[5].patches]
        np.testing.assert_allclose(times, expected_times, equal_nan=True)
        quality = [bar.get_height() for bar in figure.axes[0].patches]
        np.testing.assert_allclose(quality, [80] * 6)
        for ax in figure.axes[:5]:
            assert ax.get_ylim() == pytest.approx((0, 120))
        assert figure.axes[5].get_ylim() == pytest.approx(
            (0, 1 if no_shared_timing else max(expected_times) * 1.3)
        )
        for ax in figure.axes:
            assert all(text.get_gid() != "reference-highlight-label" for text in ax.texts)
            assert all(text.arrow_patch is None for text in ax.texts)
            assert all(bar.get_linewidth() == 0.4 for bar in ax.patches)
        assert "3 shared non-PII datasets" in figure.axes[0].get_xticklabels()[0].get_text()
        count = 0 if no_shared_timing else 2
        assert f"{count} shared timed datasets" in figure.axes[5].get_xticklabels()[0].get_text()
        text = "\n".join(item.get_text() for item in figure.texts)
        assert ("None" if no_shared_timing else "ToxiGen, HateCheck") in text
        assert "v2" not in text
        assert "supplied summary" in text and "single-run timing" in text
        positions = [ax.get_position() for ax in figure.axes]
        for left, right in ((0, 3), (1, 2), (4, 5)):
            assert positions[left].x1 < positions[right].x0
            assert positions[left].y0 == pytest.approx(positions[right].y0)
            assert positions[left].width > 0.35
        assert positions[0].y0 > positions[1].y1
        assert positions[1].y0 > positions[4].y1
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert figure.legends[0].get_window_extent(renderer).y0 > max(
            ax.title.get_window_extent(renderer).y1 for ax in figure.axes
        )
        for ax in figure.axes:
            assert all(label.get_fontsize() >= 11 for label in ax.texts if label.arrow_patch is None)
            labels = [
                label.get_window_extent(renderer) for label in ax.texts
                if label.arrow_patch is None and label.get_gid() != "reference-highlight-label"
            ]
            assert all(left.x1 < right.x0 for left, right in zip(labels, labels[1:]))
            bounds = ax.get_window_extent(renderer)
            assert all(bounds.x0 <= label.x0 < label.x1 <= bounds.x1 for label in labels)
        note = figure.texts[-1].get_window_extent(renderer)
        assert note.y0 > 0
        assert note.y1 < min(
            label.get_window_extent(renderer).y0 for ax in figure.axes for label in ax.get_xticklabels()
        )
        assert json.dumps(report) == saved
    finally:
        plt.close(figure)


@pytest.mark.parametrize("missing_jailbreak", [False, True])
def test_category_reference_averages_use_fixed_equal_dataset_weights(missing_jailbreak):
    records = []
    for provider in ("fastino.ai", "openjev"):
        for category, datasets in DATASET_CATEGORIES.items():
            for index, dataset in enumerate(datasets):
                row = {
                    "provider": provider, "dataset": dataset, "prompt": "v1", "threshold": 0.5,
                    "status": "complete", "rows": 10 ** (index + 1), "valid": 10,
                    "accuracy": 0.2 + 0.2 * index, "precision": 0.9, "recall": 0.4,
                    "f1": 0.1 + 0.2 * index,
                    "auroc": None if category == "pii" else 0.8,
                    "mean_elapsed_ms": None if category == "harmful_content" and index == 1 else 100 * (index + 1),
                }
                if not (missing_jailbreak and provider == "fastino.ai" and dataset == "jailbreakhub"):
                    records.append(row)
                records.extend([{**row, "prompt": "v2", "accuracy": 0.99},
                                {**row, "threshold": 0.7, "accuracy": 0.99}])
    report = {
        "external_comparison": {
            "providers": ["fastino.ai", "openjev"], "runs": records,
            "model_identities": MODEL_IDENTITIES,
        },
    }
    averages = category_reference_averages(report)
    grouped = {(row["category"], row["provider"]): row for row in averages}
    assert len(averages) == 6
    for provider in ("fastino.ai", "openjev"):
        harmful = grouped[("harmful_content", provider)]
        assert harmful["accuracy"] == pytest.approx(0.4)
        assert harmful["f1"] == pytest.approx(0.3)
        assert harmful["mean_elapsed_ms"] is None
        assert harmful["datasets"] == list(DATASET_CATEGORIES["harmful_content"])
        pii = grouped[("pii", provider)]
        assert pii["accuracy"] == pytest.approx(0.3)
        assert pii["mean_elapsed_ms"] == 150
        assert pii["auroc"] is None
        jailbreak = grouped[("jailbreak", provider)]
        if missing_jailbreak and provider == "fastino.ai":
            assert jailbreak["accuracy"] is jailbreak["f1"] is jailbreak["mean_elapsed_ms"] is None
        else:
            assert jailbreak["accuracy"] == pytest.approx(0.4)
            assert jailbreak["mean_elapsed_ms"] == 200
        assert jailbreak["datasets"] == list(DATASET_CATEGORIES["jailbreak"])
    figure = external_category_figure(report)
    try:
        assert len(figure.axes) == 6
        assert [bar.get_height() for bar in figure.axes[0].patches[:3]] == pytest.approx(
            [40, np.nan if missing_jailbreak else 40, 30], nan_ok=True,
        )
        assert figure.axes[-1].get_yscale() == "log"
        assert any("ordinary binary classification (present vs absent)" in text.get_text() for text in figure.texts)
        notice = next(text for text in figure.texts if text.get_text().startswith("PII = ordinary binary classification"))
        assert notice.get_fontweight() == "bold" and notice.get_fontsize() == 12
        assert notice.get_bbox_patch().get_facecolor() == to_rgba("#FFF3CD")
        assert [text.get_text() for text in figure.axes[0].get_xticklabels()] == [
            "Harmful content\n(3 datasets)", "Jailbreak\n(3 datasets)", "PII binary classification\n(2 datasets)",
        ]
    finally:
        plt.close(figure)


@pytest.fixture
def fine_tuned_run(report, tmp_path):
    provider = "fastino-ft-gg6-r1-ml2048-s7"
    model = "c94233af-98e6-4942-8e26-98ad204851bf"
    data = tmp_path / "data"
    data.mkdir()
    run = tmp_path / "run"
    (run / provider).mkdir(parents=True)
    samples = [
        {"sample_id": 0, "text": "Clean example", "label": 0},
        {"sample_id": 1, "text": "Unsafe example", "label": 1},
        {"sample_id": 2, "text": "Another example", "label": 1},
    ]
    hashes = {}
    for dataset, filename in DATASETS.items():
        path = data / filename
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        hashes["data/prepared/" + filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        if dataset.startswith("pii_"):
            continue
        rows = []
        for sample, score in zip(samples, [0.1, 0.2, 0.9]):
            failed = dataset == "jailbreakhub" and sample["sample_id"] == 1
            rows.append({
                **sample, "dataset": dataset, "provider": provider,
                "prompt_version": "v1", "threshold": 0.5,
                "score": "" if failed else score,
                "prediction": "" if failed else int(score >= 0.5),
                "error": "HTTPStatusError('400')" if failed else "",
            })
        with (run / provider / f"{dataset}_v1_t0.5.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "provider": provider, "model": model, "display_name": "gg6-r1-ml2048-s7 v1",
        "base_model": "Gliguard LLMGuardrails 300M",
        "datasets": list(DATASETS)[:6], "prompt_version": "v1", "decision_threshold": 0.5,
        "input_sha256": hashes,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    lines = [f"MODEL_ID={model}"]
    lines.extend(
        f"Running benchmark for {provider} | {dataset} | v1 | threshold=0.5 | total_rows=3"
        for dataset in metadata["datasets"]
    )
    for count in (1, 3):
        lines.extend(
            f"Progress: {count}/3 (100%) | provider={provider} | dataset={dataset} | prompt=v1 | elapsed={count * 2}.0s"
            for dataset in metadata["datasets"]
        )
    log = tmp_path / "run.log"
    log.write_text("\n".join(lines))
    providers = [*report["providers"], "openjev"]
    report["external_comparison"] = {
        "providers": providers, "model_identities": {**MODEL_IDENTITIES, "openjev": {}},
        "runs": [
            {
                "provider": name, "dataset": dataset, "prompt": prompt, "threshold": 0.5,
                "rows": 3, "valid": 3, "errors": 0, "status": "complete",
                "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                "auroc": None if dataset.startswith("pii_") else 0.5,
                "mean_elapsed_ms": 1000,
            }
            for name in providers for dataset in DATASETS for prompt in ("v1", "v2")
        ],
        "source": "summary.csv", "source_kind": "user_supplied_aggregate", "notes": "",
        "classification_threshold": 0.5, "latency_method": "reference",
        "repeat_thresholds": [0.5, 0.6, 0.7, 0.8],
    }
    return report, run, log, data


def test_qwen_summary_preserves_existing_scopes_and_excludes_unsupported_pii(fine_tuned_run, tmp_path):
    report, _, _, _ = fine_tuned_run
    before = copy.deepcopy(report)
    scopes_before = v1_reference_scope_averages(report)
    focused_before = three_dataset_averages(report)
    source = Path(__file__).parents[1] / "data/external/qwen3guard_stream_8b_summary.csv"
    attach_qwen_summary(report, source)
    comparison = report["external_comparison"]
    model = comparison["summary_models"][QWEN_PROVIDER]
    assert (model["rows"], model["valid"], model["errors"]) == (66794, 66733, 61)
    assert len(model["supplied_rows"]) == 12
    assert report["runs"] == before["runs"]
    assert report["providers"] == before["providers"]
    assert comparison["runs"][:-8] == before["external_comparison"]["runs"]
    supplied = {row["dataset"]: row for row in model["supplied_rows"]}
    assert all(row["mean_elapsed_ms"] == row["log_elapsed_mean_ms_per_sample"] > 0
               for row in supplied.values())
    assert all(row["latency_source_kind"] == "single_complete_run" for row in supplied.values())
    assert all(row["p95_source_kind"] == "successful_request_latency_ms" for row in supplied.values())
    assert QWEN_PROVIDER in external_reference_providers(comparison, "v1")
    assert QWEN_PROVIDER not in external_reference_providers(comparison, "v2")
    scopes_after = v1_reference_scope_averages(report)
    assert [row for row in scopes_after if row["provider"] != QWEN_PROVIDER] == scopes_before
    assert [
        row for row in three_dataset_averages(report) if row["provider"] != QWEN_PROVIDER
    ] == focused_before
    qwen_scopes = [row for row in scopes_after if row["provider"] == QWEN_PROVIDER]
    assert {row["scope"] for row in qwen_scopes} == {"non_pii", "pii", "all"}
    for row in qwen_scopes:
        if row["scope"] in ("pii", "all"):
            assert all(row[metric] is None for metric in METRICS)
            assert row["shared_elapsed_ms"] is None
        else:
            assert row["mean_elapsed_ms"] == pytest.approx(
                np.mean([supplied[dataset]["log_elapsed_mean_ms_per_sample"] for dataset in row["datasets"]])
            )
            assert row["shared_elapsed_ms"] == pytest.approx(
                np.mean([supplied[dataset]["log_elapsed_mean_ms_per_sample"] for dataset in row["timing_datasets"]])
            )
    focused = next(row for row in three_dataset_averages(report) if row["provider"] == QWEN_PROVIDER)
    assert focused["mean_elapsed_ms"] == pytest.approx(1649.0928346666665)
    pii_category = next(
        row for row in category_reference_averages(report)
        if row["provider"] == QWEN_PROVIDER and row["category"] == "pii"
    )
    assert pii_category["task_supported"] is False
    assert all(pii_category[metric] is None for metric in METRICS)
    figure = external_dataset_figure(report, "v1")
    try:
        for ax in figure.axes:
            assert sum(text.get_text() == "Unsupported" for text in ax.texts) == 2
            assert all(text.get_gid() != "reference-highlight-label" for text in ax.texts)
        assert np.isnan(figure.axes[0].patches[-1].get_height())
        assert np.isnan(figure.axes[0].patches[-2].get_height())
        np.testing.assert_allclose(
            [bar.get_height() for bar in figure.axes[-1].patches[-8:]],
            [supplied[dataset]["log_elapsed_mean_ms_per_sample"]
             if supplied[dataset]["task_supported"] else np.nan for dataset in DATASETS],
            equal_nan=True,
        )
    finally:
        plt.close(figure)
    write_summary_model_details(comparison, tmp_path)
    with (tmp_path / model["summary_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 12
    pii = next(row for row in exported if row["dataset"] == "pii_masking_mini_10k")
    assert pii["task_supported"] == "False"
    assert pii["f1"] == "0.555573"
    assert float(pii["mean_elapsed_ms"]) == float(pii["log_elapsed_mean_ms_per_sample"]) == 1348.288084
    assert pii["p95_latency_ms"] == "NaN"
    golden = next(row for row in exported if row["dataset"] == "pii-detection-golden")
    assert float(golden["log_elapsed_mean_ms_per_sample"]) == 1213.470162
    assert float(golden["p95_latency_ms"]) == 2124.974660
    assert "pii_rerun_20261029_203544.log" in golden["latency_log_file"]
    page = (tmp_path / model["detail_page"]).read_text()
    assert "66,794 rows; 66,733 valid; 61 errors" in page
    assert "Unsupported (historical safety-proxy only)" in page
    assert "2700/2700" in page and "not probabilities" in page
    assert "including failed requests, loop overhead and waiting" in page
    assert "missing P95 is not inferred from log average latency" in page
    assert "python -m scripts.plot_model_comparison --qwen-summary" in page
    attach_qwen_summary(report, source)
    assert comparison["providers"].count(QWEN_PROVIDER) == 1
    assert len([row for row in comparison["runs"] if row["provider"] == QWEN_PROVIDER]) == 8
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("column,value", [
    ("valid", "-1"), ("prompt", "v2"), ("accuracy", "1.1"),
    ("log_elapsed_mean_ms_per_sample", "inf"), ("p95_latency_ms", "-1"),
    ("latency_log_file", ""), ("dataset", "unknown"),
])
def test_qwen_summary_rejects_invalid_supplied_values(fine_tuned_run, tmp_path, column, value):
    report, _, _, _ = fine_tuned_run
    source = Path(__file__).parents[1] / "data/external/qwen3guard_stream_8b_summary.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][column] = value
    invalid = tmp_path / "invalid_summary.csv"
    with invalid.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = copy.deepcopy(report)
    with pytest.raises(ValueError):
        attach_qwen_summary(report, invalid)
    assert report == before


def test_qwen_missing_log_latency_remains_unavailable(fine_tuned_run, tmp_path):
    report, _, _, _ = fine_tuned_run
    source = Path(__file__).parents[1] / "data/external/qwen3guard_stream_8b_summary.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    next(row for row in rows if row["dataset"] == "toxigen")["log_elapsed_mean_ms_per_sample"] = "NaN"
    partial = tmp_path / "partial_summary.csv"
    with partial.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    attach_qwen_summary(report, partial)
    model = report["external_comparison"]["summary_models"][QWEN_PROVIDER]
    toxigen = next(row for row in model["supplied_rows"] if row["dataset"] == "toxigen")
    assert toxigen["mean_elapsed_ms"] is toxigen["p95_latency_ms"] is None
    focused = next(row for row in three_dataset_averages(report) if row["provider"] == QWEN_PROVIDER)
    assert focused["mean_elapsed_ms"] is None
    assert focused["f1"] == pytest.approx(0.807799)


def test_finetuned_reference_preserves_existing_models_and_untested_scopes(fine_tuned_run):
    report, run, log, data = fine_tuned_run
    before = copy.deepcopy(report)
    pii_before = [row for row in v1_reference_scope_averages(report) if row["scope"] == "pii"]
    attach_finetuned_run(report, run, log, data)
    comparison = report["external_comparison"]
    provider = "fastino-ft-gg6-r1-ml2048-s7"
    assert report["runs"] == before["runs"]
    assert report["providers"] == before["providers"]
    assert [row for row in comparison["runs"] if row["provider"] != provider] == before["external_comparison"]["runs"]
    records = [row for row in comparison["runs"] if row["provider"] == provider]
    assert len(records) == 6 and sum(row["errors"] for row in records) == 1
    assert all(row["mean_elapsed_ms"] == 2000 for row in records)
    assert provider in external_reference_providers(comparison, "v1")
    assert provider not in external_reference_providers(comparison, "v2")
    scopes = v1_reference_scope_averages(report)
    assert [row for row in scopes if row["scope"] == "pii" and row["provider"] != provider] == pii_before
    for scope in ("pii", "all"):
        row = next(row for row in scopes if row["scope"] == scope and row["provider"] == provider)
        assert row["accuracy"] is row["f1"] is row["auroc"] is None
    main = next(row for row in scopes if row["scope"] == "non_pii" and row["provider"] == provider)
    assert main["f1"] == pytest.approx((5 * 2 / 3 + 1) / 6)
    focused = next(row for row in three_dataset_averages(report) if row["provider"] == provider)
    assert focused["f1"] == pytest.approx(2 / 3) and focused["mean_elapsed_ms"] == 2000
    attach_finetuned_run(report, run, log, data)
    assert len([row for row in comparison["runs"] if row["provider"] == provider]) == 6


@pytest.fixture
def gliguard_run(fine_tuned_run):
    report, run, log, data = fine_tuned_run
    old_provider = "fastino-ft-gg6-r1-ml2048-s7"
    model = "fastino/gliguard-LLMGuardrails-300M"
    (run / old_provider).rename(run / GLIGUARD_PROVIDER)
    for path in (run / GLIGUARD_PROVIDER).glob("*.csv"):
        path.write_text(path.read_text().replace(old_provider, GLIGUARD_PROVIDER))
    metadata = json.loads((run / "run_metadata.json").read_text())
    status = {
        "provider": GLIGUARD_PROVIDER, "model": model,
        "status": "completed_with_errors", "exit_code": 0, "completed_datasets": 6,
        "prompt_version": "v1", "threshold": 0.5,
        "processed_rows": 18, "valid_rows": 17, "error_rows": 1,
        "finished_at": "2026-11-03T15:47:47+00:00",
        "datasets": {
            dataset: {"rows": 3, "sha256": metadata["input_sha256"]["data/prepared/" + DATASETS[dataset]]}
            for dataset in list(DATASETS)[:6]
        },
    }
    (run / "run_status.json").write_text(json.dumps(status))
    (run / "run.log").write_text(
        "\n".join(log.read_text().splitlines()[1:]).replace(old_provider, GLIGUARD_PROVIDER)
    )
    return report, run, data


def test_gliguard_reference_preserves_baseline_and_excludes_untested_scopes(gliguard_run):
    report, run, data = gliguard_run
    before = copy.deepcopy(report)
    attach_gliguard_run(report, run, data)
    comparison = report["external_comparison"]
    assert report["providers"] == before["providers"]
    assert report["runs"] == before["runs"]
    assert [row for row in comparison["runs"] if row["provider"] != GLIGUARD_PROVIDER] == before["external_comparison"]["runs"]
    assert GLIGUARD_PROVIDER in external_reference_providers(comparison, "v1")
    assert GLIGUARD_PROVIDER not in external_reference_providers(comparison, "v2")
    info = comparison["classification_models"][GLIGUARD_PROVIDER]
    assert (info["rows"], info["valid"], info["errors"]) == (18, 17, 1)
    focused = next(row for row in three_dataset_averages(report) if row["provider"] == GLIGUARD_PROVIDER)
    assert focused["f1"] == pytest.approx(2 / 3)
    assert focused["mean_elapsed_ms"] == 2000
    pii = next(row for row in category_reference_averages(report) if row["provider"] == GLIGUARD_PROVIDER and row["category"] == "pii")
    assert all(pii[metric] is None for metric in METRICS)
    attach_gliguard_run(report, run, data)
    assert len([row for row in comparison["runs"] if row["provider"] == GLIGUARD_PROVIDER]) == 6


@pytest.mark.parametrize("invalid", ["model", "reference", "incomplete_log", "counts"])
def test_gliguard_reference_rejects_mismatched_sources(gliguard_run, invalid):
    report, run, data = gliguard_run
    if invalid == "reference":
        (data / DATASETS["toxigen"]).write_text("different reference")
    elif invalid == "incomplete_log":
        log = run / "run.log"
        log.write_text(log.read_text().replace("Progress: 3/3", "Progress: 2/3"))
    else:
        path = run / "run_status.json"
        metadata = json.loads(path.read_text())
        metadata["model" if invalid == "model" else "valid_rows"] = "fastino/GLiDE" if invalid == "model" else 18
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        attach_gliguard_run(report, run, data)
    assert "classification_models" not in report["external_comparison"]


def test_gliguard_reference_exports_identity_counts_and_metrics(gliguard_run, tmp_path):
    report, run, data = gliguard_run
    attach_gliguard_run(report, run, data)
    destination = tmp_path / "figures"
    manifest = write_paper_figures(report, destination, dpi=72)
    assert GLIGUARD_PROVIDER in manifest["external_comparison"]["classification_models"]
    for name in ("index.html", "category_reference_comparison.html", "three_dataset_comparison.html"):
        page = (destination / name).read_text()
        assert GLIGUARD_DISPLAY_NAME in page
        assert "fastino/gliguard-LLMGuardrails-300M" in page
        assert "1 errors" in page and "not GLiDE Noul" in page
    for suffix in ("runs", "averages"):
        with (destination / f"classification_model_{suffix}.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert all(row["display_name"] == GLIGUARD_DISPLAY_NAME for row in rows)
        assert all(row["model"] == "fastino/gliguard-LLMGuardrails-300M" for row in rows)
    assert next(row for row in rows if row["category"] == "pii")["accuracy"] == ""


def test_gliguard_long_model_label_does_not_overlap_neighboring_panels(gliguard_run):
    report, run, data = gliguard_run
    attach_gliguard_run(report, run, data)
    figure = three_dataset_averages_figure(report)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for left, right in zip(figure.axes[::2], figure.axes[1::2]):
            assert all(
                label.get_window_extent(renderer).x0 > left.get_window_extent(renderer).x1
                for label in right.get_yticklabels()
            )
        assert GLIGUARD_DISPLAY_NAME in [label.get_text() for label in figure.axes[0].get_yticklabels()]
    finally:
        plt.close(figure)


def test_fastino_standalone_page_contains_only_two_models_and_embedded_panels(gliguard_run, tmp_path):
    report, run, data = gliguard_run
    attach_gliguard_run(report, run, data)
    before = copy.deepcopy(report)
    output = tmp_path / "standalone" / "fastino_comparison.html"
    assert write_fastino_comparison(report, output, dpi=72) == output
    assert list(output.parent.iterdir()) == [output]
    page = output.read_text()
    assert "Fastino GLiDE | request: fastino/GLiDE" in page
    assert GLIGUARD_DISPLAY_NAME in page
    assert page.count('src="data:image/png;base64,') == 6
    assert page.count('class="metric-panel"') == 6
    assert "GLiGuard was not tested on PII" in page
    assert "17 successful / 18 rows; 1 errors" in page
    assert all(name not in page for name in ("TypeSafe", "Laya", "OpenJev", "Qwen3Guard"))
    assert report == before


@pytest.fixture
def poc_run(gliguard_run, tmp_path):
    report, original, data = gliguard_run
    attach_gliguard_run(report, original, data)
    run = tmp_path / "poc_run"
    (run / POC_PROVIDER).mkdir(parents=True)
    for path in (original / GLIGUARD_PROVIDER).glob("*.csv"):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row, latency in zip(rows, (100, 200, 900)):
            row.update(
                provider=POC_PROVIDER, requested_model=POC_MODEL,
                response_model="" if row["error"] else "glide_v2",
                http_status="400" if row["error"] else "200", latency_ms=latency,
            )
        with (run / POC_PROVIDER / path.name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status = json.loads((original / "run_status.json").read_text())
    status.update(
        provider=POC_PROVIDER, model=POC_MODEL, pii_excluded=True,
        response_models={"glide_v2": 17},
    )
    for dataset, details in status["datasets"].items():
        errors = int(dataset == "jailbreakhub")
        details.update(processed=3, valid=3 - errors, errors=errors)
    (run / "run_status.json").write_text(json.dumps(status))
    return report, run, data


def test_poc_reference_preserves_identity_coverage_and_latency_protocol(poc_run):
    report, run, data = poc_run
    before = copy.deepcopy(report)
    attach_poc_run(report, run, data)
    comparison = report["external_comparison"]
    assert report["runs"] == before["runs"]
    assert report["providers"] == before["providers"]
    assert [row for row in comparison["runs"] if row["provider"] != POC_PROVIDER] == before["external_comparison"]["runs"]
    identity = comparison["model_identities"][POC_PROVIDER]
    assert identity["request_model"] == POC_MODEL
    assert identity["resolved_model"] is None
    assert identity["response_model"] == "glide_v2"
    assert POC_PROVIDER not in external_reference_providers(comparison, "v2")
    records = {row["dataset"]: row for row in comparison["runs"] if row["provider"] == POC_PROVIDER}
    assert records["toxigen"]["mean_elapsed_ms"] == 400
    assert records["jailbreakhub"]["mean_elapsed_ms"] == 500
    assert all(row["latency_source_kind"] == "successful_request_latency_ms" for row in records.values())
    pii = next(row for row in category_reference_averages(report) if row["provider"] == POC_PROVIDER and row["category"] == "pii")
    assert all(pii[metric] is None for metric in METRICS)
    attach_poc_run(report, run, data)
    assert comparison["providers"].count(POC_PROVIDER) == 1
    assert len([row for row in comparison["runs"] if row["provider"] == POC_PROVIDER]) == 6


@pytest.mark.parametrize("invalid", [
    "running", "reference", "counts", "response_models", "requested_model", "latency",
])
def test_poc_reference_rejects_mismatched_sources_without_mutation(poc_run, invalid):
    report, run, data = poc_run
    status_path = run / "run_status.json"
    status = json.loads(status_path.read_text())
    if invalid == "running":
        status["status"] = "running"
    elif invalid == "reference":
        (data / DATASETS["toxigen"]).write_text("different reference")
    elif invalid == "counts":
        status["valid_rows"] += 1
    elif invalid == "response_models":
        status["response_models"] = {"other-model": 17}
    else:
        path = run / POC_PROVIDER / "toxigen_v1_t0.5.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["requested_model" if invalid == "requested_model" else "latency_ms"] = (
            "other-model" if invalid == "requested_model" else "nan"
        )
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status_path.write_text(json.dumps(status))
    before = copy.deepcopy(report)
    with pytest.raises(ValueError):
        attach_poc_run(report, run, data)
    assert report == before


def test_fastino_page_adds_poc_with_unverified_alias_and_untested_pii(poc_run, tmp_path):
    report, run, data = poc_run
    attach_poc_run(report, run, data)
    before = copy.deepcopy(report)
    output = tmp_path / "fastino_comparison.html"
    write_fastino_comparison(report, output, dpi=72)
    page = output.read_text()
    assert POC_DISPLAY_NAME in page and POC_MODEL in page
    assert "response model: glide_v2 (mapping unverified)" in page
    assert "17 successful / 18 rows; 1 errors" in page
    assert "The POC model was also not tested on PII" in page
    assert "successful-request CSV mean, not log elapsed/processed" in page
    assert page.count('src="data:image/png;base64,') == 6
    assert GLIGUARD_DISPLAY_NAME in page and "Fastino GLiDE | request:" in page
    assert all(name not in page for name in ("TypeSafe", "Laya", "OpenJev", "Qwen3Guard"))
    assert report == before


@pytest.fixture
def gliner25_run(poc_run, tmp_path):
    report, original, data = poc_run
    attach_poc_run(report, original, data)
    run = tmp_path / "gliner25_run"
    (run / GLINER25_PROVIDER).mkdir(parents=True)
    for path in (original / POC_PROVIDER).glob("*.csv"):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.update(
                provider=GLINER25_PROVIDER, requested_model=GLINER25_MODEL,
                response_model="" if row["error"] else GLINER25_MODEL,
            )
        with (run / GLINER25_PROVIDER / path.name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status = json.loads((original / "run_status.json").read_text())
    status.update(
        provider=GLINER25_PROVIDER, model=GLINER25_MODEL,
        response_models={GLINER25_MODEL: 17},
    )
    (run / "run_status.json").write_text(json.dumps(status))
    return report, run, data


def test_fastino_page_appends_gliner25_without_replacing_previous_models(gliner25_run, tmp_path):
    report, run, data = gliner25_run
    before = copy.deepcopy(report)
    attach_gliner25_run(report, run, data)
    comparison = report["external_comparison"]
    assert report["runs"] == before["runs"]
    assert [row for row in comparison["runs"] if row["provider"] != GLINER25_PROVIDER] == before["external_comparison"]["runs"]
    info = comparison["classification_models"][GLINER25_PROVIDER]
    assert (info["valid"], info["errors"]) == (17, 1)
    assert info["http_error_counts"] == {"400": 1}
    assert info["mapping_unverified"] is False
    assert comparison["model_identities"][GLINER25_PROVIDER]["resolved_model"] == GLINER25_MODEL
    assert GLINER25_PROVIDER not in external_reference_providers(comparison, "v2")
    pii = next(row for row in category_reference_averages(report) if row["provider"] == GLINER25_PROVIDER and row["category"] == "pii")
    assert all(pii[metric] is None for metric in METRICS)
    attach_gliner25_run(report, run, data)
    assert comparison["providers"].count(GLINER25_PROVIDER) == 1
    snapshot = copy.deepcopy(report)
    output = tmp_path / "fastino_comparison.html"
    write_fastino_comparison(report, output, dpi=72)
    page = output.read_text()
    assert all(name in page for name in (GLINER25_DISPLAY_NAME, POC_DISPLAY_NAME, GLIGUARD_DISPLAY_NAME))
    assert "Fastino GLiDE | request:" in page
    assert "response model: glide_v2 (mapping unverified)" in page
    assert "response model: fastino/gliner2.5-decide (mapping unverified)" not in page
    assert "HTTP 400: 1 errors." in page
    assert "GLiNER2.5-Decide was not tested on PII" in page
    assert page.count('src="data:image/png;base64,') == 6
    assert report == snapshot


def test_gliner25_reference_rejects_response_alias(gliner25_run):
    report, run, data = gliner25_run
    path = run / GLINER25_PROVIDER / "toxigen_v1_t0.5.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["response_model"] = "glide_v2"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = copy.deepcopy(report)
    with pytest.raises(ValueError, match="response identity"):
        attach_gliner25_run(report, run, data)
    assert report == before


def test_luna_summary_preserves_supplied_metrics_and_separates_protocols(report, tmp_path):
    source = Path(__file__).parents[1] / "data/external/luna_summary.csv"
    before = copy.deepcopy(report)
    attach_luna_summary(report, source)
    assert {key: value for key, value in report.items() if key != "luna_summary"} == before
    model = report["luna_summary"]
    records = {row["dataset"]: row for row in model["supplied_rows"]}
    assert records["toxigen"]["accuracy"] == 0.6799
    assert records["toxigen"]["auroc"] == 0.8185
    assert records["toxigen"]["f1"] == 0.5474
    assert records["toxigen"]["threshold"] == 0.5
    pii = records["pii"]
    assert (pii["rows"], pii["valid"], pii["errors"], pii["f1"]) == (990, 990, 0, 0.8668)
    assert pii["task"] == "document_level_multilabel"
    assert pii["f1_averaging"] == "micro"
    assert pii["accuracy"] is pii["auroc"] is pii["threshold"] is None
    assert all(row["prompt_version"] is row["scorer_version_id"] is row["mean_elapsed_ms"] is None for row in records.values())
    page = luna_summary_section(model, tmp_path)
    assert "not v1/v2" in page
    assert "original task-specific presets without custom prompts" in page
    assert "Other models use dataset-specific v1/v2 prompts" in page
    assert "Task-definition differences may affect results" in page
    assert "not binary F1 or entity-extraction F1" in page
    assert "Micro Precision" in page and "Micro Recall" in page and "Micro F1" in page
    assert "86.68%" in page and "N/A" in page
    assert model["pii_revision"] in page
    with (tmp_path / "luna_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    exported_pii = next(row for row in rows if row["dataset"] == "pii")
    assert exported_pii["accuracy"] == exported_pii["auroc"] == exported_pii["threshold"] == "N/A"
    assert len(rows) == 7


@pytest.mark.parametrize("invalid", ["duplicate", "counts", "metric", "pii_scope"])
def test_luna_summary_rejects_invalid_values_without_mutation(report, tmp_path, invalid):
    source = Path(__file__).parents[1] / "data/external/luna_summary.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if invalid == "duplicate":
        rows.append(rows[0].copy())
    elif invalid == "counts":
        rows[0]["errors"] = "1"
    elif invalid == "metric":
        rows[0]["f1"] = "nan"
    else:
        rows[0].update(rows="1000", valid="1000")
    path = tmp_path / "invalid.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = copy.deepcopy(report)
    with pytest.raises(ValueError):
        attach_luna_summary(report, path)
    assert report == before


def test_luna_reference_charts_preserve_native_protocol_and_other_models(fine_tuned_run):
    report, _, _, _ = fine_tuned_run
    before = copy.deepcopy(report)
    scopes_before = v1_reference_scope_averages(report)
    attach_luna_summary(report, Path(__file__).parents[1] / "data/external/luna_summary.csv")
    comparison = report["external_comparison"]
    assert report["runs"] == before["runs"]
    assert report["providers"] == before["providers"]
    assert [row for row in comparison["runs"] if row["provider"] != LUNA_PROVIDER] == before["external_comparison"]["runs"]
    native = [row for row in comparison["runs"] if row["provider"] == LUNA_PROVIDER]
    assert len(native) == 6 and all(row["prompt"] == "native_presets" for row in native)
    assert all(row["threshold"] == 0.5 and row["mean_elapsed_ms"] is None for row in native)
    assert not any(row["dataset"].startswith("pii") for row in native)
    assert LUNA_PROVIDER in external_reference_providers(comparison, "v1")
    assert LUNA_PROVIDER not in external_reference_providers(comparison, "v2")
    scopes = v1_reference_scope_averages(report)
    assert [row for row in scopes if row["provider"] != LUNA_PROVIDER] == scopes_before
    luna = {row["scope"]: row for row in scopes if row["provider"] == LUNA_PROVIDER}
    assert luna["non_pii"]["prompt"] == "native_presets"
    assert luna["non_pii"]["f1"] == pytest.approx((0.9555 + 0.5474 + 0.3678 + 0.4627 + 0.7656 + 0.9248) / 6)
    assert luna["non_pii"]["shared_elapsed_ms"] is None
    for scope in ("pii", "all"):
        assert all(luna[scope][metric] is None for metric in METRICS)
    focused = next(row for row in three_dataset_averages(report) if row["provider"] == LUNA_PROVIDER)
    assert focused["prompt"] == "native_presets"
    assert focused["accuracy"] == pytest.approx((0.6799 + 0.8967 + 0.7780) / 3)
    attach_luna_summary(report, Path(__file__).parents[1] / "data/external/luna_summary.csv")
    assert comparison["providers"].count(LUNA_PROVIDER) == 1
    assert len([row for row in comparison["runs"] if row["provider"] == LUNA_PROVIDER]) == 6


@pytest.mark.parametrize("kind", ["dataset", "category", "averages", "three"])
def test_luna_reference_bar_values_and_protocol_labels(fine_tuned_run, kind):
    report, _, _, _ = fine_tuned_run
    attach_luna_summary(report, Path(__file__).parents[1] / "data/external/luna_summary.csv")
    builders = {
        "dataset": lambda: external_dataset_figure(report, "v1"),
        "category": lambda: external_category_figure(report),
        "averages": lambda: external_averages_figure(report),
        "three": lambda: three_dataset_averages_figure(report),
    }
    figure = builders[kind]()
    try:
        figure.canvas.draw()
        text = "\n".join(item.get_text() for item in figure.texts)
        assert "Task-definition differences may affect results" in text
        if kind == "dataset":
            index = report["external_comparison"]["providers"].index(LUNA_PROVIDER) * len(DATASETS)
            assert figure.axes[0].patches[index].get_height() == pytest.approx(67.99)
            assert figure.axes[3].patches[index].get_height() == pytest.approx(54.74)
            assert math.isnan(figure.axes[0].patches[index + 6].get_height())
        elif kind == "category":
            index = report["external_comparison"]["providers"].index(LUNA_PROVIDER) * 3
            assert figure.axes[0].patches[index].get_height() == pytest.approx((67.99 + 89.67 + 77.80) / 3)
            assert math.isnan(figure.axes[0].patches[index + 2].get_height())
        elif kind == "three":
            labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
            assert "Luna (native presets)" in labels
        if kind != "three":
            legend = "\n".join(item.get_text() for item in figure.legends[0].texts)
            assert "Luna | original task-specific presets" in legend
            assert "prompt: N/A; binary threshold: 0.5" in legend
    finally:
        plt.close(figure)


def test_luna_summary_appears_in_gallery_and_reference_exports(fine_tuned_run, tmp_path):
    report, _, _, _ = fine_tuned_run
    attach_luna_summary(report, Path(__file__).parents[1] / "data/external/luna_summary.csv")
    before = copy.deepcopy(report)
    manifest = write_paper_figures(report, tmp_path, dpi=72)
    page = (tmp_path / "index.html").read_text()
    assert 'id="luna-summary"' in page
    assert "original summary.json" in page.lower()
    assert "Scorer version IDs and latency remain N/A" in page
    assert "Luna has a separate document-level multilabel PII summary below" in page
    assert "luna_summary.csv" in page
    assert manifest["luna_summary"] == report["luna_summary"]
    assert LUNA_PROVIDER in manifest["external_comparison"]["native_preset_models"]
    with (tmp_path / "three_dataset_averages.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(row for row in csv.DictReader(stream) if row["provider"] == LUNA_PROVIDER)
    assert row["prompt"] == "native_presets"
    assert float(row["f1"]) == pytest.approx((0.5474 + 0.9248 + 0.7656) / 3)
    assert "Luna: native presets (prompt N/A)" in page
    assert report == before


@pytest.fixture
def dense_reference_report(gliner25_run):
    report, run, data = gliner25_run
    attach_gliner25_run(report, run, data)
    comparison = report["external_comparison"]
    for provider in ("openjev-32b", "microsoft-decision-1"):
        comparison["providers"].append(provider)
        comparison["model_identities"][provider] = {
            "request_model": "not supplied", "resolved_model": None,
        }
        comparison["runs"].extend([
            {**row, "provider": provider}
            for row in comparison["runs"] if row["provider"] == "openjev" and row["prompt"] == "v1"
        ])
    external = Path(__file__).parents[1] / "data/external"
    attach_qwen_summary(report, external / "qwen3guard_stream_8b_summary.csv")
    attach_luna_summary(report, external / "luna_summary.csv")
    return report


@pytest.mark.parametrize("kind", ["dataset", "category", "averages", "three"])
def test_dense_reference_labels_legends_and_notes_do_not_overlap(dense_reference_report, kind):
    report = dense_reference_report
    builders = {
        "dataset": lambda: external_dataset_figure(report, "v1"),
        "category": lambda: external_category_figure(report),
        "averages": lambda: external_averages_figure(report),
        "three": lambda: three_dataset_averages_figure(report),
    }
    figure = builders[kind]()
    try:
        figure.suptitle(
            "Per-Dataset Multi-Model Reference Comparison (v1 / Threshold 0.5) | "
            "Luna: native presets (prompt N/A)",
            fontsize=15, y=0.98,
        )
        layout_reference_figure(figure)
        size = figure.get_size_inches().copy()
        layout_reference_figure(figure)
        assert figure.get_size_inches() == pytest.approx(size, abs=1 / figure.dpi)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            numbers = [
                text for text in axis.texts
                if text.get_text() and text.get_gid() != "reference-highlight-label"
            ]
            boxes = [text.get_window_extent(renderer) for text in numbers]
            assert all(box.width > 1 and box.height > 1 for box in boxes)
            assert all(not left.overlaps(right) for left, right in combinations(boxes, 2))
            assert all(box.x0 >= 0 and box.x1 <= figure.bbox.x1 for box in boxes)
        for legend in figure.legends:
            box = legend.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= figure.bbox.x1
            assert all(not box.overlaps(axis.get_tightbbox(renderer)) for axis in figure.axes)
        boxes = [text.get_window_extent(renderer) for text in figure.texts if text.get_text()]
        assert all(
            box.x0 >= 0 and box.y0 >= 0
            and box.x1 <= figure.bbox.x1 and box.y1 <= figure.bbox.y1
            for box in boxes
        )
        assert all(not left.overlaps(right) for left, right in combinations(boxes, 2))
    finally:
        plt.close(figure)


@pytest.mark.parametrize("invalid", ["model", "reference", "incomplete_log"])
def test_finetuned_reference_rejects_mismatched_sources(fine_tuned_run, invalid):
    report, run, log, data = fine_tuned_run
    if invalid == "model":
        path = run / "run_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["model"] = "fastino/GLiDE"
        path.write_text(json.dumps(metadata))
    elif invalid == "reference":
        (data / DATASETS["toxigen"]).write_text("different reference")
    else:
        log.write_text(log.read_text().replace("Progress: 3/3", "Progress: 2/3"))
    with pytest.raises(ValueError):
        attach_finetuned_run(report, run, log, data)


def test_finetuned_reference_exports_identity_counts_and_category_summaries(fine_tuned_run, tmp_path):
    report, run, log, data = fine_tuned_run
    attach_finetuned_run(report, run, log, data)
    destination = tmp_path / "figures"
    manifest = write_paper_figures(report, destination, dpi=72)
    provider = "fastino-ft-gg6-r1-ml2048-s7"
    model = "c94233af-98e6-4942-8e26-98ad204851bf"
    assert manifest["external_comparison"]["fine_tuned_models"][provider]["model_id"] == model
    for name in ("index.html", "category_reference_comparison.html", "three_dataset_comparison.html"):
        page = (destination / name).read_text()
        assert model in page and "1 errors" in page and "not GLiDE Noul" in page
        assert "gliguard-LLMGuardrails-300M-fine-tuned" in page
    for name in ("index.html", "category_reference_comparison.html"):
        assert '<span class="legend-swatch fine-tuned-swatch"' in (destination / name).read_text()
    with (destination / "fine_tuned_model_runs.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 and sum(int(row["errors"]) for row in rows) == 1
    assert all(row["model"] == model for row in rows)
    assert all(row["display_name"] == "gliguard-LLMGuardrails-300M-fine-tuned" for row in rows)
    with (destination / "fine_tuned_model_averages.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["category"] for row in rows} == {"harmful_content", "jailbreak", "pii", "non_pii"}
    assert all(row["model"] == model for row in rows)
    assert all(row["display_name"] == "gliguard-LLMGuardrails-300M-fine-tuned" for row in rows)
    assert next(row for row in rows if row["category"] == "pii")["accuracy"] == ""


@pytest.mark.parametrize("kind", ["dataset", "category", "averages", "three_dataset"])
def test_finetuned_bars_have_distinct_style_without_resizing_other_bars(fine_tuned_run, kind):
    report, run, log, data = fine_tuned_run
    attach_finetuned_run(report, run, log, data)
    if kind == "dataset":
        figure, groups = external_dataset_figure(report, "v1"), 8
    elif kind == "category":
        figure, groups = external_category_figure(report), 3
    elif kind == "averages":
        figure, groups = external_averages_figure(report), 1
    else:
        figure, groups = three_dataset_averages_figure(report), 1
    try:
        for ax in figure.axes:
            fine_tuned = ax.patches[groups:groups * 2]
            for bar in fine_tuned:
                assert bar.get_hatch() == "////"
                assert bar.get_linewidth() in (1.5, 2.2)
                assert bar.get_edgecolor() == to_rgba("#172438")
                assert bar.get_facecolor() == to_rgba("#E7298A")
                if kind == "three_dataset":
                    assert bar.get_height() == ax.patches[0].get_height()
                else:
                    assert bar.get_width() == ax.patches[0].get_width()
            assert all(bar.get_linewidth() in (0.4, 2.2) for bar in [
                *ax.patches[:groups], *ax.patches[groups * 2:],
            ])
        legend = figure.legends[0]
        handle = legend.get_patches()[0 if kind == "three_dataset" else 1]
        assert handle.get_hatch() == "////" and handle.get_linewidth() == 1.5
        assert any("Fine-tuned · 300M" in text.get_text() for text in legend.get_texts())
        first = figure.axes[0].patches[0]
        assert (first.get_width() if kind == "three_dataset" else first.get_height()) == 50
    finally:
        plt.close(figure)


@pytest.mark.parametrize("missing_pii", [False, True])
def test_v1_reference_scopes_separate_pii_and_preserve_undefined_or_missing_metrics(missing_pii):
    records, runs = [], []
    for provider in ("fastino.ai", "openjev"):
        for dataset in DATASETS:
            pii = dataset.startswith("pii_")
            value = 0.9 if pii else 0.6
            row = {
                "provider": provider, "dataset": dataset, "prompt": "v1", "threshold": 0.5,
                "status": "complete", "rows": 1000 if pii else 10, "valid": 10,
                "source_kind": "local_predictions" if provider == "fastino.ai" else "user_supplied_aggregate",
                **{metric: value for metric in ("accuracy", "precision", "recall", "f1")},
                "auroc": None if pii else 0.8,
                "mean_elapsed_ms": 400 if pii else 300,
            }
            if provider == "fastino.ai":
                runs.append({**row, "mean_elapsed_ms": 200 if pii else 100})
                if dataset == "jailbreak-classification":
                    runs[-1]["mean_elapsed_ms"] = None
                row["mean_elapsed_ms"] = 9999
            if not (missing_pii and provider == "fastino.ai" and dataset == "pii_masking_mini_10k"):
                records.append(row)
            records.append({**row, "prompt": "v2", "accuracy": 0.01})
    report = {
        "runs": runs,
        "external_comparison": {"providers": ["fastino.ai", "openjev"], "runs": records},
    }
    summary = v1_reference_scope_averages(report)
    grouped = {(row["scope"], row["provider"]): row for row in summary}
    assert len(summary) == 6
    assert all(row["prompt"] == "v1" and row["threshold"] == 0.5 for row in summary)
    main = grouped[("non_pii", "fastino.ai")]
    assert len(main["datasets"]) == 6
    assert len(main["timing_datasets"]) == 5
    assert main["accuracy"] == pytest.approx(0.6)
    assert main["shared_elapsed_ms"] == 100
    assert main["mean_elapsed_ms"] is None
    pii = grouped[("pii", "fastino.ai")]
    all_datasets = grouped[("all", "fastino.ai")]
    assert len(pii["datasets"]) == 2
    assert len(all_datasets["datasets"]) == 8
    assert pii["auroc"] is all_datasets["auroc"] is None
    if missing_pii:
        assert pii["accuracy"] is all_datasets["accuracy"] is None
        assert grouped[("pii", "openjev")]["accuracy"] is None
    else:
        assert pii["accuracy"] == pytest.approx(0.9)
        assert pii["shared_elapsed_ms"] == 200
        assert all_datasets["accuracy"] == pytest.approx(0.675)
        assert len(all_datasets["timing_datasets"]) == 7
        assert all_datasets["shared_elapsed_ms"] == pytest.approx((100 * 5 + 200 * 2) / 7)
        assert grouped[("all", "openjev")]["mean_elapsed_ms"] == 325


def test_internal_models_are_adjacent_in_v1_only_and_missing_toxigen_stays_na(report, tmp_path):
    report["runs"] = [row for row in report["runs"] if row["dataset"] != "hatecheck"]
    for provider in report["providers"]:
        template = next(row for row in report["runs"] if row["provider"] == provider and row["dataset"] == "toxigen")
        report["runs"].extend({**template, "dataset": dataset} for dataset in ("hatecheck", "aegis2"))
    summary = tmp_path / "summary.csv"
    fields = ["dataset", "prompt", "threshold", "rows", "accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms"]
    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for prompt in ("v1", "v2"):
            for dataset in ("toxigen", "hatecheck", "aegis2"):
                writer.writerow({
                    "dataset": dataset, "prompt": prompt, "threshold": 0.5, "rows": 10,
                    "accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8,
                    "auroc": 0.8, "mean_elapsed_ms": 900,
                })
    internal = tmp_path / "internal_model_summary.csv"
    with internal.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields, "provider", "valid", "errors"])
        writer.writeheader()
        for provider in ("openjev-32b", "microsoft-decision-1"):
            for dataset in ("toxigen", "hatecheck", "aegis2"):
                writer.writerow({
                    "provider": provider, "dataset": dataset, "prompt": "v1", "threshold": 0.5,
                    "rows": 10, "valid": 7, "errors": 3,
                    **{
                        metric: "" if provider == "openjev-32b" and dataset == "toxigen" else 0.9
                        for metric in ("accuracy", "precision", "recall", "f1", "auroc")
                    },
                    "mean_elapsed_ms": 1100,
                })
    attach_openjev_summary(report, summary)
    figure = external_dataset_figure(report, "v1")
    try:
        assert len(figure.axes[0].patches) == 48
        for ax in figure.axes:
            assert all(text.get_gid() != "reference-highlight-label" for text in ax.texts)
            assert all(text.arrow_patch is None for text in ax.texts)
            assert all(bar.get_linewidth() == 0.4 for bar in ax.patches)
        assert np.isnan(figure.axes[0].patches[32].get_height())
        assert figure.axes[0].patches[33].get_height() == 90
        assert figure.axes[5].patches[32].get_height() == 1100
        assert figure.axes[0].patches[40].get_height() == 90
        names = [text.get_text().split(" |")[0] for text in figure.legends[0].texts]
        assert names[-3:] == ["OpenJev 4B", "OpenJev 32B", "Microsoft Decision-1"]
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend = figure.legends[0].get_window_extent(renderer)
        assert legend.x0 >= 0 and legend.x1 <= figure.bbox.width
        positions = [ax.get_position() for ax in figure.axes]
        assert all(position.width > 0.85 for position in positions)
        assert all(positions[index].y0 > positions[index + 1].y1 for index in range(5))
        assert legend.y0 > max(ax.get_window_extent(renderer).y1 for ax in figure.axes)
        assert figure.texts[-1].get_window_extent(renderer).y1 < min(
            ax.get_window_extent(renderer).y0 for ax in figure.axes
        )
        assert figure.axes[5].get_yscale() == "log"
        assert "log scale" in figure.axes[5].get_ylabel()
        assert all(text.get_fontsize() >= 10 for ax in figure.axes for text in ax.texts)
        missing = next(text for text in figure.axes[5].texts if text.get_text() == "N/A")
        assert missing.xy[1] >= figure.axes[5].get_ylim()[0]
    finally:
        plt.close(figure)
    figure = external_dataset_figure(report, "v2")
    try:
        assert len(figure.axes[0].patches) == 32
        assert any(text.get_gid() == "reference-highlight-label" for text in figure.axes[0].texts)
        assert all("32B" not in text.get_text() and "Decision-1" not in text.get_text()
                   for text in figure.legends[0].texts)
    finally:
        plt.close(figure)
    averages = three_dataset_averages(report)
    row = next(row for row in averages if row["provider"] == "openjev-32b")
    assert row["datasets"] == ["toxigen", "hatecheck", "aegis2"]
    assert row["accuracy"] is row["f1"] is None
    assert row["mean_elapsed_ms"] == 1100


@pytest.mark.parametrize("precision,recall,expected", [
    (0.2, 0.8, 0.32),
    (0.0, 0.8, 0.0),
    (0.0, 0.0, 0.0),
])
def test_missing_supplied_f1_uses_harmonic_mean(monkeypatch, precision, recall, expected):
    monkeypatch.setattr("benchmark.external_summary.FOCUSED_SUPPLIED_AVERAGES", (
        ("Supplied model", 0.5, precision, recall, None, 100),
    ))
    report = {"runs": [], "external_comparison": {"runs": [], "providers": []}}
    row, = three_dataset_averages(report)
    assert row["f1"] == pytest.approx(expected)
    assert row["auroc"] is None


def test_three_dataset_averages_use_only_v1_half_threshold_timing():
    records, runs = [], []
    for provider in ("fastino.ai", "openjev"):
        for dataset in ("toxigen", "hatecheck", "aegis2"):
            for prompt in ("v1", "v2"):
                records.append({
                    "provider": provider, "dataset": dataset, "prompt": prompt,
                    "status": "complete", "valid": 10, "accuracy": 0.8 if prompt == "v1" else 0.1,
                    "precision": 0.8, "recall": 0.8, "f1": 0.65, "auroc": 0.9,
                    "mean_elapsed_ms": 800 if provider == "openjev" else 9000,
                })
                for threshold in (0.5, 0.7):
                    runs.append({
                        "provider": provider, "dataset": dataset, "prompt": prompt,
                        "threshold": threshold, "status": "complete",
                        "mean_elapsed_ms": 200 if prompt == "v1" and threshold == 0.5 else 9999,
                    })
    report = {
        "runs": runs,
        "external_comparison": {"runs": records, "providers": ["fastino.ai", "openjev"]},
    }
    averages = three_dataset_averages(report)
    assert [row["prompt"] for row in averages[:2]] == ["v1", "v1"]
    assert [row["accuracy"] for row in averages[:2]] == pytest.approx([0.8, 0.8])
    assert [row["f1"] for row in averages[:2]] == pytest.approx([0.65, 0.65])
    assert all(row["f1_source"] == "mean_per_dataset" for row in averages[:2])
    assert [row["mean_elapsed_ms"] for row in averages[:2]] == [200, 800]
    runs[0]["status"] = "partial"
    assert three_dataset_averages(report)[0]["mean_elapsed_ms"] is None
