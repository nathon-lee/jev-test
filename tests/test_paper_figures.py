import base64
import csv
import html
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmark.comparison import DATASETS, MODEL_IDENTITIES, MODELS
from benchmark.external_summary import (
    attach_openjev_summary, reference_averages, three_dataset_averages, v1_reference_scope_averages,
)
from benchmark.paper_figures import (
    coverage_figure,
    dataset_metrics_figure,
    english_warning,
    external_averages_figure,
    external_dataset_figure,
    model_averages_figure,
    quality_figure,
    stability_averages_figure,
    stability_figure,
    threshold_figure,
    three_dataset_averages_figure,
    tradeoff_figure,
    write_paper_figures,
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
    assert len(manifest["figures"]) == 13
    assert manifest["external_comparison"]["source_kind"] == "user_supplied_aggregate"
    assert manifest["model_identities"] == report["model_identities"]
    gallery = (destination / "index.html").read_text()
    assert gallery.count("<img ") == 25
    assert "not a sample-aligned baseline" in gallery
    assert '<html lang="en">' in gallery
    assert not re.search(r"[\u3400-\u9fff]", gallery)
    assert 'href="../openjev_summary.csv"' in gallery
    for figure in manifest["figures"][6:9]:
        assert f'<img src="{figure["files"]["svg"]}"' not in gallery
        assert f'<a href="{figure["files"]["svg"]}">{figure["title"]}</a>' in gallery
    for figure in manifest["figures"][9:12]:
        if figure["name"] in ("fig10_openjev_reference_v1", "fig12_openjev_reference_averages"):
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
    assert re.findall(r"<section><(?:h2>|details><summary>)([^<]+)", gallery)[:4] == [
        focused["title"], v1_reference["title"], reference_averages_title, coverage_title,
    ]
    assert len(v1_reference["panels"]) == 6
    v1_card = gallery.split("<section><h2>" + v1_reference["title"] + "</h2>", 1)[1].split("</section>", 1)[0]
    primary, detailed = v1_card.split("<details>", 1)
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
    assert all(row["prompt"] == "not_supplied" and row["f1"] == "" for row in supplied.values())
    assert float(supplied["Dynamo CPU"]["mean_elapsed_ms"]) == 20000
    assert float(supplied["Dynamo GPU (A100)"]["mean_elapsed_ms"]) == 55
    assert float(supplied["Dynamo GPU (L4)"]["mean_elapsed_ms"]) == 100
    assert supplied["Mistral"]["auroc"] == supplied["Warden"]["auroc"] == ""
    assert float(supplied["Mistral"]["recall"]) == 0.834
    assert "Three-dataset Multi-model Reference Averages" in page
    assert "ToxiGen, HateCheck, AEGIS2: Multi-model Reference Averages" in page
    assert "timing protocols were not supplied" in page
    assert "Whether that dataset contributed to their supplied averages has not been confirmed" in html.unescape(page)
    reference_card = gallery.split(
        "<section><h2>" + manifest["figures"][11]["title"] + "</h2>", 1,
    )[1].split("</section>", 1)[0]
    assert "PII presence detection" in reference_card
    assert "All eight datasets" in reference_card
    assert 'href="v1_reference_scope_averages.csv"' in reference_card
    assert re.findall(r'data-metric="([^"]+)"', reference_card.split("<details>", 1)[0]) == [
        "accuracy", "f1", "precision", "recall", "auroc", "mean_elapsed_ms",
    ]
    assert 'class="reference-grid"' in reference_card.split("<details>", 1)[0]
    assert reference_card.index('class="model-legend"') < reference_card.index("<img ")
    assert reference_card.count('class="model-legend"') == 1
    assert len(manifest["figures"][11]["panels"]) == 6
    for panel in manifest["figures"][11]["panels"]:
        assert f'<img src="{panel["files"]["svg"]}"' in reference_card
        for filename in panel["files"].values():
            assert (destination / filename).stat().st_size > 1000
    before_methodology, methodology = reference_card.rsplit("<details>", 1)
    assert methodology.startswith("<summary>Methodology and data scope</summary>")
    assert html.escape(manifest["figures"][11]["caption"]) in methodology
    assert html.escape(manifest["figures"][11]["caption"]) not in before_methodology
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
        assert all(np.isnan(bar.get_width()) for bar in figure.axes[3].patches[4:])
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
            assert all(label.get_fontsize() >= 11 for label in ax.texts)
            labels = [label.get_window_extent(renderer) for label in ax.texts]
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
        assert all("32B" not in text.get_text() and "Decision-1" not in text.get_text()
                   for text in figure.legends[0].texts)
    finally:
        plt.close(figure)
    averages = three_dataset_averages(report)
    row = next(row for row in averages if row["provider"] == "openjev-32b")
    assert row["datasets"] == ["toxigen", "hatecheck", "aegis2"]
    assert row["accuracy"] is row["f1"] is None
    assert row["mean_elapsed_ms"] == 1100


def test_three_dataset_averages_use_only_v1_half_threshold_timing():
    records, runs = [], []
    for provider in ("fastino.ai", "openjev"):
        for dataset in ("toxigen", "hatecheck", "aegis2"):
            for prompt in ("v1", "v2"):
                records.append({
                    "provider": provider, "dataset": dataset, "prompt": prompt,
                    "status": "complete", "valid": 10, "accuracy": 0.8 if prompt == "v1" else 0.1,
                    "precision": 0.8, "recall": 0.8, "f1": 0.8, "auroc": 0.9,
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
    assert [row["mean_elapsed_ms"] for row in averages[:2]] == [200, 800]
    runs[0]["status"] = "partial"
    assert three_dataset_averages(report)[0]["mean_elapsed_ms"] is None
