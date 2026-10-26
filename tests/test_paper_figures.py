import base64
import csv
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmark.comparison import MODEL_IDENTITIES, MODELS
from benchmark.external_summary import attach_openjev_summary, three_dataset_averages
from benchmark.paper_figures import (
    coverage_figure,
    dataset_metrics_figure,
    external_dataset_figure,
    model_averages_figure,
    quality_figure,
    stability_averages_figure,
    stability_figure,
    threshold_figure,
    three_dataset_averages_figure,
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


def test_quality_chart_uses_paired_values_and_names(report):
    figure = quality_figure(report)
    try:
        heights = [bar.get_height() for bar in figure.axes[0].patches]
        assert heights == pytest.approx([60, 70, 80])
        legend = "\n".join(text.get_text() for text in figure.legends[0].texts)
        assert "request: jev-latest" in legend
        assert "resolved: unconfirmed (API response required)" in legend
        assert "resolved: glide_v2" in legend
        assert "serving: laya-rl-agent" in legend
        assert "routed: typed-decisions" in legend
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


def test_coverage_distinguishes_all_failed_from_missing(report):
    figure = coverage_figure(report)
    try:
        values = figure.axes[0].images[0].get_array()
        assert values[1, 0] == 0  # HateCheck, Fastino: all requests failed.
        assert np.ma.is_masked(values[1, 1])  # HateCheck, Laya: no predictions.
        labels = [text.get_text() for text in figure.axes[0].texts]
        assert any("ALL FAILED" in label for label in labels)
        assert "NOT AVAILABLE" in labels
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
        assert times[0] == 300  # Not baseline_elapsed_ms (sample-weighted).
        assert np.isnan(times[1])
        assert times[2] == 500
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


def test_export_all_figure_formats_and_reject_empty_comparison(report, tmp_path):
    with pytest.raises(ValueError, match="at least 72"):
        write_paper_figures(report, tmp_path, dpi=10)
    with pytest.raises(ValueError, match="No shared valid baseline"):
        write_paper_figures({**report, "baseline_datasets": []}, tmp_path)
    with pytest.raises(ValueError, match="Regenerate"):
        write_paper_figures({key: value for key, value in report.items() if key != "stability"}, tmp_path)
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
    assert gallery.count("<img ") == 6
    embedded = re.search(
        r'<img src="data:image/png;base64,([^"]+)" alt="共同样本分类能力对比">', gallery,
    )
    assert embedded is not None
    assert base64.b64decode(embedded.group(1), validate=True) == (tmp_path / "fig01_paired_quality.png").read_bytes()
    assert '<img src="fig01_paired_quality.svg"' not in gallery
    assert 'href="fig01_paired_quality.svg"' in gallery
    assert 'href="fig01_paired_quality.pdf"' in gallery
    for figure in manifest["figures"][-3:]:
        assert f'<img src="{figure["files"]["svg"]}"' not in gallery
        assert f'<a href="{figure["files"]["svg"]}">{figure["title"]}</a>' in gallery
    assert "逐数据集六项指标" in gallery
    assert "模型平均指标" in gallery
    assert "重复调用一致性：v1" in gallery
    assert "重复调用一致性：v2" in gallery
    assert "重复调用一致性：共同样本模型均值" in gallery
    assert 'href="../stability.csv"' in gallery
    assert "https://" not in gallery
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
    attach_openjev_summary(report, path)
    figure = external_dataset_figure(report, "v1")
    try:
        assert figure.axes[0].patches[24].get_height() == pytest.approx(81)
        assert figure.axes[5].patches[24].get_height() == 900
        assert np.isnan(figure.axes[5].patches[0].get_height())  # No complete four-run timing.
        legend = "\n".join(text.get_text() for text in figure.legends[0].texts)
        assert "OpenJev | supplied internal summary" in legend
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
    assert gallery.count("<img ") == 9
    assert "非共同样本基线" in gallery
    assert 'href="../openjev_summary.csv"' in gallery
    for figure in manifest["figures"][6:9]:
        assert f'<img src="{figure["files"]["svg"]}"' not in gallery
        assert f'<a href="{figure["files"]["svg"]}">{figure["title"]}</a>' in gallery
    for figure in manifest["figures"][9:12]:
        assert f'<img src="{figure["files"]["svg"]}"' in gallery
        for filename in figure["files"].values():
            assert (destination / filename).stat().st_size > 1000
    focused = manifest["figures"][-1]
    assert focused["name"] == "fig13_three_dataset_averages"
    assert f'<img src="{focused["files"]["svg"]}"' not in gallery
    assert 'href="three_dataset_comparison.html"' in gallery
    page = (destination / "three_dataset_comparison.html").read_text()
    assert page.count("<img ") == 1
    encoded = re.search(r'<img src="data:image/png;base64,([^"]+)"', page)
    assert encoded is not None
    assert base64.b64decode(encoded.group(1), validate=True) == (destination / focused["files"]["png"]).read_bytes()
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
    figure = three_dataset_averages_figure(report)
    try:
        assert len(figure.axes[0].patches) == 9
        assert [text.get_text() for text in figure.axes[0].get_yticklabels()][:4] == [
            "Fastino", "Laya", "TypeSafe", "OpenJev",
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
    finally:
        plt.close(figure)
    assert not plt.get_fignums()


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
