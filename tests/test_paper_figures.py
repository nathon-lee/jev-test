import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmark.comparison import MODEL_IDENTITIES, MODELS
from benchmark.paper_figures import (
    coverage_figure,
    quality_figure,
    threshold_figure,
    write_paper_figures,
)


@pytest.fixture
def report():
    providers = list(MODELS)
    paired, runs, overview = [], [], []
    for index, provider in enumerate(providers):
        for threshold, value in ((0.5, 0.6 + index * 0.1), (0.7, 0.5 + index * 0.1)):
            paired.append({
                "provider": provider, "dataset": "toxigen", "prompt": "v1",
                "threshold": threshold, "n": 10, "f1": value, "auroc": 0.8,
            })
        overview.append({
            "provider": provider, "macro_f1": 0.6 + index * 0.1, "macro_auroc": 0.8,
            "macro_fpr": 0.2, "macro_fnr": 0.3,
            "baseline_elapsed_ms": None if index == 1 else 200 + index * 100,
        })
        runs.append({
            "provider": provider, "dataset": "toxigen", "prompt": "v1", "threshold": 0.5,
            "rows": 10, "expected": 10, "valid": 10, "coverage": 1.0, "status": "complete",
            # Intentionally different: fair quality plots must use paired, not per-file metrics.
            "f1": 0.01,
        })
    runs.extend([
        {
            "provider": providers[0], "dataset": "hatecheck", "prompt": "v1", "threshold": 0.5,
            "rows": 10, "expected": 10, "valid": 0, "coverage": 0.0, "status": "complete",
        },
        {
            "provider": providers[1], "dataset": "hatecheck", "prompt": "v1", "threshold": 0.5,
            "rows": 0, "expected": 10, "valid": 0, "coverage": 0.0, "status": "missing",
        },
    ])
    return {
        "generated_at": "2026-10-21T00:00:00+00:00", "providers": providers,
        "model_identities": MODEL_IDENTITIES, "baseline_datasets": ["toxigen"],
        "overview": overview, "paired": paired, "runs": runs,
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


def test_export_all_figure_formats_and_reject_empty_comparison(report, tmp_path):
    with pytest.raises(ValueError, match="at least 72"):
        write_paper_figures(report, tmp_path, dpi=10)
    with pytest.raises(ValueError, match="No shared valid baseline"):
        write_paper_figures({**report, "baseline_datasets": []}, tmp_path)
    manifest = write_paper_figures(report, tmp_path, dpi=72)
    assert len(manifest["figures"]) == 4
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
    assert gallery.count("<img ") == 4
    assert "https://" not in gallery
    table = (tmp_path / "table01_baseline.tex").read_text()
    assert r"\toprule" in table
    assert r"glide\_v2" in table
    assert "unconfirmed (API response required)" in table
    assert (tmp_path / "paper_figures.pdf").read_bytes().startswith(b"%PDF")
    saved = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert saved["source_generated_at"] == report["generated_at"]
    assert saved["model_identities"] == report["model_identities"]
    assert not plt.get_fignums()
