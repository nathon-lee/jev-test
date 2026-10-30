import json

import pandas as pd
import pytest

from scripts.publish_corrected_comparison import QUALITY, collect_rows, macro_rows, render_html
from scripts.report_classification_rerun import build_report
from scripts.rerun_classification import export_results


def make_run(root, dataset="hatecheck", provider="fastino.ai"):
    folder = root / provider
    folder.mkdir(parents=True, exist_ok=True)
    samples = [{"sample_id": str(i), "text": f"sample {i}", "label": label}
               for i, label in enumerate([1, 0, 1, 0])]
    cached = {str(i): {"sample_id": str(i), "score": score, "error": "",
                       "resolved_model": "model", "latency_ms": 10}
              for i, score in enumerate([.9, .7, .6, .1])}
    path = folder / f"{dataset}_v1.csv"
    result = export_results(samples, cached, path, provider, dataset, "v1", "requested-model")
    manifest = {"provider": provider, "dataset": dataset, "prompt_version": "v1", "workers": 1, "result": result}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return path


def test_macro_requires_exact_dataset_scope_and_uses_equal_weights():
    rows = [{"provider": "fastino.ai", "prompt": "v1", "dataset": d,
             "paired_rows": n, **{m: value for m in QUALITY}}
            for d, n, value in (("toxigen", 100, .2), ("hatecheck", 10, .8), ("aegis2", 1, .5))]
    assert macro_rows(rows)[0]["f1"] == pytest.approx(.5)
    assert macro_rows(rows[:-1]) == []
    assert macro_rows(rows + [rows[0]]) == []


def test_published_html_renders_real_tables_images_and_links():
    rendered = render_html('# Title\n\n| a | b |\n|---|---|\n| 1 | <unsafe> |\n\n![Figure](figures/test.png)\n\n[CSV](values.csv)')
    assert '<h1>Title</h1>' in rendered
    assert '<th>a</th>' in rendered and '<td>1</td>' in rendered
    assert '<td>&lt;unsafe&gt;</td>' in rendered
    assert 'src="figures/test.png"' in rendered
    assert '<a href="values.csv">CSV</a>' in rendered


def test_running_provider_is_not_published_as_completed_and_resolved_billing_is_not_blocked(tmp_path):
    make_run(tmp_path)
    make_run(tmp_path, provider="laya-ai.com")
    (tmp_path / "provider_status.json").write_text(json.dumps({"laya-ai.com": {
        "status": "running", "expected_configurations": [["hatecheck", "v1"], ["aegis2", "v1"]]}}))
    (tmp_path / "access_restrictions.json").write_text(json.dumps([
        {"provider": "laya-ai.com", "http_status": 402, "status": "resolved"}]))
    report = build_report(tmp_path, tmp_path / "no-reference")
    assert report["completed_providers"] == ["fastino.ai"]
    assert report["access_restrictions"] == []
    assert report["provider_status"]["laya-ai.com"]["status"] == "running"
    assert "blocked by HTTP 402" not in (tmp_path / "README.md").read_text()
    make_run(tmp_path, dataset="aegis2", provider="laya-ai.com")
    report = build_report(tmp_path, tmp_path / "no-reference")
    assert report["provider_status"]["laya-ai.com"]["status"] == "complete"


def test_comparison_computes_average_precision_from_verified_scores_and_rejects_tampering(tmp_path):
    root = tmp_path / "rerun"
    path = make_run(root)
    historical = tmp_path / "historical"
    historical.mkdir()
    reference = historical / "hatecheck_v1_t0.5.csv"
    reference.write_bytes(path.read_bytes())
    report = build_report(root, historical)
    rows, thresholds, timings, pii = collect_rows(report, root, historical)
    assert rows[0]["auprc"] == pytest.approx(5 / 6)
    assert rows[0]["paired_rows"] == 4
    assert len(rows) == 2
    assert len(thresholds) == 4
    assert not pii
    assert timings[0]["workers"] == 1
    frame = pd.read_csv(reference)
    frame.loc[0, "score"] = .8
    frame.to_csv(reference, index=False)
    with pytest.raises(ValueError, match="digest"):
        collect_rows(report, root, historical)


def test_provider_failures_use_shared_successful_rows_for_every_quality_score(tmp_path):
    from scripts.rerun_classification import digest
    root = tmp_path / "rerun"
    first = make_run(root)
    second = make_run(root, provider="laya-ai.com")
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "hatecheck_v1_t0.5.csv").write_bytes(first.read_bytes())
    frame = pd.read_csv(second)
    # This false positive must disappear for *all* providers, including Jev.
    frame.loc[1, ["score", "prediction"]] = [None, None]
    frame["error"] = frame["error"].astype(object)
    frame.loc[1, "error"] = "HTTPStatusError"
    frame["http_status"] = 200
    frame.loc[1, "http_status"] = 413
    frame.to_csv(second, index=False)
    manifest_path = second.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["result"].update(file_sha256=digest(second), valid=3, errors=1)
    manifest_path.write_text(json.dumps(manifest))
    report = build_report(root, historical)
    rows, _, timings, _ = collect_rows(report, root, historical)
    assert len(rows) == 3
    assert {r["paired_rows"] for r in rows} == {3}
    assert {r["f1"] for r in rows} == {1.0}
    assert sum(r["errors"] for r in timings) == 1
    # A digest-consistent input mismatch must still fail, not be hidden by alignment.
    frame.loc[0, "text"] = "different request"
    frame.to_csv(second, index=False)
    report["runs"][1]["result"]["file_sha256"] = digest(second)
    with pytest.raises(ValueError, match="mismatch"):
        collect_rows(report, root, historical)
