import json

import pandas as pd
import pytest

from scripts.report_heldout import build
from scripts.rerun_classification import digest, export_results, load_input
from scripts.rerun_pii_extraction import successful_checkpoint


def test_report_requires_full_provider_matrix_and_rejects_legacy_runs(tmp_path):
    data, root = tmp_path / "data", tmp_path / "outputs"
    data.mkdir()
    path = data / "hatecheck.csv"
    provenance = {"source_repository": "test/data", "source_config": "default", "source_revision": "pinned",
                  "source_split": "test", "partition_role": "held_out"}
    pd.DataFrame([{**provenance, "source_row": i, "sample_id": str(i), "text": f"sample{i}", "label": i}
                  for i in range(2)]).to_csv(path, index=False)
    metadata = {**provenance, "prepared_rows": 2, "csv_sha256": digest(path)}
    path.with_suffix(".manifest.json").write_text(json.dumps(metadata))
    samples, _ = load_input(path, True)
    folder = root / "classification/fastino.ai"
    folder.mkdir(parents=True)
    for version in ("v1", "v2"):
        cached = {str(i): {"sample_id": str(i), "score": .1 if i == 0 else .9, "error": "", "resolved_model": "model", "latency_ms": 10}
                  for i in range(2)}
        csv_path = folder / f"hatecheck_{version}.csv"
        result = export_results(samples, cached, csv_path, "fastino.ai", "hatecheck", version, "model")
        manifest = {"dataset": "hatecheck", "prompt_version": version, "require_heldout": True,
                    "evaluation_partition": metadata, "result": result}
        sidecar = csv_path.with_suffix(".manifest.json")
        sidecar.write_text(json.dumps(manifest))
        report = build(root, data, ["hatecheck"])
        assert len(report["classification"]) == (0 if version == "v1" else 2)
    manifest["require_heldout"] = False
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared held-out source"):
        build(root, data, ["hatecheck"])


def test_extraction_resume_reuses_only_successful_known_ids(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    rows = [{"sample_id": "ok", "error": ""}, {"sample_id": "retry", "error": "HTTPStatusError HTTP 503"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert set(successful_checkpoint(path, {"ok", "retry"})) == {"ok"}
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        successful_checkpoint(path, {"retry"})
