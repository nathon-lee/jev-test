import json

import pytest

from benchmark.heldout_figures import METRICS, averages, publish


def test_macro_means_use_exact_declared_scope_and_equal_dataset_weights():
    rows = [{"provider": "fastino.ai", "prompt": "v1", "dataset": d, "valid": n,
             **{m: value for m in METRICS}}
            for d, n, value in (("hatecheck", 100, .9), ("aegis2", 10, .6), ("jailbreak-classification", 1, .3))]
    report = {"sources": {r["dataset"]: {} for r in rows}, "classification": rows}
    assert averages(report)[0]["f1"] == pytest.approx(.6)
    rows[0]["auroc"] = None
    assert averages(report)[0]["auroc"] is None
    report["classification"] = rows[:-1]
    with pytest.raises(ValueError, match="full declared dataset scope"):
        averages(report)
    report["classification"] = rows + [rows[0]]
    with pytest.raises(ValueError, match="full declared dataset scope"):
        averages(report)


def test_figures_cannot_be_published_without_verified_common_rows(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"same_successful_rows_verified": False}))
    with pytest.raises(ValueError, match="verified matching successful rows"):
        publish(tmp_path)
