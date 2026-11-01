import json
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmark import dataset_partitions, datasets, provider_suite
from benchmark.comparison import collect_report, read_reference
from benchmark.dataset_partitions import attach_provenance, load_evaluation_partition, select_split
from scripts import prepare_datasets
from scripts.pii_extraction import main as pii_main


@pytest.mark.parametrize("available,expected", [
    (["train", "test", "validation"], "test"),
    (["train", "validation"], "validation"), (["test"], "test"),
])
def test_defaults_prefer_official_held_out_splits(available, expected):
    assert select_split(available) == (expected, "held_out")


@pytest.mark.parametrize("requested", [None, "train", "train+test", "train[:10%]", "missing"])
def test_training_and_union_splits_require_or_refuse_explicit_policy(requested):
    with pytest.raises(ValueError):
        select_split(["train"], requested)


def test_training_reference_requires_opt_in_and_is_never_called_held_out():
    assert select_split(["train"], allow_train_eval=True) == ("train", "training_reference")
    assert select_split(["train", "test"], allow_train_eval=True) == ("test", "held_out")
    assert select_split(["train", "test"], "train", True) == ("train", "training_reference")


def test_dedicated_pii_workflow_also_requires_train_opt_in(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        pii_main(["prepare", "--dataset", "ai4privacy/pii-masking-nano-1k", "--split", "train",
                  "--output", str(tmp_path / "pii.jsonl")])
    assert not (tmp_path / "pii.jsonl").exists()


def provenance(config="default"):
    return {"source_repository": "example/data", "source_config": config,
            "source_revision": "immutable-sha", "source_split": "test", "partition_role": "held_out",
            "available_splits": ["train", "test"], "source_rows": 2}


def test_source_revision_is_resolved_before_split_inspection_and_download(monkeypatch):
    calls = []
    monkeypatch.setattr(dataset_partitions, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda repo, revision: SimpleNamespace(sha="immutable-sha")))
    def splits(repo, **kwargs):
        calls.append(("splits", kwargs))
        return ["train", "validation"]
    def load(repo, config, **kwargs):
        calls.append(("load", kwargs))
        return [1, 2]
    monkeypatch.setattr(dataset_partitions, "get_dataset_split_names", splits)
    monkeypatch.setattr(dataset_partitions, "load_dataset", load)
    _, metadata = load_evaluation_partition("example/data", revision="release-tag")
    assert calls == [("splits", {"config_name": None, "revision": "immutable-sha"}),
                     ("load", {"split": "validation", "revision": "immutable-sha"})]
    assert metadata["source_revision"] == "immutable-sha"


def test_toxigen_uses_one_annotated_split_and_retains_label_lookup_semantics(monkeypatch):
    annotated = pd.DataFrame({"text": ["harmful example", "benign example"],
                              "toxicity_human": [4.0, 1.0], "toxicity_ai": [3.5, 1.5]})
    source = SimpleNamespace(to_pandas=lambda: annotated.copy())
    calls = []
    def partition(repo, config, **kwargs):
        calls.append((repo, config, kwargs))
        return source, provenance("annotated")
    monkeypatch.setattr(datasets, "load_evaluation_partition", partition)
    def label_source(repo, config, **kwargs):
        assert config == "train"
        assert kwargs == {"split": "train", "revision": "immutable-sha"}
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame({
            "generation": ["harmful example", "benign example"], "prompt_label": [1, 0]}))
    monkeypatch.setattr(datasets, "load_dataset", label_source)
    frame = datasets.prepare_toxigen()
    assert len(calls) == 1
    assert calls[0][1] == "annotated"
    assert list(frame["label"]) == [1, 0]
    assert list(frame["source_row"]) == [0, 1]
    assert set(frame["source_split"]) == {"test"}
    assert frame.attrs["evaluation_partition"]["label_reference"]["role"] == "label_lookup_only"


def test_toxigen_refuses_empty_generated_label_mapping(monkeypatch):
    annotated = pd.DataFrame({"text": ["official held-out text"], "toxicity_human": [4.0], "toxicity_ai": [4.0]})
    monkeypatch.setattr(datasets, "load_evaluation_partition", lambda *a, **k: (
        SimpleNamespace(to_pandas=lambda: annotated.copy()), provenance("annotated")))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: SimpleNamespace(to_pandas=lambda: pd.DataFrame({
        "generation": ["different training text"], "prompt_label": [1]})))
    with pytest.raises(ValueError, match="no generated prompt_label matches"):
        datasets.prepare_toxigen()


def test_saved_data_has_partition_columns_and_matching_csv_digest(tmp_path):
    frame = attach_provenance(pd.DataFrame({"sample_id": [0], "text": ["example"], "label": [1]}, index=[17]), provenance())
    path = tmp_path / "prepared.csv"
    datasets.save_dataset(frame, path)
    metadata = json.loads(path.with_suffix(".manifest.json").read_text())
    import hashlib
    assert metadata["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["source_revision"] == "immutable-sha"
    assert pd.read_csv(path).iloc[0]["source_row"] == 17


def test_preparation_failure_does_not_replace_existing_files(tmp_path, monkeypatch):
    existing = tmp_path / "first.csv"
    existing.write_text("historical snapshot")
    def unavailable(**kwargs):
        raise ValueError("train-only source")
    monkeypatch.setattr(prepare_datasets, "PREPARERS", {
        "first": lambda **kwargs: pd.DataFrame({"label": [1]}), "second": unavailable})
    with pytest.raises(ValueError, match="train-only"):
        prepare_datasets.main(["--output-dir", str(tmp_path)])
    assert existing.read_text() == "historical snapshot"
    assert not (tmp_path / "second.csv").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_provider_outputs_keep_partition_provenance_on_success_and_error(tmp_path, monkeypatch, failure):
    frame = attach_provenance(pd.DataFrame({"sample_id": [0], "text": ["example"], "label": [1]}), provenance())
    frame.to_csv(tmp_path / "hatecheck.csv", index=False)
    monkeypatch.setattr(provider_suite, "DATA_DIR", tmp_path)
    def call(**kwargs):
        if failure:
            raise RuntimeError("test error")
        return 0.7
    monkeypatch.setattr(provider_suite, "call_provider", call)
    row = provider_suite.run_single_benchmark("fastino.ai", "hatecheck", "v1", 0.5)[0]
    assert row["source_revision"] == "immutable-sha"
    assert row["source_split"] == "test"
    assert row["partition_role"] == "held_out"
    assert bool(row["error"]) == failure


def test_comparison_refuses_training_reference_as_held_out(tmp_path):
    frame = attach_provenance(pd.DataFrame({"sample_id": [0], "text": ["example"], "label": [1]}),
                              {**provenance(), "partition_role": "training_reference", "source_split": "train"})
    path = tmp_path / "reference.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="training/reference"):
        read_reference(path)


def test_explicit_data_and_output_directories_do_not_use_historical_defaults(tmp_path, monkeypatch):
    data = tmp_path / "evaluation"
    data.mkdir()
    pd.DataFrame({"sample_id": [0], "text": ["example"], "label": [1]}).to_csv(data / "hatecheck.csv", index=False)
    monkeypatch.setattr(provider_suite, "DATA_DIR", tmp_path / "unavailable-historical-data")
    monkeypatch.setattr(provider_suite, "OUTPUT_DIR", tmp_path / "historical-outputs")
    monkeypatch.setattr(provider_suite, "call_provider", lambda **kwargs: 0.9)
    rows = provider_suite.run_single_benchmark("fastino.ai", "hatecheck", "v1", 0.5, data_dir=data)
    path = provider_suite.save_rows(rows, "fastino.ai", "hatecheck", "v1", 0.5, output_dir=tmp_path / "new-outputs")
    assert path == tmp_path / "new-outputs" / "fastino.ai" / "hatecheck_v1_t0.5.csv"
    assert not (tmp_path / "historical-outputs").exists()


def test_comparison_can_select_held_out_datasets_without_loading_train_only_files(tmp_path):
    data, outputs = tmp_path / "data", tmp_path / "outputs"
    data.mkdir()
    pd.DataFrame({"sample_id": [0, 1], "text": ["benign", "harmful"],
                  "label": [0, 1], "partition_role": ["held_out", "held_out"]}).to_csv(data / "hatecheck.csv", index=False)
    for provider in ("fastino.ai", "typesafe.ai"):
        directory = outputs / provider
        directory.mkdir(parents=True)
        pd.DataFrame({"dataset": ["hatecheck", "hatecheck"], "provider": [provider, provider],
                      "prompt_version": ["v1", "v1"], "threshold": [0.5, 0.5],
                      "sample_id": [0, 1], "text": ["benign", "harmful"], "label": [0, 1],
                      "score": [0.1, 0.9], "prediction": [0, 1], "error": ["", ""]}).to_csv(directory / "hatecheck_v1_t0.5.csv", index=False)
        # Excluded source/output should not be parsed or required.
        (directory / "jailbreakhub_v1_t0.5.csv").write_text("unusable legacy data")
    report = collect_report(outputs, data, ["fastino.ai", "typesafe.ai"], datasets=["hatecheck"])
    assert report["baseline_datasets"] == ["hatecheck"]
    assert report["selected_datasets"] == ["hatecheck"]


def test_comparison_cannot_silently_mix_new_partitions_and_legacy_files(tmp_path):
    data, outputs = tmp_path / "data", tmp_path / "outputs"
    data.mkdir()
    frame = pd.DataFrame({"sample_id": [0], "text": ["example"], "label": [1]})
    frame.to_csv(data / "toxigen.csv", index=False)
    frame.assign(partition_role="held_out").to_csv(data / "hatecheck.csv", index=False)
    for provider in ("fastino.ai", "typesafe.ai"):
        (outputs / provider).mkdir(parents=True)
    with pytest.raises(ValueError, match="unversioned legacy"):
        collect_report(outputs, data, ["fastino.ai", "typesafe.ai"], datasets=["toxigen", "hatecheck"])
