from pathlib import Path

from scripts.evaluate import resolve_prediction_path


def test_resolve_prediction_path_prefers_provider_threshold_file(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    provider_dir = outputs_dir / "typesafe.ai"
    provider_dir.mkdir(parents=True)

    target = provider_dir / "jailbreak-classification_v2_t0.5.csv"
    target.write_text("placeholder")

    root_generic = outputs_dir / "jailbreak-classification_predictions.csv"
    root_generic.write_text("placeholder")

    monkeypatch.chdir(tmp_path)
    result = resolve_prediction_path(
        "jailbreak-classification",
        prompt_version="v2",
        threshold=0.5,
    )

    assert result == target
