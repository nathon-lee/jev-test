from pathlib import Path

import pandas as pd

from scripts.evaluate import evaluate, resolve_prediction_path


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


def test_evaluate_handles_missing_latency_column(tmp_path, monkeypatch, capsys):
    outputs_dir = tmp_path / "outputs"
    provider_dir = outputs_dir / "typesafe.ai"
    provider_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "dataset": ["jailbreak-classification"],
            "sample_id": [1],
            "text": ["test text"],
            "label": [1],
            "score": [0.9],
            "prediction": [1],
            "error": [""],
        }
    )
    path = provider_dir / "jailbreak-classification_v2_t0.5.csv"
    df.to_csv(path, index=False)

    monkeypatch.chdir(tmp_path)
    evaluate(
        "jailbreak-classification",
        prompt_version="v2",
        threshold=0.5,
        debug=False,
    )

    captured = capsys.readouterr()
    assert "Latency unavailable" in captured.out
