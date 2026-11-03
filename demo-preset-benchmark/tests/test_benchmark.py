import importlib.util
from pathlib import Path

import pytest

for dependency in ("dotenv", "httpx", "pyarrow", "sklearn", "truststore"):
    pytest.importorskip(dependency, reason="Install demo-preset-benchmark/requirements-test.txt")

MODULE_PATH = Path(__file__).parents[1] / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("demo_preset_benchmark", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_binary_metrics_include_confusion_counts():
    rows = [
        {"gold": 1, "prediction": 1},
        {"gold": 1, "prediction": 0},
        {"gold": 0, "prediction": 1},
        {"gold": 0, "prediction": 0},
    ]

    result = benchmark.binary(rows)

    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0.5


def test_multilabel_metrics_cover_primary_pii_categories():
    rows = [
        {"gold": ["email"], "prediction": ["email", "ssn"]},
        {"gold": ["phone_number"], "prediction": ["phone_number"]},
    ]

    result = benchmark.multilabel(rows)

    assert set(result["per_class"]) == set(benchmark.PII_CLASSES)
    assert result["micro"]["tp"] == 2
    assert result["micro"]["fp"] == 1
    assert result["micro"]["fn"] == 0
    assert result["exact_match_accuracy"] == 0.5


def test_failed_calls_are_not_counted_as_zero_quality():
    result = benchmark.summarize(
        "pi",
        [{"error": "service unavailable", "gold": 1, "prediction": None}],
        requested=1,
    )

    assert result["successful_rows"] == 0
    assert result["failed_rows"] == 1
    assert result["coverage"] == 0
    assert "f1" not in result
