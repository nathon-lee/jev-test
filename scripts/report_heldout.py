"""Report only provenance-verified held-out results; do not mix legacy baselines."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.publish_corrected_comparison import table
from scripts.rerun_classification import digest, load_input
from scripts.report_classification_rerun import valid_rows
from benchmark.comparison import fingerprint
from benchmark.evaluation import evaluate_predictions


def build(root, data_dir, datasets):
    sources = {}
    source_samples = {}
    for dataset in datasets:
        filename = dataset.replace("-", "_") + ".csv"
        samples, provenance = load_input(data_dir / filename, True)
        sources[dataset] = {**provenance, "rows": len(samples)}
        source_samples[dataset] = {r["sample_id"]: (fingerprint(r["text"]), r["label"]) for r in samples}
    configurations = {(d, v) for d in datasets for v in ("v1", "v2")}
    records, coverage, thresholds = [], [], []
    comparison_rows = {}
    for provider in ("fastino.ai", "laya-ai.com", "typesafe.ai", "microsoft-decision-1"):
        found, signatures, provider_thresholds = {}, {}, []
        for path in sorted((root / "classification" / provider).glob("*.manifest.json")):
            run = json.loads(path.read_text())
            key = (run["dataset"], run["prompt_version"])
            if key not in configurations:
                continue
            result = run.get("result")
            if not result or not result["complete"]:
                continue
            if not run.get("require_heldout") or run["evaluation_partition"] != {k: v for k, v in sources[key[0]].items() if k != "rows"}:
                raise ValueError("Run does not match the declared held-out source")
            csv_path = path.with_name(path.name.replace(".manifest.json", ".csv"))
            if digest(csv_path) != result["file_sha256"]:
                raise ValueError("Result hash mismatch")
            frame = pd.read_csv(csv_path, dtype={"sample_id": str})
            if frame["sample_id"].duplicated().any():
                raise ValueError("Duplicate held-out prediction IDs")
            if not frame["label"].isin([0, 1]).all():
                raise ValueError("Held-out labels must be binary")
            actual = {r["sample_id"]: (fingerprint(str(r["text"])), int(r["label"])) for _, r in frame.iterrows()}
            if actual != source_samples[key[0]]:
                raise ValueError("Prediction IDs/text/labels differ from the held-out inputs")
            valid = valid_rows(frame)
            if len(valid) != result["valid"] or len(frame) - len(valid) != result["errors"]:
                raise ValueError("Coverage disagrees with the result manifest")
            signatures[key] = tuple(sorted(valid.index))
            labels = valid["label"].astype(int)
            found[key] = {"provider": provider, "dataset": key[0], "prompt": key[1],
                          "valid": result["valid"], "errors": result["errors"], "expected": result["expected_rows"],
                          **evaluate_predictions(labels, valid["prediction"].astype(int), valid["score"].astype(float)),
                          "auprc": float(average_precision_score(labels, valid["score"])) if labels.nunique() == 2 else None,
                          "resolved_models": ",".join(result["resolved_models"]),
                          "workers": run["workers"], "requests_per_second": run.get("requests_per_second"),
                          "latency_mean_ms": result["latency_mean_ms"], "latency_p50_ms": result["latency_p50_ms"],
                          "latency_p95_ms": result["latency_p95_ms"],
                          "source_split": sources[key[0]]["source_split"], "source_revision": sources[key[0]]["source_revision"]}
            for threshold in (.5, .6, .7, .8):
                provider_thresholds.append({"provider": provider, "dataset": key[0], "prompt": key[1], "threshold": threshold,
                                            **evaluate_predictions(labels, (valid["score"] >= threshold).astype(int), valid["score"])})
        complete = set(found) == configurations
        coverage.append({"provider": provider, "completed_configurations": len(found), "expected_configurations": len(configurations),
                         "status": "complete" if complete else "pending"})
        if complete:
            for key, ids in signatures.items():
                if key in comparison_rows and comparison_rows[key] != ids:
                    raise ValueError("Providers have different successful rows; paired comparison is unavailable")
                comparison_rows[key] = ids
            records.extend(found.values())
            thresholds.extend(provider_thresholds)
    extraction = []
    for path in sorted((root / "pii_extraction").glob("*.summary.json")):
        summary = json.loads(path.read_text())
        result_path = path.with_name(path.name.replace(".summary.json", ".jsonl"))
        manifest = json.loads(result_path.with_suffix(".manifest.json").read_text())
        if digest(result_path) != manifest["output_sha256"]:
            raise ValueError("Extraction output hash mismatch")
        extraction.append({"dataset": summary["partitions"][0][0], "model": summary["model"],
                           "valid": summary["successful_rows"], "errors": summary["failed_rows"],
                           "expected": summary["expected_rows"], "typed_span_f1": summary["entity_micro"]["f1"],
                           "boundary_f1": summary["boundary_micro"]["f1"], "exact_document_match": summary["exact_document_match"],
                           "source_revision": summary["partitions"][0][2], "source_split": summary["partitions"][0][1]})
    root.mkdir(parents=True, exist_ok=True)
    report = {"sources": sources, "coverage": coverage, "classification": records, "pii_extraction": extraction,
              "thresholds": thresholds, "same_successful_rows_verified": True}
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = ["# Official held-out evaluation rerun", "",
            "Separate from PR #20's historical-row parity benchmarks. Prepared with PR #19's explicit split/provenance policy. "
            "Official held-out membership does not establish absence of overlap with a model's training data; no training leak audit has been completed. "
            "Detect Jailbreak and JailbreakHub are excluded pending an agreed joint held-out split. ToxiGen is excluded by request because its official test rows do not match the legacy label lookup.", "",
            "## Provider coverage", "", table(coverage, ["provider", "status", "completed_configurations", "expected_configurations"]), "",
            "Only complete provider suites appear below. Pending/error rows are never counted as negatives. "
            "Successful sample IDs, text and labels have been verified against the source files and matched across all included providers. "
            "Primary threshold is 0.5 and scores use Noul. Threshold variants share the same inference probabilities.", "",
            "## Classification", "", table(records, ["provider", "dataset", "prompt", "valid", "errors", "expected", "accuracy", "precision", "recall", "f1", "auroc", "auprc", "fpr", "fnr", "resolved_models"]), "",
            "## PII extraction", "", "GLiNER extraction is a separate model/task from GLiDE classification. "
            "Typed span F1 requires exact boundaries and entity type; boundary F1 ignores type. Failed responses count gold entities as missed. "
            "No masking/leakage metric is claimed without a corresponding evaluated masking output.", "",
            table(extraction, ["dataset", "model", "source_split", "valid", "errors", "expected", "typed_span_f1", "boundary_f1", "exact_document_match"]), "",
            "## Classification source partitions", "", table([{"dataset": d, **s} for d, s in sources.items()], ["dataset", "source_repository", "source_revision", "source_split", "rows", "csv_sha256"]), "",
            "## Presentation", "", "[HTML comparison](comparison.html) · [Figure gallery](figures/index.html) · "
            "[Combined PDF](figures/heldout_figures.pdf) · [v1 summary PNG](figures/fig13_three_dataset_averages.png)", "",
            "The summary chart equally weights HateCheck, AEGIS2 and Jailbreak Classification. "
            "It is a different three-dataset scope from PR #20 (which includes ToxiGen). "
            "v1 and v2 stay separate; PII extraction and differently throttled request timing are separate figures.", ""]
    (root / "README.md").write_text("\n".join(text))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/heldout_2026-10-28"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/heldout_2026-10-28"))
    parser.add_argument("--datasets", default="hatecheck,aegis2,jailbreak-classification")
    args = parser.parse_args()
    build(args.root, args.data_dir, args.datasets.split(","))
    from benchmark.heldout_figures import publish
    publish(args.root)


if __name__ == "__main__":
    main()
