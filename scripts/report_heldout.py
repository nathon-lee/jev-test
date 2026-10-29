"""Report only provenance-verified held-out results; do not mix legacy baselines."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.publish_corrected_comparison import table
from scripts.rerun_classification import digest, load_input


def build(root, data_dir, datasets):
    sources = {}
    for dataset in datasets:
        filename = dataset.replace("-", "_") + ".csv"
        samples, provenance = load_input(data_dir / filename, True)
        sources[dataset] = {**provenance, "rows": len(samples)}
    configurations = {(d, v) for d in datasets for v in ("v1", "v2")}
    records, coverage = [], []
    for provider in ("fastino.ai", "laya-ai.com", "typesafe.ai"):
        found = {}
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
            frame = pd.read_csv(csv_path)
            valid = frame[frame["error"].isna() | frame["error"].eq("")]
            labels = valid["label"].astype(int)
            found[key] = {"provider": provider, "dataset": key[0], "prompt": key[1],
                          "valid": result["valid"], "errors": result["errors"], "expected": result["expected_rows"],
                          **result["threshold_metrics"]["0.5"],
                          "auprc": float(average_precision_score(labels, valid["score"])) if labels.nunique() == 2 else None,
                          "resolved_models": ",".join(result["resolved_models"]),
                          "source_split": sources[key[0]]["source_split"], "source_revision": sources[key[0]]["source_revision"]}
        complete = set(found) == configurations
        coverage.append({"provider": provider, "completed_configurations": len(found), "expected_configurations": len(configurations),
                         "status": "complete" if complete else "pending"})
        if complete:
            records.extend(found.values())
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
    report = {"sources": sources, "coverage": coverage, "classification": records, "pii_extraction": extraction}
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = ["# Official held-out evaluation rerun", "",
            "Separate from PR #20's historical-row parity benchmarks. Prepared with PR #19's explicit split/provenance policy. "
            "Official held-out membership does not establish absence of overlap with a model's training data; no training leak audit has been completed. "
            "Detect Jailbreak and JailbreakHub are excluded pending an agreed joint held-out split. ToxiGen is excluded by request because its official test rows do not match the legacy label lookup.", "",
            "## Provider coverage", "", table(coverage, ["provider", "status", "completed_configurations", "expected_configurations"]), "",
            "Only complete provider suites appear below. Pending/error rows are never counted as negatives. "
            "Error rates remain visible; future cross-provider comparisons require matching successful rows, not independent successful-subset rankings. "
            "Primary threshold is 0.5 and scores use Noul. Threshold variants share the same inference probabilities.", "",
            "## Classification", "", table(records, ["provider", "dataset", "prompt", "valid", "errors", "expected", "accuracy", "precision", "recall", "f1", "auroc", "auprc", "fpr", "fnr", "resolved_models"]), "",
            "## PII extraction", "", "GLiNER extraction is a separate model/task from GLiDE classification. "
            "Typed span F1 requires exact boundaries and entity type; boundary F1 ignores type. Failed responses count gold entities as missed. "
            "No masking/leakage metric is claimed without a corresponding evaluated masking output.", "",
            table(extraction, ["dataset", "model", "source_split", "valid", "errors", "expected", "typed_span_f1", "boundary_f1", "exact_document_match"]), "",
            "## Classification source partitions", "", table([{"dataset": d, **s} for d, s in sources.items()], ["dataset", "source_repository", "source_revision", "source_split", "rows", "csv_sha256"]), ""]
    (root / "README.md").write_text("\n".join(text))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/heldout_2026-10-28"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/heldout_2026-10-28"))
    parser.add_argument("--datasets", default="hatecheck,aegis2,jailbreak-classification")
    args = parser.parse_args()
    build(args.root, args.data_dir, args.datasets.split(","))


if __name__ == "__main__":
    main()
