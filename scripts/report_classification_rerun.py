"""Summarize a corrected live rerun without mixing in stale provider scores."""

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark.comparison import fingerprint
from benchmark.evaluation import evaluate_predictions
from scripts.rerun_classification import digest


def valid_rows(frame):
    valid = frame[frame["error"].isna() | frame["error"].eq("")].copy()
    if valid["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs")
    scores = pd.to_numeric(valid["score"], errors="raise")
    if scores.isna().any() or not scores.between(0, 1).all():
        raise ValueError("Invalid classification probability")
    if not (valid["prediction"].astype(int) == (scores >= 0.5).astype(int)).all():
        raise ValueError("Prediction does not match the stated 0.5 threshold")
    return valid.set_index("sample_id")


def paired_metrics(new, reference):
    new, reference = valid_rows(new), valid_rows(reference)
    ids = sorted(set(new.index) & set(reference.index))
    excluded_nul_rows = []
    for sample_id in ids:
        left, right = new.loc[sample_id], reference.loc[sample_id]
        if int(left["label"]) != int(right["label"]):
            raise ValueError(f"Reference label/text mismatch for sample {sample_id}")
        text, old_text = str(left["text"]), str(right["text"])
        if fingerprint(text) != fingerprint(old_text):
            # Older pandas truncated at NUL; the current parser preserves it.
            # These are different model inputs, so retain the fresh result but
            # exclude this precisely identified case from the paired comparison.
            if "\x00" in text and fingerprint(text.split("\x00", 1)[0]) == fingerprint(old_text):
                excluded_nul_rows.append(sample_id)
            else:
                raise ValueError(f"Reference label/text mismatch for sample {sample_id}")
    ids = [sample_id for sample_id in ids if sample_id not in excluded_nul_rows]
    results = {}
    for name, frame in (("rerun", new), ("historical_jev", reference)):
        aligned = frame.loc[ids]
        results[name] = evaluate_predictions(aligned["label"].astype(int), aligned["prediction"].astype(int), aligned["score"].astype(float))
    return {"common_successful_rows": len(ids), "excluded_legacy_nul_rows": excluded_nul_rows, **results}


def build_report(root, historical_dir=Path("outputs/typesafe.ai")):
    root = Path(root)
    records = []
    restriction_path = root / "access_restrictions.json"
    restrictions = json.loads(restriction_path.read_text()) if restriction_path.exists() else []
    restricted_providers = {row["provider"] for row in restrictions}
    incomplete_restricted = set()
    for path in sorted(root.glob("*/*.manifest.json")):
        manifest = json.loads(path.read_text())
        result = manifest.get("result")
        if result is None or not result["complete"]:
            if manifest["provider"] in restricted_providers:
                incomplete_restricted.add(manifest["provider"])
                continue
            raise ValueError(f"Incomplete rerun configuration: {path.name}")
        csv_path = path.with_name(path.name.replace(".manifest.json", ".csv"))
        if digest(csv_path) != result["file_sha256"]:
            raise ValueError("Rerun file does not match its recorded digest")
        row = {**manifest, "file": str(csv_path.relative_to(root))}
        baseline_path = Path(historical_dir) / f"{result['dataset']}_{result['prompt_version']}_t0.5.csv"
        if baseline_path.exists():
            new = pd.read_csv(csv_path, dtype={"sample_id": str})
            historical = pd.read_csv(baseline_path, dtype={"sample_id": str})
            row["historical_jev_file_sha256"] = digest(baseline_path)
            row["paired_baseline"] = paired_metrics(new, historical)
        records.append(row)
    if not records:
        raise ValueError("No completed rerun configurations")
    completed_providers = {row["provider"] for row in records}
    unresolved_restrictions = [row for row in restrictions if row["provider"] not in completed_providers
                               or row["provider"] in incomplete_restricted]
    report = {
        "completed_providers": sorted(completed_providers),
        "access_restrictions": unresolved_restrictions,
        "score_field": "answers.score.noul", "thresholds_derived_offline": True,
        "independent_repeat_inferences": False,
        "historical_jev_request_model": "jev-latest",
        "historical_jev_resolved_model": None,
        "runner_sha256": digest(Path(__file__).with_name("rerun_classification.py")),
        "runs": records,
    }
    root.joinpath("summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def number(value):
        return f"{value:.4f}" if value is not None else "—"
    lines = [
        "# Corrected GLiDE/Laya classification rerun",
        "",
        "These are new live requests scored strictly from `answers.score.noul`. "
        "Historical Fastino/Laya confidence-based results are not included. "
        "The old output directories remain historical snapshots.",
        "",
        "Each probability is requested once per row and prompt. Thresholds 0.5–0.8 "
        "are computed offline; they are not independent repeat inferences, and "
        "no repeatability claim is made. All four thresholds are retained in the "
        "JSON summary; 0.5 is the fixed primary classification baseline.",
        "",
        "## Classification quality at threshold 0.5",
        "",
        "Jev columns use the saved `jev-latest` outputs on exactly the same "
        "common-success rows, with source-text and label checks. Jev is a historical "
        "reference, not a contemporaneous rerun; its resolved model version is unknown.",
        "",
        "| Provider | Dataset | Prompt | Success / expected | Paired rows | Excluded NUL rows | New F1 | Historical Jev F1 | New AUROC | Historical Jev AUROC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        result, paired = row["result"], row.get("paired_baseline")
        if result["pii_detection_only"]:
            continue
        fresh = paired["rerun"] if paired else result["threshold_metrics"]["0.5"]
        old = paired["historical_jev"] if paired else {}
        lines.append(f"| {row['provider']} | {row['dataset']} | {row['prompt_version']} | "
                     f"{result['valid']} / {result['expected_rows']} | {paired['common_successful_rows'] if paired else '—'} | {len(paired['excluded_legacy_nul_rows']) if paired else '—'} | "
                     f"{number(fresh['f1'])} | {number(old.get('f1'))} | {number(fresh['auroc'])} | {number(old.get('auroc'))} |")
    lines += ["", "Historical pandas versions truncated text at NUL while the current "
              "parser preserves it. Such rows are excluded from the paired Jev "
              "comparison only when the historical text exactly matches the "
              "prefix before the first NUL. Their fresh results remain in the "
              "full-run CSV/metrics. Every other text or label mismatch fails validation.", ""]
    for row in records:
        excluded = row.get("paired_baseline", {}).get("excluded_legacy_nul_rows", [])
        if excluded:
            lines.append(f"- {row['provider']} / {row['dataset']} / {row['prompt_version']}: excluded legacy NUL sample IDs {', '.join(excluded)}.")
    lines += ["", "## PII detection diagnostics", "",
              "These prepared PII datasets have all-positive document labels. "
              "Detection rate is not entity/span extraction F1 and does not measure "
              "redaction, leakage, or false-positive behavior. No extraction result "
              "is claimed for GLiDE or Jev here.", "",
              "| Provider | Dataset | Prompt | Success / expected | Positive detection rate |",
              "|---|---|---|---:|---:|"]
    for row in records:
        result = row["result"]
        if result["pii_detection_only"]:
            metric = result["threshold_metrics"]["0.5"]
            lines.append(f"| {row['provider']} | {row['dataset']} | {row['prompt_version']} | "
                         f"{result['valid']} / {result['expected_rows']} | {number(metric['recall'])} |")
    lines += ["", "## Request timing and provenance", "",
              "Timing measures wall-clock request latency including retries under the "
              "recorded concurrent workload. It is not compared with historical serial "
              "Jev timing. Warm-up, hosting, and client/network conditions can differ.", "",
              "| Provider | Dataset | Prompt | Returned model(s) | Mean ms | p50 ms | p95 ms | Workers |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for row in records:
        result = row["result"]
        lines.append(f"| {row['provider']} | {row['dataset']} | {row['prompt_version']} | "
                     f"{', '.join(result['resolved_models'])} | {number(result['latency_mean_ms'])} | "
                     f"{number(result['latency_p50_ms'])} | {number(result['latency_p95_ms'])} | {row['workers']} |")
    lines += ["", "Per-configuration manifests record exact source-file and submitted-input "
              "hashes, prompt text, requested/returned model IDs, timestamps, coverage, "
              "and threshold metrics. CSVs retain corrected scores, diagnostic raw "
              "confidence, response status, attempts, and per-call latency. JSONL "
              "checkpoints support resuming and retain token usage where returned. "
              "Missing or failed rows are not counted as correct predictions.", "",
              ("Laya is blocked by HTTP 402 (Payment Required); partial Laya scores are not published as a completed benchmark. "
               if any(row["provider"] == "laya-ai.com" and row["http_status"] == 402 for row in unresolved_restrictions)
               else "Both GLiDE and Laya are included in this snapshot. " if "laya-ai.com" in completed_providers
               else "Laya is not included in this snapshot; its results will be added after its run completes. ") + "These results use "
              "Nathan's existing prepared rows and label mappings, not the Notion "
              "experiment's separately sampled evaluation sets.", ""]
    root.joinpath("README.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.input_dir)


if __name__ == "__main__":
    main()
