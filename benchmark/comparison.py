"""Offline comparison of saved suite outputs using the existing suite metrics."""

import csv
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

from benchmark.evaluation import evaluate_predictions


MODEL_IDENTITIES = {
    "fastino.ai": {
        "request_model": "fastino/GLiDE",
        "resolved_model": "glide_v2",
    },
    "laya-ai.com": {
        "request_model": "typed-decisions",
        "serving_model": "laya-rl-agent",
        "routed_model": "typed-decisions",
    },
    "typesafe.ai": {
        "request_model": "jev-latest",
        "resolved_model": None,
    },
}
MODELS = {
    provider: "；".join(
        f"{role.replace('_', ' ')}={model if model is not None else '待接口返回确认'}"
        for role, model in identity.items()
    )
    for provider, identity in MODEL_IDENTITIES.items()
}
DATASETS = {
    "toxigen": "toxigen.csv",
    "hatecheck": "hatecheck.csv",
    "aegis2": "aegis2.csv",
    "detect-jailbreak": "detect_jailbreak.csv",
    "jailbreak-classification": "jailbreak_classification.csv",
    "jailbreakhub": "jailbreakhub.csv",
    "pii_masking_nano_1k": "pii_masking_nano_1k.csv",
    "pii_masking_mini_10k": "pii_masking_mini_10k.csv",
}
PROMPTS = ("v1", "v2")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
STABILITY_THRESHOLD = 0.5
STABILITY_METRICS = ("mean_score_std", "mean_score_range", "decision_flip_rate")
FILENAME = re.compile(r"(.+)_(v[12])_t(0\.\d+)\.csv$")
START = re.compile(
    r"Running benchmark for (\S+) \| (\S+) \| (v[12]) \| "
    r"threshold=([\d.]+) \| total_rows=(\d+)"
)
PROGRESS = re.compile(
    r"Progress: (\d+)/(\d+).*provider=(\S+) \| dataset=(\S+) \| "
    r"prompt=(v[12]) \| elapsed=([\d.]+)s"
)


def fingerprint(text):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def read_reference(path):
    rows = {}
    # Match run_single_benchmark's pandas parser, including NA and NUL handling.
    for _, row in pd.read_csv(path).iterrows():
        sample = str(row["sample_id"])
        if sample in rows:
            raise ValueError(f"{path}: duplicate sample_id={sample}")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"{path}: invalid label for sample_id={sample}")
        rows[sample] = (label, fingerprint(str(row["text"])))
    if not rows:
        raise ValueError(f"{path}: empty reference dataset")
    return rows


def parse_logs(paths, warnings=None):
    runs = {}
    for path in sorted(paths, key=lambda p: (p.stat().st_mtime_ns, p.name)):
        try:
            records = _parse_log(path)
        except ValueError as error:
            if warnings is None:
                raise
            warnings.append(
                f"{path}: {error}；整份日志不用于逐配置耗时比较，不猜测并发进度的阈值归属。"
                "可核验的完整四轮均值仅用于四模型参考图。"
            )
            continue
        for key, record in records.items():
            own_log = path.parent.name == key[0]
            previous = runs.get(key)
            previous_own_log = previous and Path(previous["log_file"]).parent.name == key[0]
            if own_log or not previous_own_log:
                runs[key] = record
    return runs


def _parse_log(path):
    """Threshold comes from task starts, not progress lines (which omit it)."""
    runs = {}
    active = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            starts = list(START.finditer(line))
            if len(starts) > 1:
                raise ValueError("interleaved task starts cannot be assigned to progress")
            if starts:
                provider, dataset, prompt, threshold, total = starts[0].groups()
                active = (provider, dataset, prompt, float(threshold))
                runs[active] = {
                    "total": int(total), "processed": 0, "elapsed_s": None,
                    "log_file": str(path),
                }
                continue
            progress = PROGRESS.search(line)
            if not progress or active is None:
                continue
            count, total, provider, dataset, prompt, elapsed = progress.groups()
            if (provider, dataset, prompt) != active[:3]:
                raise ValueError("interleaved progress cannot be assigned to a threshold")
            record = runs[active]
            count, total, elapsed = int(count), int(total), float(elapsed)
            if total != record["total"] or not 0 < count <= total:
                raise ValueError("invalid progress counts")
            if count < record["processed"] or (
                record["elapsed_s"] is not None and elapsed < record["elapsed_s"]
            ):
                raise ValueError("non-monotonic progress")
            record.update(processed=count, elapsed_s=elapsed)
    return runs


def parse_repeat_log_timings(paths):
    """A four-run mean does not require assigning parallel completions to thresholds."""
    latest, origins = {}, {}
    for path in sorted(paths, key=lambda p: (p.stat().st_mtime_ns, p.name)):
        starts, progress, saved = defaultdict(list), defaultdict(list), defaultdict(list)
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                for match in START.finditer(line):
                    provider, dataset, prompt, threshold, total = match.groups()
                    starts[(provider, dataset, prompt)].append((float(threshold), int(total)))
                for match in PROGRESS.finditer(line):
                    count, total, provider, dataset, prompt, elapsed = match.groups()
                    progress[(provider, dataset, prompt)].append((int(count), int(total), float(elapsed)))
                for match in re.finditer(r"Saved:\s+(\S+\.csv).*?\brows=(\d+)", line):
                    filename, count = match.groups()
                    output = Path(filename.replace("\\", "/"))
                    config = FILENAME.fullmatch(output.name)
                    if config:
                        dataset, prompt, threshold = config.groups()
                        saved[(output.parent.name, dataset, prompt)].append((float(threshold), int(count)))
        for key, tasks in starts.items():
            previous = origins.get(key)
            if (
                previous and previous.parent.name == key[0]
                and path.parent.name != key[0]
            ):
                continue
            origins[key] = path
            # Even an incomplete later rerun invalidates the older four-run mean.
            latest[key] = None
            thresholds = [threshold for threshold, _ in tasks]
            totals = {total for _, total in tasks}
            if len(tasks) != len(THRESHOLDS) or set(thresholds) != set(THRESHOLDS) or len(totals) != 1:
                continue
            total = next(iter(totals))
            events = progress[key]
            if total <= 0 or not events or any(
                expected != total or not 0 < count <= total or not math.isfinite(elapsed) or elapsed < 0
                for count, expected, elapsed in events
            ):
                continue
            finished = [elapsed for count, _, elapsed in events if count == total]
            if len(finished) != len(THRESHOLDS) or Counter(saved[key]) != Counter(tasks):
                continue
            provider, dataset, prompt = key
            latest[key] = {
                "provider": provider, "dataset": dataset, "prompt": prompt,
                "repeat_thresholds": list(THRESHOLDS), "rows_per_run": total,
                "elapsed_s": finished, "mean_elapsed_ms": mean(finished) * 1000 / total,
                "log_file": str(path),
            }
    return [record for record in latest.values() if record is not None]


@dataclass(frozen=True)
class Prediction:
    label: int
    prediction: int
    score: float


def classification_metrics(rows):
    rows = list(rows)
    return evaluate_predictions(
        [row.label for row in rows],
        [row.prediction for row in rows],
        [row.score for row in rows],
    )


def stability_metrics(frames, sample_ids):
    """Compare scores from four calls, never predictions made at different thresholds."""
    if len(frames) != len(THRESHOLDS):
        raise ValueError("Stability requires all four repeat runs")
    deviations, ranges, flipped = [], [], 0
    for sample in sorted(sample_ids):
        scores = [frame[sample].score for frame in frames]
        deviations.append(pstdev(scores))
        ranges.append(max(scores) - min(scores))
        flipped += int(min(scores) < STABILITY_THRESHOLD <= max(scores))
    count = len(deviations)
    return {
        "n": count, "flipped": flipped,
        "mean_score_std": mean(deviations) if count else None,
        "mean_score_range": mean(ranges) if count else None,
        "decision_flip_rate": flipped / count if count else None,
    }


def read_predictions(path, reference, provider, dataset, prompt, threshold):
    valid, seen, errors = {}, set(), Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "dataset", "sample_id", "text", "label", "prompt_version",
            "threshold", "score", "prediction", "provider", "error",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            sample = row["sample_id"]
            if sample in seen or sample not in reference:
                raise ValueError(f"{path}:{line}: duplicate or unknown sample_id={sample}")
            seen.add(sample)
            label = int(row["label"])
            if (label, fingerprint(row["text"])) != reference[sample]:
                raise ValueError(f"{path}:{line}: reference label/text mismatch for sample_id={sample}")
            if (
                row["provider"] != provider or row["dataset"] != dataset
                or row["prompt_version"] != prompt or float(row["threshold"]) != threshold
            ):
                raise ValueError(f"{path}:{line}: metadata disagrees with filename/provider")
            if row["error"].strip():
                # Export categories, never raw errors which can contain input or credentials.
                category = re.match(
                    r"[A-Za-z_][A-Za-z0-9_]*(?:Error|Timeout)\b", row["error"].strip(),
                )
                errors[category.group() if category else "request_error"] += 1
                continue
            score, prediction = float(row["score"]), float(row["prediction"])
            if (
                not math.isfinite(score) or not 0 <= score <= 1
                or prediction not in (0, 1) or prediction != int(score >= threshold)
            ):
                raise ValueError(f"{path}:{line}: invalid score/prediction")
            valid[sample] = Prediction(label, int(prediction), score)
    return valid, len(seen), dict(errors)


def collect_report(input_dir, data_dir, providers, previous_report=None):
    if len(providers) < 2 or len(set(providers)) != len(providers):
        raise ValueError("Select at least two distinct providers")
    for provider in providers:
        if provider not in MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        if not (input_dir / provider).is_dir() and previous_report is None:
            raise FileNotFoundError(input_dir / provider)
    references = {name: read_reference(data_dir / filename) for name, filename in DATASETS.items()}
    previous = previous_report or {}
    if previous_report is not None and (
        previous["providers"] != providers
        or previous["model_identities"] != {p: MODEL_IDENTITIES[p] for p in providers}
    ):
        raise ValueError("Previous report providers/model identities must match")
    cached_runs = {
        (r["provider"], r["dataset"], r["prompt"], r["threshold"]): r
        for r in previous.get("runs", [])
    }
    for row in cached_runs.values():
        if row["expected"] != len(references[row["dataset"]]):
            raise ValueError("Previous report reference size does not match current data")
    warnings = []
    log_paths = [
        path for provider in providers for path in (input_dir / provider).glob("*.log")
    ]
    logs = parse_logs(log_paths, warnings=warnings)
    repeat_log_timings = parse_repeat_log_timings(log_paths)
    paths = {}
    for provider in providers:
        for path in sorted((input_dir / provider).glob("*.csv")):
            match = FILENAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unrecognized result filename: {path}")
            dataset, prompt, threshold = match.groups()
            threshold = float(threshold)
            if dataset not in DATASETS or threshold not in THRESHOLDS:
                raise ValueError(f"Unknown dataset/threshold: {path}")
            key = (provider, dataset, prompt, threshold)
            if key in paths:
                raise ValueError(f"Duplicate configuration: {path}")
            paths[key] = path
    mismatched_timings = set()
    bindings_path = input_dir / "log_input_hashes.json"
    if bindings_path.exists():
        bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
        for log_name, files in bindings.items():
            log = Path(log_name)
            if len(log.parts) != 2 or log.parts[0] not in MODELS or log.suffix != ".log":
                raise ValueError(f"{bindings_path}: invalid log path")
            provider = log.parts[0]
            for filename, expected_hash in files.items():
                match = FILENAME.fullmatch(filename)
                if not match or "/" in filename or "\\" in filename or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                    raise ValueError(f"{bindings_path}: invalid prediction binding")
                dataset, prompt, threshold = match.groups()
                key = (provider, dataset, prompt, float(threshold))
                path = paths.get(key)
                if path and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    mismatched_timings.add((*key, str(input_dir / log)))
    repeat_log_timings = [
        row for row in repeat_log_timings
        if not any(
            (row["provider"], row["dataset"], row["prompt"], threshold, row["log_file"]) in mismatched_timings
            for threshold in THRESHOLDS
        )
    ]
    runs, paired, pairwise = [], [], []
    if previous_report is not None:
        warnings.extend(
            warning for warning in previous.get("warnings", [])
            if not any(warning.startswith(f"{path}:") for path in log_paths)
        )
        warnings.append(
            "本次为增量更新：未提供历史逐样本 CSV 的配置保留旧报告汇总，不代表重新测试。"
            "新 CSV 替换同配置旧结果，缺少新日志时不沿用旧耗时。"
            "只有配置均未变更、原始预测可对齐，或各模型均在全量参考样本上成功时，"
            "才保留或补充共同样本指标；无法核验的交集不参与公平均值。"
        )
    stability, stability_paired = [], []

    def common_metrics(selected, records, frames, historical, context):
        if all(provider in frames for provider in selected):
            common = sorted(set.intersection(*(set(frames[p]) for p in selected)))
            return [
                (provider, len(common) / records[provider]["expected"],
                 classification_metrics(frames[provider][sample] for sample in common))
                for provider in selected
            ]
        if all(records[p].get("source_kind") == "historical_summary" for p in selected):
            return [
                (row["provider"], row["common_coverage"],
                 {key: row[key] for key in classification_metrics([])})
                for row in historical
            ]
        if all(records[p]["valid"] == records[p]["expected"] for p in selected):
            return [
                (provider, 1.0, {
                    key: records[provider][key] for key in classification_metrics([])
                })
                for provider in selected
            ]
        warnings.append(f"{context}：缺少历史逐样本预测，无法重新对齐共同成功样本，排除共同样本比较。")
        return []

    for dataset, reference in references.items():
        for prompt in PROMPTS:
            repeat_frames = {provider: [] for provider in providers}
            repeat_records = {provider: [] for provider in providers}
            for threshold in THRESHOLDS:
                frames, records = {}, {}
                for provider in providers:
                    key = (provider, dataset, prompt, threshold)
                    path, timing = paths.get(key), logs.get(key)
                    if timing and (*key, timing["log_file"]) in mismatched_timings:
                        warnings.append(
                            f"{provider}/{path.name}：历史日志对应的预测文件已被新结果替换，"
                            "不沿用该旧日志的耗时；质量指标仍使用新 CSV。"
                        )
                        timing = None
                    record = {
                        "provider": provider, "model": MODELS[provider], "dataset": dataset,
                        "prompt": prompt, "threshold": threshold, "expected": len(reference),
                        "file": str(path) if path else None,
                        "status": "missing", "rows": 0, "valid": 0, "errors": 0,
                        "error_rate": None, "coverage": 0.0, "error_categories": {},
                        "mean_elapsed_ms": None, "elapsed_s": None,
                        "log_processed": timing["processed"] if timing else None,
                        "log_total": timing["total"] if timing else None,
                        "log_elapsed_s": timing["elapsed_s"] if timing else None,
                        "log_file": timing["log_file"] if timing else None,
                    }
                    if path is None and key in cached_runs:
                        record = {**cached_runs[key], "source_kind": "historical_summary"}
                    if path:
                        frame, count, errors = read_predictions(
                            path, reference, provider, dataset, prompt, threshold,
                        )
                        frames[provider] = frame
                        record.update(
                            status="complete" if count == len(reference) else "partial",
                            rows=count, valid=len(frame), errors=count - len(frame),
                            error_rate=(count - len(frame)) / count if count else None,
                            coverage=len(frame) / len(reference), error_categories=errors,
                            **classification_metrics(frame.values()),
                        )
                        if previous_report is not None:
                            record["source_kind"] = "local_predictions"
                        if timing and timing["processed"] == count == timing["total"] and count:
                            record.update(
                                mean_elapsed_ms=timing["elapsed_s"] * 1000 / count,
                                elapsed_s=timing["elapsed_s"],
                            )
                        elif timing:
                            warnings.append(f"{provider}/{path.name}: 日志未与完整 CSV 对齐，不用于完成轮次耗时比较。")
                    if timing and not path and key not in cached_runs:
                        record["status"] = "in_progress"
                    records[provider] = record
                    runs.append(record)
                    repeat_frames[provider].append(frames.get(provider, {}))
                    repeat_records[provider].append(record)
                if all(records[p]["status"] == "complete" for p in providers):
                    historical = [
                        row for row in previous.get("paired", [])
                        if (row["dataset"], row["prompt"], row["threshold"]) == (dataset, prompt, threshold)
                    ]
                    for provider, coverage, metrics in common_metrics(
                        providers, records, frames, historical, f"{dataset}/{prompt}/{threshold}",
                    ):
                        paired.append({
                            "provider": provider, "model": MODELS[provider], "dataset": dataset,
                            "prompt": prompt, "threshold": threshold,
                            "common_coverage": coverage, **metrics,
                        })
                if prompt == "v1" and threshold == 0.5 and not dataset.startswith("pii_"):
                    for left, right in combinations(providers, 2):
                        if any(records[p]["status"] != "complete" for p in (left, right)):
                            continue
                        pair = f"{left} vs {right}"
                        historical = [
                            row for row in previous.get("pairwise", [])
                            if row["dataset"] == dataset and row["pair"] == pair
                        ]
                        for provider, coverage, metrics in common_metrics(
                            (left, right), records, frames, historical, f"{pair}/{dataset}",
                        ):
                            pairwise.append({
                                "pair": pair, "provider": provider,
                                "model": MODELS[provider], "dataset": dataset,
                                "common_coverage": coverage, **metrics,
                            })
            for provider in providers:
                records = repeat_records[provider]
                historical = next((
                    row for row in previous.get("stability", [])
                    if (row["provider"], row["dataset"], row["prompt"]) == (provider, dataset, prompt)
                ), None)
                if historical and all(
                    row.get("source_kind") == "historical_summary" for row in records
                ):
                    stability.append(historical.copy())
                    continue
                complete = all(row["status"] == "complete" for row in records)
                raw_available = all(
                    (provider, dataset, prompt, threshold) in paths for threshold in THRESHOLDS
                )
                common = (
                    set.intersection(*(set(frame) for frame in repeat_frames[provider]))
                    if complete and raw_available else set()
                )
                rows = sum(row["rows"] for row in records)
                errors = sum(row["errors"] for row in records)
                stability.append({
                    "provider": provider, "model": MODELS[provider],
                    "dataset": dataset, "prompt": prompt, "expected": len(reference),
                    "available_repeats": sum(row["file"] is not None for row in records),
                    "complete_repeats": sum(row["status"] == "complete" for row in records),
                    "status": (
                        (("complete" if common else "no_common_success") if raw_available else "unavailable_raw")
                        if complete else "incomplete_repeats"
                    ),
                    "coverage": len(common) / len(reference) if complete and raw_available else None,
                    "rows": rows, "errors": errors, "error_rate": errors / rows if rows else None,
                    **stability_metrics(repeat_frames[provider], common),
                })
            all_records = [row for records in repeat_records.values() for row in records]
            if all(row.get("source_kind") == "historical_summary" for row in all_records):
                stability_paired.extend(
                    row.copy() for row in previous.get("stability_paired", [])
                    if (row["dataset"], row["prompt"]) == (dataset, prompt)
                )
            elif all(row["status"] == "complete" for row in all_records) and all(
                (provider, dataset, prompt, threshold) in paths
                for provider in providers for threshold in THRESHOLDS
            ):
                common = set.intersection(*(
                    set(frame) for frames in repeat_frames.values() for frame in frames
                ))
                for provider in providers:
                    stability_paired.append({
                        "provider": provider, "model": MODELS[provider],
                        "dataset": dataset, "prompt": prompt, "expected": len(reference),
                        "coverage": len(common) / len(reference),
                        **stability_metrics(repeat_frames[provider], common),
                    })
    baseline = [
        row for row in paired
        if row["prompt"] == "v1" and row["threshold"] == 0.5
        and not row["dataset"].startswith("pii_") and row["n"] > 0
    ]
    baseline_datasets = sorted({row["dataset"] for row in baseline})
    overview = []
    for provider in providers:
        own = [row for row in runs if row["provider"] == provider]
        base = [row for row in baseline if row["provider"] == provider]
        latency = [
            row for row in own if row["dataset"] in baseline_datasets
            and row["prompt"] == "v1" and row["threshold"] == 0.5
        ]
        timed = latency and all(row["elapsed_s"] is not None for row in latency)
        macro = {}
        for metric in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "auroc", "fpr", "fnr"):
            values = [row[metric] for row in base if row[metric] is not None]
            macro["macro_" + metric] = mean(values) if values else None
        overview.append({
            "provider": provider, "model": MODELS[provider],
            "complete": sum(row["status"] == "complete" for row in own),
            "total": len(own), "rows": sum(row["rows"] for row in own),
            "errors": sum(row["errors"] for row in own),
            "baseline_datasets": len(base),
            **macro,
            "baseline_elapsed_ms": (
                sum(row["elapsed_s"] for row in latency) * 1000 / sum(row["rows"] for row in latency)
                if timed else None
            ),
            "macro_elapsed_ms": mean(row["mean_elapsed_ms"] for row in latency) if timed else None,
        })
    stability_overview = []
    for prompt in PROMPTS:
        for provider in providers:
            base = [
                row for row in stability_paired if row["provider"] == provider
                and row["prompt"] == prompt and not row["dataset"].startswith("pii_") and row["n"]
            ]
            datasets = sorted(row["dataset"] for row in base)
            own = [
                row for row in stability if row["provider"] == provider
                and row["prompt"] == prompt and row["dataset"] in datasets
            ]
            count = sum(row["n"] for row in base)
            expected = sum(row["expected"] for row in base)
            rows = sum(row["rows"] for row in own)
            errors = sum(row["errors"] for row in own)
            stability_overview.append({
                "provider": provider, "model": MODELS[provider], "prompt": prompt,
                "datasets": datasets, "n": count, "expected": expected,
                "coverage": count / expected if expected else None,
                "rows": rows, "errors": errors, "error_rate": errors / rows if rows else None,
                **{metric: mean(row[metric] for row in base) if base else None
                   for metric in STABILITY_METRICS},
            })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **({"previous_generated_at": previous["generated_at"]} if previous_report is not None else {}),
        "input_dir": str(input_dir), "data_dir": str(data_dir),
        "providers": providers, "baseline_datasets": baseline_datasets,
        "model_identities": {provider: MODEL_IDENTITIES[provider] for provider in providers},
        "overview": overview, "runs": runs, "paired": paired,
        "pairwise": pairwise, "warnings": list(dict.fromkeys(warnings)),
        "repeat_log_timings": repeat_log_timings,
        "stability_settings": {
            "repeat_thresholds": list(THRESHOLDS), "decision_threshold": STABILITY_THRESHOLD,
            "std_ddof": 0, "required_repeats": len(THRESHOLDS),
        },
        "stability": stability, "stability_paired": stability_paired,
        "stability_overview": stability_overview,
    }


def pairwise_summary(report):
    grouped = defaultdict(list)
    for row in report["pairwise"]:
        if row["n"]:
            grouped[(row["pair"], row["provider"])].append(row)
    summary = []
    for (pair, provider), rows in grouped.items():
        record = {
            "pair": pair, "provider": provider, "model": MODELS[provider],
            "datasets": sorted(row["dataset"] for row in rows),
        }
        for metric in ("f1", "auroc", "balanced_accuracy", "fpr", "fnr"):
            values = [row[metric] for row in rows if row[metric] is not None]
            record["macro_" + metric] = mean(values) if values else None
        summary.append(record)
    return summary


def fmt(value, percent=True):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.2f}"
    return str(value)


def conclusions(report):
    results = sorted(
        (row for row in report["overview"] if row["macro_f1"] is not None),
        key=lambda row: row["macro_f1"], reverse=True,
    )
    notes = []
    if results:
        winner = results[0]
        notes.append(
            f"固定 v1 / 0.5、共同完成的 {winner['baseline_datasets']} 个非 PII 数据集上，"
            f"{winner['provider']} / {winner['model']} 的宏平均 F1 最高"
            f"（{fmt(winner['macro_f1'])}），宏平均 AUROC 为 {fmt(winner['macro_auroc'])}。"
            "这是当前快照下的描述性结果，不代表未测试任务或统计显著性。"
        )
        ranking = "；".join(
            f"{r['provider']} / {r['model']}：F1 {fmt(r['macro_f1'])}、"
            f"误报率 {fmt(r['macro_fpr'])}、漏报率 {fmt(r['macro_fnr'])}"
            for r in results
        )
        notes.append("同一基线比较：" + ranking + "。高误报意味着正常内容被拦截，高漏报意味着危险内容被放过。")
        best_by_dataset = defaultdict(list)
        for row in report["paired"]:
            if row["prompt"] == "v1" and row["threshold"] == 0.5 and row["dataset"] in report["baseline_datasets"]:
                best_by_dataset[row["dataset"]].append(row)
        wins = []
        for dataset, rows in sorted(best_by_dataset.items()):
            best = max(row["f1"] for row in rows)
            names = " / ".join(row["provider"] for row in rows if row["f1"] == best)
            wins.append(f"{dataset}：{names}（{fmt(best)}）")
        notes.append("分数据集基线 F1 最优：" + "；".join(wins) + "。")
        weakest = min(
            (row for row in baseline_rows(report) if row["provider"] == winner["provider"]),
            key=lambda row: row["f1"],
        )
        notes.append(
            f"综合表现领先仍不等于可直接上线：{winner['provider']} / {winner['model']} "
            f"在自身最弱的共同基线数据集 {weakest['dataset']} 上 F1 为 {fmt(weakest['f1'])}、"
            f"误报率 {fmt(weakest['fpr'])}、漏报率 {fmt(weakest['fnr'])}，"
            "需结合业务容忍度单独选阈值，而不是照搬统一 0.5。"
        )
        all_positive = []
        for provider in report["providers"]:
            rows = [
                row for row in baseline_rows(report)
                if row["provider"] == provider
            ]
            if rows and all(row["positive_rate"] == 1 for row in rows):
                all_positive.append(f"{provider} / {MODELS[provider]}")
        if all_positive:
            notes.append(
                "、".join(all_positive)
                + " 在共同基线的有效样本中全部预测为正类（误报率 100%、平衡准确率 50%），"
                "因此即使个别数据集 F1 较高，也不代表具备有效区分能力，不建议直接使用该固定配置作为拦截器。"
            )
    else:
        notes.append("没有共同完成且含共同有效样本的非 PII 基线数据，暂不能给出公平总排名。")
    timed = [row for row in report["overview"] if row["baseline_elapsed_ms"] is not None]
    if len(timed) >= 2:
        fastest = min(timed, key=lambda row: row["baseline_elapsed_ms"])
        notes.append(
            f"相同基线任务中，{fastest['provider']} / {fastest['model']} 的日志摊销平均耗时最低"
            f"（{fmt(fastest['baseline_elapsed_ms'], False)} ms/样本）。"
            "包含网络、客户端开销和失败等待，且各轮非同一时刻运行；不能解释为纯模型推理速度。"
        )
    for row in report["overview"]:
        if row["complete"] < row["total"]:
            notes.append(
                f"{row['provider']} / {row['model']} 仅完整落盘 {row['complete']}/{row['total']} 个配置；"
                "缺测不记零分，也不把已有任务的均值当作全套能力，当前排名是阶段性结论。"
            )
    grouped_pairs = defaultdict(list)
    for row in pairwise_summary(report):
        if row["macro_f1"] is not None:
            grouped_pairs[row["pair"]].append(row)
    for pair, rows in grouped_pairs.items():
        if len(rows) == 2 and len(rows[0]["datasets"]) > len(report["baseline_datasets"]):
            best = max(rows, key=lambda row: row["macro_f1"])
            notes.append(
                f"扩展两两比较（{pair}，共同完整且有共同有效样本的 {len(best['datasets'])} 个非 PII 数据集）："
                + "；".join(f"{row['provider']} / {row['model']} 宏 F1 {fmt(row['macro_f1'])}" for row in rows)
                + "。此口径覆盖范围更广，但不能与三家共同基线的数值混排。"
            )
    no_success = [
        row for row in report["runs"]
        if row["status"] == "complete" and not row["valid"]
    ]
    if no_success:
        notes.append(
            "以下配置虽完整落盘但全部请求失败："
            + "；".join(
                f"{row['provider']} / {row['model']} / {row['dataset']} / "
                f"{row['prompt']} / {row['threshold']}（{row['errors']}/{row['rows']}）"
                for row in no_success
            )
            + "。这些配置没有质量分数，不参与质量均值；这是接口可靠性问题，不能当作缺测或正常能力结果忽略。"
        )
    inverted = [
        row for row in report["paired"]
        if row["prompt"] == "v1" and row["threshold"] == 0.5
        and row["auroc"] is not None and row["auroc"] < 0.5
    ]
    if inverted:
        notes.append(
            "部分基线 AUROC 低于 0.5："
            + "；".join(f"{row['provider']} / {row['dataset']}={fmt(row['auroc'])}" for row in inverted)
            + "。建议优先核对返回分数语义及接口解析；当前数据不能直接区分模型能力弱与分数映射问题，不自动翻转分数。"
        )
    notes.append(
        "PII 是“是否含 PII”的普通二分类，不是实体抽取或脱敏任务。当前数据全是正样本，"
        "恒预测“含 PII”即可获得 100% 检出率；"
        "现有输出没有脱敏文本，不能评价实体定位、泄露率或脱敏质量，不应据此选定脱敏模型。"
    )
    notes.append(
        "选型优先参考固定配置的 F1、AUROC、误报/漏报及错误率；"
        "表中的最优提示词/阈值是在同一测试集上挑选的探索性上限，部署前需独立验证集重新选参并复测。"
    )
    if report.get("external_comparison"):
        notes.append(
            "OpenJev 4B 及其他内部模型为用户提供的汇总结果，仅加入独立参考图表，"
            "未提供逐样本输出，不参与上述共同成功样本基线、排名或重复调用一致性分析。"
        )
    return notes


def baseline_rows(report):
    return [
        row for row in report["paired"]
        if row["prompt"] == "v1" and row["threshold"] == 0.5
        and not row["dataset"].startswith("pii_") and row["n"]
    ]


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def report_sections(report):
    sections = []
    sections.append(("结论", "\n\n".join(conclusions(report))))
    sections.append(("模型身份与证据", (
        "以下模型身份由用户补充，区分请求名称、实际解析名称、服务模型与路由模型，"
        "不是从历史 CSV 或日志反查得到。TypeSafe 的 resolved model 需以接口返回为准；"
        "现有记录未提供该值，因此保持未确认，不将 jev-latest 当作固定后端版本。\n\n"
        + markdown_table(
            ["供应商", "Request model", "Resolved model", "Serving model", "Routed model"],
            [[
                provider, identity["request_model"],
                identity.get("resolved_model") or (
                    "待接口返回确认" if "resolved_model" in identity else "—"
                ),
                identity.get("serving_model", "—"), identity.get("routed_model", "—"),
            ] for provider, identity in report["model_identities"].items()],
        )
    )))
    sections.append(("比较口径", (
        "固定基线：v1 / threshold=0.5。只使用所有所选供应商均完整落盘的数据集，"
        "按 sample_id 对齐共同成功样本；读取时统一 CRLF/LF 换行后核对参考文本哈希、标签、唯一 ID、分数与预测一致性。"
        "宏平均对数据集等权，避免大数据集主导；不混入 PII，不把 8 次提示词/阈值运行当作独立样本。\n\n"
        "参考输入按原 runner 的 pandas.read_csv 语义读取（包括空值转字符串和 NUL 字符截断）；"
        "这反映当时实际发给模型的内容，不代表原始数据字节完整性。"
        "例如 detect-jailbreak 原始文本中存在 NUL，原 runner 已发生截断，各供应商对比沿用相同输入。\n\n"
        "共同基线数据集：" + ("、".join(report["baseline_datasets"]) or "无") + "。\n\n"
        "质量指标排除请求错误，但覆盖率与错误数独立展示，避免将失败隐藏成高准确率。"
        "全部配置明细使用各文件自身的有效样本；跨模型公平比较请看共同样本表。"
        "complete 仅表示所有参考样本均已落盘，不表示请求全部成功。"
        "missing 表示未发现输出；in_progress 表示日志有记录但无输出，不保证进程仍在运行。\n\n"
        "日志 elapsed / processed 是整轮摊销平均每样本耗时，含失败等待、客户端及网络开销；"
        "仅 CSV 与日志完整计数一致的轮次进入耗时比较。基线耗时按处理行数加权；"
        "Avg latency 则对共同基线各数据集的整轮平均耗时等权平均，缺少任一完整计时即不提供均值。"
        "它使用全部处理样本，质量指标则使用共同成功样本。没有单请求计时，不能计算请求 P50/P95/P99。"
        "报告不会从 50 条进度块伪造请求分位数。\n\n"
        "**PII 是“是否含 PII”的普通二分类，不是脱敏质量评测。** 当前数据只有正样本，仅报告正样本检出率，"
        "单类 AUROC、误报率、平衡准确率无定义；"
        "没有预测脱敏文本，不计算 masking/span 指标。"
        "未输出原始文本、敏感字段或完整错误内容；报告可离线打开，无 CDN、无联网请求。"
    )))
    sections.append(("共同基线总览", markdown_table(
        ["供应商 / 模型", "完整配置", "共同数据集", "Average Accuracy", "Avg Precision",
         "Avg Recall", "宏 F1", "宏平衡准确率", "宏 AUROC",
         "宏误报率", "宏漏报率", "Avg latency ms（数据集等权）",
         "基线平均耗时 ms（样本加权）", "全套请求错误/已落盘行"],
        [[
            f"{r['provider']} / {r['model']}", f"{r['complete']}/{r['total']}",
            r["baseline_datasets"], fmt(r["macro_accuracy"]), fmt(r["macro_precision"]),
            fmt(r["macro_recall"]), fmt(r["macro_f1"]), fmt(r["macro_balanced_accuracy"]),
            fmt(r["macro_auroc"]), fmt(r["macro_fpr"]), fmt(r["macro_fnr"]),
            fmt(r["macro_elapsed_ms"], False), fmt(r["baseline_elapsed_ms"], False),
            f"{r['errors']}/{r['rows']}",
        ] for r in report["overview"]],
    )))
    sections.append(("逐数据集固定基线（共同成功样本）", markdown_table(
        ["数据集", "供应商 / 模型", "共同样本", "覆盖率", "Accuracy", "Precision",
         "Recall", "F1", "AUROC", "平衡准确率", "误报率", "漏报率", "多数类准确率"],
        [[
            r["dataset"], f"{r['provider']} / {r['model']}", r["n"],
            fmt(r["common_coverage"]), fmt(r["accuracy"]), fmt(r["precision"]),
            fmt(r["recall"]), fmt(r["f1"]), fmt(r["auroc"]),
            fmt(r["balanced_accuracy"]), fmt(r["fpr"]), fmt(r["fnr"]), fmt(r["majority_accuracy"]),
        ] for r in report["paired"] if r["prompt"] == "v1" and r["threshold"] == 0.5
         and not r["dataset"].startswith("pii_")],
    )))
    pair_datasets = {}
    for row in report["pairwise"]:
        pair_datasets.setdefault((row["pair"], row["dataset"]), row)
    sections.append(("两两共同基线（利用已完成任务，覆盖范围各不相同）", (
        "每一对供应商单独取共同完整数据集与共同成功样本，固定 v1 / 0.5；"
        "不同配对覆盖的数据集可能不同，因此只可在同一配对内部比较。"
        "共同成功样本为零的数据集不参与质量均值，排除情况如下表明确列出。\n\n"
        + markdown_table(
            ["配对", "供应商 / 模型", "共同数据集", "宏 F1", "宏 AUROC", "宏平衡准确率", "宏误报率", "宏漏报率"],
            [[
                r["pair"], f"{r['provider']} / {r['model']}", "、".join(r["datasets"]),
                fmt(r["macro_f1"]), fmt(r["macro_auroc"]), fmt(r["macro_balanced_accuracy"]),
                fmt(r["macro_fpr"]), fmt(r["macro_fnr"]),
            ] for r in pairwise_summary(report)],
        )
        + "\n\n" + markdown_table(
            ["配对", "数据集", "共同有效样本", "共同覆盖率", "质量评估状态"],
            [[
                row["pair"], row["dataset"], row["n"], fmt(row["common_coverage"]),
                "参与均值" if row["n"] else "无共同成功样本，排除质量均值",
            ] for row in pair_datasets.values()],
        )
    )))
    tuned = []
    for provider in report["providers"]:
        for dataset in DATASETS:
            if dataset.startswith("pii_"):
                continue
            candidates = [
                r for r in report["runs"] if r["provider"] == provider and r["dataset"] == dataset
                and r["status"] == "complete" and r["valid"]
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: (r["f1"], -r["threshold"], r["prompt"] == "v1"))
            base = next((r for r in candidates if r["prompt"] == "v1" and r["threshold"] == 0.5), None)
            tuned.append([
                dataset, f"{provider} / {MODELS[provider]}", f"{best['prompt']} / {best['threshold']}",
                fmt(best["f1"]), fmt(base["f1"] if base else None),
                fmt(best["fpr"]), fmt(best["fnr"]), f"{len(candidates)}/8",
            ])
    sections.append(("提示词与阈值探索（同测试集选优，非公平总排名）", (
        "以下按各文件自身成功样本选 F1 最优；不同轮次可能包含不同请求错误，"
        "不能将该结果当作独立验证后的提升。完整 v1/v2 × 4 阈值结果见明细。\n\n"
        + markdown_table(
            ["数据集", "供应商 / 模型", "最优提示词 / 阈值", "最优 F1", "自身基线 F1", "误报率", "漏报率", "可用配置"],
            tuned,
        )
    )))
    sections.append(("重复调用一致性：口径与逐数据集结果", (
        "将同一供应商、数据集、提示词的 0.5/0.6/0.7/0.8 四轮视作四次重复调用。"
        "阈值只在客户端用于判定，不进入模型请求；不直接比较各 CSV 原有 prediction。"
        "v1、v2 分开统计，按 sample_id 对齐，四轮均完整落盘后仅使用四次均成功的样本。"
        "少一轮或存在部分落盘即标 incomplete_repeats，不以较少轮次替代；"
        "四轮完整但无共同成功样本标 no_common_success，稳定性无定义而不是零波动。\n\n"
        "增量更新中，四轮汇总齐全但缺少重新计算所需的历史逐样本预测时标 unavailable_raw；"
        "不从汇总分数推算稳定性，也不沿用已被新 CSV 替换的旧一致性指标。\n\n"
        "每个样本计算四个 score 的总体标准差（ddof=0）与极差，再对样本取平均；"
        "分数保持原始 0–1 单位。判定统一使用 score >= 0.5，"
        "翻转率为四次判定中至少出现一次正类、一次负类的样本比例，不是相邻轮次翻转次数。"
        "三项指标均越低表示越一致。覆盖率为四次均成功样本 / 参考样本；"
        "请求失败率为已有 CSV 的失败行 / 已落盘总行，缺失轮次不算成功也不算请求失败。\n\n"
        "逐数据集表使用各模型自身的四轮成功交集，样本可能不同，不作为公平总排名。"
        "历史调用可能混有后端版本、路由或运行时段变化，无法将差异完全归因于模型随机性；"
        "仅四次调用是探索性观察，不提供伪造置信区间。稳定不代表正确，恒定错误输出也可能零波动。"
        "原阈值敏感性曲线同时混有重复调用波动，不是固定 score 下的纯阈值效应。\n\n"
        + markdown_table(
            ["数据集", "供应商 / 模型", "提示词", "状态", "已有轮次", "完整轮次",
             "四次均成功/参考", "覆盖率", "平均分数标准差", "平均分数极差",
             "判定翻转样本", "判定翻转率（统一 0.5）", "失败/已落盘请求", "请求失败率"],
            [[
                r["dataset"], f"{r['provider']} / {r['model']}", r["prompt"], r["status"],
                f"{r['available_repeats']}/4", f"{r['complete_repeats']}/4",
                f"{r['n']}/{r['expected']}", fmt(r["coverage"]),
                *(f"{r[key]:.6f}" if r[key] is not None else "—" for key in STABILITY_METRICS[:2]),
                r["flipped"] if r["n"] else "—",
                fmt(r["decision_flip_rate"]), f"{r['errors']}/{r['rows']}",
                fmt(r["error_rate"]),
            ] for r in report["stability"]],
        )
    )))
    sections.append(("重复调用一致性：共同样本模型均值", (
        "每个提示词单独选取所有供应商四轮均完整的数据集，"
        "在每个数据集内进一步取所有供应商全部四轮的成功样本交集。"
        "只对有共同成功样本的非 PII 数据集等权平均，不与上方各自样本均值混排。"
        "不同提示词的共同数据集或样本可能不同；PII 仅在逐数据集一致性中展示。"
        "覆盖率按共同样本总数 / 参考样本总数计算；请求失败率使用这些数据集全部已落盘请求。\n\n"
        + markdown_table(
            ["供应商 / 模型", "提示词", "共同数据集", "共同样本/参考", "共同覆盖率",
             "宏平均分数标准差", "宏平均分数极差", "宏平均判定翻转率", "请求失败率"],
            [[
                f"{r['provider']} / {r['model']}", r["prompt"], "、".join(r["datasets"]) or "无",
                f"{r['n']}/{r['expected']}", fmt(r["coverage"]),
                *(f"{r[key]:.6f}" if r[key] is not None else "—" for key in STABILITY_METRICS[:2]),
                fmt(r["decision_flip_rate"]), fmt(r["error_rate"]),
            ] for r in report["stability_overview"]],
        )
    )))
    sections.append(("PII 普通二分类（是否含 PII，不是脱敏质量）", markdown_table(
        ["数据集", "供应商 / 模型", "提示词 / 阈值", "状态", "有效/预期", "正样本检出率"],
        [[
            r["dataset"], f"{r['provider']} / {r['model']}", f"{r['prompt']} / {r['threshold']}",
            r["status"], f"{r['valid']}/{r['expected']}", fmt(r.get("recall")),
        ] for r in report["runs"] if r["dataset"].startswith("pii_")],
    )))
    sections.append(("配置覆盖与日志耗时", markdown_table(
        ["供应商 / 模型", "数据集", "提示词 / 阈值", "状态", "落盘/预期", "有效覆盖率",
         "错误率", "日志进度", "日志累计 s", "整轮平均 ms"],
        [[
            f"{r['provider']} / {r['model']}", r["dataset"], f"{r['prompt']} / {r['threshold']}",
            r["status"], f"{r['rows']}/{r['expected']}", fmt(r["coverage"]),
            fmt(r["error_rate"]),
            f"{r['log_processed']}/{r['log_total']}" if r["log_total"] is not None else "—",
            fmt(r["log_elapsed_s"], False), fmt(r["mean_elapsed_ms"], False),
        ] for r in report["runs"]],
    )))
    error_rows = []
    for provider in report["providers"]:
        own = [row for row in report["runs"] if row["provider"] == provider]
        categories = Counter()
        for row in own:
            categories.update(row["error_categories"])
        total = sum(row["rows"] for row in own)
        errors = sum(categories.values())
        error_rows.append([
            f"{provider} / {MODELS[provider]}", total, errors,
            fmt(errors / total if total else None),
            "；".join(f"{name}: {count}" for name, count in sorted(categories.items())) or "无",
        ])
    sections.append(("请求可靠性（各自已落盘范围，不作为同任务排名）", markdown_table(
        ["供应商 / 模型", "已落盘请求数", "失败数", "失败率", "异常类型计数"],
        error_rows,
    )))
    external = report.get("external_comparison")
    if external:
        sections.append(("OpenJev 内部汇总结果（4B，非共同样本基线）", (
            external["notes"] + "\n\n"
            + markdown_table(
                ["数据集", "提示词", "阈值", "Rows（总行数）", "有效样本数", "错误数",
                 "Accuracy", "Precision", "Recall", "F1", "AUROC", "四轮平均耗时 ms"],
                [[
                    row["dataset"], row["prompt"], row["threshold"], row["rows"],
                    "未知", "未知",
                    *(fmt(row[key]) for key in ("accuracy", "precision", "recall", "f1", "auroc")),
                    fmt(row["mean_elapsed_ms"], False),
                ] for row in external["source_rows"]],
            )
        )))
        if external.get("internal_source_rows"):
            sections.append(("新增内部模型汇总（非共同样本基线）", markdown_table(
                ["模型", "数据集", "提示词", "总行数", "有效样本数", "错误数", "有效覆盖率",
                 "Accuracy", "Precision", "Recall", "F1", "AUROC", "单轮平均耗时 ms"],
                [[
                    external["display_names"][row["provider"]], row["dataset"], row["prompt"],
                    row["rows"], row["valid"], row["errors"], fmt(row["valid"] / row["rows"]),
                    *(fmt(row[key]) for key in ("accuracy", "precision", "recall", "f1", "auroc")),
                    fmt(row["mean_elapsed_ms"], False),
                ] for row in external["internal_source_rows"]],
            )))
        sections.append(("多模型参考均值（同数据集，非同成功样本）", (
            "固定分类阈值 0.5，v1/v2 分开统计，按各提示词共同测试的非 PII 数据集，"
            "对数据集等权平均。各模型使用自身有效样本，未对齐共同 sample_id，不能当作公平排名。"
            "新增模型仅参与已有提示词的参考均值，不缩减原有模型的 v2 范围。"
            "任一所选数据集的指标缺失则该指标均值保持 N/A，不缩减数据集范围。"
            "原四模型优先使用四轮完整日志均值；v1 缺失四轮均值时，可使用与全量 CSV 对齐的完整阈值 0.5 单轮计时。"
            "新增模型为用户提供的单轮计时，协议不同，仅作参考；各数据集耗时再等权平均。"
            "任一所选数据集缺少所需计时即不提供模型平均耗时，不用部分计时替代。\n\n"
            + markdown_table(
                ["模型", "提示词", "共同数据集", "Average Accuracy", "Avg Precision",
                 "Avg Recall", "Avg F1", "Avg AUROC", "Avg latency ms"],
                [[
                    external.get("display_names", {}).get(row["provider"], row["provider"]),
                    row["prompt"], "、".join(row["datasets"]) or "无",
                    *(fmt(row[key]) for key in ("accuracy", "precision", "recall", "f1", "auroc")),
                    fmt(row["mean_elapsed_ms"], False),
                ] for row in external["overview"]],
            )
        )))
    if report["warnings"]:
        sections.append(("数据警告", "\n\n".join(report["warnings"])))
    return sections


def html_table(headers, rows, table_id=""):
    return (
        f'<div class="scroll"><table id="{table_id}"><thead><tr>'
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def markdown_html(text):
    blocks, lines = [], text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].startswith("| "):
            end = index
            while end < len(lines) and lines[end].startswith("| "):
                end += 1
            cells = [
                [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
                for line in lines[index:end]
            ]
            blocks.append(html_table(cells[0], cells[2:]))
            index = end
        else:
            if lines[index]:
                blocks.append("<p>" + html.escape(lines[index]) + "</p>")
            index += 1
    return "\n".join(blocks)


def chart(report):
    colors = ("#2563eb", "#e87924", "#059669")
    bars = []
    for index, row in enumerate(report["overview"]):
        value = row["macro_f1"]
        bars.append(
            '<div class="bar-row"><span>' + html.escape(f"{row['provider']} / {row['model']}")
            + f'</span><div class="track"><div class="bar" style="width:{(value or 0) * 100:.2f}%;'
            + f'background:{colors[index % len(colors)]}"></div></div><b>{fmt(value)}</b></div>'
        )
    return '<div class="chart"><h3>共同基线宏平均 F1（越高越好）</h3>' + "".join(bars) + "</div>"


def write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report["conclusions"] = conclusions(report)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    exports = [
        ("runs.csv", report["runs"]), ("paired.csv", report["paired"]),
        ("pairwise.csv", report["pairwise"]),
        ("stability.csv", report["stability"]),
        ("stability_paired.csv", report["stability_paired"]),
        ("stability_overview.csv", report["stability_overview"]),
    ]
    if report.get("external_comparison"):
        external = report["external_comparison"]
        exports.extend([
            ("openjev_summary.csv", external["source_rows"]),
            ("external_comparison_runs.csv", external["runs"]),
            ("external_comparison_overview.csv", external["overview"]),
        ])
        if external.get("internal_source_rows"):
            exports.append(("internal_model_summary.csv", external["internal_source_rows"]))
    for filename, records in exports:
        columns = list(dict.fromkeys(key for record in records for key in record))
        with (output_dir / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    sections = report_sections(report)
    title = "模型能力对比报告"
    timestamp = "生成时间（UTC）：" + report["generated_at"]
    markdown = f"# {title}\n\n{timestamp}\n\n" + "\n\n".join(
        f"## {heading}\n\n{body}" for heading, body in sections
    )
    (output_dir / "comparison.md").write_text(markdown + "\n", encoding="utf-8")
    details_headers = [
        "供应商 / 模型", "数据集", "提示词", "阈值", "状态", "有效/预期", "错误率",
        "准确率", "Precision", "Recall", "F1", "AUROC", "误报率", "漏报率",
        "TN", "FP", "FN", "TP", "平均 ms",
    ]
    details_rows = [
        [
            f"{r['provider']} / {r['model']}", r["dataset"], r["prompt"], r["threshold"], r["status"],
            f"{r['valid']}/{r['expected']}", fmt(r["error_rate"]),
            *(fmt(r.get(key)) if not r["dataset"].startswith("pii_") else "不适用"
              for key in ("accuracy", "precision", "recall", "f1", "auroc", "fpr", "fnr")),
            *(fmt(r.get(key), False) for key in ("tn", "fp", "fn", "tp")),
            fmt(r["mean_elapsed_ms"], False),
        ] for r in report["runs"]
    ]
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>模型能力对比报告</title><style>
body{font:15px/1.7 system-ui,sans-serif;margin:0;background:#f3f6fb;color:#172438}
main{max-width:1450px;margin:auto;padding:28px}section,.chart{background:white;padding:22px;margin:20px 0;border-radius:12px}
h1,h2,h3{line-height:1.4}h2{color:#174c90}p{max-width:1150px}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #e2e8f0;white-space:nowrap}
th{background:#edf3fc;position:sticky;top:0}tbody tr:hover{background:#f0f7ff}
nav a{display:inline-block;margin:4px 14px 4px 0;color:#174c90}
.bar-row{display:grid;grid-template-columns:minmax(180px,1fr) 2fr 80px;gap:12px;align-items:center;margin:15px 0}
.track{background:#edf2f8;height:24px;border-radius:5px;overflow:hidden}.bar{height:100%}
input{padding:10px;width:min(500px,90%);font:inherit;border:1px solid #bdcbe0;border-radius:6px}
@media print{body{background:white}main{padding:0}section{break-inside:avoid}.scroll{overflow:visible}table{font-size:9px}nav,input{display:none}}
</style></head><body><main>"""
    document += f"<h1>{title}</h1><p>{html.escape(timestamp)}</p><nav>"
    document += "".join(f'<a href="#s{i}">{html.escape(heading)}</a>' for i, (heading, _) in enumerate(sections))
    document += '<a href="#details">筛选全部配置指标</a></nav>' + chart(report)
    document += "".join(
        f'<section id="s{i}"><h2>{html.escape(heading)}</h2>{markdown_html(body)}</section>'
        for i, (heading, body) in enumerate(sections)
    )
    document += (
        '<section id="details"><h2>筛选全部配置指标</h2>'
        '<p>以下为各文件自身有效样本指标；跨模型比较请使用上方共同样本基线。'
        '搜索支持多个关键词，例如 fastino v2 toxigen；可搜索 missing / in_progress。</p>'
        '<input id="filter" aria-label="筛选配置" placeholder="供应商、数据集、提示词、阈值或状态">'
        + html_table(details_headers, details_rows, "details-table")
        + """</section></main><script>
document.getElementById('filter').addEventListener('input',function(){
 const words=this.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);
 document.querySelectorAll('#details-table tbody tr').forEach(function(row){
  const text=row.textContent.toLowerCase();
  row.hidden=!words.every(function(word){return text.includes(word);});
 });
});
</script></body></html>"""
    )
    (output_dir / "comparison.html").write_text(document + "\n", encoding="utf-8")
