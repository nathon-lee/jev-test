"""Reference-only comparisons with user-supplied aggregate results."""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from benchmark.comparison import (
    DATASETS, PROGRESS, PROMPTS, START, THRESHOLDS,
    classification_metrics, read_predictions, read_reference,
)


METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms")
FINE_TUNED_DISPLAY_NAME = "gliguard-LLMGuardrails-300M-fine-tuned"
GLIGUARD_PROVIDER = "fastino-gliguard-300m"
GLIGUARD_DISPLAY_NAME = "Fastino GLiGuard-LLMGuardrails-300M"
POC_PROVIDER = "fastino-systemone-21ac7ad2"
POC_MODEL = "21ac7ad2-f9b5-5e6f-801b-45b52e26bdc9"
POC_DISPLAY_NAME = "Fastino POC model"
GLINER25_PROVIDER = "fastino-gliner2.5-decide"
GLINER25_MODEL = "fastino/gliner2.5-decide"
GLINER25_DISPLAY_NAME = "Fastino GLiNER2.5-Decide"
LUNA_PROVIDER = "luna"
NATIVE_PRESET_PROMPT = "native_presets"
QWEN_PROVIDER = "qwen3guard-stream-8b"
STRAIKER_PROVIDER = "straiker.ai"
STRAIKER_NOTES = (
    "User-supplied aggregate results from outputs-straiker-full-vzs8KFDy; raw predictions and logs "
    "were not available for independent verification. Prompt v1; decision threshold and resolved "
    "model/version were not supplied (N/A), unlike the threshold-0.5 reference runs. "
    "All eight runs completed; JailbreakHub has one error. Quality excludes failed samples. "
    "Latency is single-run log elapsed/processed, including failures and client/network overhead, "
    "not successful-request mean latency or pure server inference time. "
    "PII is binary present/absent classification, not extraction or masking quality; "
    "positives-only PII FPR and AUROC are undefined. Equal dataset weights; no sample-aligned "
    "ranking, v2 comparison or repeat-call consistency is inferred."
)
QWEN_NOTES = (
    "User-supplied summary only; no prediction CSV or source-data integrity audit was performed. "
    "Prompt v1, threshold 0.5; 12 reported prediction CSVs. "
    "Scores are binary general-safety labels, not probabilities. Metrics exclude error rows; "
    "AUROC is unavailable for single-class valid labels. PII detection is unsupported: all three PII "
    "rows are historical safety-proxy metrics for audit only, excluded from comparison metrics. "
    "Supplied internal log: logs/qwen3guard-stream-8b_internal_20261029_163330.log (max_workers=1). "
    "The PII golden rerun log is logs/qwen3guard-stream-8b_pii_rerun_20261029_203544.log "
    "(max_workers=1), reported complete at 2700/2700 with a Saved record. "
    "Public-dataset log: logs/qwen3guard_v1_t0.5_20261027_233111.log (max_workers=9 setting), "
    "including the completed detect-jailbreak restart; duplicate backup records are not counted. "
    "Log paths are normalized from supplied OCR, not independently verified. "
    "The supplied description reports completed progress and Saved records with matching filenames/counts; "
    "these do not prove the current CSVs were never overwritten afterward. "
    "log_elapsed_mean_ms_per_sample = completed dataset elapsed seconds * 1000 / processed rows, "
    "including failed requests, loop overhead and waiting; mean_elapsed_ms in chart exports aliases this value. "
    "It is not the successful-request mean and cannot recover P95. "
    "p95_latency_ms uses successful-request latency_ms from CSVs. NaN means unavailable; "
    "no per-request latency is imputed. Latency includes client and network overhead, not pure server inference time. "
    "This update changes latency only; prior classification metrics and counts are retained where OCR conflicts. "
    "Different datasets/runs are not controlled latency comparisons. "
    "Do not pool task-level F1 scores into a model ranking. "
    "Supplied evaluation command (not executed): "
    "python -m scripts.evaluate_model_suite --provider qwen3guard-stream-8b; "
    "this reproduces classification metrics and CSV P95. Calculate log averages separately using the formula above."
)
FOCUSED_DATASETS = ("toxigen", "hatecheck", "aegis2")
DATASET_CATEGORIES = {
    "harmful_content": ("toxigen", "hatecheck", "aegis2"),
    "jailbreak": ("detect-jailbreak", "jailbreak-classification", "jailbreakhub"),
    "pii": ("pii_masking_nano_1k", "pii_masking_mini_10k"),
}
FOCUSED_SUPPLIED_AVERAGES = (
    ("Dynamo CPU", 0.8965, 0.8888, 0.9086, 0.9527, 20000.0),
    ("Dynamo GPU (A100)", 0.8941, 0.8869, 0.9106, 0.9524, 55.0),
    ("Dynamo GPU (L4)", 0.8939, 0.8869, 0.9102, 0.9524, 100.0),
    ("Mistral", 0.8333, 0.8564, 0.834, None, 1671.08),
    ("Warden", 0.7773, 0.8203, 0.7611, None, 328.53),
)
OPENJEV_IDENTITY = {"request_model": "not supplied", "resolved_model": None}
INTERNAL_IDENTITIES = {
    "openjev-32b": {
        "request_model": "not supplied", "resolved_model": None,
        "display_name": "OpenJev 32B", "source_tag": "apus-openjev-35b-a3b",
    },
    "microsoft-decision-1": {
        "request_model": "not supplied", "resolved_model": None,
        "display_name": "Microsoft Decision-1",
    },
}
INTERNAL_NOTES = (
    "新增 OpenJev 32B、Microsoft Decision-1 为用户提供的内部 v1 / 0.5 汇总，"
    "耗时来自每个数据集一次完整运行的 elapsed/processed（max workers=8），"
    "不是原四模型的四轮均值，计时协议不同，仅作参考。"
    "OpenJev 32B 的原文件标记为 apus-openjev-35b-a3b，显示名称按用户确认使用 32B，"
    "不从文件名推断实际模型版本。其 ToxiGen 五项质量指标均已由用户补全，"
    "Accuracy 为 0.693975；按完整数据集范围计算均值，不缩减范围或反推指标。"
    "新增模型质量指标排除错误行，尤其 OpenJev 32B 的部分任务请求错误较多，"
    "不能将成功样本分数当作全量表现。Microsoft 日志在八个完整任务后中断，"
    "提供的耗时仅取已完成任务；均无单请求 P95。"
    "没有逐样本预测，不加入共同成功样本排名或重复调用一致性。"
)
NOTES = (
    "OpenJev 4B 数值取自用户最初提供的清晰汇总表，分类阈值均为 0.5；"
    "补充 OCR 文本有错字，不据此修正数值或推断混淆矩阵。Rows 为总测试行数，"
    "有效样本数和错误数未可靠录入，保持未知，不将 Rows 当作有效样本数。"
    "OpenJev 只提供汇总指标，未提供逐样本 score、预测或模型版本；"
    "不能核验相同输入、共同成功样本或计算重复调用稳定性，不加入原有公平基线与排名。"
    "参考图按各模型自身成功样本统计，仅作描述性对照。"
    "原四模型优先使用每个数据集/提示词四轮完整运行的 elapsed/processed 算术平均。"
    "v1 缺少完整四轮计时时，仅在阈值 0.5 的完整单轮日志与全量 CSV 行数一致时，"
    "使用该单轮的 elapsed/processed，并标记为 single_complete_run；v2 不使用该回退。"
    "包含失败请求、等待、客户端及网络开销，不是纯推理时间；无可核验的完整计时标 N/A。"
    "并发日志无法逐阈值归属时，仅在四个阈值各启动一次、四次全量完成及保存均可核验，"
    "且无后续重跑时使用四轮耗时均值；不据此回填任一阈值的耗时。"
)


def read_openjev_summary(path):
    return _read_summary(path, internal=False)


def read_internal_summary(path):
    return _read_summary(path, internal=True)


def _read_summary(path, internal):
    records, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "prompt", "threshold", "rows", *METRICS}
        if internal:
            required.update(("provider", "valid", "errors"))
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing summary columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            dataset, prompt = row["dataset"], row["prompt"]
            provider = row["provider"] if internal else "openjev"
            if internal and provider not in INTERNAL_IDENTITIES:
                raise ValueError(f"{path}:{line}: unknown internal provider")
            if dataset not in DATASETS or prompt not in PROMPTS:
                raise ValueError(f"{path}:{line}: unknown dataset/prompt")
            key = provider, dataset, prompt
            if key in seen:
                raise ValueError(f"{path}:{line}: duplicate summary configuration: {key}")
            seen.add(key)
            threshold, rows = float(row["threshold"]), int(row["rows"])
            if threshold != 0.5 or rows <= 0:
                raise ValueError(f"{path}:{line}: summaries require threshold 0.5 and positive rows")
            metrics = {}
            for metric in METRICS:
                if row[metric] == "" and (metric == "auroc" or (internal and metric != "mean_elapsed_ms")):
                    metrics[metric] = None
                    continue
                value = float(row[metric])
                if not math.isfinite(value) or value < 0 or (metric != "mean_elapsed_ms" and value > 1):
                    raise ValueError(f"{path}:{line}: invalid {metric}")
                metrics[metric] = value
            valid, errors = None, None
            if internal:
                valid, errors = int(row["valid"]), int(row["errors"])
                if valid < 0 or errors < 0 or valid + errors != rows:
                    raise ValueError(f"{path}:{line}: invalid valid/errors counts")
            records.append({
                "provider": provider, "dataset": dataset, "prompt": prompt,
                "threshold": threshold, "rows": rows, "valid": valid, "errors": errors,
                "source_kind": "user_supplied_aggregate", **metrics,
                **({"latency_source_kind": "single_complete_run"} if internal else {}),
            })
    if not records:
        raise ValueError(f"{path}: empty summary")
    return records


def reference_prompt(row):
    # Native presets share the v1 reference panels, not the v1 prompting protocol.
    return "v1" if row["prompt"] == NATIVE_PRESET_PROMPT else row["prompt"]


def reference_averages(records, providers, datasets=None, include_pii=False):
    candidates = list(DATASETS if datasets is None else datasets)
    if datasets is not None and (
        not candidates or len(set(candidates)) != len(candidates)
        or any(dataset not in DATASETS or (not include_pii and dataset.startswith("pii_"))
               for dataset in candidates)
    ):
        raise ValueError("Reference averages require distinct " + ("" if include_pii else "non-PII ") + "datasets")
    lookup = {(row["provider"], row["dataset"], reference_prompt(row)): row for row in records}
    overview = []
    for prompt in PROMPTS:
        active = [
            provider for provider in providers
            if provider not in INTERNAL_IDENTITIES
            or any(key[0] == provider and key[2] == prompt for key in lookup)
        ]
        common = [
            dataset for dataset in candidates if (include_pii or not dataset.startswith("pii_"))
            and all(
                (row := lookup.get((provider, dataset, prompt))) is not None
                and (
                    provider == "openjev"
                    or (row.get("source_kind") == "user_supplied_aggregate" and row["valid"] > 0)
                    or (row.get("status") == "complete" and row["valid"] > 0)
                )
                for provider in active
            )
        ]
        selected = common if datasets is None else candidates
        complete = datasets is None or len(common) == len(candidates)
        for provider in active:
            rows = [lookup[(provider, dataset, prompt)] for dataset in selected] if complete else []
            metrics = {}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                metrics[metric] = mean(values) if values and all(value is not None for value in values) else None
            protocols = {
                row["prompt"] for row in records
                if row["provider"] == provider and reference_prompt(row) == prompt
            }
            overview.append({
                "provider": provider,
                "prompt": next(iter(protocols)) if len(protocols) == 1 else prompt,
                "datasets": selected, **metrics,
            })
    return overview


def category_reference_averages(report):
    comparison = report["external_comparison"]
    records = [
        row for row in comparison["runs"]
        if reference_prompt(row) == "v1" and (
            row["threshold"] == 0.5 or row["provider"] in comparison.get("aggregate_models", {})
        )
    ]
    result = []
    for category, datasets in DATASET_CATEGORIES.items():
        for provider in comparison["providers"]:
            rows = reference_averages(records, [provider], datasets, include_pii=True)
            result.extend(
                {
                    "category": category, **row,
                    "threshold": None if provider in comparison.get("aggregate_models", {}) else 0.5,
                    "task_supported": not (
                        category == "pii" and (
                            provider in comparison.get("summary_models", {})
                            or provider in comparison.get("native_preset_models", {})
                        )
                    ),
                }
                for row in rows if reference_prompt(row) == "v1"
            )
    return result


def reference_comparison_averages(comparison, datasets=None, include_pii=False):
    additional = {
        **comparison.get("fine_tuned_models", {}),
        **comparison.get("classification_models", {}),
        **comparison.get("summary_models", {}),
        **comparison.get("native_preset_models", {}),
        **comparison.get("aggregate_models", {}),
    }
    providers = [provider for provider in comparison["providers"] if provider not in additional]
    rows = reference_averages(comparison["runs"], providers, datasets, include_pii=include_pii)
    if not additional:
        return rows
    for provider in additional:
        prompts = {
            reference_prompt(row) for row in comparison["runs"] if row["provider"] == provider
        }
        for prompt in prompts:
            scope = next(row["datasets"] for row in rows if reference_prompt(row) == prompt)
            if not scope:
                continue
            rows.extend(
                row for row in reference_averages(
                    comparison["runs"], [provider], scope, include_pii=include_pii,
                ) if reference_prompt(row) == prompt
            )
    lookup = {(row["provider"], reference_prompt(row)): row for row in rows}
    return [
        lookup[(provider, prompt)] for prompt in PROMPTS for provider in comparison["providers"]
        if (provider, prompt) in lookup
    ]


def v1_reference_scope_averages(report, datasets=None):
    comparison = report["external_comparison"]
    raw = {
        (row["provider"], row["dataset"]): row for row in report["runs"]
        if row["prompt"] == "v1" and row["threshold"] == 0.5
    }
    timing, unsupported = {}, set()
    for row in comparison["runs"]:
        if reference_prompt(row) != "v1":
            continue
        key = (row["provider"], row["dataset"])
        if row.get("task_supported") is False:
            unsupported.add(key)
        if (
            row["provider"] == "openjev" or row["provider"] in INTERNAL_IDENTITIES
            or row["provider"] in comparison.get("fine_tuned_models", {})
            or row["provider"] in comparison.get("classification_models", {})
            or row["provider"] in comparison.get("summary_models", {})
            or row["provider"] in comparison.get("native_preset_models", {})
            or row["provider"] in comparison.get("aggregate_models", {})
        ):
            timing[key] = row["mean_elapsed_ms"]
        else:
            run = raw.get(key)
            timing[key] = run.get("mean_elapsed_ms") if (
                run is not None and run["status"] == "complete" and run["rows"] > 0
            ) else None
    result = []
    for scope, selected, include_pii in (
        ("non_pii", datasets, False),
        ("pii", [dataset for dataset in DATASETS if dataset.startswith("pii_")], True),
        ("all", list(DATASETS), True),
    ):
        rows = [
            row for row in reference_comparison_averages(
                comparison, selected, include_pii=include_pii,
            ) if reference_prompt(row) == "v1"
        ]
        scope_datasets = rows[0]["datasets"] if rows else []
        timed = [
            dataset for dataset in scope_datasets
            if all(
                timing.get((row["provider"], dataset)) is not None
                for row in rows
                if row["provider"] not in comparison.get("summary_models", {})
                and row["provider"] not in comparison.get("native_preset_models", {})
            )
        ]
        for row in rows:
            own = [timing.get((row["provider"], dataset)) for dataset in scope_datasets]
            supported = not any((row["provider"], dataset) in unsupported for dataset in scope_datasets)
            result.append({
                "scope": scope, **row,
                "threshold": None if row["provider"] in comparison.get("aggregate_models", {}) else 0.5,
                "timing_datasets": timed,
                "mean_elapsed_ms": mean(own) if supported and own and all(value is not None for value in own) else None,
                "shared_elapsed_ms": (
                    mean(timing[(row["provider"], dataset)] for dataset in timed)
                    if supported and timed
                    and all(timing.get((row["provider"], dataset)) is not None for dataset in timed)
                    else None
                ),
            })
    return result


def three_dataset_averages(report):
    comparison = report["external_comparison"]
    overview = [
        row for row in reference_comparison_averages(comparison, FOCUSED_DATASETS)
        if reference_prompt(row) == "v1"
    ]
    timing = {
        (row["provider"], row["dataset"]): row
        for row in report["runs"] if row["prompt"] == "v1" and row["threshold"] == 0.5
    }
    for row in overview:
        row["f1_source"] = "mean_per_dataset" if row["f1"] is not None else "unavailable"
        supplied_latency = report.get("luna_summary", {}).get("three_dataset_latency")
        if row["provider"] == LUNA_PROVIDER and supplied_latency:
            row["mean_elapsed_ms"] = supplied_latency["mean_elapsed_ms"]
        if (
            row["provider"] == "openjev" or row["provider"] in INTERNAL_IDENTITIES
            or row["provider"] in comparison.get("fine_tuned_models", {})
            or row["provider"] in comparison.get("classification_models", {})
            or row["provider"] in comparison.get("summary_models", {})
            or row["provider"] in comparison.get("native_preset_models", {})
            or row["provider"] in comparison.get("aggregate_models", {})
        ):
            continue
        runs = [timing.get((row["provider"], dataset)) for dataset in FOCUSED_DATASETS]
        row["mean_elapsed_ms"] = mean(run["mean_elapsed_ms"] for run in runs) if all(
            run is not None and run["status"] == "complete" and run["mean_elapsed_ms"] is not None
            for run in runs
        ) else None
    for provider, accuracy, precision, recall, auroc, latency in FOCUSED_SUPPLIED_AVERAGES:
        overview.append({
            "provider": provider, "prompt": "not_supplied", "datasets": list(FOCUSED_DATASETS),
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "f1_source": "harmonic_mean_of_supplied_precision_recall",
            "auroc": auroc, "mean_elapsed_ms": latency,
        })
    return overview


def attach_straiker_summary(report, path):
    path = Path(path)
    records, seen = [], set()
    counts = ("rows", "valid", "errors", "tp", "tn", "fp", "fn")
    scores = (*METRICS[:-1], "fpr", "fnr")
    required = {*counts, *scores, "dataset", "prompt", "progress", "complete",
                "elapsed_s", "mean_elapsed_ms"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing Straiker summary columns")
        for line, supplied in enumerate(reader, 2):
            dataset = supplied["dataset"]
            if dataset not in DATASETS or dataset in seen:
                raise ValueError(f"{path}:{line}: unknown or duplicate Straiker dataset")
            seen.add(dataset)
            row = {key: int(supplied[key]) for key in counts}
            if (
                row["rows"] <= 0 or row["valid"] <= 0 or min(row.values()) < 0
                or row["valid"] + row["errors"] != row["rows"]
                or sum(row[key] for key in ("tp", "tn", "fp", "fn")) != row["valid"]
            ):
                raise ValueError(f"{path}:{line}: invalid Straiker counts")
            if (
                supplied["prompt"] != "v1" or supplied["complete"] != "True"
                or supplied["progress"] != f'{row["rows"]}/{row["rows"]}'
            ):
                raise ValueError(f"{path}:{line}: Straiker summary requires complete v1 runs")
            for key in (*scores, "elapsed_s", "mean_elapsed_ms"):
                value = None if supplied[key] in ("", "NaN") else float(supplied[key])
                undefined = key in ("fpr", "auroc") and row["tn"] + row["fp"] == 0
                if (
                    (value is None and not undefined) or (undefined and value is not None)
                    or (value is not None and (
                        not math.isfinite(value) or value < 0 or (key in scores and value > 1)
                    ))
                ):
                    raise ValueError(f"{path}:{line}: invalid Straiker {key}")
                row[key] = value
            if abs(row["elapsed_s"] * 1000 / row["rows"] - row["mean_elapsed_ms"]) > 0.01:
                raise ValueError(f"{path}:{line}: Straiker timing does not match elapsed/processed")
            records.append({
                "provider": STRAIKER_PROVIDER, "dataset": dataset, "prompt": "v1",
                "threshold": None, **row, "progress": supplied["progress"], "complete": True,
                "source_kind": "user_supplied_aggregate",
                "latency_source_kind": "single_complete_run",
                "result_directory": "outputs-straiker-full-vzs8KFDy",
            })
    if seen != set(DATASETS):
        raise ValueError(f"{path}: Straiker summary must contain all eight datasets")
    comparison = report["external_comparison"]
    comparison["runs"] = [
        row for row in comparison["runs"] if row["provider"] != STRAIKER_PROVIDER
    ] + records
    if STRAIKER_PROVIDER not in comparison["providers"]:
        comparison["providers"].append(STRAIKER_PROVIDER)
    comparison["model_identities"][STRAIKER_PROVIDER] = {
        "request_model": "not supplied", "resolved_model": None, "display_name": "Straiker",
    }
    comparison.setdefault("aggregate_models", {})[STRAIKER_PROVIDER] = {
        "display_name": "Straiker", "source": str(path), "notes": STRAIKER_NOTES,
        "rows": sum(row["rows"] for row in records),
        "valid": sum(row["valid"] for row in records),
        "errors": sum(row["errors"] for row in records), "supplied_rows": records,
        "detail_page": "straiker_summary.html", "summary_csv": "straiker_summary.csv",
    }
    comparison["overview"] = reference_comparison_averages(comparison)


def attach_qwen_summary(report, path):
    path = Path(path)
    supplied_metrics = (*METRICS[:-1], "log_elapsed_mean_ms_per_sample", "p95_latency_ms")
    expected = {*DATASETS, "harmful-content", "harmful-content-golden",
                "jailbreak-golden", "pii-detection-golden"}
    records, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "prompt", "threshold", "rows", "valid", "errors",
                    *supplied_metrics, "latency_log_file"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing Qwen summary columns")
        for line, row in enumerate(reader, 2):
            dataset = row["dataset"]
            if dataset not in expected or dataset in seen:
                raise ValueError(f"{path}:{line}: unknown or duplicate Qwen dataset")
            seen.add(dataset)
            if row["prompt"] != "v1" or float(row["threshold"]) != 0.5:
                raise ValueError(f"{path}:{line}: Qwen summary requires v1 / 0.5")
            rows, valid, errors = (int(row[key]) for key in ("rows", "valid", "errors"))
            if rows <= 0 or min(valid, errors) < 0 or valid + errors != rows:
                raise ValueError(f"{path}:{line}: invalid Qwen coverage counts")
            metrics = {}
            for metric in supplied_metrics:
                value = None if row[metric] in ("", "NaN") else float(row[metric])
                if value is not None and (
                    not math.isfinite(value) or value < 0
                    or (metric in METRICS[:-1] and value > 1)
                ):
                    raise ValueError(f"{path}:{line}: invalid Qwen {metric}")
                metrics[metric] = value
            if metrics["log_elapsed_mean_ms_per_sample"] is not None and not row["latency_log_file"]:
                raise ValueError(f"{path}:{line}: Qwen log latency requires a supplied log reference")
            records.append({
                "provider": QWEN_PROVIDER, "dataset": dataset, "prompt": "v1", "threshold": 0.5,
                "rows": rows, "valid": valid, "errors": errors,
                "file": f"outputs/{QWEN_PROVIDER}/{dataset}_v1_t0.5.csv",
                "task_supported": not dataset.startswith("pii"),
                "source_kind": "user_supplied_aggregate",
                "latency_source_kind": "single_complete_run",
                "latency_log_file": row["latency_log_file"],
                "p95_source_kind": "successful_request_latency_ms",
                "mean_elapsed_ms": metrics["log_elapsed_mean_ms_per_sample"],
                **metrics,
            })
    if seen != expected:
        raise ValueError(f"{path}: Qwen summary must contain all 12 datasets")
    comparison = report["external_comparison"]
    comparison["runs"] = [
        row for row in comparison["runs"] if row["provider"] != QWEN_PROVIDER
    ] + [
        {**row, **({metric: None for metric in METRICS} if not row["task_supported"] else {})}
        for row in records if row["dataset"] in DATASETS
    ]
    if QWEN_PROVIDER not in comparison["providers"]:
        comparison["providers"].append(QWEN_PROVIDER)
    comparison["model_identities"][QWEN_PROVIDER] = {
        "request_model": QWEN_PROVIDER, "resolved_model": None,
        "display_name": "Qwen3Guard-Stream-8B",
    }
    comparison.setdefault("summary_models", {})[QWEN_PROVIDER] = {
        "display_name": "Qwen3Guard-Stream-8B", "source": str(path),
        "rows": sum(row["rows"] for row in records),
        "valid": sum(row["valid"] for row in records),
        "errors": sum(row["errors"] for row in records),
        "notes": QWEN_NOTES, "supplied_rows": records,
        "detail_page": "qwen3guard_stream_8b_summary.html",
        "summary_csv": "qwen3guard_stream_8b_summary.csv",
    }
    comparison["overview"] = reference_comparison_averages(comparison)


def attach_luna_reference_latency(report, mean_elapsed_ms):
    if not math.isfinite(mean_elapsed_ms) or mean_elapsed_ms <= 0:
        raise ValueError("Luna three-dataset latency must be finite and positive")
    if "luna_summary" not in report:
        raise ValueError("Luna latency requires an existing Luna summary")
    model = report["luna_summary"]
    model["three_dataset_latency"] = {
        "datasets": list(FOCUSED_DATASETS), "mean_elapsed_ms": mean_elapsed_ms,
        "source_kind": "user_supplied_measurement",
        "notes": (
            "User-measured latency for the ToxiGen / HateCheck / AEGIS2 reference average only. "
            "Timing protocol and dataset weighting were not supplied. "
            "Not imputed to individual datasets, other scopes or PII."
        ),
    }
    model["source_limitations"] = model["source_limitations"].replace(
        "Scorer version IDs and latency remain N/A.",
        "Scorer version IDs and per-dataset latency remain N/A.",
    )


def attach_luna_summary(report, path):
    path = Path(path)
    expected = {*list(DATASETS)[:6], "pii"}
    records, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "rows", "valid", "errors", *METRICS[:-1]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing Luna summary columns")
        for line, row in enumerate(reader, 2):
            dataset = row["dataset"]
            if dataset not in expected or dataset in seen:
                raise ValueError(f"{path}:{line}: unknown or duplicate Luna dataset")
            seen.add(dataset)
            rows, valid, errors = (int(row[key]) for key in ("rows", "valid", "errors"))
            if rows <= 0 or min(valid, errors) < 0 or valid + errors != rows:
                raise ValueError(f"{path}:{line}: invalid Luna coverage counts")
            if dataset == "pii" and rows != 990:
                raise ValueError(f"{path}:{line}: Luna PII summary requires 990 validation documents")
            metrics = {}
            for metric in METRICS[:-1]:
                value = None if row[metric] in ("", "N/A") else float(row[metric])
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"{path}:{line}: invalid Luna {metric}")
                metrics[metric] = value
            records.append({
                "dataset": dataset, "rows": rows, "valid": valid, "errors": errors,
                "task": "document_level_multilabel" if dataset == "pii" else "binary_classification",
                "f1_averaging": "micro" if dataset == "pii" else "binary",
                "prompt_version": None, "threshold": None if dataset == "pii" else 0.5,
                "scorer_version_id": None, "mean_elapsed_ms": None, **metrics,
            })
    if seen != expected:
        raise ValueError(f"{path}: Luna summary must contain six non-PII datasets and one PII dataset")
    report["luna_summary"] = {
        "display_name": "Luna", "source": str(path),
        "source_kind": "user_supplied_aggregate_unverified",
        "prompt_protocol": (
            "Luna uses original task-specific presets without custom prompts "
            "(prompt version: N/A; not v1/v2), with a binary threshold of 0.5. "
            "Other models use dataset-specific v1/v2 prompts. "
            "Task-definition differences may affect results."
        ),
        "source_limitations": (
            "Values were transcribed from the user-supplied summary, not inferred. "
            "Original summary.json and manifest.json files were not available in the repository; "
            "saved metrics, scorer version IDs and input provenance have not been independently verified. "
            "Scorer version IDs and latency remain N/A. No missing metrics are reconstructed. "
            "Non-PII results appear beside v1 reference results with their native-preset protocol retained. "
            "Task-definition differences may affect results; this is not a controlled prompt comparison. "
            "Not included in sample-aligned rankings, v2 comparisons or binary PII comparisons."
        ),
        "pii_dataset": "ai4privacy/pii-masking-mini-10k",
        "pii_split": "validation",
        "pii_revision": "7b686c2e7475b02e10e38add1eb54a9f604f4361",
        "pii_categories": ["credit_card_info", "email", "phone_number", "ssn"],
        "summary_csv": "luna_summary.csv", "supplied_rows": records,
    }
    if report.get("external_comparison"):
        comparison = report["external_comparison"]
        comparison["runs"] = [
            row for row in comparison["runs"] if row["provider"] != LUNA_PROVIDER
        ] + [
            {
                "provider": LUNA_PROVIDER, "dataset": row["dataset"],
                "prompt": NATIVE_PRESET_PROMPT, "threshold": row["threshold"],
                "rows": row["rows"], "valid": row["valid"], "errors": row["errors"],
                "source_kind": "user_supplied_aggregate",
                **{metric: row[metric] for metric in METRICS},
            }
            for row in records if row["dataset"] != "pii"
        ]
        if LUNA_PROVIDER not in comparison["providers"]:
            comparison["providers"].append(LUNA_PROVIDER)
        comparison["model_identities"][LUNA_PROVIDER] = {
            "display_name": "Luna", "request_model": "not supplied", "resolved_model": None,
        }
        comparison.setdefault("native_preset_models", {})[LUNA_PROVIDER] = {
            "source": str(path), "prompt_version": None, "threshold": 0.5,
            "source_kind": "user_supplied_aggregate_unverified",
            "scorer_version_id": None,
        }
        comparison["overview"] = reference_comparison_averages(comparison)


def attach_finetuned_run(report, run_dir, log_path, data_dir=Path("data/prepared")):
    from benchmark.provider_suite import resolve_provider

    run_dir, log_path, data_dir = Path(run_dir), Path(log_path), Path(data_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    provider = metadata["provider"]
    if provider != "fastino-ft-gg6-r1-ml2048-s7" or metadata["model"] != resolve_provider(provider)["model"]:
        raise ValueError("Unsupported fine-tuned model identity.")
    _attach_fastino_run(
        report, run_dir, log_path, data_dir, metadata,
        "fine_tuned_models", FINE_TUNED_DISPLAY_NAME,
    )


def attach_gliguard_run(report, run_dir, data_dir=Path("data/prepared")):
    from benchmark.provider_suite import resolve_provider

    run_dir, data_dir = Path(run_dir), Path(data_dir)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if (
        status["provider"] != GLIGUARD_PROVIDER
        or status["model"] != resolve_provider(GLIGUARD_PROVIDER)["model"]
        or status["status"] not in ("completed", "completed_with_errors")
        or status["exit_code"] != 0 or status["completed_datasets"] != 6
    ):
        raise ValueError("GLiGuard comparison requires a completed run with the exact model identity.")
    metadata = {
        "provider": status["provider"], "model": status["model"],
        "prompt_version": status["prompt_version"], "decision_threshold": status["threshold"],
        "datasets": list(status["datasets"]),
        "input_sha256": {
            str(Path("data/prepared") / DATASETS[dataset]): details["sha256"]
            for dataset, details in status["datasets"].items()
        },
        "counts": (status["processed_rows"], status["valid_rows"], status["error_rows"]),
        "finished_at": status["finished_at"],
    }
    _attach_fastino_run(
        report, run_dir, run_dir / "run.log", data_dir, metadata,
        "classification_models", GLIGUARD_DISPLAY_NAME,
    )


def attach_poc_run(report, run_dir, data_dir=Path("data/prepared")):
    _attach_completed_classification_run(
        report, run_dir, data_dir, POC_PROVIDER, POC_MODEL, POC_DISPLAY_NAME,
        "POC model. Actual model mapping is unverified; the response alias is not proof of model identity. "
        "Direct answers.safety.probabilities.unsafe, not choice confidence or GLiDE Noul.",
        allow_response_alias=True,
    )


def attach_gliner25_run(report, run_dir, data_dir=Path("data/prepared")):
    _attach_completed_classification_run(
        report, run_dir, data_dir, GLINER25_PROVIDER, GLINER25_MODEL, GLINER25_DISPLAY_NAME,
        "Direct unsafe label confidence from independent safe/unsafe scores, not GLiDE Noul. "
        "Both labels were requested at API threshold zero; no normalization or complement. "
        "Inputs were not truncated; overlength inputs are recorded as errors.",
    )


def _attach_completed_classification_run(
    report, run_dir, data_dir, provider, model, display_name, notes,
    *, allow_response_alias=False,
):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if (
        status["provider"] != provider or status["model"] != model
        or status["status"] not in ("completed", "completed_with_errors")
        or status["prompt_version"] != "v1" or status["threshold"] != 0.5
        or status["pii_excluded"] is not True
        or list(status["datasets"]) != list(DATASETS)[:6]
    ):
        raise ValueError(f"{display_name} comparison requires a completed six-dataset v1 / 0.5 run.")
    records, response_models, http_errors = [], {}, {}
    for dataset, details in status["datasets"].items():
        reference_path = data_dir / DATASETS[dataset]
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != details["sha256"]:
            raise ValueError(f"{dataset}: {display_name} reference fingerprint mismatch.")
        reference = read_reference(reference_path)
        path = run_dir / provider / f"{dataset}_v1_t0.5.csv"
        frame, count, errors = read_predictions(
            path, reference, provider, dataset, "v1", 0.5,
        )
        if (
            count != len(reference) or count != details["rows"]
            or details["processed"] != count or details["valid"] != len(frame)
            or details["errors"] != count - len(frame)
        ):
            raise ValueError(f"{dataset}: {display_name} result counts do not match completed status.")
        latencies = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["requested_model"] != model:
                    raise ValueError(f"{dataset}: {display_name} requested model mismatch.")
                if row["error"].strip():
                    if row["http_status"]:
                        http_errors[row["http_status"]] = http_errors.get(row["http_status"], 0) + 1
                    continue
                response_model = row["response_model"]
                latency = float(row["latency_ms"])
                if (
                    not response_model or (not allow_response_alias and response_model != model)
                    or row["http_status"] != "200" or not math.isfinite(latency) or latency < 0
                ):
                    raise ValueError(f"{dataset}: invalid {display_name} response identity or latency.")
                response_models[response_model] = response_models.get(response_model, 0) + 1
                latencies.append(latency)
        records.append({
            "provider": provider, "model": model,
            "display_name": display_name, "dataset": dataset,
            "prompt": "v1", "threshold": 0.5, "file": str(path),
            "status": "complete", "expected": len(reference), "rows": count,
            "valid": len(frame), "errors": count - len(frame),
            "error_categories": errors, "source_kind": "classification_predictions",
            "mean_elapsed_ms": mean(latencies) if latencies else None,
            "latency_source_kind": "successful_request_latency_ms",
            **classification_metrics(frame.values()),
        })
    counts = tuple(sum(row[key] for row in records) for key in ("rows", "valid", "errors"))
    if (
        counts != (status["processed_rows"], status["valid_rows"], status["error_rows"])
        or response_models != status["response_models"]
    ):
        raise ValueError(f"{display_name} aggregate counts or response models disagree with completed status.")
    comparison = report["external_comparison"]
    comparison["runs"] = [
        row for row in comparison["runs"] if row["provider"] != provider
    ] + records
    if provider not in comparison["providers"]:
        comparison["providers"].insert(comparison["providers"].index("fastino.ai") + 1, provider)
    response_model = ", ".join(sorted(response_models)) or "unavailable"
    comparison["model_identities"][provider] = {
        "request_model": model, "resolved_model": None if allow_response_alias else model,
        "response_model": response_model, "display_name": display_name,
    }
    comparison.setdefault("display_names", {})[provider] = display_name
    failure_note = " ".join(
        f"HTTP {code}: {count} errors." for code, count in sorted(http_errors.items())
    )
    comparison.setdefault("classification_models", {})[provider] = {
        "run_dir": str(run_dir), "log_file": str(run_dir / "run.log"),
        "model_id": model, "display_name": display_name,
        "response_model": response_model, "mapping_unverified": allow_response_alias,
        "http_error_counts": http_errors,
        "rows": counts[0], "valid": counts[1], "errors": counts[2],
        "finished_at": status["finished_at"],
        "notes": (
            notes + f" Requested model: {model}; response model: {response_model}. "
            "v1 / 0.5; six non-PII datasets. Quality excludes failed requests. "
            + failure_note + " "
            "Latency is the successful-request mean from CSV latency_ms, including per-request "
            "retry waits and client/network overhead, but excluding initial preflight warmup; "
            "it differs from GLiDE/GLiGuard log elapsed/processed timing. "
            "PII, v2 and repeat consistency were not tested. Not a sample-aligned ranking."
        ),
    }
    comparison["overview"] = reference_comparison_averages(comparison)


def _attach_fastino_run(report, run_dir, log_path, data_dir, metadata, model_group, display_name):
    provider = metadata["provider"]
    if (
        metadata["prompt_version"] != "v1" or metadata["decision_threshold"] != 0.5
        or metadata["datasets"] != list(DATASETS)[:6]
    ):
        raise ValueError("Fastino comparison requires the first six datasets at v1 / 0.5.")
    tasks, timings = {}, {}
    text = log_path.read_text(encoding="utf-8")
    if model_group == "fine_tuned_models" and f"MODEL_ID={metadata['model']}" not in text:
        raise ValueError("Fine-tuned log does not identify the requested model ID.")
    for match in START.finditer(text):
        name, dataset, prompt, threshold, total = match.groups()
        if name != provider or prompt != "v1" or float(threshold) != 0.5 or dataset in tasks:
            raise ValueError("Fastino log must contain one unambiguous v1 / 0.5 task per dataset.")
        tasks[dataset] = int(total)
    for match in PROGRESS.finditer(text):
        count, total, name, dataset, prompt, elapsed = match.groups()
        if name != provider or prompt != "v1" or dataset not in tasks:
            raise ValueError("Fastino progress does not match the recorded tasks.")
        current = int(count), int(total), float(elapsed)
        previous = timings.get(dataset)
        if (
            current[1] != tasks[dataset] or not 0 < current[0] <= current[1]
            or not math.isfinite(current[2]) or current[2] < 0
            or (previous and (current[0] < previous[0] or current[2] < previous[2]))
        ):
            raise ValueError("Invalid Fastino log progress.")
        timings[dataset] = current
    if set(tasks) != set(metadata["datasets"]):
        raise ValueError("Fastino log does not cover the expected datasets.")
    records = []
    for dataset in metadata["datasets"]:
        reference_path = data_dir / DATASETS[dataset]
        expected_hash = metadata["input_sha256"][str(Path("data/prepared") / DATASETS[dataset])]
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"{dataset}: Fastino run reference fingerprint mismatch.")
        reference = read_reference(reference_path)
        path = run_dir / provider / f"{dataset}_v1_t0.5.csv"
        frame, count, errors = read_predictions(path, reference, provider, dataset, "v1", 0.5)
        timing = timings.get(dataset)
        if not timing or timing[0] != timing[1] or timing[0] != count or count != len(reference):
            raise ValueError(f"{dataset}: Fastino result and complete log counts must match.")
        records.append({
            "provider": provider, "model": metadata["model"], "dataset": dataset,
            "display_name": display_name,
            "prompt": "v1", "threshold": 0.5, "file": str(path),
            "status": "complete", "expected": len(reference), "rows": count,
            "valid": len(frame), "errors": count - len(frame),
            "error_categories": errors,
            "source_kind": "fine_tuned_predictions" if model_group == "fine_tuned_models" else "classification_predictions",
            "mean_elapsed_ms": timing[2] * 1000 / count,
            "latency_source_kind": "single_complete_run", "latency_log_file": str(log_path),
            **classification_metrics(frame.values()),
        })
    counts = tuple(sum(row[key] for row in records) for key in ("rows", "valid", "errors"))
    if "counts" in metadata and counts != metadata["counts"]:
        raise ValueError("Fastino result counts do not match the completed run status.")
    comparison = report["external_comparison"]
    comparison["runs"] = [
        row for row in comparison["runs"] if row["provider"] != provider
    ] + records
    if provider not in comparison["providers"]:
        comparison["providers"].insert(comparison["providers"].index("fastino.ai") + 1, provider)
    comparison["model_identities"][provider] = {
        "request_model": metadata["model"], "resolved_model": metadata["model"],
        "display_name": display_name,
    }
    comparison.setdefault("display_names", {})[provider] = display_name
    model_info = {
        "run_dir": str(run_dir), "log_file": str(log_path), "model_id": metadata["model"],
        "display_name": display_name,
        "rows": counts[0], "valid": counts[1], "errors": counts[2],
        "notes": (
            "v1 / 0.5; six datasets only. Direct unsafe confidence from independent safe/unsafe scores, "
            "not GLiDE Noul. Quality excludes failed requests; no input truncation. "
            "Single-run elapsed includes original failure waits but excludes post-run retries. "
            "PII, v2 and repeat consistency were not tested. Training/evaluation overlap was not verified. "
            "Not a sample-aligned ranking."
        ),
    }
    if model_group == "fine_tuned_models":
        comparison["model_identities"][provider]["base_model"] = metadata["base_model"]
        model_info["training_name"] = metadata["display_name"]
    else:
        model_info["finished_at"] = metadata["finished_at"]
        model_info["notes"] = (
            "v1 / 0.5; six datasets only. Direct unsafe confidence from independent safe/unsafe scores, "
            "not GLiDE Noul. Quality excludes failed requests; truncated inputs are recorded as errors. "
            "Single-run elapsed includes failed requests, client and network overhead. "
            "PII, v2 and repeat consistency were not tested. Not a sample-aligned ranking."
        )
    comparison.setdefault(model_group, {})[provider] = model_info
    comparison["overview"] = reference_comparison_averages(comparison)


def attach_openjev_summary(report, path, internal_summary_path=None):
    source_rows = read_openjev_summary(path)
    if internal_summary_path is None:
        sibling = path.parent / "internal_model_summary.csv"
        internal_summary_path = sibling if sibling.exists() else None
    internal_rows = read_internal_summary(internal_summary_path) if internal_summary_path is not None else []
    if "openjev" in report["providers"]:
        raise ValueError("OpenJev aggregate results must not replace raw OpenJev predictions")
    records = []
    aggregate_timings = {
        (row["provider"], row["dataset"], row["prompt"]): row
        for row in report.get("repeat_log_timings", [])
    }
    for row in report["runs"]:
        if row["threshold"] != 0.5:
            continue
        repeats = [
            other for other in report["runs"]
            if other["provider"] == row["provider"] and other["dataset"] == row["dataset"]
            and other["prompt"] == row["prompt"]
        ]
        timed = len(repeats) == len(THRESHOLDS) and all(
            other["status"] == "complete" and other["mean_elapsed_ms"] is not None
            for other in repeats
        )
        aggregate = aggregate_timings.get((row["provider"], row["dataset"], row["prompt"]))
        aggregate_timed = (
            aggregate is not None and len(repeats) == len(THRESHOLDS)
            and all(
                other["status"] == "complete" and other["rows"] == other["expected"] == aggregate["rows_per_run"]
                for other in repeats
            )
        )
        single_timed = (
            not timed and not aggregate_timed and row["prompt"] == "v1"
            and row["status"] == "complete" and row.get("log_file")
            and row.get("log_processed") == row.get("log_total") == row["rows"] == row["expected"]
            and row["rows"] > 0 and row.get("mean_elapsed_ms") is not None
            and math.isfinite(row["mean_elapsed_ms"]) and row["mean_elapsed_ms"] >= 0
        )
        records.append({
            **row, "source_kind": "local_predictions",
            "mean_elapsed_ms": (
                mean(other["mean_elapsed_ms"] for other in repeats) if timed
                else aggregate["mean_elapsed_ms"] if aggregate_timed
                else row["mean_elapsed_ms"] if single_timed else None
            ),
            **({
                "latency_source_kind": "four_run_log_aggregate",
                "latency_log_file": aggregate["log_file"],
            } if aggregate_timed and not timed else {
                "latency_source_kind": "single_complete_run",
                "latency_log_file": row["log_file"],
            } if single_timed else {}),
        })
    records.extend(source_rows)
    records.extend(internal_rows)
    internal_providers = [provider for provider in INTERNAL_IDENTITIES if any(
        row["provider"] == provider for row in internal_rows
    )]
    providers = [*report["providers"], "openjev", *internal_providers]
    overview = reference_averages(records, providers)
    report["external_comparison"] = {
        "source": str(path), "source_kind": "user_supplied_aggregate", "notes": NOTES,
        **({
            "internal_source": str(internal_summary_path), "internal_source_rows": internal_rows,
            "notes": NOTES + INTERNAL_NOTES,
        } if internal_rows else {}),
        "classification_threshold": 0.5,
        "latency_method": (
            "per_model_four_run_mean_or_single_complete_run"
            if any(row.get("latency_source_kind") == "single_complete_run" for row in records)
            else "mean_elapsed_per_processed_over_four_complete_runs"
        ),
        "repeat_thresholds": list(THRESHOLDS),
        "providers": providers,
        "model_identities": {
            **report["model_identities"], "openjev": OPENJEV_IDENTITY,
            **{provider: INTERNAL_IDENTITIES[provider] for provider in internal_providers},
        },
        "display_names": {
            "openjev": "OpenJev 4B",
            **{provider: INTERNAL_IDENTITIES[provider]["display_name"] for provider in internal_providers},
        },
        "source_rows": source_rows, "runs": records, "overview": overview,
    }
