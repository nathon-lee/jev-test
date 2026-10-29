"""Reference-only comparisons with user-supplied aggregate results."""

import csv
import math
from statistics import mean

from benchmark.comparison import DATASETS, PROMPTS, THRESHOLDS


METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms")
FOCUSED_DATASETS = ("toxigen", "hatecheck", "aegis2")
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


def reference_averages(records, providers, datasets=None, include_pii=False):
    candidates = list(DATASETS if datasets is None else datasets)
    if datasets is not None and (
        not candidates or len(set(candidates)) != len(candidates)
        or any(dataset not in DATASETS or (not include_pii and dataset.startswith("pii_"))
               for dataset in candidates)
    ):
        raise ValueError("Reference averages require distinct " + ("" if include_pii else "non-PII ") + "datasets")
    lookup = {(row["provider"], row["dataset"], row["prompt"]): row for row in records}
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
            overview.append({
                "provider": provider, "prompt": prompt, "datasets": selected, **metrics,
            })
    return overview


def v1_reference_scope_averages(report, datasets=None):
    comparison = report["external_comparison"]
    raw = {
        (row["provider"], row["dataset"]): row for row in report["runs"]
        if row["prompt"] == "v1" and row["threshold"] == 0.5
    }
    timing = {}
    for row in comparison["runs"]:
        if row["prompt"] != "v1":
            continue
        key = (row["provider"], row["dataset"])
        if row["provider"] == "openjev" or row["provider"] in INTERNAL_IDENTITIES:
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
            row for row in reference_averages(
                comparison["runs"], comparison["providers"], selected, include_pii=include_pii,
            ) if row["prompt"] == "v1"
        ]
        scope_datasets = rows[0]["datasets"] if rows else []
        timed = [
            dataset for dataset in scope_datasets
            if all(timing.get((row["provider"], dataset)) is not None for row in rows)
        ]
        for row in rows:
            own = [timing.get((row["provider"], dataset)) for dataset in scope_datasets]
            result.append({
                "scope": scope, **row, "threshold": 0.5, "timing_datasets": timed,
                "mean_elapsed_ms": mean(own) if own and all(value is not None for value in own) else None,
                "shared_elapsed_ms": (
                    mean(timing[(row["provider"], dataset)] for dataset in timed) if timed else None
                ),
            })
    return result


def three_dataset_averages(report):
    comparison = report["external_comparison"]
    overview = [
        row for row in reference_averages(comparison["runs"], comparison["providers"], FOCUSED_DATASETS)
        if row["prompt"] == "v1"
    ]
    timing = {
        (row["provider"], row["dataset"]): row
        for row in report["runs"] if row["prompt"] == "v1" and row["threshold"] == 0.5
    }
    for row in overview:
        if row["provider"] == "openjev" or row["provider"] in INTERNAL_IDENTITIES:
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
            "f1": None, "auroc": auroc, "mean_elapsed_ms": latency,
        })
    return overview


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
