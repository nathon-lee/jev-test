"""Reference-only comparisons with user-supplied aggregate results."""

import csv
import math
from statistics import mean

from benchmark.comparison import DATASETS, PROMPTS, THRESHOLDS


METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms")
OPENJEV_IDENTITY = {"request_model": "not supplied", "resolved_model": None}
NOTES = (
    "OpenJev 数值取自用户最初提供的清晰汇总表，分类阈值均为 0.5；"
    "补充 OCR 文本有错字，不据此修正数值或推断混淆矩阵。Rows 为总测试行数，"
    "有效样本数和错误数未可靠录入，保持未知，不将 Rows 当作有效样本数。"
    "OpenJev 只提供汇总指标，未提供逐样本 score、预测或模型版本；"
    "不能核验相同输入、共同成功样本或计算重复调用稳定性，不加入原有公平基线与排名。"
    "参考图按各模型自身成功样本统计，仅作描述性对照。"
    "耗时统一为每个数据集/提示词四轮完整运行的 elapsed/processed 算术平均，"
    "包含失败请求、等待、客户端及网络开销，不是纯推理时间；无完整四轮计时标 N/A。"
)


def read_openjev_summary(path):
    records, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "prompt", "threshold", "rows", *METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing summary columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            dataset, prompt = row["dataset"], row["prompt"]
            if dataset not in DATASETS or prompt not in PROMPTS:
                raise ValueError(f"{path}:{line}: unknown dataset/prompt")
            key = dataset, prompt
            if key in seen:
                raise ValueError(f"{path}:{line}: duplicate summary configuration: {key}")
            seen.add(key)
            threshold, rows = float(row["threshold"]), int(row["rows"])
            if threshold != 0.5 or rows <= 0:
                raise ValueError(f"{path}:{line}: summaries require threshold 0.5 and positive rows")
            metrics = {}
            for metric in METRICS:
                if metric == "auroc" and row[metric] == "":
                    metrics[metric] = None
                    continue
                value = float(row[metric])
                if not math.isfinite(value) or value < 0 or (metric != "mean_elapsed_ms" and value > 1):
                    raise ValueError(f"{path}:{line}: invalid {metric}")
                metrics[metric] = value
            records.append({
                "provider": "openjev", "dataset": dataset, "prompt": prompt,
                "threshold": threshold, "rows": rows, "valid": None, "errors": None,
                "source_kind": "user_supplied_aggregate", **metrics,
            })
    if not records:
        raise ValueError(f"{path}: empty summary")
    return records


def attach_openjev_summary(report, path):
    source_rows = read_openjev_summary(path)
    if "openjev" in report["providers"]:
        raise ValueError("OpenJev aggregate results must not replace raw OpenJev predictions")
    records = []
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
        records.append({
            **row, "source_kind": "local_predictions",
            "mean_elapsed_ms": mean(other["mean_elapsed_ms"] for other in repeats) if timed else None,
        })
    records.extend(source_rows)
    providers = [*report["providers"], "openjev"]
    lookup = {(row["provider"], row["dataset"], row["prompt"]): row for row in records}
    overview = []
    for prompt in PROMPTS:
        datasets = [
            dataset for dataset in DATASETS if not dataset.startswith("pii_")
            and all(
                (row := lookup.get((provider, dataset, prompt))) is not None
                and (provider == "openjev" or (row["status"] == "complete" and row["valid"] > 0))
                for provider in providers
            )
        ]
        for provider in providers:
            rows = [lookup[(provider, dataset, prompt)] for dataset in datasets]
            metrics = {}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                metrics[metric] = mean(values) if values and all(value is not None for value in values) else None
            overview.append({
                "provider": provider, "prompt": prompt, "datasets": datasets, **metrics,
            })
    report["external_comparison"] = {
        "source": str(path), "source_kind": "user_supplied_aggregate", "notes": NOTES,
        "classification_threshold": 0.5,
        "latency_method": "mean_elapsed_per_processed_over_four_complete_runs",
        "repeat_thresholds": list(THRESHOLDS),
        "providers": providers,
        "model_identities": {**report["model_identities"], "openjev": OPENJEV_IDENTITY},
        "source_rows": source_rows, "runs": records, "overview": overview,
    }
