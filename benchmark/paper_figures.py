"""Plots of saved comparison metrics, without new inference."""

import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from benchmark.comparison import DATASETS, THRESHOLDS, baseline_rows


COLORS = {"fastino.ai": "#0072B2", "laya-ai.com": "#D55E00", "typesafe.ai": "#009E73"}
NAMES = {"fastino.ai": "Fastino", "laya-ai.com": "Laya", "typesafe.ai": "TypeSafe"}
MARKERS = {"fastino.ai": "o", "laya-ai.com": "s", "typesafe.ai": "D"}
HATCHES = {"fastino.ai": "", "laya-ai.com": "///", "typesafe.ai": "..."}
DATASET_NAMES = {
    "aegis2": "AEGIS2",
    "toxigen": "ToxiGen",
    "hatecheck": "HateCheck",
    "detect-jailbreak": "Detect-Jailbreak",
    "jailbreak-classification": "Jailbreak-Classification",
    "jailbreakhub": "JailbreakHub",
    "pii_masking_nano_1k": "PII Nano-1K",
    "pii_masking_mini_10k": "PII Mini-10K",
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
}


def model_label(report, provider):
    identity = report["model_identities"][provider]
    lines = [f"{NAMES[provider]} | request: {identity['request_model']}"]
    for role in ("resolved_model", "serving_model", "routed_model"):
        if role in identity:
            value = identity[role] or "unconfirmed (API response required)"
            lines.append(f"{role.removesuffix('_model')}: {value}")
    return "\n".join(lines)


def decorate(fig, report, title, subtitle, note, legend_style="markers"):
    fig.suptitle(title, fontsize=15, y=0.98)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=9)
    fig.text(0.07, 0.19, note, fontsize=8, va="top")
    handles = []
    for provider in report["providers"]:
        label = model_label(report, provider)
        if legend_style == "bars":
            handle = Patch(facecolor=COLORS[provider], edgecolor="#263238",
                           hatch=HATCHES[provider], linewidth=0.4, label=label)
        else:
            handle = Line2D(
                [], [], color=COLORS[provider],
                marker=MARKERS[provider] if legend_style == "markers" else None,
                linewidth=0 if legend_style == "identity" else 1.8, label=label,
            )
        handles.append(handle)
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=len(handles), frameon=False, handlelength=2.2, columnspacing=2,
    )


def style_axis(ax, grid="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#D8DDE3", linewidth=0.6, linestyle="--", alpha=0.8)


def quality_figure(report):
    rows = baseline_rows(report)
    datasets = report["baseline_datasets"]
    lookup = {(row["provider"], row["dataset"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.33, wspace=0.22)
    width = 0.76 / len(report["providers"])
    x = np.arange(len(datasets))
    for ax, metric, panel in zip(axes, ("f1", "auroc"), ("(a)", "(b)")):
        for index, provider in enumerate(report["providers"]):
            values = [
                lookup.get((provider, dataset), {}).get(metric) for dataset in datasets
            ]
            values = [value * 100 if value is not None else np.nan for value in values]
            positions = x + (index - (len(report["providers"]) - 1) / 2) * width
            bars = ax.bar(
                positions, values, width=width * 0.92, color=COLORS[provider],
                edgecolor="#263238", linewidth=0.4, hatch=HATCHES[provider],
            )
            for bar, value in zip(bars, values):
                if math.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.1f}",
                            ha="center", fontsize=7.5)
        labels = []
        for dataset in datasets:
            count = lookup.get((report["providers"][0], dataset), {}).get("n", 0)
            labels.append(f"{DATASET_NAMES[dataset]}\n(n = {count:,})")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("F1 (%)" if metric == "f1" else "AUROC (%)")
        ax.set_title(f"{panel} " + ("Detection F1" if metric == "f1" else "Score discrimination (AUROC)"))
        if metric == "auroc":
            ax.axhline(50, color="#666666", linewidth=0.8, linestyle=":")
        style_axis(ax)
    decorate(
        fig, report, "Safety classification: paired baseline",
        "v1 | threshold = 0.5 | common completed non-PII datasets | higher is better",
        "Each dataset uses the same successful sample IDs for all selected models.\n"
        "Values are descriptive point estimates; repeated sweeps are not independent replicates.",
        legend_style="bars",
    )
    return fig


def tradeoff_figure(report):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.32, wspace=0.28)
    for row in report["overview"]:
        provider = row["provider"]
        points = (
            (row["macro_fpr"], row["macro_fnr"]),
            (row["baseline_elapsed_ms"], row["macro_auroc"]),
        )
        for index, (ax, (x, y)) in enumerate(zip(axes, points)):
            if x is None or y is None:
                continue
            x = x * 100 if index == 0 else x
            y *= 100
            ax.scatter(x, y, color=COLORS[provider], marker=MARKERS[provider],
                       s=95, edgecolors="#263238", linewidths=0.6, zorder=3, clip_on=False)
            offset = (-8, 11) if index == 0 and x > 80 else (8, 11)
            ax.annotate(
                NAMES[provider], (x, y), xytext=offset, textcoords="offset points",
                ha="right" if offset[0] < 0 else "left", fontsize=9,
            )
    axes[0].set(
        xlabel="Macro false-positive rate (%)", ylabel="Macro false-negative rate (%)",
        title="(a) Error trade-off (toward bottom-left is better)", xlim=(-4, 110), ylim=(-4, 110),
    )
    times = [row["baseline_elapsed_ms"] for row in report["overview"]
             if row["baseline_elapsed_ms"] is not None]
    axes[1].set(
        xlabel="Amortized end-to-end elapsed time (ms / sample)", ylabel="Macro AUROC (%)",
        title="(b) Quality vs. elapsed time (toward top-left is better)",
        xlim=(0, max(times, default=1) * 1.28), ylim=(0, 110),
    )
    axes[1].set_yticks(range(0, 101, 20))
    for ax in axes:
        style_axis(ax, "both")
    decorate(
        fig, report, "Baseline quality, errors and elapsed time",
        f"v1 | threshold = 0.5 | {len(report['baseline_datasets'])} shared datasets | dataset-macro quality",
        "Elapsed time = total logged elapsed / all processed baseline rows, including failed requests.\n"
        "Runs occurred at different times; this is not pure inference latency or a request-level percentile.",
    )
    return fig


def threshold_figure(report):
    datasets = report["baseline_datasets"]
    columns = min(2, max(1, len(datasets)))
    rows = max(1, math.ceil(len(datasets) / columns))
    fig, grid = plt.subplots(rows, columns, figsize=(13.6, 3.2 * rows + 2.2), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.86, bottom=0.28, hspace=0.44, wspace=0.22)
    lookup = {
        (row["provider"], row["dataset"], row["prompt"], row["threshold"]): row
        for row in report["paired"]
    }
    for index, ax in enumerate(grid.flat):
        if index >= len(datasets):
            ax.set_visible(False)
            continue
        dataset = datasets[index]
        for provider in report["providers"]:
            for prompt, linestyle, marker in (("v1", "-", "o"), ("v2", "--", "^")):
                values = []
                for threshold in THRESHOLDS:
                    row = lookup.get((provider, dataset, prompt, threshold))
                    value = row["f1"] if row and row["n"] else None
                    values.append(value * 100 if value is not None else np.nan)
                ax.plot(THRESHOLDS, values, color=COLORS[provider], linestyle=linestyle,
                        marker=marker, markersize=4.5, linewidth=1.6)
        ax.set(
            title=f"({chr(97 + index)}) {DATASET_NAMES[dataset]}",
            xlabel="Decision threshold", ylabel="F1 (%)", ylim=(-3, 105),
        )
        ax.set_xticks(THRESHOLDS)
        style_axis(ax)
    grid.flat[0].legend(
        handles=[
            Line2D([], [], color="#333333", marker="o", linestyle="-", label="Prompt v1"),
            Line2D([], [], color="#333333", marker="^", linestyle="--", label="Prompt v2"),
        ],
        loc="lower left", frameon=True, framealpha=0.95,
    )
    decorate(
        fig, report, "Sensitivity to prompt version and decision threshold",
        "Exploratory test-set sweeps, not held-out model selection",
        "Points use within-configuration paired successful samples; error-driven sample sets can differ across points.\n"
        "Missing configurations are gaps (not zero scores); lines do not imply independent replicates or confidence intervals.",
        legend_style="lines",
    )
    return fig


def coverage_figure(report):
    datasets = list(DATASETS)
    lookup = {
        (row["provider"], row["dataset"]): row for row in report["runs"]
        if row["prompt"] == "v1" and row["threshold"] == 0.5
    }
    values = np.full((len(datasets), len(report["providers"])), np.nan)
    for i, dataset in enumerate(datasets):
        for j, provider in enumerate(report["providers"]):
            row = lookup.get((provider, dataset))
            if row and row["rows"]:
                values[i, j] = row["coverage"] * 100
    fig, ax = plt.subplots(figsize=(13.6, 8.5))
    fig.subplots_adjust(left=0.235, right=0.92, top=0.855, bottom=0.28)
    cmap = matplotlib.colormaps["cividis"].with_extremes(bad="#ECEFF3")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, dataset in enumerate(datasets):
        for j, provider in enumerate(report["providers"]):
            row = lookup.get((provider, dataset))
            if row and row["rows"]:
                value = values[i, j]
                label = f"{value:.2f}%\n{row['valid']:,} / {row['expected']:,}"
                if row["status"] == "complete" and not row["valid"]:
                    label = f"ALL FAILED\n0 / {row['expected']:,}"
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#D55E00", hatch="///", linewidth=1.2))
                elif row["status"] == "partial":
                    label += "\npartial CSV"
                color = "white" if value < 45 else "#172438"
            else:
                label = "NO CSV" if row and row["status"] == "in_progress" else "NOT AVAILABLE"
                color = "#566170"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(report["providers"])), [NAMES[p] for p in report["providers"]])
    ax.set_yticks(range(len(datasets)), [DATASET_NAMES[d] for d in datasets])
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    colorbar.set_label("Successful rows / reference rows (%)")
    decorate(
        fig, report, "Baseline evaluation coverage and request failures",
        "v1 | threshold = 0.5 | counts are successful / reference rows",
        "Gray = no saved predictions; hatched = complete CSV with no successful requests, not a missing run.\n"
        "PII rows measure only evaluation coverage, not masking quality. Laya's suite is incomplete.",
        legend_style="identity",
    )
    return fig


FIGURES = (
    (
        "fig01_paired_quality", quality_figure, "共同样本分类能力对比",
        "固定 v1 / 0.5，按数据集展示 F1 与 AUROC。每个数据集内，所有模型使用完全相同的成功样本。"
        "AUROC 图中的 50% 虚线表示随机排序基线；没有伪造置信区间。",
    ),
    (
        "fig02_quality_cost_tradeoff", tradeoff_figure, "误报、漏报与耗时权衡",
        "左图越靠左下越好，右图越靠左上越好。质量指标按共同数据集等权平均；"
        "耗时来自整轮日志，按全部处理样本加权，含网络和失败等待，不能解读为纯推理时延或 P95。",
    ),
    (
        "fig03_threshold_sensitivity", threshold_figure, "提示词与阈值敏感性",
        "颜色区分模型，实线圆点表示 v1，虚线三角表示 v2。缺测留空，不补零。"
        "每个点只在其配置内使用共同成功样本，不同点的样本集合可能随请求错误变化；"
        "这是同测试集探索，不是独立验证后的性能提升。",
    ),
    (
        "fig04_evaluation_coverage", coverage_figure, "测试覆盖率与请求失败",
        "覆盖率为成功样本数 / 参考样本数。灰色为未获得预测文件，斜线为完整落盘但全部请求失败；"
        "PII 行仅展示覆盖率，不代表脱敏质量。Laya 缺测不记零分。",
    ),
)


def latex_escape(text):
    escapes = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(char, char) for char in text)


def write_latex_table(report, path):
    lines = [
        r"% Requires \usepackage{booktabs}. Identity metadata supplied by the user.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Model & Macro F1 (\%) & Macro AUROC (\%) & Macro FPR (\%) & Macro FNR (\%) & Elapsed (ms) \\",
        r"\midrule",
    ]
    for row in report["overview"]:
        label = r"\shortstack[l]{" + r"\\".join(
            latex_escape(line) for line in model_label(report, row["provider"]).splitlines()
        ) + "}"
        values = []
        for key in ("macro_f1", "macro_auroc", "macro_fpr", "macro_fnr", "baseline_elapsed_ms"):
            value = row[key]
            if value is None:
                values.append("--")
            else:
                values.append(f"{value * (1 if key == 'baseline_elapsed_ms' else 100):.2f}")
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Paired baseline on " + str(len(report["baseline_datasets"]))
        + r" common completed non-PII datasets (prompt v1, threshold 0.5). "
        r"Quality metrics are dataset-macro averages over common successful sample IDs. "
        r"Elapsed time is amortized end-to-end log time per processed sample, not pure inference latency. "
        r"TypeSafe's resolved model remains unconfirmed until an API response is available.}",
        r"\label{tab:model-comparison}", r"\end{table*}", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_figures(report, output_dir, dpi=300):
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    if not report["baseline_datasets"] or not baseline_rows(report):
        raise ValueError("No shared valid baseline datasets; cannot draw a fair quality comparison")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_generated_at": report["generated_at"],
        "model_identities": report["model_identities"],
        "baseline_datasets": report["baseline_datasets"],
        "dpi": dpi, "figures": [],
    }
    with plt.rc_context(STYLE), PdfPages(output_dir / "paper_figures.pdf") as bundle:
        for stem, builder, title, caption in FIGURES:
            fig = builder(report)
            try:
                for extension in ("png", "svg", "pdf"):
                    fig.savefig(output_dir / f"{stem}.{extension}", dpi=dpi)
                bundle.savefig(fig)
            finally:
                plt.close(fig)
            manifest["figures"].append({
                "name": stem, "title": title, "caption": caption,
                "files": {ext: f"{stem}.{ext}" for ext in ("png", "svg", "pdf")},
            })
    write_latex_table(report, output_dir / "table01_baseline.tex")
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    cards = []
    for figure in manifest["figures"]:
        downloads = " | ".join(
            f'<a href="{filename}">{extension.upper()}</a>'
            for extension, filename in figure["files"].items()
        )
        cards.append(
            "<section><h2>" + html.escape(figure["title"]) + "</h2>"
            + f'<img src="{figure["files"]["svg"]}" alt="{html.escape(figure["title"])}">'
            + "<p>" + html.escape(figure["caption"]) + "</p><p>" + downloads + "</p></section>"
        )
    gallery = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>模型对比图</title><style>
body{font:16px/1.7 system-ui,sans-serif;background:#f3f5f8;color:#172438;margin:0}
main{max-width:1250px;margin:auto;padding:28px}section{background:white;padding:24px;margin:24px 0}
img{width:100%;height:auto}a{color:#174c90}h1,h2{line-height:1.4}
@media print{body{background:white}section{break-inside:avoid;padding:0}}
</style><main><h1>模型对比图</h1>
<p>图内使用英文标签，图注使用中文。PDF / SVG 为矢量格式，PNG 默认为 300 DPI。
模型身份由用户补充；TypeSafe resolved model 需以接口返回为准。</p>
<p><a href="paper_figures.pdf">下载全部图表 PDF</a> |
<a href="table01_baseline.tex">下载 LaTeX 三线表</a> |
<a href="../comparison.html">完整指标与结论报告（默认目录）</a></p>"""
    gallery += "<p>数据快照（UTC）：" + html.escape(report["generated_at"]) + "</p>"
    gallery += "".join(cards) + "</main></html>\n"
    (output_dir / "index.html").write_text(gallery, encoding="utf-8")
    return manifest
