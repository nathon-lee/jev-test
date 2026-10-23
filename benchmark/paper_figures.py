"""Plots of saved comparison metrics, without new inference."""

import base64
import csv
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

from benchmark.comparison import DATASETS, PROMPTS, STABILITY_METRICS, THRESHOLDS, baseline_rows
from benchmark.external_summary import reference_averages, three_dataset_averages


COLORS = {"fastino.ai": "#0072B2", "laya-ai.com": "#D55E00", "typesafe.ai": "#009E73", "openjev": "#CC79A7"}
NAMES = {"fastino.ai": "Fastino", "laya-ai.com": "Laya", "typesafe.ai": "TypeSafe", "openjev": "OpenJev"}
MARKERS = {"fastino.ai": "o", "laya-ai.com": "s", "typesafe.ai": "D", "openjev": "^"}
HATCHES = {"fastino.ai": "", "laya-ai.com": "///", "typesafe.ai": "...", "openjev": "xxx"}
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
    if provider == "openjev":
        return "OpenJev | supplied internal summary\nmodel / version: not supplied (not sample-paired)"
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
        "Missing configurations are gaps, not zero scores. Curves mix threshold effects and repeat-call variation.\n"
        "Scores are not held fixed across thresholds; lines do not imply independent replicates or confidence intervals.",
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


def metric_bars(ax, report, labels, records, metric, title, latency=False,
                ylabel=None, autoscale=False, decimals=1, mark_zero=False):
    width = 0.76 / len(report["providers"])
    x = np.arange(len(labels))
    finite_values = []
    for index, provider in enumerate(report["providers"]):
        values = [record.get(provider, {}).get(metric) for record in records]
        values = [value * (1 if latency else 100) if value is not None else np.nan
                  for value in values]
        positions = x + (index - (len(report["providers"]) - 1) / 2) * width
        bars = ax.bar(
            positions, values, width=width * 0.92, color=COLORS[provider],
            edgecolor="#263238", linewidth=0.4, hatch=HATCHES[provider],
        )
        for bar, value in zip(bars, values):
            available = math.isfinite(value)
            if available:
                finite_values.append(value)
            if mark_zero and available and value == 0:
                ax.plot(
                    bar.get_x() + bar.get_width() / 2, 0, marker=MARKERS[provider],
                    color=COLORS[provider], markersize=4, linestyle="none",
                    clip_on=False, zorder=4,
                )
            ax.annotate(
                f"{value:.{decimals}f}" if available else "N/A",
                (bar.get_x() + bar.get_width() / 2, value if available else 0),
                xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=7, rotation=90 if len(labels) > 4 else 0,
                color="#172438" if available else "#566170",
            )
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel or ("ms / sample (lower is better)" if latency else "% (higher is better)"))
    ax.set_ylim(0, max(max(finite_values, default=0) * 1.3, 1) if latency or autoscale else 120)
    if not latency and not autoscale:
        ax.set_yticks(range(0, 101, 20))
    style_axis(ax)


def six_metric_panels(report, labels, records):
    fig, axes = plt.subplots(3, 2, figsize=(18.4, 13.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.29, hspace=0.7, wspace=0.18)
    for ax, metric, title in zip(
        axes.flat,
        ("accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms"),
        ("Accuracy", "Precision", "Recall", "F1", "AUROC", "AVG latency (ms)"),
    ):
        metric_bars(ax, report, labels, records, metric, title, latency=metric == "mean_elapsed_ms")
    return fig


def dataset_metrics_figure(report):
    datasets = list(DATASETS)
    lookup = {
        (row["provider"], row["dataset"]): row for row in report["runs"]
        if row["prompt"] == "v1" and row["threshold"] == 0.5
    }
    records = [
        {provider: lookup.get((provider, dataset), {}) for provider in report["providers"]}
        for dataset in datasets
    ]
    labels = [DATASET_NAMES[dataset].replace("-", "-\n") for dataset in datasets]
    fig = six_metric_panels(report, labels, records)
    decorate(
        fig, report, "Per-dataset classification metrics and average latency",
        "v1 | threshold = 0.5 | each model's own successful samples (not paired)",
        "N/A = missing result, no successful predictions, undefined metric, or no aligned complete timing log.\n"
        "PII datasets contain positives only: detection metrics do not measure masking quality; AUROC is undefined.\n"
        "Latency = logged end-to-end elapsed / all processed rows, including failed requests; not pure inference time.\n"
        "Successful sample sets can differ between models; use the paired baseline for fair quality comparisons.",
        legend_style="bars",
    )
    return fig


def model_averages_figure(report):
    records = [{row["provider"]: row for row in report["overview"]}]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.32, hspace=0.38, wspace=0.28)
    for ax, metric, title in zip(
        axes.flat,
        ("macro_accuracy", "macro_precision", "macro_recall", "macro_auroc", "macro_elapsed_ms"),
        ("Average Accuracy", "Avg Precision", "Avg Recall", "Avg AUROC", "Avg latency (ms)"),
    ):
        metric_bars(ax, report, ["Shared baseline"], records, metric, title,
                    latency=metric == "macro_elapsed_ms")
    axes.flat[-1].set_visible(False)
    decorate(
        fig, report, "Model averages on the shared baseline",
        f"v1 | threshold = 0.5 | {len(report['baseline_datasets'])} shared non-PII datasets | equal dataset weights",
        "Quality: common successful sample IDs within each dataset, then an unweighted mean across datasets.\n"
        "Latency: unweighted mean of per-dataset log elapsed / processed rows, including failures (not inference time).\n"
        "N/A timing means at least one baseline dataset has no aligned complete timing log; missing is never zero.",
        legend_style="bars",
    )
    return fig


def stability_figure(report, prompt):
    lookup = {
        (row["provider"], row["dataset"]): row for row in report["stability_paired"]
        if row["prompt"] == prompt
    }
    own = {
        (row["provider"], row["dataset"]): row for row in report["stability"]
        if row["prompt"] == prompt
    }
    datasets = [
        dataset for dataset in DATASETS
        if all(lookup.get((provider, dataset), {}).get("n", 0) > 0
               for provider in report["providers"])
    ]
    labels = [
        DATASET_NAMES[dataset].replace("-", "-\n")
        + f"\n(n = {lookup[(report['providers'][0], dataset)]['n']:,})"
        for dataset in datasets
    ]
    records = [
        {provider: {**own[(provider, dataset)], **lookup[(provider, dataset)]}
         for provider in report["providers"]}
        for dataset in datasets
    ]
    fig, axes = plt.subplots(3, 2, figsize=(18.4, 13.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.29, hspace=0.7, wspace=0.18)
    panels = (
        ("mean_score_std", "Mean score SD (population)", "Score units x100 (lower is better)", True),
        ("mean_score_range", "Mean score range", "Score units x100 (lower is better)", True),
        ("decision_flip_rate", "Decision flip rate (fixed threshold = 0.5)", "% (lower is better)", True),
        ("coverage", "All-model four-call success intersection / reference", "% (higher is better)", False),
        ("error_rate", "Request failures / saved requests", "% (lower is better)", False),
    )
    for ax, (metric, title, ylabel, autoscale) in zip(axes.flat, panels):
        metric_bars(ax, report, labels, records, metric, title,
                    ylabel=ylabel, autoscale=autoscale, decimals=2, mark_zero=True)
        if not datasets:
            ax.text(0.5, 0.5, "NO SHARED FOUR-CALL SAMPLES",
                    transform=ax.transAxes, ha="center", color="#566170")
    axes.flat[-1].set_visible(False)
    decorate(
        fig, report, f"Repeat-call consistency: prompt {prompt}",
        f"{len(datasets)} shared datasets | four calls per model | identical successful sample IDs across all models",
        "Score SD and range are displayed x100; flip rate uses score >= 0.5 for ALL four calls (not saved predictions).\n"
        "Only datasets with complete four-call CSVs for every model and nonempty common successful IDs are shown.\n"
        "Baseline markers show measured zeros, not missing data. Coverage uses all-model common IDs; failures use all saved requests.\n"
        "Historical backend / routing / time changes may contribute. Consistency is not accuracy; this is exploratory.",
        legend_style="bars",
    )
    return fig


def stability_averages_figure(report):
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.32, hspace=0.5, wspace=0.28)
    titles = ("Mean score SD", "Mean score range", "Decision flip rate")
    for prompt, row_axes in zip(PROMPTS, axes):
        overview = [row for row in report["stability_overview"] if row["prompt"] == prompt]
        count = len(overview[0]["datasets"]) if overview else 0
        records = [{row["provider"]: row for row in overview}]
        for ax, metric, title in zip(row_axes, STABILITY_METRICS, titles):
            score_metric = metric != "decision_flip_rate"
            metric_bars(
                ax, report, [f"{count} shared datasets"], records, metric, f"{prompt}: {title}",
                ylabel="Score units x100 (lower is better)" if score_metric else "% (lower is better)",
                autoscale=True, decimals=2, mark_zero=True,
            )
    decorate(
        fig, report, "Model repeat-call consistency: paired dataset-macro averages",
        "Prompt v1 and v2 are separate | common four-call successful IDs | shared non-PII datasets only",
        "Equal dataset weights. Score SD / range displayed x100; decision flips use the fixed threshold 0.5.\n"
        "Baseline markers show measured zeros. No eligible shared datasets = N/A; datasets and IDs may differ between prompts.\n"
        "See the report for shared sample counts, coverage and failure rates; low variation does not imply correctness.",
        legend_style="bars",
    )
    return fig


def external_dataset_figure(report, prompt):
    comparison = report["external_comparison"]
    lookup = {
        (row["provider"], row["dataset"]): row for row in comparison["runs"]
        if row["prompt"] == prompt
    }
    labels = [DATASET_NAMES[dataset].replace("-", "-\n") for dataset in DATASETS]
    records = [
        {provider: lookup.get((provider, dataset), {}) for provider in comparison["providers"]}
        for dataset in DATASETS
    ]
    fig = six_metric_panels(comparison, labels, records)
    decorate(
        fig, comparison, f"Four-model reference comparison: prompt {prompt}",
        "threshold = 0.5 | own successful samples | OpenJev: supplied aggregate, NOT sample-paired",
        "OpenJev values are transcribed from the supplied summary; valid/error counts and model version are unconfirmed.\n"
        "Latency for each model = mean of elapsed / processed over four complete runs (including failed requests).\n"
        "N/A = missing quality result, single-class AUROC, or incomplete four-run timing; not zero.\n"
        "Descriptive only, not a fair ranking. PII values measure presence detection, not masking quality.",
        legend_style="bars",
    )
    return fig


def external_averages_figure(report, datasets=None):
    comparison = report["external_comparison"]
    overview = comparison["overview"] if datasets is None else reference_averages(
        comparison["runs"], comparison["providers"], datasets,
    )
    labels, records = [], []
    for prompt in PROMPTS:
        rows = [row for row in overview if row["prompt"] == prompt]
        scope = "shared" if datasets is None else "selected"
        labels.append(f"{prompt}\n({len(rows[0]['datasets'])} {scope} datasets)")
        records.append({row["provider"]: row for row in rows})
    fig = six_metric_panels(comparison, labels, records)
    decorate(
        fig, comparison,
        "Four-model reference averages (not sample-paired)" if datasets is None
        else "Four-model reference averages: ToxiGen / HateCheck / AEGIS2",
        "threshold = 0.5 | shared non-PII dataset names | equal dataset weights | prompts kept separate"
        if datasets is None else
        "threshold = 0.5 | ToxiGen / HateCheck / AEGIS2 | equal dataset weights | prompts kept separate",
        "Each model uses its own successful samples; no common OpenJev sample IDs are available. Not a fair ranking.\n"
        "Latency = mean per-run elapsed / processed across four complete runs, then an equal-weight dataset mean.\n"
        + (
            "Missing any selected dataset's four-run timing makes average latency N/A; failure waiting is included.\n"
            if datasets is None else
            "Missing required data makes quality N/A; incomplete four-run timing makes latency N/A. Failures included.\n"
        )
        +
        "OpenJev summary precision is retained as supplied; raw predictions and repeat-call stability are unavailable.",
        legend_style="bars",
    )
    return fig


def three_dataset_averages_figure(report):
    rows = three_dataset_averages(report)
    supplied_colors = {
        "Dynamo CPU": "#6A3D9A", "Dynamo GPU (A100)": "#8C6BB1",
        "Dynamo GPU (L4)": "#BC9BD3", "Mistral": "#B15928", "Warden": "#666666",
    }
    labels = [NAMES.get(row["provider"], row["provider"]) for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(18.4, 16.5))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.14, hspace=0.3, wspace=0.65)
    for ax, metric, title in zip(
        axes.flat,
        ("accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms"),
        ("Accuracy", "Precision", "Recall", "F1", "AUROC", "AVG latency (ms)"),
    ):
        latency = metric == "mean_elapsed_ms"
        values = [
            row[metric] * (1 if latency else 100) if row[metric] is not None else np.nan
            for row in rows
        ]
        bars = ax.barh(
            np.arange(len(rows)), values,
            color=[COLORS[row["provider"]] if row["provider"] in COLORS
                   else supplied_colors[row["provider"]] for row in rows],
            edgecolor="#263238", linewidth=0.4,
        )
        for bar, value in zip(bars, values):
            available = math.isfinite(value)
            ax.annotate(
                f"{value:,.2f}" if available else "N/A",
                (value if available else 0, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
            )
        ax.set_yticks(np.arange(len(rows)), labels)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(title)
        if latency:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(0, max((value for value in values if math.isfinite(value)), default=1) * 6)
            ax.set_xlabel("ms / sample (log scale above 1 ms; lower is better)")
        else:
            ax.set_xlim(0, 120)
            ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel("% (higher is better)")
        style_axis(ax, "x")
    fig.suptitle("Three-dataset model reference averages", fontsize=16, y=0.97)
    fig.text(
        0.5, 0.935, "ToxiGen / HateCheck / AEGIS2 | equal dataset weights | not sample-paired",
        ha="center", fontsize=10,
    )
    fig.text(
        0.07, 0.095,
        "Original four models: prompt v1, threshold 0.5; quality and latency are equal-weight dataset means.\n"
        "Local latency uses only the v1 / 0.5 run per dataset; OpenJev uses its supplied summary latency.\n"
        "Dynamo / Mistral / Warden: user-supplied three-dataset means, scope confirmed by user (original '4' was a typo).\n"
        "Their prompt, threshold and timing protocol were not supplied; means are shown once.\n"
        "N/A = not supplied or required source data missing. F1 is not inferred from average precision / recall.\n"
        "Successful samples and timing protocols are not aligned; descriptive reference only, not a fair ranking.",
        fontsize=9, va="top",
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
        "各阈值重新调用模型，曲线混有重复调用波动，不是固定 score 下的纯阈值效应；"
        "这是同测试集探索，不是独立验证后的性能提升。",
    ),
    (
        "fig04_evaluation_coverage", coverage_figure, "测试覆盖率与请求失败",
        "覆盖率为成功样本数 / 参考样本数。灰色为未获得预测文件，斜线为完整落盘但全部请求失败；"
        "PII 行仅展示覆盖率，不代表脱敏质量。Laya 缺测不记零分。",
    ),
    (
        "fig05_dataset_metrics", dataset_metrics_figure, "逐数据集六项指标",
        "固定 v1 / 0.5，展示全部数据集的 Accuracy、Precision、Recall、F1、AUROC 和 AVG latency (ms)。"
        "质量指标使用各模型自身成功样本，不是共同样本公平排名；请求错误不计入质量分数。"
        "PII 仅衡量是否含 PII 的检出，不是脱敏质量，单类 AUROC 无定义。"
        "耗时为完整对齐日志的 elapsed / processed，包含失败等待及网络开销。"
        "缺测、无成功预测、无定义指标或无完整计时均标 N/A，不补零；请求全失败仍可有日志耗时。",
    ),
    (
        "fig06_model_averages", model_averages_figure, "模型平均指标",
        "固定 v1 / 0.5，在共同完成且有共同成功样本的非 PII 数据集上，展示 Average Accuracy、"
        "Avg Precision、Avg Recall、Avg AUROC 和 Avg latency (ms)。"
        "分类指标先按共同成功样本计算，再对数据集等权平均；耗时也对各数据集的整轮平均耗时等权平均，"
        "与图 2 按样本数加权的耗时口径不同。缺少任一共同数据集完整计时，平均耗时即标 N/A。",
    ),
    (
        "fig07_repeat_consistency_v1", lambda report: stability_figure(report, "v1"),
        "重复调用一致性：v1 共同数据集",
        "仅展示所有模型四轮均完整且有共同成功样本的数据集，每个数据集内使用所有模型四次均成功的同一批样本。"
        "平均分数总体标准差、平均分数极差、统一按 0.5 判定的翻转样本比例均越低越一致；"
        "前两项图内乘以 100 展示，原始分数单位为 0–1。"
        "同时展示所有模型四轮共同成功覆盖率及各模型的请求失败率。缺轮或无共同成功样本的数据集不展示。"
        "已测零值用基线标记显示，不是缺测或补零；一致不代表正确，历史服务变化也会影响结果。",
    ),
    (
        "fig08_repeat_consistency_v2", lambda report: stability_figure(report, "v2"),
        "重复调用一致性：v2 共同数据集",
        "与 v1 相同，仅展示所有模型四轮完整且有共同成功样本的数据集，按所有模型共同样本计算；v2 独立统计。"
        "直接比较四次 score；判定统一使用 score >= 0.5，不比较原 CSV 各不相同阈值的 prediction。"
        "仅四次调用，属于探索性观察，无法将差异完全归因于模型本身的随机性。"
        "零值用基线标记显示，表示已测无波动，不是没有数据。"
        "覆盖率使用所有模型四轮共同成功样本，失败率针对这些数据集各模型的全部已落盘请求。",
    ),
    (
        "fig09_repeat_consistency_averages", stability_averages_figure,
        "重复调用一致性：共同样本模型均值",
        "v1、v2 分开统计。每个数据集要求所有模型四轮完整，使用所有模型四次均成功的同一批样本，"
        "再对有共同成功样本的非 PII 数据集等权平均。三项均越低越一致。"
        "不同提示词的共同数据集或样本可能不同，无可用共同数据集标 N/A。"
        "已测零值用基线标记显示，不是缺测或补零。"
        "原始指标、共同样本数、覆盖率及失败率见完整报告与 CSV。",
    ),
)

EXTERNAL_FIGURES = (
    (
        "fig10_openjev_reference_v1", lambda report: external_dataset_figure(report, "v1"),
        "含 OpenJev 的四模型参考对比：v1",
        "固定阈值 0.5。OpenJev 为用户提供的内部汇总结果，其他模型使用各自成功样本；"
        "无法按 OpenJev 的共同成功样本对齐，不作为公平排名。"
        "Rows 是总行数，不将其当作有效样本数。PII 仅为是否含 PII 的检出，单类 AUROC 标 N/A。"
        "四模型耗时统一取每个数据集/提示词四轮完整日志的 elapsed/processed 等权均值，"
        "包含失败等待、客户端及网络开销；缺少完整四轮计时标 N/A。"
        "原有共同样本基线与稳定性分析保持不变，不加入仅有汇总数据的 OpenJev。",
    ),
    (
        "fig11_openjev_reference_v2", lambda report: external_dataset_figure(report, "v2"),
        "含 OpenJev 的四模型参考对比：v2",
        "口径与 v1 参考图相同，v2 独立统计。分类指标对应阈值 0.5，不对四个阈值的质量分数取平均。"
        "耗时则对同数据集/提示词四轮完整运行的 elapsed/processed 取均值。"
        "OpenJev 有效样本数及错误数未从有 OCR 错字的文本推断；原始数值采用最初清晰表格，不擅自修正。",
    ),
    (
        "fig12_openjev_reference_averages", external_averages_figure,
        "含 OpenJev 的四模型参考均值",
        "仅对所有模型均有可用质量结果的非 PII 数据集等权平均，v1/v2 分开。"
        "展示 Average Accuracy、Avg Precision、Avg Recall、Avg F1、Avg AUROC、Avg latency。"
        "这里仅共同数据集名称相同，各模型成功样本未对齐，不能与原有共同样本模型均值混排。"
        "计时先对各数据集完整四轮取均值，再对数据集等权平均；任一计时缺失则平均耗时为 N/A。",
    ),
    (
        "fig13_three_dataset_averages",
        three_dataset_averages_figure,
        "ToxiGen, HateCheck, AEGIS2: Multi-model Reference Averages",
        "Accuracy, Precision, Recall, F1, AUROC and average latency are reported for ToxiGen, HateCheck and AEGIS2, "
        "with equal weights of one third per dataset; Detect-Jailbreak and all other datasets are excluded. "
        "The original four models use only prompt v1 at threshold 0.5, each with its own successful samples; "
        "OpenJev results are supplied internal summaries. "
        "Local model latency is elapsed/processed from each dataset's v1 / 0.5 run, averaged equally across datasets; "
        "OpenJev latency comes from the corresponding supplied summary entries. "
        "Missing quality results or complete timing for any required dataset make the corresponding mean N/A; "
        "the dataset scope is not reduced. "
        "Dynamo CPU, Dynamo GPU (A100), Dynamo GPU (L4), Mistral and Warden use user-supplied three-dataset means. "
        "The user confirmed that '4' in the original Scope column was a typo. These means are not averaged again; "
        "their prompts, thresholds and timing protocols were not supplied. "
        "Unreported F1 or AUROC values are N/A; F1 is not inferred from average Precision/Recall. "
        "The latency axis uses a logarithmic scale above 1 ms. "
        "Successful samples and timing protocols are not aligned across models; this is a descriptive reference, not a fair ranking.",
    ),
)


def gallery_header(title):
    escaped = html.escape(title)
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + escaped + """</title><style>
body{font:16px/1.7 system-ui,sans-serif;background:#f3f5f8;color:#172438;margin:0}
main{max-width:1250px;margin:auto;padding:28px}section{background:white;padding:24px;margin:24px 0}
img{width:100%;height:auto}a{color:#174c90}h1,h2{line-height:1.4}
@media print{body{background:white}section{break-inside:avoid;padding:0}}
</style><main><h1>""" + escaped + "</h1>\n"


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
    if any(key not in report for key in (
        "stability", "stability_paired", "stability_overview", "stability_settings",
    )):
        raise ValueError(
            "Saved report has no repeat-call stability metrics. "
            "Regenerate with python -m scripts.compare_models before plotting."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_generated_at": report["generated_at"],
        "model_identities": report["model_identities"],
        "baseline_datasets": report["baseline_datasets"],
        "stability_settings": report["stability_settings"],
        "dpi": dpi, "figures": [],
    }
    figures = FIGURES
    if report.get("external_comparison"):
        figures += EXTERNAL_FIGURES
        comparison = report["external_comparison"]
        manifest["external_comparison"] = {
            key: comparison[key] for key in (
                "source", "source_kind", "notes", "providers", "model_identities",
                "classification_threshold", "latency_method", "repeat_thresholds",
            )
        }
    with plt.rc_context(STYLE), PdfPages(output_dir / "paper_figures.pdf") as bundle:
        for stem, builder, title, caption in figures:
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
    cards, linked_figures = [], []
    for figure in manifest["figures"]:
        downloads = " | ".join(
            f'<a href="{filename}">{extension.upper()}</a>'
            for extension, filename in figure["files"].items()
        )
        if figure["name"].startswith(("fig07_", "fig08_", "fig09_")):
            linked_figures.append(
                f'<p><a href="{figure["files"]["svg"]}">{html.escape(figure["title"])}</a>'
                + "（" + downloads + "）</p>"
            )
            continue
        image_source = figure["files"]["svg"]
        if figure["name"] in ("fig01_paired_quality", "fig13_three_dataset_averages"):
            image_data = base64.b64encode((output_dir / figure["files"]["png"]).read_bytes()).decode("ascii")
            image_source = "data:image/png;base64," + image_data
        card = (
            "<section><h2>" + html.escape(figure["title"]) + "</h2>"
            + f'<img src="{image_source}" alt="{html.escape(figure["title"])}">'
            + "<p>" + html.escape(figure["caption"]) + "</p><p>" + downloads + "</p></section>"
        )
        if figure["name"] == "fig13_three_dataset_averages":
            averages = three_dataset_averages(report)
            with (output_dir / "three_dataset_averages.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(averages[0]))
                writer.writeheader()
                writer.writerows({**row, "datasets": ",".join(row["datasets"])} for row in averages)
            page = gallery_header("Three-dataset Multi-model Reference Averages")
            page += '<p><a href="index.html">返回完整图表报告</a> | '
            page += '<a href="three_dataset_averages.csv">下载三数据集均值 CSV</a></p>'
            page += "<p>数据快照（UTC）：" + html.escape(report["generated_at"]) + "</p>"
            page += card + "</main></html>\n"
            (output_dir / "three_dataset_comparison.html").write_text(page, encoding="utf-8")
            continue
        cards.append(card)
    gallery = gallery_header("模型对比图")
    gallery += """<p>图内使用英文标签，图注使用中文。PDF / SVG 为矢量格式，PNG 默认为 300 DPI。
模型身份由用户补充；TypeSafe resolved model 需以接口返回为准。</p>
<p><a href="paper_figures.pdf">下载全部图表 PDF</a> |
<a href="table01_baseline.tex">下载 LaTeX 三线表</a> |
<a href="../comparison.html">完整指标与结论报告（默认目录）</a></p>"""
    gallery += (
        '<p>重复调用一致性数据：<a href="../stability.csv">逐数据集 CSV</a> | '
        '<a href="../stability_paired.csv">共同样本 CSV</a> | '
        '<a href="../stability_overview.csv">模型均值 CSV</a></p>'
    )
    if report.get("external_comparison"):
        gallery += (
            '<p>OpenJev 内部汇总（非共同样本基线）：'
            '<a href="../openjev_summary.csv">汇总数据 CSV</a> | '
            '<a href="../external_comparison_runs.csv">四模型参考明细 CSV</a> | '
            '<a href="../external_comparison_overview.csv">参考均值 CSV</a></p>'
            '<p><a href="three_dataset_comparison.html">'
            'View separately: ToxiGen, HateCheck, AEGIS2 Multi-model Reference Averages</a></p>'
        )
    gallery += "<p>数据快照（UTC）：" + html.escape(report["generated_at"]) + "</p>"
    gallery += "".join(cards)
    gallery += "<section><h2>重复调用一致性（点击查看）</h2>" + "".join(linked_figures) + "</section>"
    gallery += "</main></html>\n"
    (output_dir / "index.html").write_text(gallery, encoding="utf-8")
    return manifest
