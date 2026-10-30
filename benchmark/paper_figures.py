"""Plots of saved comparison metrics, without new inference."""

import base64
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from benchmark.comparison import DATASETS, PROMPTS, STABILITY_METRICS, THRESHOLDS, baseline_rows
from benchmark.external_summary import (
    FINE_TUNED_DISPLAY_NAME, INTERNAL_IDENTITIES, METRICS, category_reference_averages,
    three_dataset_averages, v1_reference_scope_averages,
)


COLORS = {
    "fastino.ai": "#0072B2", "laya-ai.com": "#D55E00", "typesafe.ai": "#009E73",
    "openjev": "#CC79A7", "openjev-32b": "#6A3D9A", "microsoft-decision-1": "#B8860B",
    "fastino-ft-gg6-r1-ml2048-s7": "#E7298A",
}
NAMES = {
    "fastino.ai": "Fastino", "laya-ai.com": "Laya", "typesafe.ai": "TypeSafe", "openjev": "OpenJev 4B",
    **{provider: identity["display_name"] for provider, identity in INTERNAL_IDENTITIES.items()},
    "fastino-ft-gg6-r1-ml2048-s7": FINE_TUNED_DISPLAY_NAME,
}
MARKERS = {
    "fastino.ai": "o", "laya-ai.com": "s", "typesafe.ai": "D", "openjev": "^",
    "openjev-32b": "v", "microsoft-decision-1": "P",
    "fastino-ft-gg6-r1-ml2048-s7": "D",
}
HATCHES = {
    "fastino.ai": "", "laya-ai.com": "///", "typesafe.ai": "...", "openjev": "xxx",
    "openjev-32b": "\\\\", "microsoft-decision-1": "++",
    "fastino-ft-gg6-r1-ml2048-s7": "////",
}
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
REFERENCE_AVERAGE_PANEL_ORDER = ("accuracy", "f1", "precision", "recall", "auroc", "mean_elapsed_ms")
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
    if provider in report.get("fine_tuned_models", {}):
        model = report["fine_tuned_models"][provider]
        return f"{model['display_name']}\nFine-tuned · 300M\nmodel ID: {model['model_id']}"
    if provider == "openjev":
        return "OpenJev 4B | supplied internal summary\nmodel / version: not supplied (not sample-paired)"
    if provider in INTERNAL_IDENTITIES:
        return f"{NAMES[provider]} | supplied internal summary\nsingle-run timing (not sample-paired)"
    identity = report["model_identities"][provider]
    lines = [f"{NAMES[provider]} | request: {identity['request_model']}"]
    for role in ("resolved_model", "serving_model", "routed_model"):
        if role in identity:
            value = identity[role] or "unconfirmed (API response required)"
            lines.append(f"{role.removesuffix('_model')}: {value}")
    return "\n".join(lines)


def bar_style(report, provider):
    fine_tuned = provider in report.get("fine_tuned_models", {})
    return {
        "edgecolor": "#172438" if fine_tuned else "#263238",
        "linewidth": 1.5 if fine_tuned else 0.4,
        "hatch": HATCHES[provider],
    }


def decorate(fig, report, title, subtitle, note, legend_style="markers", stacked=False):
    height_points = fig.get_figheight() * 72
    subtitle_y = min(0.975, 0.995 - 30 / height_points) if stacked else 0.925
    legend_y = min(0.955, subtitle_y - 24 / height_points) if stacked else 0.015
    fig.suptitle(title, fontsize=15, y=0.995 if stacked else 0.98)
    fig.text(0.5, subtitle_y, subtitle, ha="center", fontsize=9)
    note_y = 0.055 if stacked else 0.25 if len(report["providers"]) > 4 else 0.19
    fig.text(0.07, note_y, note, fontsize=9 if stacked else 8, va="top")
    handles = []
    for provider in report["providers"]:
        label = model_label(report, provider)
        if legend_style == "bars":
            handle = Patch(facecolor=COLORS[provider], **bar_style(report, provider), label=label)
        else:
            handle = Line2D(
                [], [], color=COLORS[provider],
                marker=MARKERS[provider] if legend_style == "markers" else None,
                linewidth=0 if legend_style == "identity" else 1.8, label=label,
            )
        handles.append(handle)
    fig.legend(
        handles=handles, loc="upper center" if stacked else "lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3 if len(handles) > 4 else len(handles), frameon=False, handlelength=2.2, columnspacing=2,
    )


def style_axis(ax, grid="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#D8DDE3", linewidth=0.6, linestyle="--", alpha=0.8)


def quality_figure(report):
    rows = baseline_rows(report)
    datasets = report["baseline_datasets"]
    lookup = {(row["provider"], row["dataset"]): row for row in rows}
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 11.5))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.16, hspace=0.6)
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
                **bar_style(report, provider),
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
        fig, report, "Sample-Aligned Classification Comparison",
        "v1 | threshold = 0.5 | common completed non-PII datasets | higher is better",
        "Each dataset uses the same successful sample IDs for all selected models.\n"
        "Values are descriptive point estimates; repeated sweeps are not independent replicates.",
        legend_style="bars",
        stacked=True,
    )
    return fig


def shared_baseline_timing(report):
    runs = {
        (row["provider"], row["dataset"]): row for row in report["runs"]
        if row["prompt"] == "v1" and row["threshold"] == 0.5
    }
    timed_datasets = [
        dataset for dataset in report["baseline_datasets"]
        if all(
            (run := runs.get((provider, dataset))) is not None
            and run["status"] == "complete" and run["rows"] > 0
            and run.get("mean_elapsed_ms") is not None
            for provider in report["providers"]
        )
    ]
    return timed_datasets, runs


def tradeoff_figure(report):
    paired = {(row["provider"], row["dataset"]): row for row in baseline_rows(report)}
    timed_datasets, runs = shared_baseline_timing(report)
    timed_datasets = [
        dataset for dataset in timed_datasets
        if all(paired.get((provider, dataset), {}).get("auroc") is not None
               for provider in report["providers"])
    ]
    timing = {}
    if timed_datasets:
        for provider in report["providers"]:
            own = [runs[(provider, dataset)] for dataset in timed_datasets]
            timing[provider] = (
                sum(row["mean_elapsed_ms"] * row["rows"] for row in own) / sum(row["rows"] for row in own),
                sum(paired[(provider, dataset)]["auroc"] for dataset in timed_datasets) / len(timed_datasets),
            )
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.32, wspace=0.28)
    for row in report["overview"]:
        provider = row["provider"]
        points = (
            (row["macro_fpr"], row["macro_fnr"]),
            timing.get(provider, (None, None)),
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
    times = [value[0] for value in timing.values()]
    axes[1].set(
        xlabel="Amortized end-to-end elapsed time (ms / sample)", ylabel="Macro AUROC (%)",
        title="(b) Quality vs. elapsed time (toward top-left is better)",
        xlim=(0, max(times, default=1) * 1.28), ylim=(0, 110),
    )
    axes[1].set_yticks(range(0, 101, 20))
    if not timed_datasets:
        axes[1].text(
            0.5, 0.5, "N/A\nNo shared datasets with complete v1 / 0.5 timing",
            transform=axes[1].transAxes, ha="center", va="center",
        )
    for ax in axes:
        style_axis(ax, "both")
    decorate(
        fig, report, "Error Rates and Latency Trade-offs",
        f"v1 | threshold = 0.5 | (a) {len(report['baseline_datasets'])} shared datasets"
        f" | (b) {len(timed_datasets)} shared fully timed datasets",
        "Panel (b) recomputes BOTH macro AUROC and sample-weighted elapsed on the same shared timed subset:\n"
        + (", ".join(DATASET_NAMES[dataset] for dataset in timed_datasets) or "None")
        + ". No four-run timing averages substitute for missing v1 / 0.5 timing.\n"
        "Elapsed time = total logged elapsed / all processed subset rows, including failed requests.\n"
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
    own_lookup = {
        (row["provider"], row["dataset"], row["prompt"], row["threshold"]): row
        for row in report["runs"]
    }
    for index, ax in enumerate(grid.flat):
        if index >= len(datasets):
            ax.set_visible(False)
            continue
        dataset = datasets[index]
        for provider in report["providers"]:
            for prompt, linestyle, marker in (("v1", "-", "o"), ("v2", "--", "^")):
                values = []
                paired_indices, own_thresholds, own_values = [], [], []
                for threshold in THRESHOLDS:
                    key = (provider, dataset, prompt, threshold)
                    row = lookup.get(key)
                    value = row["f1"] if row and row["n"] else None
                    if value is not None:
                        paired_indices.append(len(values))
                    else:
                        own_row = own_lookup.get(key)
                        value = own_row.get("f1") if own_row and own_row["valid"] else None
                        if value is not None:
                            own_thresholds.append(threshold)
                            own_values.append(value * 100)
                    values.append(value * 100 if value is not None else np.nan)
                ax.plot(THRESHOLDS, values, color=COLORS[provider], linestyle=linestyle,
                        marker=marker, markevery=paired_indices, markersize=4.5, linewidth=1.6)
                if own_values:
                    ax.scatter(own_thresholds, own_values, color=COLORS[provider],
                               marker=marker, facecolors="none", s=4.5 ** 2, zorder=3)
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
            Line2D([], [], color="#333333", marker="o", linestyle="none",
                   label="Paired successful samples"),
            Line2D([], [], color="#333333", marker="o", markerfacecolor="none", linestyle="none",
                   label="Own successful samples"),
        ],
        loc="lower left", ncol=2, frameon=True, framealpha=0.95,
    )
    decorate(
        fig, report, "Sensitivity to prompt version and decision threshold",
        "Exploratory test-set sweeps, not held-out model selection",
        "Filled markers: paired successful samples; hollow: own successes when paired metrics are unavailable.\n"
        "Sample sets can differ across points; this is not a fair ranking. Missing or all-failed runs remain gaps.\n"
        "Curves mix threshold effects and repeat-call variation.\n"
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
        fig, report, "Evaluation Coverage and Request Failures",
        "v1 | threshold = 0.5 | counts are successful / reference rows",
        "Gray = no saved predictions; hatched = complete CSV with no successful requests, not a missing run.\n"
        "PII is ordinary binary classification (PII present vs absent); rows show coverage, not masking quality.",
        legend_style="identity",
    )
    fig.legends[0].remove()
    bounds = ax.get_position()
    fig.texts[-1].set_position((bounds.x0, 0.10))
    for index, provider in enumerate(report["providers"]):
        center = bounds.x0 + (index + 0.5) * bounds.width / len(report["providers"])
        fig.text(center, 0.205, model_label(report, provider), ha="center", va="top", fontsize=8)
    return fig


def metric_bars(ax, report, labels, records, metric, title, latency=False,
                ylabel=None, autoscale=False, decimals=1, mark_zero=False, label_fontsize=7):
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
            **bar_style(report, provider),
        )
        for bar, value, record in zip(bars, values, records):
            available = math.isfinite(value)
            if available:
                finite_values.append(value)
            if mark_zero and available and value == 0:
                ax.plot(
                    bar.get_x() + bar.get_width() / 2, 0, marker=MARKERS[provider],
                    color=COLORS[provider], markersize=4, linestyle="none",
                    clip_on=False, zorder=4,
                )
            label = f"{value:.{decimals}f}" if available else "N/A"
            if available and latency and record.get(provider, {}).get("latency_source_kind") == "single_complete_run":
                label += "*"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, value if available else 0),
                xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=label_fontsize, rotation=90 if len(labels) > 4 else 0,
                color="#172438" if available else "#566170",
            )
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel or ("ms / sample (lower is better)" if latency else "% (higher is better)"))
    ax.set_ylim(0, max(max(finite_values, default=0) * 1.3, 1) if latency or autoscale else 120)
    if not latency and not autoscale:
        ax.set_yticks(range(0, 101, 20))
    style_axis(ax)


def six_metric_panels(report, labels, records, latency_labels=None, single_column=False):
    expanded = len(report["providers"]) > 4
    if single_column:
        fig, axes = plt.subplots(6, 1, figsize=(14.5, 28))
        fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.09, hspace=0.65)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(20, 15.2) if expanded else (18.4, 13.6))
        fig.subplots_adjust(
            left=0.06, right=0.985, top=0.87, bottom=0.35 if expanded else 0.29, hspace=0.7, wspace=0.18,
        )
    for ax, metric, title in zip(
        axes.flat,
        METRICS,
        ("Accuracy", "Precision", "Recall", "F1", "AUROC", "AVG latency (ms)"),
    ):
        latency = metric == "mean_elapsed_ms"
        panel_labels = latency_labels if latency and latency_labels is not None else labels
        metric_bars(ax, report, panel_labels, records, metric, title, latency=latency,
                    label_fontsize=10 if single_column else 7)
        if single_column:
            ax.tick_params(labelsize=11)
            ax.title.set_fontsize(14)
            ax.yaxis.label.set_fontsize(12)
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
        "PII is ordinary binary classification (PII present vs absent), not extraction or masking quality.\n"
        "PII datasets contain positives only; AUROC is undefined.\n"
        "Latency = logged end-to-end elapsed / all processed rows, including failed requests; not pure inference time.\n"
        "Successful sample sets can differ between models; use the paired baseline for fair quality comparisons.",
        legend_style="bars",
    )
    return fig


def model_averages_figure(report):
    timed_datasets, runs = shared_baseline_timing(report)
    averages = {}
    for row in report["overview"]:
        provider = row["provider"]
        averages[provider] = {
            **row,
            "shared_elapsed_ms": (
                sum(runs[(provider, dataset)]["mean_elapsed_ms"] for dataset in timed_datasets)
                / len(timed_datasets) if timed_datasets else None
            ),
        }
    records = [averages]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.32, hspace=0.38, wspace=0.28)
    for ax, metric, title in zip(
        axes.flat,
        ("macro_accuracy", "macro_precision", "macro_recall", "macro_auroc", "shared_elapsed_ms"),
        ("Average Accuracy", "Avg Precision", "Avg Recall", "Avg AUROC",
         f"Avg latency (ms)\n{len(timed_datasets)} shared fully timed datasets"),
    ):
        latency = metric == "shared_elapsed_ms"
        labels = ["Shared timed subset" if latency else "Shared baseline"]
        metric_bars(ax, report, labels, records, metric, title, latency=latency)
    axes.flat[-1].set_visible(False)
    decorate(
        fig, report, "Model averages: shared quality baseline and timed subset",
        f"v1 | threshold = 0.5 | quality: {len(report['baseline_datasets'])} shared non-PII datasets"
        f" | latency: {len(timed_datasets)} shared fully timed datasets | equal dataset weights",
        "Quality: common successful sample IDs within each dataset, then an unweighted mean across datasets.\n"
        "Latency: unweighted mean of per-dataset log elapsed / processed rows on the SAME timed subset for all models:\n"
        + (", ".join(DATASET_NAMES[dataset] for dataset in timed_datasets) or "None")
        + ". Quality and latency dataset scopes can differ.\n"
        "Elapsed includes failures and network overhead, not pure inference time. No four-run timing substitutes.\n"
        "N/A timing means no shared fully timed baseline datasets; missing is never zero.",
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


def external_reference_providers(comparison, prompt):
    return [
        provider for provider in comparison["providers"]
        if (provider not in INTERNAL_IDENTITIES and provider not in comparison.get("fine_tuned_models", {}))
        or any(row["provider"] == provider and row["prompt"] == prompt for row in comparison["runs"])
    ]


def fine_tuned_note(report):
    return "\n".join(
        f"{model['display_name']} | model ID: {model['model_id']}\n"
        f"{model['valid']:,} successful / {model['rows']:,} rows; {model['errors']} errors. "
        + model["notes"]
        for model in report.get("external_comparison", {}).get("fine_tuned_models", {}).values()
    )


def reference_latency_log_axis(fig):
    latency_axis = fig.axes[-1]
    positive = [
        bar.get_height() for bar in latency_axis.patches
        if math.isfinite(bar.get_height()) and bar.get_height() > 0
    ]
    lower = min(positive) / 1.8 if positive else 1
    upper = max(positive) * 3 if positive else 10
    latency_axis.set_yscale("log")
    latency_axis.set_ylim(lower, upper)
    latency_axis.set_ylabel("ms / sample (log scale; lower is better)", fontsize=12)
    for annotation in latency_axis.texts:
        if annotation.xy[1] <= 0:
            annotation.xy = (annotation.xy[0], lower * 1.1)


def highlight_pii_task(fig, stacked=False):
    fig.text(
        0.07, 0.078 if stacked else 0.25,
        "PII = ordinary binary classification (present vs absent), NOT masking quality.",
        fontsize=12, fontweight="bold", color="#78350F", va="top",
        bbox={"facecolor": "#FFF3CD", "edgecolor": "#D97706", "linewidth": 1.2, "boxstyle": "round,pad=0.4"},
    )


def external_category_figure(report):
    source = report["external_comparison"]
    comparison = {**source, "providers": external_reference_providers(source, "v1")}
    averages = category_reference_averages(report)
    categories = (
        ("harmful_content", "Harmful content\n(3 datasets)"),
        ("jailbreak", "Jailbreak\n(3 datasets)"),
        ("pii", "PII binary classification\n(2 datasets)"),
    )
    labels = [label for _, label in categories]
    records = [
        {row["provider"]: row for row in averages if row["category"] == category}
        for category, _ in categories
    ]
    fig = six_metric_panels(comparison, labels, records, single_column=True)
    reference_latency_log_axis(fig)
    decorate(
        fig, comparison, "Model reference averages by dataset category",
        "v1 / threshold = 0.5 | equal dataset weights within each category | NOT sample-paired",
        "Harmful content: ToxiGen, HateCheck, AEGIS2. Jailbreak: Detect-Jailbreak, Jailbreak-Classification, JailbreakHub.\n"
        "PII: Nano-1K, Mini-10K; ordinary binary classification (present vs absent), not extraction or masking quality.\n"
        "Current PII datasets contain positives only; AUROC is undefined.\n"
        "Each metric is the arithmetic mean of per-dataset metrics, not pooled samples or F1 recomputed from means.\n"
        "Missing required data or undefined metrics = N/A; category scopes are never reduced to available datasets.\n"
        "Latency averages the same reference timings as the per-dataset chart; protocols differ and include failure waits.\n"
        "Own successful samples / supplied aggregates; descriptive only, not a fair ranking.",
        legend_style="bars", stacked=True,
    )
    highlight_pii_task(fig, stacked=True)
    return fig


def external_dataset_figure(report, prompt):
    source = report["external_comparison"]
    comparison = {**source, "providers": external_reference_providers(source, prompt)}
    lookup = {
        (row["provider"], row["dataset"]): row for row in comparison["runs"]
        if row["prompt"] == prompt
    }
    labels = [DATASET_NAMES[dataset].replace("-", "-\n") for dataset in DATASETS]
    records = [
        {provider: lookup.get((provider, dataset), {}) for provider in comparison["providers"]}
        for dataset in DATASETS
    ]
    stacked = prompt == "v1"
    fig = six_metric_panels(comparison, labels, records, single_column=stacked)
    if stacked:
        reference_latency_log_axis(fig)
    note = (
        "OpenJev 4B: four-run summary timing; local models prefer four-run log averages. 4B valid/error counts are unknown.\n"
    )
    if stacked:
        note += "* = single-run v1 / 0.5 timing; local models require complete, matched logs; internal models use supplied summaries.\n"
    if any(provider in INTERNAL_IDENTITIES for provider in comparison["providers"]):
        note += "OpenJev 32B / Microsoft Decision-1: supplied single-run timing, including failures; protocols differ.\n"
        counts = [
            row for row in source["runs"] if row["provider"] == "openjev-32b" and row["prompt"] == prompt
        ]
        if counts:
            note += (
                f"32B has {sum(row['errors'] for row in counts):,} errors / "
                f"{sum(row['rows'] for row in counts):,} rows; quality excludes errors.\n"
            )
    note += (
        "N/A = missing quality, undefined metric, or incomplete timing; missing values are not zero.\n"
        "Descriptive only, not a fair ranking. PII is ordinary binary classification (present vs absent), not masking quality."
    )
    decorate(
        fig, comparison, f"Model reference comparison: prompt {prompt}",
        "threshold = 0.5 | own successful samples | internal models: aggregates, NOT sample-paired",
        note,
        legend_style="bars",
        stacked=stacked,
    )
    highlight_pii_task(fig, stacked=stacked)
    return fig


def external_averages_figure(report, datasets=None):
    comparison = report["external_comparison"]
    rows = [row for row in v1_reference_scope_averages(report, datasets) if row["scope"] == "non_pii"]
    scope_datasets = rows[0]["datasets"]
    timed_datasets = rows[0]["timing_datasets"]
    labels = [f"v1\n({len(scope_datasets)} shared non-PII datasets)"]
    records = [{row["provider"]: {**row, "mean_elapsed_ms": row["shared_elapsed_ms"]} for row in rows}]
    latency_labels = [f"v1\n({len(timed_datasets)} shared timed datasets)"]
    fig = six_metric_panels(comparison, labels, records, latency_labels=latency_labels)
    fig.set_size_inches(14.5, 14.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.20, hspace=0.65, wspace=0.28)
    axes = dict(zip(METRICS, fig.axes))
    grid = fig.axes[0].get_subplotspec().get_gridspec()
    for index, metric in enumerate(REFERENCE_AVERAGE_PANEL_ORDER):
        ax = axes[metric]
        ax.set_subplotspec(grid[index])
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.tick_params(labelsize=10)
        ax.title.set_fontsize(13)
        for label in ax.texts:
            label.set_fontsize(11)
    decorate(
        fig, comparison,
        "Model reference averages (not sample-paired)" if datasets is None
        else "Model reference averages: ToxiGen / HateCheck / AEGIS2",
        "v1 | threshold = 0.5 | shared non-PII dataset names | equal dataset weights | PII reported separately",
        "Each model uses its own successful samples; aggregate models have no paired sample IDs. Not a fair ranking.\n"
        "Latency: equal dataset weights on the SAME fully timed v1 / 0.5 subset for all models:\n"
        + (", ".join(DATASET_NAMES[dataset] for dataset in timed_datasets) or "None")
        + ". Quality and latency dataset scopes can differ.\n"
        "APIs: v1 / 0.5 single-run timing. OpenJev 4B: supplied summary; 32B / Decision-1: supplied single-run timing.\n"
        "PII is ordinary binary classification (present vs absent), not masking quality; positives-only results are separate.\n"
        "Missing quality stays N/A; no shared timing subset = N/A latency. Timing protocols differ; not inference time.",
        legend_style="bars",
        stacked=True,
    )
    fig.texts[-1].set_y(0.11)
    return fig


def three_dataset_averages_figure(report):
    rows = three_dataset_averages(report)
    supplied_colors = {
        "Dynamo CPU": "#6A3D9A", "Dynamo GPU (A100)": "#8C6BB1",
        "Dynamo GPU (L4)": "#BC9BD3", "Mistral": "#B15928", "Warden": "#666666",
    }
    labels = [NAMES.get(row["provider"], row["provider"]) for row in rows]
    groups = [
        "api" if (
            row["provider"] in report["providers"]
            or row["provider"] in report["external_comparison"].get("fine_tuned_models", {})
        )
        else "internal" if row["provider"] == "openjev" or row["provider"] in INTERNAL_IDENTITIES
        else "dynamo" if row["provider"].startswith("Dynamo ") else "other"
        for row in rows
    ]
    boundaries = [index + 0.5 for index in range(len(rows) - 1) if groups[index] != groups[index + 1]]
    has_fine_tuned = bool(report["external_comparison"].get("fine_tuned_models"))
    fig, axes = plt.subplots(3, 2, figsize=(20 if has_fine_tuned else 16, 20))
    fig.subplots_adjust(
        left=0.22 if has_fine_tuned else 0.16, right=0.98, top=0.91, bottom=0.14,
        hspace=0.38, wspace=0.9 if has_fine_tuned else 0.6,
    )
    for ax, metric, title in zip(
        axes.flat,
        METRICS,
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
        for bar, row in zip(bars, rows):
            if row["provider"] in report["external_comparison"].get("fine_tuned_models", {}):
                bar.set(**bar_style(report["external_comparison"], row["provider"]))
        for bar, value in zip(bars, values):
            available = math.isfinite(value)
            ax.annotate(
                f"{value:,.2f}" if available else "N/A",
                (value if available else 0, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=11,
            )
        ax.set_yticks(np.arange(len(rows)), labels)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(title, fontsize=14)
        ax.tick_params(labelsize=12)
        for boundary in boundaries:
            ax.axhline(boundary, color="#BFC8D2", linewidth=0.8)
        if latency:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(0, max((value for value in values if math.isfinite(value)), default=1) * 6)
            ax.set_xlabel("ms / sample (log scale above 1 ms; lower is better)", fontsize=12)
        else:
            ax.set_xlim(0, 120)
            ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel("% (higher is better)", fontsize=12)
        style_axis(ax, "x")
    fig.suptitle("Three-dataset model reference averages", fontsize=17, y=0.98)
    fig.text(
        0.5, 0.955, "Computed averages: ToxiGen / HateCheck / AEGIS2 | supplied-model dataset scopes unverified",
        ha="center", fontsize=11,
    )
    fig.text(
        0.07, 0.09,
        "Computed models: no internal dataset was available; v1 / 0.5 means use only the three shared datasets.\n"
        "Local latency uses v1 / 0.5 runs; OpenJev 4B / 32B and Decision-1 use their supplied summary latency.\n"
        "Dynamo / Mistral / Warden: supplied averages; their evaluations may also include an internal dataset.\n"
        "Its inclusion in their averages is unconfirmed; prompt, threshold and timing protocols were not supplied.\n"
        "N/A = not supplied or required source data missing. F1 is not inferred from average precision / recall.\n"
        "Successful samples and timing protocols are not aligned; descriptive reference only, not a fair ranking.",
        fontsize=9, va="top",
    )
    if report["external_comparison"].get("fine_tuned_models"):
        models = report["external_comparison"]["fine_tuned_models"]
        fig.legend(
            handles=[
                Patch(
                    facecolor=COLORS[provider],
                    **bar_style(report["external_comparison"], provider),
                    label="Fine-tuned · 300M",
                )
                for provider in models
            ],
            loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=10,
        )
        fig.text(
            0.07, 0.025,
            "\n".join(f"{model['display_name']} | model ID: {model['model_id']}" for model in models.values()),
            fontsize=9, va="top",
        )
    return fig


FIGURES = (
    (
        "fig01_paired_quality", quality_figure, "Sample-Aligned Classification Comparison",
        "Fixed prompt v1 and threshold 0.5; F1 and AUROC are shown per dataset. "
        "Within each dataset, all models use exactly the same successful sample IDs, "
        "unlike the multi-model reference pages, which use each model's own successful samples. "
        "The dotted 50% line in the AUROC panel is the random-ranking baseline. No confidence intervals are shown.",
    ),
    (
        "fig02_quality_cost_tradeoff", tradeoff_figure, "Error Rates and Latency Trade-offs",
        "Lower-left is better in the left panel, which uses all shared baseline datasets. "
        "Upper-left is better in the right panel, which uses only shared datasets with complete v1 / 0.5 timing "
        "for every model; both macro AUROC and sample-weighted elapsed time are recomputed on that same subset. "
        "The dataset scopes may differ between panels and are stated in the figure. "
        "Four-run average timing never substitutes for missing single-run timing. "
        "Elapsed time includes network overhead and waits for failed requests; it is not pure inference latency or P95. "
        "Timing is N/A when no shared fully timed subset exists.",
    ),
    (
        "fig03_threshold_sensitivity", threshold_figure, "Prompt and Threshold Sensitivity",
        "Colors identify models; solid lines with circles indicate v1, and dashed lines with triangles indicate v2. "
        "Filled markers use shared successful samples at the same configuration. "
        "When paired metrics are unavailable, hollow markers show the model's own successful samples, "
        "so missing results from another model do not hide available measurements. "
        "Missing or all-failed configurations remain gaps, without zero-filling or interpolation. "
        "Sample sets may differ across points; this is not a fair ranking. "
        "Each threshold requires a new model call, so the curves mix threshold effects with repeat-call variation "
        "rather than isolating threshold effects on fixed scores. "
        "This is exploration on the same test set, not an independently validated performance improvement.",
    ),
    (
        "fig04_evaluation_coverage", coverage_figure, "Evaluation Coverage and Request Failures",
        "Coverage is successful samples / reference samples. Gray indicates an unavailable prediction file; "
        "hatching indicates a fully saved run in which every request failed. "
        "PII is ordinary binary classification (PII present vs absent); rows show coverage, not masking quality. "
        "Missing Laya results are not scored as zero.",
    ),
    (
        "fig05_dataset_metrics", dataset_metrics_figure, "Per-Dataset Performance Metrics",
        "Fixed prompt v1 and threshold 0.5; Accuracy, Precision, Recall, F1, AUROC and average latency (ms) "
        "are shown for all datasets. Quality metrics use each model's own successful samples, "
        "not a fair sample-aligned ranking; request errors are excluded from quality scores. "
        "PII is ordinary binary classification (PII present vs absent), not entity extraction or masking quality; "
        "the current positives-only labels make AUROC undefined. "
        "Elapsed time is elapsed / processed from complete, matched logs, including network overhead and failure waits. "
        "Missing results, no successful predictions, undefined metrics or incomplete timing are N/A, never zero-filled. "
        "A run with all requests failed may still have recorded elapsed time.",
    ),
    (
        "fig06_model_averages", model_averages_figure, "Sample-Aligned Model Averages",
        "Fixed prompt v1 and threshold 0.5; average Accuracy, Precision, Recall, AUROC and latency (ms) "
        "are reported for jointly completed non-PII datasets with shared successful samples. "
        "Classification metrics are computed on shared successful samples, then averaged equally across datasets. "
        "Latency also gives equal weight to each dataset's full-run average, but uses only the shared baseline subset "
        "with complete v1 / 0.5 timing for every model. All models use exactly the same timing dataset scope. "
        "Quality and timing scopes may differ; actual counts and timing datasets are stated in the figure. "
        "Four-run averages never substitute for missing single-run timing. "
        "This differs from the sample-weighted timing in Figure 2. "
        "Latency is N/A only when there is no shared fully timed subset.",
    ),
    (
        "fig07_repeat_consistency_v1", lambda report: stability_figure(report, "v1"),
        "Repeat-Call Consistency: v1",
        "Only datasets with all four rounds complete for every model and shared successful samples are shown. "
        "Within each dataset, every model uses the same samples that succeeded in all four calls for all models. "
        "Lower mean population standard deviation of scores, mean score range and decision-flip rate "
        "at a fixed threshold of 0.5 indicate greater consistency. "
        "The first two metrics are multiplied by 100 in the figure; original scores range from 0 to 1. "
        "Shared four-round success coverage and each model's request failure rate are also shown. "
        "Datasets with missing rounds or no shared successful samples are excluded. "
        "Measured zeros use baseline markers; they are not missing or imputed values. "
        "Consistency does not imply correctness, and historical service changes may also affect results.",
    ),
    (
        "fig08_repeat_consistency_v2", lambda report: stability_figure(report, "v2"),
        "Repeat-Call Consistency: v2",
        "As for v1, only datasets with four complete rounds for every model and shared successful samples are shown. "
        "Metrics use samples shared by all models; v2 is analyzed separately. "
        "The four raw scores are compared directly, with decisions uniformly defined as score >= 0.5, "
        "rather than comparing prediction columns produced at different CSV thresholds. "
        "Four calls provide exploratory evidence only; differences cannot be attributed entirely to model randomness. "
        "Zero values use baseline markers and mean measured stability, not missing data. "
        "Coverage uses samples successful across all models and all four rounds; "
        "failure rates use each model's complete saved requests for those datasets.",
    ),
    (
        "fig09_repeat_consistency_averages", stability_averages_figure,
        "Repeat-Call Consistency Averages",
        "v1 and v2 are analyzed separately. Each dataset requires four complete rounds for every model "
        "and uses the same samples that succeeded for all models in all four calls. "
        "Metrics are then averaged equally across non-PII datasets with shared successful samples. "
        "Lower values indicate greater consistency for all three metrics. "
        "Shared datasets or samples may differ between prompts. If no shared dataset is available, the result is N/A. "
        "Measured zeros use baseline markers and are neither missing nor imputed. "
        "See the full report and CSV files for raw metrics, shared sample counts, coverage and failure rates.",
    ),
)

EXTERNAL_FIGURES = (
    (
        "fig14_category_reference_averages", external_category_figure,
        "Dataset-Category Multi-Model Reference Averages (v1 / Threshold 0.5)",
        "Fixed prompt v1 and threshold 0.5. Harmful content averages ToxiGen, HateCheck and AEGIS2; "
        "Jailbreak averages Detect-Jailbreak, Jailbreak-Classification and JailbreakHub; "
        "PII averages Nano-1K and Mini-10K. Every dataset within a category has equal weight. "
        "All six metrics are averaged separately from per-dataset values; F1 is not recomputed from mean Precision/Recall. "
        "Missing required data or undefined metrics remain N/A; no category drops datasets to fill missing values. "
        "PII is ordinary binary classification (PII present vs absent), not entity extraction or masking quality. "
        "Its current labels contain positives only, so AUROC is undefined. "
        "Quality uses each model's own successful samples or supplied aggregates, not sample-aligned comparisons. "
        "Latency averages the reference timings used in the per-dataset chart, including network and failure waits; "
        "single-run and four-run timing protocols differ and are reference-only. "
        "Historical PII and other models' snapshots are retained; categories need not share an evaluation date. "
        "Accuracy, F1 and Latency are visible by default; expand Precision, Recall and AUROC for details.",
    ),
    (
        "fig10_openjev_reference_v1", lambda report: external_dataset_figure(report, "v1"),
        "Per-Dataset Multi-Model Reference Comparison (v1 / Threshold 0.5)",
        "Fixed threshold 0.5. The original OpenJev is labeled 4B; OpenJev 32B and Microsoft Decision-1 have v1 summaries only. "
        "Internal models use user-supplied aggregate results; other models use their own successful samples. "
        "Sample-level alignment with OpenJev is unavailable, so this is not a fair ranking. "
        "The Rows field counts total rows, not valid samples. "
        "PII is ordinary binary classification (PII present vs absent), not extraction or masking quality; single-class AUROC is N/A. "
        "The original four models prefer complete four-run averages for timing. "
        "For local APIs in v1, when no complete four-run mean is available, elapsed/processed from a complete "
        "threshold-0.5 run is used only if log counts match the full CSV. "
        "The two added models use supplied single-run timing. Asterisks mark single-run timings, not four-run means; "
        "protocols differ. Timing includes failure waits, client and network overhead; unverifiable timing is N/A. "
        "All five ToxiGen quality metrics for 32B are now supplied; tasks with many errors reflect successful samples only. "
        "The original sample-aligned baseline and stability analyses are unchanged and exclude aggregate-only internal models. "
        "One enlarged metric chart is shown per row, with a shared model legend at the top. "
        "Accuracy, F1 and Latency are visible by default; expand Precision, Recall and AUROC for details. "
        "Latency uses a logarithmic vertical axis with actual ms/sample labels. "
        "The complete six-metric figure and each individual metric are available for download.",
    ),
    (
        "fig11_openjev_reference_v2", lambda report: external_dataset_figure(report, "v2"),
        "Per-Dataset Reference Comparison: v2 / Threshold 0.5",
        "v2 is analyzed separately, retaining only the original four models with v2 records; "
        "v1 results from the two added models are not copied into v2. "
        "Classification metrics use threshold 0.5; quality scores are not averaged across the four thresholds. "
        "Timing is the average elapsed/processed across four complete runs for the same dataset and prompt. "
        "OpenJev valid and error counts are not inferred from text containing OCR errors. "
        "Original values come from the initial legible table and are not arbitrarily corrected.",
    ),
    (
        "fig12_openjev_reference_averages", external_averages_figure,
        "Multi-Model Reference Averages (v1 / Threshold 0.5)",
        "Fixed prompt v1 and threshold 0.5; only v1 is shown. "
        "The main charts give equal weight to six jointly tested non-PII datasets. "
        "PII is ordinary binary classification (PII present vs absent), not extraction or masking quality. "
        "The two PII datasets are summarized separately, and the eight-dataset averages are supplementary only. "
        "AUROC is undefined for both the PII and all-eight-dataset scopes and remains N/A. "
        "Average Accuracy, Precision, Recall, F1, AUROC and latency are shown. "
        "Only dataset names are shared; successful samples are not aligned across models, "
        "so these results must not be ranked together with the original sample-aligned model averages. "
        "Quality averages retain the full shared dataset scope; a missing metric on any dataset makes that average N/A. "
        "Latency gives equal weight to the shared subset with available timing for every model; "
        "the figure states its scope and count. Latency is N/A only when no shared timing subset exists; "
        "datasets are never dropped separately for individual models. "
        "The three APIs use single-run v1 / 0.5 logs, without substituting other thresholds or four-run averages. "
        "OpenJev 4B retains supplied summary timing; 32B and Decision-1 use supplied single-run timing. "
        "Protocols differ, so timing is reference-only. "
        "32B's ToxiGen metrics are now complete and included normally in quality averages. "
        "All six metrics are visible in two columns and three rows: Accuracy / F1, Precision / Recall, AUROC / Latency. "
        "Narrow screens use one column; the shared model color legend appears at the top.",
    ),
    (
        "fig13_three_dataset_averages",
        three_dataset_averages_figure,
        "ToxiGen, HateCheck, AEGIS2: Multi-model Reference Averages",
        "For Fastino, Laya, TypeSafe, OpenJev 4B, OpenJev 32B and Microsoft Decision-1, "
        "Accuracy, Precision, Recall, F1, AUROC and average latency are reported for ToxiGen, HateCheck and AEGIS2, "
        "with equal weights of one third per dataset; Detect-Jailbreak and all other datasets are excluded from these means. "
        "These six models use prompt v1 at threshold 0.5, each with its own successful samples; "
        "OpenJev 4B, OpenJev 32B and Microsoft Decision-1 results are supplied internal summaries. "
        "Local model latency is elapsed/processed from each dataset's v1 / 0.5 run, averaged equally across datasets; "
        "OpenJev latency comes from the corresponding supplied summary entries. "
        "Missing quality results or complete timing for any required dataset make the corresponding mean N/A; "
        "the dataset scope is not reduced. "
        "Dynamo CPU, Dynamo GPU (A100), Dynamo GPU (L4), Mistral and Warden use user-supplied averages retained as provided. "
        "These means are not averaged again; "
        "their prompts, thresholds and timing protocols were not supplied. "
        "Unreported F1 or AUROC values are N/A; F1 is not inferred from average Precision/Recall. "
        "The latency axis uses a logarithmic scale above 1 ms. "
        "Successful samples and timing protocols are not aligned across models; this is a descriptive reference, not a fair ranking. "
        "All six metrics are visible together in a two-column grid (one column on narrow screens). "
        "Model order is fixed across panels, with light group separators; methodology can be expanded separately. "
        "Additional dataset-scope limitation: no internal dataset was available when the six models above were evaluated, "
        "so their averages use only the three shared datasets. "
        "Other model evaluations may also have included an additional internal dataset. "
        "Whether that dataset contributed to their supplied averages has not been confirmed; "
        "their dataset coverage must not be assumed identical to the six-model three-dataset scope.",
    ),
)


def reference_scope_table(rows):
    headers = ("Model", "Accuracy", "Precision", "Recall", "F1", "AUROC", "Shared-subset latency (ms)")
    body = []
    for row in rows:
        cells = ["<td>" + html.escape(NAMES[row["provider"]]) + "</td>"]
        for metric in METRICS[:-1]:
            value = row[metric]
            cells.append("<td>" + (f"{value * 100:.2f}%" if value is not None else "N/A") + "</td>")
        elapsed = row["shared_elapsed_ms"]
        cells.append("<td>" + (f"{elapsed:,.2f}" if elapsed is not None else "N/A") + "</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    datasets = rows[0]["datasets"] if rows else []
    timed = rows[0]["timing_datasets"] if rows else []
    note = (
        f"Equal weights across {len(datasets)} datasets. Latency uses {len(timed)} shared timed datasets: "
        + (", ".join(DATASET_NAMES[dataset] for dataset in timed) or "None")
        + ". Quality and timing scopes may differ."
    )
    return (
        "<p>" + html.escape(note) + '</p><div class="table-scroll"><table class="scope-table"><thead><tr>'
        + "".join("<th>" + html.escape(header) + "</th>" for header in headers)
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def english_warning(warning):
    """Translate legacy warnings without rewriting the saved evaluation data."""
    for original, translated in (
        (
            "；整份日志不用于逐配置耗时比较，不猜测并发进度的阈值归属。可核验的完整四轮均值仅用于四模型参考图。",
            "; the entire log is excluded from per-configuration timing comparisons; "
            "threshold assignments are not inferred from interleaved progress. "
            "Verifiable complete four-run averages are used only in the four-model reference figures.",
        ),
        (
            "：历史日志对应的预测文件已被新结果替换，不沿用该旧日志的耗时；质量指标仍使用新 CSV。",
            ": the prediction file associated with the historical log has been replaced by new results. "
            "Timing from that old log is not reused; quality metrics still use the new CSV.",
        ),
    ):
        warning = warning.replace(original, translated)
    return warning


def gallery_header(title):
    escaped = html.escape(title)
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>""" + escaped + """</title><style>
body{font:16px/1.7 system-ui,sans-serif;background:#f3f5f8;color:#172438;margin:0}
main{max-width:1250px;margin:auto;padding:28px}section{background:white;padding:24px;margin:24px 0}
img{width:100%;height:auto}a{color:#174c90}h1,h2{line-height:1.4}
h1{margin-bottom:6px}
.report-subtitle{margin:0;color:#526174}
.report-meta{margin:12px 0;font-size:13px;color:#526174}
.report-actions{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}
.report-actions a{padding:8px 16px;border:1px solid #174c90;border-radius:6px;background:white;text-decoration:none;font-weight:600}
.report-actions .primary-action{background:#174c90;color:white}
.report-note{padding:10px 14px;border-left:3px solid #607d8b;background:white;font-size:14px}
.pii-task-notice{margin:18px 0;padding:16px 20px;background:#FFF3CD;color:#78350F;border:1px solid #D97706;border-left:6px solid #D97706;border-radius:6px}
.pii-task-notice strong{display:block;font-size:19px;font-weight:800;line-height:1.5}
.report-resources{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start;margin:20px 0}
.report-resources>details{padding:12px 16px;background:white;border:1px solid #D8DDE3;border-radius:6px;min-width:0}
.report-resources p{font-size:14px}
summary{cursor:pointer;font-weight:600}
.model-legend{list-style:none;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.model-legend li{display:flex;gap:8px;font-size:13px;white-space:pre-line}
.legend-swatch{flex:0 0 14px;height:14px;margin-top:5px;border:1px solid #263238}
.legend-swatch.fine-tuned-swatch{background-image:repeating-linear-gradient(135deg,#172438 0,#172438 1px,transparent 1px,transparent 5px);border:2px solid #172438}
.metric-panel{margin:24px 0}.metric-downloads{font-size:14px}
.reference-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
.reference-grid .metric-panel{margin:0;min-width:0}
.table-scroll{overflow-x:auto}.scope-table{border-collapse:collapse;width:100%;font-size:14px}
.scope-table th,.scope-table td{padding:8px;border-bottom:1px solid #D8DDE3;text-align:right;white-space:nowrap}
.scope-table th:first-child,.scope-table td:first-child{text-align:left}
@media(max-width:900px){.reference-grid,.report-resources{grid-template-columns:1fr}}
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
        "warnings": [english_warning(warning) for warning in report.get("warnings", [])],
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
        if comparison.get("internal_source"):
            manifest["external_comparison"]["internal_source"] = comparison["internal_source"]
        if comparison.get("fine_tuned_models"):
            manifest["external_comparison"]["fine_tuned_models"] = comparison["fine_tuned_models"]
    with plt.rc_context(STYLE), PdfPages(output_dir / "paper_figures.pdf") as bundle:
        for stem, builder, title, caption in figures:
            if stem in {item[0] for item in EXTERNAL_FIGURES} and stem != "fig11_openjev_reference_v2":
                note = fine_tuned_note(report)
                if note:
                    caption += " " + note.replace("\n", " ")
            scoped_title = title
            if stem in (
                "fig03_threshold_sensitivity", "fig05_dataset_metrics", "fig06_model_averages",
                "fig07_repeat_consistency_v1", "fig08_repeat_consistency_v2", "fig09_repeat_consistency_averages",
            ):
                scoped_title += " (" + ", ".join(NAMES[provider] for provider in report["providers"]) + ")"
            elif stem == "fig11_openjev_reference_v2":
                providers = external_reference_providers(report["external_comparison"], "v2")
                scoped_title += " (" + ", ".join(NAMES[provider] for provider in providers) + ")"
            fig = builder(report)
            panels = []
            try:
                if scoped_title != title:
                    fig.suptitle(scoped_title, fontsize=15, y=0.98)
                for extension in ("png", "svg", "pdf"):
                    fig.savefig(output_dir / f"{stem}.{extension}", dpi=dpi)
                bundle.savefig(fig)
                if stem in (
                    "fig10_openjev_reference_v1", "fig12_openjev_reference_averages",
                    "fig13_three_dataset_averages", "fig14_category_reference_averages",
                ):
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    axes = dict(zip(METRICS, fig.axes))
                    panel_order = REFERENCE_AVERAGE_PANEL_ORDER if stem == "fig12_openjev_reference_averages" else METRICS
                    for metric in panel_order:
                        ax = axes[metric]
                        panel_stem = f"{stem}_{metric}"
                        bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.15)
                        for extension in ("png", "svg", "pdf"):
                            fig.savefig(output_dir / f"{panel_stem}.{extension}", dpi=dpi, bbox_inches=bounds)
                        panels.append({
                            "metric": metric, "title": ax.get_title(),
                            "files": {ext: f"{panel_stem}.{ext}" for ext in ("png", "svg", "pdf")},
                        })
            finally:
                plt.close(fig)
            entry = {
                "name": stem, "title": scoped_title, "caption": caption,
                "files": {ext: f"{stem}.{ext}" for ext in ("png", "svg", "pdf")},
            }
            if panels:
                entry["panels"] = panels
            manifest["figures"].append(entry)
    write_latex_table(report, output_dir / "table01_baseline.tex")
    scope_averages = []
    if report.get("external_comparison"):
        category_averages = category_reference_averages(report)
        with (output_dir / "category_reference_averages.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(category_averages[0]))
            writer.writeheader()
            writer.writerows({**row, "datasets": ",".join(row["datasets"])} for row in category_averages)
        manifest["external_comparison"]["category_averages_csv"] = "category_reference_averages.csv"
        if report["external_comparison"].get("fine_tuned_models"):
            providers = report["external_comparison"]["fine_tuned_models"]
            fine_tuned_runs = [
                row for row in report["external_comparison"]["runs"] if row["provider"] in providers
            ]
            with (output_dir / "fine_tuned_model_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(fine_tuned_runs[0]))
                writer.writeheader()
                writer.writerows(fine_tuned_runs)
            averages = [
                {
                    **row, "model": providers[row["provider"]]["model_id"],
                    "display_name": providers[row["provider"]]["display_name"],
                }
                for row in category_averages if row["provider"] in providers
            ]
            averages.extend(
                {
                    "category": "non_pii", **row, "model": providers[row["provider"]]["model_id"],
                    "display_name": providers[row["provider"]]["display_name"], "threshold": 0.5,
                }
                for row in report["external_comparison"]["overview"]
                if row["provider"] in providers and row["prompt"] == "v1"
            )
            with (output_dir / "fine_tuned_model_averages.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(averages[0]))
                writer.writeheader()
                writer.writerows({**row, "datasets": ",".join(row["datasets"])} for row in averages)
        scope_averages = v1_reference_scope_averages(report)
        with (output_dir / "v1_reference_scope_averages.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scope_averages[0]))
            writer.writeheader()
            writer.writerows({
                **row, "datasets": ",".join(row["datasets"]),
                "timing_datasets": ",".join(row["timing_datasets"]),
            } for row in scope_averages)
        manifest["external_comparison"]["v1_scope_averages_csv"] = "v1_reference_scope_averages.csv"
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    cards, linked_figures = {}, []
    pii_notice = (
        '<aside class="pii-task-notice" aria-label="PII task definition">'
        "<strong>PII = ordinary binary classification (PII present vs absent).</strong>"
        "Not entity extraction or masking quality. "
        "The current datasets contain positives only, so AUROC is undefined.</aside>"
    )
    for figure in manifest["figures"]:
        downloads = " | ".join(
            f'<a href="{filename}">{extension.upper()}</a>'
            for extension, filename in figure["files"].items()
        )
        if figure["name"].startswith(("fig07_", "fig08_", "fig09_")):
            linked_figures.append(
                f'<p><a href="{figure["files"]["svg"]}">{html.escape(figure["title"])}</a>'
                + " (" + downloads + ")</p>"
            )
            continue
        image_source = figure["files"]["svg"]
        if figure["name"] == "fig01_paired_quality":
            image_data = base64.b64encode((output_dir / figure["files"]["png"]).read_bytes()).decode("ascii")
            image_source = "data:image/png;base64," + image_data
        card_title = figure["title"]
        if figure["name"] in ("fig01_paired_quality", "fig02_quality_cost_tradeoff"):
            card_title += " (" + ", ".join(NAMES[provider] for provider in report["providers"]) + ")"
        content = (
            f'<img src="{image_source}" alt="{html.escape(card_title)}">'
            + ("" if figure["name"] == "fig12_openjev_reference_averages"
               else "<p>" + html.escape(figure["caption"]) + "</p>")
            + "<p>" + downloads + "</p>"
        )
        if figure.get("panels"):
            focused = figure["name"] == "fig13_three_dataset_averages"
            averaged = figure["name"] == "fig12_openjev_reference_averages"
            legend = "" if focused else '<ul class="model-legend">' + "".join(
                '<li><span class="legend-swatch'
                + (" fine-tuned-swatch" if provider in report["external_comparison"].get("fine_tuned_models", {}) else "")
                + '" style="background-color:' + COLORS[provider] + '"></span>'
                + html.escape(model_label(report["external_comparison"], provider)) + "</li>"
                for provider in report["external_comparison"]["providers"]
            ) + "</ul>"
            primary, detailed, overview_panels = [], [], []
            for panel in figure["panels"]:
                links = " | ".join(
                    f'<a href="{filename}">{extension.upper()}</a>'
                    for extension, filename in panel["files"].items()
                )
                timing_note = (
                    '<p class="metric-downloads">* Single-run timing (v1 / threshold 0.5), '
                    'not a four-run mean; timing protocols differ.</p>'
                    if figure["name"] == "fig10_openjev_reference_v1" and panel["metric"] == "mean_elapsed_ms"
                    else ""
                )
                panel_card = (
                    '<div class="metric-panel" data-metric="' + panel["metric"] + '">'
                    + f'<img src="{panel["files"]["svg"]}" alt="{html.escape(panel["title"])}" loading="lazy">'
                    + timing_note
                    + '<p class="metric-downloads">' + links + "</p></div>"
                )
                if focused or averaged:
                    overview_panels.append(panel_card)
                elif panel["metric"] in ("accuracy", "f1", "mean_elapsed_ms"):
                    primary.append(panel_card)
                else:
                    detailed.append(panel_card)
            if focused:
                content = (
                    "<p>Computed averages: ToxiGen / HateCheck / AEGIS2 | equal dataset weights.<br>"
                    "Supplied model averages may have additional dataset coverage; see Methodology and data scope.<br>"
                    "Reference only: successful samples and timing protocols differ; not a fair ranking.<br>"
                    "N/A = not supplied or required source data missing, not zero.</p>"
                    + '<div class="reference-grid">' + "".join(overview_panels) + "</div>"
                    + "<details><summary>Methodology and data scope</summary><p>"
                    + html.escape(figure["caption"]) + "</p></details>"
                    + "<p>Download all six metrics: " + downloads
                    + ' | <a href="three_dataset_averages.csv">CSV</a></p>'
                )
            elif averaged:
                content = (
                    legend + '<div class="reference-grid">' + "".join(overview_panels) + "</div>"
                    + "<p>Download all six metrics: " + downloads + "</p>"
                )
            else:
                content = (
                    legend + "".join(primary)
                    + "<details><summary>Detailed metrics: Precision, Recall and AUROC</summary>"
                    + "".join(detailed) + "</details><p>Download all six metrics: " + downloads + "</p>"
                    + "<details><summary>Methodology and data scope</summary><p>"
                    + html.escape(figure["caption"]) + "</p></details>"
                )
        if figure["name"] in {item[0] for item in EXTERNAL_FIGURES} and figure["name"] != "fig11_openjev_reference_v2":
            note = fine_tuned_note(report)
            if note:
                content = '<p class="report-note">' + html.escape(note).replace("\n", "<br>") + "</p>" + content
        if figure["name"] in (
            "fig14_category_reference_averages", "fig10_openjev_reference_v1", "fig11_openjev_reference_v2",
        ):
            content = pii_notice + content
        if figure["name"] == "fig12_openjev_reference_averages":
            content += (
                "<h3>PII binary classification — two-dataset averages (v1 / threshold 0.5)</h3>"
                + pii_notice +
                "<p>Ordinary binary classification: PII present vs absent, not entity extraction or masking quality. "
                "The current datasets contain positives only, so AUROC is undefined. "
                "These results are separate from the non-PII main chart.</p>"
                + reference_scope_table([row for row in scope_averages if row["scope"] == "pii"])
                + "<details><summary>All eight datasets — supplementary averages (PII included)</summary>"
                "<p>Supplementary only: combining PII and non-PII tasks can inflate classification averages. "
                "AUROC is N/A because it is undefined for the two PII datasets; undefined values are not dropped.</p>"
                + reference_scope_table([row for row in scope_averages if row["scope"] == "all"])
                + '</details><p><a href="v1_reference_scope_averages.csv">'
                "Download v1 scope averages CSV (non-PII / PII / all datasets)</a></p>"
                + "<details><summary>Methodology and data scope</summary><p>"
                + html.escape(figure["caption"]) + "</p></details>"
            )
        if figure["name"] in (
            "fig01_paired_quality", "fig02_quality_cost_tradeoff", "fig03_threshold_sensitivity", "fig05_dataset_metrics",
            "fig06_model_averages", "fig11_openjev_reference_v2",
        ) or (
            figure["name"] == "fig04_evaluation_coverage" and not report.get("external_comparison")
        ):
            card = (
                "<section><details><summary>" + html.escape(card_title) + "</summary>"
                + content + "</details></section>"
            )
        else:
            card = "<section><h2>" + html.escape(card_title) + "</h2>" + content + "</section>"
        if figure["name"] == "fig14_category_reference_averages":
            content_link = (
                '<p><a href="category_reference_comparison.html">Open dataset-category comparison page</a> | '
                '<a href="category_reference_averages.csv">Download category averages CSV</a></p>'
            )
            card = card.replace("</h2>", "</h2>" + content_link, 1)
            page = gallery_header(figure["title"])
            page += '<p><a href="index.html">Back to the full figure gallery</a></p>'
            page += "<p>Data snapshot (UTC): " + html.escape(report["generated_at"]) + "</p>"
            standalone_card = card
            for panel in figure["panels"]:
                image_data = base64.b64encode((output_dir / panel["files"]["png"]).read_bytes()).decode("ascii")
                standalone_card = standalone_card.replace(
                    f'src="{panel["files"]["svg"]}"', f'src="data:image/png;base64,{image_data}"',
                )
            page += standalone_card + "</main></html>\n"
            (output_dir / "category_reference_comparison.html").write_text(page, encoding="utf-8")
        if figure["name"] == "fig13_three_dataset_averages":
            averages = three_dataset_averages(report)
            with (output_dir / "three_dataset_averages.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(averages[0]))
                writer.writeheader()
                writer.writerows({**row, "datasets": ",".join(row["datasets"])} for row in averages)
            page = gallery_header("Three-dataset Multi-model Reference Averages")
            page += '<p><a href="index.html">Back to the full figure gallery</a> | '
            page += '<a href="three_dataset_averages.csv">Download three-dataset averages CSV</a></p>'
            page += "<p>Data snapshot (UTC): " + html.escape(report["generated_at"]) + "</p>"
            standalone_card = card
            for panel in figure["panels"]:
                image_data = base64.b64encode((output_dir / panel["files"]["png"]).read_bytes()).decode("ascii")
                standalone_card = standalone_card.replace(
                    f'src="{panel["files"]["svg"]}"', f'src="data:image/png;base64,{image_data}"',
                )
            page += standalone_card + "</main></html>\n"
            (output_dir / "three_dataset_comparison.html").write_text(page, encoding="utf-8")
        cards[figure["name"]] = card
    snapshot = datetime.fromisoformat(report["generated_at"]).astimezone(timezone.utc)
    gallery = gallery_header("Safety Model Evaluation")
    gallery += (
        '<p class="report-subtitle">Classification quality, latency, coverage, and repeat-call consistency.</p>'
        '<p class="report-meta">Data snapshot: <time datetime="' + html.escape(report["generated_at"]) + '">'
        + snapshot.strftime("%Y-%m-%d %H:%M UTC") + "</time></p>"
        '<nav class="report-actions" aria-label="Report actions">'
        '<a class="primary-action" href="paper_figures.pdf">Download Figures (PDF)</a>'
        '<a href="../comparison.html">View Evaluation Report</a></nav>'
        '<p class="report-note">Reference comparisons use different successful samples and timing protocols; '
        'they are not fair rankings. PII is ordinary binary classification (present vs absent), not masking quality.</p>'
        '<div class="report-resources"><details><summary>Data &amp; Downloads</summary>'
        '<p>PDF and SVG are vector formats; PNG defaults to 300 DPI.</p>'
        '<p><a href="table01_baseline.tex">Download the LaTeX comparison table</a></p>'
    )
    gallery += (
        '<p>Repeat-call consistency data: <a href="../stability.csv">Per-dataset CSV</a> | '
        '<a href="../stability_paired.csv">Shared-sample CSV</a> | '
        '<a href="../stability_overview.csv">Model averages CSV</a></p>'
    )
    if report.get("external_comparison"):
        gallery += (
            '<p>OpenJev internal summaries (not a sample-aligned baseline): '
            '<a href="../openjev_summary.csv">Summary data CSV</a> | '
            '<a href="../external_comparison_runs.csv">Reference details CSV</a> | '
            '<a href="../external_comparison_overview.csv">Reference averages CSV</a></p>'
            '<p><a href="three_dataset_comparison.html">'
            'View separately: ToxiGen, HateCheck, AEGIS2 Multi-model Reference Averages</a></p>'
        )
        if report["external_comparison"].get("internal_source_rows"):
            gallery += (
                '<p>Additional internal models (v1 / 0.5 only): '
                '<a href="../internal_model_summary.csv">OpenJev 32B / Microsoft Decision-1 summary CSV</a></p>'
            )
        if report["external_comparison"].get("fine_tuned_models"):
            gallery += (
                '<p>Fine-tuned model: <a href="fine_tuned_model_runs.csv">Per-dataset summary CSV</a> | '
                '<a href="fine_tuned_model_averages.csv">Category and six-dataset averages CSV</a></p>'
            )
    gallery += (
        '</details><details><summary>Methodology &amp; Limitations</summary>'
        "<p>Model identities were supplied by the user; TypeSafe's resolved model must be confirmed by an API response.</p>"
    )
    if report.get("external_comparison") and report["external_comparison"].get("internal_source_rows"):
        gallery += (
            '<p>Additional internal models have v1 / 0.5 summaries only. '
            "OpenJev 32B's ToxiGen quality metrics are now complete. Request errors are substantial; "
            'quality computed only on successful samples does not represent overall performance. '
            'The original models prefer complete four-run means; v1 API gaps may use verified single-run logs. '
            'The added internal models use supplied single-run timings. Protocols differ; timings are reference-only.</p>'
        )
    gallery += "</details></div>"
    card_order = list(cards)
    if report.get("external_comparison"):
        leading = [
            "fig13_three_dataset_averages", "fig14_category_reference_averages", "fig10_openjev_reference_v1",
            "fig12_openjev_reference_averages", "fig04_evaluation_coverage",
        ]
        card_order = leading + [name for name in card_order if name not in leading]
    gallery += "".join(cards[name] for name in card_order)
    stability_title = "Repeat-Call Consistency (" + ", ".join(NAMES[provider] for provider in report["providers"]) + ")"
    gallery += (
        "<section><details><summary>" + html.escape(stability_title) + "</summary>"
        + "".join(linked_figures) + "</details></section>"
    )
    if manifest["warnings"]:
        gallery += "<section><details><summary>Data Warnings</summary>" + "".join(
            "<p>" + html.escape(warning) + "</p>" for warning in manifest["warnings"]
        ) + "</details></section>"
    gallery += "</main></html>\n"
    (output_dir / "index.html").write_text(gallery, encoding="utf-8")
    return manifest
