"""Offline presentation for verified, matched held-out benchmark results."""

import csv
import html
import json
import re
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from scripts.publish_corrected_comparison import render_html, table
from scripts.rerun_classification import digest


METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "auprc")
NAMES = {"fastino.ai": "GLiDE", "typesafe.ai": "Jev", "laya-ai.com": "Laya",
         "microsoft-decision-1": "Microsoft Decision-1"}
COLORS = {"fastino.ai": "#2166ac", "typesafe.ai": "#666666", "laya-ai.com": "#d97706",
          "microsoft-decision-1": "#008568"}


def averages(report):
    """Equal dataset weights; never silently shrink the declared scope."""
    datasets = set(report["sources"])
    rows = report["classification"]
    result = []
    for provider, prompt in sorted({(r["provider"], r["prompt"]) for r in rows}):
        selected = [r for r in rows if r["provider"] == provider and r["prompt"] == prompt]
        if len(selected) != len(datasets) or {r["dataset"] for r in selected} != datasets:
            raise ValueError("Macro comparison requires the full declared dataset scope exactly once")
        result.append({"provider": provider, "prompt": prompt, "datasets": ",".join(sorted(datasets)),
                       "rows": sum(r["valid"] for r in selected),
                       **{m: mean(r[m] for r in selected) if all(r[m] is not None for r in selected) else None for m in METRICS}})
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def publish(root):
    root = Path(root)
    report = json.loads((root / "summary.json").read_text())
    if not report.get("same_successful_rows_verified"):
        raise ValueError("Charts require verified matching successful rows")
    rows, pii = report["classification"], report["pii_extraction"]
    if not rows:
        raise ValueError("No completed classification provider to plot")
    macro = averages(report)
    datasets = list(report["sources"])
    providers = sorted({r["provider"] for r in rows})
    scope = " / ".join(f"{d} (n={report['sources'][d]['rows']:,})" for d in datasets)
    note = "Official held-out partitions; matching rows verified. Training overlap has not been audited."
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for filename, records in (("macro_averages.csv", macro), ("classification_metrics.csv", rows),
                              ("threshold_metrics.csv", report["thresholds"]), ("pii_extraction.csv", pii),
                              ("coverage.csv", report["coverage"])):
        write_csv(root / filename, records)
    timing_columns = ("provider", "dataset", "prompt", "workers", "requests_per_second", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms")
    timing = [{k: r[k] for k in timing_columns} for r in rows]
    write_csv(root / "request_timing.csv", timing)
    write_csv(figures / "three_dataset_averages.csv", [r for r in macro if r["prompt"] == "v1"])
    manifest = []
    with PdfPages(figures / "heldout_figures.pdf") as pdf:
        def save(fig, name, caption):
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(figures / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            manifest.append({"name": name, "caption": caption,
                             "files": {suffix: {"path": f"{name}.{suffix}", "sha256": digest(figures / f"{name}.{suffix}")}
                                       for suffix in ("png", "svg", "pdf")}})
        for prompt in ("v1", "v2"):
            selected = [r for r in macro if r["prompt"] == prompt]
            fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
            for ax, metric in zip(axes.flat, METRICS):
                for i, row in enumerate(selected):
                    value = row[metric]
                    if value is None:
                        ax.text(.02, i, "N/A", va="center")
                    else:
                        ax.barh(i, value, color=COLORS[row["provider"]])
                        ax.text(value + .012, i, f"{value:.4f}", va="center", fontsize=10)
                ax.set_yticks(range(len(selected)), [NAMES[r["provider"]] for r in selected])
                ax.set_ylim(len(selected) - .4, -.6)
                ax.set_xlim(0, 1.16)
                ax.set_xticks([0, .25, .5, .75, 1])
                ax.set_title("AUPRC (average precision)" if metric == "auprc" else metric.upper())
                ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"Held-out classification — {prompt}, threshold 0.5\n{scope}\nEqual dataset weights; ToxiGen excluded. Training overlap not audited.", fontsize=11)
            name = "fig13_three_dataset_averages" if prompt == "v1" else "fig02_macro_averages_v2"
            save(fig, name, f"{prompt} / 0.5; equal weights over {scope}. Unlike PR #20, ToxiGen is excluded and Jailbreak Classification is included. {note}")
            fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
            for ax, metric in zip(axes.flat, METRICS):
                values = np.array([[next(r[metric] for r in rows if r["provider"] == provider and r["dataset"] == dataset and r["prompt"] == prompt)
                                    for dataset in datasets] for provider in providers], dtype=float)
                ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                for i in range(len(providers)):
                    for j in range(len(datasets)):
                        value = values[i, j]
                        ax.text(j, i, f"{value:.4f}" if np.isfinite(value) else "N/A", ha="center", va="center", color="white" if value > .65 else "black")
                ax.set_xticks(range(len(datasets)), [d.replace("-", "\n") for d in datasets], fontsize=9)
                ax.set_yticks(range(len(providers)), [NAMES[p] for p in providers], fontsize=9)
                ax.set_title(metric.upper())
            fig.suptitle(f"Per-dataset held-out quality — {prompt}, threshold 0.5\n{note}", fontsize=12)
            save(fig, f"fig03_dataset_quality_{prompt}", f"All six quality metrics on matched successful rows, {prompt}. {scope}.")
        fig, axes = plt.subplots(2, len(datasets), figsize=(15, 8), squeeze=False, layout="constrained")
        for i, prompt in enumerate(("v1", "v2")):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                for provider in providers:
                    selected = sorted([r for r in report["thresholds"] if r["provider"] == provider and r["dataset"] == dataset and r["prompt"] == prompt], key=lambda r: r["threshold"])
                    ax.plot([r["threshold"] for r in selected], [r["f1"] for r in selected], marker="o", color=COLORS[provider], label=NAMES[provider])
                ax.set_title(f"{dataset} / {prompt}")
                ax.set_ylim(0, 1.03)
                ax.set_xlabel("Threshold (offline)")
                ax.set_ylabel("F1")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Threshold sensitivity — one live probability per row/prompt\nPrimary threshold stays 0.5. These are not repeated inferences or tuned held-out winners.", fontsize=12)
        save(fig, "fig04_threshold_sensitivity", "Thresholds 0.5–0.8 derived offline from saved probabilities. No threshold selected using these test results.")
        fig, ax = plt.subplots(figsize=(14, 10), layout="constrained")
        for i, row in enumerate(timing):
            ax.plot([row["latency_p50_ms"], row["latency_p95_ms"]], [i, i], marker="o", color=COLORS[row["provider"]])
        ax.set_yticks(range(len(timing)), [f"{NAMES[r['provider']]} / {r['dataset']} / {r['prompt']} / workers={r['workers']} / cap={r['requests_per_second'] or 'none'} rps" for r in timing], fontsize=8)
        ax.set_xlabel("Wall-clock request latency, p50 → p95 (ms), including retries and pacing")
        ax.set_title("Timing under recorded workloads — not an unconstrained speed ranking\nConcurrency and request caps differ; no historical latency or speedup comparison.")
        save(fig, "fig05_request_timing", "Timing includes retries and rate-limit waits. Microsoft is capped at 4 rps; worker counts/caps are in request_timing.csv. Do not rank deployment speed from this chart.")
        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        for i, (metric, color) in enumerate((("typed_span_f1", "#2166ac"), ("boundary_f1", "#008568"), ("exact_document_match", "#777777"))):
            positions = np.arange(len(pii)) + (i - 1) * .24
            ax.bar(positions, [r[metric] for r in pii], width=.23, color=color, label=metric.replace("_", " "))
            for x, r in zip(positions, pii):
                ax.text(x, r[metric] + .02, f"{r[metric]:.4f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(pii)), [f"{r['dataset']}\nvalidation n={r['expected']}" for r in pii])
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=9)
        ax.set_title("GLiNER PII extraction — separate from GLiDE classification\nExact typed spans, boundary spans and whole-document matches; failures count as misses.")
        save(fig, "fig06_pii_extraction", "Model: fastino/gliner2.5-multi-v1. Full Nano/Mini validation partitions. No document-detection, leakage or masking-success substitution for entity extraction quality.")
        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        for i, provider in enumerate(providers):
            selected = [r for r in rows if r["provider"] == provider]
            valid, expected, errors = (sum(r[key] for r in selected) for key in ("valid", "expected", "errors"))
            ax.barh(i, valid, color=COLORS[provider])
            ax.text(valid + max(1, expected * .01), i, f"{valid:,}/{expected:,} successful; errors={errors}", va="center")
        ax.set_yticks(range(len(providers)), [NAMES[p] for p in providers])
        ax.set_xlim(0, max(sum(r["expected"] for r in rows if r["provider"] == p) for p in providers) * 1.55)
        ax.set_xlabel("Successful live requests across both prompts")
        ax.set_title("Held-out classification coverage\nError rows are not counted as negative predictions; missing providers are excluded.")
        save(fig, "fig07_coverage", "Each provider attempts the same declared datasets/prompts. Coverage is request count, not number of unique texts or independent repeat runs.")
    metadata = {"source_summary_sha256": digest(root / "summary.json"), "scope": datasets,
                "aggregation": "equal weight per dataset, separate prompts", "figures": manifest}
    (figures / "figure_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    macro_text = "\n\n## Equal-weight held-out averages\n\n![Held-out v1 summary](figures/fig13_three_dataset_averages.png)\n\n" + table(macro, ["provider", "prompt", *METRICS]) + "\n\n"
    # Put the finished chart before the detailed data tables, and keep refreshes
    # idempotent whether the previous summary was in the middle or at the end.
    readme = re.sub(r"\n\n## Equal-weight held-out averages\n.*?(?=\n## |\Z)", "", (root / "README.md").read_text(), flags=re.S)
    readme = readme.replace("\n## Provider coverage", macro_text + "\n## Provider coverage", 1)
    (root / "README.md").write_text(readme)
    style = '<style>body{font:16px system-ui;line-height:1.55;margin:32px auto;max-width:1400px;padding:0 24px;color:#183044}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #ccd6df;text-align:left}th{background:#edf3f8}img{max-width:100%;height:auto}a{color:#155da6}h1,h2{line-height:1.2}code{background:#edf3f8;padding:2px}</style>'
    header = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Held-out benchmark comparison</title>' + style
    nav = '<p><a href="figures/index.html">All eight figures</a> · <a href="figures/heldout_figures.pdf">Combined PDF</a> · <a href="macro_averages.csv">Macro CSV</a> · <a href="classification_metrics.csv">Per-dataset CSV</a> · <a href="summary.json">Auditable JSON</a></p>'
    (root / "comparison.html").write_text(header + '<body>' + nav + render_html(readme) + '</body></html>')
    gallery = header + '<body><h1>Held-out benchmark figures</h1><p>' + html.escape(scope + ". " + note) + '</p><p><a href="../comparison.html">Full report</a> · <a href="heldout_figures.pdf">Combined PDF</a> · <a href="three_dataset_averages.csv">v1 macro CSV</a></p>'
    for item in manifest:
        name = item["name"]
        gallery += f'<h2>{html.escape(name)}</h2><p>{html.escape(item["caption"])}</p><img src="{name}.png" alt="{html.escape(item["caption"])}"><p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p>'
    (figures / "index.html").write_text(gallery + '</body></html>')
    (figures / "three_dataset_comparison.html").write_text(header + '<body><h1>Held-out summary — v1 / 0.5</h1><p>' + html.escape(scope + ". " + note) + '</p>' +
        render_html(table([r for r in macro if r["prompt"] == "v1"], ["provider", *METRICS])) +
        '<img src="fig13_three_dataset_averages.png" alt="Held-out v1 quality"><p><a href="../comparison.html">Full report</a> · <a href="three_dataset_averages.csv">CSV</a></p></body></html>')
    return metadata
