"""Publish the active comparison from verified corrected reruns, without API calls."""

import argparse
import csv
import html
import json
import re
import shutil
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from sklearn.metrics import average_precision_score

from benchmark.comparison import fingerprint
from benchmark.evaluation import evaluate_predictions
from scripts.report_classification_rerun import build_report, valid_rows
from scripts.rerun_classification import digest


FOCUSED = ("toxigen", "hatecheck", "aegis2")
QUALITY = ("accuracy", "precision", "recall", "f1", "auroc", "auprc")
NAMES = {"fastino.ai": "GLiDE (corrected)", "laya-ai.com": "Laya (corrected)",
         "historical_jev": "Jev (historical)"}
COLORS = {"fastino.ai": "#2166ac", "laya-ai.com": "#008568", "historical_jev": "#777777"}
SCOPE = ("Historical prepared rows; not the new held-out split policy. "
         "v1 / threshold 0.5 unless stated. Jev is a saved historical reference, not a rerun.")


def macro_rows(rows, datasets=FOCUSED):
    result = []
    identities = sorted({(r["provider"], r["prompt"]) for r in rows})
    for provider, prompt in identities:
        selected = [r for r in rows if r["provider"] == provider and r["prompt"] == prompt and r["dataset"] in datasets]
        if {r["dataset"] for r in selected} != set(datasets) or len(selected) != len(datasets):
            continue
        result.append({"provider": provider, "prompt": prompt, "datasets": ",".join(datasets),
                       "paired_rows": sum(r["paired_rows"] for r in selected),
                       **{m: mean(r[m] for r in selected) if all(r[m] is not None for r in selected) else None for m in QUALITY}})
    return result


def collect_rows(report, root, historical_dir):
    rows, thresholds, timings, pii = [], [], [], []
    seen_jev = {}
    # Build one common-success population per dataset/prompt before any quality
    # comparison. Request-size failures stay visible in coverage, never as labels.
    frames, shared_ids = {}, {}
    for run in report["runs"]:
        if run["result"]["pii_detection_only"]:
            continue
        paired = run.get("paired_baseline")
        if not paired:
            raise ValueError(f"Missing historical paired reference: {run['dataset']}/{run['prompt_version']}")
        new_path = root / run["file"]
        old_path = historical_dir / f"{run['dataset']}_{run['prompt_version']}_t0.5.csv"
        if digest(new_path) != run["result"]["file_sha256"] or digest(old_path) != run["historical_jev_file_sha256"]:
            raise ValueError("Source digest mismatch")
        fresh, old = (valid_rows(pd.read_csv(p, dtype={"sample_id": str})) for p in (new_path, old_path))
        ids = (set(fresh.index) & set(old.index)) - set(paired["excluded_legacy_nul_rows"])
        key = (run["dataset"], run["prompt_version"])
        shared_ids[key] = shared_ids[key] & ids if key in shared_ids else ids
        frames[(run["provider"], *key)] = (fresh, old)
    for run in report["runs"]:
        result = run["result"]
        provider, dataset, prompt = run["provider"], run["dataset"], run["prompt_version"]
        common = {"provider": provider, "dataset": dataset, "prompt": prompt}
        timings.append({**common, "workers": run["workers"], "mean_ms": result["latency_mean_ms"],
                        "p50_ms": result["latency_p50_ms"], "p95_ms": result["latency_p95_ms"],
                        "valid": result["valid"], "errors": result["errors"], "expected": result["expected_rows"]})
        if result["pii_detection_only"]:
            pii.append({**common, "detection_rate": result["threshold_metrics"]["0.5"]["recall"],
                        "valid": result["valid"], "expected": result["expected_rows"]})
            continue
        paired = run["paired_baseline"]
        key = (dataset, prompt)
        fresh, old = frames[(provider, *key)]
        ids = sorted(shared_ids[key])
        if not ids:
            raise ValueError(f"No common successful rows: {dataset}/{prompt}")
        for sample_id in ids:
            left, right = fresh.loc[sample_id], old.loc[sample_id]
            if int(left["label"]) != int(right["label"]) or fingerprint(str(left["text"])) != fingerprint(str(right["text"])):
                raise ValueError(f"Reference label/text mismatch for sample {sample_id}")
        signature = tuple((i, fingerprint(str(fresh.loc[i, "text"])), int(fresh.loc[i, "label"])) for i in ids)
        key = (dataset, prompt)
        if key in seen_jev and seen_jev[key] != signature:
            raise ValueError("Providers have different paired rows; a shared Jev baseline would be misleading")
        for identity, frame in ((provider, fresh), ("historical_jev", old)):
            if identity == "historical_jev" and key in seen_jev:
                continue
            aligned = frame.loc[ids]
            labels = aligned["label"].astype(int)
            metric = evaluate_predictions(labels, aligned["prediction"].astype(int), aligned["score"].astype(float))
            rows.append({**common, "provider": identity, **metric,
                         "auprc": float(average_precision_score(labels, aligned["score"].astype(float))) if labels.nunique() == 2 else None,
                         "paired_rows": len(ids), "excluded_nul_rows": len(paired["excluded_legacy_nul_rows"])})
        seen_jev[key] = signature
        for threshold, metric in result["threshold_metrics"].items():
            thresholds.append({**common, "threshold": float(threshold), **metric})
    return rows, thresholds, timings, pii


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows, columns):
    def value(v):
        return "N/A" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)] +
                     ["| " + " | ".join(value(row.get(k)) for k in columns) + " |" for row in rows])


def render_html(markdown):
    def inline(text):
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    lines = markdown.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("|"):
            cells = []
            while index < len(lines) and lines[index].startswith("|"):
                cells.append([v.strip() for v in lines[index].strip("|").split("|")])
                index += 1
            blocks.append('<div style="overflow:auto"><table><tr>' + ''.join('<th>' + inline(v) + '</th>' for v in cells[0]) + '</tr>' +
                          ''.join('<tr>' + ''.join('<td>' + inline(v) + '</td>' for v in row) + '</tr>' for row in cells[2:]) + '</table></div>')
            continue
        image = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)
        if image:
            blocks.append(f'<img alt="{html.escape(image[1])}" src="{html.escape(image[2])}">')
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(f'<h{level}>' + inline(line[level:].strip()) + f'</h{level}>')
        elif line:
            blocks.append('<p>' + inline(line) + '</p>')
        index += 1
    return '\n'.join(blocks)


def metric_figure(rows, title, pending=False):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    for ax, metric in zip(axes.flat, QUALITY):
        for i, row in enumerate(rows):
            value = row[metric]
            if value is not None:
                ax.barh(i, value, color=COLORS[row["provider"]])
                ax.text(value + .012, i, f"{value:.4f}", va="center", fontsize=10)
        labels = [NAMES[r["provider"]] for r in rows]
        if pending:
            labels.append("Laya — pending full rerun")
            ax.text(.03, len(rows), "No completed-suite result", va="center", color="#777777")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels) - .4, -.6)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_title("AUPRC (average precision)" if metric == "auprc" else metric.upper())
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title + "\nCorrected Noul vs saved historical Jev; historical prepared rows, not new holdouts."
                 "\nEqual weight per dataset; quality only. Latency is shown separately.", fontsize=11)
    return fig


def publish(root, output, historical_dir, reference_dir):
    report = build_report(root, historical_dir)
    rows, thresholds, timings, pii = collect_rows(report, root, historical_dir)
    averages = macro_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    pending = "laya-ai.com" not in report["completed_providers"]
    sources = {"rerun": str(root), "source_summary_sha256": digest(root / "summary.json"),
               "score_field": report["score_field"], "scope": SCOPE,
               "provider_status": report.get("provider_status", {}), "runs": rows,
               "thresholds": thresholds, "timing": timings, "pii_detection_only": pii,
               "three_dataset_averages": averages}
    (output / "comparison.json").write_text(json.dumps(sources, indent=2, allow_nan=False) + "\n")
    for name, records in (("external_comparison_runs.csv", rows), ("external_comparison_overview.csv", averages),
                          ("threshold_metrics.csv", thresholds), ("request_timing.csv", timings), ("pii_detection.csv", pii)):
        write_csv(output / name, records)
    write_csv(figures / "three_dataset_averages.csv", [r for r in averages if r["prompt"] == "v1"])
    supplied = []
    with (reference_dir / "figures/three_dataset_averages.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["provider"] not in ("fastino.ai", "laya-ai.com", "typesafe.ai"):
                supplied.append({**row, "evidence": "supplied aggregate; matching rows and timing conditions unverified"})
    write_csv(figures / "supplied_reference_averages.csv", supplied)
    for name in ("openjev_summary.csv", "internal_model_summary.csv"):
        shutil.copyfile(reference_dir / name, output / name)

    manifest = []
    with PdfPages(figures / "paper_figures.pdf") as pdf:
        def save(fig, name, caption):
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(figures / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            manifest.append({"name": name, "caption": caption, "png_sha256": digest(figures / f"{name}.png")})
        for prompt in ("v1", "v2"):
            selected = [r for r in averages if r["prompt"] == prompt]
            name = "fig13_three_dataset_averages" if prompt == "v1" else "fig02_three_dataset_averages_v2"
            save(metric_figure(selected, f"ToxiGen / HateCheck / AEGIS2 — {prompt}, threshold 0.5", pending), name,
                 f"{prompt}: macro means over exactly three datasets. Corrected Noul versus paired historical Jev. {SCOPE}")
        for prompt in ("v1", "v2"):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
            for ax, dataset in zip(axes.flat, sorted({r["dataset"] for r in rows})):
                selected = [r for r in rows if r["dataset"] == dataset and r["prompt"] == prompt]
                for i, row in enumerate(selected):
                    ax.bar(i, row["f1"], color=COLORS[row["provider"]])
                    ax.text(i, row["f1"] + .025, f"{row['f1']:.4f}", ha="center")
                ax.set_xticks(range(len(selected)), [NAMES[r["provider"]].replace(" ", "\n", 1) for r in selected], fontsize=9)
                ax.set_ylim(0, 1.15)
                ax.set_title(dataset)
            fig.suptitle(f"Paired classification F1 — {prompt}, threshold 0.5\n" + SCOPE, fontsize=11)
            save(fig, f"fig03_classification_f1_{prompt}", "Per-dataset paired rows; Detect Jailbreak excludes legacy NUL sample 15837.")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for ax, dataset in zip(axes.flat, sorted({r["dataset"] for r in thresholds})):
            for provider in report["completed_providers"]:
                for prompt, style in (("v1", "-"), ("v2", "--")):
                    selected = sorted([r for r in thresholds if r["dataset"] == dataset and r["provider"] == provider and r["prompt"] == prompt], key=lambda r: r["threshold"])
                    ax.plot([r["threshold"] for r in selected], [r["f1"] for r in selected], style, color=COLORS[provider], marker="o", label=f"{NAMES[provider]} {prompt}")
            ax.set_title(dataset)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Threshold (derived offline)")
            ax.set_ylabel("F1")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Threshold sensitivity — full successful rerun rows\nDescriptive only; thresholds are not tuned on a held-out validation set or repeated inferences.", fontsize=12)
        save(fig, "fig04_threshold_sensitivity", "Full rerun population; differs from paired Detect Jailbreak by one NUL row. Fixed primary threshold remains 0.5.")
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        for i, r in enumerate(timings):
            ax.plot([r["p50_ms"], r["p95_ms"]], [i, i], color=COLORS[r["provider"]], marker="o")
        ax.set_yticks(range(len(timings)), [f"{NAMES[r['provider']]} / {r['dataset']} / {r['prompt']}" for r in timings], fontsize=8)
        ax.set_xlabel("Request latency ms: p50 to p95, including retries")
        ax.set_title("Corrected live request timing\nRecorded concurrency varies by provider; no historical speedup comparison.")
        save(fig, "fig05_request_latency", "Per-request timing under concurrent workload, not elapsed/row throughput. Worker counts and means are in request_timing.csv.")
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        ax.barh(range(len(pii)), [r["detection_rate"] for r in pii], color=[COLORS[r["provider"]] for r in pii])
        ax.set_yticks(range(len(pii)), [f"{NAMES[r['provider']]} / {r['dataset']} / {r['prompt']}" for r in pii], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Positive document detection rate")
        ax.set_title("PII diagnostics — all-positive document labels\nNot span/entity F1; no false-positive, leakage or redaction-quality claim.")
        save(fig, "fig06_pii_detection_only", "Excluded from classification macro averages. Extraction benchmarking is a separate workflow.")
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
        for ax, metric in zip(axes.flat, QUALITY[:-1]):
            for i, r in enumerate(supplied):
                value = float(r[metric]) if r[metric] else None
                if value is not None:
                    ax.barh(i, value, color="#887799")
                    ax.text(value + .015, i, f"{value:.4f}", va="center", fontsize=9)
                else:
                    ax.text(.02, i, "Not supplied", va="center", color="#777777")
            ax.set_yticks(range(len(supplied)), [r["provider"] for r in supplied], fontsize=9)
            ax.set_xlim(0, 1.18)
            ax.set_ylim(len(supplied) - .4, -.6)
            ax.set_title(metric.upper())
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, .9, "Reference material only\n\nDataset scopes and successful rows unverified.\nPrompt configuration not uniformly supplied.\nNo shared-row ranking against corrected runs.\nLatency conditions differ; see source CSV.\nMissing F1 is not inferred from precision/recall.", va="top", fontsize=11)
        fig.suptitle("Supplied historical aggregates — unpaired references\nPreserved separately from the corrected live comparison; no fresh evaluation claimed.", fontsize=12)
        save(fig, "fig07_supplied_references", "Reference-only model aggregates preserved from the archived presentation; do not combine with corrected live scores for a leaderboard.")
    (figures / "figure_manifest.json").write_text(json.dumps({"source_summary_sha256": digest(root / "summary.json"), "figures": manifest}, indent=2) + "\n")
    total_errors = sum(r["errors"] for r in timings)
    status = "Laya: resumed after billing was restored; full rerun pending. Partial scores are excluded." if pending else f"Both provider suites have finished all configurations. Recorded request failures: {total_errors}; quality uses shared successful rows. See coverage for failure details."
    findings = []
    for fresh in [r for r in rows if r["provider"] != "historical_jev"]:
        old = next(r for r in rows if r["provider"] == "historical_jev" and r["dataset"] == fresh["dataset"] and r["prompt"] == fresh["prompt"])
        findings.append({"provider": fresh["provider"], "dataset": fresh["dataset"], "prompt": fresh["prompt"],
                         "f1_delta": fresh["f1"] - old["f1"], "auroc_delta": fresh["auroc"] - old["auroc"]})
    body = ["# Corrected benchmark comparison", "", SCOPE, "", status, "",
            "Scores use `answers.score.noul`. Historical confidence-based GLiDE/Laya scores are excluded. "
            "The previous presentation is archived in `../comparison_historical_confidence/`. "
            "The old 11-model chart is superseded: supplied summaries are retained separately below, without a combined ranking.", "",
            "## Findings against historical Jev", "", "Positive deltas favor the corrected live provider. "
            "Higher AUROC with lower fixed-threshold F1 suggests threshold calibration deserves investigation; "
            "it does not establish better classification performance or justify choosing thresholds on these test rows.", "",
            table(findings, ["provider", "dataset", "prompt", "f1_delta", "auroc_delta"]), "",
            "## Three-dataset macro averages", "", "Equal dataset weights, not pooled rows. AUPRC means average precision. v1 is the original figure's configuration; v2 is separate.", "",
            table(averages, ["provider", "prompt", "accuracy", "precision", "recall", "f1", "auroc", "auprc", "paired_rows"]), "",
            "![Corrected three-dataset averages](figures/fig13_three_dataset_averages.png)", "",
            "## Classification by dataset", "", "Quality excludes request errors and uses matching successful sample IDs, text and labels. "
            "The same common rows are required for all included providers. Detect Jailbreak excludes historical NUL-truncated sample 15837 from paired quality only. "
            "Historical Jev's requested model was `jev-latest`; its resolved model version is unknown.", "",
            table(rows, ["provider", "dataset", "prompt", "paired_rows", "excluded_nul_rows", "accuracy", "precision", "recall", "f1", "auroc", "auprc", "fpr", "fnr"]), "",
            "## Coverage and request timing", "", "Missing/error rows are not counted as negatives. Latency includes retries under the listed concurrency. "
            "Historical serial timings and supplied aggregate timings are not comparable to these measurements and are omitted from quality charts.", "",
            table(timings, ["provider", "dataset", "prompt", "valid", "errors", "expected", "workers", "mean_ms", "p50_ms", "p95_ms"]), "",
            "## PII detection diagnostics", "", "All-positive document labels: detection rate alone cannot measure extraction, false positives, leakage or over-masking. "
            "Use the separate PII span-extraction workflow for extraction quality.", "",
            table(pii, ["provider", "dataset", "prompt", "valid", "expected", "detection_rate"]), "",
            "## Supplied aggregate references — unpaired", "", "Preserved source values, not a fresh evaluation. Dataset membership, labels, successful rows, prompt configuration and timing are not uniformly verified. "
            "No combined ranking or latency speedup is claimed. F1 remains missing where not supplied.", "",
            table(supplied, ["provider", "prompt", "accuracy", "precision", "recall", "f1", "auroc", "mean_elapsed_ms"]), "",
            "## Reproduction and limitations", "", "Run `python -m scripts.publish_corrected_comparison` after completing or resuming a provider suite. "
            "The publisher checks source hashes, revalidates paired text/labels and computes AUPRC from saved probabilities; it makes no API calls. "
            "Provider status declares the full expected dataset/prompt matrix, preventing partial-suite publication. "
            "Thresholds 0.5–0.8 use one probability per row/prompt, not independent repeat inferences. "
            "These are historical-row parity benchmarks; split-policy PR #19 requires a separate rerun. "
            "The Notion fine-tuning evaluation populations also differ.", "",
            "[Figure gallery](figures/index.html) · [All figures PDF](figures/paper_figures.pdf) · "
            "[Machine-readable report](comparison.json) · [Three-dataset CSV](figures/three_dataset_averages.csv)", ""]
    markdown = "\n".join(body)
    for name in ("comparison.md", "model_comparison_report.md"):
        (output / name).write_text(markdown)
    # Tables and figures remain readable with no JavaScript or external assets.
    page = '<!doctype html><meta charset="utf-8"><title>Corrected benchmark comparison</title><style>body{font:16px system-ui;margin:32px;line-height:1.5}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ccc}img{max-width:100%}</style>' + render_html(markdown)
    (output / "comparison.html").write_text(page)
    gallery = '<!doctype html><meta charset="utf-8"><title>Corrected benchmark figures</title><style>body{font:16px system-ui;max-width:1200px;margin:24px auto}img{max-width:100%}</style><h1>Corrected benchmark figures</h1><p>' + html.escape(SCOPE + " " + status) + '</p><p><a href="../comparison.html">Full report</a> · <a href="paper_figures.pdf">All figures PDF</a> · <a href="three_dataset_averages.csv">Three-dataset CSV</a></p>'
    for item in manifest:
        name = item["name"]
        gallery += f'<h2>{html.escape(name)}</h2><p>{html.escape(item["caption"])}</p><img src="{name}.png" alt="{html.escape(item["caption"])}"><p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p>'
    (figures / "index.html").write_text(gallery)
    (figures / "three_dataset_comparison.html").write_text('<!doctype html><meta charset="utf-8"><title>Corrected three-dataset averages</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}td,th{padding:8px}</style>' + render_html(
        '# Corrected three-dataset averages\n\n' + SCOPE + '\n\n' + status + '\n\n' +
        table([r for r in averages if r["prompt"] == "v1"], ["provider", "accuracy", "precision", "recall", "f1", "auroc", "auprc"]) +
        '\n\n![Corrected three-dataset averages](fig13_three_dataset_averages.png)\n\n[CSV](three_dataset_averages.csv) · [Full report](../comparison.html) · [Figure gallery](index.html)'))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/corrected_classification_2026-10-28"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument("--historical-dir", type=Path, default=Path("outputs/typesafe.ai"))
    parser.add_argument("--reference-dir", type=Path, default=Path("outputs/comparison_historical_confidence"))
    args = parser.parse_args()
    publish(args.input_dir, args.output_dir, args.historical_dir, args.reference_dir)


if __name__ == "__main__":
    main()
