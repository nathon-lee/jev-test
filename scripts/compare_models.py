import argparse
import json
from pathlib import Path

from benchmark.comparison import MODELS, collect_report, write_report


def main():
    parser = argparse.ArgumentParser(description="Generate an offline Chinese model comparison from saved outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison_historical_confidence"))
    parser.add_argument("--providers", default=",".join(MODELS), help="Comma-separated provider directory names")
    parser.add_argument("--datasets", help="Comma-separated dataset names; use this to exclude train-only reference datasets")
    parser.add_argument("--paper-figures", action="store_true", help="Also export comparison PNG, SVG and PDF figures")
    parser.add_argument("--external-summary", type=Path, help="Optional OpenJev aggregate CSV (reference-only comparison)")
    parser.add_argument("--internal-summary", type=Path, help="Additional internal model aggregate CSV (reference-only)")
    parser.add_argument("--previous-report", type=Path, help="Preserve historical summaries when raw CSVs are unavailable")
    args = parser.parse_args()
    providers = [name.strip() for name in args.providers.split(",") if name.strip()]
    previous = json.loads(args.previous_report.read_text(encoding="utf-8")) if args.previous_report else None
    if args.internal_summary and not (args.external_summary or (previous and previous.get("external_comparison"))):
        parser.error("--internal-summary requires --external-summary or a previous external comparison")
    report = collect_report(args.input_dir, args.data_dir, providers, previous_report=previous,
                            datasets=args.datasets.split(",") if args.datasets else None)
    if args.external_summary is not None:
        from benchmark.external_summary import attach_openjev_summary

        attach_openjev_summary(report, args.external_summary, args.internal_summary)
    elif previous and previous.get("external_comparison"):
        from benchmark.external_summary import attach_openjev_summary

        external = previous["external_comparison"]
        internal = args.internal_summary or (Path(external["internal_source"]) if external.get("internal_source") else None)
        attach_openjev_summary(report, Path(external["source"]), internal)
    write_report(report, args.output_dir)
    if args.paper_figures:
        from benchmark.paper_figures import write_paper_figures

        write_paper_figures(report, args.output_dir / "figures")
        print(f"Figure gallery: {args.output_dir / 'figures' / 'index.html'}")
    print(f"Report: {args.output_dir / 'comparison.html'}")
    print(f"Markdown: {args.output_dir / 'comparison.md'}")
    for conclusion in report["conclusions"]:
        print(conclusion)


if __name__ == "__main__":
    main()
