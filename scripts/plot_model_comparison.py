import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark.paper_figures import write_fastino_comparison, write_paper_figures


def main():
    parser = argparse.ArgumentParser(description="Render comparison figures from the saved comparison JSON.")
    parser.add_argument("--report", type=Path, default=Path("outputs/comparison/comparison.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison/figures"))
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (minimum 72; vector exports are unaffected)")
    parser.add_argument("--fine-tuned-run", type=Path, help="Add an isolated fine-tuned run to reference-only comparisons")
    parser.add_argument("--fine-tuned-log", type=Path, help="Complete benchmark log for --fine-tuned-run")
    parser.add_argument("--gliguard-run", type=Path, help="Add a completed Fastino GLiGuard 300M run to reference-only comparisons")
    parser.add_argument("--poc-run", type=Path, help="Add a completed Fastino SystemOne POC run to reference-only comparisons")
    parser.add_argument("--gliner25-run", type=Path, help="Add a completed Fastino GLiNER2.5-Decide run to reference-only comparisons")
    parser.add_argument("--qwen-summary", type=Path, help="User-supplied 12-dataset Qwen3Guard summary CSV (v1 / 0.5)")
    parser.add_argument("--fastino-page", type=Path, help="Write only a self-contained per-dataset Fastino comparison HTML page")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    if bool(args.fine_tuned_run) != bool(args.fine_tuned_log):
        parser.error("--fine-tuned-run and --fine-tuned-log must be supplied together")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.fine_tuned_run:
        from benchmark.external_summary import attach_finetuned_run

        attach_finetuned_run(report, args.fine_tuned_run, args.fine_tuned_log)
    if args.gliguard_run:
        from benchmark.external_summary import attach_gliguard_run

        attach_gliguard_run(report, args.gliguard_run)
    if args.poc_run:
        from benchmark.external_summary import attach_poc_run

        attach_poc_run(report, args.poc_run)
    if args.gliner25_run:
        from benchmark.external_summary import attach_gliner25_run

        attach_gliner25_run(report, args.gliner25_run)
    if args.qwen_summary:
        from benchmark.external_summary import attach_qwen_summary

        attach_qwen_summary(report, args.qwen_summary)
    if args.fine_tuned_run or args.gliguard_run or args.poc_run or args.gliner25_run:
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
    if args.fastino_page:
        write_fastino_comparison(report, args.fastino_page, dpi=args.dpi)
        print(f"Fastino comparison: {args.fastino_page}")
        return
    if args.fine_tuned_run or args.gliguard_run or args.poc_run or args.gliner25_run:
        from benchmark.comparison import write_report

        write_report(report, args.report.parent)
    write_paper_figures(report, args.output_dir, dpi=args.dpi)
    print(f"Figure gallery: {args.output_dir / 'index.html'}")
    print(f"Combined PDF: {args.output_dir / 'paper_figures.pdf'}")
    print(f"LaTeX table: {args.output_dir / 'table01_baseline.tex'}")


if __name__ == "__main__":
    main()
