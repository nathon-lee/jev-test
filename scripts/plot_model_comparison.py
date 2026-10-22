import argparse
import json
from pathlib import Path

from benchmark.paper_figures import write_paper_figures


def main():
    parser = argparse.ArgumentParser(description="Render comparison figures from the saved comparison JSON.")
    parser.add_argument("--report", type=Path, default=Path("outputs/comparison/comparison.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison/figures"))
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (minimum 72; vector exports are unaffected)")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    write_paper_figures(report, args.output_dir, dpi=args.dpi)
    print(f"Figure gallery: {args.output_dir / 'index.html'}")
    print(f"Combined PDF: {args.output_dir / 'paper_figures.pdf'}")
    print(f"LaTeX table: {args.output_dir / 'table01_baseline.tex'}")


if __name__ == "__main__":
    main()
