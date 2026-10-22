import argparse
from pathlib import Path

from benchmark.comparison import MODELS, collect_report, write_report


def main():
    parser = argparse.ArgumentParser(description="Generate an offline Chinese model comparison from saved outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    parser.add_argument("--providers", default=",".join(MODELS), help="Comma-separated provider directory names")
    args = parser.parse_args()
    providers = [name.strip() for name in args.providers.split(",") if name.strip()]
    report = collect_report(args.input_dir, args.data_dir, providers)
    write_report(report, args.output_dir)
    print(f"Report: {args.output_dir / 'comparison.html'}")
    print(f"Markdown: {args.output_dir / 'comparison.md'}")
    for conclusion in report["conclusions"]:
        print(conclusion)


if __name__ == "__main__":
    main()
