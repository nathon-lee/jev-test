#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"


def load_env_file() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return

    try:
        dotenv_module = __import__("dotenv", fromlist=["load_dotenv"])
        dotenv_module.load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    repo_root = str(ROOT_DIR)
    env["PYTHONPATH"] = repo_root if not existing else repo_root + os.pathsep + existing
    return env


def ensure_compatibility_aliases(dataset_name: str, prompt_version: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if dataset_name == "all":
        dataset_names = ["toxigen", "hatecheck", "aegis2"]
    else:
        dataset_names = [dataset_name]

    for dataset in dataset_names:
        src = OUTPUT_DIR / f"{dataset}_{prompt_version}_predictions.csv"
        dst = OUTPUT_DIR / f"{dataset}_predictions.csv"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")


def run_command(command: list[str], env: dict, dry_run: bool = False) -> None:
    print("$", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(ROOT_DIR), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single benchmark case and automatically export the result to the compatibility "
            "filename expected by the evaluator."
        )
    )
    parser.add_argument("--dataset", choices=["toxigen", "hatecheck", "aegis2", "all"], default="all")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation script after the benchmark finishes.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    args = parser.parse_args()

    load_env_file()
    env = build_env()

    datasets = ["toxigen", "hatecheck", "aegis2"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        command = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_benchmark.py"),
            "--dataset",
            dataset,
            "--prompt-version",
            args.prompt_version,
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])

        run_command(command, env, dry_run=args.dry_run)
        ensure_compatibility_aliases(dataset, args.prompt_version)

        if args.evaluate:
            evaluate_command = [
                sys.executable,
                str(ROOT_DIR / "scripts" / "evaluate.py"),
                "--dataset",
                dataset,
            ]
            run_command(evaluate_command, env, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No commands were executed.")
    else:
        print("\nBenchmark step complete. Results are available under outputs/.")


if __name__ == "__main__":
    main()
