import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark.provider_suite import (
    DEFAULT_THRESHOLDS,
    available_datasets,
    normalize_provider_name,
    parse_thresholds,
    resolve_provider,
    run_single_benchmark,
    save_rows,
)


def provider_list(raw_value: str | None) -> list[str]:
    if raw_value is None or raw_value.lower() == "all":
        return list(sorted(resolve_provider(name)["kind"] and name for name in [
            "typesafe.ai",
            "fastino.ai",
            "laya-ai.com",
            "openjev",
            "galileo.ai",
            "microsoft-decision-1",
        ]))
    return [normalize_provider_name(part.strip()) for part in raw_value.split(",") if part.strip()]


def dataset_list(raw_value: str | None) -> list[str]:
    if raw_value is None or raw_value.lower() == "all":
        return available_datasets()
    aliases = {
        "aegis-ai-content-safety-dataset-2.0": "aegis2",
        "guardrailsai-detect-jailbreak": "detect-jailbreak",
        "guardrailsai/detect-jailbreak": "detect-jailbreak",
        "jackhhao-jailbreak-classification": "jailbreak-classification",
        "walledai-jailbreakhub": "jailbreakhub",
        "pii_masking_nano_1k": "pii_masking_nano_1k",
        "pii-masking-nano-1k": "pii_masking_nano_1k",
        "ai4privacy/pii-masking-nano-1k": "pii_masking_nano_1k",
        "pii_masking_mini_10k": "pii_masking_mini_10k",
        "pii-masking-mini-10k": "pii_masking_mini_10k",
        "ai4privacy/pii-masking-mini-10k": "pii_masking_mini_10k",
    }
    names = []
    for part in raw_value.split(","):
        key = part.strip().lower()
        names.append(aliases.get(key, key))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple providers against the local and Hugging Face datasets.")
    parser.add_argument("--providers", default="all", help="Comma-separated provider names or all")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or all")
    parser.add_argument("--prompt-versions", default="v1,v2", help="Comma-separated prompt versions: v1,v2")
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8", help="Comma-separated thresholds")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of benchmark combinations to run in parallel")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for rows per dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output root; results use provider subdirectories (default: outputs)")
    parser.add_argument("--api-key", default=None, help="Optional API key override for the selected provider")
    parser.add_argument("--base-url", default=None, help="Optional base URL override for the selected provider")
    parser.add_argument("--debug", action="store_true", help="Print the exact outgoing request URL, headers, and payload for each provider call")
    args = parser.parse_args()

    providers = provider_list(args.providers)
    datasets = dataset_list(args.datasets)
    prompt_versions = [part.strip().lower() for part in args.prompt_versions.split(",") if part.strip()]
    thresholds = parse_thresholds(args.thresholds)
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    print("Providers:", providers)
    print("Datasets:", datasets)
    print("Prompt versions:", prompt_versions)
    print("Thresholds:", thresholds)
    print("Max workers:", args.max_workers)

    tasks: list[tuple[str, str, str, float]] = []
    for provider in providers:
        for dataset in datasets:
            for prompt_version in prompt_versions:
                for threshold in thresholds:
                    tasks.append((provider, dataset, prompt_version, threshold))

    def run_task(task: tuple[str, str, str, float]) -> tuple[Path, int]:
        provider, dataset, prompt_version, threshold = task
        print(f"\n=== {provider} | {dataset} | {prompt_version} | threshold={threshold} ===")
        rows = run_single_benchmark(
            provider_name=provider,
            dataset_name=dataset,
            prompt_version=prompt_version,
            threshold=threshold,
            limit=args.limit,
            api_key=args.api_key,
            base_url=args.base_url,
            debug=args.debug,
        )
        saved_path = save_rows(rows, provider, dataset, prompt_version, threshold, output_dir=args.output_dir)
        return saved_path, len(rows)

    failures: list[tuple[tuple[str, str, str, float], Exception]] = []

    if args.max_workers == 1:
        for task in tasks:
            provider, dataset, prompt_version, threshold = task
            try:
                saved_path, row_count = run_task(task)
                print(f"Saved: {saved_path} (rows={row_count})")
            except Exception as exc:
                failures.append((task, exc))
                print(
                    f"FAILED: {provider} | {dataset} | {prompt_version} | threshold={threshold} | error={repr(exc)}"
                )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {executor.submit(run_task, task): task for task in tasks}
            completed = 0
            total = len(tasks)
            for future in as_completed(future_map):
                completed += 1
                task = future_map[future]
                provider, dataset, prompt_version, threshold = task
                try:
                    saved_path, row_count = future.result()
                    print(
                        f"[{completed}/{total}] Saved: {saved_path} "
                        f"({provider} | {dataset} | {prompt_version} | t={threshold} | rows={row_count})"
                    )
                except Exception as exc:
                    failures.append((task, exc))
                    print(
                        f"[{completed}/{total}] FAILED: "
                        f"{provider} | {dataset} | {prompt_version} | threshold={threshold} | error={repr(exc)}"
                    )

    if failures:
        print("\nSome benchmark tasks failed:")
        for task, exc in failures:
            provider, dataset, prompt_version, threshold = task
            print(f"- {provider} | {dataset} | {prompt_version} | threshold={threshold} -> {repr(exc)}")
        raise SystemExit(1)

    print("\nAll benchmark runs are complete.")


if __name__ == "__main__":
    main()
