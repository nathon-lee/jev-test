from pathlib import Path
import argparse
from benchmark.datasets import (
    prepare_toxigen,
    prepare_hatecheck,
    prepare_aegis2,
    prepare_detect_jailbreak,
    prepare_jailbreak_classification,
    prepare_jailbreakhub,
    prepare_pii_masking_nano_1k,
    prepare_pii_masking_mini_10k,
    save_dataset,
)

OUTPUT_DIR = Path("data/prepared")


def canonical_dataset_file_stem(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("-", "_")


PREPARERS = {
    "toxigen": prepare_toxigen, "hatecheck": prepare_hatecheck, "aegis2": prepare_aegis2,
    "detect-jailbreak": prepare_detect_jailbreak,
    "jailbreak-classification": prepare_jailbreak_classification, "jailbreakhub": prepare_jailbreakhub,
    "pii_masking_nano_1k": prepare_pii_masking_nano_1k,
    "pii_masking_mini_10k": prepare_pii_masking_mini_10k,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare explicit, revision-pinned evaluation partitions")
    parser.add_argument("--datasets", default="all", help="Comma-separated suite dataset names or all")
    parser.add_argument("--split", help="Exact split override; unions/slices are not accepted")
    parser.add_argument("--revision", default="main", help="Resolved to a commit SHA for each source dataset")
    parser.add_argument("--allow-train-eval", action="store_true", help="Explicitly permit labelled training-reference data")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    names = list(PREPARERS) if args.datasets == "all" else args.datasets.split(",")
    if not names or any(name not in PREPARERS for name in names):
        parser.error(f"Choose datasets from {list(PREPARERS)}")
    # Prepare all selected frames before replacing any existing data files. A
    # train-only source must not leave a half-updated dataset directory on failure.
    frames = {name: PREPARERS[name](split=args.split, revision=args.revision,
                                   allow_train_eval=args.allow_train_eval) for name in names}
    for name, frame in frames.items():
        save_dataset(frame, args.output_dir / f"{canonical_dataset_file_stem(name)}.csv")
if __name__ == "__main__":
    main()
