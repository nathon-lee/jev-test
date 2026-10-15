from pathlib import Path
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


def main():
    toxigen = prepare_toxigen()
    save_dataset(
        toxigen,
        OUTPUT_DIR / f"{canonical_dataset_file_stem('toxigen')}.csv",
    )

    hatecheck = prepare_hatecheck()
    save_dataset(
        hatecheck,
        OUTPUT_DIR / f"{canonical_dataset_file_stem('hatecheck')}.csv",
    )

    aegis2 = prepare_aegis2()
    save_dataset(
        aegis2,
        OUTPUT_DIR / f"{canonical_dataset_file_stem('aegis2')}.csv",
    )

    detect_jailbreak = prepare_detect_jailbreak()
    save_dataset(
        detect_jailbreak,
        OUTPUT_DIR / f"{canonical_dataset_file_stem('detect-jailbreak')}.csv",
    )

    jailbreak_classification = prepare_jailbreak_classification()
    save_dataset(
        jailbreak_classification,
        OUTPUT_DIR / f"{canonical_dataset_file_stem('jailbreak-classification')}.csv",
    )

    jailbreakhub = prepare_jailbreakhub()
    save_dataset(
        jailbreakhub,
        OUTPUT_DIR / f"{canonical_dataset_file_stem('jailbreakhub')}.csv",
    )

    pii_nano_1k = prepare_pii_masking_nano_1k()
    save_dataset(
        pii_nano_1k,
        OUTPUT_DIR / "pii_masking_nano_1k.csv",
    )

    pii_mini_10k = prepare_pii_masking_mini_10k()
    save_dataset(
        pii_mini_10k,
        OUTPUT_DIR / "pii_masking_mini_10k.csv",
    )
if __name__ == "__main__":
    main()
