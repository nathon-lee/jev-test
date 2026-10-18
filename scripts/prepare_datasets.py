from pathlib import Path
from benchmark.datasets import (
    prepare_toxigen,
    prepare_hatecheck,
    prepare_aegis2,
    prepare_detect_jailbreak,
    prepare_jailbreak_classification,
    prepare_jailbreakhub,
    save_dataset,
)

OUTPUT_DIR = Path("data/prepared")


def main():
    toxigen = prepare_toxigen()
    save_dataset(
        toxigen,
        OUTPUT_DIR / "toxigen.csv",
    )

    hatecheck = prepare_hatecheck()
    save_dataset(
        hatecheck,
        OUTPUT_DIR / "hatecheck.csv",
    )

    aegis2 = prepare_aegis2()
    save_dataset(
        aegis2,
        OUTPUT_DIR / "aegis2.csv",
    )

    detect_jailbreak = prepare_detect_jailbreak()
    save_dataset(
        detect_jailbreak,
        OUTPUT_DIR / "detect_jailbreak.csv",
    )

    jailbreak_classification = prepare_jailbreak_classification()
    save_dataset(
        jailbreak_classification,
        OUTPUT_DIR / "jailbreak_classification.csv",
    )

    jailbreakhub = prepare_jailbreakhub()
    save_dataset(
        jailbreakhub,
        OUTPUT_DIR / "jailbreakhub.csv",
    )


if __name__ == "__main__":
    main()
