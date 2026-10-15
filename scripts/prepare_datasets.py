from pathlib import Path
from benchmark.datasets import (
    prepare_toxigen,
    prepare_hatecheck,
    prepare_aegis2,
    prepare_pii_masking_300k,
    prepare_pii_masking_43k,
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
    pii_300k = prepare_pii_masking_300k()
    save_dataset(
        pii_300k,
        OUTPUT_DIR / "pii_masking_300k.csv",
    )
    pii_43k = prepare_pii_masking_43k()
    save_dataset(
        pii_43k,
        OUTPUT_DIR / "pii_masking_43k.csv",
    )
if __name__ == "__main__":
    main()
