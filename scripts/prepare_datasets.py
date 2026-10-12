from pathlib import Path
from benchmark.datasets import (
    prepare_toxigen,
    prepare_hatecheck,
    prepare_aegis2,
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
if __name__ == "__main__":
    main()
