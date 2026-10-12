import pandas as pd
import httpx
from typesafe_sdk import Noul, TypeSafeClient

from benchmark.jev_client import (
    get_question,
    run_jev,
)


DATASETS = [
    "toxigen",
    "hatecheck",
    "aegis2",
]


def main():

    # ========================================================
    # Put your existing Jev initialization here
    # ========================================================

    http_client = httpx.Client(
        verify=False
    )

    client = TypeSafeClient(
        http_client=http_client,
        timeout=30.0
    )

    # ========================================================

    for dataset in DATASETS:

        print("\n")
        print("=" * 80)
        print(dataset.upper())
        print("=" * 80)

        path = f"data/prepared/{dataset}.csv"

        df = pd.read_csv(path)

        # Important:
        # don't simply take head(10), because it may contain
        # only one class.

        negative = df[
            df["label"] == 0
        ].head(5)

        positive = df[
            df["label"] == 1
        ].head(5)

        samples = pd.concat(
            [negative, positive]
        )

        print("Question:")
        print(get_question(dataset))

        for _, row in samples.iterrows():

            print("\n" + "-" * 80)

            print("sample_id:", row["sample_id"])
            print("ground_truth:", row["label"])

            print("text:")
            print(row["text"])

            try:

                result = run_jev(
                    client=client,
                    dataset=dataset,
                    text=row["text"],
                )

                print("\nJev prediction:")
                print(result["prediction"])

                print(
                    "latency:",
                    f"{result["latency_ms"]:.2f} ms"
                )

            except Exception as e:

                print(
                    "ERROR:",
                    repr(e)
                )


if __name__ == "__main__":
    main()