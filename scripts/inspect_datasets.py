from benchmark.datasets import (
    load_toxigen,
    load_hatecheck,
    load_aegis,
)

def inspect(ds):
    ds.summary()
    print("\nSchema:")
    print(ds.data.dtypes)
    print("\nNull count:")
    print(ds.data.isnull().sum())
    print("\nLabel ratio:")
    print(
        ds.data["label"]
        .value_counts(normalize=True)
        .sort_index()
    )
    print("\nSample texts:")
    for _, row in ds.data.head(3).iterrows():
        print("-" * 80)
        print(
            f"id={row['sample_id']} "
            f"label={row['label']}"
        )
        print(
            row["text"][:500]
        )

def main():
    datasets = [
        load_toxigen(),
        # load_hatecheck(),
        # load_aegis(),
    ]
    for ds in datasets:
        inspect(ds)

if __name__ == "__main__":
    main()

