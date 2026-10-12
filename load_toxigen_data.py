from datasets import get_dataset_config_names
configs = get_dataset_config_names(
    "toxigen/toxigen-data"
)
print(configs)

from datasets import (
    get_dataset_config_names,
    load_dataset,
)
repo = "toxigen/toxigen-data"
configs = get_dataset_config_names(repo)
print("Configs:", configs)
for config in configs:
    print("\n" + "=" * 80)
    print("CONFIG:", config)
    print("=" * 80)
    ds = load_dataset(
        repo,
        config
    )
    print(ds)
    for split in ds.keys():
        print(f"\nSplit: {split}")
        print("Rows:", len(ds[split]))
        print("Columns:")
        print(ds[split].column_names)


import pandas as pd

url = (
    "https://huggingface.co/datasets/"
    "toxigen/toxigen-data/resolve/main/toxigen.csv"
)

df = pd.read_csv(url, nrows=5)

print(df.columns.tolist())
print(df.to_string())