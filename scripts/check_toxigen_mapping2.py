from datasets import load_dataset
import pandas as pd

ds = load_dataset(
    "toxigen/toxigen-data",
    "annotations",
    split="train",
)

df = ds.to_pandas()

print("Rows:", len(df))

# --------------------------------------------------
# Key fields
# --------------------------------------------------

fields = [
    "Input.prompt",
    "Input.text",
    "Input.prompt_label",
    "Input.binary_prompt_label",
]

for field in fields:
    print("\n" + "=" * 60)
    print(field)

    if field not in df.columns:
        print("NOT FOUND")
        continue

    print("dtype:", df[field].dtype)
    print("non-null:", df[field].notna().sum())
    print("unique:", df[field].nunique(dropna=False))

    print("\nValue counts:")
    print(
        df[field]
        .value_counts(dropna=False)
        .head(2)
    )


# --------------------------------------------------
# First rows
# --------------------------------------------------

print("\n=== SAMPLE ===")

print(
    df[fields]
    .head(2)
    .to_string()
)