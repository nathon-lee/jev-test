from datasets import load_dataset
import pandas as pd


# ============================================================
# 1. Load 9.9k human annotated samples
# ============================================================

annotated = load_dataset(
    "toxigen/toxigen-data",
    "annotated"
)

annotated_df = pd.concat(
    [
        annotated["train"].to_pandas(),
        annotated["test"].to_pandas(),
    ],
    ignore_index=True,
)

print("Annotated rows:", len(annotated_df))


# ============================================================
# 2. Load original 250k generated samples
# ============================================================

train = load_dataset(
    "toxigen/toxigen-data",
    "train",
    split="train"
)

train_df = train.to_pandas()

print("Original rows:", len(train_df))


# ============================================================
# 3. Exact text matching
# ============================================================

merged = annotated_df.merge(
    train_df[
        [
            "generation",
            "prompt",
            "prompt_label",
            "generation_method",
            "group",
        ]
    ],
    left_on="text",
    right_on="generation",
    how="left",
    indicator=True,
)

print("\n=== Exact match result ===")

print(
    merged["_merge"].value_counts()
)

matched = merged[
    merged["_merge"] == "both"
]

print(
    f"\nMatched: {len(matched)} / {len(annotated_df)}"
)

print(
    f"Match rate: "
    f"{len(matched) / len(annotated_df):.2%}"
)


# ============================================================
# 4. Inspect prompt_label
# ============================================================

print("\n=== prompt_label distribution ===")

print(
    matched["prompt_label"]
    .value_counts(dropna=False)
)


# ============================================================
# 5. Compare prompt_label with human toxicity
# ============================================================

print("\n=== Human toxicity by prompt_label ===")

print(
    matched
    .groupby("prompt_label")["toxicity_human"]
    .describe()
)


# ============================================================
# 6. Show examples
# ============================================================

print("\n=== Example matched rows ===")

cols = [
    "text",
    "toxicity_human",
    "toxicity_ai",
    "prompt_label",
    "roberta_prediction",
]

# roberta_prediction wasn't included above, so skip if absent
cols = [
    c for c in cols
    if c in matched.columns
]

print(
    matched[cols]
    .head(20)
    .to_string()
)