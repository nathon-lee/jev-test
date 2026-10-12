from datasets import load_dataset
import pandas as pd


# ============================================================
# annotated: 9900
# ============================================================

annotated = load_dataset(
    "toxigen/toxigen-data",
    "annotated"
)

ann = pd.concat(
    [
        annotated["train"].to_pandas(),
        annotated["test"].to_pandas(),
    ],
    ignore_index=True,
)

ann["sample_id"] = range(len(ann))


# ============================================================
# raw annotations: 27450
# ============================================================

raw = load_dataset(
    "toxigen/toxigen-data",
    "annotations",
    split="train",
).to_pandas()


print("Annotated rows:", len(ann))
print("Annotated unique text:", ann["text"].nunique())

print()

print("Raw annotation rows:", len(raw))
print(
    "Raw unique text:",
    raw["Input.text"].nunique()
)


# ============================================================
# Build unique text -> prompt label mapping
# ============================================================

label_map = (
    raw[
        [
            "Input.text",
            "Input.prompt_label",
            "Input.binary_prompt_label",
        ]
    ]
    .drop_duplicates()
)


# ============================================================
# Check whether same text ever has conflicting labels
# ============================================================

conflicts = (
    label_map
    .groupby("Input.text")
    ["Input.binary_prompt_label"]
    .nunique()
)

conflicts = conflicts[
    conflicts > 1
]

print("\nConflicting text labels:")
print(len(conflicts))


# ============================================================
# Deduplicate by text after conflict check
# ============================================================

unique_label_map = (
    label_map
    .drop_duplicates(
        subset=["Input.text"]
    )
)


# ============================================================
# Map labels to annotated 9900
# ============================================================

result = ann.merge(
    unique_label_map,
    left_on="text",
    right_on="Input.text",
    how="left",
)


# ============================================================
# IMPORTANT: count by sample_id
# ============================================================

matched = result[
    "Input.binary_prompt_label"
].notna()


print("\n=== Mapping Result ===")

print("Total annotated:", len(result))
print("Matched:", matched.sum())
print("Unmatched:", (~matched).sum())

print(
    "Match rate:",
    f"{matched.mean():.2%}"
)


# ============================================================
# Ground truth distribution
# ============================================================

print("\n=== Binary Prompt Label ===")

print(
    result.loc[
        matched,
        "Input.binary_prompt_label"
    ].value_counts()
)


# ============================================================
# Human toxicity vs prompt label
# ============================================================

print(
    "\n=== toxicity_human vs prompt label ==="
)

print(
    result.loc[matched]
    .groupby(
        "Input.binary_prompt_label"
    )["toxicity_human"]
    .describe()
)


# ============================================================
# Unmatched examples
# ============================================================

print("\n=== Unmatched examples ===")

print(
    result.loc[
        ~matched,
        [
            "sample_id",
            "text",
            "toxicity_human",
        ]
    ]
    .head(20)
    .to_string(index=False)
)