from datasets import load_dataset
import pandas as pd


# ============================================================
# Load all three configs
# ============================================================

annotated_ds = load_dataset(
    "toxigen/toxigen-data",
    "annotated",
)

annotated = pd.concat(
    [
        annotated_ds["train"].to_pandas(),
        annotated_ds["test"].to_pandas(),
    ],
    ignore_index=True,
)

annotations = load_dataset(
    "toxigen/toxigen-data",
    "annotations",
    split="train",
).to_pandas()

train = load_dataset(
    "toxigen/toxigen-data",
    "train",
    split="train",
).to_pandas()


print("=== DATASET SIZE ===")
print("annotated:", len(annotated))
print("annotations:", len(annotations))
print("train:", len(train))


# ============================================================
# Normalization
# ============================================================

def normalize(s):
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


a_text = normalize(annotated["text"])

raw_text = normalize(annotations["Input.text"])
raw_prompt = normalize(annotations["Input.prompt"])

train_generation = normalize(train["generation"])
train_prompt = normalize(train["prompt"])


# ============================================================
# Compare unique text sets
# ============================================================

def compare(name1, s1, name2, s2):

    set1 = set(s1)
    set2 = set(s2)

    intersection = set1 & set2

    print("\n" + "=" * 70)
    print(f"{name1} <-> {name2}")
    print("=" * 70)

    print(f"{name1} unique:", len(set1))
    print(f"{name2} unique:", len(set2))

    print("Intersection:", len(intersection))

    print(
        f"{name1} coverage:",
        f"{len(intersection) / len(set1):.2%}"
    )

    print(
        f"{name2} coverage:",
        f"{len(intersection) / len(set2):.2%}"
    )


# ------------------------------------------------------------
# The important comparisons
# ------------------------------------------------------------

compare(
    "annotated.text",
    a_text,
    "annotations.Input.text",
    raw_text,
)

compare(
    "annotated.text",
    a_text,
    "annotations.Input.prompt",
    raw_prompt,
)

compare(
    "annotated.text",
    a_text,
    "train.generation",
    train_generation,
)

compare(
    "annotated.text",
    a_text,
    "train.prompt",
    train_prompt,
)


# ============================================================
# Also inspect examples
# ============================================================

print("\n\n=== ANNOTATED EXAMPLES ===")

for i in range(5):
    print("\n---", i, "---")
    print(repr(annotated.iloc[i]["text"]))


print("\n\n=== ANNOTATIONS Input.text EXAMPLES ===")

for i in range(5):
    print("\n---", i, "---")
    print(repr(annotations.iloc[i]["Input.text"]))


print("\n\n=== TRAIN generation EXAMPLES ===")

for i in range(5):
    print("\n---", i, "---")
    print(repr(train.iloc[i]["generation"]))