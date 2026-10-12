from datasets import load_dataset
import pandas as pd


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


# ------------------------------------------------------------
# annotated 9.9k
# ------------------------------------------------------------

ds = load_dataset(
    "toxigen/toxigen-data",
    "annotated",
)

annotated = pd.concat(
    [
        ds["train"].to_pandas(),
        ds["test"].to_pandas(),
    ],
    ignore_index=True,
)

annotated["match_text"] = normalize_text(
    annotated["text"]
)


# ------------------------------------------------------------
# original train
# ------------------------------------------------------------

train = load_dataset(
    "toxigen/toxigen-data",
    "train",
    split="train",
).to_pandas()

train["match_text"] = normalize_text(
    train["generation"]
)


# ------------------------------------------------------------
# Remove conflicting generation labels
# ------------------------------------------------------------

label_count = (
    train
    .groupby("match_text")["prompt_label"]
    .nunique()
)

conflicts = set(
    label_count[label_count > 1].index
)

train = train[
    ~train["match_text"].isin(conflicts)
].copy()


mapping = (
    train[
        ["match_text", "prompt_label"]
    ]
    .drop_duplicates()
)


# ------------------------------------------------------------
# Join
# ------------------------------------------------------------

df = annotated.merge(
    mapping,
    on="match_text",
    how="left",
    validate="many_to_one",
)


matched = df[
    df["prompt_label"].notna()
].copy()


print("Matched:", len(matched))


# ============================================================
# 1. Mean toxicity by prompt_label
# ============================================================

print("\n=== By prompt_label ===")

print(
    matched
    .groupby("prompt_label")[
        ["toxicity_human", "toxicity_ai"]
    ]
    .agg(["mean", "median", "std"])
)


# ============================================================
# 2. Human score distribution
# ============================================================

print("\n=== toxicity_human vs prompt_label ===")

print(
    pd.crosstab(
        matched["toxicity_human"],
        matched["prompt_label"],
        margins=True,
    )
)


# ============================================================
# 3. AI score distribution
# ============================================================

print("\n=== toxicity_ai vs prompt_label ===")

print(
    pd.crosstab(
        matched["toxicity_ai"],
        matched["prompt_label"],
        margins=True,
    )
)


# ============================================================
# 4. Combined score
# ============================================================

matched["combined"] = (
    matched["toxicity_human"]
    + matched["toxicity_ai"]
)


print("\n=== combined score vs prompt_label ===")

print(
    matched
    .groupby("prompt_label")[
        "combined"
    ]
    .describe()
)


# ============================================================
# 5. Test simple thresholds
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


y = matched["prompt_label"].astype(int)


def evaluate_rule(name, prediction):

    print(f"\n{name}")

    print(
        "Accuracy:",
        round(
            accuracy_score(y, prediction),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(y, prediction),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(y, prediction),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(y, prediction),
            4
        )
    )


# Human only

for threshold in [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]:

    pred = (
        matched["toxicity_human"]
        >= threshold
    ).astype(int)

    evaluate_rule(
        f"human >= {threshold}",
        pred,
    )


# AI only

for threshold in [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]:

    pred = (
        matched["toxicity_ai"]
        >= threshold
    ).astype(int)

    evaluate_rule(
        f"ai >= {threshold}",
        pred,
    )


# Combined score

for threshold in [
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
]:

    pred = (
        matched["combined"]
        > threshold
    ).astype(int)

    evaluate_rule(
        f"human + ai > {threshold}",
        pred,
    )