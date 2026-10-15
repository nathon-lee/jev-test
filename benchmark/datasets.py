import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
# ============================================================
# Common helpers
# ============================================================
def normalize_text(series: pd.Series) -> pd.Series:
    """
    Used ONLY for matching text across ToxiGen datasets.
    We keep the original text as the actual benchmark input.
    """
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
def save_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    print(f"Rows : {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())
# ============================================================
# ToxiGen
# ============================================================
def prepare_toxigen() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Preparing ToxiGen")
    print("=" * 70)
    # --------------------------------------------------------
    # 1. Load the 9.9k human-annotated dataset
    # --------------------------------------------------------
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
    annotated["sample_id"] = range(len(annotated))
    print("Annotated rows:", len(annotated))
    # --------------------------------------------------------
    # 2. Load original generated dataset containing
    #    generation + prompt_label
    # --------------------------------------------------------
    train = load_dataset(
        "toxigen/toxigen-data",
        "train",
        split="train",
    ).to_pandas()
    print("Original train rows:", len(train))
    # --------------------------------------------------------
    # 3. Normalize ONLY for matching
    # --------------------------------------------------------
    annotated["match_text"] = normalize_text(
        annotated["text"]
    )
    train["match_text"] = normalize_text(
        train["generation"]
    )
    # --------------------------------------------------------
    # 4. Verify that one generation does not have conflicting
    #    ground-truth labels
    # --------------------------------------------------------
    label_counts = (
        train
        .groupby("match_text")["prompt_label"]
        .nunique()
    )
    conflicting_texts = set(
        label_counts[label_counts > 1].index
    )
    print(
        "Generation texts with conflicting labels:",
        len(conflicting_texts)
    )
    # IMPORTANT:
    # Do NOT silently choose a label if conflicts exist.
    # Remove those texts from benchmark construction.
    if conflicting_texts:
        train = train[
            ~train["match_text"].isin(conflicting_texts)
        ].copy()
    # --------------------------------------------------------
    # 5. generation -> prompt_label mapping
    # --------------------------------------------------------
    mapping = (
        train[
            [
                "match_text",
                "prompt_label",
            ]
        ]
        .drop_duplicates()
    )
    # There should now be one label per normalized text.
    assert not mapping["match_text"].duplicated().any()
    # --------------------------------------------------------
    # 6. Map ground truth onto annotated 9.9k
    # --------------------------------------------------------
    merged = annotated.merge(
        mapping,
        on="match_text",
        how="left",
        validate="many_to_one",
    )
    matched = merged[
        merged["prompt_label"].notna()
    ].copy()
    unmatched = merged[
        merged["prompt_label"].isna()
    ].copy()
    print("\nToxiGen mapping:")
    print("Total annotated :", len(merged))
    print("Matched         :", len(matched))
    print("Unmatched       :", len(unmatched))
    print(
        "Match rate      :",
        f"{len(matched) / len(merged):.2%}"
    )
    # --------------------------------------------------------
    # 7. Standardize ground truth
    #
    # prompt_label:
    # 0 = benign / neutral
    # 1 = harmful / hate
    # --------------------------------------------------------
    matched["label"] = (
        matched["prompt_label"]
        .astype(int)
    )
    matched["text"] = matched["text"].astype(str)
    # --------------------------------------------------------
    # 8. Produce standardized benchmark
    # --------------------------------------------------------
    result = matched[
        [
            "sample_id",
            "text",
            "label",
            "toxicity_human",
            "toxicity_ai"
        ]
    ].copy()
    result.insert(0, "dataset", "toxigen")
    # Defensive checks
    assert result["text"].notna().all()
    assert result["label"].isin([0, 1]).all()
    assert result["sample_id"].is_unique
    return result
# ============================================================
# HateCheck
# ============================================================
def prepare_hatecheck() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Preparing HateCheck")
    print("=" * 70)
    ds = load_dataset(
        "Paul/hatecheck",
        split="test",
    )
    df = ds.to_pandas()
    print("Original rows:", len(df))
    print("Columns:", list(df.columns))
    # Expected:
    # test_case
    # label_gold
    required = {
        "test_case",
        "label_gold",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"HateCheck missing columns: {missing}"
        )
    label_map = {
        "non-hateful": 0,
        "hateful": 1,
    }
    unknown = (
        set(df["label_gold"].dropna().unique())
        - set(label_map)
    )
    if unknown:
        raise ValueError(
            f"Unknown HateCheck labels: {unknown}"
        )
    result = pd.DataFrame({
        "dataset": "hatecheck",
        "sample_id": range(len(df)),
        "text": df["test_case"].astype(str),
        "label": df["label_gold"].map(label_map),
    })
    result["label"] = result["label"].astype(int)
    assert result["label"].isin([0, 1]).all()
    return result
# ============================================================
# AEGIS 2.0
# ============================================================
def prepare_aegis2() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("Preparing AEGIS 2.0")
    print("=" * 70)
    ds = load_dataset(
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        split="test",
    )
    df = ds.to_pandas()
    print("Original rows:", len(df))
    print("Columns:", list(df.columns))
    required = {
        "prompt",
        "prompt_label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"AEGIS missing columns: {missing}"
        )
    # Normalize just the label representation
    labels = (
        df["prompt_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    print(
        "prompt_label values:",
        labels.value_counts(dropna=False)
    )
    label_map = {
        "safe": 0,
        "unsafe": 1,
    }
    unknown = set(labels.unique()) - set(label_map)
    if unknown:
        raise ValueError(
            f"Unknown AEGIS labels: {unknown}"
        )
    result = pd.DataFrame({
        "dataset": "aegis2",
        "sample_id": range(len(df)),
        "text": df["prompt"].astype(str),
        "label": labels.map(label_map),
    })
    result["label"] = result["label"].astype(int)
    assert result["label"].isin([0, 1]).all()
    return result

# ============================================================
# PII Masking benchmarks
# ============================================================
def _parse_privacy_mask(mask_value):
    if mask_value is None or (isinstance(mask_value, float) and pd.isna(mask_value)):
        return []

    if isinstance(mask_value, list):
        return mask_value

    if isinstance(mask_value, str):
        text = mask_value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []

    if isinstance(mask_value, dict):
        return [mask_value]

    return []


def prepare_pii_masking(dataset_name: str, limit: int | None = None) -> pd.DataFrame:
    """
    Prepare AI4Privacy PII masking datasets into the benchmark CSV format.

    Each row retains the original source text plus the ground-truth masked target
    and a parsed privacy_mask list describing PII spans and labels.
    """
    print("\n" + "=" * 70)
    print(f"Preparing {dataset_name}")
    print("=" * 70)

    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()
    if limit is not None:
        df = df.head(limit)

    required = {"source_text", "target_text", "privacy_mask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name} missing columns: {missing}")

    df = df.loc[:, ["source_text", "target_text", "privacy_mask"]].copy()
    df["source_text"] = df["source_text"].fillna("").astype(str)
    df["target_text"] = df["target_text"].fillna("").astype(str)
    df["privacy_mask"] = df["privacy_mask"].map(_parse_privacy_mask)

    # Keep only rows containing sensitive spans; this matches the intention of
    # privacy-masking benchmarks and avoids empty negative examples.
    df = df[df["privacy_mask"].map(bool)].copy()

    df["dataset"] = dataset_name.replace("/", "_").replace("-", "_")
    df["sample_id"] = range(len(df))
    df["text"] = df["source_text"]
    df["label"] = 1

    result = df[
        [
            "dataset",
            "sample_id",
            "text",
            "label",
            "source_text",
            "target_text",
            "privacy_mask",
        ]
    ].copy()

    assert result["sample_id"].is_unique
    assert result["label"].isin([1]).all()
    return result


def prepare_pii_masking_300k(limit: int | None = None) -> pd.DataFrame:
    return prepare_pii_masking("ai4privacy/pii-masking-300k", limit=limit)


def prepare_pii_masking_43k(limit: int | None = None) -> pd.DataFrame:
    return prepare_pii_masking("ai4privacy/pii-masking-43k", limit=limit)
