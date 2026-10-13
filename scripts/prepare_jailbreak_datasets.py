from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "prepared"


def pick_text_column(df: pd.DataFrame) -> str:
    for col in [
        "Goal",
        "goal",
        "prompt",
        "text",
        "Target",
        "target_response",
        "user_input",
        "instruction",
    ]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find text column in {df.columns.tolist()}")


def pick_label_column(df: pd.DataFrame) -> str:
    for col in [
        "human_majority",
        "human_majority_label",
        "label",
        "binary_label",
        "harmbench_cf",
        "gpt4_cf",
        "llamaguard2_cf",
        "llama3_cf",
    ]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find label column in {df.columns.tolist()}")


def normalize_label(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "safe": 0,
        "unsafe": 1,
        "benign": 0,
        "harmful": 1,
        "non-hateful": 0,
        "hateful": 1,
        "non_hateful": 0,
        "hateful_content": 1,
        "negative": 0,
        "positive": 1,
        "false": 0,
        "true": 1,
    }
    out = values.map(mapping)
    if out.isna().any():
        unknown = sorted(set(values[~out.notna()]))
        raise ValueError(f"Unknown label values: {unknown}")
    return out.astype(int)


def standardize(file_path: Path, dataset_name: str, default_label: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "sample_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "sample_id"})

    text_col = pick_text_column(df)
    text = df[text_col].fillna("").astype(str)

    if default_label is not None:
        label = pd.Series([default_label] * len(df), index=df.index)
    else:
        label_col = pick_label_column(df)
        label = normalize_label(df[label_col])

    out = pd.DataFrame({
        "dataset": dataset_name,
        "sample_id": df["sample_id"].astype(int),
        "text": text,
        "label": label.astype(int),
    })

    return out


def main():
    files = [
        (DATA_DIR / "benign-behaviors.csv", "benign-behaviors", 0),
        (DATA_DIR / "harmful-behaviors.csv", "harmful-behaviors", 1),
        (DATA_DIR / "judge-comparison.csv", "judge-comparison", None),
    ]

    frames = []
    for file_path, dataset_name, default_label in files:
        print(f"Processing {file_path.name}")
        frames.append(standardize(file_path, dataset_name, default_label))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["dataset", "sample_id", "text", "label"]]
    combined = combined.reset_index(drop=True)
    combined["sample_id"] = range(len(combined))

    output_path = DATA_DIR / "jailbreak_all.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(combined)}")
    print(combined["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
