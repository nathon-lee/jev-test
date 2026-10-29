"""Explicit, revision-pinned evaluation partitions shared by all datasets."""

import json
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset
from huggingface_hub import HfApi

PROVENANCE_COLUMNS = (
    "source_repository", "source_config", "source_revision", "source_split",
    "partition_role", "source_row",
)
HELD_OUT_SPLITS = ("test", "validation", "valid", "dev")


def select_split(available, requested=None, allow_train_eval=False):
    available = list(available)
    if requested is None:
        requested = next((name for name in HELD_OUT_SPLITS if name in available), None)
        if requested is None and allow_train_eval and "train" in available:
            requested = "train"
    if requested not in available:
        raise ValueError(
            f"No requested/official held-out split available; available={available}. "
            "Select an existing split explicitly. Train-only reference evaluation "
            "requires --allow-train-eval; it is not a held-out benchmark."
        )
    if requested == "train":
        if not allow_train_eval:
            raise ValueError("Training data requires explicit --allow-train-eval")
        return requested, "training_reference"
    if requested not in HELD_OUT_SPLITS:
        raise ValueError("Only declared test/validation/valid/dev splits are accepted as held-out evaluation")
    return requested, "held_out"


def resolve_source_revision(repository, revision=None):
    """Resolve an explicit override or the version-controlled benchmark source pin."""
    if revision is None:
        pins = json.loads((Path(__file__).resolve().parents[1] / "data/source-revisions.json").read_text())
        revision = pins.get(repository)
        if revision is None:
            raise ValueError(f"No committed revision for {repository}; provide --revision explicitly")
    return HfApi().dataset_info(repository, revision=revision).sha


def load_evaluation_partition(repository, config=None, *, split=None, revision=None, allow_train_eval=False):
    resolved = resolve_source_revision(repository, revision)
    available = get_dataset_split_names(repository, config_name=config, revision=resolved)
    selected, role = select_split(available, split, allow_train_eval)
    dataset = load_dataset(repository, config, split=selected, revision=resolved)
    actual_config = getattr(getattr(dataset, "info", None), "config_name", None) or config or "default"
    provenance = {"source_repository": repository, "source_config": actual_config,
                  "source_revision": resolved, "source_split": selected,
                  "partition_role": role, "available_splits": available,
                  "source_rows": len(dataset)}
    return dataset, provenance


def attach_provenance(frame, provenance):
    frame = frame.copy()
    for name in PROVENANCE_COLUMNS:
        if name != "source_row":
            frame[name] = provenance[name]
    # Preparation preserves source row order/index through filtering. Keep that
    # identity as well as the existing suite sample_id convention.
    frame["source_row"] = frame.index
    frame.attrs["evaluation_partition"] = {**provenance, "prepared_rows": len(frame)}
    return frame
