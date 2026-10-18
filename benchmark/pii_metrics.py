import re
from difflib import SequenceMatcher
from typing import Any


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_sensitive_values(privacy_mask: Any):
    if privacy_mask is None:
        return []
    if isinstance(privacy_mask, list):
        items = privacy_mask
    elif isinstance(privacy_mask, dict):
        items = [privacy_mask]
    else:
        items = []

    values = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is not None:
            values.append(str(value))
    return values


def _mask_sensitive_values(text: str, sensitive_values):
    cleaned = text
    for value in sorted(sensitive_values, key=len, reverse=True):
        if not value:
            continue
        cleaned = re.sub(re.escape(value), " PII ", cleaned, flags=re.IGNORECASE)
    return _normalize_spaces(cleaned)


def _count_mask_tokens(text: str):
    return len(re.findall(r"\[[A-Z0-9 _-]+\]", text or ""))


def compute_pii_metrics(
    source_text: str,
    predicted_text: str,
    target_text: str,
    privacy_mask: Any,
):
    """Compute PII privacy metrics with realistic masking semantics.

    Why not use Accuracy / Precision / Recall / AUROC here?
    ------------------------------------------------------
    The PII dataset is constructed as a masking task, not a conventional binary
    classification task. Each row is effectively a positive case (contains
    sensitive values to be masked), so a standard label-only metric is not a
    meaningful performance signal:

      - all labels may be 1, causing class imbalance and single-class warnings
      - AUROC becomes undefined when y_true has only one class
      - a model can score "high" on accuracy simply because every sample is
        treated as positive, even if it fails to redact the actual sensitive
        spans

    For PII evaluation, what matters is whether the model actually hides the
    sensitive values, whether it leaks them, whether it over-masks unrelated
    text, and whether it preserves the non-sensitive content. These are the
    reasons we report leakage, masking success, over-masking, and text
    preservation instead of binary-classification metrics.

    We additionally report span-level precision/recall/F1 because those are the
    closest analogues to entity-level masking quality: they tell us whether the
    model redacted the right sensitive spans without introducing extra masking.
    """
    source = source_text or ""
    predicted = predicted_text or ""
    target = target_text or ""
    sensitive_values = _extract_sensitive_values(privacy_mask)

    if not sensitive_values:
        return {
            "pii_leakage_rate": 0.0,
            "masking_success_rate": 1.0,
            "over_masking_rate": 0.0,
            "text_preservation": 1.0,
            "span_precision": 1.0,
            "span_recall": 1.0,
            "span_f1": 1.0,
        }

    leaked = 0
    successfully_masked = 0
    for value in sensitive_values:
        if value and re.search(re.escape(value), predicted, flags=re.IGNORECASE):
            leaked += 1
        if value and not re.search(re.escape(value), predicted, flags=re.IGNORECASE):
            successfully_masked += 1

    pii_leakage_rate = leaked / len(sensitive_values)
    masking_success_rate = successfully_masked / len(sensitive_values)

    predicted_placeholder_count = _count_mask_tokens(predicted)
    target_placeholder_count = _count_mask_tokens(target)
    extra_mask_tokens = max(predicted_placeholder_count - target_placeholder_count, 0)
    over_masking_rate = extra_mask_tokens / max(predicted_placeholder_count, 1)

    source_clean = _mask_sensitive_values(source, sensitive_values)
    predicted_clean = re.sub(r"\[[A-Z0-9 _-]+\]", " PII ", predicted or "")
    predicted_clean = _normalize_spaces(predicted_clean)
    if not source_clean:
        text_preservation = 1.0
    else:
        text_preservation = SequenceMatcher(None, source_clean, predicted_clean).ratio()

    # Span-level metrics reflect the entity-masking task more faithfully than
    # binary classification metrics. We treat each sensitive value as a discrete
    # span to mask. Precision answers: "of the spans we masked, how many were
    # actually sensitive?" Recall answers: "of all sensitive spans, how many did
    # we successfully mask?" F1 balances the two.
    true_positives = successfully_masked
    false_negatives = leaked
    false_positives = max(predicted_placeholder_count - target_placeholder_count, 0)
    span_precision = true_positives / max(true_positives + false_positives, 1)
    span_recall = true_positives / max(len(sensitive_values), 1)
    span_f1 = 2 * span_precision * span_recall / max(span_precision + span_recall, 1e-9)

    return {
        "pii_leakage_rate": pii_leakage_rate,
        "masking_success_rate": masking_success_rate,
        "over_masking_rate": over_masking_rate,
        "text_preservation": text_preservation,
        "span_precision": span_precision,
        "span_recall": span_recall,
        "span_f1": span_f1,
    }
