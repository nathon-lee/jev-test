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
    """Compute PII privacy metrics with realistic masking semantics."""
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

    return {
        "pii_leakage_rate": pii_leakage_rate,
        "masking_success_rate": masking_success_rate,
        "over_masking_rate": over_masking_rate,
        "text_preservation": text_preservation,
    }
