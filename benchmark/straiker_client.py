import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                date = parsedate_to_datetime(retry_after)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = float("nan")
        if math.isfinite(delay):
            return max(0.0, delay)
    return float(2 ** attempt)


def detect_straiker(
    text: str,
    api_key: str | None,
    base_url: str,
    timeout: float = 60.0,
    debug: bool = False,
) -> dict[str, Any]:
    if not api_key or not api_key.startswith("sk_agt_") or api_key != api_key.strip():
        raise ValueError(
            "Straiker requires a full sk_agt_ integration key. "
            "Set STRAIKER_INTEGRATION_KEY or pass --api-key; PATs and JWTs are not accepted."
        )
    if not base_url:
        raise ValueError(
            "Set STRAIKER_BASE to your tenant's regional API base URL or pass --base-url."
        )
    url = base_url.rstrip("/")
    if not url.endswith("/api/v3/detect"):
        url += "/api/v3/detect"
    payload = {"content": text, "role": "user"}
    if debug:
        print("[DEBUG][straiker.ai] URL:", url)
        print("[DEBUG][straiker.ai] PAYLOAD:", payload)

    for attempt in range(4):
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except httpx.TransportError:
            if attempt == 3:
                raise
            print(f"[straiker.ai] Transport failure; retry {attempt + 1}/3.")
            time.sleep(_retry_delay(None, attempt))
            continue
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt < 3:
                print(f"[straiker.ai] HTTP {response.status_code}; retry {attempt + 1}/3.")
                time.sleep(_retry_delay(response.headers.get("Retry-After"), attempt))
                continue
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or result.get("action") not in ("allow", "block"):
            raise ValueError("Invalid Straiker response: action must be 'allow' or 'block'.")
        return result
    raise RuntimeError("Straiker detect exhausted its request attempts.")
