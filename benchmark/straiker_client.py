import json
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


def validate_straiker_config(api_key: str | None, base_url: str) -> str:
    if not api_key:
        credential_type = "missing"
    elif api_key != api_key.strip():
        credential_type = "whitespace"
    elif api_key.startswith("s6r_pat_"):
        credential_type = "PAT"
    elif api_key.startswith("eyJ"):
        credential_type = "JWT"
    elif api_key.startswith("sk_agt_") and len(api_key) > len("sk_agt_"):
        credential_type = "integration_key"
    else:
        credential_type = "unknown"
    if credential_type != "integration_key":
        raise ValueError(
            f"Straiker configuration error: credential_type={credential_type}. "
            "Straiker requires a full sk_agt_ integration key without surrounding whitespace. "
            "Set STRAIKER_INTEGRATION_KEY or pass --api-key. A new PAT is still a PAT: "
            "exchange it for a JWT, create an integration, then issue its integration key."
        )
    if not base_url:
        raise ValueError(
            "Set STRAIKER_BASE to your tenant's regional API base URL or pass --base-url."
        )
    try:
        parsed_url = httpx.URL(base_url)
    except httpx.InvalidURL:
        raise ValueError("Invalid STRAIKER_BASE: expected an HTTP(S) API base URL.") from None
    if (
        parsed_url.scheme not in ("http", "https") or not parsed_url.host
        or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment
    ):
        raise ValueError(
            "Invalid STRAIKER_BASE: use an HTTP(S) API base URL without "
            "credentials, a query string, or a fragment."
        )
    url = str(parsed_url).rstrip("/")
    if not url.endswith("/api/v3/detect"):
        url += "/api/v3/detect"
    return url


def straiker_error_message(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        hint = {
            401: "Check the integration key, its revocation status, and tenant region; PATs/JWTs cannot call detect.",
            403: "Access denied; check integration access with your Straiker administrator.",
            404: "Check the regional API base URL and /api/v3/detect endpoint.",
            429: "Rate limit reached; reduce concurrency and confirm tenant limits.",
        }.get(status, "Straiker returned an HTTP error; contact support if it persists.")
        return f"HTTP {status}: {hint}"
    if isinstance(error, httpx.RequestError):
        return f"{type(error).__name__}: Straiker request failed; check connectivity, DNS, TLS, and timeouts."
    if isinstance(error, json.JSONDecodeError):
        return "Invalid Straiker response: expected JSON, but the response could not be decoded."
    if isinstance(error, ValueError):
        return "Invalid Straiker response: expected a JSON object with action 'allow' or 'block'."
    return f"{type(error).__name__}: Straiker request processing failed."


def detect_straiker(
    text: str,
    api_key: str | None,
    base_url: str,
    timeout: float = 60.0,
    debug: bool = False,
) -> dict[str, Any]:
    url = validate_straiker_config(api_key, base_url)
    payload = {"content": text, "role": "user"}

    for attempt in range(4):
        started = time.perf_counter()
        if debug:
            print(
                f"[DEBUG][straiker.ai] POST {url} attempt={attempt + 1}/4 "
                f"role=user content_chars={len(text)}"
            )
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
        except httpx.TransportError as exc:
            if attempt == 3:
                raise
            print(f"[straiker.ai] {type(exc).__name__}; retry {attempt + 1}/3.")
            time.sleep(_retry_delay(None, attempt))
            continue
        if debug:
            elapsed_ms = (time.perf_counter() - started) * 1000
            print(
                f"[DEBUG][straiker.ai] HTTP {response.status_code} "
                f"attempt={attempt + 1}/4 elapsed_ms={elapsed_ms:.2f}"
            )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt < 3:
                print(f"[straiker.ai] HTTP {response.status_code}; retry {attempt + 1}/3.")
                time.sleep(_retry_delay(response.headers.get("Retry-After"), attempt))
                continue
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or result.get("action") not in ("allow", "block"):
            raise ValueError("Invalid Straiker response: action must be 'allow' or 'block'.")
        if debug:
            print(f"[DEBUG][straiker.ai] action={result['action']}")
        return result
    raise RuntimeError("Straiker detect exhausted its request attempts.")
