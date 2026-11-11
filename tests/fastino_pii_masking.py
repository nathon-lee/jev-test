
import os
import json
import requests

API_URL = "https://api.fastino.ai/v1/chat/completions"
API_KEY = os.environ["FASTINO_API_KEY"]
MODEL = "fastino/gliguard-PII-multi"

# PII categories
PII_LABELS = [
    "person",
    "email",
    "phone_number",
    "address",

    # Government identifiers
    "national_id",
    "passport_number",
    "tax_identification_number",
    "driver_license_number",
    "social_security_number",

    # Financial information
    "credit_card",
    "bank_account_number",
    "iban",

    # Personal information
    "date_of_birth",
    "date",
    "age",
    "gender",

    # Digital identifiers
    "ip_address",
    "username",
]


def mask_pii(text):

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }

    payload = {
        "model": MODEL,

        "messages": [
            {
                "role": "system",
                "content": """
You are a multilingual PII detection system.

Identify personally identifiable information (PII)
in the input text.

Support multilingual PII detection, including
English, Indonesian, and Malay.

Important Indonesian terminology:

- SIM means Surat Izin Mengemudi
  (Indonesian driver's license).
- Nomor SIM means driver's license number.
- Nomor pajak means tax identification number.
- KTP means Indonesian national identity card.
- NIK means Nomor Induk Kependudukan
  (Indonesian national identification number).

Detection requirements:

1. Extract exact text spans.
2. Do not translate or normalize extracted text.
3. Do not infer information that is not explicitly present.
4. Distinguish ordinary dates from dates of birth.
5. Identify driver's license numbers when explicitly
   associated with SIM or driver's license context.
6. Identify tax identification numbers when associated
   with tax-related context.
"""
            },
            {
                "role": "user",
                "content": text,
            }
        ],

        "schema": {
            "entities": PII_LABELS
        },

        "include_confidence": True,
        "include_spans": True,
    }

    # Call Fastino API
    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text}"
        )

    data = response.json()

    # Parse model output
    content = data["choices"][0]["message"]["content"]

    if isinstance(content, str):
        result = json.loads(content)
    else:
        result = content

    entities = result.get("entities", {})

    # Collect detected spans
    spans = []

    for label, items in entities.items():

        for item in items:

            start = item.get("start")
            end = item.get("end")

            if not isinstance(start, int) or not isinstance(end, int):
                continue

            if not (0 <= start < end <= len(text)):
                continue

            # Validate that the span matches the original text
            if "text" in item and text[start:end] != item["text"]:
                print(
                    f"Warning: span mismatch for {label}: "
                    f"{item['text']}"
                )
                continue

            spans.append((start, end, label))

    # Resolve overlapping spans:
    # Keep the longer span if two entities overlap.
    spans.sort(
        key=lambda x: (x[1] - x[0], x[0]),
        reverse=True
    )

    selected_spans = []

    for candidate in spans:
        start, end, label = candidate

        if any(
            start < existing_end and end > existing_start
            for existing_start, existing_end, _ in selected_spans
        ):
            continue

        selected_spans.append(candidate)

    # Replace from right to left
    masked_text = text

    for start, end, label in sorted(
        selected_spans,
        key=lambda x: x[0],
        reverse=True
    ):
        masked_text = (
            masked_text[:start]
            + f"[{label.upper()}]"
            + masked_text[end:]
        )

    return {
        "original_text": text,
        "entities": entities,
        "masked_text": masked_text,
        "model": MODEL,
        "api_usage": data.get("usage", {}),
    }


def main():

    print("=" * 65)
    print("Fastino PII Detection & Masking")
    print("Model:", MODEL)
    print("Languages: English / Indonesian / Malay")
    print("Enter 'exit' or 'quit' to stop.")
    print("=" * 65)

    while True:

        print("\nPlease input your text:")

        text = input("> ").strip()

        if text.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        if not text:
            print("Input cannot be empty.")
            continue

        try:

            result = mask_pii(text)

            print("\n=== Detected PII ===")

            print(json.dumps(
                result["entities"],
                indent=2,
                ensure_ascii=False
            ))

            print("\n=== Masked Text ===")
            print(result["masked_text"])

            print("\n=== API Usage ===")
            print(json.dumps(
                result["api_usage"],
                indent=2
            ))

        except Exception as e:

            print("\nRequest failed:")
            print(str(e))


if __name__ == "__main__":
    main()