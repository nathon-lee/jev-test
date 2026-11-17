---
license: other
license_name: cc-by-4.0
language:
  - en
  - fr
  - de
  - es
  - it
  - nl
  - pt
  - bg
  - cs
  - da
  - el
  - et
  - fi
  - hr
  - hu
  - lt
  - lv
  - pl
  - ro
  - sk
  - sl
  - sr
  - sv
  - id
  - ja
  - ko
  - ms
  - tl
  - vi
  - zh
task_categories:
  - token-classification
  - text-generation
tags:
  - privacy
  - pii
  - ner
  - synthetic
  - multilingual
  - ai4privacy
  - openpii
  - asia-pacific
pretty_name: "PII Masking Nano: Multilingual Sample"
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
---

# PII Masking Nano: Multilingual Sample

<p align="center">
  <img src="assets/pii_masking_3m_coverage.png" alt="PII-Masking-3M Coverage" width="500"/>
</p>

A **nano-sized** stratified sample of [pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m),
the flagship release of the **PII-Masking-3M** family. Sampled proportionally by
`(source_dataset, language)` so every locale and label gets representation.
Asia Pacific rows appear first.

📖 **More information:** [www.ai4privacy.com/datasets/pii-masking-3m-asia-pacific](https://www.ai4privacy.com/datasets/pii-masking-3m-asia-pacific/)

## Dataset Details

| Total Examples | Train | Validation | Labels | Languages | Regions | Annotations | Format | License |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **990** | 900 | 90 | 19 | 30 | 37 | 7,611 | JSONL | CC-BY-4.0 |

## Language Coverage

<p align="center">
  <img src="assets/language_distribution.png" alt="Language Distribution" width="800"/>
</p>

## Label Taxonomy (19 Labels)

<p align="center">
  <img src="assets/label_distribution.png" alt="Label Distribution" width="800"/>
</p>

| Label | Count | Label | Count | Label | Count |
| :---|---: | :---|---: | :---|---: |
| `GIVENNAME` | 864 | `STREET` | 386 | `SEX` | 228 |
| `DATE` | 856 | `BUILDINGNUM` | 382 | `TAXNUM` | 226 |
| `SURNAME` | 737 | `AGE` | 371 | `DRIVERLICENSENUM` | 203 |
| `EMAIL` | 545 | `ZIPCODE` | 337 | `PASSPORTNUM` | 175 |
| `CITY` | 527 | `IDCARDNUM` | 262 | `SOCIALNUM` | 170 |
| `TITLE` | 461 | `CREDITCARDNUMBER` | 246 |  |  |
| `TELEPHONENUM` | 405 | `GENDER` | 230 |  |  |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("ai4privacy/pii-masking-nano-1k")
```

## PII-Masking-3M Tiers

| Tier | Size | Link |
|:---|:---|:---|
| Nano | 1k | [pii-masking-nano-1k](https://huggingface.co/datasets/ai4privacy/pii-masking-nano-1k) |
| Mini | 10k | [pii-masking-mini-10k](https://huggingface.co/datasets/ai4privacy/pii-masking-mini-10k) |
| Micro | 100k | [pii-masking-micro-100k](https://huggingface.co/datasets/ai4privacy/pii-masking-micro-100k) |
| Full | 1.5M | [pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) |

## License

CC-BY-4.0. Copyright © 2026 Ai Suisse SA. Contains **synthetic PII only**, no real personal data.

## Legal Disclaimer

Provided "as is" without warranties of any kind. Users are responsible for compliance with all
applicable data privacy laws (GDPR, CCPA, PDPA, APPI, PIPA, UU PDP) and AI regulations.
