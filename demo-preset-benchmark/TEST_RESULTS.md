# Demo cluster verification results

Tested on 29 October 2026 against `https://api.demo.sao.splunkcloud.com`. Both original presets returned valid model scores through Agent Control. All 1,990 requested rows were scored successfully, with zero errors.

Project: `b33d2a32-5320-47ef-b351-cf4c4fb84a17` (`galileo-demo`).

| Metric / category | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Prompt Injection | 1,000 | 30.16% | 87.96% | 44.92% |
| PII: credit_card_info | 990 | 100.00% | 57.08% | 72.68% |
| PII: email | 990 | 100.00% | 94.20% | 97.01% |
| PII: phone_number | 990 | 94.43% | 93.68% | 94.06% |
| PII: ssn | 990 | 95.00% | 29.38% | 44.88% |
| PII: micro average, four classes | 990 | 97.66% | 77.60% | 86.48% |

PII macro F1: **77.16%**. This is a document-level comparison of four mapped categories, not an overall entity extraction score. See README.md for the mapping and its limitations.

PI confusion counts: TP 95, FP 220, FN 13, TN 672. Threshold: 0.5.

## Original preset versions

| Metric | Preset name | Scorer ID | Version ID |
|---|---|---|---|
| PI | `prompt_injection_luna` | `11d765b1-a625-44e9-b0ad-a64cd3b68932` | `10e8c94c-28f9-437f-b22c-5b41af8eb7f2` |
| PII | `input_pii` | `283321c0-e7e6-4be1-a7bc-e0fcc2b91d4d` | `14153c77-5bc9-4699-98e7-fae5a50d1185` |

No prompts, preset versions, or weights were changed. The script created isolated benchmark streams and observing controls in the existing project. The API identifies these as SLM presets; the cluster determines the model deployment behind them.

The 1,000-row PI sample matches the prior benchmark sample exactly (108 positive and 892 negative labels). PII covers all 990 validation documents, with 33 documents in each of 30 languages.

## PII by language

Each class cell shows F1 and the number of reference-positive documents in parentheses. Every language has 33 successfully scored documents.

| Language | Credit card F1 (support) | Email F1 (support) | Phone F1 (support) | Social ID F1 (support) | Micro F1 |
|---|---:|---:|---:|---:|---:|
| bg | 75.00% (5) | 97.44% (20) | 100.00% (16) | 0.00% (3) | 92.68% |
| cs | 73.68% (12) | 94.74% (20) | 92.86% (14) | 28.57% (6) | 84.78% |
| da | 71.43% (9) | 100.00% (12) | 96.00% (12) | 57.14% (5) | 88.57% |
| de | 54.55% (8) | 88.00% (14) | 95.24% (11) | 36.36% (9) | 76.47% |
| el | 87.50% (9) | 96.55% (15) | 88.89% (14) | 50.00% (9) | 85.71% |
| en | 72.73% (7) | 96.30% (14) | 90.91% (12) | 50.00% (6) | 85.29% |
| es | 90.91% (6) | 97.14% (18) | 96.55% (14) | 100.00% (4) | 96.39% |
| et | 80.00% (9) | 97.30% (19) | 96.00% (13) | 0.00% (4) | 88.89% |
| fi | 71.43% (9) | 92.31% (14) | 92.86% (15) | 0.00% (6) | 81.08% |
| fr | 80.00% (12) | 96.55% (15) | 90.00% (10) | 66.67% (10) | 85.71% |
| hr | 57.14% (5) | 100.00% (17) | 96.77% (15) | 0.00% (3) | 90.67% |
| hu | 40.00% (12) | 96.97% (17) | 96.55% (14) | 0.00% (6) | 79.52% |
| id | 72.73% (7) | 95.65% (12) | 96.30% (13) | 75.00% (5) | 89.86% |
| it | 71.43% (9) | 92.31% (21) | 83.87% (18) | 88.89% (5) | 86.02% |
| ja | 36.36% (9) | 100.00% (10) | 93.33% (15) | 28.57% (6) | 79.41% |
| ko | 60.00% (7) | 96.55% (15) | 90.00% (10) | 22.22% (7) | 79.41% |
| lt | 80.00% (6) | 100.00% (19) | 100.00% (15) | 16.67% (11) | 86.67% |
| lv | 54.55% (8) | 100.00% (15) | 91.67% (11) | 40.00% (8) | 82.67% |
| ms | 60.00% (7) | 100.00% (14) | 100.00% (8) | 80.00% (8) | 89.86% |
| nl | 90.00% (11) | 100.00% (14) | 95.24% (10) | 50.00% (6) | 90.91% |
| pl | 94.12% (9) | 100.00% (19) | 94.12% (8) | 18.18% (9) | 86.75% |
| pt | 66.67% (6) | 87.50% (18) | 87.50% (9) | 80.00% (6) | 83.58% |
| ro | 72.73% (7) | 100.00% (14) | 100.00% (12) | 28.57% (6) | 88.57% |
| sk | 75.00% (5) | 92.68% (22) | 100.00% (12) | 0.00% (5) | 87.18% |
| sl | 66.67% (4) | 100.00% (16) | 95.24% (10) | 33.33% (10) | 84.51% |
| sr | 71.43% (9) | 100.00% (12) | 96.55% (14) | 25.00% (7) | 85.33% |
| sv | 75.00% (5) | 96.30% (14) | 82.35% (8) | 22.22% (8) | 78.69% |
| tl | 76.92% (8) | 97.14% (18) | 89.66% (13) | 50.00% (6) | 87.06% |
| vi | 83.33% (7) | 100.00% (18) | 96.97% (17) | 100.00% (4) | 96.63% |
| zh | 80.00% (6) | 100.00% (17) | 90.32% (17) | 50.00% (6) | 89.16% |

## Verification

Overall and per-language precision, recall, and F1 were independently checked with scikit-learn. Every response matched its requested scorer, pinned version, and control ID. Checkpoint resume was tested, including multilingual inputs with Unicode line separators. The presets retained their original default versions.

Test runtime: Python 3.14.7. Dependencies: `httpx==0.28.1`, `truststore==0.10.5`, `python-dotenv==1.2.4`, `pyarrow==26.0.0`, `scikit-learn==1.9.1`.

The shared package contains no API key, bearer token, local `.env`, downloaded dataset, or prediction files.
