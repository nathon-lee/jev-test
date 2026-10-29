# Corrected GLiDE/Laya classification rerun

These are new live requests scored strictly from `answers.score.noul`. Historical Fastino/Laya confidence-based results are not included. The old output directories remain historical snapshots.

Each probability is requested once per row and prompt. Thresholds 0.5–0.8 are computed offline; they are not independent repeat inferences, and no repeatability claim is made. All four thresholds are retained in the JSON summary; 0.5 is the fixed primary classification baseline.

## Classification quality at threshold 0.5

Jev columns use the saved `jev-latest` outputs on exactly the same common-success rows, with source-text and label checks. Jev is a historical reference, not a contemporaneous rerun; its resolved model version is unknown.

| Provider | Dataset | Prompt | Success / expected | Paired rows | Excluded NUL rows | New F1 | Historical Jev F1 | New AUROC | Historical Jev AUROC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fastino.ai | aegis2 | v1 | 1964 / 1964 | 1964 | 0 | 0.8400 | 0.8159 | 0.9198 | 0.9049 |
| fastino.ai | aegis2 | v2 | 1964 / 1964 | 1964 | 0 | 0.8349 | 0.7967 | 0.9193 | 0.9054 |
| fastino.ai | detect-jailbreak | v1 | 18258 / 18258 | 18257 | 1 | 0.3966 | 0.5313 | 0.9419 | 0.9285 |
| fastino.ai | detect-jailbreak | v2 | 18258 / 18258 | 18257 | 1 | 0.3993 | 0.5015 | 0.9311 | 0.9195 |
| fastino.ai | hatecheck | v1 | 3728 / 3728 | 3728 | 0 | 0.9833 | 0.9633 | 0.9966 | 0.9928 |
| fastino.ai | hatecheck | v2 | 3728 / 3728 | 3728 | 0 | 0.9951 | 0.9636 | 0.9995 | 0.9943 |
| fastino.ai | jailbreak-classification | v1 | 1044 / 1044 | 1044 | 0 | 0.9494 | 0.9506 | 0.9932 | 0.9943 |
| fastino.ai | jailbreak-classification | v2 | 1044 / 1044 | 1044 | 0 | 0.9581 | 0.9516 | 0.9941 | 0.9948 |
| fastino.ai | jailbreakhub | v1 | 15140 / 15140 | 15140 | 0 | 0.3367 | 0.4257 | 0.9142 | 0.8973 |
| fastino.ai | jailbreakhub | v2 | 15140 / 15140 | 15140 | 0 | 0.3653 | 0.4233 | 0.9046 | 0.8937 |
| fastino.ai | toxigen | v1 | 6789 / 6789 | 6789 | 0 | 0.4815 | 0.6587 | 0.8400 | 0.8752 |
| fastino.ai | toxigen | v2 | 6789 / 6789 | 6789 | 0 | 0.6301 | 0.6928 | 0.8504 | 0.8746 |

Historical pandas versions truncated text at NUL while the current parser preserves it. Such rows are excluded from the paired Jev comparison only when the historical text exactly matches the prefix before the first NUL. Their fresh results remain in the full-run CSV/metrics. Every other text or label mismatch fails validation.

- fastino.ai / detect-jailbreak / v1: excluded legacy NUL sample IDs 15837.
- fastino.ai / detect-jailbreak / v2: excluded legacy NUL sample IDs 15837.

## PII detection diagnostics

These prepared PII datasets have all-positive document labels. Detection rate is not entity/span extraction F1 and does not measure redaction, leakage, or false-positive behavior. No extraction result is claimed for GLiDE or Jev here.

| Provider | Dataset | Prompt | Success / expected | Positive detection rate |
|---|---|---|---:|---:|
| fastino.ai | pii_masking_mini_10k | v1 | 9000 / 9000 | 0.9262 |
| fastino.ai | pii_masking_mini_10k | v2 | 9000 / 9000 | 0.9530 |
| fastino.ai | pii_masking_nano_1k | v1 | 900 / 900 | 0.9233 |
| fastino.ai | pii_masking_nano_1k | v2 | 900 / 900 | 0.9456 |

## Request timing and provenance

Timing measures wall-clock request latency including retries under the recorded concurrent workload. It is not compared with historical serial Jev timing. Warm-up, hosting, and client/network conditions can differ.

| Provider | Dataset | Prompt | Returned model(s) | Mean ms | p50 ms | p95 ms | Workers |
|---|---|---|---|---:|---:|---:|---:|
| fastino.ai | aegis2 | v1 | glide_v2 | 498.5629 | 447.8188 | 837.0714 | 32 |
| fastino.ai | aegis2 | v2 | glide_v2 | 491.8338 | 444.5702 | 799.5886 | 32 |
| fastino.ai | detect-jailbreak | v1 | glide_v2 | 504.6368 | 471.1110 | 765.7865 | 32 |
| fastino.ai | detect-jailbreak | v2 | glide_v2 | 549.4541 | 470.4219 | 829.2212 | 32 |
| fastino.ai | hatecheck | v1 | glide_v2 | 558.4088 | 490.7888 | 1088.8150 | 32 |
| fastino.ai | hatecheck | v2 | glide_v2 | 503.0429 | 448.5488 | 823.6389 | 32 |
| fastino.ai | jailbreak-classification | v1 | glide_v2 | 605.2778 | 531.5999 | 1221.1739 | 32 |
| fastino.ai | jailbreak-classification | v2 | glide_v2 | 568.5412 | 498.1759 | 1095.6176 | 32 |
| fastino.ai | jailbreakhub | v1 | glide_v2 | 509.4604 | 461.9128 | 858.4194 | 32 |
| fastino.ai | jailbreakhub | v2 | glide_v2 | 553.1703 | 469.8756 | 1041.7241 | 32 |
| fastino.ai | pii_masking_mini_10k | v1 | glide_v2 | 451.9370 | 425.3704 | 696.9498 | 32 |
| fastino.ai | pii_masking_mini_10k | v2 | glide_v2 | 500.1238 | 470.9615 | 756.0167 | 32 |
| fastino.ai | pii_masking_nano_1k | v1 | glide_v2 | 563.3786 | 512.0003 | 1082.9768 | 32 |
| fastino.ai | pii_masking_nano_1k | v2 | glide_v2 | 505.5566 | 455.7670 | 899.4038 | 32 |
| fastino.ai | toxigen | v1 | glide_v2 | 488.3889 | 447.1087 | 785.4195 | 32 |
| fastino.ai | toxigen | v2 | glide_v2 | 537.7799 | 477.3067 | 1040.6861 | 32 |

Per-configuration manifests record exact source-file and submitted-input hashes, prompt text, requested/returned model IDs, timestamps, coverage, and threshold metrics. CSVs retain corrected scores, diagnostic raw confidence, response status, attempts, and per-call latency. JSONL checkpoints support resuming and retain token usage where returned. Missing or failed rows are not counted as correct predictions.

Laya is not included in this snapshot; its results will be added after its run completes. These results use Nathan's existing prepared rows and label mappings, not the Notion experiment's separately sampled evaluation sets.
