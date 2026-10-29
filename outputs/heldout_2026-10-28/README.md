# Official held-out evaluation rerun

Separate from PR #20's historical-row parity benchmarks. Prepared with PR #19's explicit split/provenance policy. Official held-out membership does not establish absence of overlap with a model's training data; no training leak audit has been completed. Detect Jailbreak and JailbreakHub are excluded pending an agreed joint held-out split. ToxiGen is excluded until its test-label rule is agreed.

## Provider coverage

| provider | status | completed_configurations | expected_configurations |
|---|---|---|---|
| fastino.ai | complete | 6 | 6 |
| laya-ai.com | pending | 2 | 6 |
| typesafe.ai | pending | 0 | 6 |

Only complete provider suites appear below. Pending/error rows are never counted as negatives. Error rates remain visible; future cross-provider comparisons require matching successful rows, not independent successful-subset rankings. Primary threshold is 0.5 and scores use Noul. Threshold variants share the same inference probabilities.

## Classification

| provider | dataset | prompt | valid | errors | expected | accuracy | precision | recall | f1 | auroc | auprc | fpr | fnr | resolved_models |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastino.ai | aegis2 | v1 | 1964 | 0 | 1964 | 0.8371 | 0.8927 | 0.7932 | 0.8400 | 0.9198 | 0.9356 | 0.1116 | 0.2068 | glide_v2 |
| fastino.ai | aegis2 | v2 | 1964 | 0 | 1964 | 0.8315 | 0.8848 | 0.7904 | 0.8349 | 0.9193 | 0.9335 | 0.1204 | 0.2096 | glide_v2 |
| fastino.ai | hatecheck | v1 | 3728 | 0 | 3728 | 0.9769 | 0.9762 | 0.9906 | 0.9833 | 0.9966 | 0.9984 | 0.0532 | 0.0094 | glide_v2 |
| fastino.ai | hatecheck | v2 | 3728 | 0 | 3728 | 0.9933 | 0.9919 | 0.9984 | 0.9951 | 0.9995 | 0.9998 | 0.0180 | 0.0016 | glide_v2 |
| fastino.ai | jailbreak-classification | v1 | 262 | 0 | 262 | 0.9389 | 0.9184 | 0.9712 | 0.9441 | 0.9814 | 0.9888 | 0.0976 | 0.0288 | glide_v2 |
| fastino.ai | jailbreak-classification | v2 | 262 | 0 | 262 | 0.9504 | 0.9315 | 0.9784 | 0.9544 | 0.9817 | 0.9889 | 0.0813 | 0.0216 | glide_v2 |

## PII extraction

GLiNER extraction is a separate model/task from GLiDE classification. Typed span F1 requires exact boundaries and entity type; boundary F1 ignores type. Failed responses count gold entities as missed. No masking/leakage metric is claimed without a corresponding evaluated masking output.

| dataset | model | source_split | valid | errors | expected | typed_span_f1 | boundary_f1 | exact_document_match |
|---|---|---|---|---|---|---|---|---|
| ai4privacy/pii-masking-mini-10k | fastino/gliner2.5-multi-v1 | validation | 990 | 0 | 990 | 0.4888 | 0.5690 | 0.0404 |
| ai4privacy/pii-masking-nano-1k | fastino/gliner2.5-multi-v1 | validation | 90 | 0 | 90 | 0.4621 | 0.5333 | 0.0000 |

## Classification source partitions

| dataset | source_repository | source_revision | source_split | rows | csv_sha256 |
|---|---|---|---|---|---|
| hatecheck | Paul/hatecheck | 9d2ac89df04254e5c427bcc8d61b6d6c83a1f59b | test | 3728 | c2a5c8928ae38b9e2150517291df3f5460029544aff4a057b8f4e01a4282c36c |
| aegis2 | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | d86bb8bedff51d25ac834ab7838f1cc61acb7a2c | test | 1964 | 8b37befa9221dcb4bb1d2f6c1ec418e121931dda00c67c94763ee141149aa4b9 |
| jailbreak-classification | jackhhao/jailbreak-classification | 2f2ceeb39658696fd3f462403562b6eea5306287 | test | 262 | c6eb3a7e4e566f27011e300fad471b09cfd6ed177eaef4c6569b5c4c2707d2c8 |
