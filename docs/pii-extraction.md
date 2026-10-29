# PII extraction benchmarking

The classification suite asks whether a document contains PII. AI4Privacy masking
rows are mostly positive, so binary accuracy/F1 cannot measure whether a model
found the right entities. Extraction evaluates typed character spans against the
original annotations. It does not infer success from missing sensitive strings
in a generated response, count placeholders, or interpret a decision probability
as redacted text.

## Prepare one reproducible evaluation partition

Run commands from the repository root after installing `requirements.txt`:

```bash
python -m scripts.pii_extraction prepare \
  --dataset ai4privacy/pii-masking-nano-1k \
  --split validation \
  --revision main \
  --output data/pii_extraction/nano-validation.jsonl
```

`main` is resolved to a Hugging Face commit SHA **before** loading data. Each row
records the dataset, exact revision, split, stable `uid` as `sample_id`, source
text, and gold spans. Reuse that SHA for subsequent runs. Mini uses the same
command with `ai4privacy/pii-masking-mini-10k`. Validation is the extraction
default; `--split train` is explicit for development. Do not mix train/full/test
results or tune thresholds on the evaluation partition. `--limit N` takes the
first N rows, so it is a smoke test, not a representative multilingual sample.

Old `data/prepared/pii_masking_*.csv` files contain empty `privacy_mask` lists even
when their masked targets change. Regenerate them or use the new preparation
command. The old preparation helper now preserves Arrow/numpy array annotations
instead of silently replacing them with `[]`. Invalid/missing annotations fail
validation. Empty gold is valid only when explicitly annotated; a changed masked
target with empty gold is rejected. No spans are guessed from string differences.

## Run hosted Fastino extraction

Set `FASTINO_API_KEY` in the environment, then run:

```bash
python -m scripts.pii_extraction run \
  --input data/pii_extraction/nano-validation.jsonl \
  --model fastino/gliner2.5-multi-v1 \
  --threshold 0.5 \
  --output outputs/pii_extraction/fastino-nano-validation.jsonl \
  --summary outputs/pii_extraction/fastino-nano-validation.metrics.json
```

This calls the documented [Fastino GLiNER chat API](https://docs.fastino.ai/inference/chat-completions)
with a fixed 19-type AI4Privacy entity schema and requests half-open character
spans. Only the source text and fixed schema are sent; gold spans and masked
targets are never sent to the model. TLS verification is enabled, inference
storage is disabled, and input truncation is disabled. Truncated responses are
failures. `FASTINO_BASE_URL` or `--base-url` can select a compatible deployment.

The default model is a multilingual GLiNER extraction model. Confirm availability
in the [live model catalog](https://docs.fastino.ai/api-reference/inference/list-base-models)
for your account and override `--model` as needed. Requested and returned model
IDs are saved; do not label these results as GLiDE results. GLiDE's existing
`/v1/systemone` integration returns Noul/Choice/Score decisions, not arbitrary
entity spans, and is explicitly rejected by this extraction command.

The extraction adapter currently supports Fastino GLiNER only. No span extraction
API is assumed for Jev, Laya, or the other classification providers. Their span
predictions can be evaluated offline using the canonical format below once an
actual extractor produces them. This avoids comparing one model's extraction
against another model's binary detection scores.

The run saves predictions plus references as JSONL, and a JSON summary containing
the input SHA-256, partition identities, selected limit, extraction schema,
threshold, model identity, row counts, coverage, and quality metrics. Per-row
latency and errors are retained. Use identical inputs, model revisions, and
threshold selection procedures for comparisons. An alias returned by a provider
may not identify an immutable model revision; the benchmark does not invent one.

## Score spans offline

```bash
python -m scripts.pii_extraction evaluate \
  --input outputs/pii_extraction/fastino-nano-validation.jsonl \
  --summary outputs/pii_extraction/recomputed.metrics.json
```

Each successful JSONL row requires these fields:

```json
{"sample_id":"example-1","source_text":"Email alice@example.com","gold_entities":[{"start":6,"end":23,"label":"EMAIL"}],"predicted_entities":[{"start":6,"end":23,"label":"EMAIL"}],"error":""}
```

CSV inputs are also accepted; entity columns must contain JSON arrays or legacy
Python literal lists. `privacy_mask` can substitute for `gold_entities`. Offsets
are Python Unicode character indices, **not bytes or UTF-16 code units**; end is
exclusive. An optional `text` or `value` on a span must exactly equal the source
slice. Labels are stripped and uppercased, with no automatic taxonomy merging.
Missing labels/offsets, invalid ranges, and value-only outputs fail validation.
Exact duplicate entities are deduplicated; repeated mentions at different
positions remain separate. Overlapping spans are valid and compared exactly.

## Metrics and failures

- `entity_micro`: corpus TP/FP/FN and precision/recall/F1 for exact
  `(start, end, label)` matches. A partial boundary match or wrong type creates
  an FP and FN. Counts are summed before computing ratios.
- `boundary_micro`: exact `(start, end)` matches ignoring type, reported as a
  diagnostic rather than a substitute for typed entity F1.
- `per_type`: entity metrics plus gold support for every gold or predicted type.
  `entity_macro_f1` averages these type F1s, including predicted-only types.
- `exact_document_match`: fraction of rows whose complete entity sets match.
  Empty-gold negatives count false extractions as FP and affect exact match;
  empty/empty rows do not inflate entity micro F1. Undefined ratios use zero,
  including an all-empty corpus's entity F1; exact match can still be one.
- `request_success_rate`, `successful_rows`, and `failed_rows`: coverage is always
  visible. The main end-to-end entity counts treat failed requests as empty
  extractions (all gold entities become FN); failed rows never receive exact
  match credit. If some requests fail, `successful_only` separately reports
  conditional extraction quality. Failure-as-empty cannot estimate false
  positives from unavailable responses, so always compare quality with coverage.

Failed responses have `predicted_entities: null` and a nonempty `error`. A
successful response must have an explicit list, including `[]` for no detected
PII. Failed requests, including HTTP 401/402/403/429, are saved and cause exit
status 1 after summary generation. Error logs contain exception class and HTTP
status without response bodies or credentials. There are no automatic retries;
resolve access/billing/rate limits before a fresh run. If the API key is missing,
the run stops before sending requests. No live quality numbers are implied by
the offline tests.

## Existing classification and masking reports

`run_model_suite.py` keeps its existing binary behavior, prompts, thresholds, and
provider contracts, including Fastino/GLiDE. Its PII runs now print a detection-only
notice. To inspect historical binary PII results explicitly:

```bash
python -m scripts.evaluate --dataset pii_masking_nano_1k \
  --provider fastino.ai --prompt-version v1 --threshold 0.5 \
  --task classification
```

Those detection metrics are not extraction scores. The old PII masking evaluator
now requires actual redacted text (`predicted_text`, or legacy text-valued
`prediction`) and intact annotations. It rejects numeric predictions instead of
giving them false masking credit. Its historical leakage/placeholder/preservation
metrics, including heuristic `span_*` fields, are masking diagnostics only;
use the exact entity metrics above for extraction comparisons. New extraction
files live in their own directory and are not fed into the classification
comparison/plot pipeline.

## Tests

```bash
python -m pytest tests/test_pii_extraction.py -q
python -m pytest tests -q
```

Extraction tests run offline with mocked HTTP responses. They cover strict span
matching, per-type and micro aggregation, empty negatives, duplicate/repeated
mentions, Unicode, malformed/missing annotations, truncated and numeric outputs,
array preservation, partition metadata, HTTP restrictions, and CLI round trips.
