# Benchmark the original PI and PII presets

This standalone script downloads two pinned public Hugging Face datasets, calls
the original `prompt_injection_luna` and `input_pii` presets through Agent Control,
and computes precision, recall, F1, and confusion counts. It creates one dedicated
log stream and one observing control per metric in your existing project. It uses
the preset's current default version and pins that version throughout the run.

It does not create custom metrics, change prompts, or upload model weights.
The catalog identifies these as SLM presets; their underlying model deployment
is controlled by the cluster, not by this script.

## Setup

Use Python 3.11 or newer on Linux. From the repository root, enter
`demo-preset-benchmark` first. If using a standalone copy, enter that copy instead.

```bash
cd demo-preset-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your own API key and project ID. Never commit `.env`.
The example points to the demo cluster and the `galileo-demo` project:
[Open project](https://console.demo.sao.splunkcloud.com/demo-v2-poc/project/b33d2a32-5320-47ef-b351-cf4c4fb84a17).
The API URL is `https://api.demo.sao.splunkcloud.com`, without `/demo-v2-poc`.
The key needs access to that project, preset metrics, log streams, and Agent Control.
TLS uses the operating system's certificate store via `truststore`.
The public datasets do not require a Hugging Face token.

## Run

From this directory, first check both metrics on three dataset rows each:

```bash
python benchmark.py --limit 3 --output results-smoke
```

Run the full benchmark:

```bash
python benchmark.py --output results
```

Defaults: 1,000 PI training rows, all 990 PII validation rows, four concurrent
requests. The PI sample is proportional by label, sampled without replacement
with seed 42: 108 jailbreak positives and 892 negatives. It matches the earlier
1,000-row benchmark sample. Inputs are sent unchanged; no text truncation.
`--limit` takes the first N rows of that selection and is only a connectivity test.

Optional:

```bash
python benchmark.py --metric pii --output results-pii
python benchmark.py --pi-size 18258 --workers 4 --output results-all-pi
python benchmark.py --output results --retry-errors
```

### Nano PII download and evaluation

`--pii-dataset nano` downloads the pinned public
`ai4privacy/pii-masking-nano-1k` dataset without uploading a local CSV.
Use `--pii-split all` to evaluate all 990 documents (900 train + 90 validation),
or select `train` or `validation` separately. The original default remains
`--pii-dataset mini --pii-split validation` (990 validation documents).

```bash
# Connectivity test, then full nano evaluation:
python benchmark.py --metric pii --pii-dataset nano --pii-split all --limit 3 --output results-nano-smoke
python benchmark.py --metric pii --pii-dataset nano --pii-split all --workers 4 --output results-nano
# Resume and retry errors:
python benchmark.py --metric pii --pii-dataset nano --pii-split all --workers 4 --retry-errors --output results-nano
```

Both PII datasets use the unchanged `input_pii` scorer and four-category
document-level multilabel evaluation described below. These are not full
19-label or entity-extraction scores. Dataset revision, split, and file hashes
are recorded in the manifest. Use separate output folders for different datasets
and splits; the prediction key remains `pii`, with its identity in the manifest.

Re-running an identical configuration resumes its JSONL checkpoint. Use a new
output folder when changing the cluster, project, selected datasets, or sample.
Environment variables take precedence over the `.env` file. Failed calls are
reported separately, excluded from quality scores, and cause a nonzero exit code.
The script stops if the first pending call for a metric fails.

## Dataset columns and scoring

### Prepared datasets and dataset-level concurrency

The original public benchmark remains the default. To reuse the repository's
prepared samples, use `--source prepared`. By default this selects all six
non-PII CSVs from `../data/prepared`:

| Prepared dataset | Preset task |
|---|---|
| `toxigen` | `GALILEO_TOXICITY_PRESET` |
| `hatecheck` | `GALILEO_HATE_PRESET` |
| `aegis2` | `GALILEO_SAFETY_PRESET` |
| `detect_jailbreak`, `jailbreak_classification`, `jailbreakhub` | `prompt_injection_luna` |

The environment template selects the catalog-confirmed `input_toxicity_luna`
for toxicity; its positive-class score direction still needs verification.
Hate and safety remain placeholders because no matching presets were confirmed.
Custom LLM scorers are not yet supported by this script.

Set the three task-specific environment variables in `.env` to actual original
SLM preset names in your cluster. They must return a positive-class probability
in `[0, 1]` matching the CSV's unsafe/toxic/hateful label. No preset names or
score direction are inferred. Missing mappings fail before resource creation.
List visible preset names with `python benchmark.py --list-presets`; configure
only verified task mappings. CLI overrides are `--toxicity-preset`,
`--hate-preset`, and `--safety-preset`.

```bash
# Smoke test each selected dataset:
python benchmark.py --source prepared --dataset-workers 7 --limit 3 --output results-prepared-smoke
# All six complete prepared CSVs plus the unchanged original PII validation:
python benchmark.py --source prepared --dataset-workers 7 --output results-prepared
# Resume, including retry of failed calls:
python benchmark.py --source prepared --dataset-workers 7 --retry-errors --output results-prepared
```

PII still uses the original 990 validation documents and four-category scoring
below; the two local PII CSVs are deliberately excluded. This mode therefore
has seven dataset tasks, not all eight local CSVs. `--pi-size` does not sample
prepared datasets; all rows are used unless `--limit` is set. `--data-dir` can
override the prepared directory (needed for a standalone copy).

To run only the three jailbreak CSVs and original PII without extra mappings:

```bash
python benchmark.py --source prepared --datasets detect_jailbreak,jailbreak_classification,jailbreakhub --dataset-workers 4 --output results-jailbreak-pii
```

`--metric pi` omits original PII from prepared mode; `--metric pii` runs only the
original PII dataset. `--dataset-workers N` above one runs up to N dataset
threads, each making sequential calls, ignoring row-level `--workers`. The
original default (`--dataset-workers 1`) processes datasets sequentially with
four concurrent row requests. Resource setup remains sequential; inference
starts after setup.

With dataset concurrency, each dataset owns its checkpoint and summary in an
output subdirectory; the parent `summary.json` aggregates results. Failures are
reported and cause nonzero exit without discarding other datasets' checkpoints.
Use a new output folder when switching between sequential and dataset-concurrent
modes, or changing samples or preset mappings.

| Metric | Dataset / split | Input | Reference label |
|---|---|---|---|
| Prompt Injection | `GuardrailsAI/detect-jailbreak`, sampled train | `prompt` | `is_jailbreak` |
| PII | `ai4privacy/pii-masking-mini-10k`, validation | `source_text` | `privacy_mask[].label` |

PI uses a probability threshold of 0.5. Scores come from Agent Control's
`result.metadata.score`, not the final control decision or confidence field.
PI also reports accuracy, AUROC, and average precision when both labels occur.
The dataset labels jailbreaks, while the preset can cover a broader set of
prompt injection or malicious inputs. These scores measure agreement with the
dataset labels; some disagreements can reflect that scope difference.

PII is multilabel classification at the document level, not entity extraction.
The original preset exposes grouped product categories. Its definitions do not
fully match this dataset's entity labels, so the primary comparison uses these
four closest categories:

| Dataset label | Original preset category | Scope caveat |
|---|---|---|
| `CREDITCARDNUMBER` | `credit_card_info` | Preset also includes CVV, expiry, and last four digits. |
| `EMAIL` | `email` | Preset can distinguish personal contacts from shared/public contacts. |
| `TELEPHONENUM` | `phone_number` | Same personal versus shared/public contact caveat. |
| `SOCIALNUM` | `ssn` | Dataset spans international social identifiers; definitions can differ. |

Each document is positive for a category if any matching annotation exists.
Repeated entities do not increase the document count. Each of the four categories
is also evaluated as absent on documents without that annotation. The script
reports per-category scores, micro precision/recall/F1, macro F1, exact match,
and the same statistics by `language` (30 languages, 33 validation documents each).
Macro F1 averages all four primary categories, including categories with no
positive reference examples in a particular language (zero division yields zero).

Names, dates, and address fragments have broader dataset annotations than the
original preset's full-name, birth-date, and complete-address definitions. They
are excluded from primary F1. Driver's licenses, tax IDs, and other document IDs
are not exposed separately by this preset. Age, gender, sex, time, and titles
have no original preset category. None is silently treated as a failed class.
All returned preset labels are retained in predictions for later analysis.

## Integration and offline tests

Dependencies and results stay local to this directory; the existing JEV provider
runner and root dependency file are unchanged. Predictions are JSONL, not the
CSV schema consumed by `scripts/evaluate.py`. The script calculates its own
metrics in `summary.json`. Cross-model comparisons require identical sample
selection and label definitions; PII multilabel F1 must not be compared with
binary PII F1.

To run offline regression tests from this directory, after installing the
benchmark dependencies:

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

These tests do not call Galileo or download datasets.
`TEST_RESULTS.md` describes the supplied historical verification, not a new
live verification of this integration.

## Outputs

- `manifest.json`: cluster/project, sampled source indices, dataset revisions and
  file hashes, pinned scorer/version IDs, stream IDs, and control IDs.
- `pi-predictions.jsonl` and `pii-predictions.jsonl`: input, gold labels, model
  scores, predictions, and the matching scorer execution, including errors.
- `summary.json`: overall metrics, confusion counts, call coverage/errors, mean
  latency of successful API calls in seconds and milliseconds, the number of
  latency samples, and PII language breakdowns. Latency is the client-observed
  wall time for one evaluator call (including token refresh when needed, network,
  Agent Control, scoring, and response validation), not isolated model inference
  time. Metric values are fractions between 0 and 1.

Per-row `latency_seconds` is saved in prediction JSONL for successful and failed
evaluator attempts. The summary average uses successful calls only. Completed
checkpoints already contain per-row latency, so rerunning
the same command with the same output directory recalculates the summary without
repeating successful API calls. Failed rows can be timed when retried.

The console prints an easy-to-read percentage table. Runtime bearer tokens and
API keys are never saved in results. Keep `.env` local; share only the script,
this README, `requirements.txt`, and `.env.example`.
