# Jev Test

This repository evaluates a TypeSafe-based Jev model on several safety classification datasets. The project is designed to:

- prepare benchmark datasets
- run model inference over each sample
- save per-sample predictions and latency
- compute standard classification metrics
- compare results across datasets

The main workflow is:

1. prepare data under `data/prepared/`
2. run `scripts/run_benchmark.py` to call the remote TypeSafe model
3. copy generated prediction files to the name expected by `scripts/evaluate.py`
4. run `scripts/evaluate.py` to print metrics

---

## Project structure

- `benchmark/` : dataset preparation logic and Jev client wrapper
- `data/prepared/` : standardized benchmark CSV files used for evaluation
- `outputs/` : prediction files generated during benchmarking
- `scripts/` : dataset preparation, benchmark execution, and evaluation scripts
- `requirements.txt` : Python dependencies

---

## Environment setup

Create a virtual environment and install dependencies:

```bash
cd /workspace/jev-test
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install typesafe-sdk
```

On Windows PowerShell:

```powershell
cd d:\open_source\jev-xx\jev-test
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install typesafe-sdk
```

---

## Required API key

This project does not run a local model. It calls the remote TypeSafe API through `TypeSafeClient`.

You must set the environment variable before running any benchmark:

```bash
export TYPESAFE_API_KEY="your_key_here"
```

PowerShell:

```powershell
$env:TYPESAFE_API_KEY="your_key_here"
```

If this variable is missing, the SDK will raise a configuration error.

---

## Fix Python import path

Because the project uses imports like `from benchmark.jev_client import run_jev`, always run commands from the repo root and add the repo root to `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD
```

PowerShell:

```powershell
$env:PYTHONPATH="d:\open_source\jev-xx\jev-test"
```

---

## Dataset preparation

The scripts in `benchmark/datasets.py` create standardized benchmark CSVs in `data/prepared/`.

Current datasets:

- `toxigen.csv`
- `hatecheck.csv`
- `aegis2.csv`

Run all dataset builders:

```bash
cd /workspace/jev-test
export PYTHONPATH=$PWD
python scripts/prepare_datasets.py
```

This generates:

```text
data/prepared/toxigen.csv
data/prepared/hatecheck.csv
data/prepared/aegis2.csv
```

---

## Benchmark command

The benchmark script iterates row by row and sends each sample to the Jev model.

Example: run ToxiGen with a subset of rows:

```bash
cd /workspace/jev-test
export PYTHONPATH=$PWD
export TYPESAFE_API_KEY="your_key_here"
python scripts/run_benchmark.py --dataset toxigen --limit 6789 --prompt-version v1
```

Example: run HateCheck with 3728 rows:

```bash
cd /workspace/jev-test
export PYTHONPATH=$PWD
export TYPESAFE_API_KEY="your_key_here"
python scripts/run_benchmark.py --dataset hatecheck --limit 3728 --prompt-version v1
```

Example: run AEGIS 2.0 with 1964 rows:

```bash
cd /workspace/jev-test
export PYTHONPATH=$PWD
export TYPESAFE_API_KEY="your_key_here"
python scripts/run_benchmark.py --dataset aegis2 --limit 1964 --prompt-version v1
```

Example: run all datasets:

```bash
cd /workspace/jev-test
export PYTHONPATH=$PWD
export TYPESAFE_API_KEY="your_key_here"
python scripts/run_benchmark.py --dataset all --prompt-version v1
```

The benchmark writes files such as:

```text
outputs/toxigen_v1_predictions.csv
outputs/hatecheck_v1_predictions.csv
outputs/aegis2_v1_predictions.csv
```

---

## Evaluation command

The evaluation script reads the generated prediction CSV and prints metrics.

Important: the script expects a file named like `outputs/{dataset}_predictions.csv`.

Because the benchmark writes `..._v1_predictions.csv`, copy it to the expected name before evaluation:

```bash
cp outputs/toxigen_v1_predictions.csv outputs/toxigen_predictions.csv
python scripts/evaluate.py --dataset toxigen
```

Same pattern for the other datasets:

```bash
cp outputs/hatecheck_v1_predictions.csv outputs/hatecheck_predictions.csv
python scripts/evaluate.py --dataset hatecheck

cp outputs/aegis2_v1_predictions.csv outputs/aegis2_predictions.csv
python scripts/evaluate.py --dataset aegis2
```

---

## Example full run

```bash
cd /workspace/jev-test
export PYTHONPATH=$PWD
export TYPESAFE_API_KEY="your_key_here"

python scripts/run_benchmark.py --dataset toxigen --limit 6789 --prompt-version v1
cp outputs/toxigen_v1_predictions.csv outputs/toxigen_predictions.csv
python scripts/evaluate.py --dataset toxigen

python scripts/run_benchmark.py --dataset hatecheck --limit 3728 --prompt-version v1
cp outputs/hatecheck_v1_predictions.csv outputs/hatecheck_predictions.csv
python scripts/evaluate.py --dataset hatecheck

python scripts/run_benchmark.py --dataset aegis2 --limit 1964 --prompt-version v1
cp outputs/aegis2_v1_predictions.csv outputs/aegis2_predictions.csv
python scripts/evaluate.py --dataset aegis2
```

---

## Key terminology

### TypeSafe
A remote API service that exposes structured LLM outputs. This project uses it as the backend for model inference.

### Jev
The model being evaluated in this repository. The benchmark calls `client.system_one(...)` and interprets the returned score as a probability-like value for the positive class.

### prompt_version
The prompt wording used to ask the model a safety question. The project supports:

- `v1`
- `v2`

The logic is defined in `benchmark/jev_client.py`.

### score
A numeric confidence value produced by the model. In this project, it is converted to a binary prediction:

```python
prediction = int(score >= 0.5)
```

### latency_ms
The elapsed time for one model call, measured in milliseconds.

### confusion matrix
A 2x2 table for binary classification. The project reports:

- TN = True Negative
- FP = False Positive
- FN = False Negative
- TP = True Positive

### Accuracy
The share of all examples predicted correctly.

### Precision
Of the examples predicted as positive, how many were truly positive.

### Recall
Of the truly positive examples, how many were correctly found.

### F1
The harmonic mean of precision and recall.

### AUROC
Area Under the Receiver Operating Characteristic curve. It measures ranking quality across thresholds.

---

## Current datasets used in this project

### ToxiGen
A toxicity / hate classification benchmark built from the ToxiGen dataset.

The project standardizes it to a binary label:

- `0` = benign / non-toxic
- `1` = toxic / hateful

Current benchmark size in this workspace is 6789 samples after filtering.

### HateCheck
A benchmark focused on hateful content patterns and edge cases.

Binary label mapping:

- `0` = non-hateful
- `1` = hateful

Current benchmark size in this workspace is 3728 samples.

### AEGIS 2.0
A safety benchmark for unsafe or harmful content.

Binary label mapping:

- `0` = safe
- `1` = unsafe

Current benchmark size in this workspace is 1964 samples.

---

## Output files

After benchmarking, files are written to `outputs/`.

Example files:

```text
outputs/toxigen_v1_predictions.csv
outputs/hatecheck_v1_predictions.csv
outputs/aegis2_v1_predictions.csv
```

Each row contains the model output for a single sample and includes fields such as:

- dataset
- sample_id
- text
- label
- prompt_version
- score
- prediction
- latency_ms
- model
- error

---

## Typical troubleshooting

### ModuleNotFoundError: No module named 'benchmark'
Run from the repo root and set `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD
```

### No API key was provided
Set the API key:

```bash
export TYPESAFE_API_KEY="your_key_here"
```

### evaluate.py cannot find prediction files
The benchmark writes `*_v1_predictions.csv`, while `evaluate.py` expects `*_predictions.csv`.

Copy the file before evaluation:

```bash
cp outputs/toxigen_v1_predictions.csv outputs/toxigen_predictions.csv
```

---

## Quick start

```bash
cd /workspace/jev-test
export PYTHONPATH=$PWD
export TYPESAFE_API_KEY="your_key_here"

python scripts/prepare_datasets.py
python scripts/run_benchmark.py --dataset toxigen --limit 10 --prompt-version v1
cp outputs/toxigen_v1_predictions.csv outputs/toxigen_predictions.csv
python scripts/evaluate.py --dataset toxigen
```

This is the minimal path to verify the repo is working end-to-end.

---

## Notes

- This project is an evaluation benchmark, not a local inference stack.
- The actual model computation is handled by the remote TypeSafe service.
- GPU is not required for the current repository design.
- If you want a local/offline deployment, you must replace the remote TypeSafe client logic with a local model server or compatible API wrapper.
