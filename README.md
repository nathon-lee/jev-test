# JEV Benchmark Suite

This project benchmarks multiple providers against safety classification datasets, including prompt-version comparison (v1/v2) and threshold tuning.

## Overview

Supported providers:
- typesafe.ai
- fastino.ai
- laya-ai.com
- openjev
- galileo.ai
- microsoft-decision-1

Supported datasets:
- toxigen
- hatecheck
- aegis2
- detect-jailbreak
- jailbreak-classification
- jailbreakhub
- pii_masking_nano_1k
- pii_masking_mini_10k

## Create a virtual environment

On Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

Set the project root on `PYTHONPATH` so the package imports work when you run scripts directly:

```bash
export PYTHONPATH=$PWD
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD"
```

## Quick start

1. Copy environment template

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

2. Fill in your own keys / URLs in `.env`

Example values:

```bash
TYPESAFE_API_KEY=your_key_here
FASTINO_API_KEY=your_key_here
LAYA_API_KEY=your_key_here

OPENJEV_API_KEY=your_key_here
OPENJEV_BASE_URL=http://localhost:8000/v1

GALILEO_API_KEY=your_key_here
GALILEO_BASE_URL=https://your-internal-url/v1

MICROSOFT_DECISION_1_API_KEY=your_key_here
MICROSOFT_DECISION_1_BASE_URL=https://your-internal-url/v1
```

3. Prepare datasets

```bash
python scripts/prepare_datasets.py
```

The first three datasets (`toxigen`, `hatecheck`, `aegis2`) are already included in the project or fetched via the standard Hugging Face dataset loaders in the repo. The remaining three datasets (`detect-jailbreak`, `jailbreak-classification`, `jailbreakhub`) are the ones that are pulled from Hugging Face. The PII datasets are also prepared here:

- `pii_masking_nano_1k`
- `pii_masking_mini_10k`

These are loaded from `ai4privacy/pii-masking-nano-1k` and `ai4privacy/pii-masking-mini-10k`.

4. Run benchmark

Bare benchmark runner (single dataset, direct model inference):

```bash
python scripts/run_benchmark.py --dataset toxigen --prompt-version v1
python scripts/run_benchmark.py --dataset pii_masking_nano_1k --prompt-version v1
```

Compatibility runner (single case + optional evaluation):

```bash
python scripts/run_case.py --dataset toxigen --prompt-version v1
python scripts/run_case.py --dataset pii_masking_nano_1k --prompt-version v1
python scripts/run_case.py --dataset toxigen --prompt-version v1 --evaluate
python scripts/run_case.py --dataset pii_masking_nano_1k --prompt-version v1 --evaluate
```

Single dataset, v2:

```bash
python scripts/run_case.py --dataset toxigen --prompt-version v2
```

All datasets, v1 and v2:

```bash
python scripts/run_case.py --dataset all --prompt-version v1
python scripts/run_case.py --dataset all --prompt-version v2
```

All providers, all datasets, v1 + v2, thresholds 0.5/0.6/0.7/0.8:

```bash
python scripts/run_model_suite.py --providers all --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
```

Only one provider:

```bash
python scripts/run_model_suite.py --providers typesafe.ai --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
```

Only one dataset:

```bash
python scripts/run_model_suite.py --providers all --datasets toxigen --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
```

PII-specific multi-provider sweep:

```bash
python scripts/run_model_suite.py --providers typesafe.ai --datasets pii_masking_nano_1k --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
python scripts/run_model_suite.py --providers typesafe.ai --datasets pii_masking_mini_10k --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
```

## PII dataset quick start

Prepare the PII datasets first:

```bash
python scripts/prepare_datasets.py
```

Then generate predictions and evaluate them:

```bash
python scripts/run_benchmark.py --dataset pii_masking_nano_1k --prompt-version v1
python scripts/run_benchmark.py --dataset pii_masking_mini_10k --prompt-version v2

python scripts/evaluate.py --dataset pii_masking_nano_1k --provider typesafe.ai --debug-path
python scripts/evaluate.py --dataset pii_masking_mini_10k --provider typesafe.ai
```

The PII-specific evaluation path is triggered automatically whenever the dataset name contains `pii`.

## Compatibility runner

The project also includes a simplified runner that handles the repeated setup steps:

```bash
python scripts/run_case.py --dataset toxigen --prompt-version v1 --evaluate
python scripts/run_case.py --dataset pii_masking_nano_1k --prompt-version v1 --evaluate
```

This automatically does the equivalent of:
- export PYTHONPATH
- run the benchmark script
- copy the versioned prediction file to the compatibility file expected by the evaluator
- run the evaluation step

On PowerShell:

```powershell
.\scripts\run_case.ps1 -Dataset toxigen -PromptVersion v1 -Evaluate
.\scripts\run_case.ps1 -Dataset pii_masking_nano_1k -PromptVersion v1 -Evaluate
```

## Run evaluation manually

```bash
python scripts/evaluate.py --dataset toxigen
```

For the multi-provider summary:

```bash
python scripts/evaluate_model_suite.py
```

Optional single-provider summary:

```bash
python scripts/evaluate_model_suite.py --provider typesafe.ai
```

## Important notes

- If you run scripts directly with `python scripts/...`, set `PYTHONPATH` in the current shell:
  - Linux / macOS: `export PYTHONPATH=$PWD`
  - PowerShell: `$env:PYTHONPATH = "$PWD"`
- `scripts/run_case.py` also sets this automatically for the compatibility runner.
- The compatibility runner copies results like:
  - `outputs/toxigen_v1_predictions.csv` -> `outputs/toxigen_predictions.csv`
- This is useful because the older evaluator expects the shorter filename.
- For threshold tuning, use `--thresholds 0.5,0.6,0.7,0.8` in `scripts/run_model_suite.py`
- For version tuning, use `--prompt-versions v1,v2`

## Directory layout

```text
.
├── .env.example
├── .env
├── benchmark/
│   ├── datasets.py
│   ├── provider_suite.py
│   └── jev_client.py
├── scripts/
│   ├── prepare_datasets.py
│   ├── run_benchmark.py
│   ├── run_case.py
│   ├── run_model_suite.py
│   ├── evaluate.py
│   ├── evaluate_model_suite.py
│   └── run_case.ps1
├── data/
│   └── prepared/
├── outputs/
└── README.md
```

## Typical test flows

### 1. Single provider quick check

```bash
python scripts/run_case.py --dataset toxigen --prompt-version v1 --evaluate
```

### 2. Compare v1 vs v2 on one dataset

```bash
python scripts/run_case.py --dataset hatecheck --prompt-version v1
python scripts/run_case.py --dataset hatecheck --prompt-version v2
```

### 3. Full threshold sweep for one provider

```bash
python scripts/run_model_suite.py --providers typesafe.ai --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
```

### 4. Full benchmark sweep for all providers

```bash
python scripts/run_model_suite.py --providers all --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
```

## Troubleshooting

If a provider fails to run:
- confirm the matching API key variable exists in `.env`
- confirm the provider base URL points to the correct deployment
- ensure the dataset has been prepared with `python scripts/prepare_datasets.py`
- verify the provider name matches the registry exactly, for example `typesafe.ai` or `openjev`

If evaluating the legacy output file fails:
- run the compatibility script or manually copy the versioned file to `outputs/<dataset>_predictions.csv`

## License

This project is intended for internal benchmark and evaluation workflows.
