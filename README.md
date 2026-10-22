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

Parallel run (for example with 6 workers):

```bash
python scripts/run_model_suite.py --providers all --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8 --max-workers 6
```

Limit rows per dataset (for quick validation):

```bash
python scripts/run_model_suite.py --providers all --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8 --limit 100
```

`--max-workers` controls how many provider/dataset/prompt/threshold combinations run at the same time.
- Default: `1` (fully serial, same behavior as before)
- Typical value: `4` to `8` for network-bound provider calls
- If provider-side rate limits are strict, lower this number

`--limit` controls the maximum number of rows processed for each dataset in each benchmark task.
- Example: `--limit 100` means each dataset run uses at most 100 samples.

Only one provider:

```bash
python scripts/run_model_suite.py --providers typesafe.ai --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8
```

Run in the background with timestamped logs (recommended for remote / SSH sessions):

```bash
cd /home/nathon/jev-test && mkdir -p logs && nohup bash -lc 'stdbuf -o0 -e0 python -u scripts/run_model_suite.py --providers typesafe.ai,fastino.ai,laya-ai.com --datasets all --prompt-versions v1,v2 --thresholds 0.5,0.6,0.7,0.8 2>&1 | awk "{ print strftime(\"[%Y-%m-%d %H:%M:%S]\"), \$0; fflush(); }" >> /home/nathon/jev-test/logs/run_model_suite.log' >/dev/null 2>&1 &
```

Alternative background command (timestamp in log filename + startup self-check):

```bash
cd /workspace/jev-test
mkdir -p logs
LOG="logs/run_model_suite_$(date +%F_%H%M%S).log"

nohup stdbuf -o0 -e0 python -u scripts/run_model_suite.py \
  --providers typesafe.ai \
  --datasets all \
  --prompt-versions v1,v2 \
  --thresholds 0.5,0.6,0.7,0.8 \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID LOG=$LOG"
sleep 1
ps -p "$PID" -o pid,cmd
tail -n 30 "$LOG"
```

Check the live log:

```bash
tail -f /home/nathon/jev-test/logs/run_model_suite.log
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

### 可视化模型对比报告

在项目根目录运行（无需 API Key，不会重新请求模型）。首次使用先安装依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 已注明对比生成依赖的用途：`pandas`、`numpy`、`scikit-learn`
用于数据处理与评分，`matplotlib` 用于对比图表；`pytest` 仅用于相关测试，
不是生成报告的运行时依赖。这些包已统一列入依赖文件，无需再分别安装。

**推荐：一条命令生成对比报告和全部图表。**

```bash
python -m scripts.compare_models --paper-figures
```

报告入口为 `outputs/comparison/comparison.html`，图表入口为
`outputs/comparison/figures/index.html`。

如果只需要生成对比报告，不绘图：

```bash
python -m scripts.compare_models
```

打开 `outputs/comparison/comparison.html` 查看中文结论、模型名称、固定配置排名、
分数据集指标、提示词/阈值探索、覆盖率和日志耗时，以及可搜索的全部配置明细。
同时生成 `comparison.md`、`comparison.json`、`runs.csv`、`paired.csv` 和 `pairwise.csv`。
Laya 后续测试完成后重新运行同一命令即可更新快照。

报告复用 `evaluate_model_suite.py` 的分类指标实现（提取于
`benchmark/evaluation.py`），使用本地 `data/prepared` 校验样本，忽略隐藏的 notebook
检查点，只扫描供应商目录下的 CSV 和日志。支持自定义路径或仅对比两家供应商：

```bash
python -m scripts.compare_models --input-dir outputs --data-dir data/prepared --output-dir outputs/comparison
python -m scripts.compare_models --providers fastino.ai,typesafe.ai --output-dir outputs/comparison_two
```

公平基线固定为 `v1 / 0.5`，只比较共同完整落盘的数据集及共同成功样本；缺测不计零分。
报告还包含两两共同基线，让已完成模型利用更多共同数据集比较；不同配对的分数不混排。
PII 目前只有全正样本分类输出，不能用于评价实际脱敏质量。
日志只能给出整轮平均每样本耗时，不能据此计算单请求 P95。
报告按用户补充的模型身份区分请求、解析、服务和路由名称：

| 供应商 | Request model | Resolved model | Serving model | Routed model |
| --- | --- | --- | --- | --- |
| TypeSafe | `jev-latest` | 需以接口返回为准，当前未确认 | — | — |
| Fastino | `fastino/GLiDE` | `glide_v2` | — | — |
| Laya | `typed-decisions` | — | `laya-rl-agent` | `typed-decisions` |

这些身份信息不是从历史 CSV 或日志反查得到；`jev-latest` 不代表固定的后端版本。

### 对比图表

已有对比报告时，无需重新读取预测文件或调用模型：

```bash
python -m scripts.plot_model_comparison
```

打开 `outputs/comparison/figures/index.html` 查看以下图表及中文图注：

- 共同样本的分数据集 F1 / AUROC 分组柱状图。
- 误报—漏报、AUROC—日志平均耗时散点图。
- v1 / v2 提示词与阈值敏感性曲线。
- 全数据集测试覆盖率热力图，区分缺测和全部请求失败。

每张图同时导出 300 DPI PNG、矢量 SVG 和 PDF；`paper_figures.pdf` 汇总全部图，
`table01_baseline.tex` 为 LaTeX 三线表（需 `booktabs`）。
图内采用英文标签、衬线字体、固定色盲友好配色，配合填充纹理和线型区分模型与提示词。
模型的 request / resolved / serving / routed 身份保留在图例中。
不以重复运行伪造置信区间，也不把测试集选参结果解释为独立验证后的提升。

也可一条命令更新报告并生成图表：

```bash
python -m scripts.compare_models --paper-figures
```

仅从已有报告重绘为 600 DPI PNG（同时保留矢量 SVG / PDF）：

```bash
python -m scripts.plot_model_comparison --report outputs/comparison/comparison.json --output-dir outputs/comparison/figures --dpi 600
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
