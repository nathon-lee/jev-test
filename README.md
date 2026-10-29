> **Latest held-out results:** [HTML report](outputs/heldout_2026-10-28/comparison.html) · [v1 summary PNG](outputs/heldout_2026-10-28/figures/fig13_three_dataset_averages.png) · [all charts](outputs/heldout_2026-10-28/figures/index.html) · [scope and reproduction](HELDOUT_RERUN.md). Four completed providers on identical official evaluation rows; this is separate from the historical-row comparison below.

> **Historical-row results (PR #20):** [corrected comparison](outputs/comparison/comparison.md) · [updated charts](outputs/comparison/figures/index.html). Older comparison instructions and figures below are archived historical material; use `python -m scripts.publish_corrected_comparison` for the current presentation.

# JEV Benchmark Suite

This project benchmarks multiple providers against safety classification datasets, including prompt-version comparison (v1/v2) and threshold tuning.

Dataset preparation now follows an explicit [evaluation partition policy](docs/evaluation-partitions.md):
official test first, then validation; train-only sources require reference-only
opt-in. Every new prepared file records its source revision, split, role, and
checksum. Regenerate every provider's predictions before comparing scores on the
new partitions; historical outputs use the older mixed split policy.

PII **extraction** has a separate [span/entity benchmark](docs/pii-extraction.md):
`python -m scripts.pii_extraction {prepare,run,evaluate}`. It preserves gold
annotations and reports exact typed entity precision/recall/F1, per-type results,
boundary metrics, and request coverage. The existing model suite remains a
classification runner; its binary PII outputs do not measure extraction quality.
Regenerate the checked-in PII data before extraction: older prepared CSVs lost
their annotations. Fastino extraction uses GLiNER with an explicit model identity,
while existing Fastino classification continues to use GLiDE.

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

Fastino and Laya classification scores are read strictly from `answers.score.noul`,
not `confidence`. The value must be a finite JSON number in `[0, 1]`; missing or
invalid values are recorded as error rows with no score or prediction, without
falling back to confidence or other numeric fields.

Historical Fastino/Laya CSVs produced by the previous generic score parser may
contain confidence values instead of Noul scores. They do not retain both response
fields, so they cannot be reliably repaired in place. Preserve old outputs separately,
rerun affected benchmarks, and regenerate comparison reports and figures before
using their metrics.

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

加入用户提供的 OpenJev 内部汇总表：

```bash
python -m scripts.compare_models --external-summary data/external/openjev_summary.csv --paper-figures
```

该表保留最初提供的 16 行 v1/v2、阈值 0.5 的指标及精度，PII 的 AUROC 留空。
原 OpenJev 在图中标为 **OpenJev 4B**。同目录的 `internal_model_summary.csv` 会自动加入
OpenJev 32B 和 Microsoft Decision-1；也可用 `--internal-summary <path>` 指定该附加汇总表。
新增两家仅有 v1 / 0.5，加入逐数据集参考图、参考均值与独立三数据集页面，不加入公平基线或稳定性分析，
也不改变原四模型的 v2 范围。32B 原始文件标签为 `apus-openjev-35b-a3b`，显示名称按用户确认保留 32B。
32B 的 ToxiGen 五项质量指标均已补全，正常展示并参与均值；缺失指标仍按 N/A 处理，
不缩减平均范围或反推数据。
新增模型保留每个数据集一次完整运行的 `elapsed/processed`，与原四模型四轮均值协议不同；
图注注明差异，并展示有效/错误行数，避免高错误率下的成功样本分数被误当作全量表现。

仅加入单独的四模型参考图与共同非 PII 数据集等权均值，不加入原有共同成功样本基线或稳定性分析。
OpenJev 没有逐样本预测，不能核验共同输入/成功样本；总 Rows 不当作有效样本数，
也不从有错字的 OCR 文本推断错误数或混淆矩阵。模型版本保持未确认。
参考图优先对每个数据集/提示词的四轮完整日志 elapsed/processed 等权平均，包含失败等待。
v1 本地模型缺少四轮均值时，仅可使用与全量 CSV 行数一致的完整 `v1 / 0.5` 单轮日志，
保留日志来源，并用 `*` 标明单轮计时，不将其伪装成四轮均值；v2 仍要求完整四轮计时。
新增内部模型的单轮计时也用 `*` 标明；无可核验完整计时仍标 N/A。
并发日志若能核验四阈值各一次启动、四次全量完成及保存且无后续重跑，可直接计算四轮均值；
此时不需要推断完成记录与阈值的对应关系，也不回填各阈值的耗时。
`index.html` 展示三组新增参考图，稳定性三组仍仅保留超链接；汇总 PDF 保留全部图。
该次完整报告生成后，`python -m scripts.plot_model_comparison` 可从保存的 JSON 重绘全部图表。
另有 `outputs/comparison_historical_confidence/figures/three_dataset_comparison.html` 独立页面，仅对 ToxiGen、HateCheck、
AEGIS2 三个数据集等权平均，原四模型仅取 v1、阈值 0.5 的六项参考均值，并提供 PNG / SVG / PDF / CSV。
该页本地模型耗时也只取对应 v1 / 0.5 运行的 elapsed/processed，不对四轮运行平均；
OpenJev 取所提供汇总表中对应配置的耗时。其他参考图仍保留原有口径。
该页另加入 Dynamo CPU、Dynamo GPU (A100)、Dynamo GPU (L4)、Mistral、Warden 的用户提供均值；
当时没有内部数据，Fastino、Laya、TypeSafe、OpenJev 4B、OpenJev 32B、Microsoft Decision-1
仅使用共享的三个数据集计算均值。其他模型可能额外测试过内部数据，是否计入所提供均值尚未核实，
不能默认其数据集范围完全一致；保留所提供均值，不重新计算。新增配置各展示一次，
图中不再区分 v1/v2，未提供的 F1 / AUROC 标 N/A，不从平均 Precision/Recall 推算 F1。
新增配置的提示词、阈值与计时协议未知，仅作参考；耗时轴在 1 ms 以上使用对数刻度。
该页由 `index.html` 超链接进入，不替换原有参考均值图；指定数据集缺失时标 N/A，不缩减范围。

报告入口为 `outputs/comparison_historical_confidence/comparison.html`，图表入口为
`outputs/comparison_historical_confidence/figures/index.html`。

如果只需要生成对比报告，不绘图：

```bash
python -m scripts.compare_models
```

打开 `outputs/comparison_historical_confidence/comparison.html` 查看中文结论、模型名称、固定配置排名、
分数据集指标、提示词/阈值探索、覆盖率和日志耗时，以及可搜索的全部配置明细。
同时生成 `comparison.md`、`comparison.json`、`runs.csv`、`paired.csv` 和 `pairwise.csv`。
Laya 后续测试完成后重新运行同一命令即可更新快照。

如果本地只补充了部分 CSV、历史逐样本文件已不在目录中，使用旧报告增量更新，避免丢失历史结果：

```bash
python -m scripts.compare_models --previous-report outputs/comparison_historical_confidence/comparison.json --paper-figures
```

新 CSV 替换同配置旧结果；未提供的配置保留历史汇总，未测试配置仍然缺失。
新 CSV 没有匹配日志时不沿用旧耗时。共同样本指标仅在配置均未变化、逐样本数据可重新对齐，
或所有模型均成功覆盖全量参考样本时保留/补充；否则排除并提示，不能用各自样本指标冒充公平对比。
重复调用仍要求四轮；缺少重新计算所需的历史逐样本数据时标 `unavailable_raw`。
旧报告附带的 OpenJev 参考结果会重新合并，源汇总 CSV 需仍然存在。
恢复历史包时保留已补充的新 CSV。`outputs/log_input_hashes.json` 记录发生替换的历史 CSV
与原日志的 SHA-256 绑定；指纹不匹配时，旧日志不用于该配置或其四轮均值，避免行数相同却误配耗时。

报告复用 `evaluate_model_suite.py` 的分类指标实现（提取于
`benchmark/evaluation.py`），使用本地 `data/prepared` 校验样本，忽略隐藏的 notebook
检查点，只扫描供应商目录下的 CSV 和日志。支持自定义路径或仅对比两家供应商：

```bash
python -m scripts.compare_models --input-dir outputs --data-dir data/prepared --output-dir outputs/comparison_historical_confidence
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

打开 `outputs/comparison_historical_confidence/figures/index.html` 查看以下图表；页面说明、图注、链接及数据警告均使用英文：
首页概览保留 PDF 与完整报告入口、UTC 数据快照及简短可比性提示；
CSV、LaTeX 与独立页入口收纳在默认折叠的 Data & Downloads，
模型来源和协议限制收纳在默认折叠的 Methodology & Limitations。

误报、漏报与耗时权衡图的左侧使用全部共同基线数据集；右侧只取所有模型均有完整
`v1 / 0.5` 单轮计时的共同子集，并在该子集上同时重算宏 AUROC 和样本加权耗时。
两侧范围可能不同，具体数据集在图中注明；不把四轮平均计时当作缺失单轮计时。
无可用共同计时子集时明确显示 N/A，不留无说明的空图。

提示词与阈值敏感性图优先使用同配置共同成功样本（实心点）；无可用共同样本指标时，
以空心点展示该模型自身成功样本，避免一家缺测导致其他模型已有结果被隐藏。
缺测或全部请求失败仍留空，不补零、不插值；不同点样本集合可能不同，该探索图不用于公平排名。

- 共同样本的分数据集 F1 / AUROC 分组柱状图。
- 误报—漏报、AUROC—日志平均耗时散点图。
- v1 / v2 提示词与阈值敏感性曲线。
- 全数据集测试覆盖率热力图，区分缺测和全部请求失败。
- 逐数据集 Accuracy / Precision / Recall / F1 / AUROC / AVG latency (ms) 分组柱状图。
- 模型 Average Accuracy / Avg Precision / Avg Recall / Avg AUROC / Avg latency 分组柱状图。
- v1 / v2 各自的逐数据集重复调用一致性，以及共同样本的模型宏平均一致性图。

分类指标固定使用 `v1 / 0.5`。逐数据集图展示各模型自身成功样本（含 PII 检出，不代表脱敏质量），
模型平均图则使用共同非 PII 基线的共同成功样本，对数据集等权平均。
模型平均图的 Avg latency 仅在所有模型均有完整 `v1 / 0.5` 单轮计时的共同基线子集上，
对各数据集的日志平均耗时等权平均，各模型使用同一范围；无共同完整计时子集时标 N/A。
质量与耗时的数据集范围可能不同，图内注明实际数量与耗时数据集，不用四轮均值替代缺失单轮计时。
这与散点图的样本加权耗时不同；
均包含网络及失败等待，不是纯推理时延。缺测、无有效样本或无定义的 AUROC 标为 N/A，不补零。
含 OpenJev 的多模型参考均值页固定为 `v1 / 0.5`：主图对六个非 PII 数据集等权平均，
两个 PII 数据集单独汇总，八数据集总均值放在补充区；不跳过 PII 的未定义 AUROC 来拼出全八集 AUROC。
三家 API 耗时使用 `v1 / 0.5` 单轮日志，不沿用四轮均值；4B 保留提供的汇总耗时，
32B、Decision-1 保留提供的单轮计时，协议不同，不解释为统一速度排名。
各范围的耗时只在所有模型均有计时的共同子集上等权平均，注明实际范围，无共同计时子集时标 N/A。
`figures/v1_reference_scope_averages.csv` 同时导出非 PII、PII、全八集的质量指标、
完整范围耗时（缺项则为空）及共同子集耗时，不修改历史 v2 结果。
该均值页的六项指标全部展开，按 Accuracy / F1、Precision / Recall、AUROC / Latency
两列三行排列，窄屏自动单列；模型颜色分类图例置于开头，数值标签保持清晰；
PII 汇总、八数据集补充区与末尾折叠的统计口径说明保持分开。
逐数据集多模型参考对比的 v1 图采用单列放大布局和顶部共用图例；索引默认展示
Accuracy、F1、Latency，Precision、Recall、AUROC 可展开查看。Latency 使用对数纵轴，
柱上仍标注实际 ms/样本；完整六指标图与单指标图均提供 PNG / SVG / PDF 下载。
三数据集参考均值在索引及独立页面同时展示六项指标，采用两列三行的独立大图，
窄屏自动单列；统一模型顺序、放大名称与数值，并以细线区分模型组，不折叠任何指标。
详细口径可展开，完整图、单指标图与 CSV 仍可下载，独立页面内嵌六张图片便于离线查看。
旧版 `comparison.json` 需先运行 `python -m scripts.compare_models` 补齐模型平均及一致性指标后再重绘。

重复调用一致性复用四个阈值轮次的 `score`，无需重新调用模型：阈值只用于本地判定，不进入请求。
v1、v2 分开统计，要求四轮完整落盘，并按 `sample_id` 取四次均成功的样本。
每个样本计算总体标准差（`ddof=0`）、极差，再取样本平均；判定翻转率则将四次分数统一按
`score >= 0.5` 判定，统计出现过不同判定的样本占比，不能直接比较原 CSV 的不同阈值 `prediction`。
三项均越低越一致，但稳定不代表正确；历史服务版本、路由、运行时段变化也可能影响结果。
图内标准差与极差乘以 100 展示，CSV 保留 0–1 原始分数单位，同时报告成功交集覆盖率与请求失败率。
模型宏平均使用各提示词内所有模型四次均成功的共同样本，对共同非 PII 数据集等权平均；
缺轮、部分落盘或无共同成功样本的指标标 N/A，不补零。
逐数据集稳定性图也只展示所有模型四轮完整且有共同成功样本的数据集，并使用同一批共同样本计算；
零波动以基线标记和数值 0 显示，不是缺测。完整逐模型记录仍保留在报告与 CSV 中。
原阈值敏感性曲线混有重复调用波动，不代表固定分数下的纯阈值效应。
原始结果导出到 `stability.csv`、`stability_paired.csv`、`stability_overview.csv`。

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
python -m scripts.plot_model_comparison --report outputs/comparison_historical_confidence/comparison.json --output-dir outputs/comparison_historical_confidence/figures --dpi 600
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

## Corrected live classification reruns

Corrected live reruns using `answers.score.noul` are published separately in
[the corrected classification results](outputs/corrected_classification_2026-10-28/README.md).
The original Fastino/Laya raw output directories remain historical snapshots.
The active [comparison report](outputs/comparison/comparison.md) and
[figure gallery](outputs/comparison/figures/index.html) now use corrected Noul scores.
The previous report and figures are archived under
`outputs/comparison_historical_confidence/`; the older comparison instructions
above reproduce that historical workflow and are superseded for current results.

To reproduce or resume the GLiDE rerun with the existing prepared rows:

```bash
python -m scripts.rerun_classification --provider fastino.ai \
  --output-dir outputs/corrected_classification_2026-10-28 --workers 32
python -m scripts.report_classification_rerun \
  --input-dir outputs/corrected_classification_2026-10-28
```

Set `FASTINO_API_KEY` in the environment. Laya uses the same command with
`--provider laya-ai.com` and `LAYA_API_KEY`; `--env-file PATH` can load a local
credential file. The runner checkpoints each response, retries transient failures,
and stops on authorization or billing denials. Resuming reuses successful requests
only when input hashes, prompt, model, and endpoint still match. Use a fresh output
directory for a new independent run. One live probability is requested per row and
prompt; threshold metrics for 0.5–0.8 are derived offline, not repeated API calls.
The new report compares against historical Jev outputs only after matching text,
labels, and successful sample IDs. PII classification remains detection-only.

Refresh the complete corrected presentation without additional API calls:

```bash
python -m scripts.publish_corrected_comparison
```

This refreshes reports, JSON/CSV data, PNG/SVG/PDF charts, the figure gallery,
and `fig13_three_dataset_averages.png`. The original figure uses v1 / 0.5;
v2 results are shown separately. The three datasets receive equal weight.
Supplied aggregate results are retained in a separate reference table, not a
combined ranking. Historical latency is not compared with concurrent request
latency. All-positive PII document diagnostics remain outside classification
quality averages. AUPRC is average precision computed from verified saved scores.

`provider_status.json` in the corrected rerun directory declares the expected
dataset/prompt matrix. A provider is published only after that matrix completes;
Laya remains pending while its resumed run proceeds. Billing restrictions retain
their history with a resolved status once successful calls resume. These reports
use the historical prepared rows, not the new split policy from PR #19.
