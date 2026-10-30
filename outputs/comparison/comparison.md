# 模型能力对比报告

生成时间（UTC）：2026-10-30T03:55:14.909883+00:00

## 结论

固定 v1 / 0.5、共同完成的 5 个非 PII 数据集上，typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 的宏平均 F1 最高（76.28%），宏平均 AUROC 为 93.29%。这是当前快照下的描述性结果，不代表未测试任务或统计显著性。

同一基线比较：typesafe.ai / request model=jev-latest；resolved model=待接口返回确认：F1 76.28%、误报率 11.33%、漏报率 18.75%；fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2：F1 71.82%、误报率 12.85%、漏报率 19.40%；laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions：F1 46.43%、误报率 23.35%、漏报率 42.41%。高误报意味着正常内容被拦截，高漏报意味着危险内容被放过。

分数据集基线 F1 最优：aegis2：fastino.ai（84.00%）；hatecheck：fastino.ai（98.33%）；jailbreak-classification：laya-ai.com（99.62%）；jailbreakhub：typesafe.ai（42.57%）；toxigen：typesafe.ai（65.87%）。

综合表现领先仍不等于可直接上线：typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 在自身最弱的共同基线数据集 jailbreakhub 上 F1 为 42.57%、误报率 22.72%、漏报率 12.89%，需结合业务容忍度单独选阈值，而不是照搬统一 0.5。

相同基线任务中，fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 的日志摊销平均耗时最低（223.70 ms/样本）。包含网络、客户端开销和失败等待，且各轮非同一时刻运行；不能解释为纯模型推理速度。

laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions 仅完整落盘 49/64 个配置；缺测不记零分，也不把已有任务的均值当作全套能力，当前排名是阶段性结论。

扩展两两比较（fastino.ai vs laya-ai.com，共同完整且有共同有效样本的 6 个非 PII 数据集）：fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 宏 F1 66.46%；laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions 宏 F1 42.86%。此口径覆盖范围更广，但不能与三家共同基线的数值混排。

扩展两两比较（fastino.ai vs typesafe.ai，共同完整且有共同有效样本的 6 个非 PII 数据集）：fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 宏 F1 66.46%；typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 宏 F1 72.43%。此口径覆盖范围更广，但不能与三家共同基线的数值混排。

以下配置虽完整落盘但全部请求失败：fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 / jailbreak-classification / v1 / 0.6（1044/1044）。这些配置没有质量分数，不参与质量均值；这是接口可靠性问题，不能当作缺测或正常能力结果忽略。

PII 是“是否含 PII”的普通二分类，不是实体抽取或脱敏任务。当前数据全是正样本，恒预测“含 PII”即可获得 100% 检出率；现有输出没有脱敏文本，不能评价实体定位、泄露率或脱敏质量，不应据此选定脱敏模型。

选型优先参考固定配置的 F1、AUROC、误报/漏报及错误率；表中的最优提示词/阈值是在同一测试集上挑选的探索性上限，部署前需独立验证集重新选参并复测。

OpenJev 4B 及其他内部模型为用户提供的汇总结果，仅加入独立参考图表，未提供逐样本输出，不参与上述共同成功样本基线、排名或重复调用一致性分析。

## 模型身份与证据

以下模型身份由用户补充，区分请求名称、实际解析名称、服务模型与路由模型，不是从历史 CSV 或日志反查得到。TypeSafe 的 resolved model 需以接口返回为准；现有记录未提供该值，因此保持未确认，不将 jev-latest 当作固定后端版本。

| 供应商 | Request model | Resolved model | Serving model | Routed model |
| --- | --- | --- | --- | --- |
| fastino.ai | fastino/GLiDE | glide_v2 | — | — |
| laya-ai.com | typed-decisions | — | laya-rl-agent | typed-decisions |
| typesafe.ai | jev-latest | 待接口返回确认 | — | — |

## 比较口径

固定基线：v1 / threshold=0.5。只使用所有所选供应商均完整落盘的数据集，按 sample_id 对齐共同成功样本；读取时统一 CRLF/LF 换行后核对参考文本哈希、标签、唯一 ID、分数与预测一致性。宏平均对数据集等权，避免大数据集主导；不混入 PII，不把 8 次提示词/阈值运行当作独立样本。

参考输入按原 runner 的 pandas.read_csv 语义读取（包括空值转字符串和 NUL 字符截断）；这反映当时实际发给模型的内容，不代表原始数据字节完整性。例如 detect-jailbreak 原始文本中存在 NUL，原 runner 已发生截断，各供应商对比沿用相同输入。

共同基线数据集：aegis2、hatecheck、jailbreak-classification、jailbreakhub、toxigen。

质量指标排除请求错误，但覆盖率与错误数独立展示，避免将失败隐藏成高准确率。全部配置明细使用各文件自身的有效样本；跨模型公平比较请看共同样本表。complete 仅表示所有参考样本均已落盘，不表示请求全部成功。missing 表示未发现输出；in_progress 表示日志有记录但无输出，不保证进程仍在运行。

日志 elapsed / processed 是整轮摊销平均每样本耗时，含失败等待、客户端及网络开销；仅 CSV 与日志完整计数一致的轮次进入耗时比较。基线耗时按处理行数加权；Avg latency 则对共同基线各数据集的整轮平均耗时等权平均，缺少任一完整计时即不提供均值。它使用全部处理样本，质量指标则使用共同成功样本。没有单请求计时，不能计算请求 P50/P95/P99。报告不会从 50 条进度块伪造请求分位数。

**PII 是“是否含 PII”的普通二分类，不是脱敏质量评测。** 当前数据只有正样本，仅报告正样本检出率，单类 AUROC、误报率、平衡准确率无定义；没有预测脱敏文本，不计算 masking/span 指标。未输出原始文本、敏感字段或完整错误内容；报告可离线打开，无 CDN、无联网请求。

## 共同基线总览

| 供应商 / 模型 | 完整配置 | 共同数据集 | Average Accuracy | Avg Precision | Avg Recall | 宏 F1 | 宏平衡准确率 | 宏 AUROC | 宏误报率 | 宏漏报率 | Avg latency ms（数据集等权） | 基线平均耗时 ms（样本加权） | 全套请求错误/已落盘行 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 64/64 | 5 | 81.42% | 78.86% | 80.60% | 71.82% | 83.88% | 93.28% | 12.85% | 19.40% | 222.48 | 223.70 | 5386/454584 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 49/64 | 5 | 60.74% | 72.97% | 57.59% | 46.43% | 67.12% | 77.46% | 23.35% | 42.41% | 1148.44 | 1037.82 | 3712/366162 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 64/64 | 5 | 84.62% | 80.09% | 81.25% | 76.28% | 84.96% | 93.29% | 11.33% | 18.75% | 228.52 | 230.65 | 0/454584 |

## 逐数据集固定基线（共同成功样本）

| 数据集 | 供应商 / 模型 | 共同样本 | 覆盖率 | Accuracy | Precision | Recall | F1 | AUROC | 平衡准确率 | 误报率 | 漏报率 | 多数类准确率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 6789 | 100.00% | 65.49% | 94.77% | 32.28% | 48.15% | 84.00% | 65.26% | 1.76% | 67.72% | 50.35% |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 6789 | 100.00% | 55.77% | 90.89% | 12.13% | 21.41% | 72.66% | 55.47% | 1.20% | 87.87% | 50.35% |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 6789 | 100.00% | 73.91% | 94.00% | 50.70% | 65.87% | 87.52% | 73.75% | 3.19% | 49.30% | 50.35% |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 3728 | 100.00% | 97.69% | 97.62% | 99.06% | 98.33% | 99.66% | 96.87% | 5.32% | 0.94% | 68.75% |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 3728 | 100.00% | 70.20% | 82.01% | 72.57% | 77.00% | 77.24% | 68.77% | 35.02% | 27.43% | 68.75% |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 3728 | 100.00% | 94.82% | 93.92% | 98.87% | 96.33% | 99.28% | 92.40% | 14.08% | 1.13% | 68.75% |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 1964 | 100.00% | 83.71% | 89.27% | 79.32% | 84.00% | 91.98% | 84.08% | 11.16% | 20.68% | 53.92% |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 1964 | 100.00% | 49.29% | 81.19% | 7.74% | 14.14% | 66.14% | 52.82% | 2.10% | 92.26% | 53.92% |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 1964 | 100.00% | 80.96% | 85.20% | 78.28% | 81.59% | 90.49% | 81.18% | 15.91% | 21.72% | 53.92% |
| jailbreak-classification | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 1044 | 100.00% | 94.73% | 92.14% | 97.91% | 94.94% | 99.32% | 94.70% | 8.51% | 2.09% | 50.48% |
| jailbreak-classification | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 1044 | 100.00% | 99.62% | 99.62% | 99.62% | 99.62% | 99.99% | 99.62% | 0.39% | 0.38% | 50.48% |
| jailbreak-classification | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 1044 | 100.00% | 95.21% | 99.18% | 91.27% | 95.06% | 99.43% | 95.25% | 0.77% | 8.73% | 50.48% |
| jailbreakhub | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 15135 | 99.97% | 65.47% | 20.48% | 94.44% | 33.66% | 91.43% | 78.48% | 37.49% | 5.56% | 90.72% |
| jailbreakhub | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 15135 | 99.97% | 28.83% | 11.16% | 95.87% | 19.99% | 71.26% | 58.92% | 78.03% | 4.13% | 90.72% |
| jailbreakhub | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 15135 | 99.97% | 78.20% | 28.17% | 87.11% | 42.57% | 89.74% | 82.20% | 22.72% | 12.89% | 90.72% |

## 两两共同基线（利用已完成任务，覆盖范围各不相同）

每一对供应商单独取共同完整数据集与共同成功样本，固定 v1 / 0.5；不同配对覆盖的数据集可能不同，因此只可在同一配对内部比较。共同成功样本为零的数据集不参与质量均值，排除情况如下表明确列出。

| 配对 | 供应商 / 模型 | 共同数据集 | 宏 F1 | 宏 AUROC | 宏平衡准确率 | 宏误报率 | 宏漏报率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai vs laya-ai.com | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2、detect-jailbreak、hatecheck、jailbreak-classification、jailbreakhub、toxigen | 66.46% | 93.43% | 83.38% | 16.58% | 16.66% |
| fastino.ai vs laya-ai.com | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2、detect-jailbreak、hatecheck、jailbreak-classification、jailbreakhub、toxigen | 42.86% | 76.35% | 66.49% | 30.61% | 36.40% |
| fastino.ai vs typesafe.ai | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2、detect-jailbreak、hatecheck、jailbreak-classification、jailbreakhub、toxigen | 66.46% | 93.43% | 83.38% | 16.58% | 16.66% |
| fastino.ai vs typesafe.ai | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2、detect-jailbreak、hatecheck、jailbreak-classification、jailbreakhub、toxigen | 72.43% | 93.22% | 85.02% | 12.32% | 17.65% |
| laya-ai.com vs typesafe.ai | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2、hatecheck、jailbreak-classification、jailbreakhub、toxigen | 46.43% | 77.46% | 67.12% | 23.35% | 42.41% |
| laya-ai.com vs typesafe.ai | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2、hatecheck、jailbreak-classification、jailbreakhub、toxigen | 76.28% | 93.29% | 84.96% | 11.33% | 18.75% |

| 配对 | 数据集 | 共同有效样本 | 共同覆盖率 | 质量评估状态 |
| --- | --- | --- | --- | --- |
| fastino.ai vs laya-ai.com | toxigen | 6789 | 100.00% | 参与均值 |
| fastino.ai vs typesafe.ai | toxigen | 6789 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | toxigen | 6789 | 100.00% | 参与均值 |
| fastino.ai vs laya-ai.com | hatecheck | 3728 | 100.00% | 参与均值 |
| fastino.ai vs typesafe.ai | hatecheck | 3728 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | hatecheck | 3728 | 100.00% | 参与均值 |
| fastino.ai vs laya-ai.com | aegis2 | 1964 | 100.00% | 参与均值 |
| fastino.ai vs typesafe.ai | aegis2 | 1964 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | aegis2 | 1964 | 100.00% | 参与均值 |
| fastino.ai vs laya-ai.com | detect-jailbreak | 18253 | 99.97% | 参与均值 |
| fastino.ai vs typesafe.ai | detect-jailbreak | 18258 | 100.00% | 参与均值 |
| fastino.ai vs laya-ai.com | jailbreak-classification | 1044 | 100.00% | 参与均值 |
| fastino.ai vs typesafe.ai | jailbreak-classification | 1044 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | jailbreak-classification | 1044 | 100.00% | 参与均值 |
| fastino.ai vs laya-ai.com | jailbreakhub | 15135 | 99.97% | 参与均值 |
| fastino.ai vs typesafe.ai | jailbreakhub | 15140 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | jailbreakhub | 15135 | 99.97% | 参与均值 |

## 提示词与阈值探索（同测试集选优，非公平总排名）

以下按各文件自身成功样本选 F1 最优；不同轮次可能包含不同请求错误，不能将该结果当作独立验证后的提升。完整 v1/v2 × 4 阈值结果见明细。

| 数据集 | 供应商 / 模型 | 最优提示词 / 阈值 | 最优 F1 | 自身基线 F1 | 误报率 | 漏报率 | 可用配置 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.5 | 55.89% | 48.15% | 91.93% | 25.07% | 8/8 |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | 98.33% | 98.33% | 5.32% | 0.94% | 8/8 |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | 84.00% | 84.00% | 11.16% | 20.68% | 8/8 |
| detect-jailbreak | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | 39.67% | 39.67% | 35.22% | 3.00% | 8/8 |
| jailbreak-classification | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | 94.94% | 94.94% | 8.51% | 2.09% | 7/8 |
| jailbreakhub | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | 33.66% | 33.66% | 37.51% | 5.55% | 8/8 |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.5 | 66.34% | 21.41% | 100.00% | 0.00% | 8/8 |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.5 | 81.48% | 77.00% | 100.00% | 0.00% | 8/8 |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.5 | 70.06% | 14.14% | 100.00% | 0.00% | 8/8 |
| detect-jailbreak | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | 25.01% | 25.01% | 66.93% | 6.36% | 7/8 |
| jailbreak-classification | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | 99.62% | 99.62% | 0.39% | 0.38% | 2/8 |
| jailbreakhub | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.8 | 23.07% | 19.99% | 59.14% | 11.54% | 8/8 |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.5 | 69.28% | 65.87% | 3.95% | 44.88% | 8/8 |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.7 | 97.43% | 96.33% | 4.46% | 3.08% | 8/8 |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.5 | 81.59% | 81.59% | 15.91% | 21.72% | 8/8 |
| detect-jailbreak | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.8 | 64.23% | 53.13% | 8.92% | 17.71% | 8/8 |
| jailbreak-classification | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.5 | 95.16% | 95.06% | 0.77% | 8.54% | 8/8 |
| jailbreakhub | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.8 | 51.76% | 42.57% | 13.11% | 20.36% | 8/8 |

## 重复调用一致性：口径与逐数据集结果

将同一供应商、数据集、提示词的 0.5/0.6/0.7/0.8 四轮视作四次重复调用。阈值只在客户端用于判定，不进入模型请求；不直接比较各 CSV 原有 prediction。v1、v2 分开统计，按 sample_id 对齐，四轮均完整落盘后仅使用四次均成功的样本。少一轮或存在部分落盘即标 incomplete_repeats，不以较少轮次替代；四轮完整但无共同成功样本标 no_common_success，稳定性无定义而不是零波动。

增量更新中，四轮汇总齐全但缺少重新计算所需的历史逐样本预测时标 unavailable_raw；不从汇总分数推算稳定性，也不沿用已被新 CSV 替换的旧一致性指标。

每个样本计算四个 score 的总体标准差（ddof=0）与极差，再对样本取平均；分数保持原始 0–1 单位。判定统一使用 score >= 0.5，翻转率为四次判定中至少出现一次正类、一次负类的样本比例，不是相邻轮次翻转次数。三项指标均越低表示越一致。覆盖率为四次均成功样本 / 参考样本；请求失败率为已有 CSV 的失败行 / 已落盘总行，缺失轮次不算成功也不算请求失败。

逐数据集表使用各模型自身的四轮成功交集，样本可能不同，不作为公平总排名。历史调用可能混有后端版本、路由或运行时段变化，无法将差异完全归因于模型随机性；仅四次调用是探索性观察，不提供伪造置信区间。稳定不代表正确，恒定错误输出也可能零波动。原阈值敏感性曲线同时混有重复调用波动，不是固定 score 下的纯阈值效应。

| 数据集 | 供应商 / 模型 | 提示词 | 状态 | 已有轮次 | 完整轮次 | 四次均成功/参考 | 覆盖率 | 平均分数标准差 | 平均分数极差 | 判定翻转样本 | 判定翻转率（统一 0.5） | 失败/已落盘请求 | 请求失败率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.286341 | 0.661275 | 4823 | 71.04% | 0/27156 | 0.00% |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 6781/6789 | 99.88% | 0.218065 | 0.503599 | 6333 | 93.39% | 8/27156 | 0.03% |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.005146 | 0.012672 | 79 | 1.16% | 0/27156 | 0.00% |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/27156 | 0.00% |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 6780/6789 | 99.87% | 0.000000 | 0.000000 | 0 | 0.00% | 9/27156 | 0.03% |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.004798 | 0.011797 | 88 | 1.30% | 0/27156 | 0.00% |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.123127 | 0.284349 | 1174 | 31.49% | 0/14912 | 0.00% |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.052235 | 0.120632 | 1460 | 39.16% | 0/14912 | 0.00% |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.005095 | 0.012538 | 50 | 1.34% | 0/14912 | 0.00% |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/14912 | 0.00% |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 3726/3728 | 99.95% | 0.000000 | 0.000000 | 0 | 0.00% | 2/14912 | 0.01% |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.004581 | 0.011274 | 35 | 0.94% | 0/14912 | 0.00% |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.204869 | 0.473125 | 1078 | 54.89% | 0/7856 | 0.00% |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.230586 | 0.532515 | 1863 | 94.86% | 0/7856 | 0.00% |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.006415 | 0.015850 | 31 | 1.58% | 0/7856 | 0.00% |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 1961/1964 | 99.85% | 0.000000 | 0.000000 | 0 | 0.00% | 3/7856 | 0.04% |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/7856 | 0.00% |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.005369 | 0.013299 | 34 | 1.73% | 0/7856 | 0.00% |
| detect-jailbreak | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | unavailable_raw | 4/4 | 4/4 | 0/18258 | — | — | — | — | — | 2/73032 | 0.00% |
| detect-jailbreak | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | unavailable_raw | 4/4 | 4/4 | 0/18258 | — | — | — | — | — | 29/73032 | 0.04% |
| detect-jailbreak | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 18258/18258 | 100.00% | 0.005082 | 0.012416 | 155 | 0.85% | 0/73032 | 0.00% |
| detect-jailbreak | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 14634/18258 | 80.15% | 0.018741 | 0.042424 | 624 | 4.26% | 3624/73032 | 4.96% |
| detect-jailbreak | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 3/4 | 3/4 | 0/18258 | — | — | — | — | — | 15/54774 | 0.03% |
| detect-jailbreak | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 18258/18258 | 100.00% | 0.004493 | 0.010897 | 144 | 0.79% | 0/73032 | 0.00% |
| jailbreak-classification | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | no_common_success | 4/4 | 4/4 | 0/1044 | 0.00% | — | — | — | — | 1754/4176 | 42.00% |
| jailbreak-classification | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | incomplete_repeats | 1/4 | 1/4 | 0/1044 | — | — | — | — | — | 0/1044 | 0.00% |
| jailbreak-classification | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 1044/1044 | 100.00% | 0.002424 | 0.005766 | 4 | 0.38% | 0/4176 | 0.00% |
| jailbreak-classification | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 1044/1044 | 100.00% | 0.014281 | 0.028561 | 24 | 2.30% | 0/4176 | 0.00% |
| jailbreak-classification | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 1/4 | 1/4 | 0/1044 | — | — | — | — | — | 0/1044 | 0.00% |
| jailbreak-classification | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 1044/1044 | 100.00% | 0.002192 | 0.005239 | 2 | 0.19% | 0/4176 | 0.00% |
| jailbreakhub | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 15140/15140 | 100.00% | 0.230010 | 0.531962 | 10904 | 72.02% | 0/60560 | 0.00% |
| jailbreakhub | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 15130/15140 | 99.93% | 0.056335 | 0.130100 | 3074 | 20.32% | 24/60560 | 0.04% |
| jailbreakhub | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 15140/15140 | 100.00% | 0.005015 | 0.012223 | 142 | 0.94% | 0/60560 | 0.00% |
| jailbreakhub | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 15139/15140 | 99.99% | 0.000112 | 0.000260 | 2 | 0.01% | 1/60560 | 0.00% |
| jailbreakhub | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 15129/15140 | 99.93% | 0.000000 | 0.000000 | 0 | 0.00% | 28/60560 | 0.05% |
| jailbreakhub | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 15140/15140 | 100.00% | 0.004183 | 0.010139 | 109 | 0.72% | 0/60560 | 0.00% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/3600 | 0.00% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | incomplete_repeats | 3/4 | 3/4 | 0/900 | — | — | — | — | — | 1184/2700 | 43.85% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.004437 | 0.010789 | 8 | 0.89% | 0/3600 | 0.00% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/3600 | 0.00% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 294/900 | 32.67% | 0.000000 | 0.000000 | 0 | 0.00% | 2413/3600 | 67.03% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.002678 | 0.006433 | 1 | 0.11% | 0/3600 | 0.00% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 8998/9000 | 99.98% | 0.003892 | 0.008988 | 53 | 0.59% | 2/36000 | 0.01% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | incomplete_repeats | 1/4 | 1/4 | 0/9000 | — | — | — | — | — | 0/9000 | 0.00% |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 9000/9000 | 100.00% | 0.004347 | 0.010594 | 57 | 0.63% | 0/36000 | 0.00% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 9000/9000 | 100.00% | 0.000611 | 0.001415 | 5 | 0.06% | 0/36000 | 0.00% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 0/4 | 0/4 | 0/9000 | — | — | — | — | — | 0/0 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 9000/9000 | 100.00% | 0.002616 | 0.006252 | 26 | 0.29% | 0/36000 | 0.00% |

## 重复调用一致性：共同样本模型均值

每个提示词单独选取所有供应商四轮均完整的数据集，在每个数据集内进一步取所有供应商全部四轮的成功样本交集。只对有共同成功样本的非 PII 数据集等权平均，不与上方各自样本均值混排。不同提示词的共同数据集或样本可能不同；PII 仅在逐数据集一致性中展示。覆盖率按共同样本总数 / 参考样本总数计算；请求失败率使用这些数据集全部已落盘请求。

| 供应商 / 模型 | 提示词 | 共同数据集 | 共同样本/参考 | 共同覆盖率 | 宏平均分数标准差 | 宏平均分数极差 | 宏平均判定翻转率 | 请求失败率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | aegis2、hatecheck、jailbreakhub、toxigen | 27603/27621 | 99.93% | 0.211119 | 0.487752 | 57.37% | 0.00% |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | aegis2、hatecheck、jailbreakhub、toxigen | 27603/27621 | 99.93% | 0.139305 | 0.321712 | 61.93% | 0.03% |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | aegis2、hatecheck、jailbreakhub、toxigen | 27603/27621 | 99.93% | 0.005418 | 0.013320 | 1.26% | 0.00% |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | aegis2、hatecheck、jailbreakhub、toxigen | 27595/27621 | 99.91% | 0.000028 | 0.000065 | 0.00% | 0.00% |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | aegis2、hatecheck、jailbreakhub、toxigen | 27595/27621 | 99.91% | 0.000000 | 0.000000 | 0.00% | 0.04% |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | aegis2、hatecheck、jailbreakhub、toxigen | 27595/27621 | 99.91% | 0.004734 | 0.011631 | 1.17% | 0.00% |

## PII 普通二分类（是否含 PII，不是脱敏质量）

| 数据集 | 供应商 / 模型 | 提示词 / 阈值 | 状态 | 有效/预期 | 正样本检出率 |
| --- | --- | --- | --- | --- | --- |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | complete | 900/900 | 86.11% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | complete | 900/900 | 100.00% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.5 | complete | 900/900 | 94.33% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.6 | complete | 900/900 | 82.33% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.6 | complete | 310/900 | 82.90% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.6 | complete | 900/900 | 93.00% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.7 | complete | 900/900 | 77.78% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.7 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.7 | complete | 900/900 | 91.78% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.8 | complete | 900/900 | 70.11% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.8 | complete | 306/900 | 5.88% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.8 | complete | 900/900 | 89.11% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.5 | complete | 900/900 | 92.56% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.5 | complete | 299/900 | 100.00% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.5 | complete | 900/900 | 95.67% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.6 | complete | 900/900 | 90.11% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.6 | complete | 298/900 | 60.40% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.6 | complete | 900/900 | 95.22% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.7 | complete | 900/900 | 87.78% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.7 | complete | 296/900 | 21.28% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.7 | complete | 900/900 | 93.67% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.8 | complete | 900/900 | 83.67% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.8 | complete | 294/900 | 3.74% |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.8 | complete | 900/900 | 92.33% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | complete | 9000/9000 | 85.96% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | complete | 9000/9000 | 100.00% |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.5 | complete | 9000/9000 | 94.94% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.6 | complete | 9000/9000 | 82.18% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.6 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.6 | complete | 9000/9000 | 93.70% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.7 | complete | 8999/9000 | 77.52% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.7 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.7 | complete | 9000/9000 | 92.17% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.8 | complete | 8999/9000 | 68.99% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.8 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.8 | complete | 9000/9000 | 89.30% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.5 | complete | 9000/9000 | 92.91% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.5 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.5 | complete | 9000/9000 | 95.97% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.6 | complete | 9000/9000 | 90.51% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.6 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.6 | complete | 9000/9000 | 95.38% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.7 | complete | 9000/9000 | 87.76% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.7 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.7 | complete | 9000/9000 | 94.54% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.8 | complete | 9000/9000 | 83.04% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.8 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.8 | complete | 9000/9000 | 93.12% |

## 配置覆盖与日志耗时

| 供应商 / 模型 | 数据集 | 提示词 / 阈值 | 状态 | 落盘/预期 | 有效覆盖率 | 错误率 | 日志进度 | 日志累计 s | 整轮平均 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1489.31 | 219.37 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 5810.44 | 855.86 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1552.10 | 228.62 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v1 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 3133.50 | 461.56 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v1 / 0.6 | complete | 6789/6789 | 99.99% | 0.01% | 6789/6789 | 2970.18 | 437.50 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v1 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1560.43 | 229.85 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v1 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2662.40 | 392.16 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v1 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 3039.83 | 447.76 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v1 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1564.38 | 230.43 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v1 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2503.40 | 368.74 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v1 / 0.8 | complete | 6789/6789 | 99.90% | 0.10% | 6789/6789 | 3331.06 | 490.66 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v1 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1536.03 | 226.25 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v2 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2312.30 | 340.60 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v2 / 0.5 | complete | 6789/6789 | 99.96% | 0.04% | 6789/6789 | 3996.74 | 588.71 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v2 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1557.01 | 229.34 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v2 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2261.50 | 333.11 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v2 / 0.6 | complete | 6789/6789 | 99.99% | 0.01% | 6789/6789 | 3799.09 | 559.60 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v2 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1567.75 | 230.93 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v2 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2205.70 | 324.89 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v2 / 0.7 | complete | 6789/6789 | 99.97% | 0.03% | 6789/6789 | 4839.83 | 712.89 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v2 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1640.30 | 241.61 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v2 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2191.40 | 322.79 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v2 / 0.8 | complete | 6789/6789 | 99.96% | 0.04% | 6789/6789 | 3853.96 | 567.68 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | toxigen | v2 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1565.98 | 230.66 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v1 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 831.63 | 223.08 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v1 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 3771.80 | 1011.75 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v1 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 855.00 | 229.35 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v1 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1236.90 | 331.79 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v1 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1572.98 | 421.94 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v1 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 857.90 | 230.12 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v1 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1090.90 | 292.62 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v1 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1477.33 | 396.28 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v1 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 841.24 | 225.65 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v1 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1422.60 | 381.60 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v1 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1586.41 | 425.54 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v1 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 855.70 | 229.53 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v2 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1719.00 | 461.11 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v2 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1712.24 | 459.29 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v2 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 846.55 | 227.08 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v2 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1310.10 | 351.42 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v2 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1648.89 | 442.30 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v2 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 853.83 | 229.03 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v2 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1697.90 | 455.45 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v2 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1611.52 | 432.27 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v2 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 831.23 | 222.97 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v2 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1462.30 | 392.25 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v2 / 0.8 | complete | 3728/3728 | 99.95% | 0.05% | 3728/3728 | 1990.75 | 534.00 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | hatecheck | v2 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 835.13 | 224.02 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 434.24 | 221.10 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 2582.22 | 1314.78 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 435.62 | 221.80 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v1 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 755.10 | 384.47 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v1 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 898.52 | 457.49 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v1 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 428.99 | 218.43 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v1 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 995.80 | 507.03 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v1 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 927.80 | 472.40 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v1 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 427.75 | 217.80 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v1 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1112.80 | 566.60 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v1 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1009.67 | 514.09 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v1 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 438.14 | 223.08 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v2 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1028.60 | 523.73 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v2 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1066.86 | 543.21 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v2 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 434.07 | 221.01 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v2 / 0.6 | complete | 1964/1964 | 99.85% | 0.15% | 1964/1964 | 1977.70 | 1006.98 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v2 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1018.75 | 518.71 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v2 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 434.81 | 221.39 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v2 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1209.70 | 615.94 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v2 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1043.24 | 531.18 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v2 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 437.34 | 222.68 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v2 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1048.30 | 533.76 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v2 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1112.68 | 566.54 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2 | v2 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 441.41 | 224.75 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4189.81 | 229.48 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 17633.25 | 965.78 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4091.16 | 224.08 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v1 / 0.6 | complete | 18258/18258 | 99.99% | 0.01% | 18258/18258 | 7457.90 | 408.47 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v1 / 0.6 | complete | 18258/18258 | 99.92% | 0.08% | 18258/18258 | 15582.19 | 853.44 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v1 / 0.6 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4119.35 | 225.62 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v1 / 0.7 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 6750.40 | 369.72 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v1 / 0.7 | complete | 18258/18258 | 99.98% | 0.02% | 18258/18258 | 16128.16 | 883.35 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v1 / 0.7 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4142.22 | 226.87 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v1 / 0.8 | complete | 18258/18258 | 99.99% | 0.01% | 18258/18258 | 5721.80 | 313.39 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v1 / 0.8 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 15647.33 | 857.01 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v1 / 0.8 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4103.78 | 224.77 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v2 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 6108.10 | 334.54 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v2 / 0.5 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 14717.17 | 806.07 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v2 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4160.19 | 227.86 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v2 / 0.6 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 8883.70 | 486.56 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v2 / 0.6 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 15273.93 | 836.56 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v2 / 0.6 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4080.60 | 223.50 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v2 / 0.7 | complete | 18258/18258 | 99.80% | 0.20% | 18258/18258 | 10732.20 | 587.81 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v2 / 0.7 | complete | 18258/18258 | 99.98% | 0.02% | 18258/18258 | 15966.81 | 874.51 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v2 / 0.7 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4104.36 | 224.80 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v2 / 0.8 | complete | 18258/18258 | 80.35% | 19.65% | 18258/18258 | 6199.20 | 339.53 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v2 / 0.8 | in_progress | 0/18258 | 0.00% | — | 3500/18258 | 19214.10 | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | detect-jailbreak | v2 / 0.8 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4043.71 | 221.48 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 232.46 | 222.66 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 1568.00 | 1501.92 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 239.84 | 229.73 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.6 | complete | 1044/1044 | 0.00% | 100.00% | 1044/1044 | 246.40 | 236.02 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.6 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 249.70 | 239.18 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.7 | complete | 1044/1044 | 31.99% | 68.01% | 1044/1044 | 299.80 | 287.16 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.7 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 241.61 | 231.43 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | — | — | — |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.8 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 239.06 | 228.99 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | — | — | — |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 783.63 | 750.60 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 241.01 | 230.85 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 299.70 | 287.07 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.6 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 249.60 | 239.08 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 349.60 | 334.87 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.7 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 246.67 | 236.27 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | — | — | — |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.8 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 240.36 | 230.23 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3424.81 | 226.21 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.5 | complete | 15140/15140 | 99.97% | 0.03% | 15140/15140 | 16016.56 | 1057.90 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3529.11 | 233.10 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 4845.10 | 320.02 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.6 | complete | 15140/15140 | 99.96% | 0.04% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3548.75 | 234.40 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 5602.20 | 370.03 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.7 | complete | 15140/15140 | 99.95% | 0.05% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3520.66 | 232.54 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 8423.20 | 556.35 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.8 | complete | 15140/15140 | 99.96% | 0.04% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3505.52 | 231.54 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 7124.20 | 470.55 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.5 | complete | 15140/15140 | 99.96% | 0.04% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3475.07 | 229.53 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 5187.80 | 342.66 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.6 | complete | 15140/15140 | 99.96% | 0.04% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3447.00 | 227.68 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 6065.60 | 400.63 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.7 | complete | 15140/15140 | 99.93% | 0.07% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3496.82 | 230.97 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.8 | complete | 15140/15140 | 99.99% | 0.01% | 15140/15140 | 6052.10 | 399.74 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.8 | complete | 15140/15140 | 99.96% | 0.04% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3400.87 | 224.63 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 376.80 | 418.67 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 527.51 | 586.12 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.96 | 219.95 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 318.20 | 353.56 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.6 | complete | 900/900 | 34.44% | 65.56% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 200.23 | 222.48 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 301.40 | 334.89 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.7 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 201.80 | 224.23 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 300.20 | 333.56 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.8 | complete | 900/900 | 34.00% | 66.00% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.96 | 219.95 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 370.00 | 411.11 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.5 | complete | 900/900 | 33.22% | 66.78% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.29 | 219.21 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 311.30 | 345.89 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.6 | complete | 900/900 | 33.11% | 66.89% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.06 | 218.95 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 334.90 | 372.11 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.7 | complete | 900/900 | 32.89% | 67.11% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 203.91 | 226.56 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 346.20 | 384.67 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.8 | complete | 900/900 | 32.67% | 67.33% | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 200.88 | 223.20 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v1 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 3028.20 | 336.47 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v1 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 5179.08 | 575.45 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v1 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2039.10 | 226.57 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v1 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 3119.40 | 346.60 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v1 / 0.6 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v1 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2012.36 | 223.60 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v1 / 0.7 | complete | 9000/9000 | 99.99% | 0.01% | 9000/9000 | 3326.00 | 369.56 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v1 / 0.7 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v1 / 0.7 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2031.76 | 225.75 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v1 / 0.8 | complete | 9000/9000 | 99.99% | 0.01% | 9000/9000 | 4030.30 | 447.81 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v1 / 0.8 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v1 / 0.8 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2042.81 | 226.98 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v2 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 5494.10 | 610.46 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v2 / 0.5 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v2 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2048.51 | 227.61 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v2 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 5078.10 | 564.23 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v2 / 0.6 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v2 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2040.31 | 226.70 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v2 / 0.7 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 3599.00 | 399.89 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v2 / 0.7 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v2 / 0.7 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2032.40 | 225.82 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v2 / 0.8 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2859.70 | 317.74 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v2 / 0.8 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_mini_10k | v2 / 0.8 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2097.96 | 233.11 |

## 请求可靠性（各自已落盘范围，不作为同任务排名）

| 供应商 / 模型 | 已落盘请求数 | 失败数 | 失败率 | 异常类型计数 |
| --- | --- | --- | --- | --- |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 454584 | 5386 | 1.18% | ConnectError: 1；HTTPStatusError: 5379；ReadTimeout: 5；RemoteProtocolError: 1 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 366162 | 3712 | 1.01% | ConnectError: 13；HTTPStatusError: 3693；ReadTimeout: 3；RemoteProtocolError: 3 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 454584 | 0 | 0.00% | 无 |

## OpenJev 内部汇总结果（4B，非共同样本基线）

OpenJev 4B 数值取自用户最初提供的清晰汇总表，分类阈值均为 0.5；补充 OCR 文本有错字，不据此修正数值或推断混淆矩阵。Rows 为总测试行数，有效样本数和错误数未可靠录入，保持未知，不将 Rows 当作有效样本数。OpenJev 只提供汇总指标，未提供逐样本 score、预测或模型版本；不能核验相同输入、共同成功样本或计算重复调用稳定性，不加入原有公平基线与排名。参考图按各模型自身成功样本统计，仅作描述性对照。原四模型优先使用每个数据集/提示词四轮完整运行的 elapsed/processed 算术平均。v1 缺少完整四轮计时时，仅在阈值 0.5 的完整单轮日志与全量 CSV 行数一致时，使用该单轮的 elapsed/processed，并标记为 single_complete_run；v2 不使用该回退。包含失败请求、等待、客户端及网络开销，不是纯推理时间；无可核验的完整计时标 N/A。并发日志无法逐阈值归属时，仅在四个阈值各启动一次、四次全量完成及保存均可核验，且无后续重跑时使用四轮耗时均值；不据此回填任一阈值的耗时。新增 OpenJev 32B、Microsoft Decision-1 为用户提供的内部 v1 / 0.5 汇总，耗时来自每个数据集一次完整运行的 elapsed/processed（max workers=8），不是原四模型的四轮均值，计时协议不同，仅作参考。OpenJev 32B 的原文件标记为 apus-openjev-35b-a3b，显示名称按用户确认使用 32B，不从文件名推断实际模型版本。其 ToxiGen 五项质量指标均已由用户补全，Accuracy 为 0.693975；按完整数据集范围计算均值，不缩减范围或反推指标。新增模型质量指标排除错误行，尤其 OpenJev 32B 的部分任务请求错误较多，不能将成功样本分数当作全量表现。Microsoft 日志在八个完整任务后中断，提供的耗时仅取已完成任务；均无单请求 P95。没有逐样本预测，不加入共同成功样本排名或重复调用一致性。

| 数据集 | 提示词 | 阈值 | Rows（总行数） | 有效样本数 | 错误数 | Accuracy | Precision | Recall | F1 | AUROC | 四轮平均耗时 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aegis2 | v1 | 0.5 | 1964 | 未知 | 未知 | 80.15% | 87.08% | 74.12% | 80.08% | 88.98% | 883.25 |
| aegis2 | v2 | 0.5 | 1964 | 未知 | 未知 | 77.48% | 90.97% | 64.68% | 75.61% | 88.81% | 907.94 |
| detect-jailbreak | v1 | 0.5 | 18258 | 未知 | 未知 | 89.93% | 52.03% | 79.79% | 62.99% | 92.85% | 1015.00 |
| detect-jailbreak | v2 | 0.5 | 18258 | 未知 | 未知 | 89.14% | 49.65% | 76.38% | 60.18% | 91.78% | 1013.58 |
| hatecheck | v1 | 0.5 | 3728 | 未知 | 未知 | 84.13% | 83.41% | 96.02% | 89.27% | 87.80% | 861.90 |
| hatecheck | v2 | 0.5 | 3728 | 未知 | 未知 | 86.47% | 87.46% | 93.75% | 90.50% | 89.94% | 915.34 |
| jailbreak-classification | v1 | 0.5 | 1044 | 未知 | 未知 | 89.73% | 99.76% | 79.81% | 88.68% | 98.89% | 990.77 |
| jailbreak-classification | v2 | 0.5 | 1044 | 未知 | 未知 | 92.04% | 99.78% | 84.41% | 91.45% | 98.60% | 978.70 |
| jailbreakhub | v1 | 0.5 | 15140 | 未知 | 未知 | 86.94% | 39.48% | 76.87% | 52.17% | 88.98% | 1017.51 |
| jailbreakhub | v2 | 0.5 | 15140 | 未知 | 未知 | 86.66% | 39.11% | 79.29% | 52.38% | 89.10% | 908.04 |
| pii_masking_mini_10k | v1 | 0.5 | 9000 | 未知 | 未知 | 89.92% | 100.00% | 89.92% | 94.69% | — | 916.30 |
| pii_masking_mini_10k | v2 | 0.5 | 9000 | 未知 | 未知 | 94.28% | 100.00% | 94.28% | 97.06% | — | 851.40 |
| pii_masking_nano_1k | v1 | 0.5 | 900 | 未知 | 未知 | 90.43% | 100.00% | 90.43% | 94.98% | — | 1050.33 |
| pii_masking_nano_1k | v2 | 0.5 | 900 | 未知 | 未知 | 93.18% | 100.00% | 93.18% | 96.47% | — | 953.69 |
| toxigen | v1 | 0.5 | 6789 | 未知 | 未知 | 68.91% | 83.99% | 46.19% | 59.60% | 80.09% | 859.56 |
| toxigen | v2 | 0.5 | 6789 | 未知 | 未知 | 70.68% | 86.76% | 48.31% | 62.06% | 80.78% | 859.95 |

## 新增内部模型汇总（非共同样本基线）

| 模型 | 数据集 | 提示词 | 总行数 | 有效样本数 | 错误数 | 有效覆盖率 | Accuracy | Precision | Recall | F1 | AUROC | 单轮平均耗时 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenJev 32B | aegis2 | v1 | 1964 | 1963 | 1 | 99.95% | 72.64% | 86.85% | 58.03% | 69.58% | 80.08% | 1076.03 |
| OpenJev 32B | detect-jailbreak | v1 | 18258 | 5425 | 12833 | 29.71% | 87.50% | 77.33% | 73.49% | 75.36% | 87.66% | 1257.87 |
| OpenJev 32B | hatecheck | v1 | 3728 | 3723 | 5 | 99.87% | 91.94% | 90.94% | 98.05% | 94.36% | 97.48% | 999.50 |
| OpenJev 32B | jailbreak-classification | v1 | 1044 | 1041 | 3 | 99.71% | 81.27% | 99.40% | 63.31% | 77.35% | 94.17% | 1183.80 |
| OpenJev 32B | jailbreakhub | v1 | 15140 | 5432 | 9708 | 35.88% | 86.30% | 73.75% | 72.81% | 73.28% | 87.69% | 1516.69 |
| OpenJev 32B | pii_masking_mini_10k | v1 | 9000 | 5436 | 3564 | 60.40% | 95.79% | 100.00% | 95.79% | 97.85% | — | 2550.39 |
| OpenJev 32B | pii_masking_nano_1k | v1 | 900 | 899 | 1 | 99.89% | 95.33% | 100.00% | 95.33% | 97.61% | — | 1196.94 |
| OpenJev 32B | toxigen | v1 | 6789 | 5444 | 1345 | 80.19% | 69.40% | 94.12% | 41.26% | 57.37% | 82.63% | 3380.10 |
| Microsoft Decision-1 | aegis2 | v1 | 1964 | 1962 | 2 | 99.90% | 80.78% | 88.55% | 73.89% | 80.56% | 89.91% | 1241.10 |
| Microsoft Decision-1 | detect-jailbreak | v1 | 18258 | 18255 | 3 | 99.98% | 72.79% | 27.18% | 91.04% | 41.87% | 91.87% | 1251.58 |
| Microsoft Decision-1 | hatecheck | v1 | 3728 | 3726 | 2 | 99.95% | 97.58% | 98.97% | 97.50% | 98.23% | 99.78% | 1238.43 |
| Microsoft Decision-1 | jailbreak-classification | v1 | 1044 | 1044 | 0 | 100.00% | 95.21% | 96.86% | 93.55% | 95.17% | 98.90% | 1235.97 |
| Microsoft Decision-1 | jailbreakhub | v1 | 15140 | 15136 | 4 | 99.97% | 71.23% | 23.02% | 89.61% | 36.64% | 88.36% | 1250.54 |
| Microsoft Decision-1 | pii_masking_mini_10k | v1 | 9000 | 8997 | 3 | 99.97% | 94.90% | 100.00% | 94.90% | 97.38% | — | 1244.82 |
| Microsoft Decision-1 | pii_masking_nano_1k | v1 | 900 | 900 | 0 | 100.00% | 94.78% | 100.00% | 94.78% | 97.32% | — | 1232.74 |
| Microsoft Decision-1 | toxigen | v1 | 6789 | 6788 | 1 | 99.99% | 71.20% | 93.33% | 45.22% | 60.92% | 85.93% | 1239.27 |

## 多模型参考均值（同数据集，非同成功样本）

固定分类阈值 0.5，v1/v2 分开统计，按各提示词共同测试的非 PII 数据集，对数据集等权平均。各模型使用自身有效样本，未对齐共同 sample_id，不能当作公平排名。新增模型仅参与已有提示词的参考均值，不缩减原有模型的 v2 范围。任一所选数据集的指标缺失则该指标均值保持 N/A，不缩减数据集范围。原四模型优先使用四轮完整日志均值；v1 缺失四轮均值时，可使用与全量 CSV 对齐的完整阈值 0.5 单轮计时。新增模型为用户提供的单轮计时，协议不同，仅作参考；各数据集耗时再等权平均。任一所选数据集缺少所需计时即不提供模型平均耗时，不用部分计时替代。

| 模型 | 提示词 | 共同数据集 | Average Accuracy | Avg Precision | Avg Recall | Avg F1 | Avg AUROC | Avg latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai | v1 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 79.22% | 69.87% | 83.34% | 66.46% | 93.43% | 334.77 |
| gliguard-LLMGuardrails-300M-fine-tuned | v1 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 93.73% | 89.22% | 85.55% | 87.13% | 96.25% | 228.59 |
| laya-ai.com | v1 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 57.21% | 63.22% | 63.60% | 42.86% | 76.35% | 1118.00 |
| typesafe.ai | v1 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 84.40% | 73.09% | 82.35% | 72.43% | 93.22% | 228.05 |
| OpenJev 4B | v1 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 83.30% | 74.29% | 75.47% | 72.13% | 89.60% | 938.00 |
| OpenJev 32B | v1 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 81.51% | 87.06% | 67.82% | 74.55% | 88.29% | 1569.00 |
| Microsoft Decision-1 | v1 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 81.47% | 71.32% | 81.80% | 68.90% | 92.46% | 1242.82 |
| fastino.ai | v2 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 43.81% | 39.85% | 88.85% | 51.17% | 62.68% | — |
| laya-ai.com | v2 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 40.47% | 40.47% | 100.00% | 53.56% | 43.37% | — |
| typesafe.ai | v2 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 84.20% | 73.00% | 82.09% | 72.16% | 93.04% | 228.01 |
| OpenJev 4B | v2 | toxigen、hatecheck、aegis2、detect-jailbreak、jailbreak-classification、jailbreakhub | 83.75% | 75.62% | 74.47% | 72.03% | 89.83% | 930.59 |

## 数据警告

outputs/laya-ai.com/run_model_suite_2026-10-21_083713.log: interleaved task starts cannot be assigned to progress；整份日志不用于逐配置耗时比较，不猜测并发进度的阈值归属。可核验的完整四轮均值仅用于四模型参考图。

outputs/laya-ai.com/run_model_suite_2026-10-21_084817.log: interleaved task starts cannot be assigned to progress；整份日志不用于逐配置耗时比较，不猜测并发进度的阈值归属。可核验的完整四轮均值仅用于四模型参考图。

fastino.ai/jailbreak-classification_v1_t0.5.csv：历史日志对应的预测文件已被新结果替换，不沿用该旧日志的耗时；质量指标仍使用新 CSV。

fastino.ai/jailbreak-classification_v1_t0.8.csv：历史日志对应的预测文件已被新结果替换，不沿用该旧日志的耗时；质量指标仍使用新 CSV。

fastino.ai/jailbreak-classification_v2_t0.5.csv：历史日志对应的预测文件已被新结果替换，不沿用该旧日志的耗时；质量指标仍使用新 CSV。

fastino.ai/jailbreak-classification_v2_t0.8.csv：历史日志对应的预测文件已被新结果替换，不沿用该旧日志的耗时；质量指标仍使用新 CSV。

本次为增量更新：未提供历史逐样本 CSV 的配置保留旧报告汇总，不代表重新测试。新 CSV 替换同配置旧结果，缺少新日志时不沿用旧耗时。只有配置均未变更、原始预测可对齐，或各模型均在全量参考样本上成功时，才保留或补充共同样本指标；无法核验的交集不参与公平均值。

detect-jailbreak/v1/0.5：缺少历史逐样本预测，无法重新对齐共同成功样本，排除共同样本比较。

fastino.ai vs laya-ai.com/detect-jailbreak：缺少历史逐样本预测，无法重新对齐共同成功样本，排除共同样本比较。

Fastino 本次六个配置使用 2026-10-29 六数据集并发运行日志；comparison 日志仅按数据集重排原始记录，未改变 elapsed。两个 HTTP 503 样本已补跑成功，质量指标包含补跑结果；主轮 elapsed 不含随后补跑耗时，不能解释为单请求延迟。

fastino.ai/detect-jailbreak_v1_t0.6.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

fastino.ai/detect-jailbreak_v1_t0.7.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

fastino.ai/detect-jailbreak_v1_t0.8.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

fastino.ai/detect-jailbreak_v2_t0.5.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

fastino.ai/detect-jailbreak_v2_t0.6.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

fastino.ai/detect-jailbreak_v2_t0.7.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

fastino.ai/detect-jailbreak_v2_t0.8.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

laya-ai.com/detect-jailbreak_v1_t0.6.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

laya-ai.com/detect-jailbreak_v1_t0.7.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

laya-ai.com/detect-jailbreak_v1_t0.8.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

laya-ai.com/detect-jailbreak_v2_t0.5.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

laya-ai.com/detect-jailbreak_v2_t0.6.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

laya-ai.com/detect-jailbreak_v2_t0.7.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v1_t0.5.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v1_t0.6.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v1_t0.7.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v1_t0.8.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v2_t0.5.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v2_t0.6.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v2_t0.7.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

typesafe.ai/detect-jailbreak_v2_t0.8.csv：历史原始文件与当前参考文本不匹配，保留上一快照汇总，不将该原始文件用于共同样本重算。

laya-ai.com vs typesafe.ai/detect-jailbreak：缺少历史逐样本预测，无法重新对齐共同成功样本，排除共同样本比较。

本次更新 Laya 前六个数据集的 v1 / 0.5：46,923 行，46,913 行成功、10 行失败。质量指标排除错误行；六个参考配置使用本轮完整单次 elapsed/processed，不混入历史其他阈值耗时。PII、v2、其他阈值及其他模型保留历史结果；提示词与重复调用探索混合历史轮次，不能把差异仅归因于提示词、阈值或随机性。
