# 模型能力对比报告

生成时间（UTC）：2026-10-23T04:34:55.578736+00:00

## 结论

固定 v1 / 0.5、共同完成的 4 个非 PII 数据集上，typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 的宏平均 F1 最高（74.15%），宏平均 AUROC 为 92.52%。这是当前快照下的描述性结果，不代表未测试任务或统计显著性。

同一基线比较：typesafe.ai / request model=jev-latest；resolved model=待接口返回确认：F1 74.15%、误报率 12.60%、漏报率 21.08%；laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions：F1 59.28%、误报率 100.00%、漏报率 0.00%；fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2：F1 55.48%、误报率 87.10%、漏报率 14.28%。高误报意味着正常内容被拦截，高漏报意味着危险内容被放过。

分数据集基线 F1 最优：aegis2：typesafe.ai（81.59%）；detect-jailbreak：typesafe.ai（52.81%）；hatecheck：typesafe.ai（96.33%）；toxigen：laya-ai.com（66.36%）。

综合表现领先仍不等于可直接上线：typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 在自身最弱的共同基线数据集 detect-jailbreak 上 F1 为 52.81%、误报率 17.23%、漏报率 12.17%，需结合业务容忍度单独选阈值，而不是照搬统一 0.5。

laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions 在共同基线的有效样本中全部预测为正类（误报率 100%、平衡准确率 50%），因此即使个别数据集 F1 较高，也不代表具备有效区分能力，不建议直接使用该固定配置作为拦截器。

相同基线任务中，typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 的日志摊销平均耗时最低（225.57 ms/样本）。包含网络、客户端开销和失败等待，且各轮非同一时刻运行；不能解释为纯模型推理速度。

laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions 仅完整落盘 31/64 个配置；缺测不记零分，也不把已有任务的均值当作全套能力，当前排名是阶段性结论。

扩展两两比较（fastino.ai vs typesafe.ai，共同完整且有共同有效样本的 5 个非 PII 数据集）：fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 宏 F1 48.11%；typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 宏 F1 67.90%。此口径覆盖范围更广，但不能与三家共同基线的数值混排。

以下配置虽完整落盘但全部请求失败：fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 / jailbreak-classification / v1 / 0.5（1044/1044）；fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 / jailbreak-classification / v1 / 0.6（1044/1044）。这些配置没有质量分数，不参与质量均值；这是接口可靠性问题，不能当作缺测或正常能力结果忽略。

部分基线 AUROC 低于 0.5：fastino.ai / toxigen=23.80%；laya-ai.com / toxigen=27.87%；laya-ai.com / hatecheck=43.11%；fastino.ai / aegis2=46.15%；laya-ai.com / aegis2=34.10%。建议优先核对返回分数语义及接口解析；当前数据不能直接区分模型能力弱与分数映射问题，不自动翻转分数。

PII 数据全是正样本，恒预测“含 PII”即可获得 100% 检出率；现有输出没有脱敏文本，不能评价实体定位、泄露率或脱敏质量，不应据此选定脱敏模型。

选型优先参考固定配置的 F1、AUROC、误报/漏报及错误率；表中的最优提示词/阈值是在同一测试集上挑选的探索性上限，部署前需独立验证集重新选参并复测。

OpenJev 为用户提供的内部汇总结果，仅加入独立参考图表，未提供逐样本输出，不参与上述共同成功样本基线、排名或重复调用一致性分析。

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

共同基线数据集：aegis2、detect-jailbreak、hatecheck、toxigen。

质量指标排除请求错误，但覆盖率与错误数独立展示，避免将失败隐藏成高准确率。全部配置明细使用各文件自身的有效样本；跨模型公平比较请看共同样本表。complete 仅表示所有参考样本均已落盘，不表示请求全部成功。missing 表示未发现输出；in_progress 表示日志有记录但无输出，不保证进程仍在运行。

日志 elapsed / processed 是整轮摊销平均每样本耗时，含失败等待、客户端及网络开销；仅 CSV 与日志完整计数一致的轮次进入耗时比较。基线耗时按处理行数加权；Avg latency 则对共同基线各数据集的整轮平均耗时等权平均，缺少任一完整计时即不提供均值。它使用全部处理样本，质量指标则使用共同成功样本。没有单请求计时，不能计算请求 P50/P95/P99。报告不会从 50 条进度块伪造请求分位数。

PII 仅报告正样本检出率，单类 AUROC、误报率、平衡准确率无定义；没有预测脱敏文本，不计算 masking/span 指标。未输出原始文本、敏感字段或完整错误内容；报告可离线打开，无 CDN、无联网请求。

## 共同基线总览

| 供应商 / 模型 | 完整配置 | 共同数据集 | Average Accuracy | Avg Precision | Avg Recall | 宏 F1 | 宏平衡准确率 | 宏 AUROC | 宏误报率 | 宏漏报率 | Avg latency ms（数据集等权） | 基线平均耗时 ms（样本加权） | 全套请求错误/已落盘行 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 64/64 | 4 | 47.45% | 44.65% | 85.72% | 55.48% | 49.31% | 57.20% | 87.10% | 14.28% | 407.78 | 432.75 | 6430/454584 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 31/64 | 4 | 45.74% | 45.74% | 100.00% | 59.28% | 50.00% | 40.44% | 100.00% | 0.00% | 562.22 | 717.83 | 89/227654 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 64/64 | 4 | 83.25% | 77.72% | 78.92% | 74.15% | 83.16% | 92.52% | 12.60% | 21.08% | 225.96 | 225.57 | 0/454584 |

## 逐数据集固定基线（共同成功样本）

| 数据集 | 供应商 / 模型 | 共同样本 | 覆盖率 | Accuracy | Precision | Recall | F1 | AUROC | 平衡准确率 | 误报率 | 漏报率 | 多数类准确率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 6789 | 100.00% | 39.46% | 43.49% | 73.18% | 54.56% | 23.80% | 39.69% | 93.80% | 26.82% | 50.35% |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 6789 | 100.00% | 49.65% | 49.65% | 100.00% | 66.36% | 27.87% | 50.00% | 100.00% | 0.00% | 50.35% |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 6789 | 100.00% | 73.91% | 94.00% | 50.70% | 65.87% | 87.52% | 73.75% | 3.19% | 49.30% | 50.35% |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 3727 | 99.97% | 69.68% | 70.17% | 97.23% | 81.51% | 78.32% | 53.16% | 90.90% | 2.77% | 68.74% |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 3727 | 99.97% | 68.74% | 68.74% | 100.00% | 81.48% | 43.11% | 50.00% | 100.00% | 0.00% | 68.74% |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 3727 | 99.97% | 94.82% | 93.92% | 98.87% | 96.33% | 99.28% | 92.40% | 14.08% | 1.13% | 68.74% |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 1964 | 100.00% | 50.81% | 52.81% | 82.63% | 64.43% | 46.15% | 48.11% | 86.41% | 17.37% | 53.92% |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 1964 | 100.00% | 53.92% | 53.92% | 100.00% | 70.06% | 34.10% | 50.00% | 100.00% | 0.00% | 53.92% |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 1964 | 100.00% | 80.96% | 85.20% | 78.28% | 81.59% | 90.49% | 81.18% | 15.91% | 21.72% | 53.92% |
| detect-jailbreak | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 18227 | 99.83% | 29.83% | 12.15% | 89.84% | 21.41% | 80.55% | 56.27% | 77.31% | 10.16% | 89.36% |
| detect-jailbreak | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 18227 | 99.83% | 10.64% | 10.64% | 100.00% | 19.23% | 56.68% | 50.00% | 100.00% | 0.00% | 89.36% |
| detect-jailbreak | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 18227 | 99.83% | 83.30% | 37.76% | 87.83% | 52.81% | 92.80% | 85.30% | 17.23% | 12.17% | 89.36% |

## 两两共同基线（利用已完成任务，覆盖范围各不相同）

每一对供应商单独取共同完整数据集与共同成功样本，固定 v1 / 0.5；不同配对覆盖的数据集可能不同，因此只可在同一配对内部比较。共同成功样本为零的数据集不参与质量均值，排除情况如下表明确列出。

| 配对 | 供应商 / 模型 | 共同数据集 | 宏 F1 | 宏 AUROC | 宏平衡准确率 | 宏误报率 | 宏漏报率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai vs laya-ai.com | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2、detect-jailbreak、hatecheck、toxigen | 55.48% | 57.20% | 49.31% | 87.10% | 14.28% |
| fastino.ai vs laya-ai.com | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2、detect-jailbreak、hatecheck、toxigen | 59.28% | 40.44% | 50.00% | 100.00% | 0.00% |
| fastino.ai vs typesafe.ai | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2、detect-jailbreak、hatecheck、jailbreakhub、toxigen | 48.11% | 60.70% | 50.40% | 85.76% | 13.44% |
| fastino.ai vs typesafe.ai | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2、detect-jailbreak、hatecheck、jailbreakhub、toxigen | 67.90% | 91.97% | 82.97% | 14.63% | 19.43% |
| laya-ai.com vs typesafe.ai | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2、detect-jailbreak、hatecheck、toxigen | 59.28% | 40.44% | 50.00% | 100.00% | 0.00% |
| laya-ai.com vs typesafe.ai | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | aegis2、detect-jailbreak、hatecheck、toxigen | 74.15% | 92.52% | 83.16% | 12.60% | 21.08% |

| 配对 | 数据集 | 共同有效样本 | 共同覆盖率 | 质量评估状态 |
| --- | --- | --- | --- | --- |
| fastino.ai vs laya-ai.com | toxigen | 6789 | 100.00% | 参与均值 |
| fastino.ai vs typesafe.ai | toxigen | 6789 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | toxigen | 6789 | 100.00% | 参与均值 |
| fastino.ai vs laya-ai.com | hatecheck | 3727 | 99.97% | 参与均值 |
| fastino.ai vs typesafe.ai | hatecheck | 3728 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | hatecheck | 3727 | 99.97% | 参与均值 |
| fastino.ai vs laya-ai.com | aegis2 | 1964 | 100.00% | 参与均值 |
| fastino.ai vs typesafe.ai | aegis2 | 1964 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | aegis2 | 1964 | 100.00% | 参与均值 |
| fastino.ai vs laya-ai.com | detect-jailbreak | 18227 | 99.83% | 参与均值 |
| fastino.ai vs typesafe.ai | detect-jailbreak | 18258 | 100.00% | 参与均值 |
| laya-ai.com vs typesafe.ai | detect-jailbreak | 18227 | 99.83% | 参与均值 |
| fastino.ai vs typesafe.ai | jailbreak-classification | 0 | 0.00% | 无共同成功样本，排除质量均值 |
| fastino.ai vs typesafe.ai | jailbreakhub | 15140 | 100.00% | 参与均值 |

## 提示词与阈值探索（同测试集选优，非公平总排名）

以下按各文件自身成功样本选 F1 最优；不同轮次可能包含不同请求错误，不能将该结果当作独立验证后的提升。完整 v1/v2 × 4 阈值结果见明细。

| 数据集 | 供应商 / 模型 | 最优提示词 / 阈值 | 最优 F1 | 自身基线 F1 | 误报率 | 漏报率 | 可用配置 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.5 | 55.89% | 54.56% | 91.93% | 25.07% | 8/8 |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.8 | 82.13% | 81.52% | 76.14% | 6.20% | 8/8 |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | 64.43% | 64.43% | 86.41% | 17.37% | 8/8 |
| detect-jailbreak | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.8 | 31.24% | 21.64% | 54.24% | 13.05% | 8/8 |
| jailbreak-classification | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.5 | 65.21% | — | 95.55% | 6.26% | 6/8 |
| jailbreakhub | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.8 | 21.14% | 18.41% | 59.75% | 19.15% | 8/8 |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | 66.36% | 66.36% | 100.00% | 0.00% | 8/8 |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.5 | 81.48% | 81.48% | 100.00% | 0.00% | 8/8 |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | 70.06% | 70.06% | 100.00% | 0.00% | 8/8 |
| detect-jailbreak | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.7 | 23.47% | 19.23% | 66.48% | 13.39% | 7/8 |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.5 | 69.28% | 65.87% | 3.95% | 44.88% | 8/8 |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.7 | 97.43% | 96.33% | 4.46% | 3.08% | 8/8 |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.5 | 81.59% | 81.59% | 15.91% | 21.72% | 8/8 |
| detect-jailbreak | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.8 | 64.23% | 53.13% | 8.92% | 17.71% | 8/8 |
| jailbreak-classification | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.5 | 95.16% | 95.06% | 0.77% | 8.54% | 8/8 |
| jailbreakhub | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.8 | 51.76% | 42.57% | 13.11% | 20.36% | 8/8 |

## 重复调用一致性：口径与逐数据集结果

将同一供应商、数据集、提示词的 0.5/0.6/0.7/0.8 四轮视作四次重复调用。阈值只在客户端用于判定，不进入模型请求；不直接比较各 CSV 原有 prediction。v1、v2 分开统计，按 sample_id 对齐，四轮均完整落盘后仅使用四次均成功的样本。少一轮或存在部分落盘即标 incomplete_repeats，不以较少轮次替代；四轮完整但无共同成功样本标 no_common_success，稳定性无定义而不是零波动。

每个样本计算四个 score 的总体标准差（ddof=0）与极差，再对样本取平均；分数保持原始 0–1 单位。判定统一使用 score >= 0.5，翻转率为四次判定中至少出现一次正类、一次负类的样本比例，不是相邻轮次翻转次数。三项指标均越低表示越一致。覆盖率为四次均成功样本 / 参考样本；请求失败率为已有 CSV 的失败行 / 已落盘总行，缺失轮次不算成功也不算请求失败。

逐数据集表使用各模型自身的四轮成功交集，样本可能不同，不作为公平总排名。历史调用可能混有后端版本、路由或运行时段变化，无法将差异完全归因于模型随机性；仅四次调用是探索性观察，不提供伪造置信区间。稳定不代表正确，恒定错误输出也可能零波动。原阈值敏感性曲线同时混有重复调用波动，不是固定 score 下的纯阈值效应。

| 数据集 | 供应商 / 模型 | 提示词 | 状态 | 已有轮次 | 完整轮次 | 四次均成功/参考 | 覆盖率 | 平均分数标准差 | 平均分数极差 | 判定翻转样本 | 判定翻转率（统一 0.5） | 失败/已落盘请求 | 请求失败率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/27156 | 0.00% |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 6781/6789 | 99.88% | 0.000000 | 0.000000 | 0 | 0.00% | 8/27156 | 0.03% |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.005146 | 0.012672 | 79 | 1.16% | 0/27156 | 0.00% |
| toxigen | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/27156 | 0.00% |
| toxigen | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 6780/6789 | 99.87% | 0.000000 | 0.000000 | 0 | 0.00% | 9/27156 | 0.03% |
| toxigen | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 6789/6789 | 100.00% | 0.004798 | 0.011797 | 88 | 1.30% | 0/27156 | 0.00% |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/14912 | 0.00% |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 3727/3728 | 99.97% | 0.000000 | 0.000000 | 0 | 0.00% | 1/14912 | 0.01% |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.005095 | 0.012538 | 50 | 1.34% | 0/14912 | 0.00% |
| hatecheck | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/14912 | 0.00% |
| hatecheck | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 3726/3728 | 99.95% | 0.000000 | 0.000000 | 0 | 0.00% | 2/14912 | 0.01% |
| hatecheck | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 3728/3728 | 100.00% | 0.004581 | 0.011274 | 35 | 0.94% | 0/14912 | 0.00% |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/7856 | 0.00% |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/7856 | 0.00% |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.006415 | 0.015850 | 31 | 1.58% | 0/7856 | 0.00% |
| aegis2 | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 1961/1964 | 99.85% | 0.000000 | 0.000000 | 0 | 0.00% | 3/7856 | 0.04% |
| aegis2 | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/7856 | 0.00% |
| aegis2 | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 1964/1964 | 100.00% | 0.005369 | 0.013299 | 34 | 1.73% | 0/7856 | 0.00% |
| detect-jailbreak | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 18256/18258 | 99.99% | 0.000000 | 0.000000 | 0 | 0.00% | 2/73032 | 0.00% |
| detect-jailbreak | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | complete | 4/4 | 4/4 | 18215/18258 | 99.76% | 0.000000 | 0.000000 | 0 | 0.00% | 55/73032 | 0.08% |
| detect-jailbreak | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 18258/18258 | 100.00% | 0.005082 | 0.012416 | 155 | 0.85% | 0/73032 | 0.00% |
| detect-jailbreak | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 14634/18258 | 80.15% | 0.018741 | 0.042424 | 624 | 4.26% | 3624/73032 | 4.96% |
| detect-jailbreak | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 3/4 | 3/4 | 0/18258 | — | — | — | — | — | 14/54774 | 0.03% |
| detect-jailbreak | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 18258/18258 | 100.00% | 0.004493 | 0.010897 | 144 | 0.79% | 0/73032 | 0.00% |
| jailbreak-classification | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | no_common_success | 4/4 | 4/4 | 0/1044 | 0.00% | — | — | — | — | 2798/4176 | 67.00% |
| jailbreak-classification | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | incomplete_repeats | 0/4 | 0/4 | 0/1044 | — | — | — | — | — | 0/0 | — |
| jailbreak-classification | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 1044/1044 | 100.00% | 0.002424 | 0.005766 | 4 | 0.38% | 0/4176 | 0.00% |
| jailbreak-classification | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 1044/1044 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/4176 | 0.00% |
| jailbreak-classification | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 0/4 | 0/4 | 0/1044 | — | — | — | — | — | 0/0 | — |
| jailbreak-classification | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 1044/1044 | 100.00% | 0.002192 | 0.005239 | 2 | 0.19% | 0/4176 | 0.00% |
| jailbreakhub | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 15140/15140 | 100.00% | 0.000931 | 0.002151 | 35 | 0.23% | 0/60560 | 0.00% |
| jailbreakhub | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | incomplete_repeats | 0/4 | 0/4 | 0/15140 | — | — | — | — | — | 0/0 | — |
| jailbreakhub | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 15140/15140 | 100.00% | 0.005015 | 0.012223 | 142 | 0.94% | 0/60560 | 0.00% |
| jailbreakhub | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 15139/15140 | 99.99% | 0.000112 | 0.000260 | 2 | 0.01% | 1/60560 | 0.00% |
| jailbreakhub | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 0/4 | 0/4 | 0/15140 | — | — | — | — | — | 0/0 | — |
| jailbreakhub | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 15140/15140 | 100.00% | 0.004183 | 0.010139 | 109 | 0.72% | 0/60560 | 0.00% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/3600 | 0.00% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | incomplete_repeats | 0/4 | 0/4 | 0/900 | — | — | — | — | — | 0/0 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.004437 | 0.010789 | 8 | 0.89% | 0/3600 | 0.00% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.000000 | 0.000000 | 0 | 0.00% | 0/3600 | 0.00% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 0/4 | 0/4 | 0/900 | — | — | — | — | — | 0/0 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 900/900 | 100.00% | 0.002678 | 0.006433 | 1 | 0.11% | 0/3600 | 0.00% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | complete | 4/4 | 4/4 | 8998/9000 | 99.98% | 0.003892 | 0.008988 | 53 | 0.59% | 2/36000 | 0.01% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | incomplete_repeats | 0/4 | 0/4 | 0/9000 | — | — | — | — | — | 0/0 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | complete | 4/4 | 4/4 | 9000/9000 | 100.00% | 0.004347 | 0.010594 | 57 | 0.63% | 0/36000 | 0.00% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | complete | 4/4 | 4/4 | 9000/9000 | 100.00% | 0.000611 | 0.001415 | 5 | 0.06% | 0/36000 | 0.00% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | incomplete_repeats | 0/4 | 0/4 | 0/9000 | — | — | — | — | — | 0/0 | — |
| pii_masking_mini_10k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | complete | 4/4 | 4/4 | 9000/9000 | 100.00% | 0.002616 | 0.006252 | 26 | 0.29% | 0/36000 | 0.00% |

## 重复调用一致性：共同样本模型均值

每个提示词单独选取所有供应商四轮均完整的数据集，在每个数据集内进一步取所有供应商全部四轮的成功样本交集。只对有共同成功样本的非 PII 数据集等权平均，不与上方各自样本均值混排。不同提示词的共同数据集或样本可能不同；PII 仅在逐数据集一致性中展示。覆盖率按共同样本总数 / 参考样本总数计算；请求失败率使用这些数据集全部已落盘请求。

| 供应商 / 模型 | 提示词 | 共同数据集 | 共同样本/参考 | 共同覆盖率 | 宏平均分数标准差 | 宏平均分数极差 | 宏平均判定翻转率 | 请求失败率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 | aegis2、detect-jailbreak、hatecheck、toxigen | 30685/30739 | 99.82% | 0.000000 | 0.000000 | 0.00% | 0.00% |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 | aegis2、detect-jailbreak、hatecheck、toxigen | 30685/30739 | 99.82% | 0.000000 | 0.000000 | 0.00% | 0.05% |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 | aegis2、detect-jailbreak、hatecheck、toxigen | 30685/30739 | 99.82% | 0.005435 | 0.013370 | 1.23% | 0.00% |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 | aegis2、hatecheck、toxigen | 12467/12481 | 99.89% | 0.000000 | 0.000000 | 0.00% | 0.01% |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 | aegis2、hatecheck、toxigen | 12467/12481 | 99.89% | 0.000000 | 0.000000 | 0.00% | 0.02% |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 | aegis2、hatecheck、toxigen | 12467/12481 | 99.89% | 0.004918 | 0.012130 | 1.32% | 0.00% |

## PII 检出（不是脱敏质量）

| 数据集 | 供应商 / 模型 | 提示词 / 阈值 | 状态 | 有效/预期 | 正样本检出率 |
| --- | --- | --- | --- | --- | --- |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | complete | 900/900 | 86.11% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.5 | complete | 900/900 | 94.33% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.6 | complete | 900/900 | 82.33% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.6 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.6 | complete | 900/900 | 93.00% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.7 | complete | 900/900 | 77.78% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.7 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.7 | complete | 900/900 | 91.78% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.8 | complete | 900/900 | 70.11% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.8 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v1 / 0.8 | complete | 900/900 | 89.11% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.5 | complete | 900/900 | 92.56% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.5 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.5 | complete | 900/900 | 95.67% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.6 | complete | 900/900 | 90.11% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.6 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.6 | complete | 900/900 | 95.22% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.7 | complete | 900/900 | 87.78% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.7 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.7 | complete | 900/900 | 93.67% |
| pii_masking_nano_1k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v2 / 0.8 | complete | 900/900 | 83.67% |
| pii_masking_nano_1k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v2 / 0.8 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | v2 / 0.8 | complete | 900/900 | 92.33% |
| pii_masking_mini_10k | fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | v1 / 0.5 | complete | 9000/9000 | 85.96% |
| pii_masking_mini_10k | laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | v1 / 0.5 | missing | 0/9000 | — |
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
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 3376.80 | 497.39 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2870.91 | 422.88 |
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
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | hatecheck | v1 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1213.50 | 325.51 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | hatecheck | v1 / 0.5 | complete | 3728/3728 | 99.97% | 0.03% | 3728/3728 | 1664.83 | 446.57 |
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
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 728.60 | 370.98 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 922.82 | 469.87 |
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
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 7983.40 | 437.25 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 99.83% | 0.17% | 18258/18258 | 16606.74 | 909.56 |
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
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | detect-jailbreak | v2 / 0.5 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 15068.20 | 825.29 |
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
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.5 | complete | 1044/1044 | 0.00% | 100.00% | 1044/1044 | 254.30 | 243.58 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.5 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 239.84 | 229.73 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.6 | complete | 1044/1044 | 0.00% | 100.00% | 1044/1044 | 246.40 | 236.02 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.6 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 249.70 | 239.18 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.7 | complete | 1044/1044 | 31.99% | 68.01% | 1044/1044 | 299.80 | 287.16 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.7 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 241.61 | 231.43 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v1 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 314.90 | 301.63 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v1 / 0.8 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v1 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 239.06 | 228.99 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 365.50 | 350.10 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.5 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 241.01 | 230.85 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 299.70 | 287.07 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.6 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 249.60 | 239.08 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 349.60 | 334.87 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.7 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 246.67 | 236.27 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreak-classification | v2 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 342.10 | 327.68 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreak-classification | v2 / 0.8 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreak-classification | v2 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 240.36 | 230.23 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 4721.70 | 311.87 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.5 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3529.11 | 233.10 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 4845.10 | 320.02 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.6 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3548.75 | 234.40 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 5602.20 | 370.03 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.7 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3520.66 | 232.54 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v1 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 8423.20 | 556.35 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v1 / 0.8 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v1 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3505.52 | 231.54 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 7124.20 | 470.55 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.5 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3475.07 | 229.53 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 5187.80 | 342.66 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.6 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3447.00 | 227.68 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 6065.60 | 400.63 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.7 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3496.82 | 230.97 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | jailbreakhub | v2 / 0.8 | complete | 15140/15140 | 99.99% | 0.01% | 15140/15140 | 6052.10 | 399.74 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | jailbreakhub | v2 / 0.8 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | jailbreakhub | v2 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3400.87 | 224.63 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 376.80 | 418.67 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.5 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.96 | 219.95 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 318.20 | 353.56 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.6 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 200.23 | 222.48 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 301.40 | 334.89 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.7 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 201.80 | 224.23 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v1 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 300.20 | 333.56 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v1 / 0.8 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v1 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.96 | 219.95 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 370.00 | 411.11 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.5 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.29 | 219.21 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 311.30 | 345.89 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.6 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.06 | 218.95 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 334.90 | 372.11 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.7 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 203.91 | 226.56 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_nano_1k | v2 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 346.20 | 384.67 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_nano_1k | v2 / 0.8 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | pii_masking_nano_1k | v2 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 200.88 | 223.20 |
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | pii_masking_mini_10k | v1 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 3028.20 | 336.47 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | pii_masking_mini_10k | v1 / 0.5 | missing | 0/9000 | 0.00% | — | — | — | — |
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
| fastino.ai / request model=fastino/GLiDE；resolved model=glide_v2 | 454584 | 6430 | 1.41% | ConnectError: 1；HTTPStatusError: 6423；ReadTimeout: 5；RemoteProtocolError: 1 |
| laya-ai.com / request model=typed-decisions；serving model=laya-rl-agent；routed model=typed-decisions | 227654 | 89 | 0.04% | ConnectError: 13；HTTPStatusError: 72；ReadTimeout: 1；RemoteProtocolError: 3 |
| typesafe.ai / request model=jev-latest；resolved model=待接口返回确认 | 454584 | 0 | 0.00% | 无 |

## OpenJev 内部汇总结果（非共同样本基线）

OpenJev 数值取自用户最初提供的清晰汇总表，分类阈值均为 0.5；补充 OCR 文本有错字，不据此修正数值或推断混淆矩阵。Rows 为总测试行数，有效样本数和错误数未可靠录入，保持未知，不将 Rows 当作有效样本数。OpenJev 只提供汇总指标，未提供逐样本 score、预测或模型版本；不能核验相同输入、共同成功样本或计算重复调用稳定性，不加入原有公平基线与排名。参考图按各模型自身成功样本统计，仅作描述性对照。耗时统一为每个数据集/提示词四轮完整运行的 elapsed/processed 算术平均，包含失败请求、等待、客户端及网络开销，不是纯推理时间；无完整四轮计时标 N/A。

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

## 四模型参考均值（同数据集，非同成功样本）

固定分类阈值 0.5，v1/v2 分开统计，仅选择所有模型均有可用质量结果的非 PII 数据集，对数据集等权平均。各模型使用自身有效样本，未对齐共同 sample_id，不能当作公平排名。耗时为四轮完整日志的每轮 elapsed/processed 等权平均，再按数据集等权平均；任一所选数据集缺少完整四轮计时即不提供模型平均耗时，不用部分计时替代。

| 模型 | 提示词 | 共同数据集 | Average Accuracy | Avg Precision | Avg Recall | Avg F1 | Avg AUROC | Avg latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai | v1 | toxigen、hatecheck、aegis2、detect-jailbreak | 47.47% | 44.69% | 85.74% | 55.54% | 57.22% | 400.58 |
| laya-ai.com | v1 | toxigen、hatecheck、aegis2、detect-jailbreak | 45.74% | 45.74% | 100.00% | 59.28% | 40.44% | 556.65 |
| typesafe.ai | v1 | toxigen、hatecheck、aegis2、detect-jailbreak | 83.25% | 77.80% | 78.93% | 74.23% | 92.54% | 225.77 |
| openjev | v1 | toxigen、hatecheck、aegis2、detect-jailbreak | 80.78% | 76.63% | 74.03% | 72.98% | 87.43% | 904.93 |
| fastino.ai | v2 | toxigen、hatecheck、aegis2、detect-jailbreak | 46.89% | 44.53% | 86.59% | 55.59% | 59.33% | 463.15 |
| laya-ai.com | v2 | toxigen、hatecheck、aegis2、detect-jailbreak | 45.76% | 45.76% | 100.00% | 59.33% | 38.87% | — |
| typesafe.ai | v2 | toxigen、hatecheck、aegis2、detect-jailbreak | 82.94% | 77.70% | 78.65% | 73.86% | 92.35% | 226.44 |
| openjev | v2 | toxigen、hatecheck、aegis2、detect-jailbreak | 80.94% | 78.71% | 70.78% | 72.09% | 87.83% | 924.20 |
