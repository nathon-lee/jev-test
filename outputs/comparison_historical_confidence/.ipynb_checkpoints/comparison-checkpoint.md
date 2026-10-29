# 模型能力对比报告

生成时间（UTC）：2026-10-21T23:39:07.307845+00:00

## 结论

固定 v1 / 0.5、共同完成的 4 个非 PII 数据集上，typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） 的宏平均 F1 最高（74.15%），宏平均 AUROC 为 92.52%。这是当前快照下的描述性结果，不代表未测试任务或统计显著性。

同一基线比较：typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录）：F1 74.15%、误报率 12.60%、漏报率 21.08%；laya-ai.com / typed-decisions：F1 59.28%、误报率 100.00%、漏报率 0.00%；fastino.ai / fastino/GLiDE：F1 55.48%、误报率 87.10%、漏报率 14.28%。高误报意味着正常内容被拦截，高漏报意味着危险内容被放过。

分数据集基线 F1 最优：aegis2：typesafe.ai（81.59%）；detect-jailbreak：typesafe.ai（52.81%）；hatecheck：typesafe.ai（96.33%）；toxigen：laya-ai.com（66.36%）。

综合表现领先仍不等于可直接上线：typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） 在自身最弱的共同基线数据集 detect-jailbreak 上 F1 为 52.81%、误报率 17.23%、漏报率 12.17%，需结合业务容忍度单独选阈值，而不是照搬统一 0.5。

laya-ai.com / typed-decisions 在共同基线的有效样本中全部预测为正类（误报率 100%、平衡准确率 50%），因此即使个别数据集 F1 较高，也不代表具备有效区分能力，不建议直接使用该固定配置作为拦截器。

相同基线任务中，typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） 的日志摊销平均耗时最低（225.57 ms/样本）。包含网络、客户端开销和失败等待，且各轮非同一时刻运行；不能解释为纯模型推理速度。

laya-ai.com / typed-decisions 仅完整落盘 31/64 个配置；缺测不记零分，也不把已有任务的均值当作全套能力，当前排名是阶段性结论。

扩展两两比较（fastino.ai vs typesafe.ai，共同完整且有共同有效样本的 5 个非 PII 数据集）：fastino.ai / fastino/GLiDE 宏 F1 48.11%；typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） 宏 F1 67.90%。此口径覆盖范围更广，但不能与三家共同基线的数值混排。

以下配置虽完整落盘但全部请求失败：fastino.ai / fastino/GLiDE / jailbreak-classification / v1 / 0.5（1044/1044）；fastino.ai / fastino/GLiDE / jailbreak-classification / v1 / 0.6（1044/1044）。这些配置没有质量分数，不参与质量均值；这是接口可靠性问题，不能当作缺测或正常能力结果忽略。

部分基线 AUROC 低于 0.5：fastino.ai / toxigen=23.80%；laya-ai.com / toxigen=27.87%；laya-ai.com / hatecheck=43.11%；fastino.ai / aegis2=46.15%；laya-ai.com / aegis2=34.10%。建议优先核对返回分数语义及接口解析；当前数据不能直接区分模型能力弱与分数映射问题，不自动翻转分数。

PII 数据全是正样本，恒预测“含 PII”即可获得 100% 检出率；现有输出没有脱敏文本，不能评价实体定位、泄露率或脱敏质量，不应据此选定脱敏模型。

选型优先参考固定配置的 F1、AUROC、误报/漏报及错误率；表中的最优提示词/阈值是在同一测试集上挑选的探索性上限，部署前需独立验证集重新选参并复测。

## 模型身份与证据

| 供应商 | 模型名称 / 调用方式 | 证据及限制 |
| --- | --- | --- |
| fastino.ai | fastino/GLiDE | provider_suite.py 的请求显式传入该模型；输出未记录后端修订版本 |
| laya-ai.com | typed-decisions | provider_suite.py 的请求显式传入该模型；输出未记录后端修订版本 |
| typesafe.ai | typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | 配置含此别名，但 SDK 调用不传 model；无法确认具体部署模型 |

## 比较口径

固定基线：v1 / threshold=0.5。只使用所有所选供应商均完整落盘的数据集，按 sample_id 对齐共同成功样本；读取时统一 CRLF/LF 换行后核对参考文本哈希、标签、唯一 ID、分数与预测一致性。宏平均对数据集等权，避免大数据集主导；不混入 PII，不把 8 次提示词/阈值运行当作独立样本。

参考输入按原 runner 的 pandas.read_csv 语义读取（包括空值转字符串和 NUL 字符截断）；这反映当时实际发给模型的内容，不代表原始数据字节完整性。例如 detect-jailbreak 原始文本中存在 NUL，原 runner 已发生截断，各供应商对比沿用相同输入。

共同基线数据集：aegis2、detect-jailbreak、hatecheck、toxigen。

质量指标排除请求错误，但覆盖率与错误数独立展示，避免将失败隐藏成高准确率。全部配置明细使用各文件自身的有效样本；跨模型公平比较请看共同样本表。complete 仅表示所有参考样本均已落盘，不表示请求全部成功。missing 表示未发现输出；in_progress 表示日志有记录但无输出，不保证进程仍在运行。

日志 elapsed / processed 是整轮摊销平均每样本耗时，含失败等待、客户端及网络开销；仅 CSV 与日志完整计数一致的轮次进入耗时比较。基线耗时按处理行数加权；它使用全部处理样本，质量指标则使用共同成功样本。没有单请求计时，不能计算请求 P50/P95/P99。报告不会从 50 条进度块伪造请求分位数。

PII 仅报告正样本检出率，单类 AUROC、误报率、平衡准确率无定义；没有预测脱敏文本，不计算 masking/span 指标。未输出原始文本、敏感字段或完整错误内容；报告可离线打开，无 CDN、无联网请求。

## 共同基线总览

| 供应商 / 模型 | 完整配置 | 共同数据集 | 宏 F1 | 宏平衡准确率 | 宏 AUROC | 宏误报率 | 宏漏报率 | 基线平均耗时 ms | 全套请求错误/已落盘行 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai / fastino/GLiDE | 64/64 | 4 | 55.48% | 49.31% | 57.20% | 87.10% | 14.28% | 432.75 | 6430/454584 |
| laya-ai.com / typed-decisions | 31/64 | 4 | 59.28% | 50.00% | 40.44% | 100.00% | 0.00% | 717.83 | 89/227654 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | 64/64 | 4 | 74.15% | 83.16% | 92.52% | 12.60% | 21.08% | 225.57 | 0/454584 |

## 逐数据集固定基线（共同成功样本）

| 数据集 | 供应商 / 模型 | 共同样本 | 覆盖率 | F1 | AUROC | 平衡准确率 | 误报率 | 漏报率 | 多数类准确率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxigen | fastino.ai / fastino/GLiDE | 6789 | 100.00% | 54.56% | 23.80% | 39.69% | 93.80% | 26.82% | 50.35% |
| toxigen | laya-ai.com / typed-decisions | 6789 | 100.00% | 66.36% | 27.87% | 50.00% | 100.00% | 0.00% | 50.35% |
| toxigen | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | 6789 | 100.00% | 65.87% | 87.52% | 73.75% | 3.19% | 49.30% | 50.35% |
| hatecheck | fastino.ai / fastino/GLiDE | 3727 | 99.97% | 81.51% | 78.32% | 53.16% | 90.90% | 2.77% | 68.74% |
| hatecheck | laya-ai.com / typed-decisions | 3727 | 99.97% | 81.48% | 43.11% | 50.00% | 100.00% | 0.00% | 68.74% |
| hatecheck | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | 3727 | 99.97% | 96.33% | 99.28% | 92.40% | 14.08% | 1.13% | 68.74% |
| aegis2 | fastino.ai / fastino/GLiDE | 1964 | 100.00% | 64.43% | 46.15% | 48.11% | 86.41% | 17.37% | 53.92% |
| aegis2 | laya-ai.com / typed-decisions | 1964 | 100.00% | 70.06% | 34.10% | 50.00% | 100.00% | 0.00% | 53.92% |
| aegis2 | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | 1964 | 100.00% | 81.59% | 90.49% | 81.18% | 15.91% | 21.72% | 53.92% |
| detect-jailbreak | fastino.ai / fastino/GLiDE | 18227 | 99.83% | 21.41% | 80.55% | 56.27% | 77.31% | 10.16% | 89.36% |
| detect-jailbreak | laya-ai.com / typed-decisions | 18227 | 99.83% | 19.23% | 56.68% | 50.00% | 100.00% | 0.00% | 89.36% |
| detect-jailbreak | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | 18227 | 99.83% | 52.81% | 92.80% | 85.30% | 17.23% | 12.17% | 89.36% |

## 两两共同基线（利用已完成任务，覆盖范围各不相同）

每一对供应商单独取共同完整数据集与共同成功样本，固定 v1 / 0.5；不同配对覆盖的数据集可能不同，因此只可在同一配对内部比较。共同成功样本为零的数据集不参与质量均值，排除情况如下表明确列出。

| 配对 | 供应商 / 模型 | 共同数据集 | 宏 F1 | 宏 AUROC | 宏平衡准确率 | 宏误报率 | 宏漏报率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai vs laya-ai.com | fastino.ai / fastino/GLiDE | aegis2、detect-jailbreak、hatecheck、toxigen | 55.48% | 57.20% | 49.31% | 87.10% | 14.28% |
| fastino.ai vs laya-ai.com | laya-ai.com / typed-decisions | aegis2、detect-jailbreak、hatecheck、toxigen | 59.28% | 40.44% | 50.00% | 100.00% | 0.00% |
| fastino.ai vs typesafe.ai | fastino.ai / fastino/GLiDE | aegis2、detect-jailbreak、hatecheck、jailbreakhub、toxigen | 48.11% | 60.70% | 50.40% | 85.76% | 13.44% |
| fastino.ai vs typesafe.ai | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2、detect-jailbreak、hatecheck、jailbreakhub、toxigen | 67.90% | 91.97% | 82.97% | 14.63% | 19.43% |
| laya-ai.com vs typesafe.ai | laya-ai.com / typed-decisions | aegis2、detect-jailbreak、hatecheck、toxigen | 59.28% | 40.44% | 50.00% | 100.00% | 0.00% |
| laya-ai.com vs typesafe.ai | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2、detect-jailbreak、hatecheck、toxigen | 74.15% | 92.52% | 83.16% | 12.60% | 21.08% |

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
| toxigen | fastino.ai / fastino/GLiDE | v2 / 0.5 | 55.89% | 54.56% | 91.93% | 25.07% | 8/8 |
| hatecheck | fastino.ai / fastino/GLiDE | v2 / 0.8 | 82.13% | 81.52% | 76.14% | 6.20% | 8/8 |
| aegis2 | fastino.ai / fastino/GLiDE | v1 / 0.5 | 64.43% | 64.43% | 86.41% | 17.37% | 8/8 |
| detect-jailbreak | fastino.ai / fastino/GLiDE | v2 / 0.8 | 31.24% | 21.64% | 54.24% | 13.05% | 8/8 |
| jailbreak-classification | fastino.ai / fastino/GLiDE | v2 / 0.5 | 65.21% | — | 95.55% | 6.26% | 6/8 |
| jailbreakhub | fastino.ai / fastino/GLiDE | v1 / 0.8 | 21.14% | 18.41% | 59.75% | 19.15% | 8/8 |
| toxigen | laya-ai.com / typed-decisions | v1 / 0.5 | 66.36% | 66.36% | 100.00% | 0.00% | 8/8 |
| hatecheck | laya-ai.com / typed-decisions | v2 / 0.5 | 81.48% | 81.48% | 100.00% | 0.00% | 8/8 |
| aegis2 | laya-ai.com / typed-decisions | v1 / 0.5 | 70.06% | 70.06% | 100.00% | 0.00% | 8/8 |
| detect-jailbreak | laya-ai.com / typed-decisions | v2 / 0.7 | 23.47% | 19.23% | 66.48% | 13.39% | 7/8 |
| toxigen | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.5 | 69.28% | 65.87% | 3.95% | 44.88% | 8/8 |
| hatecheck | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.7 | 97.43% | 96.33% | 4.46% | 3.08% | 8/8 |
| aegis2 | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.5 | 81.59% | 81.59% | 15.91% | 21.72% | 8/8 |
| detect-jailbreak | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.8 | 64.23% | 53.13% | 8.92% | 17.71% | 8/8 |
| jailbreak-classification | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.5 | 95.16% | 95.06% | 0.77% | 8.54% | 8/8 |
| jailbreakhub | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.8 | 51.76% | 42.57% | 13.11% | 20.36% | 8/8 |

## PII 检出（不是脱敏质量）

| 数据集 | 供应商 / 模型 | 提示词 / 阈值 | 状态 | 有效/预期 | 正样本检出率 |
| --- | --- | --- | --- | --- | --- |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v1 / 0.5 | complete | 900/900 | 86.11% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v1 / 0.5 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.5 | complete | 900/900 | 94.33% |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v1 / 0.6 | complete | 900/900 | 82.33% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v1 / 0.6 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.6 | complete | 900/900 | 93.00% |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v1 / 0.7 | complete | 900/900 | 77.78% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v1 / 0.7 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.7 | complete | 900/900 | 91.78% |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v1 / 0.8 | complete | 900/900 | 70.11% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v1 / 0.8 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.8 | complete | 900/900 | 89.11% |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v2 / 0.5 | complete | 900/900 | 92.56% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v2 / 0.5 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.5 | complete | 900/900 | 95.67% |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v2 / 0.6 | complete | 900/900 | 90.11% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v2 / 0.6 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.6 | complete | 900/900 | 95.22% |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v2 / 0.7 | complete | 900/900 | 87.78% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v2 / 0.7 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.7 | complete | 900/900 | 93.67% |
| pii_masking_nano_1k | fastino.ai / fastino/GLiDE | v2 / 0.8 | complete | 900/900 | 83.67% |
| pii_masking_nano_1k | laya-ai.com / typed-decisions | v2 / 0.8 | missing | 0/900 | — |
| pii_masking_nano_1k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.8 | complete | 900/900 | 92.33% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v1 / 0.5 | complete | 9000/9000 | 85.96% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v1 / 0.5 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.5 | complete | 9000/9000 | 94.94% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v1 / 0.6 | complete | 9000/9000 | 82.18% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v1 / 0.6 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.6 | complete | 9000/9000 | 93.70% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v1 / 0.7 | complete | 8999/9000 | 77.52% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v1 / 0.7 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.7 | complete | 9000/9000 | 92.17% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v1 / 0.8 | complete | 8999/9000 | 68.99% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v1 / 0.8 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v1 / 0.8 | complete | 9000/9000 | 89.30% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v2 / 0.5 | complete | 9000/9000 | 92.91% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v2 / 0.5 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.5 | complete | 9000/9000 | 95.97% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v2 / 0.6 | complete | 9000/9000 | 90.51% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v2 / 0.6 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.6 | complete | 9000/9000 | 95.38% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v2 / 0.7 | complete | 9000/9000 | 87.76% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v2 / 0.7 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.7 | complete | 9000/9000 | 94.54% |
| pii_masking_mini_10k | fastino.ai / fastino/GLiDE | v2 / 0.8 | complete | 9000/9000 | 83.04% |
| pii_masking_mini_10k | laya-ai.com / typed-decisions | v2 / 0.8 | missing | 0/9000 | — |
| pii_masking_mini_10k | typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | v2 / 0.8 | complete | 9000/9000 | 93.12% |

## 配置覆盖与日志耗时

| 供应商 / 模型 | 数据集 | 提示词 / 阈值 | 状态 | 落盘/预期 | 有效覆盖率 | 错误率 | 日志进度 | 日志累计 s | 整轮平均 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastino.ai / fastino/GLiDE | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 3376.80 | 497.39 |
| laya-ai.com / typed-decisions | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2870.91 | 422.88 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v1 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1552.10 | 228.62 |
| fastino.ai / fastino/GLiDE | toxigen | v1 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 3133.50 | 461.56 |
| laya-ai.com / typed-decisions | toxigen | v1 / 0.6 | complete | 6789/6789 | 99.99% | 0.01% | 6789/6789 | 2970.18 | 437.50 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v1 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1560.43 | 229.85 |
| fastino.ai / fastino/GLiDE | toxigen | v1 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2662.40 | 392.16 |
| laya-ai.com / typed-decisions | toxigen | v1 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 3039.83 | 447.76 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v1 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1564.38 | 230.43 |
| fastino.ai / fastino/GLiDE | toxigen | v1 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2503.40 | 368.74 |
| laya-ai.com / typed-decisions | toxigen | v1 / 0.8 | complete | 6789/6789 | 99.90% | 0.10% | 6789/6789 | 3331.06 | 490.66 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v1 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1536.03 | 226.25 |
| fastino.ai / fastino/GLiDE | toxigen | v2 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2312.30 | 340.60 |
| laya-ai.com / typed-decisions | toxigen | v2 / 0.5 | complete | 6789/6789 | 99.96% | 0.04% | 6789/6789 | 3996.74 | 588.71 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v2 / 0.5 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1557.01 | 229.34 |
| fastino.ai / fastino/GLiDE | toxigen | v2 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2261.50 | 333.11 |
| laya-ai.com / typed-decisions | toxigen | v2 / 0.6 | complete | 6789/6789 | 99.99% | 0.01% | 6789/6789 | 3799.09 | 559.60 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v2 / 0.6 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1567.75 | 230.93 |
| fastino.ai / fastino/GLiDE | toxigen | v2 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2205.70 | 324.89 |
| laya-ai.com / typed-decisions | toxigen | v2 / 0.7 | complete | 6789/6789 | 99.97% | 0.03% | 6789/6789 | 4839.83 | 712.89 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v2 / 0.7 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1640.30 | 241.61 |
| fastino.ai / fastino/GLiDE | toxigen | v2 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 2191.40 | 322.79 |
| laya-ai.com / typed-decisions | toxigen | v2 / 0.8 | complete | 6789/6789 | 99.96% | 0.04% | 6789/6789 | 3853.96 | 567.68 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | toxigen | v2 / 0.8 | complete | 6789/6789 | 100.00% | 0.00% | 6789/6789 | 1565.98 | 230.66 |
| fastino.ai / fastino/GLiDE | hatecheck | v1 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1213.50 | 325.51 |
| laya-ai.com / typed-decisions | hatecheck | v1 / 0.5 | complete | 3728/3728 | 99.97% | 0.03% | 3728/3728 | 1664.83 | 446.57 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v1 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 855.00 | 229.35 |
| fastino.ai / fastino/GLiDE | hatecheck | v1 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1236.90 | 331.79 |
| laya-ai.com / typed-decisions | hatecheck | v1 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1572.98 | 421.94 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v1 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 857.90 | 230.12 |
| fastino.ai / fastino/GLiDE | hatecheck | v1 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1090.90 | 292.62 |
| laya-ai.com / typed-decisions | hatecheck | v1 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1477.33 | 396.28 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v1 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 841.24 | 225.65 |
| fastino.ai / fastino/GLiDE | hatecheck | v1 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1422.60 | 381.60 |
| laya-ai.com / typed-decisions | hatecheck | v1 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1586.41 | 425.54 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v1 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 855.70 | 229.53 |
| fastino.ai / fastino/GLiDE | hatecheck | v2 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1719.00 | 461.11 |
| laya-ai.com / typed-decisions | hatecheck | v2 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1712.24 | 459.29 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v2 / 0.5 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 846.55 | 227.08 |
| fastino.ai / fastino/GLiDE | hatecheck | v2 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1310.10 | 351.42 |
| laya-ai.com / typed-decisions | hatecheck | v2 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1648.89 | 442.30 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v2 / 0.6 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 853.83 | 229.03 |
| fastino.ai / fastino/GLiDE | hatecheck | v2 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1697.90 | 455.45 |
| laya-ai.com / typed-decisions | hatecheck | v2 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1611.52 | 432.27 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v2 / 0.7 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 831.23 | 222.97 |
| fastino.ai / fastino/GLiDE | hatecheck | v2 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 1462.30 | 392.25 |
| laya-ai.com / typed-decisions | hatecheck | v2 / 0.8 | complete | 3728/3728 | 99.95% | 0.05% | 3728/3728 | 1990.75 | 534.00 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | hatecheck | v2 / 0.8 | complete | 3728/3728 | 100.00% | 0.00% | 3728/3728 | 835.13 | 224.02 |
| fastino.ai / fastino/GLiDE | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 728.60 | 370.98 |
| laya-ai.com / typed-decisions | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 922.82 | 469.87 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v1 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 435.62 | 221.80 |
| fastino.ai / fastino/GLiDE | aegis2 | v1 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 755.10 | 384.47 |
| laya-ai.com / typed-decisions | aegis2 | v1 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 898.52 | 457.49 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v1 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 428.99 | 218.43 |
| fastino.ai / fastino/GLiDE | aegis2 | v1 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 995.80 | 507.03 |
| laya-ai.com / typed-decisions | aegis2 | v1 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 927.80 | 472.40 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v1 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 427.75 | 217.80 |
| fastino.ai / fastino/GLiDE | aegis2 | v1 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1112.80 | 566.60 |
| laya-ai.com / typed-decisions | aegis2 | v1 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1009.67 | 514.09 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v1 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 438.14 | 223.08 |
| fastino.ai / fastino/GLiDE | aegis2 | v2 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1028.60 | 523.73 |
| laya-ai.com / typed-decisions | aegis2 | v2 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1066.86 | 543.21 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v2 / 0.5 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 434.07 | 221.01 |
| fastino.ai / fastino/GLiDE | aegis2 | v2 / 0.6 | complete | 1964/1964 | 99.85% | 0.15% | 1964/1964 | 1977.70 | 1006.98 |
| laya-ai.com / typed-decisions | aegis2 | v2 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1018.75 | 518.71 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v2 / 0.6 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 434.81 | 221.39 |
| fastino.ai / fastino/GLiDE | aegis2 | v2 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1209.70 | 615.94 |
| laya-ai.com / typed-decisions | aegis2 | v2 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1043.24 | 531.18 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v2 / 0.7 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 437.34 | 222.68 |
| fastino.ai / fastino/GLiDE | aegis2 | v2 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1048.30 | 533.76 |
| laya-ai.com / typed-decisions | aegis2 | v2 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 1112.68 | 566.54 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | aegis2 | v2 / 0.8 | complete | 1964/1964 | 100.00% | 0.00% | 1964/1964 | 441.41 | 224.75 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 7983.40 | 437.25 |
| laya-ai.com / typed-decisions | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 99.83% | 0.17% | 18258/18258 | 16606.74 | 909.56 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v1 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4091.16 | 224.08 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v1 / 0.6 | complete | 18258/18258 | 99.99% | 0.01% | 18258/18258 | 7457.90 | 408.47 |
| laya-ai.com / typed-decisions | detect-jailbreak | v1 / 0.6 | complete | 18258/18258 | 99.92% | 0.08% | 18258/18258 | 15582.19 | 853.44 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v1 / 0.6 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4119.35 | 225.62 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v1 / 0.7 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 6750.40 | 369.72 |
| laya-ai.com / typed-decisions | detect-jailbreak | v1 / 0.7 | complete | 18258/18258 | 99.98% | 0.02% | 18258/18258 | 16128.16 | 883.35 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v1 / 0.7 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4142.22 | 226.87 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v1 / 0.8 | complete | 18258/18258 | 99.99% | 0.01% | 18258/18258 | 5721.80 | 313.39 |
| laya-ai.com / typed-decisions | detect-jailbreak | v1 / 0.8 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 15647.33 | 857.01 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v1 / 0.8 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4103.78 | 224.77 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v2 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 6108.10 | 334.54 |
| laya-ai.com / typed-decisions | detect-jailbreak | v2 / 0.5 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 15068.20 | 825.29 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v2 / 0.5 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4160.19 | 227.86 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v2 / 0.6 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 8883.70 | 486.56 |
| laya-ai.com / typed-decisions | detect-jailbreak | v2 / 0.6 | complete | 18258/18258 | 99.97% | 0.03% | 18258/18258 | 15273.93 | 836.56 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v2 / 0.6 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4080.60 | 223.50 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v2 / 0.7 | complete | 18258/18258 | 99.80% | 0.20% | 18258/18258 | 10732.20 | 587.81 |
| laya-ai.com / typed-decisions | detect-jailbreak | v2 / 0.7 | complete | 18258/18258 | 99.98% | 0.02% | 18258/18258 | 15966.81 | 874.51 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v2 / 0.7 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4104.36 | 224.80 |
| fastino.ai / fastino/GLiDE | detect-jailbreak | v2 / 0.8 | complete | 18258/18258 | 80.35% | 19.65% | 18258/18258 | 6199.20 | 339.53 |
| laya-ai.com / typed-decisions | detect-jailbreak | v2 / 0.8 | in_progress | 0/18258 | 0.00% | — | 3500/18258 | 19214.10 | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | detect-jailbreak | v2 / 0.8 | complete | 18258/18258 | 100.00% | 0.00% | 18258/18258 | 4043.71 | 221.48 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v1 / 0.5 | complete | 1044/1044 | 0.00% | 100.00% | 1044/1044 | 254.30 | 243.58 |
| laya-ai.com / typed-decisions | jailbreak-classification | v1 / 0.5 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v1 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 239.84 | 229.73 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v1 / 0.6 | complete | 1044/1044 | 0.00% | 100.00% | 1044/1044 | 246.40 | 236.02 |
| laya-ai.com / typed-decisions | jailbreak-classification | v1 / 0.6 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v1 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 249.70 | 239.18 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v1 / 0.7 | complete | 1044/1044 | 31.99% | 68.01% | 1044/1044 | 299.80 | 287.16 |
| laya-ai.com / typed-decisions | jailbreak-classification | v1 / 0.7 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v1 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 241.61 | 231.43 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v1 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 314.90 | 301.63 |
| laya-ai.com / typed-decisions | jailbreak-classification | v1 / 0.8 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v1 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 239.06 | 228.99 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v2 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 365.50 | 350.10 |
| laya-ai.com / typed-decisions | jailbreak-classification | v2 / 0.5 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v2 / 0.5 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 241.01 | 230.85 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v2 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 299.70 | 287.07 |
| laya-ai.com / typed-decisions | jailbreak-classification | v2 / 0.6 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v2 / 0.6 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 249.60 | 239.08 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v2 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 349.60 | 334.87 |
| laya-ai.com / typed-decisions | jailbreak-classification | v2 / 0.7 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v2 / 0.7 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 246.67 | 236.27 |
| fastino.ai / fastino/GLiDE | jailbreak-classification | v2 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 342.10 | 327.68 |
| laya-ai.com / typed-decisions | jailbreak-classification | v2 / 0.8 | missing | 0/1044 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreak-classification | v2 / 0.8 | complete | 1044/1044 | 100.00% | 0.00% | 1044/1044 | 240.36 | 230.23 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v1 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 4721.70 | 311.87 |
| laya-ai.com / typed-decisions | jailbreakhub | v1 / 0.5 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v1 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3529.11 | 233.10 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v1 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 4845.10 | 320.02 |
| laya-ai.com / typed-decisions | jailbreakhub | v1 / 0.6 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v1 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3548.75 | 234.40 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v1 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 5602.20 | 370.03 |
| laya-ai.com / typed-decisions | jailbreakhub | v1 / 0.7 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v1 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3520.66 | 232.54 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v1 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 8423.20 | 556.35 |
| laya-ai.com / typed-decisions | jailbreakhub | v1 / 0.8 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v1 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3505.52 | 231.54 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v2 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 7124.20 | 470.55 |
| laya-ai.com / typed-decisions | jailbreakhub | v2 / 0.5 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v2 / 0.5 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3475.07 | 229.53 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v2 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 5187.80 | 342.66 |
| laya-ai.com / typed-decisions | jailbreakhub | v2 / 0.6 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v2 / 0.6 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3447.00 | 227.68 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v2 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 6065.60 | 400.63 |
| laya-ai.com / typed-decisions | jailbreakhub | v2 / 0.7 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v2 / 0.7 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3496.82 | 230.97 |
| fastino.ai / fastino/GLiDE | jailbreakhub | v2 / 0.8 | complete | 15140/15140 | 99.99% | 0.01% | 15140/15140 | 6052.10 | 399.74 |
| laya-ai.com / typed-decisions | jailbreakhub | v2 / 0.8 | missing | 0/15140 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | jailbreakhub | v2 / 0.8 | complete | 15140/15140 | 100.00% | 0.00% | 15140/15140 | 3400.87 | 224.63 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v1 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 376.80 | 418.67 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v1 / 0.5 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v1 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.96 | 219.95 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v1 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 318.20 | 353.56 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v1 / 0.6 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v1 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 200.23 | 222.48 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v1 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 301.40 | 334.89 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v1 / 0.7 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v1 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 201.80 | 224.23 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v1 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 300.20 | 333.56 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v1 / 0.8 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v1 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.96 | 219.95 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v2 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 370.00 | 411.11 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v2 / 0.5 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v2 / 0.5 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.29 | 219.21 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v2 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 311.30 | 345.89 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v2 / 0.6 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v2 / 0.6 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 197.06 | 218.95 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v2 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 334.90 | 372.11 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v2 / 0.7 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v2 / 0.7 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 203.91 | 226.56 |
| fastino.ai / fastino/GLiDE | pii_masking_nano_1k | v2 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 346.20 | 384.67 |
| laya-ai.com / typed-decisions | pii_masking_nano_1k | v2 / 0.8 | missing | 0/900 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_nano_1k | v2 / 0.8 | complete | 900/900 | 100.00% | 0.00% | 900/900 | 200.88 | 223.20 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v1 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 3028.20 | 336.47 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v1 / 0.5 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v1 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2039.10 | 226.57 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v1 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 3119.40 | 346.60 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v1 / 0.6 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v1 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2012.36 | 223.60 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v1 / 0.7 | complete | 9000/9000 | 99.99% | 0.01% | 9000/9000 | 3326.00 | 369.56 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v1 / 0.7 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v1 / 0.7 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2031.76 | 225.75 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v1 / 0.8 | complete | 9000/9000 | 99.99% | 0.01% | 9000/9000 | 4030.30 | 447.81 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v1 / 0.8 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v1 / 0.8 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2042.81 | 226.98 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v2 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 5494.10 | 610.46 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v2 / 0.5 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v2 / 0.5 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2048.51 | 227.61 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v2 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 5078.10 | 564.23 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v2 / 0.6 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v2 / 0.6 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2040.31 | 226.70 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v2 / 0.7 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 3599.00 | 399.89 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v2 / 0.7 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v2 / 0.7 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2032.40 | 225.82 |
| fastino.ai / fastino/GLiDE | pii_masking_mini_10k | v2 / 0.8 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2859.70 | 317.74 |
| laya-ai.com / typed-decisions | pii_masking_mini_10k | v2 / 0.8 | missing | 0/9000 | 0.00% | — | — | — | — |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | pii_masking_mini_10k | v2 / 0.8 | complete | 9000/9000 | 100.00% | 0.00% | 9000/9000 | 2097.96 | 233.11 |

## 请求可靠性（各自已落盘范围，不作为同任务排名）

| 供应商 / 模型 | 已落盘请求数 | 失败数 | 失败率 | 异常类型计数 |
| --- | --- | --- | --- | --- |
| fastino.ai / fastino/GLiDE | 454584 | 6430 | 1.41% | ConnectError: 1；HTTPStatusError: 6423；ReadTimeout: 5；RemoteProtocolError: 1 |
| laya-ai.com / typed-decisions | 227654 | 89 | 0.04% | ConnectError: 13；HTTPStatusError: 72；ReadTimeout: 1；RemoteProtocolError: 3 |
| typesafe.ai / typesafe-standard（配置别名；实际为 SDK system_one，后端版本未记录） | 454584 | 0 | 0.00% | 无 |
