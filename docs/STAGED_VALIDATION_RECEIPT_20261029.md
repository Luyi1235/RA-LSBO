# RA-LSBO 分阶段实现验证回执（2026-10-29）

固定运行提交：`cfdf7c30216585b3920f23723236708859435ce7`。
代码树：`cf8cddaed966f7897ce83f3ee17083e5c7311bbe`。
分支：`research/discovery-stages-20261029`，PR #3。
最终成功CI：https://github.com/Luyi1235/RA-LSBO/actions/runs/40819328407 。
Job：123267828773。数据来自实际下载的JUnit、plans/results/audits和原始证据，不是仅按配置推算。

## 最终验收

影响范围回归：**229 passed、0 failures、0 errors、0 skipped**。
范围为两份新测试与七份既有相关测试；不是全仓库全部历史测试。
Linux/Python3.11.17、CPU torch2.8.0+cpu、numpy2.0.2、scipy1.13.1、gpytorch1.11、botorch0.10.0、scikit-learn1.6.1、matplotlib3.9.4、pytest8.3.5。
其他告警没有全局屏蔽；完整版本见CI artifact的environment.txt。Windows工作机必须另行验收。

| smoke | 实际调用/上限 | trial | 在线提案 | 来源绑定 | 审计证据文件 | 运行/审计/报告 |
|---|---:|---:|---:|---:|---:|---|
| discovery | 104/104 | 20 | 40 | 162 | 88 | 全部通过 |
| representation | 128/128 | 32 | 64 | 162 | 136 | 全部通过 |
| baselines | 128/128 | 32 | 64 | 162 | 144 | 全部通过 |

最终三套smoke共360次数学模型调用，0次RCAIDE MDA。保护组在这些smoke中阈值内近重复送评为0，只验证有限测试轨迹，不证明全局性能。原生LSBO停止测试使用人工构造的已结算标签，并明确禁止调用真实problem.evaluate。

## 本轮开发累计调用及失败记录

| CI run | 结果 | 研究smoke调用 |
|---|---|---:|
| 40818187979 | setup-python依赖缓存未指定requirements-models.txt而失败；测试未启动 | 0 |
| 40818515157 | 225项通过，首版三套smoke通过 | 360 |
| 40819020974 | 228通过、1项新原生LSBO测试夹具失败：1初值违反冻结算法至少4初值要求；smoke未启动 | 0 |
| 40819328407 | 修正夹具后229通过，最终三套smoke通过 | 360 |

本轮开发新增研究smoke共**720次**，不是只计算最后360次。单元测试内部廉价/人工求值另计，未对全部测试内部评价器统一计数；没有把这些调用冒称为0。完整pilot未启动，旧实验费用未改写，新增RCAIDE MDA为0。

## 证据身份与边界

最终artifact：`staged-validation-cfdf7c30216585b3920f23723236708859435ce7`；ID12874095919。
入口：https://github.com/Luyi1235/RA-LSBO/actions/runs/40819328407/artifacts/12874095919
ZIP SHA256：`e8d296789d03d4daae4e4f895a18363175a55a123ac28bd72a4019a3dc0c70a8`。

Artifact保留14天，包含三套smoke的完整源快照、逐步提案/观测/账本、计划/审计/结果/报告及JUnit环境记录。下载后逐项复核三套审计所列证据哈希及全部来源快照，均相符。冷启动、近重复、覆盖和选点索引不变量已检查。

不是每一步GP拟合的独立算法重放，不是物理真值校核，不代表20种子优势或安全保证。数学区间主要是符号，不能证明一般有限区间的收益。CBOB、全新分支权衡任务、确认性统计未冒充已完成；真实飞机MDA未开放。

原RA-v2、冻结LSBO、旧外部适配器源码及历史数据未改。新实现位于discovery_stages，保守SEGO与符号一致UTB采用不同新标识，未重命名历史结果冒充修复后结果。详细产物哈希见`STAGED_VALIDATION_FACTS_20261029.json`。本文件为摘要，不冒充原始CI日志。
