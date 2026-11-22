# 工作机开发任务：X-57/RCAIDE 冷启动

在工作机用原 X-57 LSBO 和 RCAIDE 评价器开发一套可审计的配对试验。第一轮回传开发分支、接口清单、测试记录和资格试验计划。当前开发验收使用解析或模拟 worker，真实 RCAIDE 的首批运行另按明确的额度启动。

## 固定参考代码

| 用途 | 入口 |
|---|---|
| 原生 X-57 LSBO 基线 | [冻结源树](https://github.com/Luyi1235/RA-LSBO/tree/61c265c626957cb8d69761087270721b846025b5/runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base)。读 [lsbo/optimizer.py](https://github.com/Luyi1235/RA-LSBO/blob/61c265c626957cb8d69761087270721b846025b5/runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base/lsbo/optimizer.py) 的 `propose`、`search_phase`、`evaluate`，以及同目录的 `space.py`、`representation.py`、`surrogate.py`。工作机的原项目有记录的路径为 `D:/LSBO/LSBO/X57-LSBO-refactored`，先核对现有副本和冻结参考的差异，再记录实际采用的源提交或文件哈希。 |
| X-57 物理评价与预检 | [problems/x57](https://github.com/Luyi1235/RA-LSBO/tree/61c265c626957cb8d69761087270721b846025b5/runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base/problems/x57) 下的 `problem.py`、`adapter.py`、`worker.py`。参考 [X57LiveProblem](https://github.com/Luyi1235/RA-LSBO/blob/61c265c626957cb8d69761087270721b846025b5/ra_lsbo/x57_live.py) 和 [run_x57_preflight.py](https://github.com/Luyi1235/RA-LSBO/blob/61c265c626957cb8d69761087270721b846025b5/run_x57_preflight.py) 的独立 worker 与冻结契约。 |
| 定版风险模块 | [run_aircraft_native_risk_lsbo.py](https://github.com/Luyi1235/RA-LSBO/blob/fddfe2e476b09db2a38e8b957646bac4d7017055/run_aircraft_native_risk_lsbo.py) 中的两阶段提案和回退，[aircraft_structured_risk.py](https://github.com/Luyi1235/RA-LSBO/blob/fddfe2e476b09db2a38e8b957646bac4d7017055/aircraft_structured_risk.py) 中的构型层级核，以及 [completion_refinement.py](https://github.com/Luyi1235/RA-LSBO/blob/fddfe2e476b09db2a38e8b957646bac4d7017055/ra_lsbo/risk_core/completion_refinement.py) 的风险加权。该 runner 的任务评价器是降阶模型，迁移风险模块时接入 X-57 worker。 |
| 既有反馈证据 | [X-57 历史反馈审计](https://github.com/Luyi1235/RA-LSBO/blob/697346ead98bfa02be21fe904b447eb97d8dca8f/results/aircraft-risk-x57-feedback-audit-20261115-61c265c/report.json)。同旧任务的 37 次启动没有可用设计，因此先核实新接口与任务的可行性和反馈覆盖。 |

## 开发内容与验收

1. **统一评价接口和结果语义。** 以既有 X-57 worker 为评价入口，为三臂统一候选规范化、工程预筛、状态映射和账本。冻结任务配置、RCAIDE 版本、模型及适配器哈希。保留原始 worker 记录，并映射为评价完成、工程可行、目标及约束的精确值或缺失、可信风险范围、数值失败和基础设施状态。正常完成但违反工程约束的记录仍标为完成。风险区间须对应可解释的同一风险指标和已验证的边界；一般数值不收敛只在有可靠依据时提供状态信息。基础设施未知独立记账。
2. **保持原 LSBO 的选点和更新。** 保留构型潜空间、目标与约束代理、候选生成和局部搜索。用适配层处理共享初始化和预算账本。先做记录回放，核对原提案及更新保持一致。原生代码有连续缺失目标停止规则。完整预算版本若改动这一项，单独标记为 `LSBO-full-budget`，公布与原停止策略的差异，保留提前停止原因和实际费用。公共的系统故障处理对各臂一致。
3. **接入两个风险对照臂。** `structured_sign` 与 `structured_interval` 使用相同的结构化表示、可用设计门槛、候选生成、工程筛选、后续决策和回退。在具有可靠反馈的同一批观测类型上，符号臂仅使用标志或单侧符号，区间臂使用已核验的范围。两臂保留相同的目标与工程约束记录。冷启动采用层级核，在筛选后的候选中按完成概率选点；达到预设可用设计数量后，沿用定版的完成概率加权 LSBO 改进评分。原定版门槛为 3 个可用设计，资格试验若需修改须先记录依据，正式批次前冻结。对未能可靠构造区间的工况，符号臂可验证完成状态学习，区间贡献单独标为尚待验证。
4. **实现顺序运行与可复核结果。** 每个种子只求解一次共享初始化。评价预检失败、发起调用、结算和基础设施中断分别记录；只要实际发起 RCAIDE 即计费。保存每次候选、构型、阶段、选点来源、完成概率、观测、worker 原始工件位置和耗时。分别输出首次可用评价次数、首次达到冻结的任务能耗目标所需评价次数、达标率和最佳可行能耗。查验第 32、64、128、256、320 次节点。并行以相互隔离的种子或运行作业为单位，各条 BO 轨迹内保持逐次观察和更新。
5. **本地验收。** 覆盖两种构型的变量激活、共享初始化和计费、完成但违约、可信区间与符号、无效区间、数值非收敛、系统失败、代理切换、无可用解和受控停止。确保规划或 dry-run 在工作机环境可运行，并回传 Python 与 RCAIDE 版本。继续沿用 55 nmi 作为资格试验候选任务，先预登记 16–24 次资格点，覆盖两构型和少量复验。确认两条分支的可用区域及反馈口径后再登记正式运行的种子、初始化、目标和预算。

## 正式预算与初始化

- **建议每种子 16 个共享初始点，320 次总评价含初始化。** 两类推进布局各 8 点，在各自有效的连续子空间内分散取点，覆盖预先固定的代表性离散层级。候选可先做确定性工程预筛，后续 RCAIDE 结果不参与初始点选择。若资格检查显示条件分支覆盖不足，可在正式批次冻结前改用 24 个。4 个或 8 个初始化保留为可选的更极端冷启动敏感性，与 16 使用嵌套设计并保持各自总预算一致。
- 报告全部配对种子，同时按共享初始观测划分零可用的冷启动组。以初始化结果决定组别，固定并保留全部种子。资格试验用于核实任务与评价接口，并估算耗时，资格点和历史标签留在正式优化训练数据之外。
- 按 16 个共享初始点，三臂各做 304 次在线评价。每种子实际 RCAIDE 上限为 928 次，六个配对种子在单工况下的上限为 5,568 次，另计资格试验。用资格试验的耗时和失败比例排正式批次。
