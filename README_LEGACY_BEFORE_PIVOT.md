# RA-LSBO 初步推进

## 当前双实验包（2026-10-26）

数学基准和简化飞机已拆成两个独立闭环包，共用可审计的单写入执行引擎：
`ra_lsbo/experiments/mathematical_benchmarks/` 与
`ra_lsbo/experiments/simplified_aircraft/`。各包均提供
`prepare/run/audit/report`，并有独立问题入口、任务过滤、预算计划和预注册协议
（分别见 `docs/MATHEMATICAL_EXPERIMENTS_PROTOCOL.md`、
`docs/SIMPLIFIED_AIRCRAFT_EXPERIMENTS_PROTOCOL.md`）。

两包 smoke 均已完成：各 120 次模型调用，合计 240 次；审计通过，新增 RCAIDE MDA 为 0。
这是闭环与接口验证，不是算法独立回放，也不足以证明稳定优化优势。
旧综合实现及其 smoke、partial-latent、partial-refinement 和 SOC revision 结果已归档于
`archive/historical_research/2026-10-26/`；共享冻结飞机基线保留在 `runs/` 供当前包使用。

2026-10-26已完成SOC重复不可行诊断、隔离修订与492次新廉价评价，0次新增RCAIDE MDA，见
[SOC诊断及三层实验简报](docs/RA_LSBO_SOC_AND_THREE_LEVEL_REPORT_20261026.md)和
[逐种子技术记录](docs/SOC_REVISION_RESULTS_20261026.md)。
仅修订利用评分后，飞机SOC中止由6/36降至3/36、新增可行由12/36升至18/36，最优能耗实用持平；
完整核平均组合在Branin三种子均退化，未过预定门槛，不替换默认方法。
新飞机面板6/30可行，不支持沿用旧2/30宣称稳定稀疏；六组初始集均有可行点，冷启动尚未验证。
30项新针对性测试通过；360个选点、30个面板模型、492次调用及126项来源绑定全量审计通过。
新实现入口为 `ra_lsbo/soc_revision/`。
三层规划为数学机制、简化电动飞机和单一RCAIDE/X57工程案例，后续任务扩展尚未执行。

2026-10-26进一步保留并接入原LSBO的分支精英邻域、活跃连续变量局部优化及自适应半径，
形成混合观测GP驱动的全局候选与局部连续搜索组合，见
[局部优化组合结果](docs/PARTIAL_REFINEMENT_RESULTS_20261026.md)及
[冻结主协议](docs/PARTIAL_REFINEMENT_PROTOCOL_20261026.md)。
独立140km简化飞机资格面板2/30可行，点估计门槛通过，但95%区间仍较宽。
30组主比较与完整预算原LSBO次要控制共新增430次廉价模型调用，0次RCAIDE MDA。
新组合在Branin对完整预算原LSBO为3胜0负，飞机为2胜1负；飞机一组明显退化，
潜空间、分组核和可行性概率质量尚无稳定优势，不替换原默认方法。
71项针对性测试通过；346个主选点、18个面板模型和72个完整预算对照选点恢复一致，审计通过。
本阶段实现入口为 `ra_lsbo/partial_refinement/`。
旧MDA第37次中断调用另有[只读核查](docs/COMPACT_MDA_INTERRUPTION_20261026.md)，终态仍未知。

2026-10-26已按 GPML 与部分观测贝叶斯优化思想搭建新的组合框架，见
[代码与实验结果](docs/PARTIAL_LATENT_BO_RESULTS_20261026.md)及
[方法与预注册协议](docs/PARTIAL_LATENT_BO_PROTOCOL_20261026.md)。
本轮基准函数、简化电动飞机及独立航程补充合计491次廉价模型调用，0次新增RCAIDE MDA；
42项针对性测试通过，327个在线选点恢复重放通过，旧冻结源码未改。
组合方法尚未稳定优于原LSBO；100km飞机任务14/30随机可行，不足以作为稀疏场景。
125km只作为下一轮资格验证候选，不能提前宣称已满足稀疏性或创新组件有效。
该阶段实现入口为 `ra_lsbo/partial_bo/` 和 `tools/run_partial_bo_development.py`。

2026-10-25根据用户进一步明确的范围，主线限定为飞机概念设计、稀疏可行域与风险感知。
先用少量廉价实验直接筛选“三种风险核心与同一结构化潜空间”的适配性，
再推进一个飞机MDA任务，物理并发固定为1。见[当前主线及筛选协议](docs/AIRCRAFT_MAINLINE_20261025.md)。
此前涡扇发动机和大规模多任务提案已撤出执行主线，保留为历史设计记录。
本轮少量筛选已完成20组、376条廉价标签、0次新增真实MDA，
见[适配筛选结果](docs/LATENT_CORE_SCREEN_RESULTS_20261025.md)。混合观测GP与校准分支并列，
按简约规则保留混合GP为研发底座；其潜空间组合保留种子验证未过门槛，
新增可行为4/48，原LSBO为13/48，不能宣布已选出效果合格的最终方法。

2026-10-25完成三条独立风险核心的并行实现、56组筛选及保留种子组合验证，
见[本轮选择与组合结果](docs/RISK_CORE_SELECTION_RESULTS_20261025.md)。
实际487条受控标签、0次新增真实MDA；403项测试及423次完整提案重放通过。
边界集成在三条新路线中按预定规则排名第一，但组合方案仅6/56次新增可行，
原LSBO为12/55；隐藏失败分别为39和19，升级门槛未通过。
目前没有同时满足效果、创新性与科学证据要求的最终方案，不替换原默认方法。
以下保留此前各阶段的历史记录。

2026-10-25按 SMO 研究标准继续推进，见
[研究标准与差距](docs/SMO_RESEARCH_ALIGNMENT_20261025.md)和
[本轮实现及结果](docs/SMO_PROGRESS_20261025.md)。
完成20种子/60组/480条廉价真值的发现阶段诊断，240次完整采集与120次原LSBO提案重放通过；
可行性优先修改尚无充分改善证据，仍落后于原LSBO。
新增支持观测缺失、相关输出和联合后验抽样的多任务约束GP及选点接口；
128条真实历史训练、262个去重设计的回顾性诊断中，联合数值预测改善但条件可行性Brier略差，
不能宣布在线优化收益。原物理目录变化已用历史快照隔离恢复，
300次上限新研究的独立资格检验已在恢复目录启动；终态与费用以该阶段结果和账本为准。
当前仍未达到完整论文效果已验证的状态。

2026-10-21进一步按“分析原因、修改不确定性逻辑、重设计稀疏真实MDA任务”推进，
见[本轮诊断与实算结果](docs/POSTERIOR_AND_MDA_RESULTS_20261021.md)。
新增完成状态变分GP、约束后验抽样和无可行初值的违反量搜索；30组小样本对照、
480条新增受控标签、216次完整采集值恢复通过，仍未稳定优于原LSBO。
新建紧凑机翼/长航程场景并完成6次真实MDA，证明SOC、端口能量和失速约束存在竞争，
但独立稀疏性资格验证尚未执行，不能把选定锚点结果当作稀疏率。
新研究完整上限300次，当前6次计入其中；旧v1--v5累计500次保持不变。
新协议见 `docs/POSTERIOR_MDA_PROTOCOL_20261021.md`，最新246项测试通过。
以下保留此前阶段的历史记录与结论。

2026-10-21按用户新授权完成两轮方法修订，详见[方法修订与结果](docs/METHOD_REVISION_RESULTS_20261021.md)。新增部分标签联合终止网络、约束余量/对比监督风险表示、bootstrap信息分歧、固定参考在线学习与条件约束GP桥接。两轮各66组，另有6组事后配对控制，共138组、2,853次廉价受控在线调用；1,937个选点风险预测恢复回放，最新235项测试通过。新增真实MDA为0，旧冻结结果保留。后续试验可用的归一化模型工厂见 `docs/STABLE_MARGIN_USAGE_20261021.md`，其自身尚无新的优化效果试验。

新直接联合模型在262个唯一历史飞机设计上的原始Brier为0.013915，旧共同参考为0.032574；这是回顾性预测改善，不是真实优化收益。第二轮普通受控任务完整模型平均改进0.278101%，原LSBO为0.277286%；稀疏任务为0.105914%与0.248644%，仍落后。潜变量、机制分解和信息探索尚无一致优势，其他四类真实隐藏失效仍无正例，不替换原默认。文献与两轮协议见 `docs/METHOD_REVISION_RESEARCH_20261021.md`、`docs/ADAPTIVE_RESEARCH_PROTOCOL_20261021.md`、`docs/MARGIN_JOINT_PROTOCOL_20261021.md`。

本目录已形成 RA-LSBO 的研究原型：风险数据协议、潜空间与多头模型、时间隔离校准、风险账本、统一候选审批及合成问题优化闭环。已接入工作区 LSBO 进行配对比较，但当前实验未证明 RA-LSBO 优于 LSBO。2026-10-17 已完成单次 v3 真实检查及 v4 前瞻试点；后者完成 6 次后因累计 3 个失速违规触发预定停止。2026-10-20 用户批准 120 点边界验证，并授权直接推进不高于 500 次的完整评估计划；超过 500 次才另行确认，不拆分计划规避门槛。

本项目不修改 `../LSBO/X57-LSBO-refactored` 的优化器、物理模型、参数或历史结果。模型层使用本机已有的 `Rcaide_env`，没有安装依赖或修改环境。既有文档中的“未调用新 MDA”描述的是 2026-09-30 阶段，不代表本轮状态。

本轮机制数据补充与有限完整系统验证已经结束，总报告见 [最终结果](docs/SYSTEM_VALIDATION_FINAL_RESULTS_20261021.md)。全部修订版本累计500次真实MDA，包括140次训练补采、72次全新校准和8方法×3种子×12次的288次真实优化；独立审计通过。共同历史632条，实验后去重保留894个设计、无冲突，完整权重已保存并通过恢复预测检查；218项软件测试通过。

真实任务失败共46个正例，其余四类仍无真实正例；留出整个失效锚点族时，两种表示的失败召回均为0，不能判为泛化充分。完整RA平均改进1.57644%，原LSBO为1.62606%，两者新增可行均为33/36，未证明完整组合稳定领先。实际选点上的完整RA最终可行性Brier由原始0.04141变为校准后0.05326，没有改善。两个五机制受控案例单独保留，不进入飞机训练。模型与数据用法见 `docs/SYSTEM_DATA_AND_MODELS_20261021.md`；判定口径见 `docs/MECHANISM_DATA_ACCEPTANCE_20261021.md`。

上一阶段已按用户要求先完善结构化编码器，再做实验，见 `docs/STRUCTURED_ENCODER_RESULTS_20261020.md`。已新增独立的论文对齐 VAE、五种特征视图、风险/残差 GP/校准接口、增量更新及 checkpoint；当时 185 项测试通过、无跳过。8 维、600 轮在三个种子的训练语法内均达到 100% 完整构型恢复，九次历史增量更新也全部保持；该配置历史风险测试有所改善，但边界局部评分仍落后于层级 one-hot，未支持跨分布稳定领先，不替换旧默认模型。该编码器阶段新增真实 MDA 为 0，旧拟议的 192 次闭环方案未实施。

此前 120 次真实验证全部完成，53 点可行、67 点显式违规，无未知结果或重试，审计通过，见 `docs/BOUNDARY_RESULTS_20261020.md`；推进记录见 `docs/PROGRESS_20261020.md`。约束感知局部校准的留出 Brier 从 0.31424 降至 0.17456，但 0.8 门槛仅接受 1 点，尚未证明风险受控的持续优化有效。2026-10-17 的 135 组合成比较、冷启动修复和边界诊断保留在 `docs/CONTINUATION_RESULTS_20261017.md`。早期 150 组结果见 `docs/PROGRESS_20261017.md`，真实试点见 `docs/PILOT_RESULTS_20261017.md`。新版论文衔接见 `docs/PAPER_ALIGNMENT_20261017.md`。论文 24 变量与当前 19 变量工程分支保持区分。

## 当前交付

| 文件 | 用途 |
| --- | --- |
| `ra_lsbo/soc_revision/` | 独立开关的核证据平均、逐分量概率/EI及有可行解后的风险利用评分 |
| `tools/run_soc_revision_study.py` | 492次新标签的五组机制验证、费用绑定与全量无真值回放 |
| `ra_lsbo/partial_refinement/` | 混合观测GP、分组核、分支精英邻域及活跃连续变量局部搜索 |
| `tools/run_partial_refinement_study.py` | 冻结的基准函数和140km简化飞机比较、资格检验及全量选点恢复 |
| `tools/continue_lsbo_full_budget_control.py` | 原LSBO完整预算次要控制，核验并复用前缀，只补未执行评价 |
| `ra_lsbo/records.py` | 统一风险评估结果及有效性校验 |
| `ra_lsbo/legacy.py` | 将现有 LSBO 的 `Evaluation.to_dict()` 转为风险结果 |
| `ra_lsbo/training.py` | 分别导出目标、数值约束、隐藏失效及最终可行性训练行 |
| `ra_lsbo/audit.py` | 只读审计单个已结束运行，生成带来源哈希的数据快照 |
| `ra_lsbo/features.py` | 隔离加载既有表示模块，管理时间前缀及表示快照 |
| `ra_lsbo/structured_encoder.py` | 新的论文对齐结构化 VAE、多视图、增量更新和 checkpoint |
| `ra_lsbo/structured_models.py` | 新编码器的风险/GP/校准接口及可选优化循环策略 |
| `ra_lsbo/adaptive_risk.py` / `ra_lsbo/adaptive_system.py` | 独立实验版本：部分标签事件集成、风险表示、固定参考更新和软风险价格 |
| `ra_lsbo/margin_joint.py` / `ra_lsbo/normalized_margin_joint.py` | 完成条件下的GP约束桥接、完整参数恢复及显式数值归一化 |
| `tools/run_adaptive_research.py` / `tools/run_margin_research.py` | 两轮零新增真实MDA的冻结开发试验；旧默认和旧实验不改写 |
| `tools/report_method_revision.py` | 两轮与事后配对控制核验、唯一历史设计评分及结果图表 |
| `ra_lsbo/mechanism_system.py` | 实际可训练机制头与约束感知评分、增量拟合及独立受控串行问题 |
| `ra_lsbo/system_experiment.py` | 八方法配对闭环、原 LSBO 原生搜索适配和逐次真实计费 |
| `ra_lsbo/responses.py` | 目标性能先验与残差 GP、独立数值约束 GP |
| `ra_lsbo/risk_model.py` | 总体隐藏成功、最终可行性和五个失效机制分类头 |
| `ra_lsbo/metrics.py` | Brier、对数损失、ECE 和可靠性分箱 |
| `ra_lsbo/replay.py` | 逐前缀重新训练、先保存预测再揭示结果 |
| `ra_lsbo/calibration.py` | 冻结模型的逐头 sigmoid 校准、样本门槛及降级 |
| `ra_lsbo/calibration_replay.py` | 训练 / 校准 / 测试时间块隔离、配对评分与来源复核 |
| `run_audit.py` / `run_replay.py` | 数据审计 / 离线模型回放入口 |
| `run_calibration.py` | 真实历史数据三段式校准检查 |
| `run_calibration_synthetic.py` | 固定种子的已知失准合成检查及可靠性图 |
| `ra_lsbo/budget.py` | 持久化预约、提交、结算与可回放预算账本 |
| `ra_lsbo/selection.py` | Log-EI、约束边界与分类器分歧联合评分，最终候选统一审批 |
| `ra_lsbo/optimizer.py` | 分块重训与重新校准、开发 / 边界 / 发现模式的优化闭环 |
| `ra_lsbo/discovery_learning.py` | 校准不可用时逐次使用全历史更新原始发现模型，不升级概率身份 |
| `run_discovery_benchmark.py` | 原预算下 60 组新鲜学习与另 75 组约束引导发现检验 |
| `ra_lsbo/constraint_risk.py` | 约束感知原始风险评分及身份绑定，不宣称联合概率 |
| `run_constraint_risk_diagnostic.py` | 固定历史切分上的探索性诊断，不调用 MDA |
| `ra_lsbo/boundary_validation.py` / `run_boundary_validation.py` | 单独批准的 60 校准 + 60 留出真实验证入口 |
| `tools/verify_boundary_request.py` / `tools/audit_boundary_validation.py` | 大批量执行前检查与执行后核验 |
| `tools/analyze_boundary_validation.py` | 固定留出评分、接受风险、分层配对区间及图 |
| `tools/report_boundary_validation.py` | 终止且审计通过后的结果报告，保留提前停止与未执行 |
| `tools/risk_feature_ablation.py` / `tools/summarize_risk_feature_ablation.py` | 历史风险特征消融、原标签独立核验和图 |
| `tools/boundary_feature_validation.py` | 同一批真实设计上的次要特征预测冻结、局部校准与留出比较，无新增 MDA |
| `tools/encoder_framework_experiment.py` / `tools/audit_encoder_framework.py` | 新旧编码器、损失消融与 one-hot 的回顾性比较及独立预测重建 |
| `tools/encoder_training_budget.py` | 结构恢复的潜维数/训练预算诊断，不使用风险测试评分选配置 |
| `tools/encoder_candidate_validation.py` | 结构合格配置的风险对照、历史增量保持与独立预测重建 |
| `tools/run_system_validation_v2.py` / `tools/audit_system_validation_v2.py` | 包含首版调用的修订真实验证、全新校准段与独立审计 |
| `tools/run_system_validation_v3.py` / `tools/audit_system_validation_v3.py` | 真实失效邻域定向补采、继承所有版本费用及不变的配对优化 |
| `tools/verify_system_v3_request.py` | 零MDA核验真实物理候选的合法性、去重、邻域距离与完整预算 |
| `ra_lsbo/partial_observation.py` | 严格来源校验后的有效能耗部分观测；缺失约束与可行性保持未知 |
| `tools/run_system_validation_v4.py` / `tools/audit_system_validation_v4.py` | 保留68次费用及原始未知记录、采用部分观测契约继续未执行实验 |
| `tools/run_system_validation_v5.py` / `tools/audit_system_validation_v5.py` | 保留此前128次费用、单因素构型训练补采、全新分层校准及24个真实对照 |
| `tools/verify_system_v5_request.py` / `tools/analyze_system_validation_v5.py` | 500次累计上限、候选与来源预检及实际结果分析 |
| `tools/analyze_system_validation.py` | 保留早停、共同实际调用数比较、费用归属及真实结果图 |
| `tools/reconstruct_system_forecasts.py` | 从历史前缀和 checkpoint 重建风险、GP 及校准预测，不调用 MDA |
| `tools/release_system_training_data.py` | 同契约数据去重、冲突隔离、实验后模型拟合及完整权重包保存/加载 |
| `tools/release_shared_system_model.py` | 单独保存不包含优化结果的共同参考模型及全新留出校准块 |
| `tools/validate_mechanism_transfer.py` | 同时留出锚点和全部邻居的失效区域迁移诊断，禁止未来校准和优化数据进入训练 |
| `tools/diagnose_system_components.py` | 真实模型权重重载后检查机制头、总体隐藏头与约束评分的额外贡献及标签冗余 |
| `tools/score_system_risk_forecasts.py` | 在实际新增选点上评分既有风险预测，不用校准拟合块冒充独立检验 |
| `tools/export_best_system_design.py` | 从独立审计通过的实际响应导出最优已观察可行构型和约束证据 |
| `tools/summarize_system_acceptance.py` | 汇集数据、模型、真实与受控对照、预测重建及仍未满足的研究证据 |
| `ra_lsbo/synthetic.py` | 层级变量、稀疏可行域、串行失效和多约束违规测试问题 |
| `ra_lsbo/legacy_benchmark.py` | 不修改原代码的原 LSBO 比较适配器 |
| `run_synthetic_benchmark.py` / `plot_benchmark.py` | 多种子配对实验与逐种子图 |
| `run_x57_preflight.py` | 历史物理契约、当前配置与校准证据预检，无 MDA |
| `run_x57_smoke.py` | 准备或执行已批准的单次真实评估请求 |
| `ra_lsbo/catalog.py` / `run_catalog.py` | 多目录终止日志清单、原样副本排除与物理契约分组 |
| `ra_lsbo/transfer_validation.py` / `run_transfer_validation.py` | 冻结模型的跨搜索策略验证，排除共享初始化和重复设计 |
| `ra_lsbo/x57_pilot.py` / `run_x57_pilot.py` | 预生成最多 8 次的冻结候选试验，另行批准后执行 |
| `tools/plot_transfer_results.py` | 跨策略评分表与可靠性图 |
| `tests/` | 数据协议、模型掩码、表示一致性及无未来标签泄漏测试 |
| `docs/IMPLEMENTATION_PLAN.md` | 后续模型、校准、选点和实验的接入顺序 |
| `docs/SYSTEM_VALIDATION_FINAL_RESULTS_20261021.md` | 本轮500次真实MDA、完整对照、负面证据和实际交付 |
| `docs/SYSTEM_DATA_AND_MODELS_20261021.md` | 两套权重包、去重训练数据和只读加载方法 |
| `docs/BOUNDARY_RESULTS_20261020.md` | 120 次真实验证、独立留出结果及前瞻次要特征比较 |
| `docs/STRUCTURED_ENCODER_FRAMEWORK_20261020.md` | 结构化编码器框架、论文衔接及接口 |
| `docs/STRUCTURED_ENCODER_RESULTS_20261020.md` | 框架完成项、风险诊断、结构恢复结果及使用示例 |
| `docs/MODEL_VALIDATION.md` | 第二阶段真实回放设置、结果及适用边界 |
| `docs/CALIBRATION_VALIDATION.md` | 第三阶段校准协议、验证结果及尚未完成项 |
| `docs/CLOSED_LOOP_VALIDATION.md` | 优化闭环、比较结果、真实数据扩展验证及待授权事项 |

## 运行

在本目录使用 Python 3.9 或更高版本。仅审计数据：

```powershell
python run_audit.py --source "D:\LSBO\LSBO\X57-LSBO-refactored\runs\trial\run_20260928_142256_6443ac03"
```

运行全部测试和模型回放，使用已验证的本机环境：

```powershell
& "D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe" -m unittest discover -s tests -v
& "D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe" run_replay.py --source "D:\LSBO\LSBO\X57-LSBO-refactored\runs\trial\run_20260928_142256_6443ac03"
```

依赖版本记录在 `requirements-models.txt`。缺少模型依赖时，模型测试会跳过；不能把这种结果视为模型验证通过。2026-09-30 在上述环境中执行了 104 项测试，全部通过且无跳过。

`--source` 必须显式指定。工具不自动选择最新目录，不审计正在运行的实验，不加载旧 VAE 权重，不读取论文数值。回放默认使用前 10 条记录初始化，VAE 训练 60 epochs，GP 训练 45 steps，种子为 0。

默认在 `runs/audit_<UTC时间>_<唯一标识>/` 生成：

- `report.json`：状态统计、各失效类别正负样本数量、缺失值及问题清单。
- `observations.jsonl`：保留候选、评估编号、阶段、提案信息的标准化风险记录。
- `training_rows.json`：每个模型头各自可使用的候选与标签，尚未编码或训练。
- `evaluator_contract.json`：源运行保存的物理配置、约束、变量、依赖和评估器代码哈希。

原始日志不变。输出目录不能位于源运行内部；导出前复核源文件哈希以拒绝读取过程中发生变化的记录。每次导出新建目录，不覆盖之前的报告。来源文件与转换代码的 SHA-256 一并记录。

回放另建 `runs/replay_<UTC时间>_<唯一标识>/`，生成 `forecasts.jsonl`（不含当次结果）、`scored_predictions.jsonl`（预测后附加结果）和 `report.json`。报告记录逐头评分、先验与拟合状态、环境版本和来源哈希；逐次预测记录训练编号和表示快照。

## 概率校准

```powershell
& "D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe" run_calibration.py --source "D:\LSBO\LSBO\X57-LSBO-refactored\runs\trial\run_20260928_142256_6443ac03" --train-count 18 --calibration-count 6
& "D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe" run_calibration_synthetic.py
```

第一条命令采用固定的 18 / 6 / 6 时间分块，VAE 和原始分类头只用前 18 条数据训练，并在校准及测试期间冻结。分块大小需要显式指定，不能看完测试评分后重新挑选。它与 `run_replay.py` 的逐前缀重新训练是两种不同的验证协议，不混用已有回放评分拟合冻结模型校准器。

校准方法固定为 logit 概率上的正则化 sigmoid，默认至少 30 个有效校准样本、正负类各 5 个、3 个不同分数，且原始分类器必须已拟合。门槛是工程启用条件，不是可靠性或覆盖率保证。不足时 `risk_probabilities_calibrated` 为 `null`，`risk_probabilities_effective` 保留明确标记的原始预测；预算审批不能仅凭 effective 字段把降级结果视为可信。

新增合并事件 `any_hidden_failure` 来自 `1 - hidden_success`，不累加各机制概率。合并事件不替代缺少数据的单机制校准，也不是仅包含稀有类型的专用分组。

真实检查在 `runs/calibration_<UTC时间>_<唯一标识>/` 保存报告、校准样本、未附真值的测试预测和评分记录。合成检查另存 `runs/calibration_synthetic_.../`，包括可靠性图；绘图使用本机已有的 Matplotlib 3.9.4。

真实校准段仅有 6 个样本且全部可行，所有校准器均正确降级。独立合成检查使用 1,000 个校准样本和 2,000 个测试样本，Brier 从 0.2928 降至 0.2131。这只验证预设合成失准情形中的实现，不证明飞机数据校准有效。详细协议与结果见 `docs/CALIBRATION_VALIDATION.md`。

后续对已完成的 400 次真实日志采用 240 / 80 / 80 划分。最终可行性校准器能够拟合，但留出测试 Brier 从 0.2087 恶化到 0.2299。该结果已保留，没有据测试集调阈值或改选校准方法，不应将该校准器当作已经验证可靠的真实候选筛选器。

## 优化闭环与比较

```powershell
& "D:\ProgramData\anaconda3\envs\Rcaide_env\python.exe" run_synthetic_benchmark.py
```

默认运行 3 个种子、3 个场景、5 种方法，共 45 组。比较原 LSBO、独立约束 BO、RA-LSBO、去边界消融和纯随机采样。各组共享初始候选和 112 次调用上限，其中 88 次初始化也计费；RA 保留最近 40 条作校准，每 8 条新数据更新模型。这里只是小规模研究试跑，不是论文统计结论。

所有 RA 候选先规范化、做廉价预检，再根据最终实际坐标审批。开发候选须有经验校准值且达到固定阈值；边界候选有单独数量上限；无可行点或校准不可用时走独立限额的发现策略。候选修复后预测绑定失效，必须重新评分。选择使用 Log-EI 保持极小改进量的排序。

账本写入并刷新后才提交评估。初始化和未校准发现按每次 1 的最坏情况扣减，经验校准候选按 `1-p` 记预计不可行量；成功不退还该历史风险记账。预计不可行量不是实际失败次数的确定上界。未启动的预约可取消；已启动但未知结果的调用保留收费，不自动重试。超时或接口错误与物理不可行分开统计。

所有早停和未找到可行解的种子都保留。当前试跑中，边界探索能缓解部分停滞，但总体仍未优于原 LSBO / 约束 BO；预算控制正确不等于优化收益已经成立。

## 真实评估保护

2026-09-30 阶段的历史 400 次日志使用 v2 任务配置，之后 v3 将主下降率从 500 改为 450 ft/min。2026-10-17 核查时，旧目录已不存在，未发现仍在运行的 LSBO 进程；旧状态不再当作当前事实。新增 v4 比较数据还改变了航程、推进与滑流模型、运行约束，不能与 v3 合并。

`run_x57_smoke.py --prepare` 只生成 `operator_approved=false` 的请求，记录当前物理代码、资产、配置、Python 和基础数值库身份，并进行廉价预检。真正执行还须显式批准、确认现有实验已结束、核对冻结身份一致。每个请求最多一次真实调用、风险额度 1，无自动重试。实际调用仍使用原 LSBO 的独立 worker 和其超时设置，不更改飞机模型。

本次已批准的 v3 默认候选检查结果为 feasible，目标 46,721.4338 Wh，10 项所需约束均有效，账本 1 次调用、无未结算项。源结果与适配输出可逐字段核对；不为了接口比较重复调用物理模型。

`run_x57_pilot.py --prepare SOURCE_RUN` 仅训练历史模型、廉价筛选候选并写入计划。试验固定最多 8 次顺序评估、4 次边界候选、经验风险额度 4.8；累计 3 个非可行/未知结果或首个基础设施/未知异常即停止，不自动重试。模型和候选在执行前冻结，本入口是前瞻接口与概率检查，不是自适应优化效果试验。历史拟合与校准使用的 240 次评估单独披露，不能作为免费初始化掩盖成本。

## 模型边界

- 复用原 LSBO 的连续量、激活掩码和 VAE 潜变量。每个时间前缀从头训练表示，不使用未来观测；已知完整设计语法不视为未来观测。
- 目标使用全部有效能耗，包括显式违规设计。约束 GP 按各自有效性掩码训练，不以全局成功标记替代输出有效性。
- 风险头使用随机森林原始概率。默认正负类各至少 3 个样本才拟合；不足时返回 Beta(1,1) 平滑计数估计，并标记 `prior_only`，不表示已经校准。
- 五个机制头只使用对应标签已知的记录，输出受标签观测条件限制。未知机制不补成负类，机制概率和约束边际满足概率不直接相乘。
- 最终可行性使用独立二分类头，不冒充严格联合概率。GP 标准差也不是对表示误差和物理误差的完整不确定性估计。
- 训练数据身份或表示版本不一致时拒绝使用旧模型，避免更新 VAE 后继续使用过期模型头。

## 评估语义

所有显式约束沿用现有 LSBO 约定：`g(x) <= 0` 为满足约束。本层不改变符号、阈值或单位。

| 状态 | 目标训练 | 失效训练 | 最终可行标签 |
| --- | --- | --- | --- |
| 有效目标、全部约束通过 | 使用真实目标 | 成功标签 | True |
| 有效目标、一个或多个约束违规 | 使用真实目标 | 隐藏评估成功；显式违规分开记录 | False |
| 有效目标、必要约束缺失且无已知违规 | 使用真实目标 | 隐藏评估成功 | 未知 |
| 已识别的隐藏失效 | 不使用目标 | 只使用有证据的失效标签 | False |
| 超时、崩溃、无返回等运行故障 | 不使用目标 | 不直接当作飞机物理失效 | 未知 |
| 无法归因的异常 | 不使用目标 | 保留原始状态，等待诊断 | 未知 |
| 廉价解析预检拒绝 | 不使用目标 | 不作为隐藏失败 | False，但不进入昂贵评估校准集 |

`hidden_success` 表示当前固定评估流程是否产生可信目标，不是飞行安全认证。隐藏失效的 False 可行标签表示该流程未获得有效可行设计，不证明设计在物理上永远无解。

五个隐藏失效头为几何生成失败、网络配置失败、电机设计失败、任务求解不收敛、能量审计失败。它们描述最终评估事件，不描述求解器已恢复的中间试探异常。

历史通用 `ValueError` 不会仅凭错误文本包含 SOC、geometry 等词就被自动归因。能量审计明确失败、任务段明确未收敛或受支持的最终状态码才构成分类证据。几何、网络和电机头已定义，但旧日志未提供明确证据时仍为未知；本轮未改造 worker 以生成这些新状态码。

失败后尚未执行的检查保持 `null`，不能自动标记为 0。一次任务失败也不意味着失速、SOC 等全部违规。有效数值约束允许同时违反，多标签记录为 `constraint:<约束名>`。

## 缺失值与训练隔离

- `objective_observed=False` 时目标必须为 `null`，不能写入大罚值。
- `constraint_observed` 对每个必要约束单独标记。失败记录里的残留数字默认不可信；只有额外的布尔有效性标记明确为 True 才允许使用。
- 成功任务中的有限约束值默认视为观测；`NaN`、无穷大、布尔伪数值或缺失值均不能成为有效余量。
- 解析几何、翼展及当前低阶颤振预检量保留在记录中，但不进入昂贵约束 GP 训练表。
- 失败种子根的负功率诊断保留来源信息，不作为任务目标或完整任务约束余量。
- 超时等已提交日志中的失败调用仍记录预算消耗；`mda_evaluated` 不明时保持未知，不能以预算消耗推断任务已完成。
- `training_rows.json` 是未划分的数据选择结果，不是训练集、校准集、测试集已经独立分离的证明。

## 已完成的真实日志审计

源运行：`run_20260928_142256_6443ac03`，记录状态为 complete，共 30 次评估。

| 项目 | 数量 |
| --- | ---: |
| 可行设计 | 25 |
| 有效目标但显式约束违规 | 2 |
| 任务求解失败 | 3 |
| 可用于目标头的真实能耗值 | 27 |
| SOC 违规 | 1 |
| 失速速度违规 | 1 |
| 可用端口能量违规 | 1 |

显式违规数不能直接相加作为不可行设计数，因为同一设计可以违反多个约束。

任务失败头有 3 个正样本和 27 个负样本。其余四个隐藏失效头均没有正样本，不能据此宣称这些风险不存在，也不能宣称分类或校准已经有效。

已完成第 11 至 30 次候选的 20 次逐前缀预测：最终可行性头有 12 次达到拟合门槛，任务失败头仅有 2 次，其余使用标明状态的先验输出。最终可行性的全体原始预测 Brier 为 0.1640；17 个有效目标的预测 RMSE 为 1266.12 Wh。这是单次小样本历史诊断，不是优化收益或校准有效性的证据。完整结果见 `docs/MODEL_VALIDATION.md`。

## 下一步

2026-10-21真实系统验证v5已结束：
`runs/system_validation_v5_20261020T192312Z_5f6531eb/`。
所有版本累计实际500次，包含前版128次；140次训练补采、
72次全新校准和288次配对优化均完成。旧校准并入训练，
新增构型及机制邻域校准在真实优化之前冻结，算法参数没有按结果调整。
完整24个真实对照、独立审计及预测重建已完成。
总报告见 `docs/SYSTEM_VALIDATION_FINAL_RESULTS_20261021.md`。

本轮500次授权已用完。后续先明确worker阶段诊断、其他真实机制的可审计失效场景、独立物理任务及能吸收新增标签的长期更新协议；新的真实计划超出本轮上限时需另行确认。此前120点方案已完成，不重复执行。当前结果不能包装成算法优越性的论文结论。

现有 LSBO 仍独立运行。审计和回放入口不执行 MDA，合成优化只调用合成函数；真实入口受上述授权与版本检查保护。
