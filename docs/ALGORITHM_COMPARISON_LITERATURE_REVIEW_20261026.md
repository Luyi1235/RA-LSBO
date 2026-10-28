# RA-LSBO 首轮算法对比文献核查与建议

日期：2026-10-26

## 结论

首轮先比较外部算法，不做组件消融。建议主比较组为：

1. RA-LSBO（本文方法）。
2. Uniform Random Search（随机搜索）。
3. Standard constrained GP-BO（独立目标/约束 GP + logCEI；低维数学任务可称 logCEI，飞机层级/缺失反馈处理须报告为适配版）。
4. SCBO（数学任务使用原始问题设定；混合层级飞机任务若经改造，明确标注 mixed/partial-feedback adaptation，而非原版复现）。
5. SEGO-UTB（飞机设计约束 BO 的强相关参照；若无法复现其算法细节或软件依赖，不能以同名近似算法代替）。
6. NSGA-II（混合变量飞机概念设计中已有直接先例；单目标问题以预先固定的可行性优先排序处理约束）。

这是候选名单，不代表当前代码已实现这些算法。最终冻结名单前，应先确认每个算法都能在相同设计域、初始标签、昂贵评价预算与失败反馈下运行。若 SEGO-UTB 无法忠实实现，则从主表移除并说明原因；不应仅凭名称写成该算法。可将 COBYLA/NOMAD 作为连续、完整反馈控制实验的补充，但不建议直接放进层级混合主比较：二者原生处理的是连续无导数问题，类别变量、条件激活变量以及仿真中途失败都需要显著适配。

原 LSBO 暂不列入首轮主比较，也不删除其历史代码或结果。它是方法来源/继承基线，不是独立外部算法；首轮的目标是先观察 RA-LSBO 相对于公认约束优化方法的表现。若后续论文仍主张“在 LSBO 上的改进”，最终确认实验应另加一个冻结的原 LSBO 对照。

## 文献做法

### 飞机设计与混合约束 BO

- Priem et al., “Upper trust bound feasibility criterion for mixed constrained Bayesian optimization with application to aircraft design,” *Aerospace Science and Technology*, 105, 105980 (2020), DOI: 10.1016/j.ast.2020.105980。该研究以 SEGO-UTB 处理混合约束，并将其用于概念飞机构型设计。论文算法比较中包含其他 SEGO 类方法及无导数优化器；不同问题规模下比较集合有所变化。其价值在于提供与飞机混合约束高度贴近的 BO 参照，而非证明某一固定算法集合适用于所有任务。
- Priem et al., “An efficient application of Bayesian optimization to an industrial MDO framework for aircraft design,” AIAA AVIATION 2020, DOI: 10.2514/6.2020-3152。研究将 SEGO/MOE 类约束 BO 用于 Bombardier 飞机 MDO 案例，并与工程软件中常见优化器比较，强调昂贵 MDA 调用数而非单纯 wall-clock 时间。
- Saves et al., “Bayesian optimization for mixed variables using an adaptive dimension reduction process: applications to aircraft design,” AIAA SciTech 2022, DOI: 10.2514/6.2022-0082。该研究围绕混合变量与飞机应用，提出 PLS 降维 BO，并在公开版本中与 NSGA-II、Bandit-BO 等参照比较，说明进化算法和已有混合变量 BO 都是飞机概念设计中有依据的外部基线。
- “Bayesian optimization with hidden constraints for aircraft design,” *Structural and Multidisciplinary Optimization* (2024), DOI: 10.1007/s00158-024-03833-8。该工作直接研究飞机设计隐藏约束，支持将隐藏失败处理作为算法能力比较的一部分；其工业仿真受知识产权限制，公开复现边界需要谨慎。
- “Bayesian optimization of cooperative components for multi-stage aero-structural compressor blade design,” *Structural and Multidisciplinary Optimization* (2025), DOI: 10.1007/s00158-025-03998-w。该文应用 CoEGO 于高维压气机叶片问题，并在研究背景中梳理 PCA-BO、SCBO、TuRBO 和 AutoOpti 等相关方法。它可证明这些方法属于工程 BO 文献中的常见参照家族，但不应误述为该论文对所有方法都做了同条件实证对比。

### 约束优化算法对照的跨领域依据

- Maathuis et al., “Autoencoder-enhanced joint dimensionality reduction for constrained Bayesian optimisation,” *Machine Learning: Science and Technology* 6 (2025), 045028。其基线包括 PCA-GP SCBO、logCEI vanilla BO、二次罚函数、COBYLA、CMA-ES 与随机搜索。该集合是近期昂贵约束优化论文可参考的“BO + 经典无导数 + 随机”组合，但其中部分算法不是飞机设计专属方法。
- Bartoli et al., “Adaptive modeling strategy for constrained global optimization with application to aerodynamic wing design,” *Aerospace Science and Technology* 90 (2019), 85–102, DOI: 10.1016/j.ast.2019.03.041。数学约束问题中比较 SEGOMOE 与 COBYLA、NOMAD、ALBO 等无导数/约束 BO 方法；机翼设计案例另与梯度法比较。它提示应按问题结构分层比较，不需要强行让每个算法覆盖每个工程案例。
- Priem et al. (2020) 的飞机设计研究在实际飞机问题上由于计算代价，缩减了可行的对比集合；论文报告多个独立运行的收敛分布，而不是只挑单次最优结果。

## 与本项目的适配判断

本项目候选设计是条件层级混合变量，评估还可能返回部分约束区间或隐藏失败。因而：

- Random Search 是必要下界参照，保留。
- logCEI 和 SCBO 都是受约束昂贵 BO 的通用强参照；需分别实现并命名，不复用当前 `conventional_eic` 后改名。
- SEGO-UTB 是飞机混合约束设计中最相关的工程方法候选。须依据论文的采集/可行性规则实现并留存参数；若做不到，宁可标记“未纳入”，也不声称复现。
- NSGA-II 能自然表示离散设计，但经典版本通常依赖完整数值目标/约束。必须预先定义对未完成仿真的可行性排序，不得对隐藏失败事后挑选罚值。
- COBYLA、NOMAD、CMA-ES 可作为额外传统优化器，但需要处理层级类别与缺失反馈；若用分支枚举或罚函数，适配会改变算法含义，故不应挤占首轮核心比较，除非实现并验证了统一、事先固定的反馈策略。
- 原 LSBO 保留为冻结历史/补充参照，不进入本轮首筛主组。

## 公平比较及报告要求

1. 每个算法使用相同冻结初始设计及初始标签；不允许算法专属预训练标签或面板标签进入选点。
2. 对齐的是昂贵模型调用预算，而不是迭代轮数。失败、区间和未知结果都计一次调用，且按算法原生可用信息或预注册的统一适配规则处理。
3. 同种子成对运行，保留无可行解的运行；同时报告预算内可行发现率、首个可行解成本、最佳可行目标轨迹、隐藏失败/约束违规次数及运行间分布。
4. 数学基准、简化飞机和 RCAIDE MDA 分开陈述。廉价层的筛选结果不能替代真实 MDA 的独立验证。
5. 首轮两三个种子仅用于实现筛查和发现明显失效，不用于发表级优越性结论。确认实验种子数由效应与变异估计另行设计，不以首轮赢家临时追加或调参。

## 当前代码状态与待实现事项

当前实验包声明的七个 arms 是 `ra_lsbo`、`original_lsbo`、三个消融、`conventional_eic` 和 `random`。其中 `conventional_eic` 在代码注释中已明确为本项目的数值输出、层级 one-hot 适配，而不是官方 SCBO/CBOB 或标准化基线。因此现有 smoke/pilot 输出不能当作上述外部算法对比结果。

建议新建版本化的“external algorithm comparison”实验协议，不覆盖旧实验目录、不改写其来源快照。完成 baseline 实现与单元/回放测试后，再冻结一份不超过已授权调用上限的 prospective plan。正式启动前须核对六组名单各自的完整预算，并确保资格样本、独立面板、初始数据及飞机场景验证均已计入。

## 参考文献与入口

- Priem et al. (2020), AST: https://doi.org/10.1016/j.ast.2020.105980
- Priem et al. (2020), AIAA AVIATION: https://doi.org/10.2514/6.2020-3152
- Saves et al. (2022), AIAA SciTech: https://doi.org/10.2514/6.2022-0082
- Hidden-constraint aircraft BO, SMO (2024): https://doi.org/10.1007/s00158-024-03833-8
- CoEGO compressor blade design, SMO (2025): https://doi.org/10.1007/s00158-025-03998-w
- Maathuis et al. (2025), MLST: https://doi.org/10.1088/2632-2153/ae0efe
- Bartoli et al. (2019), AST: https://doi.org/10.1016/j.ast.2019.03.041
