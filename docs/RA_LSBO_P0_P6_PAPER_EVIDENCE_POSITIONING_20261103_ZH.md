# RA-LSBO P0–P6 论文证据与创新定位备忘录

日期：2026-11-03。性质：内部研究判断，不是论文结论或新的实验协议。本次仅整理已有记录与文献，新增简化飞机、RCAIDE、GP 拟合和 VAE 训练均为 0。文献检索不是系统性综述或专利检索；以下“创新空间”不等于已证明新颖。

## 一、执行摘要

**当前最可信的定位是：面向层级混合变量、部分可观测约束飞行器设计的研究型 BO 框架，探索如何让不同输出使用不同结构表征，并利用已结算反馈审计风险决策。** 已有工作证明各组件能够按登记的接口和时间顺序运转；但现有结果**尚不支持**“动态权重普遍提升优化效率、可行发现率、风险校准或飞行安全”的主张，也尚不能支持 SMO 级别的最终性能结论。

尤其应区分三件事：第一，H1 的目标潜空间与约束显式空间是候选建模架构，已有小样本对照；第二，H2 动态混合权重有严格的提交预测—结算—更新机制，P3 的实际优化结果是混合的；第三，P6 在 2 个可识别历史前缀上给出了候选生成与排序的有限单步干预证据，并不是对整条优化轨迹或能耗改善的因果证明。

**结论：可撰写问题定义、方法和探索性结果初稿；不宜现在提交以算法优越性或真实航空风险保证为主结论的完整论文。** 应先补齐任务稀疏度资格、预测/失效解释、强外部基线、独立种子及独立真实 MDA 验证，再决定主算法。预注册探索与独立确认分开，既不重写过去也不以结果反选阈值。

## 二、可追溯证据账本

本文统一使用**新增简化飞机求值调用**，不把离线 GP 拟合、数学测试、初始结构采样、RCAIDE MDA 混为一次类型。以下上限和实际调用均以各阶段的原始登记/审计为准；不同阶段不能当作独立同分布重复或直接合并排行榜。P0–P6 是研究阶段，不是七个独立性能基线。

| 阶段/方法 | 可主张的机制与直接对照 | 预算、实际费用与证据强度 | 尚不能得出的结论 |
|---|---|---|---|
| 先前 `old30`、`corpus_stable`、`onehot_stable` | 中断试验中的表示探索；`old30` 从头训练，`corpus_stable` 使用 2,000 条保留重复的结构样本及持久编码器，one-hot 为显式结构表征 | 原 corpus pilot 已结算 **2,212** 次后中断，不能作为完整结果；D29/SOC 原 8 组核失败，稳定尾部矩固定输入下 8 组收敛，只是数值故障恢复。[E1] | 不可据此认定持久表示导致中断，也不可把数值收敛等同预测校准或优化收益。2,000 条结构样本不是 2,000 次真实飞机标签或 2,000 个不同构型。 |
| H1 / `hybrid_onehot` | 潜空间目标 GP + 分层 one-hot 完成/约束 GP；与全 one-hot、全潜空间及 `hybrid_guard` 同预算对照 | smoke 实际 **37/42**；四种子 pilot **856/856**，2 任务、5 方法；`random` 是唯一明显外部风格下界，其余主要是内部变体。[E2] | pilot 报告模板写“Eight pilot seeds”是已知笔误，实际 **4**；不能声称 8 个种子或外部强基线胜出。100 km onehot 中位能耗低于 hybrid_onehot；不同场景排序不同。 |
| H1 / `hybrid_guard` | 同 `hybrid_onehot`，对 SOC 额外建潜空间头，并按候选取较小的预测可行概率 | 对照 `hybrid_onehot` 可考察这条保守规则，但两者仍是四种子开发性数据。[E2] | `min(P)` 不是已校准下置信界、失效概率保证或现实安全保证；140 km 结果提示可能付出能耗代价，但样本太少，不能推广。 |
| H1 同前缀预测面板 | 相同 24 个历史前缀、两个 SOC 表征及跨五策略 120 个目标记录，考察相同候选上的预测 | **48 个 GP 头/384 个核候选拟合**；0 次新飞机/VAE/RCAIDE，原 one-hot 预测重现。[E3] | 120 个相关候选不是 120 个独立种子；删失概率质量与精确值密度分开评分；极端区间点影响较大，尚无独立场景校准。 |
| H2 P0 | 稳定高均值高偏移下的高斯混合方差计算；测试预报先提交、结算后更新和重放审计 | 零飞机调用的回归与登记；不把方差公式改写称作新概率模型。[E4] | 数值稳定及无未来标签泄漏不等于预测更准或优化更好。 |
| H2 P1 / `h2_causal` smoke | 六组流程、GP/VAE 接口和账本；对照包括 `random`、`onehot_stable`、`corpus_stable`、`hybrid_onehot`、`hybrid_guard` | 固定种子 11061，实际 **43/48**；其中某些 140 km 组没有可行 incumbent，未使用的复核预算不转移。[E5] | 仅 2 个在线步骤，低于 4 个已评分反馈的 warmup，无法验证飞机闭环中的动态权重有效作用。 |
| H2 P2 / `h2_causal` | 前缀因果、配对密度/区间得分驱动 onehot 与潜空间 SOC GP 的有界混合；可行概率用两个分量 CDF，乐观几何用矩匹配近似 | 四种子 11062–11065、六方法、48 trial；**1,007/1,008**：96 初值、864 在线、47 次 5 s 数值复核；142 次评分、118 次有效权重变化，审计通过。[E6] | 相对 `hybrid_onehot` 同时改变“双头混合”和“动态权重”，不是动态权重的独立效果；冷启动可行解可发生在 warmup 之前。5 s 复核不是风险/飞行安全验证。 |
| H2 P3 / `h2_dynamic` vs `h2_fixed50` | 共享初始化、两 SOC 头和搜索规则，仅动态更新对固定 0.5/0.5；这是动态权重的直接闭环机制对照 | smoke **24/24**；8 种子 12071–12078、2 任务、32 trial 的 pilot **800/800**（192 初值、576 在线、32 次复核）。审计通过。[E7] | 同初始数据不意味着分歧后的训练历史相同；8 种子是开发性实验，不是独立确认。P3 结果见下节。 |
| P4 固定池离线分析 | 在首次候选分歧前匹配历史与全局池；仅对既存增强池重算权重 | 16/16 同全局池、16/16 最终局部增强池不同；固定既存池重排 8/16；新增飞机/GP/VAE 均 0。[E8] | 8/16 是**固定池模型排序**，不能转述成 8/16 实际完整选点由权重造成。分歧后轨迹不是同历史单步对照。 |
| P5 完整单步重放 | 恢复检查点与过去历史，每组重建一次 GP；原动态和固定两臂各重复两次，再以权重干预完整选择 | 16/16 的已保存起点、终点、局部报告、最终池和选点通过重放；**1,152** 个 GP 核候选拟合，0 次飞机/RCAIDE/VAE 拟合；预定阈值下仅 **2/16** 可识别，2/2 选点不同。[E9] | 14 组近并列不应当作零效果。原历史不含全部 SLSQP 内部查询记录，因此不能声称逐查询轨迹获得历史验证。 |
| P6 生成/排序 2×2 | DD、FF、DF、FD 在同一历史及 GP 状态上分别控制局部候选生成与最终排序权重；DD/FF 首先各重放两次 | **1,152** 个离线 GP 核候选拟合；0 飞机/RCAIDE/VAE 训练；在预定 `1e-10` 四单元门槛下仅 **2/16** 可识别。[E10] | 可识别的 2 组中，固定任一排序权重时改变生成权重均 2/2 换点；固定任一生成池时改变排序权重均 0/2 换点。不是总体频率、能耗收益或安全收益。 |

**共同研究边界：**上表 100/140 km 为简化电动飞机的固定任务与确定性设定，历史标签 `aircraft_sparse` 并不证明可行域占比低于 10%；共用初值中暖启动偏多。5 s 同设计积分步长复核属于数值一致性检查，**不等于** RCAIDE MDA 验证。原先 316 次资格检查、中断的 2,212 次、后续各阶段预算各自记账，不把重复引用的资格样本再次算成新调用，也不把不同目标的研究总额混作单个算法预算。

## 三、P3 混合结果与 P6 因果边界

P2 在 140 km 的 4 个配对种子上，`h2_causal` 相比 `hybrid_onehot` 的名义最终能耗均较低，配对均差约 **-1402.28 Wh**；但该对照同时改变 SOC 双头融合与动态权重，且其中冷启动可行发现可能早于动态 warmup 结束，因此不能把这一差值归因于动态调权。[E6,E7]

P3 以每个任务内的配对种子报告 `dynamic - fixed50`，负值代表动态组名义可行能耗较低；两组在两个任务中各 8/8 找到名义可行解。100 km：动态组 5 胜 3 负，均值 **-42.460 Wh**，中位配对差 **-0.558 Wh**。140 km：4 胜 4 负，均值 **-262.908 Wh**，中位配对差 **+0.379 Wh**。140 km 的种子 12075 为 **-2167.313 Wh**；仅作敏感性说明，去除它后该任务均值变为 **+9.150 Wh**，并非可以删去该种子或改变主要结果。[E7,E8]

P6 的问题是给定相同历史、已验证 GP、状态与候选池后，仅改变当前步候选生成权重或最终排序权重，选点是否改变。它不是“两个方法最终真实能耗的因果分解”。预先固定 `1e-10` 的前二采集分数间隔门槛，4 个单元任一个不满足即该组不识别；14/16 组被排除出可识别计数而非判定“无作用”。在两个通过者上，局部生成两个条件对照均为 2/2 变点，排序两个条件对照均为 0/2。两项条件效应是分类结果，不能相加、外推至 16 个种子或解释为真实安全收益。[E10]

阈值敏感性必须与主结果并列：`1e-12`、`1e-11`、`1e-10`、`1e-9`、`1e-8` 下可识别数依次为 **13、5、2、2、2**；低阈值下排序效应也出现变化，不能事后挑选有利结果。重复计算可重现已保存的 SLSQP 起点/终点/状态与增强池，但过去未封存内部逐查询路径；现在新生成的查询轨迹不能冒称历史真值。[E9,E10]

## 四、与文献逐项对照

| 候选主张 | 已有相关工作及差异 | 可辩护的本项目表述 / 缺口 |
|---|---|---|
| 部分反馈下的精确/区间 GP、EP | Wang 和 Li 的 CBOB 已处理部分观测约束、异质反馈与 EP，并设计平衡改进采集 [L1]；飞机隐藏约束 BO 已研究不可计算区域 [L2,L3]。 | **不是首次提出删失或部分观测 BO。** 当前可研究“早终止后的物理约束界是否有可信来源、与层级飞机表征如何结合”；应对相同反馈接口做 full/interval/sign/missing 对照，并明确未知基础设施故障不能伪装成物理不可行。 |
| 层级/混合变量潜空间及不同视图 | Zhang 等和 Oune/Bostanabad 已研究混合变量潜在 GP [L4,L5]；Saves 等已在绿色飞机 MDO 中研究混合类别 GP [L6]；Bussemaker 等比较昂贵层级架构的表征/优化策略 [L7]。 | **不是首次提出潜在表示、one-hot 或混合核。** 可以检验“目标和安全相关约束采用不同视图是否更有利于预测和预算内可行优化”；必须对照全潜空间、全 one-hot 和目标/风险分离，固定表征预算与训练标签来源。 |
| SOC 两头概率融合与历史得分动态权重 | 专家聚合、预序贯评分和混合分布属于既有方法家族；本次未完成覆盖全部在线专家/BO 文献的系统查新。 | 潜在贡献仅限特定约束、双视图、部分观测和可审计反馈的组合与可验证假设；P3 不支持性能优越。精确 SOC 使用密度、删失 SOC 使用区间概率，二者不作未经定义的混合平均。 |
| 风险感知约束与飞机局部优化 | Priem 等已有面向飞机设计的混合约束 UTB 准则 [L8]；SCBO 将约束处理与局部信赖域结合 [L9]；自编码器降维与约束 BO 也已有研究 [L10]。 | 不能把 GP 可行概率、保守 min、SLSQP 或局部搜索单独称创新；P6 可作为**该实现的决策机制诊断**。现有置信度主要表示代理认知不确定性，未显式建模真实场景随机性。 |
| 风险目标的航空工程意义 | 电动/混动飞机设计已有针对关键参数假设与设计可信度的 MDO 研究 [L11]。 | 如果论文题目声称面向风险，应预先定义气象、载荷、电池容量/热状态等不确定变量、来源与物理关联，给出可行概率或尾部损失目标及独立验证；目前不能把 GP 方差当成物理故障率。 |

说明：文献条目以出版方 Crossref 元数据和可访问的出版方/作者页面核验；[L1] 的 AAAI 页面本次遭反爬拦截，项目已有只读原文缓存及检查记录 [E11]；[L8,L10] 结合项目已有文献核查，具体外部基线实现仍须再次按原文逐项核对。对尚未读到的全文不虚构实现细节或声称穷尽先例。

## 五、候选创新点排序

1. **首选研究主线：输出/证据敏感的层级表征与约束决策。** 将结构潜空间服务于目标建模，显式层级表征服务于失败/约束，并保留可验证的精确、区间和未知语义。价值是可以提出明确反事实：同一历史和预算下，响应头表征变化是否改善跨构型预测、首次可行发现和最终目标。**目前是候选科学问题，不是已证明贡献。** 需与 `onehot_stable`、`corpus_stable`、`hybrid_onehot` 公平对照，且与已有混合变量飞机 GP 比较。
2. **次选：部分证据驱动的双 SOC 模型融合与时间因果账本。** 可作为特定工程任务的可复现建模方案及消融对象；与固定 one-hot、固定 latent、固定 0.5、保守 guard 必须同条件对比。动态调权当前属于**待证假设**，不应被写为贡献已提升效率的结论。精确 CDF 与矩匹配几何近似必须同时披露。
3. **支撑性机制证据：生成与排序分离的离线同前缀干预。** P6 的四单元设计有助于解释该实现中的候选分歧，可作为方法分析/消融，而不是独立宣称广义 BO 新算法；2/16 与 14/16 的分母和历史 SLSQP 记录缺口必须同时出现。
4. **工程必要条件而非独立创新：稳定尾部 EP、失败账本、局部连续优化与数值复核。** 它们提高可运行性、诊断性或可复现性，但数值修复、一般性局部优化以及开发期 smoke 本身不足以支撑 SMO 创新主张。

建议暂用工作性贡献表述：“提出并审计一种面向层级混合设计、可处理可信部分反馈的异质表征约束优化框架；通过匹配历史及消融分析其预测、选点机制与成本”。**仅当新增实验支持时**再加入“提高效率/可靠性”；否则诚实改为条件性结论、负结果或研究报告。

## 六、下一阶段实验必要性与门槛

1. **先做零真值再分析，不自动发明新算法。** 审查 P6 近并列的候选距离、同一局部最优邻域内采样点及求解器状态；报告按原登记阈值的结论，并把其它阈值与候选合并规则仅作探索性敏感性分析。任何新规则必须另行预注册，不能回改 P5/P6 判定。复核 12075 的物理约束、初始状态与历程，不排除该种子。
2. **重新定义和资格化任务。** 用独立、预先冻结的合法层级构型抽样，分别估计解析约束、数值完成、物理约束和联合可行率，给出抽样分布、种子、样本量和不确定区间；按预先确定的区间分层（包括冷启动困难但并非不可解的任务）。抽样分布决定“稀疏”，不能把优化轨迹可行率当作整个设计空间体积。资格样本和参数选择不得泄漏到确认集。
3. **分层评价预测与风险。** 在固定前缀、共同候选及独立场景候选上，分别报告精确 SOC 密度 NLL、区间概率质量 NLL、可解析标签 Brier、覆盖率与决策阈值附近误判；按任务/种子/场景聚类，报告置信区间、失效样本数及选择偏差。校准是经验结果，不能由 EP 收敛或一小组命中率直接证明。
4. **独立、同成本优化对照。** 在新种子、先注册的调用预算下比较候选主法、随机、one-hot、全潜空间、固定/动态混合；增加适配清楚且能忠实复现的外部约束 BO，如 CBOB、SCBO、SEGO-UTB，必要时加混合变量进化基线。相同初始信息、反馈契约、失败收费、离线训练和总计算费用；不把适配实现冒称原版，不用测试表现挑参数。保留原 LSBO 作为论文声称相对于 LSBO 改进时的必要继承对照。
5. **真实 MDA 与风险实验需单独登记。** 对固定的 RCAIDE 飞机/任务及版本先验证质量、任务和约束输出，明确条件变量与求解失败；用独立确定设计/不确定场景集合估计实际约束违规、能耗分布和计算费用。区分认知不确定性与工况/参数不确定性；5 s 同模型复核不代替 MDA 或物理验证。是否做少量关键设计复评还是完整优化，应先根据可信预算、独立资格结果和方法筛选判断，不在本文自动授权。
6. **统计与论文闸门。** 事先固定主要对照、主要终点、期望可检测效应与功效/样本量依据；报告逐种子结果、成对差与区间、首次可行成本曲线、预算内可行率、无解/失败、名义目标、经验违规及代价。先报告所有种子，再做预注册敏感性；不依据旧结果选择种子、把多个相关候选当独立样本或将条件可行能耗冒充无条件成功率。多任务或多对照的结论需处理比较数量与不确定性。

**阶段判断：**目前不建议进入大规模确认性 RCAIDE 优化或声称已具备风险保证。先完成零真值诊断和独立任务资格，随后用预算可承受的简化模型对候选主线进行预注册筛选；只有当风险定义、外部基线、数值鲁棒性与效果量均达到事先规定门槛，才申请独立 RCAIDE 验证。上述是研究建议，不是任何新调用授权。

## 七、项目证据索引

- [E1] 原中断 2,212 次及 D29 数值诊断：PR [#14](https://github.com/Luyi1235/RA-LSBO/pull/14)、稳定性候选 PR [#15](https://github.com/Luyi1235/RA-LSBO/pull/15)。完整中断档案 Release `aircraft-corpus-pilot-interrupted-20261031`，不得作为已完成 pilot。
- [E2] H1 预算/五组方案和独立审计：PR [#17](https://github.com/Luyi1235/RA-LSBO/pull/17)、[#18](https://github.com/Luyi1235/RA-LSBO/pull/18)；原 H1 完整 ZIP SHA-256 `3d9cb4f684bbc9198c523bb44bf466463c2f85053d87ca1770eca9111b48fed1`。
- [E3] 24 前缀跨策略面板及独立审计：PR [#19](https://github.com/Luyi1235/RA-LSBO/pull/19)、[#20](https://github.com/Luyi1235/RA-LSBO/pull/20)。
- [E4–E6] H2 P0/P1/P2：PR [#22](https://github.com/Luyi1235/RA-LSBO/pull/22)、[#23](https://github.com/Luyi1235/RA-LSBO/pull/23)、[#24](https://github.com/Luyi1235/RA-LSBO/pull/24)、[#25](https://github.com/Luyi1235/RA-LSBO/pull/25)、[#26](https://github.com/Luyi1235/RA-LSBO/pull/26)。P2 执行代码 `7ca420613ccb50e138dc3c9d4292ac8e863c4ecf`。
- [E7] P3 协议、smoke、pilot 和独立审计：PR [#27](https://github.com/Luyi1235/RA-LSBO/pull/27)、[#28](https://github.com/Luyi1235/RA-LSBO/pull/28)、[#29](https://github.com/Luyi1235/RA-LSBO/pull/29)、[#30](https://github.com/Luyi1235/RA-LSBO/pull/30)、[#31](https://github.com/Luyi1235/RA-LSBO/pull/31)。P3 代码 `57e390fc8f6280db9f201940aec14089e81c104c`；原完整 ZIP SHA-256 `bd170195201b45942e4fe2c2e04e1e994263bb0a4566a467b28816cdce0473b4`。
- [E8–E10] P4、P5、P6：PR [#32](https://github.com/Luyi1235/RA-LSBO/pull/32)、[#33](https://github.com/Luyi1235/RA-LSBO/pull/33)、[#34](https://github.com/Luyi1235/RA-LSBO/pull/34)；P6 数据固定提交 `5ea0e20c5e6def03d7178500b48dd8a10be622ed`，其 `results/h2-p6-two-stage-attribution-20261103/` 含逐单元、敏感性与 SHA-256 清单。
- [E11] 项目此前原文核验与基线综述：`docs/THREE_FIDELITY_LITERATURE_NOTES_20261026.md`、`docs/ALGORITHM_COMPARISON_LITERATURE_REVIEW_20261026.md`；对应研究分支和 P6 固定提交中的文件，不能把文献建议误作现有基线已验证。

## 八、主要参考文献

- [L1] Wang S, Li K. Constrained Bayesian Optimization under Partial Observations: Balanced Improvements and Provable Convergence. AAAI, 2024. https://doi.org/10.1609/aaai.v38i14.29488
- [L2] Tfaily A, Diouane Y, Bartoli N, et al. Bayesian optimization with hidden constraints for aircraft design. *Structural and Multidisciplinary Optimization*, 2024. https://doi.org/10.1007/s00158-024-03833-8
- [L3] Bussemaker J H, Saves P, Bartoli N, et al. Surrogate-based optimization of system architectures subject to hidden constraints. AIAA AVIATION, 2024. https://doi.org/10.2514/6.2024-4401 （作者公开版：https://arxiv.org/abs/2504.08721 ）
- [L4] Zhang Y, Apley D W, Chen W. Bayesian Optimization for Materials Design with Mixed Quantitative and Qualitative Variables. *Scientific Reports*, 2020. https://doi.org/10.1038/s41598-020-60652-9
- [L5] Oune N, Bostanabad R. Latent map Gaussian processes for mixed variable metamodeling. *Computer Methods in Applied Mechanics and Engineering*, 2021. https://doi.org/10.1016/j.cma.2021.114128
- [L6] Saves P, Diouane Y, Bartoli N, et al. High-dimensional mixed-categorical Gaussian processes with application to multidisciplinary design optimization for a green aircraft. *Structural and Multidisciplinary Optimization*, 2024. https://doi.org/10.1007/s00158-024-03785-z
- [L7] Bussemaker J H, Saves P, Bartoli N, et al. System architecture optimization strategies: dealing with expensive hierarchical problems. *Journal of Global Optimization*, 2025. https://doi.org/10.1007/s10898-024-01443-8
- [L8] Priem R, Bartoli N, Diouane Y, et al. Upper trust bound feasibility criterion for mixed constrained Bayesian optimization with application to aircraft design. *Aerospace Science and Technology*, 2020. https://doi.org/10.1016/j.ast.2020.105980
- [L9] Eriksson D, Poloczek M. Scalable Constrained Bayesian Optimization. AISTATS, 2021. https://proceedings.mlr.press/v130/eriksson21a.html
- [L10] Maathuis H, De Breuker R, Castro S G P. Autoencoder-enhanced joint dimensionality reduction for constrained Bayesian optimisation. *Machine Learning: Science and Technology*, 2025. https://doi.org/10.1088/2632-2153/ae0efe
- [L11] Wahler N F M, Maruyama D, Elham A, et al. Credibility-Based Multidisciplinary Design Optimization of Hybrid and Fully Electric Aircraft. *Journal of Aircraft*, 2025. https://doi.org/10.2514/1.C037516
